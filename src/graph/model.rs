//! In-memory code graph model + its on-disk `graph.json` representation
//! (Step 3 of `docs/ideation/trm-code-graph/2026-08-19-scoped-graph-mvp-plan.md`).
//!
//! Deliberately a thin, hand-shaped model — `File`/`Function`/`Struct`/
//! `Enum`/`Trait`/`Impl` nodes, `Contains`/`Calls`/`Implements`/`Uses`
//! edges — not graphify's full taxonomy. No confidence/audit-trail tiers
//! (EXTRACTED/INFERRED/AMBIGUOUS): every edge this crate ever writes is a
//! deterministic AST fact, so that distinction doesn't apply here.

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    File,
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    Contains,
    Calls,
    Implements,
    Uses,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Stable, human-readable id — e.g. `src/foo.rs::FooStruct::new` — the
    /// on-disk join key across `build`/`update` runs and every query verb's
    /// user-facing handle. Not `petgraph`'s own `NodeIndex`, which is only
    /// stable for the lifetime of one in-memory graph and would silently
    /// break across the very next `update`.
    pub id: String,
    pub kind: NodeKind,
    pub name: String,
}

/// On-disk shape of `graph.json` — plain nodes plus edges addressed by
/// node id, not `petgraph`'s own internal representation. Keeps the file
/// human-readable and independent of `petgraph`'s index layout, which
/// shifts across mutations and was never meaningful across separate
/// `build`/`update` runs to begin with.
#[derive(Debug, Serialize, Deserialize)]
struct GraphFile {
    nodes: Vec<Node>,
    edges: Vec<(String, String, EdgeKind)>,
}

/// In-memory graph plus the id→index lookup every caller needs —
/// extraction, incremental update, and every query verb all address
/// nodes by their stable `id`, never by `NodeIndex` directly.
#[derive(Debug, Default)]
pub struct CodeGraph {
    pub graph: DiGraph<Node, EdgeKind>,
    index: HashMap<String, NodeIndex>,
}

impl CodeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a node if `id` isn't already present, else overwrite its
    /// weight in place; returns its index either way. Idempotent by
    /// design — extraction re-inserts the same file's nodes on every
    /// `update` pass, and this is the one place that has to tolerate that
    /// without producing duplicates.
    pub fn upsert_node(&mut self, node: Node) -> NodeIndex {
        if let Some(&idx) = self.index.get(&node.id) {
            self.graph[idx] = node;
            idx
        } else {
            let id = node.id.clone();
            let idx = self.graph.add_node(node);
            self.index.insert(id, idx);
            idx
        }
    }

    pub fn node_index(&self, id: &str) -> Option<NodeIndex> {
        self.index.get(id).copied()
    }

    /// Best-effort lookup by kind + display name, not by stable id — used
    /// to resolve a `Calls`/`Implements` target when extraction only has an
    /// identifier's short name to go on (no full type resolution). Picks
    /// the first match encountered; a real ambiguity (two functions named
    /// `new` in the same extraction run) is an accepted MVP imprecision,
    /// not a bug — see the plan doc's Risks section.
    pub fn find_by_name(&self, kind: NodeKind, name: &str) -> Option<String> {
        self.graph
            .node_weights()
            .find(|n| n.kind == kind && n.name == name)
            .map(|n| n.id.clone())
    }

    /// Add an edge between two already-inserted node ids. Returns `false`
    /// (no panic) if either endpoint isn't present yet — extraction runs
    /// emit nodes before edges within one file, but a `Calls` edge can
    /// legitimately target a not-yet-extracted or unresolved function; the
    /// caller decides whether that's worth logging, this layer just
    /// reports it didn't happen.
    pub fn add_edge(&mut self, from: &str, to: &str, kind: EdgeKind) -> bool {
        match (self.index.get(from), self.index.get(to)) {
            (Some(&f), Some(&t)) => {
                self.graph.add_edge(f, t, kind);
                true
            }
            _ => false,
        }
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Serialize to `graph.json` at `path` — atomic write, same discipline
    /// every other durable file in this crate already follows.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let nodes: Vec<Node> = self.graph.node_weights().cloned().collect();
        let edges: Vec<(String, String, EdgeKind)> = self
            .graph
            .edge_indices()
            .filter_map(|e| {
                let (from, to) = self.graph.edge_endpoints(e)?;
                let kind = *self.graph.edge_weight(e)?;
                Some((self.graph[from].id.clone(), self.graph[to].id.clone(), kind))
            })
            .collect();
        let file = GraphFile { nodes, edges };
        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        crate::atomic::write(path, &json)
    }

    /// Load from `graph.json` at `path`. A missing file yields an empty
    /// graph rather than an error — `trm graph build` on a bank with no
    /// prior graph is the normal first-run case, not a failure.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let contents = std::fs::read_to_string(path)?;
        let file: GraphFile = serde_json::from_str(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut g = Self::new();
        for node in file.nodes {
            g.upsert_node(node);
        }
        for (from, to, kind) in file.edges {
            g.add_edge(&from, &to, kind);
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> CodeGraph {
        let mut g = CodeGraph::new();
        g.upsert_node(Node {
            id: "src/foo.rs".to_string(),
            kind: NodeKind::File,
            name: "foo.rs".to_string(),
        });
        g.upsert_node(Node {
            id: "src/foo.rs::Foo".to_string(),
            kind: NodeKind::Struct,
            name: "Foo".to_string(),
        });
        g.upsert_node(Node {
            id: "src/foo.rs::Foo::new".to_string(),
            kind: NodeKind::Function,
            name: "new".to_string(),
        });
        g.add_edge("src/foo.rs", "src/foo.rs::Foo", EdgeKind::Contains);
        g.add_edge("src/foo.rs", "src/foo.rs::Foo::new", EdgeKind::Contains);
        g
    }

    #[test]
    fn upsert_node_is_idempotent_by_id() {
        let mut g = CodeGraph::new();
        let node = Node {
            id: "src/foo.rs".to_string(),
            kind: NodeKind::File,
            name: "foo.rs".to_string(),
        };
        g.upsert_node(node.clone());
        g.upsert_node(node);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn upsert_node_replaces_weight_for_an_existing_id() {
        let mut g = CodeGraph::new();
        g.upsert_node(Node {
            id: "src/foo.rs::Foo".to_string(),
            kind: NodeKind::Struct,
            name: "Foo".to_string(),
        });
        g.upsert_node(Node {
            id: "src/foo.rs::Foo".to_string(),
            kind: NodeKind::Struct,
            name: "FooRenamed".to_string(),
        });
        let idx = g.node_index("src/foo.rs::Foo").unwrap();
        assert_eq!(g.graph[idx].name, "FooRenamed");
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn add_edge_between_known_nodes_succeeds() {
        let g = sample_graph();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn add_edge_with_an_unknown_endpoint_reports_false_without_panicking() {
        let mut g = CodeGraph::new();
        g.upsert_node(Node {
            id: "src/foo.rs".to_string(),
            kind: NodeKind::File,
            name: "foo.rs".to_string(),
        });
        assert!(!g.add_edge("src/foo.rs", "src/does_not_exist.rs", EdgeKind::Contains));
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn save_then_load_round_trips_nodes_and_edges() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("graph.json");
        let g = sample_graph();
        g.save(&path).unwrap();

        let loaded = CodeGraph::load(&path).unwrap();
        assert_eq!(loaded.node_count(), g.node_count());
        assert_eq!(loaded.edge_count(), g.edge_count());
        let idx = loaded.node_index("src/foo.rs::Foo::new").unwrap();
        assert_eq!(loaded.graph[idx].kind, NodeKind::Function);
        assert_eq!(loaded.graph[idx].name, "new");
    }

    #[test]
    fn load_of_a_missing_file_yields_an_empty_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("does-not-exist.json");
        let g = CodeGraph::load(&path).unwrap();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }
}
