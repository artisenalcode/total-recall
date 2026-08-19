//! No-LLM, AST-derived code graph for Rust sources
//! (`docs/ideation/trm-code-graph/2026-08-19-scoped-graph-mvp-plan.md`).
//! `model` holds the graph type and its `graph.json` (de)serialization;
//! `extract` builds one from a directory via tree-sitter. Query verbs
//! (`query`/`path`/`god-nodes`) land in a follow-up pass — see the plan
//! doc's remaining Steps.

pub mod extract;
pub mod model;

use crate::bank::BankPaths;
use std::path::PathBuf;

/// Where a bank's `graph.json` lives — `paths.graph` is a directory tier
/// (parallel to `wiki`/`raw`/`sessions`), this is the one file in it.
pub fn graph_file_path(paths: &BankPaths) -> PathBuf {
    paths.graph.join("graph.json")
}

/// Rank nodes by total degree (in + out), highest first — graphify's own
/// "god nodes" concept: the nodes with the most fan-in/fan-out, a cheap
/// refactor-risk signal. Pure graph algorithm, no extraction or embedding
/// involved.
pub fn god_nodes(graph: &model::CodeGraph, n: usize) -> Vec<(String, usize)> {
    use petgraph::Direction;
    let mut ranked: Vec<(String, usize)> = graph
        .graph
        .node_indices()
        .map(|idx| {
            let degree = graph.graph.edges_directed(idx, Direction::Incoming).count()
                + graph.graph.edges_directed(idx, Direction::Outgoing).count();
            (graph.graph[idx].id.clone(), degree)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(n);
    ranked
}

/// Shortest path between two node ids — unweighted BFS following edge
/// direction (no edge weights exist in this graph yet, and traversing
/// `Calls`/`Contains`/`Implements` forward is the meaningful direction for
/// "how does A reach B"). Returns the full id sequence including both
/// endpoints, or `None` if either id is unknown or no directed path
/// exists between them.
pub fn shortest_path(graph: &model::CodeGraph, from: &str, to: &str) -> Option<Vec<String>> {
    use petgraph::visit::EdgeRef;
    use std::collections::{HashMap, HashSet, VecDeque};

    let start = graph.node_index(from)?;
    let goal = graph.node_index(to)?;
    if start == goal {
        return Some(vec![from.to_string()]);
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut prev = HashMap::new();
    visited.insert(start);
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == goal {
            break;
        }
        for edge in graph.graph.edges(current) {
            let next = edge.target();
            if visited.insert(next) {
                prev.insert(next, current);
                queue.push_back(next);
            }
        }
    }

    if !visited.contains(&goal) {
        return None;
    }

    let mut path = vec![goal];
    let mut node = goal;
    while node != start {
        node = *prev.get(&node)?;
        path.push(node);
    }
    path.reverse();
    Some(
        path.into_iter()
            .map(|idx| graph.graph[idx].id.clone())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::model::{CodeGraph, EdgeKind, Node, NodeKind};

    fn hub_graph() -> CodeGraph {
        // hub <- a, hub <- b, hub -> c : degree(hub) = 3, others = 1.
        let mut g = CodeGraph::new();
        for id in ["hub", "a", "b", "c"] {
            g.upsert_node(Node {
                id: id.to_string(),
                kind: NodeKind::Function,
                name: id.to_string(),
            });
        }
        g.add_edge("a", "hub", EdgeKind::Calls);
        g.add_edge("b", "hub", EdgeKind::Calls);
        g.add_edge("hub", "c", EdgeKind::Calls);
        g
    }

    #[test]
    fn god_nodes_ranks_highest_degree_first() {
        let g = hub_graph();
        let ranked = god_nodes(&g, 2);
        assert_eq!(ranked[0].0, "hub");
        assert_eq!(ranked[0].1, 3);
    }

    #[test]
    fn god_nodes_respects_the_requested_limit() {
        let g = hub_graph();
        assert_eq!(god_nodes(&g, 1).len(), 1);
        assert_eq!(god_nodes(&g, 10).len(), 4);
    }

    #[test]
    fn graph_file_path_is_graph_dot_json_under_the_graph_tier() {
        let paths = crate::bank::paths_for(std::path::Path::new("/tmp/root"), "global");
        assert_eq!(
            graph_file_path(&paths),
            std::path::Path::new("/tmp/root/banks/global/graph/graph.json")
        );
    }

    #[test]
    fn shortest_path_follows_a_multi_hop_directed_chain() {
        let g = hub_graph();
        // a -> hub -> c
        let path = shortest_path(&g, "a", "c").unwrap();
        assert_eq!(path, vec!["a", "hub", "c"]);
    }

    #[test]
    fn shortest_path_returns_none_when_no_directed_path_exists() {
        let g = hub_graph();
        // c has no outgoing edges -- nothing reachable from it.
        assert!(shortest_path(&g, "c", "a").is_none());
    }

    #[test]
    fn shortest_path_returns_none_for_an_unknown_endpoint() {
        let g = hub_graph();
        assert!(shortest_path(&g, "a", "does-not-exist").is_none());
    }

    #[test]
    fn shortest_path_of_a_node_to_itself_is_a_single_element_path() {
        let g = hub_graph();
        assert_eq!(
            shortest_path(&g, "hub", "hub"),
            Some(vec!["hub".to_string()])
        );
    }
}
