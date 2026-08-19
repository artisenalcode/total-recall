//! Embedder-based node-name matching for `trm graph query` -- cosine similarity against node *names* only, no LLM, no body reasoning.

use crate::embeddings::{Embedder, cosine_similarity};
use crate::graph::model::CodeGraph;

/// Rank every node by cosine similarity between its `name` and `query`, highest first, truncated to `limit`. Returns `(node id, score)`.
pub fn semantic_query(
    graph: &CodeGraph,
    query: &str,
    embedder: &mut Embedder,
    limit: usize,
) -> Result<Vec<(String, f32)>, String> {
    let ids: Vec<String> = graph.graph.node_weights().map(|n| n.id.clone()).collect();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let names: Vec<String> = graph.graph.node_weights().map(|n| n.name.clone()).collect();

    let mut texts = names;
    texts.push(query.to_string());
    let mut vectors = embedder.embed(&texts)?;
    let query_vec = vectors.pop().expect("query text was just pushed");

    let mut scored: Vec<(String, f32)> = ids
        .into_iter()
        .zip(vectors)
        .map(|(id, v)| (id, cosine_similarity(&v, &query_vec)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::model::{EdgeKind, Node, NodeKind};

    fn embedder() -> Embedder {
        Embedder::new(crate::bank::data_root().join("models")).unwrap()
    }

    fn sample_graph() -> CodeGraph {
        let mut g = CodeGraph::new();
        g.upsert_node(Node {
            id: "src/graph/model.rs::CodeGraph::save".to_string(),
            kind: NodeKind::Function,
            name: "save".to_string(),
        });
        g.upsert_node(Node {
            id: "src/wiki.rs::slugify".to_string(),
            kind: NodeKind::Function,
            name: "slugify".to_string(),
        });
        g.add_edge(
            "src/graph/model.rs::CodeGraph::save",
            "src/wiki.rs::slugify",
            EdgeKind::Calls,
        );
        g
    }

    #[test]
    fn semantic_query_ranks_the_closer_name_match_first() {
        let g = sample_graph();
        let mut e = embedder();
        let results = semantic_query(&g, "persist to disk", &mut e, 5).unwrap();
        assert_eq!(results[0].0, "src/graph/model.rs::CodeGraph::save");
    }

    #[test]
    fn semantic_query_respects_the_requested_limit() {
        let g = sample_graph();
        let mut e = embedder();
        let results = semantic_query(&g, "anything", &mut e, 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn semantic_query_on_an_empty_graph_returns_no_matches() {
        let g = CodeGraph::new();
        let mut e = embedder();
        let results = semantic_query(&g, "anything", &mut e, 5).unwrap();
        assert!(results.is_empty());
    }
}
