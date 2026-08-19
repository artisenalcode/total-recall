//! Rust-specific structural extraction (originally Step 4 of
//! `docs/ideation/trm-code-graph/2026-08-19-scoped-graph-mvp-plan.md`,
//! since generalized to a per-language module — see `extract/mod.rs`).
//!
//! Deterministic AST facts only. Explicitly **not** attempted here: macro
//! expansion, cross-crate resolution, or real type inference — `Calls` and
//! `Implements` edges are resolved by matching an identifier's short text
//! against node *names* already seen in the same extraction run.

use crate::graph::model::{CodeGraph, EdgeKind, Node, NodeKind};
use tree_sitter::Node as TsNode;

pub fn ts_language() -> tree_sitter::Language {
    tree_sitter_rust::LANGUAGE.into()
}

fn text(node: &TsNode, source: &str) -> String {
    source[node.byte_range()].to_string()
}

/// Recursively walk `node`'s direct item children, emitting nodes/edges
/// into `graph`. `current_impl_type` is `Some(type_name)` only while
/// walking the body of an `impl` block, so a method's stable id can be
/// scoped under its type (`file.rs::Type::method`) instead of colliding
/// with a free function of the same name.
pub fn extract_items(
    node: TsNode,
    source: &str,
    rel: &str,
    current_impl_type: Option<&str>,
    graph: &mut CodeGraph,
    pending_calls: &mut Vec<(String, String)>,
    pending_impls: &mut Vec<(String, String)>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_item" => {
                let Some(name_node) = child.child_by_field_name("name") else {
                    continue;
                };
                let name = text(&name_node, source);
                let id = match current_impl_type {
                    Some(ty) => format!("{rel}::{ty}::{name}"),
                    None => format!("{rel}::{name}"),
                };
                graph.upsert_node(Node {
                    id: id.clone(),
                    kind: NodeKind::Function,
                    name,
                });
                graph.add_edge(rel, &id, EdgeKind::Contains);
                if let Some(body) = child.child_by_field_name("body") {
                    collect_calls(body, source, &id, pending_calls);
                }
            }
            "struct_item" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = text(&name_node, source);
                    let id = format!("{rel}::{name}");
                    graph.upsert_node(Node {
                        id: id.clone(),
                        kind: NodeKind::Struct,
                        name,
                    });
                    graph.add_edge(rel, &id, EdgeKind::Contains);
                }
            }
            "enum_item" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = text(&name_node, source);
                    let id = format!("{rel}::{name}");
                    graph.upsert_node(Node {
                        id: id.clone(),
                        kind: NodeKind::Enum,
                        name,
                    });
                    graph.add_edge(rel, &id, EdgeKind::Contains);
                }
            }
            "trait_item" => {
                // Node only, not its default methods — default-method
                // extraction is future scope, not attempted here.
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = text(&name_node, source);
                    let id = format!("{rel}::{name}");
                    graph.upsert_node(Node {
                        id: id.clone(),
                        kind: NodeKind::Trait,
                        name,
                    });
                    graph.add_edge(rel, &id, EdgeKind::Contains);
                }
            }
            "impl_item" => {
                let ty_name = child.child_by_field_name("type").map(|n| text(&n, source));
                let trait_name = child.child_by_field_name("trait").map(|n| text(&n, source));
                if let (Some(ty), Some(tr)) = (&ty_name, &trait_name) {
                    pending_impls.push((format!("{rel}::{ty}"), tr.clone()));
                }
                if let (Some(ty), Some(body)) = (&ty_name, child.child_by_field_name("body")) {
                    extract_items(
                        body,
                        source,
                        rel,
                        Some(ty.as_str()),
                        graph,
                        pending_calls,
                        pending_impls,
                    );
                }
            }
            _ => {}
        }
    }
}

/// Recursively collect every `call_expression` under `node` into
/// `pending_calls` as `(caller_id, callee_short_name)`. Does not descend
/// into nested item definitions specially — a closure or nested `fn`'s
/// calls are still attributed to the enclosing function, an accepted
/// simplification for MVP granularity.
fn collect_calls(
    node: TsNode,
    source: &str,
    caller_id: &str,
    pending_calls: &mut Vec<(String, String)>,
) {
    if node.kind() == "call_expression"
        && let Some(func_node) = node.child_by_field_name("function")
        && let Some(name) = callee_name(func_node, source)
    {
        pending_calls.push((caller_id.to_string(), name));
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_calls(child, source, caller_id, pending_calls);
    }
}

/// The short, identifier-level name a call expression's callee resolves
/// to for name-matching purposes: `foo()` -> `foo`, `Foo::new()` ->
/// `new`, `self.foo()` -> `foo`. Anything else yields `None` and is
/// simply not resolved — no error, just an edge that doesn't get added.
fn callee_name(node: TsNode, source: &str) -> Option<String> {
    match node.kind() {
        "identifier" => Some(text(&node, source)),
        "scoped_identifier" => node.child_by_field_name("name").map(|n| text(&n, source)),
        "field_expression" => node.child_by_field_name("field").map(|n| text(&n, source)),
        _ => None,
    }
}
