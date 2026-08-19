//! Rust-specific structural extraction. Deterministic AST facts only -- no macro expansion, cross-crate resolution, or type inference.

use crate::graph::model::{CodeGraph, EdgeKind, Node, NodeKind};
use tree_sitter::Node as TsNode;

pub fn ts_language() -> tree_sitter::Language {
    tree_sitter_rust::LANGUAGE.into()
}

fn text(node: &TsNode, source: &str) -> String {
    source[node.byte_range()].to_string()
}

/// Recursively walk `node`'s direct item children. `current_impl_type` is `Some` only inside an `impl` block, scoping a method's id
/// under its type (`file.rs::Type::method`) instead of colliding with a free function of the same name.
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
                // Node only, not its default methods.
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

/// Recursively collect every `call_expression` under `node` as `(caller_id, callee_short_name)`. A closure/nested `fn`'s calls are
/// still attributed to the enclosing function.
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

/// The short callee name for name-matching: `foo()` -> `foo`, `Foo::new()` -> `new`, `self.foo()` -> `foo`. Else `None`, no error.
fn callee_name(node: TsNode, source: &str) -> Option<String> {
    match node.kind() {
        "identifier" => Some(text(&node, source)),
        "scoped_identifier" => node.child_by_field_name("name").map(|n| text(&n, source)),
        "field_expression" => node.child_by_field_name("field").map(|n| text(&n, source)),
        _ => None,
    }
}
