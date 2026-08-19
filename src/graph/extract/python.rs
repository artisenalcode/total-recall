//! Python structural extraction. Deterministic AST facts only.
//!
//! A class's base classes (`class Dog(Animal):`) are emitted as pending
//! `Implements`-style edges, resolved against any node — `Struct` or
//! `Trait` — matching that name, since Python has no separate
//! interface/trait kind to disambiguate against (unlike Rust). Not
//! attempted: decorators, `self`-attribute call targets beyond a plain
//! `self.method()`, multiple inheritance beyond the first base's edge —
//! every base still gets its own pending edge, just no MRO reasoning.

use crate::graph::model::{CodeGraph, EdgeKind, Node, NodeKind};
use tree_sitter::Node as TsNode;

pub fn ts_language() -> tree_sitter::Language {
    tree_sitter_python::LANGUAGE.into()
}

fn text(node: &TsNode, source: &str) -> String {
    source[node.byte_range()].to_string()
}

/// `current_class` scopes a nested `function_definition`'s id under its
/// enclosing class, same convention as the Rust extractor's
/// `current_impl_type`.
pub fn extract_items(
    node: TsNode,
    source: &str,
    rel: &str,
    current_class: Option<&str>,
    graph: &mut CodeGraph,
    pending_calls: &mut Vec<(String, String)>,
    pending_impls: &mut Vec<(String, String)>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_definition" => {
                let Some(name_node) = child.child_by_field_name("name") else {
                    continue;
                };
                let name = text(&name_node, source);
                let id = match current_class {
                    Some(cls) => format!("{rel}::{cls}::{name}"),
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
            "class_definition" => {
                let Some(name_node) = child.child_by_field_name("name") else {
                    continue;
                };
                let class_name = text(&name_node, source);
                let class_id = format!("{rel}::{class_name}");
                graph.upsert_node(Node {
                    id: class_id.clone(),
                    kind: NodeKind::Struct,
                    name: class_name.clone(),
                });
                graph.add_edge(rel, &class_id, EdgeKind::Contains);

                if let Some(bases) = child.child_by_field_name("superclasses") {
                    let mut bc = bases.walk();
                    for base in bases.children(&mut bc) {
                        if base.kind() == "identifier" {
                            pending_impls.push((class_id.clone(), text(&base, source)));
                        }
                    }
                }

                if let Some(body) = child.child_by_field_name("body") {
                    extract_items(
                        body,
                        source,
                        rel,
                        Some(class_name.as_str()),
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

fn collect_calls(
    node: TsNode,
    source: &str,
    caller_id: &str,
    pending_calls: &mut Vec<(String, String)>,
) {
    if node.kind() == "call"
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

/// `foo()` -> `foo`, `self.foo()` -> `foo`.
fn callee_name(node: TsNode, source: &str) -> Option<String> {
    match node.kind() {
        "identifier" => Some(text(&node, source)),
        "attribute" => node
            .child_by_field_name("attribute")
            .map(|n| text(&n, source)),
        _ => None,
    }
}
