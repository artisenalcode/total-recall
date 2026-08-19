//! Go structural extraction. Deterministic AST facts only.
//!
//! No `Implements` edges: Go's interface satisfaction is structural (no
//! `implements` keyword anywhere in the source), so there is no AST fact
//! to extract it from without real type-checking — out of scope for
//! this MVP, same posture as skipping type inference elsewhere in this
//! crate. `Trait`-kind nodes are still emitted for `interface` types (a
//! useful node to `path`/`god-nodes` even with no incoming edges yet).

use crate::graph::model::{CodeGraph, EdgeKind, Node, NodeKind};
use tree_sitter::Node as TsNode;

pub fn ts_language() -> tree_sitter::Language {
    tree_sitter_go::LANGUAGE.into()
}

fn text(node: &TsNode, source: &str) -> String {
    source[node.byte_range()].to_string()
}

pub fn extract_items(
    node: TsNode,
    source: &str,
    rel: &str,
    graph: &mut CodeGraph,
    pending_calls: &mut Vec<(String, String)>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "type_declaration" => {
                let mut spec_cursor = child.walk();
                for spec in child.children(&mut spec_cursor) {
                    if spec.kind() != "type_spec" {
                        continue;
                    }
                    let Some(name_node) = spec.child_by_field_name("name") else {
                        continue;
                    };
                    let Some(ty) = spec.child_by_field_name("type") else {
                        continue;
                    };
                    let kind = match ty.kind() {
                        "struct_type" => NodeKind::Struct,
                        "interface_type" => NodeKind::Trait,
                        _ => continue,
                    };
                    let name = text(&name_node, source);
                    let id = format!("{rel}::{name}");
                    graph.upsert_node(Node {
                        id: id.clone(),
                        kind,
                        name,
                    });
                    graph.add_edge(rel, &id, EdgeKind::Contains);
                }
            }
            "function_declaration" => {
                let Some(name_node) = child.child_by_field_name("name") else {
                    continue;
                };
                let name = text(&name_node, source);
                let id = format!("{rel}::{name}");
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
            "method_declaration" => {
                let Some(name_node) = child.child_by_field_name("name") else {
                    continue;
                };
                let name = text(&name_node, source);
                let id = match receiver_type_name(child, source) {
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
            _ => {}
        }
    }
}

/// `func (d Dog) Speak()` / `func (d *Dog) Speak()` -> `Some("Dog")`.
fn receiver_type_name(method_node: TsNode, source: &str) -> Option<String> {
    let receiver = method_node.child_by_field_name("receiver")?;
    let mut cursor = receiver.walk();
    let param = receiver
        .children(&mut cursor)
        .find(|c| c.kind() == "parameter_declaration")?;
    let ty = param.child_by_field_name("type")?;
    let ty = if ty.kind() == "pointer_type" {
        let mut pc = ty.walk();
        ty.children(&mut pc)
            .find(|c| c.kind() == "type_identifier")?
    } else {
        ty
    };
    Some(text(&ty, source))
}

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

/// `bark()` -> `bark`, `d.Speak()` -> `Speak`.
fn callee_name(node: TsNode, source: &str) -> Option<String> {
    match node.kind() {
        "identifier" => Some(text(&node, source)),
        "selector_expression" => node.child_by_field_name("field").map(|n| text(&n, source)),
        _ => None,
    }
}
