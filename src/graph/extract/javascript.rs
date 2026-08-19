//! TypeScript/TSX/JavaScript structural extraction. One extractor for all three -- TS/JS grammars share the same node/field names for
//! the shapes this crate cares about; TS/TSX additionally produce `interface_declaration`/`implements_clause`, which plain JS never emits.
//! Deterministic AST facts only. Not attempted: arrow-function expressions, decorators, or generic type resolution.

use crate::graph::model::{CodeGraph, EdgeKind, Node, NodeKind};
use tree_sitter::Node as TsNode;

pub fn ts_language() -> tree_sitter::Language {
    tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
}

pub fn tsx_language() -> tree_sitter::Language {
    tree_sitter_typescript::LANGUAGE_TSX.into()
}

pub fn js_language() -> tree_sitter::Language {
    tree_sitter_javascript::LANGUAGE.into()
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
    pending_impls: &mut Vec<(String, String)>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
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
            "class_declaration" => {
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

                for base in heritage_targets(child, source) {
                    pending_impls.push((class_id.clone(), base));
                }

                if let Some(body) = child.child_by_field_name("body") {
                    let mut body_cursor = body.walk();
                    for member in body.children(&mut body_cursor) {
                        if member.kind() == "method_definition"
                            && let Some(m_name_node) = member.child_by_field_name("name")
                        {
                            let m_name = text(&m_name_node, source);
                            let m_id = format!("{class_id}::{m_name}");
                            graph.upsert_node(Node {
                                id: m_id.clone(),
                                kind: NodeKind::Function,
                                name: m_name,
                            });
                            graph.add_edge(rel, &m_id, EdgeKind::Contains);
                            if let Some(m_body) = member.child_by_field_name("body") {
                                collect_calls(m_body, source, &m_id, pending_calls);
                            }
                        }
                    }
                }
            }
            "interface_declaration" => {
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
            _ => {}
        }
    }
}

/// Base-class/interface names a `class_declaration`'s heritage refers to. TS wraps them in `extends_clause`/`implements_clause`; plain
/// JS produces a bare `extends <identifier>` with no `implements_clause`. Both shapes handled here.
fn heritage_targets(class_node: TsNode, source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut top_cursor = class_node.walk();
    let Some(heritage) = class_node
        .children(&mut top_cursor)
        .find(|c| c.kind() == "class_heritage")
    else {
        return out;
    };

    let mut cursor = heritage.walk();
    for part in heritage.children(&mut cursor) {
        match part.kind() {
            "extends_clause" => {
                if let Some(value) = part.child_by_field_name("value") {
                    out.push(text(&value, source));
                }
            }
            "implements_clause" => {
                let mut ic = part.walk();
                for ty in part.children(&mut ic) {
                    if ty.kind() == "type_identifier" {
                        out.push(text(&ty, source));
                    }
                }
            }
            // Plain JS: no `extends_clause` wrapper.
            "identifier" => out.push(text(&part, source)),
            _ => {}
        }
    }
    out
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

/// `foo()` -> `foo`, `obj.foo()` -> `foo`. Anything else (a computed
/// member call, an IIFE, ...) yields `None` and is simply not resolved.
fn callee_name(node: TsNode, source: &str) -> Option<String> {
    match node.kind() {
        "identifier" => Some(text(&node, source)),
        "member_expression" => node
            .child_by_field_name("property")
            .map(|n| text(&n, source)),
        _ => None,
    }
}
