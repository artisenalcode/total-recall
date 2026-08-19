//! tree-sitter-based structural extraction for Rust sources (Step 4 of
//! `docs/ideation/trm-code-graph/2026-08-19-scoped-graph-mvp-plan.md`).
//!
//! Deterministic AST facts only. Explicitly **not** attempted here: macro
//! expansion, cross-crate resolution, or real type inference — `Calls` and
//! `Implements` edges are resolved by matching an identifier's short text
//! against node *names* already seen in the same extraction run, the same
//! boundary graphify itself draws between its EXTRACTED and INFERRED
//! tiers (this crate only ever produces the EXTRACTED equivalent — no LLM
//! layer exists here at all, by design).

use crate::graph::manifest::Manifest;
use crate::graph::model::{CodeGraph, EdgeKind, Node, NodeKind};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tree_sitter::{Node as TsNode, Parser};

/// Walk `root` for `*.rs` files (skipping `target/` and any
/// dot-directory), parse each with tree-sitter-rust, and extract a fresh
/// `CodeGraph`: one `File` node per file, `Function`/`Struct`/`Enum`/
/// `Trait` nodes for top-level items (plus methods inside `impl` blocks),
/// `Contains` edges from each file to its items, `Implements` edges from
/// an `impl Trait for Type` block's type to that trait, and best-effort
/// `Calls` edges from a function to whatever function-named node its body
/// calls.
pub fn extract_dir(root: &Path) -> std::io::Result<CodeGraph> {
    let mut graph = CodeGraph::new();
    let files = collect_rust_files(root)?;
    let mut pending_calls: Vec<(String, String)> = Vec::new();
    let mut pending_impls: Vec<(String, String)> = Vec::new();
    extract_files(
        &files,
        root,
        &mut graph,
        &mut pending_calls,
        &mut pending_impls,
    );
    resolve_pending(&mut graph, pending_impls, pending_calls);
    Ok(graph)
}

/// Summary of one `update_dir` run, for the CLI to report.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UpdateSummary {
    pub files_changed: usize,
    pub files_deleted: usize,
    pub node_count: usize,
    pub edge_count: usize,
}

/// Incremental counterpart to `extract_dir`: loads the graph and
/// manifest already saved at `graph_path`/`manifest_path` (an empty
/// graph/manifest if this is the first run for this bank — the same
/// "missing file is not an error" convention `CodeGraph::load` and
/// `Manifest::load` already follow), re-extracts only files whose
/// content-hash changed or that are new, drops nodes for files deleted
/// since the last run, and saves both back.
///
/// Known limitation, accepted for this MVP (see the plan doc's Risks
/// section): a `Calls`/`Implements` edge from an *unchanged* file into a
/// symbol that a changed file renamed is not re-resolved to the new
/// name — only files actually re-extracted this run get their pending
/// calls/impls re-resolved. A full `build` always recomputes correctly.
pub fn update_dir(
    root: &Path,
    graph_path: &Path,
    manifest_path: &Path,
) -> std::io::Result<UpdateSummary> {
    let mut graph = CodeGraph::load(graph_path)?;
    let mut manifest = Manifest::load(manifest_path);

    let files = collect_rust_files(root)?;
    let mut current_rels: HashSet<String> = HashSet::new();
    let mut to_extract: Vec<PathBuf> = Vec::new();
    let mut changed_rels: HashSet<String> = HashSet::new();

    for path in &files {
        let rel = rel_of(path, root);
        current_rels.insert(rel.clone());
        let Ok(source) = std::fs::read_to_string(path) else {
            continue;
        };
        let hash = crate::embed_cache::content_hash(&source);
        if manifest.get(&rel) != Some(hash) {
            changed_rels.insert(rel);
            to_extract.push(path.clone());
        }
    }

    let deleted_rels: HashSet<String> = manifest
        .paths()
        .filter(|rel| !current_rels.contains(rel.as_str()))
        .cloned()
        .collect();

    let mut to_remove = changed_rels.clone();
    to_remove.extend(deleted_rels.iter().cloned());
    graph.remove_files(&to_remove);

    let mut pending_calls: Vec<(String, String)> = Vec::new();
    let mut pending_impls: Vec<(String, String)> = Vec::new();
    extract_files(
        &to_extract,
        root,
        &mut graph,
        &mut pending_calls,
        &mut pending_impls,
    );
    resolve_pending(&mut graph, pending_impls, pending_calls);

    for rel in &deleted_rels {
        manifest.remove(rel);
    }
    for path in &to_extract {
        let rel = rel_of(path, root);
        if let Ok(source) = std::fs::read_to_string(path) {
            manifest.set(rel, crate::embed_cache::content_hash(&source));
        }
    }

    graph.save(graph_path)?;
    manifest.save(manifest_path)?;

    Ok(UpdateSummary {
        files_changed: changed_rels.len(),
        files_deleted: deleted_rels.len(),
        node_count: graph.node_count(),
        edge_count: graph.edge_count(),
    })
}

/// Hash every current `*.rs` file under `root`, independent of any
/// existing graph or manifest — used after a full `extract_dir` (`trm
/// graph build`) to (re)seed the manifest so a later `trm graph update`
/// diffs against real content hashes instead of treating every file as
/// new on its very first run.
pub fn manifest_for(root: &Path) -> std::io::Result<Manifest> {
    let mut manifest = Manifest::default();
    for path in collect_rust_files(root)? {
        let rel = rel_of(&path, root);
        if let Ok(source) = std::fs::read_to_string(&path) {
            manifest.set(rel, crate::embed_cache::content_hash(&source));
        }
    }
    Ok(manifest)
}

fn rel_of(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Parse each of `files` (relative to `root`) and fold its nodes/edges
/// into `graph`, appending any not-yet-resolvable `Calls`/`Implements`
/// targets to `pending_calls`/`pending_impls` for `resolve_pending` to
/// settle afterward. Shared by both a full `extract_dir` run (every file)
/// and `update_dir` (only changed/new files).
fn extract_files(
    files: &[PathBuf],
    root: &Path,
    graph: &mut CodeGraph,
    pending_calls: &mut Vec<(String, String)>,
    pending_impls: &mut Vec<(String, String)>,
) {
    let language = tree_sitter_rust::LANGUAGE.into();
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .expect("tree-sitter-rust grammar failed to load");

    for path in files {
        let rel = rel_of(path, root);
        let Ok(source) = std::fs::read_to_string(path) else {
            // Non-UTF-8 or unreadable file — skip rather than fail the
            // whole extraction run over one file.
            continue;
        };

        graph.upsert_node(Node {
            id: rel.clone(),
            kind: NodeKind::File,
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
        });

        let Some(tree) = parser.parse(&source, None) else {
            continue;
        };
        extract_items(
            tree.root_node(),
            &source,
            &rel,
            None,
            graph,
            pending_calls,
            pending_impls,
        );
    }
}

/// Resolve every pending `Implements`/`Calls` target by short name
/// against `graph`'s current nodes (which may include files this run
/// didn't touch) — the second pass `extract_dir`'s own doc comment
/// describes, factored out so `update_dir` can run the same resolution
/// after a partial `extract_files` pass.
fn resolve_pending(
    graph: &mut CodeGraph,
    pending_impls: Vec<(String, String)>,
    pending_calls: Vec<(String, String)>,
) {
    for (type_id, trait_name) in pending_impls {
        if let Some(trait_id) = graph.find_by_name(NodeKind::Trait, &trait_name) {
            graph.add_edge(&type_id, &trait_id, EdgeKind::Implements);
        }
    }
    for (caller_id, callee_name) in pending_calls {
        if let Some(callee_id) = graph.find_by_name(NodeKind::Function, &callee_name) {
            graph.add_edge(&caller_id, &callee_id, EdgeKind::Calls);
        }
    }
}

fn text(node: &TsNode, source: &str) -> String {
    source[node.byte_range()].to_string()
}

/// Recursively walk `node`'s direct item children, emitting nodes/edges
/// into `graph`. `current_impl_type` is `Some(type_name)` only while
/// walking the body of an `impl` block, so a method's stable id can be
/// scoped under its type (`file.rs::Type::method`) instead of colliding
/// with a free function of the same name.
fn extract_items(
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
                // extraction is future scope (plan doc Risks), not
                // attempted here.
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
/// calls are still attributed to the enclosing function, which is an
/// accepted simplification for MVP granularity.
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
/// `new`, `self.foo()` -> `foo`. Anything else (e.g. a call through a
/// more complex expression) yields `None` and is simply not resolved —
/// no error, just an edge that doesn't get added.
fn callee_name(node: TsNode, source: &str) -> Option<String> {
    match node.kind() {
        "identifier" => Some(text(&node, source)),
        "scoped_identifier" => node.child_by_field_name("name").map(|n| text(&n, source)),
        "field_expression" => node.child_by_field_name("field").map(|n| text(&n, source)),
        _ => None,
    }
}

fn collect_rust_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                stack.push(path);
            } else if file_type.is_file() && path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_fixture(dir: &Path, rel: &str, contents: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn extracts_a_struct_its_impl_and_a_call_between_two_functions() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(
            tmp.path(),
            "src/lib.rs",
            r#"
                pub struct Greeter;

                impl Greeter {
                    pub fn greet(&self) -> String {
                        build_message()
                    }
                }

                fn build_message() -> String {
                    "hi".to_string()
                }
            "#,
        );

        let graph = extract_dir(tmp.path()).unwrap();

        assert!(
            graph.node_index("src/lib.rs").is_some(),
            "expected a File node"
        );
        assert!(
            graph.node_index("src/lib.rs::Greeter").is_some(),
            "expected a Struct node for Greeter"
        );
        assert!(
            graph.node_index("src/lib.rs::Greeter::greet").is_some(),
            "expected a Function node for the method"
        );
        assert!(
            graph.node_index("src/lib.rs::build_message").is_some(),
            "expected a Function node for the free function"
        );

        let caller = graph.node_index("src/lib.rs::Greeter::greet").unwrap();
        let callee = graph.node_index("src/lib.rs::build_message").unwrap();
        let has_calls_edge = graph
            .graph
            .edges_connecting(caller, callee)
            .any(|e| *e.weight() == EdgeKind::Calls);
        assert!(
            has_calls_edge,
            "expected a Calls edge from greet to build_message"
        );
    }

    #[test]
    fn extracts_an_implements_edge_for_a_trait_impl() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(
            tmp.path(),
            "src/lib.rs",
            r#"
                pub trait Speak {
                    fn speak(&self);
                }

                pub struct Dog;

                impl Speak for Dog {
                    fn speak(&self) {}
                }
            "#,
        );

        let graph = extract_dir(tmp.path()).unwrap();
        let dog = graph.node_index("src/lib.rs::Dog").unwrap();
        let speak = graph.node_index("src/lib.rs::Speak").unwrap();
        let has_implements_edge = graph
            .graph
            .edges_connecting(dog, speak)
            .any(|e| *e.weight() == EdgeKind::Implements);
        assert!(
            has_implements_edge,
            "expected an Implements edge from Dog to Speak"
        );
    }

    #[test]
    fn skips_target_and_hidden_directories() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path(), "src/lib.rs", "fn real() {}");
        write_fixture(tmp.path(), "target/debug/build.rs", "fn ignored() {}");
        write_fixture(tmp.path(), ".hidden/x.rs", "fn also_ignored() {}");

        let graph = extract_dir(tmp.path()).unwrap();
        assert!(graph.node_index("src/lib.rs::real").is_some());
        assert!(graph.find_by_name(NodeKind::Function, "ignored").is_none());
        assert!(
            graph
                .find_by_name(NodeKind::Function, "also_ignored")
                .is_none()
        );
    }

    fn graph_and_manifest_paths(tmp: &Path) -> (PathBuf, PathBuf) {
        (tmp.join("graph.json"), tmp.join("manifest.json"))
    }

    #[test]
    fn update_dir_on_a_bank_with_no_prior_graph_behaves_like_a_first_build() {
        let src = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        write_fixture(src.path(), "src/lib.rs", "fn real() {}");
        let (graph_path, manifest_path) = graph_and_manifest_paths(store.path());

        let summary = update_dir(src.path(), &graph_path, &manifest_path).unwrap();
        assert_eq!(summary.files_changed, 1);
        assert_eq!(summary.files_deleted, 0);

        let graph = CodeGraph::load(&graph_path).unwrap();
        assert!(graph.node_index("src/lib.rs::real").is_some());
    }

    #[test]
    fn update_dir_picks_up_a_newly_added_file_without_rescanning_unchanged_ones() {
        let src = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        write_fixture(src.path(), "src/lib.rs", "fn one() {}");
        let (graph_path, manifest_path) = graph_and_manifest_paths(store.path());
        update_dir(src.path(), &graph_path, &manifest_path).unwrap();

        write_fixture(src.path(), "src/extra.rs", "fn two() {}");
        let summary = update_dir(src.path(), &graph_path, &manifest_path).unwrap();

        assert_eq!(summary.files_changed, 1, "only the new file re-extracted");
        let graph = CodeGraph::load(&graph_path).unwrap();
        assert!(graph.node_index("src/lib.rs::one").is_some());
        assert!(graph.node_index("src/extra.rs::two").is_some());
    }

    #[test]
    fn update_dir_re_extracts_a_changed_files_new_content_and_drops_its_stale_nodes() {
        let src = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        write_fixture(src.path(), "src/lib.rs", "fn old_name() {}");
        let (graph_path, manifest_path) = graph_and_manifest_paths(store.path());
        update_dir(src.path(), &graph_path, &manifest_path).unwrap();

        write_fixture(src.path(), "src/lib.rs", "fn new_name() {}");
        let summary = update_dir(src.path(), &graph_path, &manifest_path).unwrap();

        assert_eq!(summary.files_changed, 1);
        let graph = CodeGraph::load(&graph_path).unwrap();
        assert!(graph.node_index("src/lib.rs::new_name").is_some());
        assert!(
            graph.node_index("src/lib.rs::old_name").is_none(),
            "stale node for the removed function should be gone"
        );
    }

    #[test]
    fn update_dir_drops_nodes_for_a_deleted_file() {
        let src = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        write_fixture(src.path(), "src/lib.rs", "fn stays() {}");
        write_fixture(src.path(), "src/gone.rs", "fn vanishes() {}");
        let (graph_path, manifest_path) = graph_and_manifest_paths(store.path());
        update_dir(src.path(), &graph_path, &manifest_path).unwrap();

        std::fs::remove_file(src.path().join("src/gone.rs")).unwrap();
        let summary = update_dir(src.path(), &graph_path, &manifest_path).unwrap();

        assert_eq!(summary.files_deleted, 1);
        assert_eq!(summary.files_changed, 0);
        let graph = CodeGraph::load(&graph_path).unwrap();
        assert!(graph.node_index("src/lib.rs::stays").is_some());
        assert!(graph.node_index("src/gone.rs").is_none());
        assert!(graph.node_index("src/gone.rs::vanishes").is_none());
    }

    #[test]
    fn update_dir_with_no_changes_is_a_no_op() {
        let src = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        write_fixture(src.path(), "src/lib.rs", "fn stable() {}");
        let (graph_path, manifest_path) = graph_and_manifest_paths(store.path());
        update_dir(src.path(), &graph_path, &manifest_path).unwrap();

        let summary = update_dir(src.path(), &graph_path, &manifest_path).unwrap();
        assert_eq!(summary.files_changed, 0);
        assert_eq!(summary.files_deleted, 0);
        assert_eq!(summary.node_count, 2); // File node + one Function node
    }
}
