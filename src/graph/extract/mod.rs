//! tree-sitter-based structural extraction, multi-language (Rust, TS/TSX/JS, Go, Python) behind one `Language` dispatch.
//! Deterministic AST facts only: no macro/generic expansion, no cross-file type resolution. `Calls`/`Implements` edges resolve by
//! matching an identifier's short text against node names already seen in the same extraction run.

mod go;
mod javascript;
mod python;
mod rust;

use crate::graph::manifest::Manifest;
use crate::graph::model::{CodeGraph, EdgeKind, NodeKind};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use tree_sitter::Parser;

/// Directory names skipped everywhere, on top of any dot-directory -- build/dependency output for the supported languages.
const SKIP_DIRS: &[&str] = &[
    "target",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    "vendor",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Language {
    Rust,
    TypeScript,
    Tsx,
    JavaScript,
    Go,
    Python,
}

impl Language {
    fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Language::Rust),
            "ts" => Some(Language::TypeScript),
            "tsx" => Some(Language::Tsx),
            "js" | "jsx" | "mjs" | "cjs" => Some(Language::JavaScript),
            "go" => Some(Language::Go),
            "py" => Some(Language::Python),
            _ => None,
        }
    }

    fn ts_language(&self) -> tree_sitter::Language {
        match self {
            Language::Rust => rust::ts_language(),
            Language::TypeScript => javascript::ts_language(),
            Language::Tsx => javascript::tsx_language(),
            Language::JavaScript => javascript::js_language(),
            Language::Go => go::ts_language(),
            Language::Python => python::ts_language(),
        }
    }
}

/// Walk `root` for supported source files, parse each with the matching tree-sitter grammar, and extract a fresh `CodeGraph`.
pub fn extract_dir(root: &Path) -> std::io::Result<CodeGraph> {
    let mut graph = CodeGraph::new();
    let files = collect_source_files(root)?;
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

/// Incremental counterpart to `extract_dir`: re-extracts only files whose content-hash changed or that are new, drops nodes for deleted
/// files, and saves both graph and manifest back. Known limitation: an edge from an *unchanged* file into a symbol a changed file
/// renamed isn't re-resolved -- only files re-extracted this run get their pending calls/impls resolved. A full `build` always recomputes.
pub fn update_dir(
    root: &Path,
    graph_path: &Path,
    manifest_path: &Path,
) -> std::io::Result<UpdateSummary> {
    let mut graph = CodeGraph::load(graph_path)?;
    let mut manifest = Manifest::load(manifest_path);

    let files = collect_source_files(root)?;
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

/// Hash every current supported-language file under `root`, used to (re)seed the manifest after a full `extract_dir` build.
pub fn manifest_for(root: &Path) -> std::io::Result<Manifest> {
    let mut manifest = Manifest::default();
    for path in collect_source_files(root)? {
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

/// Parse each of `files` with the grammar matching its extension and fold its nodes/edges into `graph`, queuing unresolvable
/// `Calls`/`Implements` targets for `resolve_pending`. One `Parser` per language, built lazily and reused.
fn extract_files(
    files: &[PathBuf],
    root: &Path,
    graph: &mut CodeGraph,
    pending_calls: &mut Vec<(String, String)>,
    pending_impls: &mut Vec<(String, String)>,
) {
    let mut parsers: HashMap<Language, Parser> = HashMap::new();

    for path in files {
        let Some(lang) = path
            .extension()
            .and_then(|e| e.to_str())
            .and_then(Language::from_extension)
        else {
            continue;
        };
        let rel = rel_of(path, root);
        let Ok(source) = std::fs::read_to_string(path) else {
            // Non-UTF-8 or unreadable -- skip rather than fail the whole run.
            continue;
        };

        graph.upsert_node(crate::graph::model::Node {
            id: rel.clone(),
            kind: NodeKind::File,
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
        });

        let parser = parsers.entry(lang).or_insert_with(|| {
            let mut p = Parser::new();
            p.set_language(&lang.ts_language())
                .expect("tree-sitter grammar failed to load");
            p
        });
        let Some(tree) = parser.parse(&source, None) else {
            continue;
        };

        match lang {
            Language::Rust => rust::extract_items(
                tree.root_node(),
                &source,
                &rel,
                None,
                graph,
                pending_calls,
                pending_impls,
            ),
            Language::TypeScript | Language::Tsx | Language::JavaScript => {
                javascript::extract_items(
                    tree.root_node(),
                    &source,
                    &rel,
                    graph,
                    pending_calls,
                    pending_impls,
                )
            }
            Language::Go => {
                go::extract_items(tree.root_node(), &source, &rel, graph, pending_calls)
            }
            Language::Python => python::extract_items(
                tree.root_node(),
                &source,
                &rel,
                None,
                graph,
                pending_calls,
                pending_impls,
            ),
        }
    }
}

/// Resolve every pending `Implements`/`Calls` target by short name against `graph`'s current nodes. `Implements` tries `Trait` first,
/// then falls back to `Struct` (class inheritance in TS/JS/Python has no separate "Inherits" edge kind).
fn resolve_pending(
    graph: &mut CodeGraph,
    pending_impls: Vec<(String, String)>,
    pending_calls: Vec<(String, String)>,
) {
    for (type_id, target_name) in pending_impls {
        let target = graph
            .find_by_name(NodeKind::Trait, &target_name)
            .or_else(|| graph.find_by_name(NodeKind::Struct, &target_name));
        if let Some(target_id) = target {
            graph.add_edge(&type_id, &target_id, EdgeKind::Implements);
        }
    }
    for (caller_id, callee_name) in pending_calls {
        if let Some(callee_id) = graph.find_by_name(NodeKind::Function, &callee_name) {
            graph.add_edge(&caller_id, &callee_id, EdgeKind::Calls);
        }
    }
}

fn collect_source_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
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
                if name.starts_with('.') || SKIP_DIRS.contains(&name.as_ref()) {
                    continue;
                }
                stack.push(path);
            } else if file_type.is_file()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| Language::from_extension(e).is_some())
            {
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
    use crate::graph::model::NodeKind;
    use std::fs;

    fn write_fixture(dir: &Path, rel: &str, contents: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    // --- Rust (unchanged behavior, moved from the old single-file test suite) ---

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

        assert!(graph.node_index("src/lib.rs").is_some());
        assert!(graph.node_index("src/lib.rs::Greeter").is_some());
        assert!(graph.node_index("src/lib.rs::Greeter::greet").is_some());
        assert!(graph.node_index("src/lib.rs::build_message").is_some());

        let caller = graph.node_index("src/lib.rs::Greeter::greet").unwrap();
        let callee = graph.node_index("src/lib.rs::build_message").unwrap();
        let has_calls_edge = graph
            .graph
            .edges_connecting(caller, callee)
            .any(|e| *e.weight() == EdgeKind::Calls);
        assert!(has_calls_edge);
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
        assert!(has_implements_edge);
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

    #[test]
    fn skips_node_modules_and_pycache_directories() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path(), "src/lib.rs", "fn real() {}");
        write_fixture(
            tmp.path(),
            "node_modules/pkg/index.js",
            "function ignored() {}",
        );
        write_fixture(tmp.path(), "__pycache__/x.py", "def also_ignored(): pass");

        let graph = extract_dir(tmp.path()).unwrap();
        assert!(graph.node_index("src/lib.rs::real").is_some());
        assert!(graph.find_by_name(NodeKind::Function, "ignored").is_none());
        assert!(
            graph
                .find_by_name(NodeKind::Function, "also_ignored")
                .is_none()
        );
    }

    // --- TypeScript ---

    #[test]
    fn typescript_extracts_class_interface_implements_and_a_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(
            tmp.path(),
            "src/dog.ts",
            r#"
                interface Speak {
                    speak(): void;
                }

                class Dog implements Speak {
                    speak(): void {
                        bark();
                    }
                }

                function bark() {}
            "#,
        );

        let graph = extract_dir(tmp.path()).unwrap();
        assert!(graph.node_index("src/dog.ts::Dog").is_some());
        assert!(graph.node_index("src/dog.ts::Speak").is_some());
        assert!(graph.node_index("src/dog.ts::Dog::speak").is_some());
        assert!(graph.node_index("src/dog.ts::bark").is_some());

        let dog = graph.node_index("src/dog.ts::Dog").unwrap();
        let speak = graph.node_index("src/dog.ts::Speak").unwrap();
        assert!(
            graph
                .graph
                .edges_connecting(dog, speak)
                .any(|e| *e.weight() == EdgeKind::Implements)
        );

        let caller = graph.node_index("src/dog.ts::Dog::speak").unwrap();
        let callee = graph.node_index("src/dog.ts::bark").unwrap();
        assert!(
            graph
                .graph
                .edges_connecting(caller, callee)
                .any(|e| *e.weight() == EdgeKind::Calls)
        );
    }

    // --- JavaScript ---

    #[test]
    fn javascript_extracts_class_extends_and_a_method_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(
            tmp.path(),
            "src/dog.js",
            r#"
                class Animal {}

                class Dog extends Animal {
                    speak() {
                        bark();
                    }
                }

                function bark() {}
            "#,
        );

        let graph = extract_dir(tmp.path()).unwrap();
        let dog = graph.node_index("src/dog.js::Dog").unwrap();
        let animal = graph.node_index("src/dog.js::Animal").unwrap();
        assert!(
            graph
                .graph
                .edges_connecting(dog, animal)
                .any(|e| *e.weight() == EdgeKind::Implements)
        );
        assert!(graph.node_index("src/dog.js::Dog::speak").is_some());
    }

    // --- Go ---

    #[test]
    fn go_extracts_struct_interface_and_a_method_scoped_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(
            tmp.path(),
            "dog.go",
            r#"
                package main

                type Speaker interface {
                    Speak()
                }

                type Dog struct {
                    Name string
                }

                func (d Dog) Speak() {
                    bark()
                }

                func bark() {}
            "#,
        );

        let graph = extract_dir(tmp.path()).unwrap();
        assert!(graph.node_index("dog.go::Speaker").is_some());
        assert!(graph.node_index("dog.go::Dog").is_some());
        assert!(graph.node_index("dog.go::Dog::Speak").is_some());

        let caller = graph.node_index("dog.go::Dog::Speak").unwrap();
        let callee = graph.node_index("dog.go::bark").unwrap();
        assert!(
            graph
                .graph
                .edges_connecting(caller, callee)
                .any(|e| *e.weight() == EdgeKind::Calls)
        );
    }

    // --- Python ---

    #[test]
    fn python_extracts_class_base_and_a_method_scoped_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(
            tmp.path(),
            "dog.py",
            r#"
class Animal:
    pass

class Dog(Animal):
    def speak(self):
        bark()

def bark():
    pass
"#,
        );

        let graph = extract_dir(tmp.path()).unwrap();
        let dog = graph.node_index("dog.py::Dog").unwrap();
        let animal = graph.node_index("dog.py::Animal").unwrap();
        assert!(
            graph
                .graph
                .edges_connecting(dog, animal)
                .any(|e| *e.weight() == EdgeKind::Implements)
        );

        let caller = graph.node_index("dog.py::Dog::speak").unwrap();
        let callee = graph.node_index("dog.py::bark").unwrap();
        assert!(
            graph
                .graph
                .edges_connecting(caller, callee)
                .any(|e| *e.weight() == EdgeKind::Calls)
        );
    }

    // --- Incremental update across languages ---

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
    fn update_dir_picks_up_a_newly_added_file_of_a_different_language() {
        let src = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        write_fixture(src.path(), "src/lib.rs", "fn one() {}");
        let (graph_path, manifest_path) = graph_and_manifest_paths(store.path());
        update_dir(src.path(), &graph_path, &manifest_path).unwrap();

        write_fixture(src.path(), "src/extra.py", "def two():\n    pass\n");
        let summary = update_dir(src.path(), &graph_path, &manifest_path).unwrap();

        assert_eq!(summary.files_changed, 1, "only the new file re-extracted");
        let graph = CodeGraph::load(&graph_path).unwrap();
        assert!(graph.node_index("src/lib.rs::one").is_some());
        assert!(graph.node_index("src/extra.py::two").is_some());
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
        assert!(graph.node_index("src/lib.rs::old_name").is_none());
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
