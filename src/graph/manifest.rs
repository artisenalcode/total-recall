//! Per-file content-hash manifest for `trm graph update` -- lets it re-extract only changed files and drop nodes for deleted ones.
//! Reuses `embed_cache::content_hash` rather than duplicating the hashing choice.

use crate::atomic;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest(HashMap<String, u64>);

impl Manifest {
    /// A missing or unparseable file is "no manifest yet", not a failure -- matches `CodeGraph::load`'s missing-file handling.
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        atomic::write(path, &json)
    }

    pub fn get(&self, rel: &str) -> Option<u64> {
        self.0.get(rel).copied()
    }

    pub fn set(&mut self, rel: String, hash: u64) {
        self.0.insert(rel, hash);
    }

    pub fn remove(&mut self, rel: &str) {
        self.0.remove(rel);
    }

    pub fn paths(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_then_load_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("manifest.json");
        let mut m = Manifest::default();
        m.set("src/lib.rs".to_string(), 42);
        m.set("src/main.rs".to_string(), 7);
        m.save(&path).unwrap();

        let loaded = Manifest::load(&path);
        assert_eq!(loaded.get("src/lib.rs"), Some(42));
        assert_eq!(loaded.get("src/main.rs"), Some(7));
    }

    #[test]
    fn load_of_a_missing_file_yields_an_empty_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let m = Manifest::load(&tmp.path().join("does-not-exist.json"));
        assert_eq!(m.paths().count(), 0);
    }

    #[test]
    fn remove_drops_an_entry() {
        let mut m = Manifest::default();
        m.set("src/lib.rs".to_string(), 1);
        m.remove("src/lib.rs");
        assert_eq!(m.get("src/lib.rs"), None);
    }
}
