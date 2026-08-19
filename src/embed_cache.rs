//! Per-entry embedding cache, keyed by content hash, so `wiki::semantic_search` doesn't re-embed every file on every `recall` call.
//! Hand-rolled text format, not JSON -- this crate has no `serde_json` dependency. Any parse failure returns `None`, never a hard error.

use crate::atomic;
use std::fs;
use std::path::{Path, PathBuf};

/// One embedded window. `end_offset` (not a length) so a caller can slice `content[offset..end_offset]` directly.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedWindow {
    pub offset: usize,
    pub end_offset: usize,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntryCache {
    pub content_hash: u64,
    pub windows: Vec<CachedWindow>,
}

/// Same `DefaultHasher` technique `wiki::slugify` uses -- only needs to detect "did the content change," not resist collisions.
pub fn content_hash(content: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

fn cache_dir(wiki_dir: &Path) -> PathBuf {
    wiki_dir.join(".embeddings")
}

fn cache_path(wiki_dir: &Path, slug: &str) -> PathBuf {
    cache_dir(wiki_dir).join(format!("{slug}.cache"))
}

/// `None` on a missing file, I/O error, or parse failure -- callers treat it exactly like "not cached yet."
pub fn load(wiki_dir: &Path, slug: &str) -> Option<EntryCache> {
    let contents = fs::read_to_string(cache_path(wiki_dir, slug)).ok()?;
    parse(&contents)
}

pub fn save(wiki_dir: &Path, slug: &str, cache: &EntryCache) -> std::io::Result<()> {
    atomic::write(&cache_path(wiki_dir, slug), &serialize(cache))
}

fn serialize(cache: &EntryCache) -> String {
    let mut out = format!("hash {:x}\n", cache.content_hash);
    for w in &cache.windows {
        let vector: Vec<String> = w.vector.iter().map(|f| f.to_string()).collect();
        out.push_str(&format!(
            "w {} {} {}\n",
            w.offset,
            w.end_offset,
            vector.join(" ")
        ));
    }
    out
}

fn parse(contents: &str) -> Option<EntryCache> {
    let mut lines = contents.lines();
    let hash_line = lines.next()?;
    let content_hash = u64::from_str_radix(hash_line.strip_prefix("hash ")?, 16).ok()?;

    let mut windows = Vec::new();
    for line in lines {
        let mut parts = line.split_whitespace();
        if parts.next()? != "w" {
            return None;
        }
        let offset = parts.next()?.parse().ok()?;
        let end_offset = parts.next()?.parse().ok()?;
        let vector: Option<Vec<f32>> = parts.map(|p| p.parse().ok()).collect();
        windows.push(CachedWindow {
            offset,
            end_offset,
            vector: vector?,
        });
    }

    Some(EntryCache {
        content_hash,
        windows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_then_load_round_trips_exactly() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = EntryCache {
            content_hash: 0xdead_beef_cafe_babe,
            windows: vec![
                CachedWindow {
                    offset: 0,
                    end_offset: 42,
                    vector: vec![0.1, -0.2, 3.5, 0.0],
                },
                CachedWindow {
                    offset: 30,
                    end_offset: 80,
                    vector: vec![1.0, -1.0],
                },
            ],
        };
        save(tmp.path(), "my-slug", &cache).unwrap();
        let loaded = load(tmp.path(), "my-slug").unwrap();
        assert_eq!(loaded, cache);
    }

    #[test]
    fn load_on_missing_cache_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(load(tmp.path(), "never-saved").is_none());
    }

    #[test]
    fn load_on_corrupted_cache_returns_none_not_a_panic() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = cache_dir(tmp.path());
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("bad-slug.cache"),
            "not a valid cache file\ngarbage",
        )
        .unwrap();
        assert!(load(tmp.path(), "bad-slug").is_none());
    }

    #[test]
    fn content_hash_differs_for_different_content() {
        assert_ne!(content_hash("alpha"), content_hash("beta"));
    }

    #[test]
    fn content_hash_is_stable_for_identical_content() {
        assert_eq!(content_hash("same text"), content_hash("same text"));
    }

    #[test]
    fn save_creates_the_dot_embeddings_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = EntryCache {
            content_hash: 1,
            windows: vec![],
        };
        save(tmp.path(), "slug", &cache).unwrap();
        assert!(cache_dir(tmp.path()).is_dir());
    }

    #[test]
    fn cache_subdir_is_never_surfaced_by_wiki_list() {
        let tmp = tempfile::tempdir().unwrap();
        crate::wiki::write_named(tmp.path(), "real-entry", "hello world").unwrap();
        save(
            tmp.path(),
            "real-entry",
            &EntryCache {
                content_hash: 1,
                windows: vec![],
            },
        )
        .unwrap();
        let slugs = crate::wiki::list(tmp.path()).unwrap();
        assert_eq!(slugs, vec!["real-entry".to_string()]);
    }
}
