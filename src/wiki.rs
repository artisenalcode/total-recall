use crate::atomic;
use std::fs;
use std::io;
use std::path::Path;

/// Derive a filesystem-safe, human-legible slug from memory content: a
/// kebab-cased prefix of the text plus a short content hash, so two
/// entries with the same opening words never collide.
pub fn slugify(content: &str) -> String {
    let prefix: String = content
        .chars()
        .take(40)
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let collapsed = prefix
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let collapsed = if collapsed.is_empty() {
        "memory".to_string()
    } else {
        collapsed
    };

    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{collapsed}-{:06x}", hasher.finish() & 0xff_ffff)
}

/// Write `content` into `wiki_dir/<slug>.md` atomically. Returns the slug used.
pub fn write(wiki_dir: &Path, content: &str) -> io::Result<String> {
    let slug = slugify(content);
    let final_path = wiki_dir.join(format!("{slug}.md"));
    atomic::write(&final_path, content)?;
    Ok(slug)
}

/// Write `content` under an exact, caller-chosen slug -- no re-hashing, unlike `write`.
pub fn write_named(wiki_dir: &Path, slug: &str, content: &str) -> io::Result<()> {
    let final_path = wiki_dir.join(format!("{slug}.md"));
    atomic::write(&final_path, content)
}

pub fn read(wiki_dir: &Path, slug: &str) -> io::Result<String> {
    fs::read_to_string(wiki_dir.join(format!("{slug}.md")))
}

/// List every stored slug in `wiki_dir` (empty if the dir doesn't exist yet).
pub fn list(wiki_dir: &Path) -> io::Result<Vec<String>> {
    if !wiki_dir.exists() {
        return Ok(Vec::new());
    }
    let mut slugs = Vec::new();
    for entry in fs::read_dir(wiki_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(slug) = name.strip_suffix(".md")
            && !name.starts_with('.')
        {
            slugs.push(slug.to_string());
        }
    }
    slugs.sort();
    Ok(slugs)
}

/// A single ranked recall hit.
#[derive(Debug, PartialEq)]
pub struct RankedMatch {
    pub slug: String,
    pub score: f32,
    pub snippet: String,
    /// Char offset of the window that produced this match's score, so a caller can tell which part of a long file matched. 0 for a single-window file.
    pub window_offset: usize,
}

/// Factored out purely to satisfy `clippy::type_complexity`, no behavioral meaning beyond that.
type ResolvedWindow = (usize, usize, Vec<f32>);
type ResolvedWindows = Vec<Option<Vec<ResolvedWindow>>>;

/// Semantic recall: rank every entry in `wiki_dir` by cosine similarity to `query`, returning matches at or above `min_score`, highest first.
pub fn semantic_search(
    wiki_dir: &Path,
    query: &str,
    embedder: &mut crate::embeddings::Embedder,
    min_score: f32,
    limit: usize,
) -> Result<Vec<RankedMatch>, String> {
    let slugs = list(wiki_dir).map_err(|e| e.to_string())?;
    if slugs.is_empty() {
        return Ok(Vec::new());
    }

    // Per-slug resolved windows, reused from cache or freshly embedded below; `None` is filled in once the batch embed call returns.
    let mut contents: Vec<String> = Vec::with_capacity(slugs.len());
    let mut resolved: ResolvedWindows = Vec::with_capacity(slugs.len());

    // Batch-embed only what's dirty (missing or stale cache) -- an unchanged bank costs one query embed and zero file re-embeds.
    let mut texts = vec![query.to_string()];
    let mut dirty_slug_idx = Vec::new();
    let mut dirty_window_span = Vec::new(); // (offset, end_offset) per queued text

    for (slug_idx, slug) in slugs.iter().enumerate() {
        let content = read(wiki_dir, slug).map_err(|e| e.to_string())?;
        let hash = crate::embed_cache::content_hash(&content);

        match crate::embed_cache::load(wiki_dir, slug) {
            Some(cache) if cache.content_hash == hash => {
                resolved.push(Some(
                    cache
                        .windows
                        .into_iter()
                        .map(|w| (w.offset, w.end_offset, w.vector))
                        .collect(),
                ));
            }
            _ => {
                resolved.push(None); // filled in after the batch embed below
                for (offset, window_text) in crate::window::windows(
                    &content,
                    crate::window::WINDOW_WORDS,
                    crate::window::OVERLAP_WORDS,
                ) {
                    let end_offset = offset + window_text.len();
                    texts.push(window_text);
                    dirty_slug_idx.push(slug_idx);
                    dirty_window_span.push((offset, end_offset));
                }
            }
        }
        contents.push(content);
    }

    if !dirty_slug_idx.is_empty() {
        let vectors = embedder.embed(&texts)?;
        // vectors[0] is the query; dirty windows start at vectors[1].
        let mut per_slug: Vec<Vec<(usize, usize, Vec<f32>)>> = vec![Vec::new(); slugs.len()];
        for (i, &slug_idx) in dirty_slug_idx.iter().enumerate() {
            let (offset, end_offset) = dirty_window_span[i];
            per_slug[slug_idx].push((offset, end_offset, vectors[i + 1].clone()));
        }
        for &slug_idx in &dirty_slug_idx {
            if resolved[slug_idx].is_none() {
                let windows = std::mem::take(&mut per_slug[slug_idx]);
                let cache = crate::embed_cache::EntryCache {
                    content_hash: crate::embed_cache::content_hash(&contents[slug_idx]),
                    windows: windows
                        .iter()
                        .map(
                            |(offset, end_offset, vector)| crate::embed_cache::CachedWindow {
                                offset: *offset,
                                end_offset: *end_offset,
                                vector: vector.clone(),
                            },
                        )
                        .collect(),
                };
                let _ = crate::embed_cache::save(wiki_dir, &slugs[slug_idx], &cache);
                resolved[slug_idx] = Some(windows);
            }
        }

        // The query only needs embedding when something was actually dirty.
        let query_vec = vectors[0].clone();
        return rank(slugs, contents, resolved, &query_vec, min_score, limit);
    }

    // Fully cached: still need the query embedded to score against it.
    let query_vec = embedder.embed(&[query.to_string()])?.remove(0);
    rank(slugs, contents, resolved, &query_vec, min_score, limit)
}

fn rank(
    slugs: Vec<String>,
    contents: Vec<String>,
    resolved: ResolvedWindows,
    query_vec: &[f32],
    min_score: f32,
    limit: usize,
) -> Result<Vec<RankedMatch>, String> {
    let mut matches: Vec<RankedMatch> = slugs
        .into_iter()
        .zip(contents)
        .zip(resolved)
        .filter_map(|((slug, content), windows)| {
            let windows = windows?;
            windows
                .iter()
                .map(|(offset, end_offset, vector)| {
                    let score = crate::embeddings::cosine_similarity(query_vec, vector);
                    (score, *offset, *end_offset)
                })
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(score, offset, end_offset)| RankedMatch {
                    slug,
                    score,
                    snippet: snippet_of(content[offset..end_offset].trim_end()),
                    window_offset: offset,
                })
        })
        .filter(|m| m.score >= min_score)
        .collect();

    matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    matches.truncate(limit);
    Ok(matches)
}

fn snippet_of(content: &str) -> String {
    let one_line: String = content.split_whitespace().collect::<Vec<_>>().join(" ");
    let snippet: String = one_line.chars().take(120).collect();
    if one_line.chars().count() > 120 {
        format!("{snippet}…")
    } else {
        snippet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_round_trips_content() {
        let tmp = tempfile::tempdir().unwrap();
        let slug = write(tmp.path(), "user prefers terse commit messages").unwrap();
        assert_eq!(
            read(tmp.path(), &slug).unwrap(),
            "user prefers terse commit messages"
        );
    }

    #[test]
    fn write_leaves_no_stray_temp_files_behind() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "some memory").unwrap();
        let leftovers: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp file(s) left behind: {leftovers:?}"
        );
    }

    #[test]
    fn write_named_preserves_the_exact_given_slug() {
        let tmp = tempfile::tempdir().unwrap();
        write_named(tmp.path(), "boris-cherny", "some frontmatter + content").unwrap();
        assert_eq!(
            read(tmp.path(), "boris-cherny").unwrap(),
            "some frontmatter + content"
        );
    }

    #[test]
    fn two_different_contents_do_not_collide() {
        let tmp = tempfile::tempdir().unwrap();
        let slug_a = write(tmp.path(), "fact A").unwrap();
        let slug_b = write(tmp.path(), "fact B").unwrap();
        assert_ne!(slug_a, slug_b);
        assert_eq!(read(tmp.path(), &slug_a).unwrap(), "fact A");
        assert_eq!(read(tmp.path(), &slug_b).unwrap(), "fact B");
    }

    #[test]
    fn list_returns_all_written_slugs_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        let slug_a = write(tmp.path(), "zzz last").unwrap();
        let slug_b = write(tmp.path(), "aaa first").unwrap();
        let mut expected = vec![slug_a, slug_b];
        expected.sort();
        assert_eq!(list(tmp.path()).unwrap(), expected);
    }

    #[test]
    fn list_on_missing_dir_returns_empty_not_error() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist-yet");
        assert_eq!(list(&missing).unwrap(), Vec::<String>::new());
    }

    fn embedder() -> crate::embeddings::Embedder {
        crate::embeddings::Embedder::new(crate::bank::data_root().join("models")).unwrap()
    }

    #[test]
    fn semantic_search_ranks_the_closer_match_first() {
        let tmp = tempfile::tempdir().unwrap();
        write_named(
            tmp.path(),
            "close",
            "user prefers short, terse commit messages",
        )
        .unwrap();
        write_named(
            tmp.path(),
            "far",
            "podman containers require explicit network setup",
        )
        .unwrap();

        let mut e = embedder();
        let matches =
            semantic_search(tmp.path(), "keep commit messages brief", &mut e, 0.0, 10).unwrap();
        assert_eq!(matches[0].slug, "close");
        assert!(matches[0].score > matches[1].score);
    }

    #[test]
    fn semantic_search_respects_min_score_floor() {
        let tmp = tempfile::tempdir().unwrap();
        write_named(
            tmp.path(),
            "fact",
            "podman containers require explicit network setup",
        )
        .unwrap();

        let mut e = embedder();
        let matches =
            semantic_search(tmp.path(), "keep commit messages brief", &mut e, 0.99, 10).unwrap();
        assert!(
            matches.is_empty(),
            "unrelated content shouldn't clear a near-1.0 floor"
        );
    }

    #[test]
    fn semantic_search_persists_an_embedding_cache_file_after_first_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_named(tmp.path(), "fact", "podman containers, not docker").unwrap();

        let mut e = embedder();
        semantic_search(tmp.path(), "containers", &mut e, 0.0, 10).unwrap();

        assert!(
            crate::embed_cache::load(tmp.path(), "fact").is_some(),
            "a cache file should exist for the entry after a real recall call"
        );
    }

    #[test]
    fn semantic_search_does_not_recompute_an_unchanged_entrys_cache_on_a_second_call() {
        let tmp = tempfile::tempdir().unwrap();
        write_named(tmp.path(), "fact", "podman containers, not docker").unwrap();

        let mut e = embedder();
        semantic_search(tmp.path(), "containers", &mut e, 0.0, 10).unwrap();
        let cache_path = tmp.path().join(".embeddings/fact.cache");
        let mtime_after_first = fs::metadata(&cache_path).unwrap().modified().unwrap();

        // A second call against unchanged content must not rewrite the cache file -- proof the entry wasn't re-embedded.
        std::thread::sleep(std::time::Duration::from_millis(10));
        semantic_search(tmp.path(), "different query entirely", &mut e, 0.0, 10).unwrap();
        let mtime_after_second = fs::metadata(&cache_path).unwrap().modified().unwrap();

        assert_eq!(
            mtime_after_first, mtime_after_second,
            "cache file should not be rewritten when the entry's content hasn't changed"
        );
    }

    #[test]
    fn semantic_search_invalidates_the_cache_when_content_changes() {
        let tmp = tempfile::tempdir().unwrap();
        write_named(tmp.path(), "fact", "the weather today is sunny and warm").unwrap();

        let mut e = embedder();
        semantic_search(tmp.path(), "containers", &mut e, 0.0, 10).unwrap();
        let cache = crate::embed_cache::load(tmp.path(), "fact").unwrap();
        let old_hash = cache.content_hash;

        // Overwrite the same slug with unrelated new content.
        write_named(
            tmp.path(),
            "fact",
            "podman containers require explicit networking",
        )
        .unwrap();
        let matches = semantic_search(tmp.path(), "podman networking", &mut e, 0.3, 10).unwrap();

        let refreshed = crate::embed_cache::load(tmp.path(), "fact").unwrap();
        assert_ne!(
            refreshed.content_hash, old_hash,
            "cache should reflect the new content's hash, not the stale one"
        );
        assert!(
            matches.iter().any(|m| m.slug == "fact"),
            "recall should find the updated content, not stale cached vectors"
        );
    }

    #[test]
    fn semantic_search_respects_limit() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 0..5 {
            write_named(
                tmp.path(),
                &format!("fact-{i}"),
                "user prefers terse commit messages",
            )
            .unwrap();
        }
        let mut e = embedder();
        let matches = semantic_search(tmp.path(), "commit messages", &mut e, 0.0, 3).unwrap();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn semantic_search_on_empty_bank_returns_empty_without_loading_a_model() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path()).unwrap();
        let mut e = embedder();
        assert!(
            semantic_search(tmp.path(), "anything", &mut e, 0.0, 10)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn snippet_of_truncates_long_content_with_ellipsis() {
        let long = "word ".repeat(60);
        let snippet = snippet_of(&long);
        assert!(snippet.ends_with('…'));
        assert!(snippet.chars().count() <= 121);
    }

    /// A phrase past the old whole-file truncation point (>300 words of filler) must still be found; before windowing this returned empty.
    /// min_score is 0.25, not the library-wide 0.3 default -- this fixture is deliberately worst-case and real scores cluster at 0.28-0.30.
    #[test]
    fn semantic_search_finds_a_match_buried_past_the_old_truncation_point() {
        let tmp = tempfile::tempdir().unwrap();
        let filler =
            "unrelated filler content about gardening tips and weather patterns ".repeat(60);
        let content = format!(
            "{filler}The quarterly onboarding checklist requires manager signoff before day three."
        );
        write_named(tmp.path(), "long-doc", &content).unwrap();

        let mut e = embedder();
        let matches = semantic_search(
            tmp.path(),
            "manager signoff onboarding checklist requirement",
            &mut e,
            0.25,
            10,
        )
        .unwrap();

        assert_eq!(
            matches.len(),
            1,
            "the buried phrase should be found once windowing embeds the part of the file that actually contains it"
        );
        assert_eq!(matches[0].slug, "long-doc");
        assert!(
            matches[0].window_offset > 0,
            "the matching window should be the later one, not the file's start"
        );
    }
}
