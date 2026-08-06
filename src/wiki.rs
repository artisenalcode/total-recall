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

/// Write `content` into `wiki_dir/<slug>.md`, atomically: write to a
/// same-directory temp file, then rename over the final path. Returns the
/// slug used, so the caller can record it in index.md.
pub fn write(wiki_dir: &Path, content: &str) -> io::Result<String> {
    let slug = slugify(content);
    let final_path = wiki_dir.join(format!("{slug}.md"));
    atomic::write(&final_path, content)?;
    Ok(slug)
}

/// Write `content` under an exact, caller-chosen slug — no re-hashing.
/// Used by import pathways that need to preserve an external identity
/// (e.g. mindforge's existing per-advisor filenames), unlike `write`
/// which always derives its own slug from content.
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
    /// Char offset into the source file of the window that produced this
    /// match's score — so a caller can tell *why* a long file matched
    /// (which part), not just that it did. 0 for a file short enough to
    /// be a single window (the common case, unchanged from before
    /// windowing existed).
    pub window_offset: usize,
}

/// Semantic recall: rank every entry in `wiki_dir` by cosine similarity
/// to `query` (local embeddings — same model/mechanism curator-scan
/// already uses, reused rather than rebuilt). Replaces an earlier
/// grep-based `search` entirely — retro finding was "no ranking," and
/// keeping both a grep path and a ranked path around would just be the
/// same dead-code problem the `clean` tier was. Returns matches scoring
/// at or above `min_score`, highest first, capped to `limit`.
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

    // Window every candidate file (a short file — the common case —
    // produces exactly one window, so this is a no-op cost-wise for
    // anything that was already fine). Every window across every file,
    // plus the query, is embedded in one batched call.
    let mut texts = vec![query.to_string()];
    let mut owner_slug_idx = Vec::new();
    let mut owner_offset = Vec::new();
    for (slug_idx, slug) in slugs.iter().enumerate() {
        let content = read(wiki_dir, slug).map_err(|e| e.to_string())?;
        for (offset, window_text) in crate::window::windows(
            &content,
            crate::window::WINDOW_WORDS,
            crate::window::OVERLAP_WORDS,
        ) {
            texts.push(window_text);
            owner_slug_idx.push(slug_idx);
            owner_offset.push(offset);
        }
    }

    let vectors = embedder.embed(&texts)?;
    let query_vec = &vectors[0];

    // Max score per source file — a file matches as well as its
    // best-matching window, and that window's text/offset become the
    // reported snippet/window_offset so a caller can tell *why* it
    // matched, not just that it did.
    let mut best: Vec<Option<(f32, usize, usize)>> = vec![None; slugs.len()];
    for (window_i, (&slug_idx, &offset)) in owner_slug_idx.iter().zip(&owner_offset).enumerate() {
        let score = crate::embeddings::cosine_similarity(query_vec, &vectors[window_i + 1]);
        let current = &mut best[slug_idx];
        if current.is_none_or(|(best_score, _, _)| score > best_score) {
            *current = Some((score, offset, window_i));
        }
    }

    let mut matches: Vec<RankedMatch> = slugs
        .into_iter()
        .zip(best)
        .filter_map(|(slug, hit)| {
            hit.map(|(score, offset, window_i)| RankedMatch {
                slug,
                score,
                snippet: snippet_of(&texts[window_i + 1]),
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

    /// The real regression this whole fix exists for: a distinctive phrase
    /// sitting past the old whole-file-embedding truncation point (>300
    /// words of unrelated filler before it — comfortably past the ~211-261
    /// word real ceiling measured in embeddings.rs's probe) must still be
    /// found. Before windowing, this returns empty — the file's embedding
    /// never saw the phrase at all.
    ///
    /// min_score here is 0.25, not the library-wide 0.3 recall default —
    /// measured (window.rs's module doc), not guessed: this fixture is
    /// deliberately worst-case (a real point diluted by 600 words of a
    /// single unrelated topic, not realistic mixed-topic content), and
    /// real scores against it cluster at 0.28-0.30 regardless of window
    /// size. The fix under test is "0 matches -> a real, findable match,"
    /// not "always clears an arbitrary universal floor on adversarial
    /// content" — real session/doc content mixing related sub-topics
    /// should score better than this intentionally-hard fixture.
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
