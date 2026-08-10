//! Mechanical, model-free concept-recurrence detection -- the answer to
//! a real limitation `cluster_index` can't solve: sentence embeddings
//! catch near-identical *phrasing*, not the same idea explained in
//! different words each time. Roy Sugarman doesn't recite DARN-C
//! verbatim in every interview; he re-explains it from scratch, so its
//! surrounding sentences never cluster. But "DARN-C" itself, as a term,
//! recurs literally -- a plain n-gram frequency count catches that with
//! no model at all.
//!
//! `lexicon_scan` counts 2-4 word phrases across every source file's
//! already-cleaned `.dedup.json` text, ranked by how many *distinct
//! source files* a phrase appears in (breadth across independent
//! occasions -- the same "returns to it unprompted across independent
//! sources" evidence bar `handover.rs`'s `PersonaBuild` criteria
//! already names for values), with total occurrence count as the
//! tiebreak. This is deliberately separate from boilerplate flagging
//! (a different problem: excluding recurring *noise* like podcast
//! intros, not finding recurring *signal*).

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

const MIN_NGRAM: usize = 2;
const MAX_NGRAM: usize = 4;

/// A minimal, deliberately small function-word stoplist -- just enough
/// to keep an n-gram's *edges* meaningful ("the purpose stack" and
/// "purpose stack" should count as the same signal, trimmed to "purpose
/// stack"). Not an attempt at full stopword removal; mid-phrase
/// function words ("of", "and") stay, since "state of flow" or "fight
/// and flight" are real terms, not noise.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "so", "to", "of", "in", "on", "at", "is", "was", "are",
    "were", "be", "been", "i", "you", "he", "she", "it", "we", "they", "this", "that", "with",
    "for", "as", "if", "then", "than", "not", "just", "like", "um", "uh",
];

fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word)
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// One ranked lexicon entry: a phrase, how many distinct source files
/// it appeared in, and how many times total across the corpus.
#[derive(Debug, Clone, PartialEq)]
pub struct LexiconEntry {
    pub term: String,
    pub distinct_files: usize,
    pub total_occurrences: usize,
}

impl From<&LexiconEntry> for Value {
    fn from(e: &LexiconEntry) -> Value {
        serde_json::json!({
            "term": e.term,
            "distinct_files": e.distinct_files,
            "total_occurrences": e.total_occurrences,
        })
    }
}

/// Counts every 2-4 word phrase (with stopword-trimmed edges) in
/// `text`, adding `source_file` to each phrase's file set and
/// incrementing its total count. `counts` maps a lowercase phrase to
/// `(display_form, files, total_occurrences)` -- the display form is
/// whichever casing was seen first, good enough without a full
/// casing-frequency vote.
fn count_ngrams(
    text: &str,
    source_file: &str,
    counts: &mut HashMap<String, (String, HashSet<String>, usize)>,
) {
    let words = tokenize(text);
    for n in MIN_NGRAM..=MAX_NGRAM {
        if words.len() < n {
            continue;
        }
        for window in words.windows(n) {
            let Some(first) = window.first() else {
                continue;
            };
            let Some(last) = window.last() else { continue };
            if is_stopword(first) || is_stopword(last) {
                continue;
            }
            let phrase = window.join(" ");
            let entry = counts
                .entry(phrase.clone())
                .or_insert_with(|| (phrase.clone(), HashSet::new(), 0));
            entry.1.insert(source_file.to_string());
            entry.2 += 1;
        }
    }
}

/// Scans every `.dedup.json` sidecar's `compressed` text for `slug`
/// and returns ranked lexicon entries (distinct-file count descending,
/// then total occurrences descending). Phrases seen in only one
/// occurrence corpus-wide are dropped -- a single mention isn't a
/// "returns to" signal, just noise. No embedding model, no network --
/// purely mechanical text counting.
pub fn scan(raw_dir: &Path, slug: &str) -> Result<Vec<LexiconEntry>, String> {
    let person_dir = raw_dir.join(slug);
    if !person_dir.is_dir() {
        return Err(format!(
            "no raw directory for slug {slug:?}: {}",
            person_dir.display()
        ));
    }

    let mut sidecar_paths: Vec<PathBuf> = std::fs::read_dir(&person_dir)
        .map_err(|e| e.to_string())?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".dedup.json"))
        })
        .collect();
    sidecar_paths.sort();

    if sidecar_paths.is_empty() {
        return Err(format!(
            "no .dedup.json sidecars for slug {slug:?} -- run `trm dedup-raw --slug {slug}` first"
        ));
    }

    let mut counts: HashMap<String, (String, HashSet<String>, usize)> = HashMap::new();
    for path in &sidecar_paths {
        let source_file = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let parsed: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let Some(compressed) = parsed.get("compressed").and_then(|v| v.as_str()) else {
            continue;
        };
        count_ngrams(compressed, &source_file, &mut counts);
    }

    let mut entries: Vec<LexiconEntry> = counts
        .into_values()
        .filter(|(_, _, total)| *total >= 2)
        .map(|(display, files, total)| LexiconEntry {
            term: display,
            distinct_files: files.len(),
            total_occurrences: total,
        })
        .collect();
    entries.sort_by(|a, b| {
        b.distinct_files
            .cmp(&a.distinct_files)
            .then(b.total_occurrences.cmp(&a.total_occurrences))
            .then(a.term.cmp(&b.term))
    });

    Ok(entries)
}

/// Runs `scan` and writes `<raw>/<slug>/lexicon.json` (capped to the
/// top 200 entries -- plenty for a synthesis pass to draw from without
/// carrying every two-word noise phrase that cleared the `total >= 2`
/// floor). Returns the written path.
pub fn write_lexicon(raw_dir: &Path, slug: &str) -> Result<PathBuf, String> {
    let mut entries = scan(raw_dir, slug)?;
    entries.truncate(200);

    let entries_json: Vec<Value> = entries.iter().map(Value::from).collect();
    let summary = serde_json::json!({ "slug": slug, "terms": entries_json });

    let lexicon_path = raw_dir.join(slug).join("lexicon.json");
    crate::atomic::write(
        &lexicon_path,
        &serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    Ok(lexicon_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_sidecar(person_dir: &Path, name: &str, compressed: &str) {
        let sidecar = serde_json::json!({ "compressed": compressed }).to_string();
        std::fs::write(person_dir.join(name), sidecar).unwrap();
    }

    #[test]
    fn scan_errors_for_a_missing_slug_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let result = scan(tmp.path(), "no-such-slug");
        assert!(result.is_err());
    }

    #[test]
    fn scan_errors_when_no_dedup_sidecars_exist() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("no-sidecars-yet");
        std::fs::create_dir_all(&person_dir).unwrap();
        std::fs::write(person_dir.join("yt-abc.md"), "raw, undeduped body").unwrap();

        let result = scan(tmp.path(), "no-sidecars-yet");
        let err = result.expect_err("should error without any .dedup.json sidecars");
        assert!(err.contains("dedup-raw"));
    }

    #[test]
    fn a_named_term_repeated_differently_phrased_across_files_ranks_first() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("roy-like");
        std::fs::create_dir_all(&person_dir).unwrap();

        // The named term "DARN-C" (well, "the purpose stack" here, to
        // keep tokenize()'s alphanumeric-only splitting simple) recurs
        // verbatim across three files, even though everything AROUND
        // it is phrased completely differently each time -- exactly
        // the case sentence-embedding clustering can't catch.
        write_sidecar(
            &person_dir,
            "a.dedup.json",
            "When I work with clients I always come back to the purpose stack as the anchor.",
        );
        write_sidecar(
            &person_dir,
            "b.dedup.json",
            "So the framework I use, completely differently explained here, is the purpose stack.",
        );
        write_sidecar(
            &person_dir,
            "c.dedup.json",
            "In a totally separate context, coaching athletes needs the purpose stack too.",
        );
        // A one-off phrase that only appears once total should never
        // outrank something appearing across three distinct files.
        write_sidecar(
            &person_dir,
            "d.dedup.json",
            "This scheduling logistics aside about printer maintenance never repeats.",
        );

        let entries = scan(tmp.path(), "roy-like").expect("scan should succeed");
        assert!(!entries.is_empty());

        let top = &entries[0];
        assert_eq!(top.term, "purpose stack");
        assert_eq!(top.distinct_files, 3);
        assert_eq!(top.total_occurrences, 3);
    }

    #[test]
    fn a_phrase_seen_only_once_corpus_wide_is_excluded() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("sparse");
        std::fs::create_dir_all(&person_dir).unwrap();
        write_sidecar(
            &person_dir,
            "a.dedup.json",
            "This unique combination of words never appears again anywhere.",
        );

        let entries = scan(tmp.path(), "sparse").expect("scan should succeed");
        assert!(
            entries.is_empty(),
            "single-occurrence phrases should be filtered: {entries:?}"
        );
    }

    #[test]
    fn ngrams_never_start_or_end_on_a_stopword() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("stopword-edges");
        std::fs::create_dir_all(&person_dir).unwrap();
        write_sidecar(
            &person_dir,
            "a.dedup.json",
            "the load bearing unit of motivation and the load bearing unit again",
        );

        let entries = scan(tmp.path(), "stopword-edges").expect("scan should succeed");
        for entry in &entries {
            let words: Vec<&str> = entry.term.split(' ').collect();
            assert!(!is_stopword(words[0]), "leading stopword in {entry:?}");
            assert!(
                !is_stopword(words[words.len() - 1]),
                "trailing stopword in {entry:?}"
            );
        }
    }
}
