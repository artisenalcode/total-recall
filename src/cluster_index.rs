//! Cross-file recurrence clustering over a persona's raw corpus, backed
//! by a throwaway per-corpus SQLite file instead of squishi's pooled,
//! windowed dedup pass.
//!
//! Real bug this replaces: squishi's `MAX_COMPARISON_WINDOW` bound
//! (added to fix a genuine O(n^2) blowup on a single document) assumes
//! near-duplicate sentences land near each other in the text. Pooling
//! many whole transcripts end to end breaks that assumption -- the same
//! idea in source file 1 and source file 15 can sit thousands of
//! sentence-positions apart, permanently outside any fixed window, so
//! they were never even compared. Pressure-tested against Roy
//! Sugarman's real 19-video corpus (2026-08-10): 5,707 of 5,714 pooled
//! "topic" clusters came back size 1 -- essentially no recurrence
//! detected at all, despite known recurring concepts (DARN-C, the
//! Purpose Stack) that should have clustered far higher.
//!
//! The fix: an index has no notion of position. `up` embeds every kept
//! sentence from every source file's `.dedup.json` sidecar (same model
//! `trm recall` already uses) into `<raw>/<slug>/cluster.sqlite`; a
//! query then compares every sentence against every other sentence in
//! the corpus, unbounded. At real per-person corpus sizes (~5-10k
//! sentences), brute-force cosine comparison in native code is
//! genuinely fast -- no approximate index (HNSW) is needed at this
//! scale, only removal of the artificial window. SQLite's WAL mode
//! gives "multiple scripts against one live index" for free (concurrent
//! readers across separate `trm` invocations), without a database
//! server process. `down` deletes the file -- the index is exactly as
//! disposable as the raw tier it's built from (ADR-0005's "raw is
//! disposable post-synthesis" convention).

use crate::embeddings::{Embedder, cosine_similarity};
use rusqlite::Connection;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Matches squishi's own `PARAPHRASE_THRESHOLD` (`main.rs`) -- the same
/// paraphrase-collapse decision, just made without a positional window.
pub const PARAPHRASE_THRESHOLD: f32 = 0.80;

/// One ranked cluster: a sentence that survived clustering, plus how
/// many other sentences (across every source file) collapsed into it.
/// High `cluster_size` on a `concept`-shaped sentence is a recurring
/// topic; on a `narrative`-shaped one, a recurring story.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrenceCluster {
    pub text: String,
    pub cluster_size: usize,
}

impl From<&RecurrenceCluster> for Value {
    fn from(c: &RecurrenceCluster) -> Value {
        serde_json::json!({ "text": c.text, "cluster_size": c.cluster_size })
    }
}

fn index_path(raw_dir: &Path, slug: &str) -> PathBuf {
    raw_dir.join(slug).join("cluster.sqlite")
}

fn embedding_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_embedding(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Reads every `.dedup.json` sidecar's `kept` array for `slug` and
/// returns `(source_file, text, shape)` triples, sorted by sidecar
/// filename for determinism. Requires squishi's `kept` field (per-file
/// `dedup-raw` must have already run against a squishi build that emits
/// it) -- errors clearly if no sidecars exist at all.
fn read_kept_sentences(
    raw_dir: &Path,
    slug: &str,
) -> Result<Vec<(String, String, String)>, String> {
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

    let mut rows = Vec::new();
    for path in &sidecar_paths {
        let source_file = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let parsed: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let Some(kept) = parsed.get("kept").and_then(|v| v.as_array()) else {
            continue;
        };
        for entry in kept {
            let (Some(text), Some(shape)) = (
                entry.get("text").and_then(|v| v.as_str()),
                entry.get("shape").and_then(|v| v.as_str()),
            ) else {
                continue;
            };
            rows.push((source_file.clone(), text.to_string(), shape.to_string()));
        }
    }
    Ok(rows)
}

/// Builds (or rebuilds, if it already exists) `<raw>/<slug>/
/// cluster.sqlite`: one row per kept sentence across every source
/// file's `.dedup.json` sidecar, with its embedding stored as a BLOB.
/// Always a full rebuild, not incremental -- this is a throwaway index
/// meant to live for one synthesis pass, not a durable store to keep in
/// sync piecemeal. Returns the index path.
pub fn up(raw_dir: &Path, slug: &str) -> Result<PathBuf, String> {
    let rows = read_kept_sentences(raw_dir, slug)?;
    if rows.is_empty() {
        return Err(format!(
            "every sidecar for slug {slug:?} had an empty or missing `kept` array -- nothing to index"
        ));
    }

    let mut embedder =
        Embedder::new(crate::bank::data_root().join("models")).map_err(|e| e.to_string())?;
    let texts: Vec<String> = rows.iter().map(|(_, text, _)| text.clone()).collect();
    let embeddings = embedder.embed(&texts)?;

    let path = index_path(raw_dir, slug);
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| e.to_string())?;
    }
    let conn = Connection::open(&path).map_err(|e| e.to_string())?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         CREATE TABLE sentences (
             id INTEGER PRIMARY KEY,
             source_file TEXT NOT NULL,
             text TEXT NOT NULL,
             shape TEXT NOT NULL,
             embedding BLOB NOT NULL
         );",
    )
    .map_err(|e| e.to_string())?;

    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    {
        let mut stmt = tx
            .prepare("INSERT INTO sentences (source_file, text, shape, embedding) VALUES (?1, ?2, ?3, ?4)")
            .map_err(|e| e.to_string())?;
        for ((source_file, text, shape), embedding) in rows.iter().zip(embeddings.iter()) {
            stmt.execute(rusqlite::params![
                source_file,
                text,
                shape,
                embedding_to_blob(embedding)
            ])
            .map_err(|e| e.to_string())?;
        }
    }
    tx.commit().map_err(|e| e.to_string())?;

    Ok(path)
}

/// Deletes the corpus's index file (and its WAL/SHM siblings, if
/// present) -- the tear-down half of `up`. Not an error if nothing
/// exists to delete (idempotent, matching `dedup_raw_files`'s posture
/// on already-done work).
pub fn down(raw_dir: &Path, slug: &str) -> Result<(), String> {
    let path = index_path(raw_dir, slug);
    for candidate in [
        path.clone(),
        PathBuf::from(format!("{}-wal", path.display())),
        PathBuf::from(format!("{}-shm", path.display())),
    ] {
        if candidate.exists() {
            std::fs::remove_file(&candidate).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Queries the already-built index for `slug` and returns
/// `(topics, stories)`, each ranked by `cluster_size` descending.
/// Unbounded greedy clustering (same first-fit-above-threshold
/// algorithm squishi's own `dedupe()` uses), but over the *entire*
/// corpus at once -- no positional window, which is the whole fix.
pub fn cluster(
    raw_dir: &Path,
    slug: &str,
) -> Result<(Vec<RecurrenceCluster>, Vec<RecurrenceCluster>), String> {
    let path = index_path(raw_dir, slug);
    if !path.exists() {
        return Err(format!(
            "no cluster index for slug {slug:?} -- run `trm cluster-index up --slug {slug}` first"
        ));
    }

    let conn = Connection::open(&path).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT source_file, text, shape, embedding FROM sentences ORDER BY id")
        .map_err(|e| e.to_string())?;
    let rows: Vec<(String, String, String, Vec<f32>)> = stmt
        .query_map([], |row| {
            let embedding_blob: Vec<u8> = row.get(3)?;
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                blob_to_embedding(&embedding_blob),
            ))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;

    // Greedy single-pass clustering, unbounded: for each sentence, find
    // the best-matching already-kept sentence across the WHOLE corpus
    // so far. Above threshold -> collapses into it (cluster grows).
    // Otherwise -> becomes a new cluster head. O(n^2) worst case, same
    // as squishi's pre-windowing algorithm -- acceptable here because a
    // real per-person corpus (~5-10k sentences) is small enough that
    // brute force is still fast (Stonebraker's own point: no need for
    // an approximate index at this scale).
    let mut kept: Vec<(usize, String, String)> = Vec::new(); // (row_index, text, shape)
    let mut kept_embeddings: Vec<&Vec<f32>> = Vec::new();
    let mut cluster_sizes: Vec<usize> = Vec::new();

    for (_source_file, text, shape, embedding) in &rows {
        let mut best: Option<(usize, f32)> = None;
        for (i, kept_emb) in kept_embeddings.iter().enumerate() {
            let sim = cosine_similarity(embedding, kept_emb);
            if best.is_none_or(|(_, best_sim)| sim > best_sim) {
                best = Some((i, sim));
            }
        }

        match best {
            Some((i, sim)) if sim >= PARAPHRASE_THRESHOLD => {
                cluster_sizes[i] += 1;
            }
            _ => {
                kept.push((kept.len(), text.clone(), shape.clone()));
                kept_embeddings.push(embedding);
                cluster_sizes.push(1);
            }
        }
    }

    let mut topics = Vec::new();
    let mut stories = Vec::new();
    for ((_, text, shape), size) in kept.into_iter().zip(cluster_sizes) {
        let cluster = RecurrenceCluster {
            text,
            cluster_size: size,
        };
        match shape.as_str() {
            "narrative" => stories.push(cluster),
            _ => topics.push(cluster),
        }
    }
    topics.sort_by_key(|c| std::cmp::Reverse(c.cluster_size));
    stories.sort_by_key(|c| std::cmp::Reverse(c.cluster_size));

    Ok((topics, stories))
}

/// Runs `cluster` and writes `<raw>/<slug>/cluster-summary.json` --
/// same output shape the old squishi-pooled `cluster_raw_files`
/// produced, now backed by the unbounded index query.
pub fn write_summary(raw_dir: &Path, slug: &str) -> Result<PathBuf, String> {
    let (topics, stories) = cluster(raw_dir, slug)?;
    let topics_json: Vec<Value> = topics.iter().map(Value::from).collect();
    let stories_json: Vec<Value> = stories.iter().map(Value::from).collect();
    let summary =
        serde_json::json!({ "slug": slug, "topics": topics_json, "stories": stories_json });

    let summary_path = raw_dir.join(slug).join("cluster-summary.json");
    crate::atomic::write(
        &summary_path,
        &serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    Ok(summary_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_sidecar(person_dir: &Path, name: &str, kept: &[(&str, &str)]) {
        let kept_json: Vec<Value> = kept
            .iter()
            .enumerate()
            .map(|(i, (text, shape))| {
                serde_json::json!({ "index": i, "text": text, "shape": shape })
            })
            .collect();
        let sidecar = serde_json::json!({ "kept": kept_json }).to_string();
        std::fs::write(person_dir.join(name), sidecar).unwrap();
    }

    #[test]
    fn up_errors_for_a_missing_slug_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let result = up(tmp.path(), "no-such-slug");
        assert!(result.is_err());
    }

    #[test]
    fn up_errors_when_no_dedup_sidecars_exist() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("no-sidecars-yet");
        std::fs::create_dir_all(&person_dir).unwrap();
        std::fs::write(person_dir.join("yt-abc.md"), "raw, undeduped body").unwrap();

        let result = up(tmp.path(), "no-sidecars-yet");
        let err = result.expect_err("should error without any .dedup.json sidecars");
        assert!(err.contains("dedup-raw"));
    }

    #[test]
    fn cluster_errors_when_index_was_never_built() {
        let tmp = tempfile::tempdir().unwrap();
        let result = cluster(tmp.path(), "never-built");
        let err = result.expect_err("should error without a prior `up`");
        assert!(err.contains("cluster-index up"));
    }

    #[test]
    fn down_is_idempotent_when_nothing_exists_to_delete() {
        let tmp = tempfile::tempdir().unwrap();
        let result = down(tmp.path(), "never-built");
        assert!(result.is_ok());
    }

    #[test]
    fn embedding_blob_round_trips_exactly() {
        let original = vec![0.1_f32, -2.5, 3.75, 0.0, 100.25];
        let blob = embedding_to_blob(&original);
        let restored = blob_to_embedding(&blob);
        assert_eq!(original, restored);
    }

    // Real end-to-end tests (real ONNX embedder, first-run model
    // download) live behind #[ignore] -- see
    // `cluster_index_ranks_a_sentence_repeated_across_files_above_a_one_off`
    // below, matching the pattern the rest of this crate already uses
    // for embedder-backed tests.

    #[test]
    #[ignore] // requires the real embedding model (network/cache on first run)
    fn cluster_index_ranks_a_sentence_repeated_across_files_above_a_one_off() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("recurring-topic");
        std::fs::create_dir_all(&person_dir).unwrap();

        let recurring = "Values are the load-bearing unit of motivation, and context tells the genes what they need to do.";
        // Genuinely distinct one-off asides, not a shared template
        // differing only by a number -- an earlier version of this test
        // used a template and the asides wrongly clustered with each
        // other too, since near-identical phrasing is exactly what this
        // mechanism is supposed to catch.
        let one_offs = [
            "My daughter had her wisdom teeth removed last Tuesday, completely unrelated to anything else here.",
            "The printer on the third floor jammed again this morning for the third time this month.",
            "We're switching the team's coffee supplier because the old one kept running out of beans.",
        ];

        for (i, one_off) in one_offs.iter().enumerate() {
            write_sidecar(
                &person_dir,
                &format!("source-{i}.dedup.json"),
                &[(recurring, "concept"), (one_off, "concept")],
            );
        }

        let index_result = up(tmp.path(), "recurring-topic");
        assert!(index_result.is_ok(), "up failed: {:?}", index_result.err());

        let (topics, _stories) =
            cluster(tmp.path(), "recurring-topic").expect("cluster should succeed");
        assert!(!topics.is_empty());

        // The recurring sentence (present, near-verbatim, in all three
        // widely-separated source files) should collapse into ONE
        // cluster of size 3 and rank first -- proving cross-file
        // recurrence is detected with no positional bias, unlike the
        // windowed-pooling approach this replaces.
        let top = &topics[0];
        assert_eq!(
            top.cluster_size, 3,
            "expected all 3 near-duplicates to collapse into one cluster: {top:?}"
        );
        assert!(top.text.to_lowercase().contains("motivation"));

        // Every one-off aside should remain its own singleton cluster.
        let singleton_count = topics.iter().filter(|c| c.cluster_size == 1).count();
        assert_eq!(
            singleton_count, 3,
            "each one-off aside should be its own cluster: {topics:?}"
        );
    }
}
