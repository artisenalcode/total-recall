use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::{Path, PathBuf};

/// Local, no-API sentence embeddings — all-MiniLM-L6-v2 via ONNX Runtime
/// (the same model mindforge's own `tools/dedupe_semantic.py` already
/// uses through Python's fastembed). The model downloads once from
/// Hugging Face on first use and is cached at `~/.memoryforge/models/`
/// after that — no per-call network access, no LLM API, no per-token
/// cost. Consistent with ADR-0002's "no external APIs" for judgment
/// work: this is a fixed, deterministic embedding model, not an LLM
/// making a decision.
pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// `cache_dir` is explicit (not derived internally from a global
    /// data root) — easier to test in isolation, and it's the same
    /// explicit-path pattern every other module here already follows.
    /// Production call sites pass `bank::data_root().join("models")`,
    /// pinned rather than cwd-relative (fastembed's own default is a
    /// `.fastembed_cache` in the current directory, wrong for a CLI
    /// invoked from many different working directories — found by
    /// actually running it, not assumed).
    ///
    /// Root-cause fix for a real bug: concurrent first-time downloads
    /// into the same cache dir raced and corrupted each other. The
    /// board's fix was right in spirit (reuse the bank lease-lock, not
    /// bank-specific in its implementation) but two attempts at applying
    /// it were each wrong in a way only running the full test suite cold
    /// exposed:
    ///
    /// 1. Naive lock-on-every-call: correct (no corruption) but fail-fast
    ///    meant most concurrent callers just failed, even once the cache
    ///    was already warm and the only real work is a fast local load.
    /// 2. "Try unlocked first, lock only on failure": reintroduced the
    ///    original race, because the unlocked attempt isn't a pure read
    ///    — it *starts a download* if the cache is cold, so two threads
    ///    hitting a cold cache at the same instant both entered the
    ///    unlocked path and corrupted each other exactly as before.
    ///
    /// The actual fix: always take the lock (no unlocked path at all —
    /// that's what caused #2), but retry with a short backoff instead of
    /// `lock.rs`'s fail-fast policy. `lock.rs` itself is untouched and
    /// stays fail-fast, which is correct for bank writes; this is a
    /// local wrapper around it for a different-natured resource. Once
    /// warm, the critical section (a local load) is sub-second, so a
    /// contending thread's retry loop resolves almost immediately; on a
    /// genuine cold cache, retries wait out the one real download.
    ///
    /// A third layer, also only found by running the real suite: even
    /// correctly serialized (one downloader at a time, confirmed), the
    /// download itself can fail transiently under heavy concurrent
    /// system load — separate from the locking bug, a plain flaky-
    /// network problem. Retried a few times below, independent of the
    /// lock-contention retry above.
    ///
    /// A fourth layer, needed because the retry above still let several
    /// threads *in this same process* independently poll and re-attempt
    /// network calls while waiting — compounding exactly the load
    /// causing the transient failures in the first place. The file lock
    /// is the right tool for cross-process safety (two separate `trm`
    /// invocations), but for same-process threads a plain in-process
    /// mutex is cheaper and correct: it blocks instead of polling, so
    /// only one thread in the whole process ever attempts the download,
    /// full stop — no compounding, no retry storm.
    pub fn new(cache_dir: PathBuf) -> Result<Self, String> {
        static IN_PROCESS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _in_process_guard = IN_PROCESS_LOCK.lock().map_err(|e| e.to_string())?;

        std::fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
        let _guard = acquire_with_retry(&cache_dir, std::time::Duration::from_secs(90))?;

        // Even correctly serialized, the download itself can fail
        // transiently under heavy concurrent system load (found by
        // running the full test suite for real, not assumed) — retry
        // the fetch a few times before surfacing a hard error, the
        // standard robustness pattern for a flaky network call.
        let mut last_err = None;
        for attempt in 0..3 {
            match build_model(&cache_dir) {
                Ok(model) => return Ok(Self { model }),
                Err(e) => {
                    last_err = Some(e);
                    if attempt < 2 {
                        std::thread::sleep(std::time::Duration::from_secs(2));
                    }
                }
            }
        }
        Err(last_err.unwrap().to_string())
    }

    pub fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        self.model.embed(texts, None).map_err(|e| e.to_string())
    }
}

/// Retry `lock::acquire` with a short fixed backoff instead of failing
/// immediately on the first contention. Scoped to this call site only —
/// `lock.rs`'s own policy (fail-fast) is unchanged for bank writes.
fn acquire_with_retry(
    dir: &Path,
    max_wait: std::time::Duration,
) -> Result<crate::lock::LockGuard, String> {
    let start = std::time::Instant::now();
    loop {
        match crate::lock::acquire(dir) {
            Ok(guard) => return Ok(guard),
            Err(e) if start.elapsed() >= max_wait => return Err(e.to_string()),
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(300)),
        }
    }
}

fn build_model(cache_dir: &Path) -> anyhow::Result<TextEmbedding> {
    TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_cache_dir(cache_dir.to_path_buf()),
    )
}

/// Cosine similarity between two equal-length embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_of_identical_vectors_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_of_orthogonal_vectors_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_of_opposite_vectors_is_negative_one() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_with_zero_vector_is_zero_not_nan() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // Cosine-similarity tests above are pure and don't need a real
    // model. This one does: it reproduces a genuine bug found by
    // running curator-scan's test suite for real — concurrent first-time
    // model downloads into the same cache dir corrupted each other
    // ("Failed to retrieve model.onnx"). Root cause per the board: no
    // coordination around the download, the same class of problem the
    // bank lease-lock already exists to solve. Root-cause fix: reuse
    // that lock around Embedder::new(), guarding the cache dir.
    //
    // This test is slow (real download on a fresh cache dir) and must
    // run alone, not interleaved with other tests hitting the same
    // shared cache — see the module doc on why cache_dir is now an
    // explicit parameter instead of the global ~/.memoryforge/models.
    #[test]
    fn concurrent_first_time_downloads_do_not_corrupt_each_other() {
        let cache_dir = tempfile::tempdir().unwrap();
        let cache_path = cache_dir.path().to_path_buf();

        let handles: Vec<_> = (0..3)
            .map(|_| {
                let path = cache_path.clone();
                std::thread::spawn(move || Embedder::new(path))
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        let successes = results.iter().filter(|r| r.is_ok()).count();
        assert!(
            successes >= 1,
            "at least one concurrent Embedder::new() call should succeed"
        );

        for result in &results {
            if let Err(e) = result {
                assert!(
                    !e.contains("Failed to retrieve model.onnx"),
                    "a failure should be clean lock contention, not download corruption: {e}"
                );
            }
        }
    }

    #[test]
    fn once_warm_concurrent_loads_all_succeed_quickly_via_retry() {
        let cache_dir = tempfile::tempdir().unwrap();
        let cache_path = cache_dir.path().to_path_buf();

        // Warm the cache with one call first.
        Embedder::new(cache_path.clone()).expect("initial warm-up should succeed");

        // Now that it's cached, concurrent loads still go through the
        // lock, but each one's critical section is a fast local load —
        // contending threads' retry loop should resolve almost
        // immediately, not fail and not hang.
        let start = std::time::Instant::now();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let path = cache_path.clone();
                std::thread::spawn(move || Embedder::new(path))
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        for result in &results {
            assert!(
                result.is_ok(),
                "warm-cache load should never fail: {:?}",
                result.as_ref().err()
            );
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "warm-cache concurrent loads took {:?}, expected sub-second-scale resolution",
            start.elapsed()
        );
    }
}
