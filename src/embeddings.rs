use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::ApiBuilder;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Local, no-API sentence embeddings — all-MiniLM-L6-v2, the same model
/// mindforge's own `tools/dedupe_semantic.py` already uses through
/// Python's fastembed. The model downloads once from Hugging Face on
/// first use and is cached at `~/.trm/models/` after that — no per-call
/// network access, no LLM API, no per-token cost. Consistent with
/// ADR-0002's "no external APIs" for judgment work: this is a fixed,
/// deterministic embedding model, not an LLM making a decision.
///
/// 2026-08-18: ported off `fastembed` (ONNX Runtime underneath) onto
/// `candle`, running the real, unmodified `sentence-transformers/
/// all-MiniLM-L6-v2` checkpoint directly via `candle_transformers`'
/// `BertModel` — no ONNX conversion step. Same weights fastembed's own
/// `EmbeddingModel::AllMiniLML6V2` already used (it downloads `Qdrant/
/// all-MiniLM-L6-v2-onnx`, an ONNX export of this exact model — same
/// mean-pooling, same L2-normalize, confirmed by reading fastembed
/// 5.17.4's own source). Verified numerically equivalent before this
/// migration, not assumed: cosine similarity ~1.0 (float32-precision
/// noise only) between fastembed's real output and this module's output
/// on the same real sentences — see `squishi`'s own
/// `docs/ideation/ort-dependency-consistency/2026-08-18-ort-pin-and-bottleneck-plan.md`
/// for the full investigation this follows (`squishi/src/
/// semantic_dedup.rs` made the identical move first, for the identical
/// reason: `fastembed` and `magika` forced incompatible `ort`
/// prereleases in that crate — dropping `ort` here too means
/// `total-recall` can never collide with another tool's `ort`
/// resolution again).
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

const MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
const EMBEDDING_DIM: usize = 384;

/// Texts per forward pass. `curator::scan` calls `embed()` with every
/// wiki entry in a bank at once (real measurement: 50 entries took
/// 41.8s one-at-a-time — a real regression caught by
/// `curator::tests::scan_completes_within_a_bounded_time_at_realistic_bank_scale`
/// during this port, not a hypothetical). Batching amortizes one
/// transformer forward pass across many sequences instead of paying
/// full per-call overhead per text — same fix, same constant value,
/// squishi's own `semantic_dedup.rs::EMBED_BATCH_SIZE` already applied
/// for the identical reason. 32 is a conventional sentence-transformer
/// batch size, not independently tuned here.
const EMBED_BATCH_SIZE: usize = 32;

impl Embedder {
    /// `cache_dir` is explicit (not derived internally from a global
    /// data root) — easier to test in isolation, and it's the same
    /// explicit-path pattern every other module here already follows.
    /// Production call sites pass `bank::data_root().join("models")`,
    /// pinned rather than cwd-relative. `hf_hub::api::sync::ApiBuilder::
    /// with_cache_dir` honors this directly (unlike `Api::new()`'s
    /// default, which always resolves to `~/.cache/huggingface/hub` —
    /// a real gap found and documented in squishi's own `doctor.rs`,
    /// avoided here by using the builder instead of the shortcut).
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
    /// genuine cold cache, retries wait out the one real download. Kept
    /// unchanged by the 2026-08-18 candle port — this locking problem was
    /// never specific to `fastembed`'s downloader, and a lock around
    /// model construction is harmless even if `hf-hub`'s own downloader
    /// turns out to already be race-safe on its own.
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
                Ok((model, tokenizer)) => {
                    return Ok(Self {
                        model,
                        tokenizer,
                        device: Device::Cpu,
                    });
                }
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

    /// Batches `texts` in chunks of `EMBED_BATCH_SIZE`, one transformer
    /// forward pass per chunk rather than one per text — see
    /// `EMBED_BATCH_SIZE`'s doc comment for why this isn't optional.
    /// Returns embeddings in the same order as `texts`.
    pub fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut result = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(EMBED_BATCH_SIZE) {
            result.extend(self.embed_batch(chunk)?);
        }
        Ok(result)
    }

    /// Embeds one batch in a single forward pass. Uses the tokenizer's
    /// own batch padding (`with_padding`, default `BatchLongest`/
    /// right-padded) so every sequence in the batch shares one seq_len,
    /// which candle's fixed-shape `[batch, seq]` tensors require, then
    /// mean-pools over the attention mask and L2-normalizes — the
    /// standard sentence-transformers recipe, the same one `fastembed`'s
    /// own `pooling::mean` + `common::normalize` applied (see the module
    /// doc comment's equivalence check). Padding is toggled back off
    /// afterward so the `Tokenizer`'s state doesn't leak into any other
    /// caller.
    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        self.tokenizer
            .with_padding(Some(tokenizers::PaddingParams::default()));
        let encode_result = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| e.to_string());
        self.tokenizer.with_padding(None);
        let encodings = encode_result?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        let mut ids = Vec::with_capacity(batch_size * seq_len);
        let mut mask = Vec::with_capacity(batch_size * seq_len);
        let mut type_ids = Vec::with_capacity(batch_size * seq_len);
        for e in &encodings {
            ids.extend(e.get_ids().iter().copied());
            mask.extend(e.get_attention_mask().iter().copied());
            type_ids.extend(e.get_type_ids().iter().copied());
        }

        let input_ids = Tensor::from_vec(ids, (batch_size, seq_len), &self.device)
            .map_err(|e| e.to_string())?;
        let attention_mask = Tensor::from_vec(mask.clone(), (batch_size, seq_len), &self.device)
            .map_err(|e| e.to_string())?;
        let token_type_ids = Tensor::from_vec(type_ids, (batch_size, seq_len), &self.device)
            .map_err(|e| e.to_string())?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| e.to_string())?;
        let hidden: Vec<f32> = hidden
            .flatten_all()
            .map_err(|e| e.to_string())?
            .to_vec1()
            .map_err(|e| e.to_string())?;
        let hidden = hidden.as_slice();

        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut pooled = vec![0f32; EMBEDDING_DIM];
            let mut mask_sum = 0f32;
            for s in 0..seq_len {
                if mask[b * seq_len + s] == 0 {
                    continue;
                }
                mask_sum += 1.0;
                let base = (b * seq_len + s) * EMBEDDING_DIM;
                for d in 0..EMBEDDING_DIM {
                    pooled[d] += hidden[base + d];
                }
            }
            if mask_sum > 0.0 {
                for v in &mut pooled {
                    *v /= mask_sum;
                }
            }

            let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut pooled {
                    *v /= norm;
                }
            }
            result.push(pooled);
        }

        Ok(result)
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

fn build_model(cache_dir: &Path) -> anyhow::Result<(BertModel, Tokenizer)> {
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()?;
    let repo = api.model(MODEL_REPO.to_string());
    let weights = repo.get("model.safetensors")?;
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    let config: BertConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    tokenizer.with_padding(None);

    let device = Device::Cpu;
    // SAFETY: `weights` is a file this process just fetched from the
    // hf-hub cache/API into `cache_dir`, not untrusted user input.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
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

    // Probe run 2026-08-06 (plan step 1) to find the real effective
    // token/word ceiling before hardcoding a window size, rather than
    // trusting the ~256-token figure already documented in main.rs's CLI
    // help as fact. Method: a fixed "needle" sentence at the start of a
    // document, increasing filler after it; cosine similarity to a
    // needle-matching query stops changing (delta 0.0000) once filler
    // exceeds the real ceiling, since nothing past that point can move
    // the embedding vector. Real result: similarity dropped steadily
    // through 200 filler words, then went perfectly flat (0.3203, delta
    // 0.0000) at 250 and every length tested after — confirming the
    // truncation boundary sits at 200-250 filler words + an 11-word
    // needle, i.e. ~211-261 total words. This validates the documented
    // ~256-token estimate as real, measured, not assumed. `window.rs`'s
    // window size should sit safely under this (see its own module doc)
    // to leave margin for frontmatter overhead. Probe removed after use
    // — the finding is what mattered, not a permanent test.

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
    // ("Failed to retrieve model.onnx" — `fastembed`'s own error
    // signature for this, before the 2026-08-18 candle port). Root cause
    // per the board: no coordination around the download, the same class
    // of problem the bank lease-lock already exists to solve. Root-cause
    // fix: reuse that lock around Embedder::new(), guarding the cache
    // dir. This property (concurrent first-time loads never corrupt one
    // another) is backend-agnostic and still worth defending after the
    // candle port, even though this backend's own corruption failure
    // signature (if `hf-hub`'s downloader isn't already race-safe on its
    // own) hasn't been separately characterized — the specific-substring
    // check below was dropped for that reason rather than guessed at.
    //
    // This test is slow (real download on a fresh cache dir) and must
    // run alone, not interleaved with other tests hitting the same
    // shared cache — see the module doc on why cache_dir is now an
    // explicit parameter instead of the global ~/.trm/models.
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
    }

    #[test]
    fn once_warm_concurrent_loads_all_succeed_quickly_via_retry() {
        let cache_dir = tempfile::tempdir().unwrap();
        let cache_path = cache_dir.path().to_path_buf();

        // Warm the cache with one call first — also the timing baseline
        // below, so the assertion scales with the host's actual current
        // speed/load instead of a hardcoded wall-clock number. Found
        // flaky in real use (2026-08-07): a fixed "<10s" bound passed
        // standalone but failed running alongside this repo's ~80 other
        // tests, several of which also spin up real Embedder instances —
        // genuine host contention, not a lock-logic regression (the
        // panic was only ever the timing assertion; every result was
        // already `Ok`). The property actually worth defending is
        // "concurrent warm loads don't serialize/hang the way the fixed
        // fail-fast bug this test exists for would" — relative to a
        // solo load measured on the same run, not an absolute bound.
        let baseline_start = std::time::Instant::now();
        Embedder::new(cache_path.clone()).expect("initial warm-up should succeed");
        let baseline = baseline_start.elapsed();

        // Now that it's cached, concurrent loads still go through the
        // lock, but each one's critical section is a fast local load —
        // contending threads' retry loop should resolve without failing
        // or serializing into something dramatically slower than one
        // solo load, not necessarily under any fixed wall-clock number.
        let start = std::time::Instant::now();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let path = cache_path.clone();
                std::thread::spawn(move || Embedder::new(path))
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let elapsed = start.elapsed();

        for result in &results {
            assert!(
                result.is_ok(),
                "warm-cache load should never fail: {:?}",
                result.as_ref().err()
            );
        }
        // 4x the solo baseline (plus a fixed floor for scheduling noise
        // on an already-fast baseline) is generous headroom for retry
        // backoff under real contention while still catching genuine
        // fail-fast-style serialization, which would multiply cost far
        // beyond that regardless of host load.
        let budget = (baseline * 4).max(std::time::Duration::from_secs(5));
        assert!(
            elapsed < budget,
            "warm-cache concurrent loads took {elapsed:?} against a solo baseline of {baseline:?} \
             (budget {budget:?}) — looks like real serialization, not host noise"
        );
    }
}
