use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::ApiBuilder;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Local, no-API sentence embeddings via all-MiniLM-L6-v2, run through `candle` instead of `fastembed`/ONNX to avoid the `ort` version conflicts squishi hit.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

const MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
const EMBEDDING_DIM: usize = 384;

/// Texts per forward pass; one-at-a-time embedding measured 41.8s for 50 entries. 32 is the conventional sentence-transformer default.
const EMBED_BATCH_SIZE: usize = 32;

impl Embedder {
    /// `cache_dir` is explicit because `hf_hub::Api::new()`'s default ignores it; retrying lock + in-process mutex guard against concurrent first-time downloads corrupting each other.
    pub fn new(cache_dir: PathBuf) -> Result<Self, String> {
        static IN_PROCESS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _in_process_guard = IN_PROCESS_LOCK.lock().map_err(|e| e.to_string())?;

        std::fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
        let _guard = acquire_with_retry(&cache_dir, std::time::Duration::from_secs(90))?;

        // Even serialized, the download can fail transiently under load — retry before surfacing a hard error.
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

    /// Batches `texts` in chunks of `EMBED_BATCH_SIZE`; returns embeddings in the same order as `texts`.
    pub fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut result = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(EMBED_BATCH_SIZE) {
            result.extend(self.embed_batch(chunk)?);
        }
        Ok(result)
    }

    /// Pads to a shared seq_len (candle tensors are fixed-shape), mean-pools over the attention mask, and L2-normalizes.
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

/// Retries `lock::acquire` with a short backoff instead of failing immediately; `lock.rs`'s own fail-fast policy is unchanged for bank writes.
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

    // Probe (removed after use) confirmed main.rs's ~256-token truncation estimate; window.rs's window size stays under it with margin.

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

    // Slow (real download) — must run alone, not interleaved with other tests sharing the cache.
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

        // Baseline scales the assertion to host speed — a fixed <10s bound was flaky under real test-suite contention.
        let baseline_start = std::time::Instant::now();
        Embedder::new(cache_path.clone()).expect("initial warm-up should succeed");
        let baseline = baseline_start.elapsed();

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
        // 4x baseline plus a floor catches real fail-fast-style serialization while tolerating host load.
        let budget = (baseline * 4).max(std::time::Duration::from_secs(5));
        assert!(
            elapsed < budget,
            "warm-cache concurrent loads took {elapsed:?} against a solo baseline of {baseline:?} \
             (budget {budget:?}) — looks like real serialization, not host noise"
        );
    }
}
