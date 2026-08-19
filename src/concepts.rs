//! Splits multi-concept content into atomic-concept sentences, collapsing near-duplicate paraphrases (sentence split + greedy cosine clustering).
//! Turns a bulk/ambiguous `stage()` payload into indexed candidate concepts for a handover's sub-agent to judge individually.

use crate::embeddings::{Embedder, cosine_similarity};
use std::collections::HashSet;

const MIN_WORDS: usize = 8;
const MAX_WORDS: usize = 40;

/// Split `text` into qualifying sentences, collapse exact duplicates, then greedily cluster by cosine similarity (dropped once `>= threshold`).
pub fn split(text: &str, embedder: &mut Embedder, threshold: f32) -> Result<Vec<String>, String> {
    let candidates = sentences(text);
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut seen = HashSet::new();
    let mut uniq = Vec::new();
    for s in candidates {
        let key: String = s
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();
        let key = key.trim().to_string();
        if !key.is_empty() && seen.insert(key) {
            uniq.push(s);
        }
    }

    let vectors = embedder.embed(&uniq)?;
    let mut kept_indices = Vec::new();
    let mut kept_vecs: Vec<&Vec<f32>> = Vec::new();
    for (i, v) in vectors.iter().enumerate() {
        let is_near_duplicate = kept_vecs
            .iter()
            .any(|kv| cosine_similarity(kv, v) >= threshold);
        if !is_near_duplicate {
            kept_indices.push(i);
            kept_vecs.push(v);
        }
    }

    Ok(kept_indices.into_iter().map(|i| uniq[i].clone()).collect())
}

/// Split on sentence-ending punctuation, keeping only sentences within `MIN_WORDS..=MAX_WORDS` -- too short/long to judge as a standalone concept.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            current.clear();
        }
    }
    let tail = current.trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out.into_iter()
        .filter(|s| (MIN_WORDS..=MAX_WORDS).contains(&s.split_whitespace().count()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedder() -> Embedder {
        Embedder::new(crate::bank::data_root().join("models")).unwrap()
    }

    #[test]
    fn sentences_drops_too_short_and_too_long() {
        let text =
            "Too short. This one has exactly enough words to qualify as a real concept here today.";
        let result = sentences(text);
        assert_eq!(result.len(), 1);
        assert!(result[0].starts_with("This one has"));
    }

    #[test]
    fn sentences_splits_on_terminal_punctuation() {
        let text = "The quarterly onboarding checklist always requires explicit manager signoff! Does the deployment pipeline actually run its nightly integration builds? Yes it runs every single night without fail, rain or shine.";
        let result = sentences(text);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn split_collapses_exact_duplicate_sentences() {
        let sentence =
            "The quarterly onboarding checklist requires manager signoff before day three.";
        let text = format!("{sentence} {sentence}");
        let mut e = embedder();
        let result = split(&text, &mut e, 0.8).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn split_collapses_a_genuine_paraphrase_pair_at_calibrated_threshold() {
        let text = "New hires must get their manager's signoff on the onboarding checklist by their third day. \
                     The onboarding checklist needs manager approval before the third day for all new employees.";
        let mut e = embedder();
        let result = split(text, &mut e, 0.8).unwrap();
        assert_eq!(
            result.len(),
            1,
            "two paraphrased sentences about the same fact should collapse to one concept, got: {result:?}"
        );
    }

    #[test]
    fn split_keeps_genuinely_distinct_concepts() {
        let text = "New hires must get their manager's signoff on the onboarding checklist by their third day. \
                     The deployment pipeline runs a full integration test suite before every production release.";
        let mut e = embedder();
        let result = split(text, &mut e, 0.8).unwrap();
        assert_eq!(
            result.len(),
            2,
            "two unrelated real facts should both survive as distinct concepts, got: {result:?}"
        );
    }

    #[test]
    fn split_on_empty_text_returns_empty_without_loading_a_model() {
        let mut e = embedder();
        assert_eq!(split("", &mut e, 0.8).unwrap(), Vec::<String>::new());
    }
}
