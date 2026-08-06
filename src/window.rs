//! Split long content into overlapping word-count windows so a fixed-
//! context embedder can represent all of it, not just the first slice.
//!
//! Window size is calibrated against real measurements, not assumed.
//!
//! First measurement (`embeddings::tests`, 2026-08-06 probe): cosine
//! similarity to a needle-matching query went perfectly flat once filler
//! exceeded 200-250 words past an 11-word needle (~211-261 total words) —
//! the real hard truncation ceiling, confirming the previously-documented
//! ~256-token estimate.
//!
//! Second measurement (`wiki::tests`, same session): the hard ceiling
//! isn't the whole story. Even well inside it, a window containing mostly
//! *topically unrelated* real content (not the first probe's nonsense
//! filler) dilutes a mean-pooled embedding hard — scores clustered at
//! ~0.28-0.30 across window sizes from 50 to 180 words against a
//! deliberately worst-case fixture (a real point buried in 600 words of a
//! single unrelated topic). Window/overlap tuning alone doesn't fully
//! solve same-topic dilution; a smaller window reduces it but can't
//! eliminate it when a real "concept" is a small fraction of any window
//! that still contains it. This is the concrete, measured case for
//! ADR-0004's other accepted fix — concept-level chunking *before*
//! persistence (small, mostly-single-topic stored units) — this windowing
//! fix is a real mitigation for already-stored large files, not a
//! substitute for storing atomic concepts in the first place.
//!
//! `WINDOW_WORDS` sits well under the measured hard ceiling, small enough
//! to keep same-topic dilution bounded on real content (which mixes
//! related sub-topics, not the maximally-adversarial single unrelated
//! topic the fixture above deliberately tested).
pub const WINDOW_WORDS: usize = 80;
pub const OVERLAP_WORDS: usize = 20;

/// `(char_offset, window_text)` pairs covering `content`. A short document
/// (at or under `WINDOW_WORDS`) produces exactly one window spanning the
/// whole thing. Windows overlap by `OVERLAP_WORDS` so a real point that
/// happens to sit near a window boundary isn't split awkwardly across two
/// windows with neither containing it in full context.
pub fn windows(content: &str, window_words: usize, overlap_words: usize) -> Vec<(usize, String)> {
    // (word_start_char_offset, word_text) — needed to recover real char
    // offsets after grouping into windows, since word boundaries aren't
    // fixed-width.
    let words: Vec<(usize, &str)> = content
        .split_word_bound_indices()
        .filter(|(_, w)| !w.trim().is_empty())
        .collect();

    if words.is_empty() {
        return Vec::new();
    }
    if words.len() <= window_words {
        return vec![(0, content.to_string())];
    }

    let stride = window_words.saturating_sub(overlap_words).max(1);
    let mut result = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + window_words).min(words.len());
        let offset = words[start].0;
        let end_offset = if end == words.len() {
            content.len()
        } else {
            words[end].0
        };
        result.push((offset, content[offset..end_offset].trim_end().to_string()));
        if end == words.len() {
            break;
        }
        start += stride;
    }
    result
}

/// Minimal word-boundary splitter (offset, word) — content here is plain
/// English prose/markdown, not requiring full Unicode segmentation
/// machinery for a correctness-critical path; whitespace-run boundaries
/// are sufficient and keep this dependency-free.
trait WordBoundIndices {
    fn split_word_bound_indices(&self) -> WordBoundIter<'_>;
}

impl WordBoundIndices for str {
    fn split_word_bound_indices(&self) -> WordBoundIter<'_> {
        WordBoundIter {
            content: self,
            pos: 0,
        }
    }
}

struct WordBoundIter<'a> {
    content: &'a str,
    pos: usize,
}

impl<'a> Iterator for WordBoundIter<'a> {
    type Item = (usize, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.content.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if self.pos >= bytes.len() {
            return None;
        }
        let start = self.pos;
        while self.pos < bytes.len() && !bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        Some((start, &self.content[start..self.pos]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_content_produces_exactly_one_window_spanning_it_all() {
        let content = "one two three four five";
        let result = windows(content, 180, 30);
        assert_eq!(result, vec![(0, content.to_string())]);
    }

    #[test]
    fn empty_content_produces_no_windows() {
        assert_eq!(windows("", 180, 30), Vec::new());
        assert_eq!(windows("   ", 180, 30), Vec::new());
    }

    #[test]
    fn long_content_produces_multiple_overlapping_windows() {
        let words: Vec<String> = (0..500).map(|i| format!("w{i}")).collect();
        let content = words.join(" ");

        let result = windows(&content, 100, 20);

        assert!(result.len() > 1, "expected multiple windows, got 1");
        // First window starts at the real beginning.
        assert_eq!(result[0].0, 0);
        assert!(result[0].1.starts_with("w0 "));
        // Consecutive windows overlap: the tail of one window's word set
        // should reappear at the head of the next.
        let first_words: Vec<&str> = result[0].1.split_whitespace().collect();
        let second_words: Vec<&str> = result[1].1.split_whitespace().collect();
        assert_eq!(
            first_words[first_words.len() - 20],
            second_words[0],
            "window 2 should start 20 words before window 1's end (the overlap)"
        );
    }

    #[test]
    fn last_window_reaches_the_real_end_of_content() {
        let words: Vec<String> = (0..250).map(|i| format!("w{i}")).collect();
        let content = words.join(" ");

        let result = windows(&content, 100, 20);
        let last = result.last().unwrap();
        assert!(
            last.1.ends_with("w249"),
            "last window should reach the true end, got: ...{}",
            &last.1[last.1.len().saturating_sub(20)..]
        );
    }

    #[test]
    fn char_offsets_are_real_positions_into_the_original_content() {
        let content = "alpha beta gamma delta epsilon zeta eta theta";
        let result = windows(content, 3, 1);
        for (offset, window_text) in &result {
            let first_word_in_window = window_text.split_whitespace().next().unwrap();
            assert!(
                content[*offset..].starts_with(first_word_in_window),
                "offset {offset} should point at {first_word_in_window:?} in the original content"
            );
        }
    }
}
