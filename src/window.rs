//! Splits long content into overlapping word-count windows so a fixed-context embedder can represent all of it, not just the first slice.
//! `WINDOW_WORDS` sits well under the measured ~211-261-word hard truncation ceiling, small enough to keep same-topic dilution bounded.
pub const WINDOW_WORDS: usize = 80;
pub const OVERLAP_WORDS: usize = 20;

/// `(char_offset, window_text)` pairs covering `content`, overlapping by `OVERLAP_WORDS` so a boundary point isn't split across two windows.
pub fn windows(content: &str, window_words: usize, overlap_words: usize) -> Vec<(usize, String)> {
    // (word_start_char_offset, word_text), since word boundaries aren't fixed-width.
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

/// Minimal word-boundary splitter -- whitespace runs are sufficient for plain prose/markdown and keep this dependency-free.
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
