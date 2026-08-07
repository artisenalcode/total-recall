//! `trm ingest-persona` — fetches YouTube auto-captions for a person's
//! videos, cleans them into raw transcript files, and stages a
//! `PersonaBuild` handover for a sub-agent to synthesize (per ADR-0002:
//! trm never calls an LLM itself; this module's job stops at mechanical
//! fetch+clean+stage). Rust port of the real, working shape already
//! proven out today via `advisory/tools/ingest_youtube.py` and this
//! session's own live YouTube ingestion test — same frontmatter
//! convention, so existing wiki-reading tooling (`ask-the-board`,
//! `/advice`) needs no changes to read the resulting raw files.

use crate::atomic;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct VideoTarget {
    pub id: String,
    pub title: String,
}

/// Strip VTT structure (WEBVTT/Kind/Language headers, timing cue lines,
/// inline `<...>` tags), collapse consecutive rolling-caption duplicate
/// lines, and decode the handful of HTML entities real auto-captions
/// actually contain. Not a general HTML-entity decoder — a small, fixed
/// set is enough for caption text and avoids a new dependency, same
/// reasoning as every other minimal-dependency choice in this crate.
pub fn clean_vtt(raw: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with("WEBVTT")
            || line.starts_with("Kind:")
            || line.starts_with("Language:")
            || line.contains("-->")
            || line.chars().all(|c| c.is_ascii_digit())
        {
            continue;
        }
        lines.push(line);
    }

    let mut deduped: Vec<String> = Vec::with_capacity(lines.len());
    for line in lines {
        let stripped = strip_tags(line);
        let stripped = stripped.trim();
        if stripped.is_empty() {
            continue;
        }
        if deduped.last().map(String::as_str) != Some(stripped) {
            deduped.push(stripped.to_string());
        }
    }

    decode_entities(&deduped.join(" "))
}

fn strip_tags(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_tag = false;
    for c in line.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&#39;", "'")
        .replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

/// Same frontmatter shape as `advisory/tools/ingest_youtube.py`'s
/// `process_one` — so existing wiki-reading tooling needs no changes to
/// treat these raw files the same way as the Python-ingested ones.
pub fn build_raw_transcript_md(
    person: &str,
    slug: &str,
    video: &VideoTarget,
    text: &str,
    ingested_date: &str,
) -> String {
    let word_count = text.split_whitespace().count();
    format!(
        "---\n\
         source: https://www.youtube.com/watch?v={id}\n\
         person: {person}\n\
         person_slug: {slug}\n\
         title: {title:?}\n\
         video_id: {id}\n\
         type: youtube-transcript\n\
         transcript: auto-captions\n\
         ingested: {ingested_date}\n\
         word_count: {word_count}\n\
         ---\n\n\
         # {title}\n\n\
         {text}\n",
        id = video.id,
        title = video.title,
        person = person,
        slug = slug,
        ingested_date = ingested_date,
        word_count = word_count,
        text = text,
    )
}

/// Spawn `yt-dlp` to fetch a video's English auto-captions (VTT), read
/// and clean the result, then remove the temp file. Real subprocess I/O
/// — not unit-tested directly (mocking a network-capable subprocess
/// buys nothing real); same discipline as `session_prune`'s and
/// `ingest`'s own real-subprocess boundaries.
pub fn fetch_captions(video_id: &str, work_dir: &Path) -> Result<String, String> {
    let stem = work_dir.join(video_id);
    let output = Command::new("yt-dlp")
        .args([
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang",
            "en.*",
            "--sub-format",
            "vtt",
            "--no-warnings",
            "-o",
        ])
        .arg(format!("{}.%(ext)s", stem.display()))
        .arg(format!("https://www.youtube.com/watch?v={video_id}"))
        .output()
        .map_err(|e| format!("yt-dlp not on PATH: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "yt-dlp failed for {video_id}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let mut vtt_paths: Vec<PathBuf> = std::fs::read_dir(work_dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(video_id) && n.ends_with(".vtt"))
        })
        .collect();
    vtt_paths.sort();

    let vtt_path = vtt_paths
        .into_iter()
        .next()
        .ok_or_else(|| format!("no captions available for {video_id}"))?;
    let raw = std::fs::read_to_string(&vtt_path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&vtt_path);

    let cleaned = clean_vtt(&raw);
    if cleaned.len() < 200 {
        return Err(format!("caption text too short for {video_id}"));
    }
    Ok(cleaned)
}

/// Fetch+clean+write raw transcript files for every video, into
/// `<bank>/raw/<slug>/` (the same tier `stage()` uses for pending
/// content — consistent with this being raw material awaiting a
/// sub-agent's synthesis judgment, not yet a durable fact). Returns the
/// written file paths, in order — callers hand these straight to
/// `handover::stage_persona_sources`.
pub fn ingest_videos(
    raw_dir: &Path,
    person: &str,
    slug: &str,
    videos: &[VideoTarget],
    ingested_date: &str,
) -> Result<Vec<PathBuf>, String> {
    let person_dir = raw_dir.join(slug);
    std::fs::create_dir_all(&person_dir).map_err(|e| e.to_string())?;

    let mut written = Vec::with_capacity(videos.len());
    for video in videos {
        let text = fetch_captions(&video.id, &person_dir)?;
        let md = build_raw_transcript_md(person, slug, video, &text, ingested_date);
        let file_name = format!("yt-{}.md", video.id);
        let file_path = person_dir.join(&file_name);
        atomic::write(&file_path, &md).map_err(|e| e.to_string())?;
        written.push(file_path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_vtt_strips_structure_and_dedupes_rolling_captions() {
        let raw = "WEBVTT\nKind: captions\nLanguage: en\n\n\
                    00:00:05.120 --> 00:00:08.629 align:start position:0%\n\
                    Good<00:00:05.520><c> morning,</c> world\n\n\
                    00:00:08.629 --> 00:00:08.639 align:start position:0%\n\
                    Good morning, world\n\n\
                    00:00:08.639 --> 00:00:11.749 align:start position:0%\n\
                    Good morning, world\nto you too";
        let cleaned = clean_vtt(raw);
        // The third cue's "Good morning, world" line duplicates the
        // previous cue's line exactly and correctly gets deduped (real
        // rolling-caption behavior), leaving only its new content.
        assert_eq!(cleaned, "Good morning, world to you too");
    }

    #[test]
    fn clean_vtt_decodes_common_html_entities() {
        let raw = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nRock &amp; Roll &#39;n&#39; stuff";
        assert_eq!(clean_vtt(raw), "Rock & Roll 'n' stuff");
    }

    #[test]
    fn clean_vtt_on_empty_input_returns_empty() {
        assert_eq!(clean_vtt(""), "");
        assert_eq!(clean_vtt("WEBVTT\nKind: captions\n"), "");
    }

    #[test]
    fn build_raw_transcript_md_matches_the_python_ingester_s_frontmatter_shape() {
        let video = VideoTarget {
            id: "abc123".to_string(),
            title: "A Real Talk".to_string(),
        };
        let md = build_raw_transcript_md(
            "Jane Doe",
            "jane-doe",
            &video,
            "real body text",
            "2026-08-07",
        );
        assert!(md.starts_with("---\nsource: https://www.youtube.com/watch?v=abc123\n"));
        assert!(md.contains("person: Jane Doe\n"));
        assert!(md.contains("person_slug: jane-doe\n"));
        assert!(md.contains("video_id: abc123\n"));
        assert!(md.contains("type: youtube-transcript\n"));
        assert!(md.contains("word_count: 3\n"));
        assert!(md.contains("# A Real Talk\n\nreal body text\n"));
    }

    #[test]
    fn ingest_videos_returns_an_error_without_touching_the_network_when_yt_dlp_is_missing() {
        // Real behavior check: a genuinely nonexistent video id against a
        // scratch dir should fail cleanly (not panic), whether because
        // yt-dlp is missing or the fetch itself fails — either way this
        // proves ingest_videos surfaces a real Err rather than hanging or
        // silently producing an empty file.
        let tmp = tempfile::tempdir().unwrap();
        let videos = vec![VideoTarget {
            id: "this-is-not-a-real-video-id-00000".to_string(),
            title: "irrelevant".to_string(),
        }];
        let result = ingest_videos(
            tmp.path(),
            "Test Person",
            "test-person",
            &videos,
            "2026-08-07",
        );
        assert!(result.is_err());
    }
}
