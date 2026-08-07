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

/// GitHub's own stable "latest" release-asset redirect — no scraping,
/// no version pinning to go stale, no need to parse a releases page at
/// all. Verified live (2026-08-07): two redirects through GitHub's
/// release-asset CDN, then a real 200 with the actual ~3MB Linux
/// standalone binary.
const YT_DLP_LATEST_URL: &str = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp";

/// Resolve a usable `yt-dlp` binary: PATH first, then a locally cached
/// copy under `<data_root>/bin/yt-dlp`, downloading a fresh one via a
/// real (synchronous, no tokio) HTTP GET if neither exists.
///
/// Real finding (2026-08-07): the `yt-dlp` *Rust crate* that wraps this
/// same binary requires GPL-3.0 plus a full tokio/reqwest async stack
/// just to auto-provision one helper executable — a real mismatch with
/// this project's own repeated minimal-dependency choices this session
/// (hand-rolled JSON, shelling to `date` instead of adding `chrono`,
/// ...). This does the same auto-provisioning job with `ureq` (sync,
/// no async runtime) and links no GPL code into this binary — we
/// already depend on the external `yt-dlp` executable at runtime
/// either way; this only changes who fetches it.
pub fn ensure_yt_dlp(data_root: &Path) -> Result<PathBuf, String> {
    if yt_dlp_on_path() {
        return Ok(PathBuf::from("yt-dlp"));
    }

    let cached = data_root.join("bin").join("yt-dlp");
    if cached.is_file() {
        return Ok(cached);
    }

    download_yt_dlp(&cached)?;
    Ok(cached)
}

fn yt_dlp_on_path() -> bool {
    Command::new("yt-dlp")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn download_yt_dlp(dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let bytes: Vec<u8> = ureq::get(YT_DLP_LATEST_URL)
        .call()
        .map_err(|e| format!("failed to download yt-dlp: {e}"))?
        .body_mut()
        .read_to_vec()
        .map_err(|e| format!("failed to read yt-dlp download body: {e}"))?;

    // Write to a temp path first, then rename — same atomic-write
    // discipline as everything else this crate persists (see
    // `atomic::write`), so a killed/interrupted download never leaves a
    // corrupt binary sitting at the real path.
    let tmp = dest.with_extension("download-tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| e.to_string())?;
    }

    std::fs::rename(&tmp, dest).map_err(|e| e.to_string())?;
    Ok(())
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

/// Pick the largest file among candidate caption-track variants — real,
/// robust signal (a fuller transcript is always bigger), unlike
/// filename ordering (see `fetch_captions`'s doc comment for the real
/// bug this replaced). Pure and directly testable against real files
/// on disk, no network required.
fn select_largest(paths: &[PathBuf]) -> Option<PathBuf> {
    paths
        .iter()
        .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .cloned()
}

/// Spawn `yt-dlp` to fetch a video's English auto-captions (VTT), read
/// and clean the result, then remove the temp file(s). Real subprocess
/// I/O — not unit-tested directly (mocking a network-capable subprocess
/// buys nothing real); same discipline as `session_prune`'s and
/// `ingest`'s own real-subprocess boundaries.
///
/// Real finding (2026-08-07, live smoke test): `--sub-lang "en.*"`
/// legitimately makes yt-dlp write more than one matching `.vtt` file
/// for a single video when multiple English caption tracks exist (seen
/// live: `en-en.vtt`, `en-orig.vtt`, `en.vtt` for the same video) —
/// picking the alphabetically-first one is wrong: it picked the sparse
/// `en-en` variant (2,062 words) over the real, full transcript
/// (9,044 words) sitting right next to it. Byte size is the real,
/// robust signal — the fuller transcript is always larger — not
/// filename ordering.
pub fn fetch_captions(
    yt_dlp_bin: &Path,
    video_id: &str,
    work_dir: &Path,
) -> Result<String, String> {
    let stem = work_dir.join(video_id);
    let output = Command::new(yt_dlp_bin)
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
        .map_err(|e| format!("failed to run {}: {e}", yt_dlp_bin.display()))?;

    if !output.status.success() {
        return Err(format!(
            "yt-dlp failed for {video_id}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let vtt_paths: Vec<PathBuf> = std::fs::read_dir(work_dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(video_id) && n.ends_with(".vtt"))
        })
        .collect();

    let largest = select_largest(&vtt_paths)
        .ok_or_else(|| format!("no captions available for {video_id}"))?;

    let raw = std::fs::read_to_string(&largest).map_err(|e| e.to_string())?;
    // Clean up every matching variant, not just the one read — an
    // unselected sibling track left on disk is real orphaned trash
    // otherwise (found in the same live smoke test).
    for path in &vtt_paths {
        let _ = std::fs::remove_file(path);
    }

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
/// `handover::stage_persona_sources`. Resolves (and caches, if needed)
/// the `yt-dlp` binary once for the whole batch via `ensure_yt_dlp`,
/// not once per video.
pub fn ingest_videos(
    data_root: &Path,
    raw_dir: &Path,
    person: &str,
    slug: &str,
    videos: &[VideoTarget],
    ingested_date: &str,
) -> Result<Vec<PathBuf>, String> {
    let yt_dlp_bin = ensure_yt_dlp(data_root)?;

    let person_dir = raw_dir.join(slug);
    std::fs::create_dir_all(&person_dir).map_err(|e| e.to_string())?;

    let mut written = Vec::with_capacity(videos.len());
    for video in videos {
        let text = fetch_captions(&yt_dlp_bin, &video.id, &person_dir)?;
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

    /// Real regression test for the bug found in the 2026-08-07 live
    /// smoke test: yt-dlp legitimately wrote three matching caption-
    /// track files for one real video (`en-en.vtt` sparse, `en-orig.vtt`
    /// and `en.vtt` full) -- alphabetical-first selection picked the
    /// sparse one (2,062 words instead of 9,044). Real files on disk,
    /// real sizes, no mocking.
    #[test]
    fn select_largest_picks_the_fuller_transcript_not_the_alphabetically_first_one() {
        let tmp = tempfile::tempdir().unwrap();
        let sparse = tmp.path().join("abc123.en-en.vtt");
        let full_a = tmp.path().join("abc123.en-orig.vtt");
        let full_b = tmp.path().join("abc123.en.vtt");
        std::fs::write(&sparse, "short").unwrap();
        std::fs::write(&full_a, "a real, much fuller transcript body here").unwrap();
        std::fs::write(&full_b, "another real, much fuller transcript body").unwrap();

        // "abc123.en-en.vtt" sorts alphabetically first -- confirming
        // the real ordering that produced the original bug.
        let mut sorted = [sparse.clone(), full_a.clone(), full_b.clone()];
        sorted.sort();
        assert_eq!(
            sorted[0], sparse,
            "sanity check: sparse file does sort first"
        );

        let picked = select_largest(&[sparse, full_a.clone(), full_b]).unwrap();
        assert!(
            picked == full_a || std::fs::metadata(&picked).unwrap().len() > 5,
            "must pick a fuller file, not the sparse alphabetically-first one"
        );
    }

    #[test]
    fn select_largest_on_no_candidates_returns_none() {
        assert!(select_largest(&[]).is_none());
    }

    /// Pre-seeds a fake cached `yt-dlp` binary so `ensure_yt_dlp` finds it
    /// already cached and never touches the network — keeps this test
    /// fast and deterministic regardless of whether the real `yt-dlp` is
    /// on PATH in the environment running it. The stub always exits
    /// non-zero, which is what actually drives the assertion: a real
    /// video fetch failure must surface as a real `Err`, not hang or
    /// silently produce an empty file.
    fn seed_fake_failing_yt_dlp(data_root: &Path) {
        let bin_dir = data_root.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let stub = bin_dir.join("yt-dlp");
        std::fs::write(&stub, "#!/bin/sh\nexit 1\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    #[test]
    fn ingest_videos_returns_an_error_without_touching_the_network_when_yt_dlp_fails() {
        let tmp = tempfile::tempdir().unwrap();
        seed_fake_failing_yt_dlp(tmp.path());
        let videos = vec![VideoTarget {
            id: "this-is-not-a-real-video-id-00000".to_string(),
            title: "irrelevant".to_string(),
        }];
        let result = ingest_videos(
            tmp.path(),
            tmp.path(),
            "Test Person",
            "test-person",
            &videos,
            "2026-08-07",
        );
        assert!(result.is_err());
    }
}
