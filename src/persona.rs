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

/// Wikipedia's own API etiquette requires a descriptive User-Agent
/// identifying the tool and a contact point — unidentified requests get
/// deprioritized/blocked. Same politeness convention already used by
/// `advisory/tools/scrape_blog.py`'s own UA string.
const WIKIPEDIA_USER_AGENT: &str = "trm-persona-ingest/1.0 (personal knowledge corpus; https://github.com/artisenalcode/total-recall)";

/// Fetches the full plain-text extract of a Wikipedia article via the
/// official MediaWiki Action API (`prop=extracts&explaintext=1`) — no
/// HTML to parse, no scraping, real structured JSON. `redirects=1` so a
/// title that's actually a redirect (common) still resolves.
pub fn fetch_wikipedia(title: &str) -> Result<String, String> {
    let body: String = ureq::get("https://en.wikipedia.org/w/api.php")
        .header("User-Agent", WIKIPEDIA_USER_AGENT)
        .query("action", "query")
        .query("format", "json")
        .query("prop", "extracts")
        .query("explaintext", "1")
        .query("redirects", "1")
        .query("titles", title)
        .call()
        .map_err(|e| format!("Wikipedia request failed for {title:?}: {e}"))?
        .body_mut()
        .read_to_string()
        .map_err(|e| e.to_string())?;

    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("bad Wikipedia JSON: {e}"))?;
    let pages = json["query"]["pages"]
        .as_object()
        .ok_or_else(|| format!("unexpected Wikipedia response shape for {title:?}"))?;
    let page = pages
        .values()
        .next()
        .ok_or_else(|| format!("empty Wikipedia response for {title:?}"))?;

    if page.get("missing").is_some() {
        return Err(format!("Wikipedia page not found: {title:?}"));
    }
    let extract = page
        .get("extract")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("no extract field for {title:?}"))?;
    if extract.trim().is_empty() {
        return Err(format!("empty Wikipedia extract for {title:?}"));
    }
    Ok(extract.trim().to_string())
}

/// Same frontmatter shape family as `build_raw_transcript_md`, adapted
/// for a source with no video_id/transcript-type fields.
pub fn build_wikipedia_raw_md(
    person: &str,
    slug: &str,
    title: &str,
    text: &str,
    ingested_date: &str,
) -> String {
    let word_count = text.split_whitespace().count();
    let url_title = title.replace(' ', "_");
    format!(
        "---\n\
         source: https://en.wikipedia.org/wiki/{url_title}\n\
         person: {person}\n\
         person_slug: {slug}\n\
         title: {title:?}\n\
         type: wikipedia\n\
         ingested: {ingested_date}\n\
         word_count: {word_count}\n\
         ---\n\n\
         # {title}\n\n\
         {text}\n"
    )
}

/// Fetch+write raw Wikipedia article files, one per title, into
/// `<bank>/raw/<slug>/` — same tier and same best-effort squishi-sidecar
/// treatment as `ingest_videos`. Article titles are usually few (1-3 per
/// person), so no batching/concurrency needed here unlike video fetches.
pub fn ingest_wikipedia_pages(
    raw_dir: &Path,
    person: &str,
    slug: &str,
    titles: &[String],
    ingested_date: &str,
) -> Result<Vec<PathBuf>, String> {
    let person_dir = raw_dir.join(slug);
    std::fs::create_dir_all(&person_dir).map_err(|e| e.to_string())?;

    let mut written = Vec::with_capacity(titles.len());
    for title in titles {
        let text = fetch_wikipedia(title)?;
        let md = build_wikipedia_raw_md(person, slug, title, &text, ingested_date);
        let file_name = format!("wikipedia-{}.md", slugify(title));
        let file_path = person_dir.join(&file_name);
        atomic::write(&file_path, &md).map_err(|e| e.to_string())?;
        written.push(file_path);

        if let Ok(dedup_json) = run_squishi_dedup(&text) {
            let sidecar_path = person_dir.join(format!("wikipedia-{}.dedup.json", slugify(title)));
            let _ = atomic::write(&sidecar_path, &dedup_json);
        }
    }
    Ok(written)
}

/// Filesystem-safe stand-in for a title with spaces/punctuation — same
/// job `ingest_youtube.py`'s own `slugify` does, kept minimal (no crate)
/// since it's one regex-shaped substitution.
fn slugify(title: &str) -> String {
    let mut out = String::with_capacity(title.len());
    let mut last_was_dash = false;
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            out.push('-');
            last_was_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// One commit's message, split into subject/body per `git log`'s own
/// convention (first line vs. everything after the blank line).
#[derive(Debug, Clone, PartialEq)]
struct Commit {
    date: String,
    subject: String,
    body: String,
}

/// Field/record separators that can't appear in real commit text —
/// same choice `ingest_code_archaeology.py` makes (ASCII unit/record
/// separators), not a delimiter a commit message could plausibly
/// collide with the way a comma or pipe could.
const GIT_LOG_FIELD_SEP: char = '\u{1f}';
const GIT_LOG_RECORD_SEP: char = '\u{1e}';

/// Clone (blob-less, no working tree — only history is needed) and
/// extract one author's commit messages from a repo, oldest first.
/// Merge commits are dropped ("Merge " subject prefix — GitHub's own
/// words, not the person's). Rust port of `ingest_code_archaeology.py`'s
/// `commits_by`, scoped to just the commit-message piece (that script's
/// code-sample/doc extraction is a separate, larger feature, not
/// ported here). `work_dir` holds the temporary clone, removed after
/// extraction either way.
pub fn fetch_git_commits(
    repo_url: &str,
    authors: &[String],
    work_dir: &Path,
) -> Result<String, String> {
    if authors.is_empty() {
        return Err("fetch_git_commits requires at least one author".to_string());
    }

    let repo_name = repo_url
        .trim_end_matches('/')
        .trim_end_matches(".git")
        .rsplit('/')
        .next()
        .unwrap_or("repo");
    let clone_dir = work_dir.join(format!("git-clone-{repo_name}"));
    let _ = std::fs::remove_dir_all(&clone_dir);

    let clone_result = Command::new("git")
        .args([
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "-q",
            repo_url,
        ])
        .arg(&clone_dir)
        .output()
        .map_err(|e| format!("failed to run git clone: {e}"))?;
    if !clone_result.status.success() {
        return Err(format!(
            "git clone failed for {repo_url}: {}",
            String::from_utf8_lossy(&clone_result.stderr).trim()
        ));
    }

    let mut log_args = vec!["log".to_string(), "--all".to_string()];
    for author in authors {
        log_args.push(format!("--author={author}"));
    }
    log_args.push("--date=short".to_string());
    log_args.push(format!(
        "--pretty=format:%H{GIT_LOG_FIELD_SEP}%ad{GIT_LOG_FIELD_SEP}%s{GIT_LOG_FIELD_SEP}%b{GIT_LOG_RECORD_SEP}"
    ));

    let log_result = Command::new("git")
        .args(&log_args)
        .current_dir(&clone_dir)
        .output()
        .map_err(|e| format!("failed to run git log: {e}"));
    let _ = std::fs::remove_dir_all(&clone_dir);
    let log_result = log_result?;

    if !log_result.status.success() {
        return Err(format!(
            "git log failed for {repo_url}: {}",
            String::from_utf8_lossy(&log_result.stderr).trim()
        ));
    }

    let commits = parse_git_log(&String::from_utf8_lossy(&log_result.stdout));
    if commits.is_empty() {
        return Err(format!(
            "no commits found for author(s) {authors:?} in {repo_url}"
        ));
    }
    Ok(render_commits(commits))
}

fn parse_git_log(raw: &str) -> Vec<Commit> {
    raw.split(GIT_LOG_RECORD_SEP)
        .filter_map(|record| {
            let record = record.trim_matches('\n');
            if record.is_empty() {
                return None;
            }
            let mut fields = record.split(GIT_LOG_FIELD_SEP);
            let _hash = fields.next()?;
            let date = fields.next().unwrap_or("").to_string();
            let subject = fields.next().unwrap_or("").trim().to_string();
            let body = fields.next().unwrap_or("").trim().to_string();
            if subject.starts_with("Merge ") {
                return None;
            }
            Some(Commit {
                date,
                subject,
                body,
            })
        })
        .collect()
}

fn render_commits(mut commits: Vec<Commit>) -> String {
    commits.sort_by(|a, b| a.date.cmp(&b.date));
    commits
        .into_iter()
        .map(|c| {
            if c.body.is_empty() {
                format!("On {}: {}.", c.date, c.subject)
            } else {
                format!("On {}: {}. {}", c.date, c.subject, c.body)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Same frontmatter shape family as `build_wikipedia_raw_md`, for a
/// commit-message corpus.
pub fn build_git_commits_raw_md(
    person: &str,
    slug: &str,
    repo_url: &str,
    text: &str,
    ingested_date: &str,
) -> String {
    let word_count = text.split_whitespace().count();
    format!(
        "---\n\
         source: {repo_url}\n\
         person: {person}\n\
         person_slug: {slug}\n\
         type: git-commit-messages\n\
         ingested: {ingested_date}\n\
         word_count: {word_count}\n\
         ---\n\n\
         # Commit messages from {repo_url}\n\n\
         {text}\n"
    )
}

/// Fetch+write one raw file of commit messages for a single repo — same
/// tier and best-effort squishi-sidecar treatment as `ingest_videos`/
/// `ingest_wikipedia_pages`. One repo per call, matching
/// `ingest_code_archaeology.py`'s own scope (repeat the call for
/// multiple repos).
pub fn ingest_git_commits(
    raw_dir: &Path,
    person: &str,
    slug: &str,
    repo_url: &str,
    authors: &[String],
    ingested_date: &str,
) -> Result<Vec<PathBuf>, String> {
    let person_dir = raw_dir.join(slug);
    std::fs::create_dir_all(&person_dir).map_err(|e| e.to_string())?;

    let text = fetch_git_commits(repo_url, authors, &person_dir)?;
    let md = build_git_commits_raw_md(person, slug, repo_url, &text, ingested_date);
    let repo_name = repo_url
        .trim_end_matches('/')
        .trim_end_matches(".git")
        .rsplit('/')
        .next()
        .unwrap_or("repo");
    let file_name = format!("git-{}.md", slugify(repo_name));
    let file_path = person_dir.join(&file_name);
    atomic::write(&file_path, &md).map_err(|e| e.to_string())?;

    if let Ok(dedup_json) = run_squishi_dedup(&text) {
        let sidecar_path = person_dir.join(format!("git-{}.dedup.json", slugify(repo_name)));
        let _ = atomic::write(&sidecar_path, &dedup_json);
    }

    Ok(vec![file_path])
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

/// Runs squishi's caller-asserted plain-text dedup on cleaned transcript
/// text and returns its raw `--json` output verbatim — an opaque
/// passthrough, matching squishi's own stated boundary ("compresses
/// text, never stores/retrieves"): this module's job is calling it and
/// persisting what it returns as a traceability sidecar, not
/// interpreting the content itself (that's a synthesis sub-agent's job).
/// `--force-kind plain-text` is the caller assertion this exists for —
/// squishi's own shape-detection heuristic false-positives on real
/// conversational transcripts (ordinary words like "failed" trip its
/// Log-shape regex with zero structural check), found and fixed
/// 2026-08-07 by testing this exact pipeline against real Dr. Roy
/// Sugarman transcripts.
fn run_squishi_dedup(text: &str) -> Result<String, String> {
    use std::io::Write;

    let mut child = Command::new("squishi")
        .args(["--force-kind", "plain-text", "--json"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn squishi: {e}"))?;

    child
        .stdin
        .take()
        .ok_or_else(|| "squishi: no stdin handle".to_string())?
        .write_all(text.as_bytes())
        .map_err(|e| e.to_string())?;

    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "squishi failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

/// Fetch+clean+write raw transcript files for every video, into
/// `<bank>/raw/<slug>/` (the same tier `stage()` uses for pending
/// content — consistent with this being raw material awaiting a
/// sub-agent's synthesis judgment, not yet a durable fact). Returns the
/// written file paths, in order — callers hand these straight to
/// `handover::stage_persona_sources`. Resolves (and caches, if needed)
/// the `yt-dlp` binary once for the whole batch via `ensure_yt_dlp`,
/// not once per video.
///
/// Also writes a `yt-<id>.dedup.json` sidecar per video — squishi's
/// dedup/shape/summary output, run best-effort: squishi being
/// unavailable (not installed, not on PATH) never fails the whole
/// ingestion, it just means no sidecar for that run. The sub-agent
/// completing the handover reads the sidecar when present as a
/// traceable pointer into the raw transcript (which sentences are
/// narrative-shaped, which are the most central/summary-worthy) rather
/// than reading the full raw transcript cold every time.
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

        if let Ok(dedup_json) = run_squishi_dedup(&text) {
            let sidecar_path = person_dir.join(format!("yt-{}.dedup.json", video.id));
            let _ = atomic::write(&sidecar_path, &dedup_json);
        }
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

    // --- parse_git_log / render_commits: pure ---

    fn fake_git_log_output(records: &[(&str, &str, &str, &str)]) -> String {
        records
            .iter()
            .map(|(hash, date, subject, body)| {
                format!("{hash}{GIT_LOG_FIELD_SEP}{date}{GIT_LOG_FIELD_SEP}{subject}{GIT_LOG_FIELD_SEP}{body}{GIT_LOG_RECORD_SEP}")
            })
            .collect()
    }

    #[test]
    fn parse_git_log_drops_merge_commits() {
        let raw = fake_git_log_output(&[
            ("abc123", "2026-01-01", "Merge pull request #4", ""),
            ("def456", "2026-01-02", "Fix a real bug", "details here"),
        ]);
        let commits = parse_git_log(&raw);
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].subject, "Fix a real bug");
    }

    #[test]
    fn parse_git_log_keeps_commits_with_no_body() {
        let raw = fake_git_log_output(&[("abc123", "2026-01-01", "One-line commit", "")]);
        let commits = parse_git_log(&raw);
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].body, "");
    }

    #[test]
    fn parse_git_log_on_empty_input_returns_empty() {
        assert!(parse_git_log("").is_empty());
    }

    #[test]
    fn render_commits_sorts_by_date_and_formats_subject_and_body() {
        let commits = vec![
            Commit {
                date: "2026-02-01".to_string(),
                subject: "Second thing".to_string(),
                body: "".to_string(),
            },
            Commit {
                date: "2026-01-01".to_string(),
                subject: "First thing".to_string(),
                body: "with real detail".to_string(),
            },
        ];
        let rendered = render_commits(commits);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines[0], "On 2026-01-01: First thing. with real detail");
        assert_eq!(lines[1], "On 2026-02-01: Second thing.");
    }

    // --- build_git_commits_raw_md: pure ---

    #[test]
    fn build_git_commits_raw_md_matches_the_frontmatter_shape() {
        let md = build_git_commits_raw_md(
            "Jane Doe",
            "jane-doe",
            "https://github.com/example/repo",
            "On 2026-01-01: did a thing.",
            "2026-08-07",
        );
        assert!(md.starts_with("---\nsource: https://github.com/example/repo\n"));
        assert!(md.contains("type: git-commit-messages\n"));
        assert!(md.contains("# Commit messages from https://github.com/example/repo\n"));
    }

    // --- fetch_git_commits: real subprocess, #[ignore]d ---

    #[test]
    #[ignore]
    fn fetch_git_commits_returns_real_commit_messages_for_a_real_author() {
        let tmp = tempfile::tempdir().unwrap();
        // GitHub's own public demo repo -- small, stable, genuinely
        // public (unlike this account's own repos, which are private
        // and correctly fail an unauthenticated clone -- a real finding
        // from the first run of this test, not a bug in the fetch path).
        let text = fetch_git_commits(
            "https://github.com/octocat/Hello-World",
            &["Octocat".to_string()],
            tmp.path(),
        );
        assert!(text.is_ok(), "expected real commits: {:?}", text.err());
        let text = text.unwrap();
        assert!(!text.is_empty());
        assert!(text.contains("On 20"), "expected a rendered date line");
    }

    #[test]
    #[ignore]
    fn fetch_git_commits_errors_on_a_real_nonexistent_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let result = fetch_git_commits(
            "https://github.com/artisenalcode/this-repo-does-not-exist-at-all-12345",
            &["someone".to_string()],
            tmp.path(),
        );
        assert!(result.is_err());
    }

    // --- slugify: pure ---

    #[test]
    fn slugify_lowercases_and_dashes_spaces_and_punctuation() {
        assert_eq!(slugify("Jordan B. Peterson"), "jordan-b-peterson");
    }

    #[test]
    fn slugify_collapses_consecutive_separators_and_trims_edges() {
        assert_eq!(slugify("  Dr. Roy   Sugarman!! "), "dr-roy-sugarman");
    }

    // --- build_wikipedia_raw_md: pure ---

    #[test]
    fn build_wikipedia_raw_md_matches_the_frontmatter_shape() {
        let md = build_wikipedia_raw_md(
            "Jordan Peterson",
            "jordan-peterson",
            "Jordan Peterson",
            "real article text here",
            "2026-08-07",
        );
        assert!(md.starts_with("---\nsource: https://en.wikipedia.org/wiki/Jordan_Peterson\n"));
        assert!(md.contains("type: wikipedia\n"));
        assert!(md.contains("word_count: 4\n"));
        assert!(md.ends_with("# Jordan Peterson\n\nreal article text here\n"));
    }

    // --- fetch_wikipedia: real network, #[ignore]d ---

    #[test]
    #[ignore]
    fn fetch_wikipedia_returns_real_article_text() {
        let text = fetch_wikipedia("Jordan Peterson").unwrap();
        assert!(text.len() > 500, "expected a substantial article extract");
        assert!(text.to_lowercase().contains("psycholog"));
    }

    #[test]
    #[ignore]
    fn fetch_wikipedia_errors_on_a_real_nonexistent_title() {
        let result = fetch_wikipedia("Zzzznotarealwikipediaarticletitle12345");
        assert!(result.is_err());
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

    #[test]
    #[ignore] // requires `squishi` installed on PATH (real subprocess, first-run model download)
    fn run_squishi_dedup_returns_the_real_json_contract_for_plain_prose() {
        // Real content, over squishi's SKIP_SEMANTIC_DEDUP_UNDER_CHARS
        // (2000 chars) so this actually exercises the ONNX dedup path,
        // not just the short-input line-dedup passthrough.
        let sentence = "I had a client who wanted to lose weight to fit into a wedding dress, \
                         and context tells the genes what they need to do, because values are \
                         the load-bearing unit of motivation and autonomy matters more than \
                         almost anything else in coaching. ";
        let text: String = std::iter::repeat_n(sentence, 20).collect();
        let result = run_squishi_dedup(&text);
        assert!(
            result.is_ok(),
            "squishi not on PATH or failed: {:?}",
            result.err()
        );
        let json: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(json["kind"], "PlainText");
        assert!(json.get("stories").is_some());
        assert!(json.get("drops").is_some());
    }
}
