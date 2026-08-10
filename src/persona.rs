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
use crate::ingest;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct VideoTarget {
    pub id: String,
    pub title: String,
}

/// Every `--session` source (ADR-0007) stages into this fixed bank,
/// regardless of the session's own recorded cwd or `trm`'s invocation
/// cwd/`-p` — the one source type in this pipeline where "whatever bank
/// you're standing in" is actively wrong (a pattern surfaced in a
/// squishi session is about the user, not squishi). Hardcoded for now,
/// not user-configurable — fine for a single-user tool.
pub const SELF_PERSONA_BANK: &str = "self";

/// squishi's own `--session-digest` only reads plain files — no gzip
/// awareness — so a `sessions/<id>.jsonl.gz` archive (ADR-0006) needs
/// decompressing to a real file first. A plain `.jsonl` source is
/// returned unchanged: no copy, `work_dir` untouched, nothing for the
/// caller to clean up.
///
/// For a `.gz` source, the returned path is a freshly-created, uniquely-
/// named temp file under `work_dir` (pid+nanos in the name, same
/// uniqueness discipline `atomic::write`'s own tmp files use) — the
/// caller owns it and must remove it after use (compare the returned
/// path against `source` to tell the two cases apart). No tmp-then-
/// rename dance is needed here the way `atomic::write` needs one: the
/// output IS a disposable temp file, not an overwrite of a stable path,
/// so a write that fails partway just orphans a filename nothing will
/// ever read again (unique per call) — it can never be mistaken for real
/// input on a later call. The orphan is still removed on error, though,
/// rather than left for a human to find later.
pub fn read_transcript_path(source: &Path, work_dir: &Path) -> Result<PathBuf, String> {
    if source.extension().and_then(|e| e.to_str()) != Some("gz") {
        return Ok(source.to_path_buf());
    }

    let compressed =
        std::fs::read(source).map_err(|e| format!("failed to read {}: {e}", source.display()))?;
    let mut decoder = flate2::read::GzDecoder::new(compressed.as_slice());
    let mut decompressed = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut decompressed)
        .map_err(|e| format!("failed to decompress {}: {e}", source.display()))?;

    std::fs::create_dir_all(work_dir).map_err(|e| e.to_string())?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("session");
    let tmp_path = work_dir.join(format!(".{stem}.tmp-{}-{}", std::process::id(), nanos));

    if let Err(e) = std::fs::write(&tmp_path, &decompressed) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(format!(
            "failed to write decompressed {}: {e}",
            source.display()
        ));
    }
    Ok(tmp_path)
}

/// Fetch+write raw session-transcript files for every `--session` source
/// (ADR-0007), into `<bank>/raw/<slug>/` — same tier and same per-item
/// resilience posture as `ingest_videos` (one bad transcript doesn't
/// abort the batch; only `Err` outright when every path failed).
///
/// **No squishi dedup-batch sidecar here**, unlike
/// `ingest_videos`/`ingest_wikipedia_pages`/`ingest_git_commits` — those
/// write *raw, uncompressed* source text and get deduped separately for
/// sub-agent traceability. A session digest is already
/// squishi-compressed (`--session-digest` runs its own internal `route()`
/// pass before returning `content`), so a second dedup pass over the
/// same content already-deduped wouldn't be a traceability aid, just
/// redundant work. Deliberate, not a missed step.
///
/// **No resumability skip** either, unlike `ingest_videos` (which skips
/// already-fetched videos to avoid re-hitting network/rate limits) —
/// digesting a local transcript has no such cost, so re-running just
/// re-digests. Revisit only if a future bulk mode makes that wasteful at
/// scale (explicitly deferred, ADR-0007 Decision 3).
pub fn ingest_sessions(
    raw_dir: &Path,
    person: &str,
    slug: &str,
    session_paths: &[PathBuf],
    ingested_date: &str,
) -> Result<(Vec<PathBuf>, FetchFailures), String> {
    let person_dir = raw_dir.join(slug);
    std::fs::create_dir_all(&person_dir).map_err(|e| e.to_string())?;

    let mut written = Vec::with_capacity(session_paths.len());
    let mut failures: FetchFailures = Vec::new();

    for source_path in session_paths {
        let label = source_path.display().to_string();

        let digest_path = match read_transcript_path(source_path, &person_dir) {
            Ok(p) => p,
            Err(e) => {
                failures.push((label, e));
                continue;
            }
        };
        let is_temp = &digest_path != source_path;

        let digest_result = ingest::run_squishi_session_digest(&digest_path);
        if is_temp {
            let _ = std::fs::remove_file(&digest_path);
        }

        let digest = match digest_result {
            Ok(d) if d.content.trim().is_empty() => {
                failures.push((label, "nothing to digest (empty session)".to_string()));
                continue;
            }
            Ok(d) => d,
            Err(e) => {
                failures.push((label, e));
                continue;
            }
        };

        // Falls back to a slug of the source path only in the
        // (defensive, not expected in practice) case squishi's digest
        // came back with no session_id at all.
        let session_id = digest.session_id.clone().unwrap_or_else(|| slugify(&label));
        let md = build_session_raw_md(person, slug, &digest, source_path, ingested_date);
        let file_path = person_dir.join(format!("session-{session_id}.md"));
        if let Err(e) = atomic::write(&file_path, &md) {
            failures.push((label, e.to_string()));
            continue;
        }
        written.push(file_path);
    }

    if written.is_empty() {
        let summary = failures
            .iter()
            .map(|(id, e)| format!("{id}: {e}"))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!("every session failed to ingest: {summary}"));
    }

    Ok((written, failures))
}

/// Same frontmatter shape family as `build_wikipedia_raw_md`/
/// `build_git_commits_raw_md`, for a Claude Code session-transcript
/// source (ADR-0007). `session_cwd` falls back to `"unknown"` the same
/// way `ingest::build_reason` does, for a digest whose transcript never
/// recorded a `cwd` (defensive — real transcripts always have one, but
/// the field is `Option` on `SessionDigest`).
pub fn build_session_raw_md(
    person: &str,
    slug: &str,
    digest: &ingest::SessionDigest,
    source_path: &Path,
    ingested_date: &str,
) -> String {
    let session_id = digest.session_id.as_deref().unwrap_or("unknown");
    let session_cwd = digest.cwd.as_deref().unwrap_or("unknown");
    let word_count = digest.content.split_whitespace().count();
    format!(
        "---\n\
         source: {source}\n\
         person: {person}\n\
         person_slug: {slug}\n\
         type: claude-code-session\n\
         session_id: {session_id}\n\
         session_cwd: {session_cwd}\n\
         ingested: {ingested_date}\n\
         word_count: {word_count}\n\
         ---\n\n\
         {content}\n",
        source = source_path.display(),
        content = digest.content,
    )
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

/// Enumerate up to `max` of a channel's most-recent videos as
/// `VideoTarget`s, via `yt-dlp --flat-playlist` (lists videos without
/// downloading anything, including captions -- a separate, much
/// cheaper network operation than `fetch_captions`). Fills the real gap
/// found 2026-08-09 reingesting a 23-person batch: `--video` is
/// explicitly hand-picked-only (see its CLI doc comment), so before
/// this there was no way to point `ingest-persona` at a channel URL
/// directly -- callers had to shell out to `yt-dlp --flat-playlist`
/// themselves and translate the output into a wall of `--video` flags.
///
/// `%(id)s|%(title)s` is the same `id|title` shape `--video` already
/// expects, so enumerated targets merge into the same `Vec<VideoTarget>`
/// pipeline `fetch_captions`/`ingest_videos` already handle -- no new
/// code path downstream, just a new way to populate the list.
///
/// A real video ID can start with `-` (found 2026-08-09: bashbunni's
/// most recent upload at the time was `-EMKMPxJrWY`) -- parsing here
/// splits on the first `|`, so a leading `-` in the id half is just
/// data, never mistaken for a flag the way it would be if this were
/// naively re-serialized into space-separated CLI args downstream (see
/// `--video`'s `allow_hyphen_values` fix in main.rs for the CLI-level
/// half of this same bug class).
pub fn enumerate_channel_videos(
    yt_dlp_bin: &Path,
    channel_url: &str,
    max: usize,
) -> Result<Vec<VideoTarget>, String> {
    let output = Command::new(yt_dlp_bin)
        .args(["--flat-playlist", "--playlist-end"])
        .arg(max.to_string())
        .args(["--print", "%(id)s|%(title)s"])
        .arg(channel_url)
        .output()
        .map_err(|e| format!("failed to run {}: {e}", yt_dlp_bin.display()))?;

    if !output.status.success() {
        return Err(format!(
            "yt-dlp channel enumeration failed for {channel_url:?}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut targets = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line.split_once('|') {
            Some((id, title)) => targets.push(VideoTarget {
                id: id.to_string(),
                title: title.to_string(),
            }),
            // A title itself could theoretically contain no pipe issue,
            // but a line with no pipe at all means yt-dlp's own output
            // format didn't match what was requested -- skip rather
            // than fail the whole channel over one malformed line.
            None => continue,
        }
    }
    Ok(targets)
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
    let mut dedup_items = Vec::with_capacity(titles.len());
    for title in titles {
        let text = fetch_wikipedia(title)?;
        let md = build_wikipedia_raw_md(person, slug, title, &text, ingested_date);
        let file_name = format!("wikipedia-{}.md", slugify(title));
        let file_path = person_dir.join(&file_name);
        atomic::write(&file_path, &md).map_err(|e| e.to_string())?;
        written.push(file_path);
        dedup_items.push((slugify(title), text, false));
    }

    if let Ok(results) = run_squishi_dedup_batch(&dedup_items) {
        for (id, dedup_json) in results {
            let sidecar_path = person_dir.join(format!("wikipedia-{id}.dedup.json"));
            let _ = atomic::write(&sidecar_path, &dedup_json);
        }
    }
    Ok(written)
}

/// Strips a raw file's YAML frontmatter (`---\n...\n---\n`), returning
/// just the body — shared by the resumability path in `ingest_videos`
/// and by `dedup_raw_files`, both of which need the same plain-text
/// content squishi actually dedupes, not the frontmatter around it.
/// Falls back to the whole content unchanged if it doesn't look like it
/// has frontmatter (defensive, not expected in practice — every raw
/// file this crate writes has one).
fn strip_frontmatter(content: &str) -> String {
    match content.split_once("---\n") {
        Some((_, rest)) => match rest.split_once("---\n") {
            Some((_, body)) => body.trim().to_string(),
            None => content.to_string(),
        },
        None => content.to_string(),
    }
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

/// Fetches a URL's agent-readable text via the locally installed
/// `agent-browser` CLI (`agent-browser read <url>`) rather than a plain
/// HTTP GET. Real browser rendering handles JS-heavy pages a static
/// fetch can't — the exact failure found 2026-08-07 on Roy Sugarman's
/// own `/media` page, where a static fetch returned nothing because the
/// embedded video list only exists after JS runs; `agent-browser` (a
/// real Chrome/Chromium session) was used by hand to work around it that
/// session. This wires the same tool into the pipeline itself so a
/// personal-site/blog source doesn't need a human to notice and
/// intervene. `agent-browser read` already returns cleaned,
/// markdown-ish text (HTML boilerplate stripped) — no separate
/// HTML-to-text step needed here.
pub fn fetch_website(url: &str) -> Result<String, String> {
    let output = Command::new("agent-browser")
        .args(["read", url])
        .output()
        .map_err(|e| format!("failed to run agent-browser (is it installed on PATH?): {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "agent-browser read failed for {url}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.len() < 200 {
        return Err(format!(
            "website text too short for {url} ({} bytes) — likely a fetch failure, not real content",
            text.len()
        ));
    }
    Ok(text)
}

/// Same frontmatter shape family as `build_wikipedia_raw_md`, for a
/// personal-site/blog page fetched via `agent-browser`.
pub fn build_website_raw_md(
    person: &str,
    slug: &str,
    url: &str,
    text: &str,
    ingested_date: &str,
) -> String {
    let word_count = text.split_whitespace().count();
    format!(
        "---\n\
         source: {url}\n\
         person: {person}\n\
         person_slug: {slug}\n\
         type: website\n\
         ingested: {ingested_date}\n\
         word_count: {word_count}\n\
         ---\n\n\
         {text}\n"
    )
}

/// Fetch+write raw website/blog files, one per URL, into
/// `<bank>/raw/<slug>/` — same tier, resilience posture (one bad URL
/// doesn't abort the batch), and best-effort squishi-sidecar treatment
/// as `ingest_wikipedia_pages`. `restore_punctuation` is always `false`
/// here, same reasoning as the Wikipedia/git-commit sources: real prose
/// already has real punctuation by construction, unlike YouTube
/// auto-captions.
pub fn ingest_websites(
    raw_dir: &Path,
    person: &str,
    slug: &str,
    urls: &[String],
    ingested_date: &str,
) -> Result<(Vec<PathBuf>, FetchFailures), String> {
    let person_dir = raw_dir.join(slug);
    std::fs::create_dir_all(&person_dir).map_err(|e| e.to_string())?;

    let mut written = Vec::with_capacity(urls.len());
    let mut dedup_items = Vec::with_capacity(urls.len());
    let mut failures: FetchFailures = Vec::new();
    for url in urls {
        let text = match fetch_website(url) {
            Ok(t) => t,
            Err(e) => {
                failures.push((url.clone(), e));
                continue;
            }
        };
        let md = build_website_raw_md(person, slug, url, &text, ingested_date);
        let id = slugify(url);
        let file_path = person_dir.join(format!("website-{id}.md"));
        if let Err(e) = atomic::write(&file_path, &md) {
            failures.push((url.clone(), e.to_string()));
            continue;
        }
        written.push(file_path);
        dedup_items.push((format!("website-{id}"), text, false));
    }

    if written.is_empty() {
        let summary = failures
            .iter()
            .map(|(id, e)| format!("{id}: {e}"))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!("every website failed to fetch: {summary}"));
    }

    if let Ok(results) = run_squishi_dedup_batch(&dedup_items) {
        for (id, dedup_json) in results {
            let sidecar_path = person_dir.join(format!("{id}.dedup.json"));
            let _ = atomic::write(&sidecar_path, &dedup_json);
        }
    }
    Ok((written, failures))
}

/// Extracts the `owner/repo` slug from a GitHub URL (with or without a
/// trailing `.git`/`/`), for `gh api` calls — those address repos by
/// slug, not clone URL. Returns `None` for a non-GitHub host: issue
/// search is a GitHub-specific API, unlike `fetch_git_commits` (plain
/// `git log`, host-agnostic).
pub fn parse_github_slug(repo_url: &str) -> Option<String> {
    let trimmed = repo_url.trim_end_matches('/').trim_end_matches(".git");
    let after_host = trimmed.split("github.com/").nth(1)?;
    let mut parts = after_host.splitn(3, '/');
    let owner = parts.next()?;
    let repo = parts.next()?;
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    Some(format!("{owner}/{repo}"))
}

/// Fetches issues/PRs authored by `github_users` (real GitHub logins —
/// a different identifier than `fetch_git_commits`'s email-matched
/// `--git-author`, since GitHub's issue-search API matches by account,
/// not commit-trailer email) in `repo_slug`, via `gh api search/issues`
/// — issue/PR data lives only on GitHub's side, not in `git log`, so a
/// clone (as `fetch_git_commits` does) can't reach it. Requires `gh`
/// authenticated on PATH (already a documented environment tool).
pub fn fetch_git_issues(repo_slug: &str, github_users: &[String]) -> Result<String, String> {
    if github_users.is_empty() {
        return Err("fetch_git_issues requires at least one --github-user".to_string());
    }

    let mut entries: Vec<(String, String)> = Vec::new();
    for user in github_users {
        let query = format!("repo:{repo_slug} author:{user}");
        let output = Command::new("gh")
            .args([
                "api",
                "search/issues",
                "-X",
                "GET",
                "-f",
                &format!("q={query}"),
                "--jq",
                r#".items[] | [.created_at, .title, (.body // "")] | @tsv"#,
            ])
            .output()
            .map_err(|e| format!("failed to run gh: {e}"))?;

        if !output.status.success() {
            return Err(format!(
                "gh api search/issues failed for {repo_slug} author:{user}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }

        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let mut fields = line.splitn(3, '\t');
            let (Some(date), Some(title), Some(body)) =
                (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            entries.push((date.to_string(), format!("{title}\n{body}")));
        }
    }

    if entries.is_empty() {
        return Err(format!(
            "no issues/PRs found for github user(s) {github_users:?} in {repo_slug}"
        ));
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries
        .into_iter()
        .map(|(date, body)| format!("On {date}: {body}"))
        .collect::<Vec<_>>()
        .join("\n\n"))
}

/// Same frontmatter shape family as `build_git_commits_raw_md`, for an
/// issue/PR-discussion corpus.
pub fn build_git_issues_raw_md(
    person: &str,
    slug: &str,
    repo_slug: &str,
    text: &str,
    ingested_date: &str,
) -> String {
    let word_count = text.split_whitespace().count();
    format!(
        "---\n\
         source: https://github.com/{repo_slug}/issues\n\
         person: {person}\n\
         person_slug: {slug}\n\
         type: git-issues\n\
         ingested: {ingested_date}\n\
         word_count: {word_count}\n\
         ---\n\n\
         # Issues/PRs from {repo_slug}\n\n\
         {text}\n"
    )
}

/// Fetch+write one raw file of issue/PR text for a single repo — same
/// tier and best-effort squishi-sidecar treatment as `ingest_git_commits`.
pub fn ingest_git_issues(
    raw_dir: &Path,
    person: &str,
    slug: &str,
    repo_url: &str,
    github_users: &[String],
    ingested_date: &str,
) -> Result<Vec<PathBuf>, String> {
    let repo_slug = parse_github_slug(repo_url).ok_or_else(|| {
        format!(
            "{repo_url:?} doesn't look like a github.com repo URL — issue search is GitHub-specific"
        )
    })?;

    let person_dir = raw_dir.join(slug);
    std::fs::create_dir_all(&person_dir).map_err(|e| e.to_string())?;

    let text = fetch_git_issues(&repo_slug, github_users)?;
    let md = build_git_issues_raw_md(person, slug, &repo_slug, &text, ingested_date);
    let file_name = format!("git-issues-{}.md", slugify(&repo_slug));
    let file_path = person_dir.join(&file_name);
    atomic::write(&file_path, &md).map_err(|e| e.to_string())?;

    let dedup_id = slugify(&repo_slug);
    let dedup_items = vec![(dedup_id.clone(), text, false)];
    if let Ok(results) = run_squishi_dedup_batch(&dedup_items) {
        for (_id, dedup_json) in results {
            let sidecar_path = person_dir.join(format!("git-issues-{dedup_id}.dedup.json"));
            let _ = atomic::write(&sidecar_path, &dedup_json);
        }
    }

    Ok(vec![file_path])
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

    let dedup_items = vec![(slugify(repo_name), text, false)];
    if let Ok(results) = run_squishi_dedup_batch(&dedup_items) {
        for (id, dedup_json) in results {
            let sidecar_path = person_dir.join(format!("git-{id}.dedup.json"));
            let _ = atomic::write(&sidecar_path, &dedup_json);
        }
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

/// Runs squishi's caller-asserted plain-text dedup on a whole batch of
/// `(id, text)` pairs in ONE squishi process via `--batch`, instead of
/// one process per item. Real fix for a real bottleneck (found
/// 2026-08-08 reingesting 19 real persona videos one-subprocess-per-
/// video): each call reloaded the ~562MB punctuation-restoration model
/// from scratch, turning a batch job into hours of redundant model
/// loading. `--batch` loads the model once, reuses it across every
/// item in the array. Returns each item's own JSON result (same shape
/// a single-item call would have produced, still one document per id)
/// so callers keep writing one `.dedup.json` sidecar per source,
/// unchanged. Best-effort at the *batch* level — matching the prior
/// per-item best-effort posture, squishi being unavailable means no
/// sidecars for this whole batch, not a partial one.
///
/// Each item's `restore_punctuation` flag is a real, source-aware
/// assertion, not left to squishi's own content-density guess: only
/// YouTube-caption sources ever set it `true` (`ingest_videos`,
/// `dedup_raw_files` inferring from a `yt-` filename prefix) —
/// Wikipedia and git-commit-message sources always set it `false`,
/// since they already have real punctuation by construction and never
/// need the ~562MB model invoked, belt-and-suspenders alongside
/// squishi's own heuristic rather than relying on it alone.
fn run_squishi_dedup_batch(
    items: &[(String, String, bool)],
) -> Result<Vec<(String, String)>, String> {
    use std::io::Write;

    if items.is_empty() {
        return Ok(Vec::new());
    }

    let input = Value::Array(
        items
            .iter()
            .map(|(id, text, restore_punctuation)| {
                serde_json::json!({ "id": id, "text": text, "restore_punctuation": restore_punctuation })
            })
            .collect(),
    );
    let input_str = serde_json::to_string(&input).map_err(|e| e.to_string())?;

    let mut child = Command::new("squishi")
        .args(["--batch", "--force-kind", "plain-text"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn squishi: {e}"))?;

    child
        .stdin
        .take()
        .ok_or_else(|| "squishi: no stdin handle".to_string())?
        .write_all(input_str.as_bytes())
        .map_err(|e| e.to_string())?;

    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "squishi --batch failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    let results: Value = serde_json::from_str(&stdout).map_err(|e| e.to_string())?;
    let array = results
        .as_array()
        .ok_or_else(|| "squishi --batch: expected a JSON array".to_string())?;

    array
        .iter()
        .map(|item| {
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "squishi --batch: result item missing id".to_string())?
                .to_string();
            let json = serde_json::to_string(item).map_err(|e| e.to_string())?;
            Ok((id, json))
        })
        .collect()
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
/// Resilient per-video: one video's fetch failing (a transient network
/// blip, a rate limit) does NOT abort the rest — it's logged into the
/// returned failure list and the loop continues. Real cost of the
/// opposite (found 2026-08-08): a 55-video job that fetched 54
/// successfully over ~2 hours failed the WHOLE run — zero sidecars,
/// nothing staged — because video 55 hit a rate limit and the old
/// fail-fast `?` discarded everything already done. Only returns `Err`
/// outright when literally every video failed (nothing to work with at
/// all).
///
/// Also writes a `yt-<id>.dedup.json` sidecar per successfully-fetched
/// video — squishi's dedup/shape/summary output, run best-effort:
/// squishi being unavailable (not installed, not on PATH) never fails
/// the whole ingestion, it just means no sidecars for this run (they
/// can be produced later via `dedup_raw_files`, without re-fetching).
/// The sub-agent completing the handover reads a sidecar when present
/// as a traceable pointer into the raw transcript (which sentences are
/// narrative-shaped, which are the most central/summary-worthy) rather
/// than reading the full raw transcript cold every time.
/// `(id, error message)` pairs for videos that failed to fetch —
/// resilience means these are reported, not silently dropped or fatal.
pub type FetchFailures = Vec<(String, String)>;

pub fn ingest_videos(
    data_root: &Path,
    raw_dir: &Path,
    person: &str,
    slug: &str,
    videos: &[VideoTarget],
    ingested_date: &str,
) -> Result<(Vec<PathBuf>, FetchFailures), String> {
    let yt_dlp_bin = ensure_yt_dlp(data_root)?;

    let person_dir = raw_dir.join(slug);
    std::fs::create_dir_all(&person_dir).map_err(|e| e.to_string())?;

    let mut written = Vec::with_capacity(videos.len());
    let mut dedup_items = Vec::with_capacity(videos.len());
    let mut failures: FetchFailures = Vec::new();
    for video in videos {
        let file_name = format!("yt-{}.md", video.id);
        let file_path = person_dir.join(&file_name);
        let sidecar_path = person_dir.join(format!("yt-{}.dedup.json", video.id));

        // Resumable, matching ingest_youtube.py's own real behavior
        // (skip a video whose output already exists) -- lost when this
        // was first ported to Rust, restored here after confirming the
        // Python script had it and this one didn't (2026-08-08). A
        // video already fetched is never re-fetched; its dedup only
        // gets (re-)run if no sidecar exists yet either.
        if file_path.exists() {
            written.push(file_path.clone());
            if !sidecar_path.exists()
                && let Ok(existing) = std::fs::read_to_string(&file_path)
            {
                dedup_items.push((video.id.clone(), strip_frontmatter(&existing), true));
            }
            continue;
        }

        let text = match fetch_captions(&yt_dlp_bin, &video.id, &person_dir) {
            Ok(t) => t,
            Err(e) => {
                failures.push((video.id.clone(), e));
                continue;
            }
        };
        let md = build_raw_transcript_md(person, slug, video, &text, ingested_date);
        if let Err(e) = atomic::write(&file_path, &md) {
            failures.push((video.id.clone(), e.to_string()));
            continue;
        }
        written.push(file_path);
        dedup_items.push((video.id.clone(), text, true));
    }

    if written.is_empty() {
        let summary = failures
            .iter()
            .map(|(id, e)| format!("{id}: {e}"))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!("every video failed to fetch: {summary}"));
    }

    if let Ok(results) = run_squishi_dedup_batch(&dedup_items) {
        for (id, dedup_json) in results {
            let sidecar_path = person_dir.join(format!("yt-{id}.dedup.json"));
            let _ = atomic::write(&sidecar_path, &dedup_json);
        }
    }
    Ok((written, failures))
}

/// Standalone transform pass: dedupe+punctuate whatever raw `.md` files
/// already exist for `slug`, independent of fetching — the answer to
/// "fetch in batches, then transform separately," and the same real
/// recovery this session ran by hand after `ingest_videos` partial
/// failures. Skips any raw file that already has a `.dedup.json`
/// sidecar (idempotent — re-running after a partial `ingest_videos`
/// only processes what's actually new), unless `force` is set. Returns
/// how many files were processed.
pub fn dedup_raw_files(raw_dir: &Path, slug: &str, force: bool) -> Result<usize, String> {
    let person_dir = raw_dir.join(slug);
    if !person_dir.is_dir() {
        return Err(format!(
            "no raw directory for slug {slug:?}: {}",
            person_dir.display()
        ));
    }

    let mut items = Vec::new();
    let entries = std::fs::read_dir(&person_dir).map_err(|e| e.to_string())?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".md") {
            continue;
        }
        let id = name.trim_end_matches(".md");
        let sidecar_path = person_dir.join(format!("{id}.dedup.json"));
        if sidecar_path.exists() && !force {
            continue;
        }
        let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        // Only a real YouTube caption source ever needs punctuation
        // restoration -- inferred from this crate's own filename
        // convention (yt-<id>.md vs wikipedia-<id>.md/git-<id>.md),
        // since dedup_raw_files works from whatever's already on disk
        // and has no other signal for which source type wrote a file.
        let restore_punctuation = id.starts_with("yt-");
        items.push((
            id.to_string(),
            strip_frontmatter(&content),
            restore_punctuation,
        ));
    }

    if items.is_empty() {
        return Ok(0);
    }

    let results = run_squishi_dedup_batch(&items)?;
    for (id, dedup_json) in &results {
        let sidecar_path = person_dir.join(format!("{id}.dedup.json"));
        atomic::write(&sidecar_path, dedup_json).map_err(|e| e.to_string())?;
    }
    Ok(results.len())
}

/// One ranked cluster from `cluster_raw_files` -- a sentence that
/// survived squishi's pooled-corpus dedup pass, plus how many other
/// sentences (across every source file) collapsed into it. High
/// `cluster_size` on a `concept`-shaped sentence is a recurring topic;
/// high `cluster_size` on a `narrative`-shaped one is a recurring story
/// -- the mechanical proxy for "what does this person keep coming back
/// to," no LLM judgment involved.
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

/// Pools every raw source file's already-cleaned `.dedup.json` sidecar
/// text for `slug` into one document and runs squishi's dedup ONE more
/// time over the pool. This is the whole mechanism: squishi's existing
/// greedy single-pass cosine-similarity clustering, run cross-file
/// instead of per-file, naturally produces cluster membership as a side
/// effect -- each `Drop` records which surviving sentence it collapsed
/// into (`kept_index`), so counting drops per kept sentence's index
/// gives a recurrence score for free. No new clustering algorithm.
///
/// Requires `dedup_raw_files` to have already run for this slug (reads
/// `.dedup.json` sidecars, not raw `.md` files, so each source's own
/// punctuation-restoration decision is respected instead of re-guessed
/// here). `restore_punctuation` is always `false` for the pooled call --
/// every sidecar's `compressed` text is already real, clean prose by
/// construction; re-running restoration on already-punctuated pooled
/// text is exactly the wrong input shape for that heuristic.
///
/// Writes `<raw_dir>/<slug>/cluster-summary.json` (`{ topics, stories }`,
/// each sorted by `cluster_size` descending) and returns its path.
pub fn cluster_raw_files(raw_dir: &Path, slug: &str) -> Result<PathBuf, String> {
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

    let mut pooled = String::new();
    for path in &sidecar_paths {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let parsed: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let Some(compressed) = parsed.get("compressed").and_then(|v| v.as_str()) else {
            continue;
        };
        if !pooled.is_empty() {
            pooled.push_str("\n\n");
        }
        pooled.push_str(compressed);
    }

    if pooled.trim().is_empty() {
        return Err(format!(
            "every sidecar for slug {slug:?} was empty or malformed -- nothing to cluster"
        ));
    }

    let results = run_squishi_dedup_batch(&[("pooled".to_string(), pooled, false)])?;
    let (_, result_json) = results
        .into_iter()
        .next()
        .ok_or_else(|| "squishi returned no result for the pooled corpus".to_string())?;
    let parsed: Value = serde_json::from_str(&result_json).map_err(|e| e.to_string())?;

    let kept = parsed
        .get("kept")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "squishi result missing `kept` array".to_string())?;
    let drops = parsed
        .get("drops")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "squishi result missing `drops` array".to_string())?;

    let mut cluster_sizes: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for drop in drops {
        if let Some(kept_index) = drop.get("kept_index").and_then(|v| v.as_u64()) {
            *cluster_sizes.entry(kept_index).or_insert(1) += 1;
        }
    }

    let mut topics = Vec::new();
    let mut stories = Vec::new();
    for entry in kept {
        let (Some(index), Some(text), Some(shape)) = (
            entry.get("index").and_then(|v| v.as_u64()),
            entry.get("text").and_then(|v| v.as_str()),
            entry.get("shape").and_then(|v| v.as_str()),
        ) else {
            continue;
        };
        let cluster_size = cluster_sizes.get(&index).copied().unwrap_or(1);
        let cluster = RecurrenceCluster {
            text: text.to_string(),
            cluster_size,
        };
        match shape {
            "narrative" => stories.push(cluster),
            _ => topics.push(cluster),
        }
    }
    topics.sort_by_key(|c| std::cmp::Reverse(c.cluster_size));
    stories.sort_by_key(|c| std::cmp::Reverse(c.cluster_size));

    let topics_json: Vec<Value> = topics.iter().map(Value::from).collect();
    let stories_json: Vec<Value> = stories.iter().map(Value::from).collect();
    let summary =
        serde_json::json!({ "slug": slug, "topics": topics_json, "stories": stories_json });
    let summary_path = person_dir.join("cluster-summary.json");
    atomic::write(
        &summary_path,
        &serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    Ok(summary_path)
}

/// Same threshold `advisory/tools/dedupe_semantic.py` calibrated on real
/// persona corpora (0.8 = high precision) -- reused as-is, not
/// recalibrated from nothing. See `crate::concepts` for the algorithm.
const CONCEPTS_SPLIT_THRESHOLD: f32 = 0.8;

/// Sentence-level concept distillation over whatever raw `.md` files
/// already exist for `slug` -- the trm-native port of `advisory/tools/
/// dedupe_semantic.py`, finally wired into the persona pipeline
/// (`crate::concepts::split` already existed for `handover.rs`'s
/// candidate-splitting but was never called from here). Complementary
/// to `dedup_raw_files`, not a replacement: that produces a coarse,
/// near-full-length punctuation-restored sidecar per file; this
/// produces a much smaller per-file sidecar of unique-concept
/// sentences, meant as a distilled synthesis input alongside the raw
/// transcript -- not judged sentence-by-sentence as a handover
/// candidate list (see `HandoverKind::PersonaBuild`'s doc comment for
/// why persona synthesis stays a holistic read, not a per-concept
/// judgment). Skips a file that already has a `.concepts.json` sidecar
/// unless `force` is set. Returns how many files were processed.
pub fn extract_concepts_files(raw_dir: &Path, slug: &str, force: bool) -> Result<usize, String> {
    let person_dir = raw_dir.join(slug);
    if !person_dir.is_dir() {
        return Err(format!(
            "no raw directory for slug {slug:?}: {}",
            person_dir.display()
        ));
    }

    let mut targets = Vec::new();
    let entries = std::fs::read_dir(&person_dir).map_err(|e| e.to_string())?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".md") {
            continue;
        }
        let id = name.trim_end_matches(".md");
        let sidecar_path = person_dir.join(format!("{id}.concepts.json"));
        if sidecar_path.exists() && !force {
            continue;
        }
        let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        targets.push((id.to_string(), sidecar_path, strip_frontmatter(&content)));
    }

    if targets.is_empty() {
        return Ok(0);
    }

    // Embedder is loaded once and reused across every file in this
    // slug -- model load is the expensive part, not the per-file split.
    let mut embedder = crate::embeddings::Embedder::new(crate::bank::data_root().join("models"))?;

    let mut processed = 0;
    for (id, sidecar_path, body) in targets {
        let concepts = crate::concepts::split(&body, &mut embedder, CONCEPTS_SPLIT_THRESHOLD)?;
        let json = serde_json::json!({ "id": id, "concepts": concepts });
        let text = serde_json::to_string_pretty(&json).map_err(|e| e.to_string())?;
        atomic::write(&sidecar_path, &text).map_err(|e| e.to_string())?;
        processed += 1;
    }
    Ok(processed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- read_transcript_path: real fixtures ---

    #[test]
    fn plain_jsonl_source_is_returned_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("sess.jsonl");
        std::fs::write(&source, "{}\n").unwrap();
        let work_dir = tmp.path().join("work");

        let result = read_transcript_path(&source, &work_dir).unwrap();
        assert_eq!(result, source);
        assert!(
            !work_dir.exists(),
            "work_dir should be untouched for a plain source"
        );
    }

    #[test]
    fn gzip_source_decompresses_to_a_real_temp_file_in_work_dir() {
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let original = "line one\nline two\nline three\n".repeat(20);
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(original.as_bytes()).unwrap();
        let compressed = encoder.finish().unwrap();

        let source = tmp.path().join("sess-1.jsonl.gz");
        std::fs::write(&source, &compressed).unwrap();
        let work_dir = tmp.path().join("work");

        let result = read_transcript_path(&source, &work_dir).unwrap();
        assert_ne!(result, source, "a gz source must produce a fresh temp path");
        assert!(result.starts_with(&work_dir));
        assert_eq!(std::fs::read_to_string(&result).unwrap(), original);
    }

    // --- build_session_raw_md: pure ---

    #[test]
    fn build_session_raw_md_matches_the_frontmatter_shape() {
        let digest = ingest::SessionDigest {
            content: "USER: a real question\n\nASSISTANT: a real answer".to_string(),
            session_id: Some("sess-abc123".to_string()),
            cwd: Some("/home/alvin/Code/_labs/squishi".to_string()),
            turn_count: 2,
            total_lines: 4,
        };
        let md = build_session_raw_md(
            "Alvin Tolentino",
            "alvin",
            &digest,
            Path::new("/home/alvin/.claude/projects/x/sess-abc123.jsonl"),
            "2026-08-09",
        );
        assert!(md.starts_with("---\nsource: /home/alvin/.claude/projects/x/sess-abc123.jsonl\n"));
        assert!(md.contains("person: Alvin Tolentino\n"));
        assert!(md.contains("person_slug: alvin\n"));
        assert!(md.contains("type: claude-code-session\n"));
        assert!(md.contains("session_id: sess-abc123\n"));
        assert!(md.contains("session_cwd: /home/alvin/Code/_labs/squishi\n"));
        assert!(md.ends_with("USER: a real question\n\nASSISTANT: a real answer\n"));
    }

    #[test]
    fn build_session_raw_md_falls_back_to_unknown_for_missing_meta() {
        let digest = ingest::SessionDigest {
            content: "USER: content with no meta".to_string(),
            session_id: None,
            cwd: None,
            turn_count: 1,
            total_lines: 1,
        };
        let md = build_session_raw_md(
            "Alvin Tolentino",
            "alvin",
            &digest,
            Path::new("/some/path.jsonl"),
            "2026-08-09",
        );
        assert!(md.contains("session_id: unknown\n"));
        assert!(md.contains("session_cwd: unknown\n"));
    }

    // --- ingest_sessions: real squishi subprocess ---

    /// Doesn't need squishi truly installed to be meaningful: whether the
    /// binary is missing (spawn itself fails) or present-but-fed-a-
    /// nonexistent-path (squishi's own file read fails), both surface as
    /// `Err` from `run_squishi_session_digest` -- exactly what this test
    /// asserts, so it runs unconditionally rather than joining the
    /// `#[ignore]`d real-squishi tests below.
    #[test]
    fn ingest_sessions_errors_when_every_path_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = vec![
            tmp.path().join("does-not-exist-1.jsonl"),
            tmp.path().join("does-not-exist-2.jsonl"),
        ];
        let result = ingest_sessions(tmp.path(), "Alvin Tolentino", "alvin", &paths, "2026-08-09");
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // requires `squishi` installed on PATH (real subprocess)
    fn ingest_sessions_digests_a_real_plain_and_gzip_transcript() {
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let raw_dir = tmp.path().join("raw");

        let user_line = |session_id: &str| {
            format!(
                r#"{{"type":"user","sessionId":"{session_id}","cwd":"/repo","timestamp":"t1","message":{{"role":"user","content":[{{"type":"text","text":"a real question worth digesting"}}]}}}}"#
            )
        };

        let plain_path = tmp.path().join("sess-plain.jsonl");
        std::fs::write(&plain_path, user_line("sess-plain")).unwrap();

        let gz_path = tmp.path().join("sess-gz.jsonl.gz");
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(user_line("sess-gz").as_bytes()).unwrap();
        std::fs::write(&gz_path, encoder.finish().unwrap()).unwrap();

        let (written, failures) = ingest_sessions(
            &raw_dir,
            "Alvin Tolentino",
            "alvin",
            &[plain_path, gz_path],
            "2026-08-09",
        )
        .expect("both sources should succeed");

        assert!(
            failures.is_empty(),
            "expected no failures, got: {failures:?}"
        );
        assert_eq!(written.len(), 2);
        assert!(written.iter().any(|p| p.ends_with("session-sess-plain.md")));
        assert!(written.iter().any(|p| p.ends_with("session-sess-gz.md")));

        let plain_content = std::fs::read_to_string(&written[0]).unwrap();
        assert!(plain_content.contains("type: claude-code-session\n"));

        // The gz source's decompressed temp file must not linger in
        // raw_dir/alvin/ once digesting is done.
        let leftovers: Vec<_> = std::fs::read_dir(raw_dir.join("alvin"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "expected no leftover temp files");
    }

    #[test]
    #[ignore] // requires `squishi` installed on PATH (real subprocess)
    fn ingest_sessions_is_resilient_to_one_missing_path() {
        let tmp = tempfile::tempdir().unwrap();
        let raw_dir = tmp.path().join("raw");

        let real_path = tmp.path().join("sess-real.jsonl");
        std::fs::write(
            &real_path,
            r#"{"type":"user","sessionId":"sess-real","cwd":"/repo","timestamp":"t1","message":{"role":"user","content":[{"type":"text","text":"a real question"}]}}"#,
        )
        .unwrap();
        let missing_path = tmp.path().join("does-not-exist.jsonl");

        let (written, failures) = ingest_sessions(
            &raw_dir,
            "Alvin Tolentino",
            "alvin",
            &[real_path, missing_path],
            "2026-08-09",
        )
        .expect("partial success should be Ok, not Err");

        assert_eq!(written.len(), 1);
        assert_eq!(failures.len(), 1);
    }

    #[test]
    fn corrupt_gzip_source_returns_an_error_not_a_panic() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("corrupt.jsonl.gz");
        std::fs::write(&source, b"not actually gzip data").unwrap();
        let work_dir = tmp.path().join("work");

        let result = read_transcript_path(&source, &work_dir);
        assert!(result.is_err());
    }

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

    // --- enumerate_channel_videos: pure parsing + fake-binary tests ---

    /// A fake yt-dlp that just echoes fixed lines to stdout, ignoring its
    /// arguments -- proves the parsing logic without touching the network.
    fn seed_fake_yt_dlp_printing(dir: &Path, lines: &[&str]) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let stub = dir.join("yt-dlp");
        let script = format!(
            "#!/bin/sh\n{}\n",
            lines
                .iter()
                .map(|l| format!("echo '{l}'"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        std::fs::write(&stub, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        stub
    }

    #[test]
    fn enumerate_channel_videos_parses_id_title_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = seed_fake_yt_dlp_printing(
            tmp.path(),
            &["abc123|First Video", "def456|Second Video: A Subtitle"],
        );
        let result = enumerate_channel_videos(&bin, "https://youtube.com/@fake/videos", 50);
        assert_eq!(
            result,
            Ok(vec![
                VideoTarget {
                    id: "abc123".to_string(),
                    title: "First Video".to_string(),
                },
                VideoTarget {
                    id: "def456".to_string(),
                    title: "Second Video: A Subtitle".to_string(),
                },
            ])
        );
    }

    #[test]
    fn enumerate_channel_videos_handles_a_dash_prefixed_id() {
        // Real bug found 2026-08-09: bashbunni's most recent upload at the
        // time had the video ID "-EMKMPxJrWY". Splitting on the first '|'
        // must keep that leading '-' as part of the id, not choke on it.
        let tmp = tempfile::tempdir().unwrap();
        let bin = seed_fake_yt_dlp_printing(tmp.path(), &["-EMKMPxJrWY|dash prefixed id"]);
        let result = enumerate_channel_videos(&bin, "https://youtube.com/@fake/videos", 50);
        assert_eq!(
            result,
            Ok(vec![VideoTarget {
                id: "-EMKMPxJrWY".to_string(),
                title: "dash prefixed id".to_string(),
            }])
        );
    }

    #[test]
    fn enumerate_channel_videos_skips_blank_and_malformed_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let bin =
            seed_fake_yt_dlp_printing(tmp.path(), &["", "no-pipe-here", "real123|A Real Title"]);
        let result = enumerate_channel_videos(&bin, "https://youtube.com/@fake/videos", 50);
        assert_eq!(
            result,
            Ok(vec![VideoTarget {
                id: "real123".to_string(),
                title: "A Real Title".to_string(),
            }])
        );
    }

    #[test]
    fn enumerate_channel_videos_on_empty_channel_returns_empty_ok_not_error() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = seed_fake_yt_dlp_printing(tmp.path(), &[]);
        let result = enumerate_channel_videos(&bin, "https://youtube.com/@fake/videos", 50);
        assert_eq!(result, Ok(Vec::new()));
    }

    #[test]
    fn enumerate_channel_videos_surfaces_a_real_error_when_yt_dlp_fails() {
        let tmp = tempfile::tempdir().unwrap();
        seed_fake_failing_yt_dlp(tmp.path());
        let bin = tmp.path().join("bin").join("yt-dlp");
        let result = enumerate_channel_videos(&bin, "https://youtube.com/@fake/videos", 50);
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // requires network + real yt-dlp on PATH
    fn enumerate_channel_videos_real_channel_respects_the_max_cap() {
        let data_root = tempfile::tempdir().unwrap();
        let yt_dlp = ensure_yt_dlp(data_root.path()).unwrap();
        // A real, stable, high-upload-volume channel -- if this ever
        // becomes flaky due to the channel changing, swap the URL, but
        // the point of this test is exercising the real network path,
        // not the specific channel.
        let result =
            enumerate_channel_videos(&yt_dlp, "https://www.youtube.com/@ippsec", 3).unwrap();
        assert_eq!(result.len(), 3, "expected exactly 3 videos (the cap)");
        for v in &result {
            assert!(!v.id.is_empty());
            assert!(!v.title.is_empty());
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
    fn ingest_videos_skips_one_failure_and_still_returns_ok_with_the_rest() {
        // Real resilience test: yt-dlp always fails here, but one of the
        // two videos already has its raw file on disk -- resumability
        // means that one is never even attempted, so it "succeeds"
        // despite a guaranteed-failing yt-dlp, and the overall call
        // returns Ok (partial success), not Err, because not everything
        // failed. This is the exact class of bug found 2026-08-08: a
        // single bad item must not discard everything else.
        let tmp = tempfile::tempdir().unwrap();
        seed_fake_failing_yt_dlp(tmp.path());
        let raw_dir = tmp.path().join("raw");
        let person_dir = raw_dir.join("test-person");
        std::fs::create_dir_all(&person_dir).unwrap();
        let already_fetched_path = person_dir.join("yt-already-fetched.md");
        std::fs::write(
            &already_fetched_path,
            "---\nsource: x\n---\n\n# Title\n\nreal cached content",
        )
        .unwrap();

        let videos = vec![
            VideoTarget {
                id: "already-fetched".to_string(),
                title: "already on disk".to_string(),
            },
            VideoTarget {
                id: "will-fail".to_string(),
                title: "yt-dlp will fail for this one".to_string(),
            },
        ];
        let result = ingest_videos(
            tmp.path(),
            &raw_dir,
            "Test Person",
            "test-person",
            &videos,
            "2026-08-07",
        );
        let (written, failures) = result.expect("partial success should be Ok, not Err");
        assert_eq!(written, vec![already_fetched_path]);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "will-fail");
    }

    // --- dedup_raw_files: pure/fast paths, no real squishi call ---

    #[test]
    fn dedup_raw_files_errors_for_a_missing_slug_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let result = dedup_raw_files(tmp.path(), "no-such-slug", false);
        assert!(result.is_err());
    }

    #[test]
    fn dedup_raw_files_on_an_empty_directory_returns_zero_without_spawning_squishi() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("empty-slug")).unwrap();
        let result = dedup_raw_files(tmp.path(), "empty-slug", false);
        assert_eq!(result, Ok(0));
    }

    #[test]
    fn dedup_raw_files_skips_a_file_that_already_has_a_sidecar_unless_forced() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("has-sidecar");
        std::fs::create_dir_all(&person_dir).unwrap();
        std::fs::write(
            person_dir.join("yt-abc.md"),
            "---\nsource: x\n---\n\nreal body",
        )
        .unwrap();
        std::fs::write(
            person_dir.join("yt-abc.dedup.json"),
            r#"{"already":"there"}"#,
        )
        .unwrap();

        // Nothing left to process -> Ok(0), never spawns squishi (which
        // would otherwise hang/error in a test environment without it).
        let result = dedup_raw_files(tmp.path(), "has-sidecar", false);
        assert_eq!(result, Ok(0));
    }

    // --- cluster_raw_files: pure/fast paths, no real squishi call ---

    #[test]
    fn cluster_raw_files_errors_for_a_missing_slug_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let result = cluster_raw_files(tmp.path(), "no-such-slug");
        assert!(result.is_err());
    }

    #[test]
    fn cluster_raw_files_errors_when_no_dedup_sidecars_exist() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("no-sidecars-yet");
        std::fs::create_dir_all(&person_dir).unwrap();
        std::fs::write(person_dir.join("yt-abc.md"), "raw, undeduped body").unwrap();

        let result = cluster_raw_files(tmp.path(), "no-sidecars-yet");
        let err = result.expect_err("should error without any .dedup.json sidecars");
        assert!(err.contains("dedup-raw"));
    }

    #[test]
    #[ignore] // requires `squishi` on PATH (real subprocess, real ONNX model)
    fn cluster_raw_files_ranks_a_sentence_repeated_across_files_above_a_one_off() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("recurring-topic");
        std::fs::create_dir_all(&person_dir).unwrap();

        // A concept sentence repeated (near-verbatim) across three
        // separate source files should out-rank a sentence that only
        // appears once anywhere in the pooled corpus -- the whole claim
        // this mechanism exists to prove.
        let recurring = "Values are the load-bearing unit of motivation, and context tells the \
                          genes what they need to do, more than almost anything else in coaching. ";
        let filler = |n: usize| -> String {
            format!(
                "{}Something entirely unrelated happens only in file {n}, a one-off aside about \
                 scheduling logistics that nobody ever repeats anywhere else in the corpus. ",
                recurring.repeat(9)
            )
        };
        for (i, body) in [filler(1), filler(2), filler(3)].into_iter().enumerate() {
            let sidecar = serde_json::json!({ "compressed": body }).to_string();
            std::fs::write(person_dir.join(format!("source-{i}.dedup.json")), sidecar).unwrap();
        }

        let result = cluster_raw_files(tmp.path(), "recurring-topic");
        assert!(
            result.is_ok(),
            "cluster_raw_files failed: {:?}",
            result.err()
        );
        let summary_path = result.unwrap();
        let summary: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&summary_path).unwrap()).unwrap();
        let topics = summary["topics"].as_array().unwrap();
        assert!(!topics.is_empty(), "expected at least one topic cluster");

        // The top-ranked topic should be (a near-paraphrase of) the
        // recurring sentence, with cluster_size > 1; the one-off aside
        // should never outrank it.
        let top = &topics[0];
        assert!(top["cluster_size"].as_u64().unwrap() > 1);
        assert!(
            top["text"]
                .as_str()
                .unwrap()
                .to_lowercase()
                .contains("motivation"),
            "expected the recurring sentence to rank first, got: {top}"
        );
    }

    // --- extract_concepts_files: pure/fast paths, no real embedder call ---

    #[test]
    fn extract_concepts_files_errors_for_a_missing_slug_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let result = extract_concepts_files(tmp.path(), "no-such-slug", false);
        assert!(result.is_err());
    }

    #[test]
    fn extract_concepts_files_on_an_empty_directory_returns_zero_without_loading_embedder() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("empty-slug")).unwrap();
        let result = extract_concepts_files(tmp.path(), "empty-slug", false);
        assert_eq!(result, Ok(0));
    }

    #[test]
    fn extract_concepts_files_skips_a_file_that_already_has_a_sidecar_unless_forced() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("has-sidecar");
        std::fs::create_dir_all(&person_dir).unwrap();
        std::fs::write(
            person_dir.join("yt-abc.md"),
            "---\nsource: x\n---\n\nreal body",
        )
        .unwrap();
        std::fs::write(
            person_dir.join("yt-abc.concepts.json"),
            r#"{"already":"there"}"#,
        )
        .unwrap();

        // Nothing left to process -> Ok(0), never loads the embedder
        // (which would otherwise download/init models in a test
        // environment that shouldn't need them for this path).
        let result = extract_concepts_files(tmp.path(), "has-sidecar", false);
        assert_eq!(result, Ok(0));
    }

    #[test]
    #[ignore] // requires the fastembed model cache (real download on first run)
    fn extract_concepts_files_writes_a_distilled_sidecar_for_a_real_file() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("real-slug");
        std::fs::create_dir_all(&person_dir).unwrap();
        let sentence = "I had a client who wanted to lose weight to fit into a wedding dress, \
                         and context tells the genes what they need to do, because values are \
                         the load-bearing unit of motivation and autonomy matters more than \
                         almost anything else in coaching. ";
        let text: String = std::iter::repeat_n(sentence, 20).collect();
        std::fs::write(
            person_dir.join("yt-real.md"),
            format!("---\nsource: x\n---\n\n{text}"),
        )
        .unwrap();

        let count = extract_concepts_files(tmp.path(), "real-slug", false).unwrap();
        assert_eq!(count, 1);

        let sidecar: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(person_dir.join("yt-real.concepts.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(sidecar["id"], "yt-real");
        let concepts = sidecar["concepts"].as_array().unwrap();
        // 20 exact repeats of one sentence should collapse to far fewer
        // than 20 -- proves the dedup actually ran, not just a passthrough.
        assert!(
            concepts.len() < 5,
            "expected heavy collapse of 20 identical sentences, got {}",
            concepts.len()
        );
    }

    // --- strip_frontmatter: pure ---

    #[test]
    fn strip_frontmatter_removes_the_yaml_block() {
        let content = "---\nsource: x\nperson: Y\n---\n\n# Title\n\nreal body text";
        assert_eq!(strip_frontmatter(content), "# Title\n\nreal body text");
    }

    #[test]
    fn strip_frontmatter_on_content_without_frontmatter_returns_it_unchanged() {
        let content = "just plain content, no frontmatter at all";
        assert_eq!(strip_frontmatter(content), content);
    }

    #[test]
    #[ignore] // requires `squishi` installed on PATH (real subprocess, first-run model download)
    fn run_squishi_dedup_batch_returns_the_real_json_contract_per_item() {
        // Real content, over squishi's SKIP_SEMANTIC_DEDUP_UNDER_CHARS
        // (2000 chars) so this actually exercises the ONNX dedup path,
        // not just the short-input line-dedup passthrough.
        let sentence = "I had a client who wanted to lose weight to fit into a wedding dress, \
                         and context tells the genes what they need to do, because values are \
                         the load-bearing unit of motivation and autonomy matters more than \
                         almost anything else in coaching. ";
        let text: String = std::iter::repeat_n(sentence, 20).collect();
        let items = vec![
            ("first".to_string(), text.clone(), true),
            ("second".to_string(), text, true),
        ];
        let result = run_squishi_dedup_batch(&items);
        assert!(
            result.is_ok(),
            "squishi not on PATH or failed: {:?}",
            result.err()
        );
        let results = result.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "first");
        assert_eq!(results[1].0, "second");
        let json: serde_json::Value = serde_json::from_str(&results[0].1).unwrap();
        assert_eq!(json["kind"], "PlainText");
        assert!(json.get("stories").is_some());
        assert!(json.get("drops").is_some());
    }

    #[test]
    fn run_squishi_dedup_batch_on_empty_input_returns_empty_without_spawning() {
        // Never touches the network/subprocess for an empty batch --
        // real, fast, no #[ignore] needed.
        let result = run_squishi_dedup_batch(&[]);
        assert_eq!(result, Ok(Vec::new()));
    }

    #[test]
    #[ignore] // requires `squishi` installed on PATH (real subprocess, first-run model download)
    fn run_squishi_dedup_batch_honors_restore_punctuation_false() {
        // Real, genuinely unpunctuated text, over squishi's 2000-char
        // semantic-dedup threshold so this actually reaches the path
        // where punctuation_restored gets set at all -- would trigger
        // restoration if allowed. Sent with restore_punctuation: false
        // (as ingest_wikipedia_pages/ingest_git_commits always do), so
        // squishi must report it was never attempted.
        let text: String = std::iter::repeat_n("word ", 600).collect();
        let items = vec![("wiki-item".to_string(), text, false)];
        let results = run_squishi_dedup_batch(&items).unwrap();
        let json: serde_json::Value = serde_json::from_str(&results[0].1).unwrap();
        assert_eq!(json["punctuation_restored"], false);
    }

    #[test]
    #[ignore] // requires `squishi` installed on PATH (real subprocess, first-run model download)
    fn dedup_raw_files_only_restores_punctuation_for_yt_prefixed_files() {
        let tmp = tempfile::tempdir().unwrap();
        let person_dir = tmp.path().join("mixed-source-slug");
        std::fs::create_dir_all(&person_dir).unwrap();

        // Real, genuinely unpunctuated text for both, over squishi's
        // 2000-char semantic-dedup threshold -- the only difference is
        // the filename prefix dedup_raw_files infers eligibility from.
        let unpunctuated: String = std::iter::repeat_n("word ", 600).collect();
        std::fs::write(
            person_dir.join("yt-realvideo.md"),
            format!("---\nsource: x\n---\n\n{unpunctuated}"),
        )
        .unwrap();
        std::fs::write(
            person_dir.join("wikipedia-realpage.md"),
            format!("---\nsource: x\n---\n\n{unpunctuated}"),
        )
        .unwrap();

        let count = dedup_raw_files(tmp.path(), "mixed-source-slug", false).unwrap();
        assert_eq!(count, 2);

        let yt_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(person_dir.join("yt-realvideo.dedup.json")).unwrap(),
        )
        .unwrap();
        let wiki_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(person_dir.join("wikipedia-realpage.dedup.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(yt_json["punctuation_restored"], true);
        assert_eq!(wiki_json["punctuation_restored"], false);
    }

    // --- build_website_raw_md: pure ---

    #[test]
    fn build_website_raw_md_matches_the_frontmatter_shape() {
        let md = build_website_raw_md(
            "Jane Doe",
            "jane-doe",
            "https://example.com/about",
            "real page text",
            "2026-08-10",
        );
        assert!(md.starts_with("---\nsource: https://example.com/about\n"));
        assert!(md.contains("type: website\n"));
        assert!(md.contains("word_count: 3\n"));
        assert!(md.ends_with("real page text\n"));
    }

    // --- fetch_website: real subprocess (agent-browser), #[ignore]d ---

    #[test]
    #[ignore] // requires `agent-browser` installed on PATH (real subprocess + browser)
    fn fetch_website_returns_real_agent_readable_text() {
        let text = fetch_website("https://example.com").unwrap();
        assert!(text.to_lowercase().contains("example"));
    }

    #[test]
    #[ignore] // requires `agent-browser` installed on PATH (real subprocess + browser)
    fn fetch_website_errors_on_a_real_nonexistent_host() {
        let result = fetch_website("https://this-host-genuinely-does-not-exist-12345.invalid");
        assert!(result.is_err());
    }

    // --- ingest_websites: real subprocess, #[ignore]d ---

    #[test]
    #[ignore] // requires `agent-browser` installed on PATH (real subprocess + browser)
    fn ingest_websites_is_resilient_to_one_bad_url() {
        let tmp = tempfile::tempdir().unwrap();
        let (written, failures) = ingest_websites(
            tmp.path(),
            "Jane Doe",
            "jane-doe",
            &[
                "https://example.com".to_string(),
                "https://this-host-genuinely-does-not-exist-12345.invalid".to_string(),
            ],
            "2026-08-10",
        )
        .expect("partial success should be Ok, not Err");
        assert_eq!(written.len(), 1);
        assert_eq!(failures.len(), 1);
    }

    // --- parse_github_slug: pure ---

    #[test]
    fn parse_github_slug_extracts_owner_repo_from_a_plain_url() {
        assert_eq!(
            parse_github_slug("https://github.com/bcherny/json-schema-to-typescript"),
            Some("bcherny/json-schema-to-typescript".to_string())
        );
    }

    #[test]
    fn parse_github_slug_strips_trailing_dot_git_and_slash() {
        assert_eq!(
            parse_github_slug("https://github.com/octocat/Hello-World.git/"),
            Some("octocat/Hello-World".to_string())
        );
    }

    #[test]
    fn parse_github_slug_returns_none_for_a_non_github_host() {
        assert_eq!(
            parse_github_slug("https://gitlab.com/someone/somerepo"),
            None
        );
    }

    #[test]
    fn parse_github_slug_returns_none_for_a_bare_github_root() {
        assert_eq!(parse_github_slug("https://github.com/"), None);
        assert_eq!(parse_github_slug("https://github.com/justowner"), None);
    }

    // --- build_git_issues_raw_md: pure ---

    #[test]
    fn build_git_issues_raw_md_matches_the_frontmatter_shape() {
        let md = build_git_issues_raw_md(
            "Jane Doe",
            "jane-doe",
            "example/repo",
            "On 2026-01-01: A real issue title.",
            "2026-08-10",
        );
        assert!(md.starts_with("---\nsource: https://github.com/example/repo/issues\n"));
        assert!(md.contains("type: git-issues\n"));
        assert!(md.contains("# Issues/PRs from example/repo\n"));
    }

    // --- fetch_git_issues: real subprocess (gh), #[ignore]d ---

    #[test]
    #[ignore] // requires `gh` authenticated on PATH (real subprocess + network)
    fn fetch_git_issues_returns_real_issues_for_a_real_author() {
        // GitHub's own public demo repo -- same choice
        // `fetch_git_commits_returns_real_commit_messages_for_a_real_author`
        // makes, for the same reason (small, stable, genuinely public).
        let text = fetch_git_issues("octocat/Hello-World", &["octocat".to_string()]);
        // octocat/Hello-World may genuinely have zero issues authored by
        // octocat specifically -- this asserts the call succeeds
        // mechanically (real `gh` subprocess, real JSON), not that
        // results are non-empty.
        assert!(
            text.is_ok() || text.as_ref().unwrap_err().contains("no issues/PRs found"),
            "expected either real results or the documented empty-result error: {text:?}"
        );
    }

    #[test]
    fn fetch_git_issues_errors_with_no_github_users() {
        let result = fetch_git_issues("example/repo", &[]);
        assert!(result.is_err());
    }
}
