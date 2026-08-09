//! `trm ingest-session` — Rust port of `mindforge/tools/session_to_trm.py`'s
//! orchestration half. squishi owns extraction+compression
//! (`squishi --session-digest`); this module owns the part squishi
//! explicitly refuses to: calling `stage`. Keeps squishi's own stated
//! boundary ("compresses text, never stores/retrieves") intact — the
//! storage-owning tool is the one that calls storage.
//!
//! The one real behavioral requirement ported exactly from the Python
//! version: the digest gets staged into the bank resolved from the
//! *session's own* cwd (parsed out of squishi's output), not whatever
//! directory `trm ingest-session` itself is invoked from. The Python
//! version did this by setting the subprocess's own `cwd=`; here it's
//! done by passing the parsed cwd straight into `bank::resolve_bank_id`
//! instead of the ambient `std::env::current_dir()`.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionDigest {
    pub content: String,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    pub turn_count: usize,
    /// squishi's real total line count for the transcript this digest
    /// came from (ADR-0006 Phase 2) — independent of whatever `start_line`
    /// the call used. An incremental caller (`--since-checkpoint`) saves
    /// this as the session's new `last_staged_line`.
    pub total_lines: usize,
}

/// Parse `squishi --session-digest ... --json`'s real output shape.
/// Pure and independently testable — the subprocess call itself
/// (`run_squishi_session_digest`) is a thin wrapper around this plus
/// real process I/O, exercised by the black-box CLI tests instead.
pub fn parse_squishi_json(json_str: &str) -> Result<SessionDigest, String> {
    let value: Value =
        serde_json::from_str(json_str).map_err(|e| format!("invalid JSON from squishi: {e}"))?;
    let content = value
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or("squishi output missing \"content\"")?
        .to_string();
    let session_id = value
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let cwd = value
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let turn_count = value
        .get("turn_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let total_lines = value
        .get("total_lines")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    Ok(SessionDigest {
        content,
        session_id,
        cwd,
        turn_count,
        total_lines,
    })
}

/// Same reason format as the Python version's `session_to_trm.py`.
pub fn build_reason(digest: &SessionDigest) -> String {
    format!(
        "Claude Code session {} in {} ({} text turns) — squishi-compressed transcript, judge which durable facts are worth keeping",
        digest.session_id.as_deref().unwrap_or("unknown"),
        digest.cwd.as_deref().unwrap_or("unknown"),
        digest.turn_count,
    )
}

/// Spawn `squishi --session-digest <path> --json` (whole-file, `--start-line
/// 0`) and parse its output. Thin wrapper over `run_squishi_session_digest_from`
/// — unchanged behavior for every existing caller.
pub fn run_squishi_session_digest(path: &Path) -> Result<SessionDigest, String> {
    run_squishi_session_digest_from(path, 0)
}

/// Same as `run_squishi_session_digest`, but with an explicit `--start-line`
/// (ADR-0006 Phase 2) — an incremental caller passes the session's saved
/// `last_staged_line` here and gets back only the delta. Real process I/O
/// — not unit-tested directly (mocking a subprocess call buys nothing
/// real); covered by the black-box CLI tests instead, same discipline as
/// every other real-subprocess boundary in this project.
pub fn run_squishi_session_digest_from(
    path: &Path,
    start_line: usize,
) -> Result<SessionDigest, String> {
    let output = Command::new("squishi")
        .arg("--session-digest")
        .arg(path)
        .arg("--start-line")
        .arg(start_line.to_string())
        .arg("--json")
        .output()
        .map_err(|e| format!("squishi not on PATH: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "squishi --session-digest failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    parse_squishi_json(&String::from_utf8_lossy(&output.stdout))
}

/// Where `trm ingest-sessions --all` (ADR-0006 Phase 1) looks for
/// transcripts. `MF_CLAUDE_PROJECTS_DIR` overrides for tests, same
/// precedent as `bank::data_root`'s `MF_DATA_ROOT`; defaults to Claude
/// Code's real, undocumented-but-observed layout, `~/.claude/projects`.
pub fn claude_projects_dir() -> PathBuf {
    if let Ok(over) = std::env::var("MF_CLAUDE_PROJECTS_DIR") {
        return PathBuf::from(over);
    }
    let home = std::env::var("HOME").expect("HOME must be set to locate ~/.claude/projects");
    PathBuf::from(home).join(".claude").join("projects")
}

/// Every `*.jsonl` file directly under `projects_dir`'s immediate
/// subdirectories (`<projects_dir>/<project>/<session-id>.jsonl`, the
/// real layout confirmed against this repo's own `~/.claude/projects/`).
/// Defensive: an unreadable `projects_dir` or subdirectory yields fewer
/// results, never a hard error — matches every other transcript-reading
/// path in this codebase's "not a versioned contract" discipline.
pub fn find_transcripts(projects_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(project_dirs) = std::fs::read_dir(projects_dir) else {
        return out;
    };
    for project_entry in project_dirs.flatten() {
        let project_path = project_entry.path();
        if !project_path.is_dir() {
            continue;
        }
        let Ok(files) = std::fs::read_dir(&project_path) else {
            continue;
        };
        for file_entry in files.flatten() {
            let file_path = file_entry.path();
            if file_path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                out.push(file_path);
            }
        }
    }
    out.sort();
    out
}

/// A cheap peek at a transcript's `sessionId`/`cwd`, without spawning
/// squishi — just the first line that carries both non-null (skips the
/// real, observed leading `{"type":"mode", ..., "cwd": null}` line).
/// Lets `ingest-sessions --all` resolve which bank's checkpoint to check
/// *before* paying for a real digest, for sessions it's about to skip
/// anyway.
pub fn peek_session_meta(path: &Path) -> Option<(String, PathBuf)> {
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    for line in std::io::BufRead::lines(reader).map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let session_id = value.get("sessionId").and_then(|v| v.as_str());
        let cwd = value.get("cwd").and_then(|v| v.as_str());
        if let (Some(session_id), Some(cwd)) = (session_id, cwd) {
            return Some((session_id.to_string(), PathBuf::from(cwd)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- find_transcripts / peek_session_meta: pure(ish), real fixture dirs ---

    #[test]
    fn find_transcripts_finds_jsonl_under_project_subdirectories() {
        let tmp = tempfile::tempdir().unwrap();
        let project_a = tmp.path().join("-home-alvin-repo-a");
        let project_b = tmp.path().join("-home-alvin-repo-b");
        std::fs::create_dir_all(&project_a).unwrap();
        std::fs::create_dir_all(&project_b).unwrap();
        std::fs::write(project_a.join("sess-1.jsonl"), "{}").unwrap();
        std::fs::write(project_a.join("not-a-transcript.txt"), "ignore me").unwrap();
        std::fs::write(project_b.join("sess-2.jsonl"), "{}").unwrap();

        let found = find_transcripts(tmp.path());
        assert_eq!(found.len(), 2);
        assert!(found.iter().any(|p| p.ends_with("sess-1.jsonl")));
        assert!(found.iter().any(|p| p.ends_with("sess-2.jsonl")));
    }

    #[test]
    fn find_transcripts_on_missing_dir_returns_empty_not_a_panic() {
        let found = find_transcripts(Path::new("/does/not/exist/at/all"));
        assert!(found.is_empty());
    }

    #[test]
    fn peek_session_meta_skips_the_leading_mode_line() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sess.jsonl");
        std::fs::write(
            &path,
            "{\"type\":\"mode\",\"sessionId\":\"sess-1\",\"cwd\":null}\n\
             {\"type\":\"user\",\"sessionId\":\"sess-1\",\"cwd\":\"/repo/a\"}\n",
        )
        .unwrap();

        let (session_id, cwd) = peek_session_meta(&path).unwrap();
        assert_eq!(session_id, "sess-1");
        assert_eq!(cwd, PathBuf::from("/repo/a"));
    }

    #[test]
    fn peek_session_meta_on_a_file_with_no_usable_line_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sess.jsonl");
        std::fs::write(&path, "not json\n{\"type\":\"mode\",\"cwd\":null}\n").unwrap();
        assert!(peek_session_meta(&path).is_none());
    }

    /// Real shape — matches squishi's actual `--session-digest --json`
    /// output exactly (verified live against the real binary before
    /// this fixture was written, not guessed; `total_lines` added
    /// ADR-0006 Phase 2, reverified live the same way).
    const REAL_SHAPE_JSON: &str = r#"{"chars_after":100,"chars_before":200,"content":"SESSION DIGEST sess-1\n\n---\ntype: session-digest\nsession_id: sess-1\ncwd: /repo\nfirst_ts: t1\nlast_ts: t2\nturn_count: 3\n---\n\nbody","cwd":"/repo","first_ts":"t1","last_ts":"t2","raw_bytes":9000,"session_id":"sess-1","total_lines":5,"truncated":false,"turn_count":3}"#;

    #[test]
    fn parse_squishi_json_extracts_the_real_fields() {
        let digest = parse_squishi_json(REAL_SHAPE_JSON).unwrap();
        assert_eq!(digest.session_id.as_deref(), Some("sess-1"));
        assert_eq!(digest.cwd.as_deref(), Some("/repo"));
        assert_eq!(digest.turn_count, 3);
        assert_eq!(digest.total_lines, 5);
        assert!(digest.content.starts_with("SESSION DIGEST sess-1"));
    }

    #[test]
    fn parse_squishi_json_rejects_invalid_json() {
        assert!(parse_squishi_json("not json").is_err());
    }

    #[test]
    fn parse_squishi_json_rejects_json_missing_content() {
        assert!(parse_squishi_json(r#"{"session_id":"s"}"#).is_err());
    }

    #[test]
    fn parse_squishi_json_tolerates_null_meta_fields() {
        let json = r#"{"content":"body","session_id":null,"cwd":null,"turn_count":0}"#;
        let digest = parse_squishi_json(json).unwrap();
        assert_eq!(digest.session_id, None);
        assert_eq!(digest.cwd, None);
        assert_eq!(digest.content, "body");
    }

    #[test]
    fn build_reason_matches_the_expected_format() {
        let digest = SessionDigest {
            content: "irrelevant".to_string(),
            session_id: Some("sess-1".to_string()),
            cwd: Some("/repo".to_string()),
            turn_count: 5,
            total_lines: 0,
        };
        let reason = build_reason(&digest);
        assert_eq!(
            reason,
            "Claude Code session sess-1 in /repo (5 text turns) — squishi-compressed transcript, judge which durable facts are worth keeping"
        );
    }

    #[test]
    fn build_reason_handles_missing_meta_gracefully() {
        let digest = SessionDigest::default();
        let reason = build_reason(&digest);
        assert!(reason.contains("unknown"));
    }
}
