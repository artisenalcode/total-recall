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
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionDigest {
    pub content: String,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    pub turn_count: usize,
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

    Ok(SessionDigest {
        content,
        session_id,
        cwd,
        turn_count,
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

/// Spawn `squishi --session-digest <path> --json` and parse its output.
/// Real process I/O — not unit-tested directly (mocking a subprocess
/// call buys nothing real); covered by the black-box CLI tests instead,
/// same discipline as every other real-subprocess boundary in this
/// project.
pub fn run_squishi_session_digest(path: &Path) -> Result<SessionDigest, String> {
    let output = Command::new("squishi")
        .arg("--session-digest")
        .arg(path)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Real shape — matches squishi's actual `--session-digest --json`
    /// output exactly (verified live against the real binary before
    /// this fixture was written, not guessed).
    const REAL_SHAPE_JSON: &str = r#"{"chars_after":100,"chars_before":200,"content":"SESSION DIGEST sess-1\n\n---\ntype: session-digest\nsession_id: sess-1\ncwd: /repo\nfirst_ts: t1\nlast_ts: t2\nturn_count: 3\n---\n\nbody","cwd":"/repo","first_ts":"t1","last_ts":"t2","raw_bytes":9000,"session_id":"sess-1","truncated":false,"turn_count":3}"#;

    #[test]
    fn parse_squishi_json_extracts_the_real_fields() {
        let digest = parse_squishi_json(REAL_SHAPE_JSON).unwrap();
        assert_eq!(digest.session_id.as_deref(), Some("sess-1"));
        assert_eq!(digest.cwd.as_deref(), Some("/repo"));
        assert_eq!(digest.turn_count, 3);
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
