//! Per-bank archival checkpoint (ADR-0006 Phase 1): tracks which session
//! transcripts have already been archived, so `trm ingest-sessions --all`
//! is resumable — a second sweep is a no-op over sessions already done.
//!
//! Phase 1 scope only: an `archived` boolean per session id, nothing
//! line-offset-based. Live in-session incremental staging (Phase 2, the
//! `PreCompact` hook) needs a `last_staged_line` too, but that's a
//! different, still-open problem (see
//! docs/ideation/memory-cli/plan-2026-08-09-automated-session-staging.md's
//! Phasing section) — not added here speculatively.
//!
//! Defensive parsing throughout, same discipline squishi's `session_prune`
//! established for transcript JSONL: this file's shape isn't a versioned
//! contract either, so a missing or malformed file degrades to "nothing
//! archived yet" rather than erroring — a fresh bank with no checkpoint
//! file at all is exactly that state, not a failure.

use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionCheckpointEntry {
    pub archived: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Checkpoint {
    pub sessions: HashMap<String, SessionCheckpointEntry>,
}

/// Load a checkpoint from `path`. Absent or malformed → empty checkpoint,
/// never an error — see module doc comment.
pub fn load(path: &Path) -> Checkpoint {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Checkpoint::default();
    };
    let Ok(Value::Object(sessions)) = serde_json::from_str(&contents) else {
        return Checkpoint::default();
    };

    let mut checkpoint = Checkpoint::default();
    for (session_id, entry) in sessions {
        let archived = entry
            .get("archived")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        checkpoint
            .sessions
            .insert(session_id, SessionCheckpointEntry { archived });
    }
    checkpoint
}

/// Write `checkpoint` to `path`, atomically (matches `atomic::write`'s
/// tmp-then-rename pattern the rest of the crate uses for anything that
/// must never end up half-written).
pub fn save(path: &Path, checkpoint: &Checkpoint) -> std::io::Result<()> {
    let mut object = serde_json::Map::new();
    for (session_id, entry) in &checkpoint.sessions {
        let mut entry_obj = serde_json::Map::new();
        entry_obj.insert("archived".to_string(), Value::from(entry.archived));
        object.insert(session_id.clone(), Value::Object(entry_obj));
    }
    let contents =
        serde_json::to_string_pretty(&Value::Object(object)).unwrap_or_else(|_| "{}".to_string());
    crate::atomic::write(path, &contents)
}

/// `false` for a session never recorded — the natural "not archived yet"
/// default, not a special case callers need to handle separately.
pub fn is_archived(checkpoint: &Checkpoint, session_id: &str) -> bool {
    checkpoint
        .sessions
        .get(session_id)
        .is_some_and(|e| e.archived)
}

/// Record `session_id` as archived. Callers persist via `save` after.
pub fn mark_archived(checkpoint: &mut Checkpoint, session_id: &str) {
    checkpoint
        .sessions
        .entry(session_id.to_string())
        .or_default()
        .archived = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unseen_session_is_not_archived() {
        let checkpoint = Checkpoint::default();
        assert!(!is_archived(&checkpoint, "sess-1"));
    }

    #[test]
    fn round_trips_through_save_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(".session-state.json");

        let mut checkpoint = Checkpoint::default();
        mark_archived(&mut checkpoint, "sess-1");
        save(&path, &checkpoint).unwrap();

        let loaded = load(&path);
        assert!(is_archived(&loaded, "sess-1"));
        assert!(!is_archived(&loaded, "sess-2"));
    }

    #[test]
    fn missing_file_loads_as_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("does-not-exist.json");
        let checkpoint = load(&path);
        assert_eq!(checkpoint, Checkpoint::default());
    }

    #[test]
    fn malformed_file_loads_as_empty_not_a_panic() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(".session-state.json");
        std::fs::write(&path, "not json at all").unwrap();
        let checkpoint = load(&path);
        assert_eq!(checkpoint, Checkpoint::default());
    }

    #[test]
    fn mark_archived_is_idempotent() {
        let mut checkpoint = Checkpoint::default();
        mark_archived(&mut checkpoint, "sess-1");
        mark_archived(&mut checkpoint, "sess-1");
        assert_eq!(checkpoint.sessions.len(), 1);
        assert!(is_archived(&checkpoint, "sess-1"));
    }
}
