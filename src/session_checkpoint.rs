//! Per-bank session checkpoint tracking two independent facts per session id: `archived` (gzip-archived and removed) and `last_staged_line`
//! (how much of a still-live session has been staged, for incremental `--since-checkpoint`). Missing/malformed file degrades to empty, never errors.

use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionCheckpointEntry {
    pub archived: bool,
    pub last_staged_line: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Checkpoint {
    pub sessions: HashMap<String, SessionCheckpointEntry>,
}

/// Load a checkpoint from `path`. Absent or malformed → empty checkpoint, never an error.
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
        let last_staged_line = entry
            .get("last_staged_line")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        checkpoint.sessions.insert(
            session_id,
            SessionCheckpointEntry {
                archived,
                last_staged_line,
            },
        );
    }
    checkpoint
}

/// Write `checkpoint` to `path` atomically, via `atomic::write`'s tmp-then-rename.
pub fn save(path: &Path, checkpoint: &Checkpoint) -> std::io::Result<()> {
    let mut object = serde_json::Map::new();
    for (session_id, entry) in &checkpoint.sessions {
        let mut entry_obj = serde_json::Map::new();
        entry_obj.insert("archived".to_string(), Value::from(entry.archived));
        entry_obj.insert(
            "last_staged_line".to_string(),
            Value::from(entry.last_staged_line),
        );
        object.insert(session_id.clone(), Value::Object(entry_obj));
    }
    let contents =
        serde_json::to_string_pretty(&Value::Object(object)).unwrap_or_else(|_| "{}".to_string());
    crate::atomic::write(path, &contents)
}

/// `false` for a session never recorded -- the natural "not archived yet" default.
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

/// `0` for a session never recorded -- matches squishi's own `--start-line 0` whole-file behavior.
pub fn last_staged_line(checkpoint: &Checkpoint, session_id: &str) -> usize {
    checkpoint
        .sessions
        .get(session_id)
        .map(|e| e.last_staged_line)
        .unwrap_or(0)
}

/// Record how many lines of `session_id` have been staged; always sets to `line` directly, never accumulates.
pub fn mark_staged(checkpoint: &mut Checkpoint, session_id: &str, line: usize) {
    checkpoint
        .sessions
        .entry(session_id.to_string())
        .or_default()
        .last_staged_line = line;
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

    // --- last_staged_line / mark_staged ---

    #[test]
    fn unseen_session_has_last_staged_line_zero() {
        let checkpoint = Checkpoint::default();
        assert_eq!(last_staged_line(&checkpoint, "sess-1"), 0);
    }

    #[test]
    fn mark_staged_advances_last_staged_line() {
        let mut checkpoint = Checkpoint::default();
        mark_staged(&mut checkpoint, "sess-1", 42);
        assert_eq!(last_staged_line(&checkpoint, "sess-1"), 42);
        // A later call sets, doesn't accumulate.
        mark_staged(&mut checkpoint, "sess-1", 90);
        assert_eq!(last_staged_line(&checkpoint, "sess-1"), 90);
    }

    #[test]
    fn last_staged_line_round_trips_through_save_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(".session-state.json");

        let mut checkpoint = Checkpoint::default();
        mark_staged(&mut checkpoint, "sess-1", 123);
        save(&path, &checkpoint).unwrap();

        let loaded = load(&path);
        assert_eq!(last_staged_line(&loaded, "sess-1"), 123);
    }

    #[test]
    fn archived_and_last_staged_line_are_independent_fields() {
        // Setting one must not disturb the other.
        let mut checkpoint = Checkpoint::default();
        mark_staged(&mut checkpoint, "sess-1", 50);
        mark_archived(&mut checkpoint, "sess-1");
        assert_eq!(last_staged_line(&checkpoint, "sess-1"), 50);
        assert!(is_archived(&checkpoint, "sess-1"));

        let mut checkpoint2 = Checkpoint::default();
        mark_archived(&mut checkpoint2, "sess-2");
        mark_staged(&mut checkpoint2, "sess-2", 7);
        assert!(is_archived(&checkpoint2, "sess-2"));
        assert_eq!(last_staged_line(&checkpoint2, "sess-2"), 7);
    }
}
