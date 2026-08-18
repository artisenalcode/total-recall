//! `trm.json` — user-configurable behavior for automated session staging
//! (ADR-0006). Two locations, merged field-by-field: a global file at
//! `<data_root>/trm.json`, and an optional project override at
//! `<repo_root>/.trm/trm.json` (same relative shape as the existing
//! `.trm-bank` file precedent) for the repo enclosing `cwd`, if any.
//! Missing files are not errors — defaults apply, and those defaults
//! reproduce today's unconditional-staging behavior exactly (both hooks
//! enabled, no thresholds, keep archives forever) so `trm.json` existing
//! at all never changes behavior for someone who hasn't written one.
//! Malformed JSON on a file that *does* exist is a hard error, surfaced
//! to the caller: a typo'd config silently ignored is worse than a loud
//! failure here, since nothing in this module touches transcript
//! content — it only gates behavior elsewhere.
//!
//! Hand-parsed against `serde_json::Value` rather than a
//! `#[derive(Deserialize)]` struct — matches every other JSON-parsing
//! boundary in this crate (`ingest::parse_squishi_json`, squishi's own
//! `build_output`), and avoids pulling in `serde`'s derive feature for a
//! small, stable schema.

use crate::bank::find_repo_root;
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct TrmConfig {
    /// Overrides `bank::data_root()`'s own `~/.trm` default — but
    /// `MF_DATA_ROOT` (the existing env var, used throughout the test
    /// suite for isolation) still wins over this field wherever both are
    /// consulted; this field only fills in when the env var is unset.
    pub data_root: Option<String>,
    pub pre_compact_enabled: bool,
    pub session_end_enabled: bool,
    pub min_turns: usize,
    pub min_bytes: usize,
    /// `None` == keep archived transcripts forever.
    pub retention_days: Option<u64>,
    /// `ccr-gc`'s default age cutoff, in days -- an entry whose `last_seen`
    /// is older than this is evicted. 3 days is generous enough to survive
    /// a slow-moving multi-day task without accumulating forever.
    pub ccr_max_age_days: u64,
    /// `ccr-gc`'s default total-size cap across the whole CCR store, in
    /// bytes -- bounds unbounded growth even before the age cutoff would
    /// otherwise kick in on a very active machine.
    pub ccr_max_bytes: u64,
}

impl Default for TrmConfig {
    fn default() -> Self {
        TrmConfig {
            data_root: None,
            pre_compact_enabled: true,
            session_end_enabled: true,
            min_turns: 1,
            min_bytes: 0,
            retention_days: None,
            ccr_max_age_days: 3,
            ccr_max_bytes: 500_000_000,
        }
    }
}

/// Resolve `TrmConfig` for a given `data_root`/`cwd`: start from defaults,
/// merge the global file if present, then merge the project override (the
/// repo enclosing `cwd`, if any) on top — later merges win per-field, not
/// per-file, so a project file that only sets one key doesn't reset the
/// rest to default.
pub fn load(data_root: &Path, cwd: &Path) -> Result<TrmConfig, String> {
    let mut config = TrmConfig::default();

    if let Some(value) = read_json_file(&data_root.join("trm.json"))? {
        merge(&mut config, &value);
    }

    if let Some(repo_root) = find_repo_root(cwd)
        && let Some(value) = read_json_file(&repo_root.join(".trm").join("trm.json"))?
    {
        merge(&mut config, &value);
    }

    Ok(config)
}

/// `Ok(None)` if the file doesn't exist — not an error. `Err` only for a
/// file that exists but isn't valid JSON.
fn read_json_file(path: &Path) -> Result<Option<Value>, String> {
    let contents = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("failed to read {}: {e}", path.display())),
    };
    serde_json::from_str(&contents)
        .map(Some)
        .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))
}

/// Overwrite only the fields actually present in `value`, leaving
/// everything else in `config` untouched — the "field-level, not
/// file-level" merge semantics `load`'s doc comment promises.
fn merge(config: &mut TrmConfig, value: &Value) {
    if let Some(s) = value.get("data_root").and_then(|v| v.as_str()) {
        config.data_root = Some(s.to_string());
    }
    if let Some(b) = value
        .pointer("/hooks/pre_compact/enabled")
        .and_then(|v| v.as_bool())
    {
        config.pre_compact_enabled = b;
    }
    if let Some(b) = value
        .pointer("/hooks/session_end/enabled")
        .and_then(|v| v.as_bool())
    {
        config.session_end_enabled = b;
    }
    if let Some(n) = value.pointer("/staging/min_turns").and_then(|v| v.as_u64()) {
        config.min_turns = n as usize;
    }
    if let Some(n) = value.pointer("/staging/min_bytes").and_then(|v| v.as_u64()) {
        config.min_bytes = n as usize;
    }
    if let Some(n) = value
        .pointer("/sessions_archive/retention_days")
        .and_then(|v| v.as_u64())
    {
        config.retention_days = Some(n);
    }
    if let Some(n) = value.pointer("/ccr/max_age_days").and_then(|v| v.as_u64()) {
        config.ccr_max_age_days = n;
    }
    if let Some(n) = value.pointer("/ccr/max_bytes").and_then(|v| v.as_u64()) {
        config.ccr_max_bytes = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn no_files_present_returns_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let config = load(tmp.path(), tmp.path()).unwrap();
        assert_eq!(config, TrmConfig::default());
    }

    #[test]
    fn global_file_overrides_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("trm.json"),
            r#"{"hooks": {"pre_compact": {"enabled": false}}, "staging": {"min_turns": 3}}"#,
        )
        .unwrap();
        let config = load(tmp.path(), tmp.path()).unwrap();
        assert!(!config.pre_compact_enabled);
        assert_eq!(config.min_turns, 3);
        // Untouched fields keep their defaults.
        assert!(config.session_end_enabled);
        assert_eq!(config.min_bytes, 0);
    }

    #[test]
    fn project_override_wins_per_field_not_per_file() {
        let data_root = tempfile::tempdir().unwrap();
        fs::write(
            data_root.path().join("trm.json"),
            r#"{"hooks": {"pre_compact": {"enabled": false}}, "staging": {"min_turns": 3}}"#,
        )
        .unwrap();

        let repo = tempfile::tempdir().unwrap();
        fs::create_dir_all(repo.path().join(".git")).unwrap();
        fs::create_dir_all(repo.path().join(".trm")).unwrap();
        fs::write(
            repo.path().join(".trm").join("trm.json"),
            r#"{"staging": {"min_turns": 5}}"#,
        )
        .unwrap();

        let config = load(data_root.path(), repo.path()).unwrap();
        // Project file only set min_turns -- pre_compact_enabled must
        // still carry the GLOBAL file's value, not reset to default.
        assert!(!config.pre_compact_enabled);
        assert_eq!(config.min_turns, 5);
    }

    #[test]
    fn malformed_global_json_is_a_hard_error() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("trm.json"), "{not valid json").unwrap();
        let result = load(tmp.path(), tmp.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("trm.json"));
    }

    #[test]
    fn ccr_defaults_apply_when_unset() {
        let tmp = tempfile::tempdir().unwrap();
        let config = load(tmp.path(), tmp.path()).unwrap();
        assert_eq!(config.ccr_max_age_days, 3);
        assert_eq!(config.ccr_max_bytes, 500_000_000);
    }

    #[test]
    fn global_file_overrides_ccr_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("trm.json"),
            r#"{"ccr": {"max_age_days": 14, "max_bytes": 1000}}"#,
        )
        .unwrap();
        let config = load(tmp.path(), tmp.path()).unwrap();
        assert_eq!(config.ccr_max_age_days, 14);
        assert_eq!(config.ccr_max_bytes, 1000);
    }

    #[test]
    fn project_override_wins_over_global_for_ccr_fields_too() {
        let data_root = tempfile::tempdir().unwrap();
        fs::write(
            data_root.path().join("trm.json"),
            r#"{"ccr": {"max_age_days": 14}}"#,
        )
        .unwrap();

        let repo = tempfile::tempdir().unwrap();
        fs::create_dir_all(repo.path().join(".git")).unwrap();
        fs::create_dir_all(repo.path().join(".trm")).unwrap();
        fs::write(
            repo.path().join(".trm").join("trm.json"),
            r#"{"ccr": {"max_age_days": 1}}"#,
        )
        .unwrap();

        let config = load(data_root.path(), repo.path()).unwrap();
        assert_eq!(config.ccr_max_age_days, 1);
        // Untouched field keeps the default, not the global file's silence.
        assert_eq!(config.ccr_max_bytes, 500_000_000);
    }

    #[test]
    fn no_enclosing_repo_only_reads_global() {
        let data_root = tempfile::tempdir().unwrap();
        fs::write(
            data_root.path().join("trm.json"),
            r#"{"staging": {"min_bytes": 10}}"#,
        )
        .unwrap();
        let outside_any_repo = tempfile::tempdir().unwrap();
        let config = load(data_root.path(), outside_any_repo.path()).unwrap();
        assert_eq!(config.min_bytes, 10);
    }
}
