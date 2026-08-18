//! Self-diagnostics — checks this tool's own real failure surface (data
//! root permissions, bank resolution/lazy-creation, lock staleness, model
//! cache reachability, whether the embedder actually loads) and reports
//! pass/warn/fail per check, plus an optional `--fix` for the one real
//! repairable thing in that surface today (a stale lock file).
//!
//! Modeled on `agent-browser doctor` (audited 2026-08-07,
//! docs/ideation/audit-repo-techniques/2026-08-07-enrichment-techniques.md):
//! one command, read-only checks by default, `--fix` gated separately,
//! `--json` for scripting, real exit codes.
//!
//! Deliberately not a shared framework with squishi's own `--doctor` —
//! two real implementations as of this plan, not a third need yet (this
//! project's own "no shared abstraction before a third real need"
//! convention, already applied twice this session).

use crate::bank;
use crate::ccr;
use crate::config;
use crate::embeddings::Embedder;
use crate::lock;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Warn,
    Fail,
    /// `--quick` skipped an expensive check — distinct from Pass so a
    /// caller can tell "verified working" from "not actually checked."
    Skipped,
}

#[derive(Debug, Clone)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct DoctorReport {
    pub checks: Vec<Check>,
}

impl DoctorReport {
    pub fn has_failures(&self) -> bool {
        self.checks.iter().any(|c| c.status == Status::Fail)
    }

    /// Hand-rolled, not `serde_json` — this crate has no JSON dependency
    /// at all today, and a flat list of three string fields per check
    /// doesn't earn adding one (same reasoning squishi's own
    /// `build_output` doc comment gives for not reaching for a derive
    /// macro on a small, controlled output shape).
    pub fn to_json(&self) -> String {
        let checks: Vec<String> = self
            .checks
            .iter()
            .map(|c| {
                format!(
                    r#"{{"name":"{}","status":"{}","message":"{}"}}"#,
                    escape_json(&c.name),
                    c.status.as_str(),
                    escape_json(&c.message)
                )
            })
            .collect();
        format!(r#"{{"checks":[{}]}}"#, checks.join(","))
    }
}

impl Status {
    fn as_str(&self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Warn => "warn",
            Status::Fail => "fail",
            Status::Skipped => "skipped",
        }
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Run every check against `data_root`/`cwd`. Read-only — never mutates
/// anything (that's `fix`'s job, called separately and explicitly).
/// `quick` skips the real embedder-load check (the one expensive,
/// network-capable check in this surface).
pub fn run(data_root: &Path, cwd: &Path, quick: bool) -> DoctorReport {
    let mut checks = Vec::new();

    checks.push(check_data_root_writable(data_root));

    let bank_id = bank::resolve_bank_id(None, cwd);
    let paths = bank::paths_for(data_root, &bank_id);
    checks.push(check_bank_resolution(&bank_id, &paths));
    checks.push(check_lock_state(&paths));

    let models_dir = data_root.join("models");
    checks.push(check_model_cache_reachable(&models_dir));
    checks.push(check_embedder_loads(&models_dir, quick));

    checks.push(check_ccr_health(data_root, cwd));

    DoctorReport { checks }
}

fn check_data_root_writable(data_root: &Path) -> Check {
    let name = "data_root_writable".to_string();
    if let Err(e) = std::fs::create_dir_all(data_root) {
        return Check {
            name,
            status: Status::Fail,
            message: format!("cannot create {}: {e}", data_root.display()),
        };
    }
    let probe = data_root.join(".doctor-write-probe");
    match std::fs::write(&probe, b"probe") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            Check {
                name,
                status: Status::Pass,
                message: format!("{} is writable", data_root.display()),
            }
        }
        Err(e) => Check {
            name,
            status: Status::Fail,
            message: format!("{} is not writable: {e}", data_root.display()),
        },
    }
}

fn check_bank_resolution(bank_id: &str, paths: &bank::BankPaths) -> Check {
    let name = "bank_resolution".to_string();
    if paths.root.exists() {
        Check {
            name,
            status: Status::Pass,
            message: format!("cwd resolves to bank {bank_id:?}, already exists"),
        }
    } else {
        // Not a failure — banks are lazily created on first write (found
        // for real this session: a bank genuinely didn't exist until
        // something first wrote to it).
        Check {
            name,
            status: Status::Warn,
            message: format!(
                "cwd resolves to bank {bank_id:?}, not created yet (created on first write)"
            ),
        }
    }
}

fn check_lock_state(paths: &bank::BankPaths) -> Check {
    let name = "lock_state".to_string();
    let lock_path = paths.root.join(".lock");
    if !lock_path.exists() {
        return Check {
            name,
            status: Status::Pass,
            message: "no lock held".to_string(),
        };
    }
    let contents = match std::fs::read_to_string(&lock_path) {
        Ok(c) => c,
        Err(e) => {
            return Check {
                name,
                status: Status::Warn,
                message: format!("lock file exists but unreadable: {e}"),
            };
        }
    };
    match lock::parse_pid(&contents) {
        Some(pid) if lock::is_pid_alive(pid) => Check {
            name,
            status: Status::Warn,
            message: format!("lock held by live pid {pid}"),
        },
        Some(pid) => Check {
            name,
            status: Status::Warn,
            message: format!("stale lock (pid {pid} not alive) — would be reclaimed automatically, or run --fix"),
        },
        None => Check {
            name,
            status: Status::Warn,
            message: "lock file exists but has no parseable pid — would be reclaimed automatically, or run --fix".to_string(),
        },
    }
}

fn check_model_cache_reachable(models_dir: &Path) -> Check {
    let name = "model_cache_reachable".to_string();
    if let Err(e) = std::fs::create_dir_all(models_dir) {
        return Check {
            name,
            status: Status::Fail,
            message: format!("cannot create {}: {e}", models_dir.display()),
        };
    }
    let probe = models_dir.join(".doctor-write-probe");
    if let Err(e) = std::fs::write(&probe, b"probe") {
        return Check {
            name,
            status: Status::Fail,
            message: format!("{} is not writable: {e}", models_dir.display()),
        };
    }
    let _ = std::fs::remove_file(&probe);

    // Real signal, no network call: does the cache dir already contain
    // anything (a prior download), or would loading the embedder for
    // real need a fresh download?
    let has_cached_files = std::fs::read_dir(models_dir)
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false);
    Check {
        name,
        status: Status::Pass,
        message: if has_cached_files {
            format!(
                "{} is writable, model files already cached",
                models_dir.display()
            )
        } else {
            format!(
                "{} is writable, nothing cached yet (first real load will download)",
                models_dir.display()
            )
        },
    }
}

fn check_embedder_loads(models_dir: &Path, quick: bool) -> Check {
    let name = "embedder_loads".to_string();
    if quick {
        return Check {
            name,
            status: Status::Skipped,
            message: "skipped (--quick)".to_string(),
        };
    }
    // This single real load covers both `recall`'s and `stage()`'s
    // concept-split viability — report it once, under one label, rather
    // than loading the model twice under two different check names.
    match Embedder::new(models_dir.to_path_buf()) {
        Ok(_) => Check {
            name,
            status: Status::Pass,
            message: "embedder loads — recall and stage()'s concept-split will both work"
                .to_string(),
        },
        Err(e) => Check {
            name,
            status: Status::Fail,
            message: format!(
                "embedder failed to load: {e} — recall and stage()'s concept-split will both degrade or fail"
            ),
        },
    }
}

/// Reports CCR store size against `trm.json`'s configured limits (or the
/// built-in defaults if unset/unreadable). WARN, never FAIL -- an
/// oversized store is a real "you should run `ccr-gc`" nudge, not a broken
/// tool, same posture as `check_bank_resolution`'s "not created yet" case.
/// A malformed `trm.json` is not re-surfaced here -- that's already a hard
/// error on the command paths that actually read it (`ingest-session`,
/// `ccr-gc`); this check falls back to defaults rather than duplicating
/// that failure under a misleading name.
fn check_ccr_health(data_root: &Path, cwd: &Path) -> Check {
    let name = "ccr_health".to_string();
    let cfg = config::load(data_root, cwd).unwrap_or_default();
    let stats = ccr::stats(&data_root.join("ccr"));

    let max_age_secs = cfg.ccr_max_age_days * 24 * 3600;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let oldest_age_secs = stats.oldest_last_seen.map(|t| now.saturating_sub(t));

    let over_bytes = stats.total_bytes > cfg.ccr_max_bytes;
    let over_age = oldest_age_secs.is_some_and(|age| age > max_age_secs);

    if over_bytes || over_age {
        Check {
            name,
            status: Status::Warn,
            message: format!(
                "{} entries, {} bytes (cap {}), oldest entry {}s old (cutoff {}s) — run `trm ccr-gc`",
                stats.entry_count,
                stats.total_bytes,
                cfg.ccr_max_bytes,
                oldest_age_secs.unwrap_or(0),
                max_age_secs
            ),
        }
    } else {
        Check {
            name,
            status: Status::Pass,
            message: format!(
                "{} entries, {} bytes, within configured limits",
                stats.entry_count, stats.total_bytes
            ),
        }
    }
}

/// The one real repair available in today's failure surface: reclaim a
/// stale (dead-pid) lock. Returns what it did, or `None` if there was
/// nothing to fix. Strictly opt-in — `run()` above never calls this.
pub fn fix(paths: &bank::BankPaths) -> Option<String> {
    let lock_path = paths.root.join(".lock");
    if !lock_path.exists() {
        return None;
    }
    let contents = std::fs::read_to_string(&lock_path).ok()?;
    let is_stale = match lock::parse_pid(&contents) {
        Some(pid) => !lock::is_pid_alive(pid),
        None => true,
    };
    if is_stale {
        std::fs::remove_file(&lock_path).ok()?;
        Some(format!("removed stale lock at {}", lock_path.display()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_root_writable_check_passes_on_a_real_writable_scratch_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let check = check_data_root_writable(tmp.path());
        assert_eq!(check.status, Status::Pass);
    }

    #[test]
    fn data_root_writable_check_fails_on_an_unwritable_path() {
        // A path under a read-only parent — real permissions, not mocked.
        let tmp = tempfile::tempdir().unwrap();
        let readonly_parent = tmp.path().join("readonly");
        std::fs::create_dir(&readonly_parent).unwrap();
        let mut perms = std::fs::metadata(&readonly_parent).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o444);
        std::fs::set_permissions(&readonly_parent, perms.clone()).unwrap();

        let unwritable_child = readonly_parent.join("data_root");
        let check = check_data_root_writable(&unwritable_child);

        // Restore permissions so tempdir cleanup can actually delete it.
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
        std::fs::set_permissions(&readonly_parent, perms).unwrap();

        assert_eq!(check.status, Status::Fail);
    }

    #[test]
    fn bank_resolution_check_warns_when_bank_not_yet_created() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = bank::paths_for(tmp.path(), "some-bank");
        let check = check_bank_resolution("some-bank", &paths);
        assert_eq!(check.status, Status::Warn);
        assert!(check.message.contains("not created yet"));
    }

    #[test]
    fn bank_resolution_check_passes_when_bank_already_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = bank::paths_for(tmp.path(), "some-bank");
        std::fs::create_dir_all(&paths.root).unwrap();
        let check = check_bank_resolution("some-bank", &paths);
        assert_eq!(check.status, Status::Pass);
    }

    #[test]
    fn lock_state_check_passes_when_no_lock_file() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = bank::paths_for(tmp.path(), "some-bank");
        std::fs::create_dir_all(&paths.root).unwrap();
        let check = check_lock_state(&paths);
        assert_eq!(check.status, Status::Pass);
    }

    #[test]
    fn lock_state_check_warns_on_a_stale_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = bank::paths_for(tmp.path(), "some-bank");
        std::fs::create_dir_all(&paths.root).unwrap();
        std::fs::write(paths.root.join(".lock"), "pid=999999999\nacquired_at=0\n").unwrap();
        let check = check_lock_state(&paths);
        assert_eq!(check.status, Status::Warn);
        assert!(check.message.contains("stale"));
    }

    #[test]
    fn lock_state_check_warns_on_a_live_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = bank::paths_for(tmp.path(), "some-bank");
        std::fs::create_dir_all(&paths.root).unwrap();
        // pid 1 (init) is always alive on Linux and never this test process.
        std::fs::write(paths.root.join(".lock"), "pid=1\nacquired_at=0\n").unwrap();
        let check = check_lock_state(&paths);
        assert_eq!(check.status, Status::Warn);
        assert!(check.message.contains("live pid 1"));
    }

    #[test]
    fn embedder_loads_check_is_skipped_in_quick_mode() {
        let tmp = tempfile::tempdir().unwrap();
        let check = check_embedder_loads(tmp.path(), true);
        assert_eq!(check.status, Status::Skipped);
    }

    #[test]
    fn fix_removes_a_real_stale_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = bank::paths_for(tmp.path(), "some-bank");
        std::fs::create_dir_all(&paths.root).unwrap();
        std::fs::write(paths.root.join(".lock"), "pid=999999999\nacquired_at=0\n").unwrap();

        let result = fix(&paths);

        assert!(result.is_some());
        assert!(!paths.root.join(".lock").exists());
    }

    #[test]
    fn fix_does_nothing_to_a_live_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = bank::paths_for(tmp.path(), "some-bank");
        std::fs::create_dir_all(&paths.root).unwrap();
        std::fs::write(paths.root.join(".lock"), "pid=1\nacquired_at=0\n").unwrap();

        let result = fix(&paths);

        assert!(result.is_none());
        assert!(paths.root.join(".lock").exists());
    }

    #[test]
    fn fix_does_nothing_when_no_lock_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = bank::paths_for(tmp.path(), "some-bank");
        std::fs::create_dir_all(&paths.root).unwrap();
        assert!(fix(&paths).is_none());
    }

    #[test]
    fn ccr_health_check_passes_on_an_empty_store() {
        let tmp = tempfile::tempdir().unwrap();
        let check = check_ccr_health(tmp.path(), tmp.path());
        assert_eq!(check.status, Status::Pass);
    }

    #[test]
    fn ccr_health_check_warns_when_over_configured_byte_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let data_root = tmp.path();
        ccr::put(&data_root.join("ccr"), b"twenty bytes of blob", None).unwrap();
        std::fs::write(data_root.join("trm.json"), r#"{"ccr": {"max_bytes": 1}}"#).unwrap();

        let check = check_ccr_health(data_root, data_root);

        assert_eq!(check.status, Status::Warn);
        assert!(check.message.contains("ccr-gc"));
    }

    #[test]
    fn ccr_health_check_warns_when_oldest_entry_exceeds_the_age_cutoff() {
        let tmp = tempfile::tempdir().unwrap();
        let data_root = tmp.path();
        let ccr_root = data_root.join("ccr");
        let handle = ccr::put(&ccr_root, b"an aging blob", None).unwrap();
        let meta = ccr::meta_path_for(&ccr_root, &handle).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&meta).unwrap()).unwrap();
        value["last_seen"] = serde_json::json!(0); // unix epoch -- guaranteed ancient
        std::fs::write(&meta, serde_json::to_string(&value).unwrap()).unwrap();
        std::fs::write(
            data_root.join("trm.json"),
            r#"{"ccr": {"max_age_days": 1}}"#,
        )
        .unwrap();

        let check = check_ccr_health(data_root, data_root);

        assert_eq!(check.status, Status::Warn);
        assert!(check.message.contains("ccr-gc"));
    }

    #[test]
    fn ccr_health_check_falls_back_to_defaults_on_malformed_config_instead_of_erroring() {
        let tmp = tempfile::tempdir().unwrap();
        let data_root = tmp.path();
        std::fs::write(data_root.join("trm.json"), "{not valid json").unwrap();

        // Must not panic and must report the (empty) real store state under
        // the default thresholds, not surface the config parse error here.
        let check = check_ccr_health(data_root, data_root);
        assert_eq!(check.status, Status::Pass);
    }

    #[test]
    fn to_json_produces_valid_json_shape_with_escaped_special_characters() {
        let report = DoctorReport {
            checks: vec![
                Check {
                    name: "data_root_writable".to_string(),
                    status: Status::Fail,
                    message: r#"cannot write "quoted" path\with\backslashes"#.to_string(),
                },
                Check {
                    name: "embedder_loads".to_string(),
                    status: Status::Skipped,
                    message: "skipped (--quick)".to_string(),
                },
            ],
        };
        let json = report.to_json();
        assert!(json.contains(r#""status":"fail""#));
        assert!(json.contains(r#""status":"skipped""#));
        assert!(json.contains(r#"\"quoted\""#));
        assert!(json.contains(r"\\with\\backslashes"));
        // Not a full JSON parser (no serde_json dependency in this
        // crate), but balanced braces/brackets is a real, cheap
        // structural sanity check.
        assert_eq!(json.matches('{').count(), json.matches('}').count());
        assert_eq!(json.matches('[').count(), json.matches(']').count());
    }

    #[test]
    fn report_has_failures_reflects_real_check_statuses() {
        let mut report = DoctorReport::default();
        assert!(!report.has_failures());
        report.checks.push(Check {
            name: "x".to_string(),
            status: Status::Warn,
            message: "".to_string(),
        });
        assert!(!report.has_failures());
        report.checks.push(Check {
            name: "y".to_string(),
            status: Status::Fail,
            message: "".to_string(),
        });
        assert!(report.has_failures());
    }
}
