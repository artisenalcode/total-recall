use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// A held lease lock on a bank directory. Dropping it releases the lock.
pub struct LockGuard {
    path: PathBuf,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum LockError {
    /// Someone else holds a live lease. Fail-fast policy: the caller
    /// decides whether to retry, not this module.
    Held {
        pid: u32,
    },
    Io(String),
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::Held { pid } => write!(f, "bank is locked by pid {pid}"),
            LockError::Io(msg) => write!(f, "lock io error: {msg}"),
        }
    }
}

/// Acquire the lease lock in `bank_dir`, reclaiming it first if the
/// existing lease's holder process is no longer alive. Fail-fast (no
/// retry/wait loop) if a live process holds it.
pub fn acquire(bank_dir: &Path) -> Result<LockGuard, LockError> {
    let lock_path = bank_dir.join(".lock");

    match try_create(&lock_path) {
        Ok(()) => return Ok(LockGuard { path: lock_path }),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(LockError::Io(e.to_string())),
    }

    let existing = fs::read_to_string(&lock_path).map_err(|e| LockError::Io(e.to_string()))?;
    let pid = parse_pid(&existing);

    match pid {
        Some(pid) if is_pid_alive(pid) => Err(LockError::Held { pid }),
        _ => {
            // Stale lease (dead PID, or unparseable content): reclaim it.
            fs::remove_file(&lock_path).map_err(|e| LockError::Io(e.to_string()))?;
            try_create(&lock_path).map_err(|e| LockError::Io(e.to_string()))?;
            Ok(LockGuard { path: lock_path })
        }
    }
}

fn try_create(lock_path: &Path) -> std::io::Result<()> {
    let mut file: File = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(lock_path)?;
    let pid = std::process::id();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    write!(file, "pid={pid}\nacquired_at={now}\n")
}

pub(crate) fn parse_pid(contents: &str) -> Option<u32> {
    contents
        .lines()
        .find_map(|line| line.strip_prefix("pid="))
        .and_then(|s| s.trim().parse().ok())
}

/// Linux-only liveness check (this project's target environment) via
/// /proc/<pid>. Not portable to macOS/Windows — deliberately not
/// abstracted yet per this session's "no premature abstraction" call.
/// `pub(crate)` (not private) so `doctor.rs` can report lock staleness
/// read-only, without calling `acquire` (which would mutate).
pub(crate) fn is_pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_on_fresh_bank_succeeds_and_writes_our_pid() {
        let tmp = tempfile::tempdir().unwrap();
        let guard = acquire(tmp.path()).unwrap();
        let contents = fs::read_to_string(tmp.path().join(".lock")).unwrap();
        assert_eq!(parse_pid(&contents), Some(std::process::id()));
        drop(guard);
    }

    #[test]
    fn release_removes_the_lease_file() {
        let tmp = tempfile::tempdir().unwrap();
        let guard = acquire(tmp.path()).unwrap();
        drop(guard);
        assert!(!tmp.path().join(".lock").exists());
    }

    #[test]
    fn acquiring_a_live_held_lock_fails_fast() {
        let tmp = tempfile::tempdir().unwrap();
        // pid 1 (init) is always alive on Linux and is never this test process.
        fs::write(tmp.path().join(".lock"), "pid=1\nacquired_at=0\n").unwrap();
        let result = acquire(tmp.path());
        assert_eq!(result.err(), Some(LockError::Held { pid: 1 }));
    }

    #[test]
    fn acquiring_a_stale_dead_pid_lock_reclaims_it() {
        let tmp = tempfile::tempdir().unwrap();
        // Extremely unlikely to be a live pid.
        fs::write(tmp.path().join(".lock"), "pid=999999999\nacquired_at=0\n").unwrap();
        let guard = acquire(tmp.path()).expect("stale lock should be reclaimed");
        let contents = fs::read_to_string(tmp.path().join(".lock")).unwrap();
        assert_eq!(parse_pid(&contents), Some(std::process::id()));
        drop(guard);
    }

    #[test]
    fn reacquire_after_release_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        drop(acquire(tmp.path()).unwrap());
        assert!(acquire(tmp.path()).is_ok());
    }
}
