//! CCR (content-addressed recovery): a short-lived, bank-agnostic store for the original bytes behind a lossy compression, at `~/.trm/ccr/`.
//! Object identity is its SHA-256 hash (`ccr_<hash16>`), sharded two levels deep with a JSON sidecar tracking `last_seen` for `gc`'s age cutoff.
//! Content-addressing gives free dedup. `put`/`get` need no lock (idempotent/read-only against a never-mutated file); `gc` deletes, so it locks.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const HANDLE_PREFIX: &str = "ccr_";
const HASH_HEX_LEN: usize = 16;

#[derive(Debug, PartialEq, Eq)]
pub enum CcrError {
    /// Rejected before any filesystem access, distinct from a real miss.
    MalformedHandle(String),
    /// Well-formed handle, but no live object exists for it.
    NotFound,
    Io(String),
}

impl std::fmt::Display for CcrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CcrError::MalformedHandle(h) => write!(f, "malformed CCR handle: {h:?}"),
            CcrError::NotFound => {
                write!(
                    f,
                    "no object found for this handle (evicted or never stored)"
                )
            }
            CcrError::Io(msg) => write!(f, "ccr io error: {msg}"),
        }
    }
}

impl From<io::Error> for CcrError {
    fn from(e: io::Error) -> Self {
        CcrError::Io(e.to_string())
    }
}

impl From<serde_json::Error> for CcrError {
    fn from(e: serde_json::Error) -> Self {
        CcrError::Io(e.to_string())
    }
}

/// Compute the handle for `bytes` without touching the filesystem.
pub fn handle_for(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    format!("{HANDLE_PREFIX}{}", &hex[..HASH_HEX_LEN])
}

fn parse_handle(handle: &str) -> Result<&str, CcrError> {
    let hash = handle
        .strip_prefix(HANDLE_PREFIX)
        .ok_or_else(|| CcrError::MalformedHandle(handle.to_string()))?;
    let well_formed = hash.len() == HASH_HEX_LEN
        && hash
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
    if !well_formed {
        return Err(CcrError::MalformedHandle(handle.to_string()));
    }
    Ok(hash)
}

fn object_path(ccr_root: &Path, hash: &str) -> PathBuf {
    ccr_root
        .join("objects")
        .join(&hash[0..2])
        .join(&hash[2..4])
        .join(format!("{hash}.bin"))
}

/// `meta_path`, but takes and validates a full handle -- lets other modules' tests backdate `last_seen` directly.
#[cfg(test)]
pub(crate) fn meta_path_for(ccr_root: &Path, handle: &str) -> Option<PathBuf> {
    parse_handle(handle)
        .ok()
        .map(|hash| meta_path(ccr_root, hash))
}

fn meta_path(ccr_root: &Path, hash: &str) -> PathBuf {
    ccr_root
        .join("objects")
        .join(&hash[0..2])
        .join(&hash[2..4])
        .join(format!("{hash}.meta.json"))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Store `bytes`, returning its handle. Idempotent, and verifies its own write before returning success.
pub fn put(ccr_root: &Path, bytes: &[u8], kind: Option<&str>) -> Result<String, CcrError> {
    let handle = handle_for(bytes);
    let hash = parse_handle(&handle).expect("handle_for always produces a well-formed handle");
    let obj_path = object_path(ccr_root, hash);
    let meta = meta_path(ccr_root, hash);

    if obj_path.exists() {
        touch_last_seen(&meta)?;
        return Ok(handle);
    }

    crate::atomic::write_bytes(&obj_path, bytes)?;

    // Read the just-written file back, not the in-memory buffer, so a corrupt-on-disk write is actually caught.
    let written = std::fs::read(&obj_path)?;
    if written != bytes {
        let _ = std::fs::remove_file(&obj_path);
        return Err(CcrError::Io(format!(
            "write verification failed for handle {handle}: on-disk content didn't match"
        )));
    }

    let now = now_secs();
    let sidecar = json!({
        "created_at": now,
        "last_seen": now,
        "size": bytes.len(),
        "kind": kind,
    });
    crate::atomic::write(&meta, &serde_json::to_string(&sidecar)?)?;

    Ok(handle)
}

fn touch_last_seen(meta_path: &Path) -> Result<(), CcrError> {
    let contents = std::fs::read_to_string(meta_path)?;
    let mut value: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|e| CcrError::Io(format!("corrupt sidecar {}: {e}", meta_path.display())))?;
    value["last_seen"] = json!(now_secs());
    crate::atomic::write(meta_path, &serde_json::to_string(&value)?)?;
    Ok(())
}

/// Recover the exact original bytes for `handle`, bumping `last_seen` on a hit. Malformed handles never probe the object store.
pub fn get(ccr_root: &Path, handle: &str) -> Result<Vec<u8>, CcrError> {
    let hash = parse_handle(handle)?;
    let obj_path = object_path(ccr_root, hash);
    if !obj_path.exists() {
        return Err(CcrError::NotFound);
    }
    let bytes = std::fs::read(&obj_path)?;
    // Best-effort -- a sidecar touch failing must never fail the recovery itself.
    let _ = touch_last_seen(&meta_path(ccr_root, hash));
    Ok(bytes)
}

#[derive(Debug, Default, PartialEq)]
pub struct CcrStats {
    pub entry_count: usize,
    pub total_bytes: u64,
    pub oldest_last_seen: Option<u64>,
}

pub fn stats(ccr_root: &Path) -> CcrStats {
    let mut out = CcrStats::default();
    for (_, meta) in walk_objects(ccr_root) {
        out.entry_count += 1;
        out.total_bytes += meta.size;
        out.oldest_last_seen = Some(match out.oldest_last_seen {
            Some(existing) => existing.min(meta.last_seen),
            None => meta.last_seen,
        });
    }
    out
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct GcReport {
    pub removed: usize,
    pub bytes_freed: u64,
}

/// Evict entries older than `max_age`, then -- if still over `max_bytes` -- evict oldest-by-`last_seen` until under the cap.
pub fn gc(ccr_root: &Path, max_age: Duration, max_bytes: u64) -> Result<GcReport, CcrError> {
    std::fs::create_dir_all(ccr_root)?;
    let _guard = crate::lock::acquire(ccr_root).map_err(|e| CcrError::Io(e.to_string()))?;

    let mut report = GcReport::default();
    let now = now_secs();
    let max_age_secs = max_age.as_secs();

    let mut entries = walk_objects(ccr_root);
    entries.retain(|(paths, meta)| {
        let age = now.saturating_sub(meta.last_seen);
        if age >= max_age_secs {
            report.removed += 1;
            report.bytes_freed += meta.size;
            let _ = std::fs::remove_file(&paths.0);
            let _ = std::fs::remove_file(&paths.1);
            false
        } else {
            true
        }
    });

    let mut total_bytes: u64 = entries.iter().map(|(_, m)| m.size).sum();
    if total_bytes > max_bytes {
        entries.sort_by_key(|(_, m)| m.last_seen);
        for (paths, meta) in entries {
            if total_bytes <= max_bytes {
                break;
            }
            report.removed += 1;
            report.bytes_freed += meta.size;
            total_bytes = total_bytes.saturating_sub(meta.size);
            let _ = std::fs::remove_file(&paths.0);
            let _ = std::fs::remove_file(&paths.1);
        }
    }

    Ok(report)
}

struct ObjectMeta {
    last_seen: u64,
    size: u64,
}

/// Walk every stored object under `ccr_root`. A sidecar that's missing or corrupt is skipped, not a panic.
fn walk_objects(ccr_root: &Path) -> Vec<((PathBuf, PathBuf), ObjectMeta)> {
    let objects_dir = ccr_root.join("objects");
    let mut out = Vec::new();
    let Ok(top) = std::fs::read_dir(&objects_dir) else {
        return out;
    };
    for shard1 in top.flatten() {
        let Ok(inner) = std::fs::read_dir(shard1.path()) else {
            continue;
        };
        for shard2 in inner.flatten() {
            let Ok(files) = std::fs::read_dir(shard2.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                    continue;
                }
                // "<hash>.bin" -> "<hash>" -> "<hash>.meta.json".
                let meta_path = path.with_extension("").with_extension("meta.json");
                let Ok(contents) = std::fs::read_to_string(&meta_path) else {
                    continue;
                };
                let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) else {
                    continue;
                };
                let (Some(last_seen), Some(size)) = (
                    value.get("last_seen").and_then(|v| v.as_u64()),
                    value.get("size").and_then(|v| v.as_u64()),
                ) else {
                    continue;
                };
                out.push(((path, meta_path), ObjectMeta { last_seen, size }));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ccr_root() -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("ccr");
        (tmp, root)
    }

    /// Backdates a stored handle's `last_seen` so real time doesn't need to pass to exercise `gc`'s cutoff logic.
    fn backdate(root: &Path, handle: &str, seconds_ago: u64) {
        let hash = handle.strip_prefix(HANDLE_PREFIX).unwrap();
        let meta = meta_path(root, hash);
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&meta).unwrap()).unwrap();
        value["last_seen"] = json!(now_secs().saturating_sub(seconds_ago));
        std::fs::write(&meta, serde_json::to_string(&value).unwrap()).unwrap();
    }

    #[test]
    fn put_then_get_roundtrips_bytes_exactly() {
        let (_tmp, root) = ccr_root();
        let original = b"a real tool result\nwith multiple lines\n";
        let handle = put(&root, original, Some("log")).unwrap();
        let recovered = get(&root, &handle).unwrap();
        assert_eq!(recovered, original);
    }

    /// Binary-safety: a raw tool result can legitimately contain non-UTF-8 bytes; CCR must not assume text.
    #[test]
    fn put_then_get_roundtrips_non_utf8_bytes_exactly() {
        let (_tmp, root) = ccr_root();
        let original: &[u8] = &[0, 159, 146, 150, 0, 255, 254, b'x', 0];
        let handle = put(&root, original, None).unwrap();
        let recovered = get(&root, &handle).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn identical_content_put_twice_returns_the_same_handle() {
        let (_tmp, root) = ccr_root();
        let content = b"same bytes both times";
        let first = put(&root, content, None).unwrap();
        let second = put(&root, content, None).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn identical_content_put_twice_stores_only_one_object_file() {
        let (_tmp, root) = ccr_root();
        let content = b"deduped content";
        put(&root, content, None).unwrap();
        put(&root, content, None).unwrap();

        let object_files: Vec<_> = walk_objects(&root);
        assert_eq!(
            object_files.len(),
            1,
            "expected exactly one stored object for identical content, got {}",
            object_files.len()
        );
    }

    #[test]
    fn get_on_unknown_but_well_formed_handle_returns_not_found() {
        let (_tmp, root) = ccr_root();
        let handle = "ccr_0123456789abcdef";
        assert_eq!(get(&root, handle), Err(CcrError::NotFound));
    }

    #[test]
    fn get_rejects_a_handle_with_the_wrong_prefix_before_touching_the_filesystem() {
        let (_tmp, root) = ccr_root();
        let result = get(&root, "sha_0123456789abcdef");
        assert!(matches!(result, Err(CcrError::MalformedHandle(_))));
        assert!(
            !root.exists(),
            "malformed handle must never create ccr_root"
        );
    }

    #[test]
    fn get_rejects_a_handle_with_the_wrong_length() {
        let (_tmp, root) = ccr_root();
        assert!(matches!(
            get(&root, "ccr_abc"),
            Err(CcrError::MalformedHandle(_))
        ));
    }

    #[test]
    fn get_rejects_a_handle_with_uppercase_hex() {
        let (_tmp, root) = ccr_root();
        assert!(matches!(
            get(&root, "ccr_0123456789ABCDEF"),
            Err(CcrError::MalformedHandle(_))
        ));
    }

    #[test]
    fn stats_reports_entry_count_and_total_bytes() {
        let (_tmp, root) = ccr_root();
        put(&root, b"one", None).unwrap();
        put(&root, b"two-two", None).unwrap();

        let s = stats(&root);
        assert_eq!(s.entry_count, 2);
        assert_eq!(s.total_bytes, 3 + 7);
    }

    #[test]
    fn gc_removes_entries_older_than_max_age() {
        let (_tmp, root) = ccr_root();
        let old = put(&root, b"stale entry", None).unwrap();
        let fresh = put(&root, b"fresh entry", None).unwrap();
        backdate(&root, &old, 10 * 24 * 3600); // 10 days old

        let report = gc(&root, Duration::from_secs(7 * 24 * 3600), u64::MAX).unwrap();

        assert_eq!(report.removed, 1);
        assert_eq!(get(&root, &old), Err(CcrError::NotFound));
        assert!(get(&root, &fresh).is_ok());
    }

    #[test]
    fn gc_respects_max_bytes_cap_and_evicts_oldest_first() {
        let (_tmp, root) = ccr_root();
        let oldest = put(&root, b"oldest-blob", None).unwrap();
        let middle = put(&root, b"middle-blob", None).unwrap();
        let newest = put(&root, b"newest-blob", None).unwrap();
        backdate(&root, &oldest, 300);
        backdate(&root, &middle, 200);
        backdate(&root, &newest, 100);

        // Each blob is 11 bytes; cap at 15 forces evicting the two oldest.
        let report = gc(&root, Duration::from_secs(u64::MAX / 2), 15).unwrap();

        assert_eq!(report.removed, 2);
        assert_eq!(get(&root, &oldest), Err(CcrError::NotFound));
        assert_eq!(get(&root, &middle), Err(CcrError::NotFound));
        assert!(get(&root, &newest).is_ok());
    }

    #[test]
    fn get_between_two_gc_runs_extends_life_past_the_first_cutoff() {
        let (_tmp, root) = ccr_root();
        let handle = put(&root, b"touched entry", None).unwrap();
        backdate(&root, &handle, 5 * 24 * 3600); // 5 days old

        // First gc at a 7-day cutoff: not old enough yet, survives.
        gc(&root, Duration::from_secs(7 * 24 * 3600), u64::MAX).unwrap();
        assert!(
            get(&root, &handle).is_ok(),
            "entry should survive the first gc"
        );

        // get() reset the clock -- backdate again to simulate 5 more days, still only 5 days old, not 10.
        backdate(&root, &handle, 5 * 24 * 3600);
        gc(&root, Duration::from_secs(7 * 24 * 3600), u64::MAX).unwrap();
        assert!(
            get(&root, &handle).is_ok(),
            "a get() between gc runs should have reset the age clock"
        );
    }

    #[test]
    fn handle_for_is_deterministic_and_content_addressed() {
        let a = handle_for(b"same content");
        let b = handle_for(b"same content");
        let c = handle_for(b"different content");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.starts_with("ccr_"));
        assert_eq!(a.len(), "ccr_".len() + 16);
    }
}
