use std::fs;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Write `contents` to `path` atomically: write to a same-directory temp
/// file, then rename over the final path. Shared by wiki (per-fact files)
/// and the bank index (index.md) — the two places Milestone 1 needs it.
pub fn write(path: &Path, contents: &str) -> io::Result<()> {
    write_bytes(path, contents.as_bytes())
}

/// Same tmp-then-rename discipline as `write`, for binary content (e.g.
/// `archive.rs`'s gzip output, which isn't valid UTF-8 to write via
/// `fs::write(&str)`). `write` is a thin wrapper over this now — one real
/// atomic-write implementation, not two.
pub fn write_bytes(path: &Path, contents: &[u8]) -> io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp_path = dir.join(format!(".{file_name}.tmp-{}-{}", std::process::id(), nanos));
    fs::write(&tmp_path, contents)?;
    fs::rename(&tmp_path, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_creates_final_file_with_no_leftover_temp() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("index.md");
        write(&target, "hello").unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "hello");
        let leftovers: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn write_overwrites_existing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("index.md");
        write(&target, "v1").unwrap();
        write(&target, "v2").unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "v2");
    }
}
