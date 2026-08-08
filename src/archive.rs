//! Archives an original Claude Code session transcript into a bank's
//! `sessions/` tier (ADR-0006 Decision 3) once its content is safely
//! staged. The one truly hard-to-reverse operation in the whole
//! automated-staging feature — `src` is only ever removed after the
//! compressed copy has been read back and confirmed byte-identical, not
//! merely "the write call returned Ok". A write that silently truncated
//! (disk full mid-write, e.g.) must never take the only copy of a
//! session's transcript down with it.

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use std::io::{self, Read, Write};
use std::path::Path;

/// Gzip-compress `src` to `dest_gz` (atomic tmp-then-rename, via
/// `atomic::write_bytes`), verify the result decompresses back to exactly
/// `src`'s original bytes, then remove `src`. `src` is left untouched on
/// any failure along the way — compression error, write error, or a
/// verification mismatch.
pub fn archive_transcript(src: &Path, dest_gz: &Path) -> io::Result<()> {
    let original = std::fs::read(src)?;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&original)?;
    let compressed = encoder.finish()?;

    crate::atomic::write_bytes(dest_gz, &compressed)?;

    // Verify before deleting anything -- read the just-written file back
    // (not the in-memory `compressed` buffer) so a corrupt-on-disk write
    // that still returned Ok from `write_bytes` is actually caught.
    let written = std::fs::read(dest_gz)?;
    let mut decoder = GzDecoder::new(written.as_slice());
    let mut roundtripped = Vec::new();
    decoder.read_to_end(&mut roundtripped)?;

    if roundtripped != original {
        return Err(io::Error::other(format!(
            "archive verification failed for {}: decompressed content didn't match the original -- source left in place",
            src.display()
        )));
    }

    std::fs::remove_file(src)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archives_a_real_file_and_removes_the_source() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("session.jsonl");
        let original = "line one\nline two\nline three\n".repeat(50);
        std::fs::write(&src, &original).unwrap();

        let dest = tmp.path().join("archived").join("session.jsonl.gz");
        archive_transcript(&src, &dest).unwrap();

        assert!(dest.exists(), "compressed archive should exist");
        assert!(!src.exists(), "source should be removed after archiving");

        let compressed = std::fs::read(&dest).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn source_is_left_in_place_when_the_destination_cannot_be_written() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("session.jsonl");
        std::fs::write(&src, "some content").unwrap();

        // A destination path whose parent doesn't exist AND can't be
        // created (parent is itself a file, not a directory) forces
        // `atomic::write_bytes`'s `create_dir_all` to fail.
        let blocker = tmp.path().join("blocker-file");
        std::fs::write(&blocker, "im a file not a dir").unwrap();
        let dest = blocker.join("nested").join("session.jsonl.gz");

        let result = archive_transcript(&src, &dest);
        assert!(result.is_err());
        assert!(src.exists(), "source must survive a failed archive attempt");
    }

    #[test]
    fn empty_transcript_archives_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("empty.jsonl");
        std::fs::write(&src, "").unwrap();
        let dest = tmp.path().join("empty.jsonl.gz");

        archive_transcript(&src, &dest).unwrap();
        assert!(!src.exists());

        let compressed = std::fs::read(&dest).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).unwrap();
        assert!(decompressed.is_empty());
    }
}
