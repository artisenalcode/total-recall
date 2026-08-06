//! Black-box tests against the actual compiled `trm` binary — the real
//! deployable artifact real callers (session_to_trm.py, a harness
//! session typing commands directly) depend on. Unit tests in
//! `src/*.rs` cover the internal logic; these cover the thing an actual
//! consumer depends on: argv/stdin in, real bank on disk out. Same
//! pattern as squishi's own `tests/cli.rs`.

use std::io::Write;
use std::process::{Command, Stdio};

/// A scratch `MF_DATA_ROOT` per test so nothing touches the real `~/.trm`.
fn scratch_data_root() -> tempfile::TempDir {
    tempfile::tempdir().expect("failed to create scratch data root")
}

fn run_trm(args: &[&str], data_root: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_trm"))
        .args(args)
        .env("MF_DATA_ROOT", data_root)
        .output()
        .expect("failed to run trm binary")
}

fn run_trm_with_stdin(
    args: &[&str],
    data_root: &std::path::Path,
    stdin_content: &str,
) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_trm"))
        .args(args)
        .env("MF_DATA_ROOT", data_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn trm binary");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin_content.as_bytes())
        .expect("failed to write to stdin");

    child.wait_with_output().expect("failed to wait on child")
}

#[test]
fn retain_reads_from_stdin_when_no_argument_given() {
    let data_root = scratch_data_root();
    let output = run_trm_with_stdin(
        &["-p", "test-bank", "retain"],
        data_root.path(),
        "a fact piped in over stdin, no positional argument at all",
    );

    assert!(
        output.status.success(),
        "trm retain exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("retained:"));

    let recall_output = run_trm(
        &["-p", "test-bank", "recall", "fact piped in over stdin"],
        data_root.path(),
    );
    let recall_stdout = String::from_utf8_lossy(&recall_output.stdout);
    assert!(
        recall_stdout.contains("piped in over stdin"),
        "stdin-retained content should be findable via recall, got: {recall_stdout}"
    );
}

#[test]
fn retain_with_positional_argument_still_works() {
    let data_root = scratch_data_root();
    let output = run_trm(
        &[
            "-p",
            "test-bank",
            "retain",
            "a fact via positional argument",
        ],
        data_root.path(),
    );

    assert!(
        output.status.success(),
        "trm retain exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("retained:"));
}

#[test]
fn stage_reads_from_stdin_when_no_argument_given() {
    let data_root = scratch_data_root();
    let output = run_trm_with_stdin(
        &["-p", "test-bank", "stage", "--reason", "a real reason"],
        data_root.path(),
        "raw content piped in over stdin for a handover",
    );

    assert!(
        output.status.success(),
        "trm stage exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("staged:"));
}

/// No positional argument AND stdin is a real pipe with nothing written
/// to it (immediate EOF, not a terminal) — a real, legitimate "empty
/// content" case, distinct from the terminal-refusal path (which this
/// black-box test can't easily simulate without a real pty). Confirms
/// the empty-stdin path doesn't hang or crash; `retain`'s own downstream
/// behavior on empty content is out of scope here.
#[test]
fn retain_with_empty_piped_stdin_does_not_hang() {
    let data_root = scratch_data_root();
    let output = run_trm_with_stdin(&["-p", "test-bank", "retain"], data_root.path(), "");
    // Whatever retain decides to do with empty content, the process must
    // exit (not hang waiting for more stdin) — that's the real contract
    // under test.
    assert!(output.status.code().is_some());
}
