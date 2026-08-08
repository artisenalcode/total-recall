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

/// `ingest-session` shells out to the real `squishi` binary for
/// extraction+compression (see ingest.rs's module doc) -- it's a hard
/// dependency, not best-effort, so there's no in-process fallback to
/// test instead. `squishi` lives in a sibling private repo; CI checks it
/// out and builds it via a fine-grained PAT (`SQUISHI_CHECKOUT_PAT`, see
/// this repo's CI workflow) when that secret is set. Locally, or on a
/// fork/PR without the secret, it may not be on PATH -- skip gracefully
/// rather than fail the whole suite on an environment gap that isn't a
/// code regression; same resilience posture this codebase already
/// applies to model/embedding availability elsewhere.
fn squishi_on_path() -> bool {
    Command::new("squishi")
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn run_trm(args: &[&str], data_root: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_trm"))
        .args(args)
        .env("MF_DATA_ROOT", data_root)
        .output()
        .expect("failed to run trm binary")
}

fn run_trm_with_envs(
    args: &[&str],
    data_root: &std::path::Path,
    envs: &[(&str, &str)],
) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_trm"));
    cmd.args(args).env("MF_DATA_ROOT", data_root);
    for (key, value) in envs {
        cmd.env(key, value);
    }
    cmd.output().expect("failed to run trm binary")
}

/// Backdate a file's mtime well past `RECENT_MTIME_GUARD_SECS` (10 min) --
/// std has no mtime setter without a extra crate, and this is the one
/// test that needs one, so shell out to the real `touch` instead of
/// adding a dependency for a single call site.
fn backdate_mtime(path: &std::path::Path) {
    let status = Command::new("touch")
        .arg("-d")
        .arg("2020-01-01T00:00:00")
        .arg(path)
        .status()
        .expect("failed to run touch");
    assert!(status.success(), "touch -d failed for {}", path.display());
}

fn run_trm_from_cwd(
    args: &[&str],
    data_root: &std::path::Path,
    invocation_cwd: &std::path::Path,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_trm"))
        .args(args)
        .env("MF_DATA_ROOT", data_root)
        .current_dir(invocation_cwd)
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

#[test]
fn complete_handover_reads_from_stdin_when_no_argument_given() {
    let data_root = scratch_data_root();
    let stage_output = run_trm_with_stdin(
        &["-p", "test-bank", "stage", "--reason", "a real reason"],
        data_root.path(),
        "raw content piped in over stdin for a handover",
    );
    let job_id = String::from_utf8_lossy(&stage_output.stdout)
        .trim()
        .strip_prefix("staged: ")
        .expect("stage should print a job id")
        .to_string();

    // A result large enough that passing it as a positional argument
    // would be exactly the fragile pattern retain/stage already moved
    // away from -- the real motivation for this fix.
    let large_result = format!("# A synthesized page\n\n{}", "real content ".repeat(2000));
    let output = run_trm_with_stdin(
        &["-p", "test-bank", "complete-handover", &job_id],
        data_root.path(),
        &large_result,
    );

    assert!(
        output.status.success(),
        "trm complete-handover exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("completed:"));
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

/// The one real behavioral requirement ported from session_to_trm.py:
/// `ingest-session` must stage into the bank resolved from the
/// SESSION's own cwd (parsed out of squishi's digest output), not
/// whatever directory `trm` itself was invoked from. Sets up a fake
/// "session repo" (dir A, with `.trm-bank` = a known marker) distinct
/// from the real invocation cwd (dir B, no bank markers at all — would
/// resolve to "global" if cwd resolution were used by mistake), and
/// confirms the staged content lands in dir A's bank, not "global".
#[test]
fn ingest_session_stages_into_the_bank_resolved_from_the_session_s_own_cwd() {
    if !squishi_on_path() {
        eprintln!(
            "skipping ingest_session_stages_into_the_bank_resolved_from_the_session_s_own_cwd: \
             squishi not on PATH (see squishi_on_path's doc comment)"
        );
        return;
    }
    let data_root = scratch_data_root();
    let session_repo = tempfile::tempdir().expect("failed to create session repo dir");
    let invocation_cwd = tempfile::tempdir().expect("failed to create invocation cwd dir");

    std::fs::create_dir(session_repo.path().join(".git")).unwrap();
    std::fs::write(
        session_repo.path().join(".trm-bank"),
        "session-bank-marker\n",
    )
    .unwrap();

    let session_cwd_json = serde_json::to_string(session_repo.path().to_str().unwrap()).unwrap();
    let user_line = format!(
        r#"{{"type":"user","sessionId":"sess-cross-cwd","cwd":{session_cwd_json},"timestamp":"t1","message":{{"role":"user","content":[{{"type":"text","text":"a real question about the project"}}]}}}}"#
    );
    let assistant_line = format!(
        r#"{{"type":"assistant","sessionId":"sess-cross-cwd","cwd":{session_cwd_json},"timestamp":"t2","message":{{"role":"assistant","content":[{{"type":"text","text":"a real answer with enough content to be worth staging"}}]}}}}"#
    );
    let transcript = format!("{user_line}\n{assistant_line}\n");
    let transcript_path = data_root.path().join("fixture-session.jsonl");
    std::fs::write(&transcript_path, transcript).unwrap();

    let output = run_trm_from_cwd(
        &["ingest-session", transcript_path.to_str().unwrap()],
        data_root.path(),
        invocation_cwd.path(),
    );

    assert!(
        output.status.success(),
        "trm ingest-session exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("staged:"));

    // Confirm it landed in the SESSION's bank, not "global" (which is
    // what invocation_cwd, having no bank markers, would resolve to).
    let recall = run_trm(&["-p", "session-bank-marker", "pending"], data_root.path());
    let recall_stdout = String::from_utf8_lossy(&recall.stdout);
    assert!(
        !recall_stdout.trim().is_empty() && !recall_stdout.contains("no pending"),
        "expected a pending handover in session-bank-marker, got: {recall_stdout}"
    );

    let global_pending = run_trm(&["-p", "global", "pending"], data_root.path());
    let global_stdout = String::from_utf8_lossy(&global_pending.stdout);
    assert!(
        global_stdout.trim().is_empty() || global_stdout.to_lowercase().contains("no pending"),
        "the global bank should NOT have received this session's content, got: {global_stdout}"
    );
}

/// ADR-0006 Phase 1's real point: `--archive-after` stages exactly as
/// today, then gzip-archives the transcript into the resolved bank's
/// `sessions/` tier and removes the original from disk.
#[test]
fn ingest_session_archive_after_archives_and_removes_the_source_transcript() {
    if !squishi_on_path() {
        eprintln!(
            "skipping ingest_session_archive_after_archives_and_removes_the_source_transcript: \
             squishi not on PATH (see squishi_on_path's doc comment)"
        );
        return;
    }
    let data_root = scratch_data_root();
    let session_repo = tempfile::tempdir().expect("failed to create session repo dir");
    std::fs::create_dir(session_repo.path().join(".git")).unwrap();
    std::fs::write(session_repo.path().join(".trm-bank"), "archive-bank\n").unwrap();

    let session_cwd_json = serde_json::to_string(session_repo.path().to_str().unwrap()).unwrap();
    let user_line = format!(
        r#"{{"type":"user","sessionId":"sess-archive-1","cwd":{session_cwd_json},"timestamp":"t1","message":{{"role":"user","content":[{{"type":"text","text":"a real question about the project"}}]}}}}"#
    );
    let assistant_line = format!(
        r#"{{"type":"assistant","sessionId":"sess-archive-1","cwd":{session_cwd_json},"timestamp":"t2","message":{{"role":"assistant","content":[{{"type":"text","text":"a real answer with enough content to be worth staging"}}]}}}}"#
    );
    let transcript = format!("{user_line}\n{assistant_line}\n");
    let transcript_dir = tempfile::tempdir().expect("failed to create transcript dir");
    let transcript_path = transcript_dir.path().join("sess-archive-1.jsonl");
    std::fs::write(&transcript_path, &transcript).unwrap();

    let output = run_trm(
        &[
            "ingest-session",
            transcript_path.to_str().unwrap(),
            "--archive-after",
        ],
        data_root.path(),
    );

    assert!(
        output.status.success(),
        "trm ingest-session --archive-after exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("(archived)"));

    assert!(
        !transcript_path.exists(),
        "source transcript should be removed after archiving"
    );

    let archived_path = data_root
        .path()
        .join("banks")
        .join("archive-bank")
        .join("sessions")
        .join("sess-archive-1.jsonl.gz");
    assert!(
        archived_path.exists(),
        "expected archived transcript at {}",
        archived_path.display()
    );
}

/// A `--trigger` whose hook is disabled in `trm.json` must skip entirely
/// -- no staging, no archiving, not even an error (a disabled hook firing
/// should look like nothing happened).
#[test]
fn ingest_session_skips_entirely_when_its_trigger_is_disabled_in_config() {
    if !squishi_on_path() {
        eprintln!(
            "skipping ingest_session_skips_entirely_when_its_trigger_is_disabled_in_config: \
             squishi not on PATH (see squishi_on_path's doc comment)"
        );
        return;
    }
    let data_root = scratch_data_root();
    std::fs::write(
        data_root.path().join("trm.json"),
        r#"{"hooks": {"pre_compact": {"enabled": false}}}"#,
    )
    .unwrap();

    let session_repo = tempfile::tempdir().expect("failed to create session repo dir");
    std::fs::create_dir(session_repo.path().join(".git")).unwrap();
    std::fs::write(session_repo.path().join(".trm-bank"), "disabled-bank\n").unwrap();

    let session_cwd_json = serde_json::to_string(session_repo.path().to_str().unwrap()).unwrap();
    let user_line = format!(
        r#"{{"type":"user","sessionId":"sess-disabled-1","cwd":{session_cwd_json},"timestamp":"t1","message":{{"role":"user","content":[{{"type":"text","text":"a real question about the project"}}]}}}}"#
    );
    let transcript_dir = tempfile::tempdir().expect("failed to create transcript dir");
    let transcript_path = transcript_dir.path().join("sess-disabled-1.jsonl");
    std::fs::write(&transcript_path, format!("{user_line}\n")).unwrap();

    let output = run_trm(
        &[
            "ingest-session",
            transcript_path.to_str().unwrap(),
            "--trigger",
            "precompact",
        ],
        data_root.path(),
    );

    assert!(
        output.status.success(),
        "a disabled trigger should exit 0, not fail: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("skipped:"));
    assert!(
        transcript_path.exists(),
        "a skipped ingest must not touch the source transcript"
    );

    let pending = run_trm(&["-p", "disabled-bank", "pending"], data_root.path());
    let pending_stdout = String::from_utf8_lossy(&pending.stdout);
    assert!(
        pending_stdout.trim().is_empty() || pending_stdout.to_lowercase().contains("no pending"),
        "a skipped ingest must not stage anything, got: {pending_stdout}"
    );
}

/// The plan's key `--all` check: three transcripts in a fake
/// `~/.claude/projects/`-shaped tree -- one already archived (checkpoint
/// says so), one new/eligible, one belonging to the "currently running"
/// session (`$CLAUDE_CODE_SESSION_ID`). Exactly the new one should get
/// archived; the other two must be left completely untouched.
#[test]
fn ingest_sessions_all_archives_only_the_one_eligible_transcript() {
    if !squishi_on_path() {
        eprintln!(
            "skipping ingest_sessions_all_archives_only_the_one_eligible_transcript: \
             squishi not on PATH (see squishi_on_path's doc comment)"
        );
        return;
    }
    let data_root = scratch_data_root();
    let session_repo = tempfile::tempdir().expect("failed to create session repo dir");
    std::fs::create_dir(session_repo.path().join(".git")).unwrap();
    std::fs::write(session_repo.path().join(".trm-bank"), "sweep-bank\n").unwrap();
    let session_cwd_json = serde_json::to_string(session_repo.path().to_str().unwrap()).unwrap();

    let transcript_line = |session_id: &str| {
        format!(
            r#"{{"type":"user","sessionId":"{session_id}","cwd":{session_cwd_json},"timestamp":"t1","message":{{"role":"user","content":[{{"type":"text","text":"a real question worth staging for {session_id}"}}]}}}}"#
        )
    };

    let projects_dir = tempfile::tempdir().expect("failed to create fake projects dir");
    let project_subdir = projects_dir.path().join("-fake-project");
    std::fs::create_dir_all(&project_subdir).unwrap();

    let already_path = project_subdir.join("sess-already.jsonl");
    let new_path = project_subdir.join("sess-new.jsonl");
    let current_path = project_subdir.join("sess-current.jsonl");
    std::fs::write(&already_path, transcript_line("sess-already")).unwrap();
    std::fs::write(&new_path, transcript_line("sess-new")).unwrap();
    std::fs::write(&current_path, transcript_line("sess-current")).unwrap();
    for path in [&already_path, &new_path, &current_path] {
        backdate_mtime(path);
    }

    // Pre-seed the checkpoint so sess-already reads as already archived --
    // matches session_checkpoint::save's own format, written directly so
    // this test doesn't depend on a prior `trm` run to set it up.
    let bank_dir = data_root.path().join("banks").join("sweep-bank");
    std::fs::create_dir_all(&bank_dir).unwrap();
    std::fs::write(
        bank_dir.join(".session-state.json"),
        r#"{"sess-already": {"archived": true}}"#,
    )
    .unwrap();

    let output = run_trm_with_envs(
        &["ingest-sessions", "--all"],
        data_root.path(),
        &[
            (
                "MF_CLAUDE_PROJECTS_DIR",
                projects_dir.path().to_str().unwrap(),
            ),
            ("CLAUDE_CODE_SESSION_ID", "sess-current"),
        ],
    );

    assert!(
        output.status.success(),
        "trm ingest-sessions --all exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("1 archived"),
        "expected exactly 1 archived, got: {stdout}"
    );
    assert!(
        stdout.contains("1 already archived"),
        "expected exactly 1 already-archived skip, got: {stdout}"
    );
    assert!(
        stdout.contains("1 current session"),
        "expected exactly 1 current-session skip, got: {stdout}"
    );

    assert!(
        already_path.exists(),
        "an already-archived session's source file must be left untouched"
    );
    assert!(
        current_path.exists(),
        "the current session's source file must never be touched"
    );
    assert!(
        !new_path.exists(),
        "the one eligible session's source file should be archived away"
    );

    let archived_gz = bank_dir.join("sessions").join("sess-new.jsonl.gz");
    assert!(
        archived_gz.exists(),
        "expected {} to exist",
        archived_gz.display()
    );
}
