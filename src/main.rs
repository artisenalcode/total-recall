mod archive;
mod atomic;
mod bank;
mod concepts;
mod config;
mod curator;
mod doctor;
mod embed_cache;
mod embeddings;
mod handover;
mod ingest;
mod lock;
mod session_checkpoint;
mod wiki;
mod window;

use clap::{Parser, Subcommand};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "trm",
    about = "trm (total-recall) — canonical cross-harness memory CLI"
)]
struct Cli {
    /// Explicit bank id, overriding auto-resolution from cwd/git remote.
    #[arg(short = 'p', long, global = true)]
    bank: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Store a single fact directly into the resolved bank's wiki tier.
    /// Omit `content` to read from stdin instead — no shell-argument size
    /// limit, no quoting/escaping fragility for large content (unlike a
    /// positional argument, which Linux caps around 128KB per element).
    Retain { content: Option<String> },
    /// Case-insensitive substring search across the resolved bank's wiki tier.
    Recall { query: String },
    /// Stage raw content for a sub-agent handover (extraction), per ADR-0002.
    /// Omit `content` to read from stdin instead — same reasoning as `retain`.
    Stage {
        content: Option<String>,
        #[arg(long)]
        reason: String,
        /// Provenance label. "direct" (default) is trusted; anything else
        /// (e.g. "web-scrape") is flagged UNTRUSTED in the handover prompt.
        #[arg(long, default_value = "direct")]
        source: String,
    },
    /// Stage a persona-build handover from an external tool's payload
    /// manifest — the whole trm/persona boundary (ADR-0009): a caller
    /// hands trm one JSON file, trm owns every internal detail of how
    /// it becomes a pending job. Manifest shape:
    /// `{ slug, bank, description, source_label, self_build, artifacts: [paths] }`
    /// -- `artifacts` is untyped from trm's side, just "files worth a
    /// sub-agent reading" (dedup sidecars, cluster/lexicon output,
    /// whatever the caller produced). `--bank` (the global flag) is an
    /// override; the manifest's own `bank` field is used otherwise --
    /// no cwd-based resolution, since an external caller has no
    /// meaningful cwd relationship to a bank.
    StagePersona {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// List open handover jobs in the resolved bank, or every bank with --all.
    Pending {
        #[arg(long)]
        all: bool,
    },
    /// Print a pending job's exact rendered prompt — feed this verbatim to
    /// a sub-agent to complete the handover.
    PendingShow { job_id: String },
    /// Scan the resolved bank for duplicate-candidate wiki entries
    /// (word-overlap heuristic, no LLM) and stage Curation handovers
    /// for any pair above the threshold.
    CuratorScan {
        #[arg(long, default_value_t = 0.8)]
        threshold: f64,
    },
    /// Commit a completed handover's result (called by the harness after
    /// a sub-agent finished the judgment work `trm` couldn't do itself).
    /// Omit `result` to read from stdin instead -- same reasoning as
    /// `retain`/`stage`: a synthesized persona wiki page can run to tens
    /// of KB, well past comfortable argv territory.
    CompleteHandover {
        job_id: String,
        result: Option<String>,
    },
    /// Serve live usage docs, so the skill stub never goes stale.
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },
    /// One-off migration: recursively import every *.md file under
    /// `source` into a bank, preserving relative filenames as slugs
    /// (minus a `MEMORY.md` index file, which is skipped — it's an
    /// index, not a fact).
    Import {
        source: PathBuf,
        #[arg(long)]
        bank: String,
    },
    /// Self-diagnostics: data root/model-cache permissions, bank
    /// resolution, lock staleness, whether the embedder actually loads.
    /// Read-only by default; --fix reclaims a real stale lock, the one
    /// repairable thing in this surface today.
    Doctor {
        /// Skip the real embedder-load check (network-capable, slower).
        #[arg(long)]
        quick: bool,
        /// Reclaim a stale lock, if the resolved bank has one. The only
        /// mutation this command ever makes, and only with this flag.
        #[arg(long)]
        fix: bool,
        #[arg(long)]
        json: bool,
    },
    /// Sweep every transcript under `~/.claude/projects/` (`
    /// MF_CLAUDE_PROJECTS_DIR` overrides, for tests) not yet archived,
    /// staging+archiving each (ADR-0006 Phase 1). Resumable: a second run
    /// is a no-op over sessions already archived. Never touches the
    /// transcript belonging to `$CLAUDE_CODE_SESSION_ID` (the invoking
    /// process's own live session, if any) regardless of its age, and
    /// skips anything modified in the last 10 minutes as a secondary
    /// guard against archiving a file some other still-running session
    /// might append to next.
    IngestSessions {
        /// Required, not a default-true flag -- an explicit ask, not
        /// something that happens by just running the bare subcommand
        /// (there's no other mode today, but making it implicit now
        /// forecloses a future narrower selection flag reading as a
        /// silent behavior change).
        #[arg(long)]
        all: bool,
    },
    /// Extract + compress a Claude Code session transcript (via the real
    /// `squishi --session-digest`) and stage the result for handover.
    /// Rust port of `session_to_trm.py`. Stages into the bank resolved
    /// from the SESSION's own cwd (parsed from squishi's output), not
    /// the cwd `trm` itself is invoked from -- `-p/--bank` still
    /// overrides both if given explicitly.
    IngestSession {
        path: PathBuf,
        /// After staging succeeds, gzip-archive the transcript into the
        /// resolved bank's `sessions/` tier and remove the original from
        /// its source location. Only safe for a FINISHED session (ADR-0006
        /// Phase 1) -- never pass this for a still-live transcript. Cannot
        /// be combined with --since-checkpoint (see its own doc comment).
        #[arg(long)]
        archive_after: bool,
        /// Stage only the delta since this session's last staged line
        /// (ADR-0006 Phase 2), instead of the whole transcript -- the
        /// live, in-session counterpart to --archive-after's
        /// finished-session archival. Resolves the bank from a cheap
        /// peek at the transcript (no full digest needed just to find
        /// the checkpoint), loads that bank's saved `last_staged_line`,
        /// passes it to squishi as `--start-line`, and advances the
        /// checkpoint to the new `total_lines` on success. "Nothing new
        /// since last checkpoint" is a normal outcome (Skipped), not a
        /// failure. Cannot be combined with --archive-after: archiving is
        /// for a session that's finished, since-checkpoint is for one
        /// that's still running -- combining them risks archiving (and
        /// deleting) a transcript a live Claude Code process may still
        /// have open.
        #[arg(long)]
        since_checkpoint: bool,
        /// Which trigger is calling this. `manual` (default, this
        /// command's original behavior) is never gated by `trm.json`'s
        /// hook-enable flags; `precompact`/`sessionend` are -- a disabled
        /// hook exits 0 silently rather than staging anyway.
        #[arg(long, value_enum, default_value = "manual")]
        trigger: Trigger,
    },
}

/// Which caller is invoking `ingest-session` (ADR-0006 Phase 1). `manual`
/// is the pre-existing, ungated behavior; `precompact`/`sessionend` are
/// checked against `trm.json`'s `hooks.*.enabled` before staging anything.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
#[value(rename_all = "lowercase")]
enum Trigger {
    Precompact,
    Sessionend,
    Manual,
}

#[derive(Subcommand)]
enum SkillAction {
    Get { topic: String },
}

/// Live usage docs served by `trm skill get core` — the skill stub in
/// agents-brain just points here (agent-browser's pattern) rather than
/// embedding instructions that can drift from the installed binary.
const CORE_DOCS: &str = r#"# trm — canonical memory CLI

## Retain a fact

    trm retain "<content>"
    <content-source> | trm retain          # stdin, since 2026-08-07

Writes into the bank resolved from your current directory:
- explicit `-p/--bank <id>` if given
- else the git remote of the enclosing repo (owner-repo slug)
- else a stable hash of the enclosing repo's path (no remote configured)
- else "global", if you're not inside any git repo

`content` is optional — omit it to read from stdin instead. Prefer stdin
for anything large or programmatically generated: a positional argument
is a real, hard Linux limit (~128KB per argv element, `MAX_ARG_STRLEN`),
found the hard way when a caller (`session_to_trm.py`) needed its own
argv-length guard before this existed. Stdin has no such ceiling and
avoids shell quoting/escaping fragility on multi-line content.

Examples:

    trm retain "user prefers terse commit messages"          # -> bank for cwd's repo
    trm -p global retain "cross-project preference: X"        # -> forces the global bank
    cat large-note.md | trm retain                            # -> stdin, no argv limit

## Recall a fact

    trm recall "<query>"

Semantic ranked search (local embeddings, same model as curator-scan) —
replaced an earlier grep-based version entirely. Prints `<score>  <slug>:
<snippet>` for the top 5 matches scoring >= 0.3 cosine similarity, or "no
matches" if none. Same bank-resolution rules as retain.

Windowed since 2026-08-07 (ADR-0004): a long entry is split into
overlapping ~80-word windows before embedding, scored per window, best
score wins — a whole-file-as-one-vector approach used to hard-truncate
anything past ~211-261 words (found empirically; the original documented
miss was `ethan-evans`, whose real content sat past that point). Windowing
fixes recall against already-stored large entries; it doesn't fully solve
same-topic dilution within a window on its own (measured: a real point
diluted by ~150 words of a single unrelated topic still scores only
~0.28-0.30 regardless of window size) — `stage`'s concept pre-split below
is the write-time complement, keeping newly-staged entries small and
mostly-single-topic in the first place rather than relying on windowing
alone to recover from a large blob.

## Handover (extraction/curation trm can't do itself — see ADR-0002)

    trm stage "<raw content>" --reason "<why this needs judgment>" [--source direct]
    <content-source> | trm stage --reason "..." [--source direct]   # stdin, since 2026-08-07
    trm pending [--all]
    trm pending-show <job-id>
    trm complete-handover <job-id> "<result>"

`stage` writes to the raw tier and drops a pending marker. `--source`
labels provenance: "direct" (default) is trusted; anything else (e.g.
"web-scrape", once a web-scrape ingestion pathway exists) is flagged
UNTRUSTED CONTENT in the rendered prompt, so a sub-agent reading staged
external content knows to treat it as data, never as instructions.
`pending` lists open job ids in the resolved bank; `--all` lists across
every bank as `<bank>/<job-id>` (needed since not every pathway resolves
its bank the same way — code archaeology resolves from the target repo
path, not the invoking directory). `pending-show <job-id>` prints a job's
exact rendered prompt — feed this verbatim to a sub-agent.

**Content over ~2000 chars gets concept-pre-split automatically**
(ADR-0004, 2026-08-07): `stage` deterministically splits large/ambiguous
content into candidate concepts (sentence-level, near-duplicate-collapsed
— same algorithm `advisory/tools/dedupe_semantic.py` proved out) *before*
rendering the prompt. **If you're the sub-agent handling a handover whose
prompt lists numbered candidate concepts: judge each one, `trm retain
"<concept>"` individually for every one genuinely worth keeping — it's
now a single judged fact, no further handover needed for it — discard the
rest, then `trm complete-handover <job-id> "<N kept, M discarded>"` to
close the audit trail.** Do not `complete-handover` the whole raw blob as
a single result for split content — that reproduces the exact bug this
mechanism exists to prevent (one giant, barely-recallable entry). Small
content (no listed candidates in the prompt) is unaffected — handle it as
a normal single-result extraction.

**The shared handover-completion workflow** (one place, not reimplemented
per ingestion pathway): run `trm pending --all`; for each `<bank>/<job-id>`,
run `trm pending-show <job-id> -p <bank>` and spawn a sub-agent with that
exact content as its prompt; the sub-agent follows the candidate-concept
pattern above if the prompt has one, otherwise does its own extraction and
calls `trm complete-handover <job-id> "<result>" -p <bank>` directly. trm
never calls an LLM itself — the calling harness is responsible for the
judgment work. `complete-handover` writes its result into the wiki tier,
updates index.md, and clears the pending marker; the raw source stays
for audit, it isn't deleted.

## Curator scan (finds Curation-handover candidates)

    trm curator-scan [--threshold 0.8]

Local sentence-embedding cosine similarity (all-MiniLM-L6-v2 via ONNX,
downloaded once and cached at `~/.trm/models/`, no external API —
the same model mindforge's own `tools/dedupe_semantic.py` already uses
through Python's fastembed, this is the Rust-native equivalent) between
every pair of wiki entries — a candidate finder, not a judgment. Stages a
Curation handover per pair above threshold, same completion flow as
extraction. An exact/fully-contained-content pre-check runs first and is
always on regardless of threshold — zero tuning, catches genuine
copy-paste duplication before the embedding comparison even runs.

Threshold calibration (found empirically against the real 48-entry
mindforge bank, not assumed): 0.8 (default) surfaces a small, high-
precision set — correctly caught `casey-muratori`/`uncle-bob` (a
documented real disagreement between the two) and `scott-tolinski`/
`syntax` (Scott co-hosts the Syntax podcast). 0.75 broadens to ~20+
candidates, still mostly plausible but needing more manual judgment — one
entry (`dan-harrison`) showed up as a five-way "hub," which may mean
genuinely broad topical overlap or an unusually generic entry; worth a
human look, not assumed to be a false positive. Below ~0.6 the candidate
count grows into the hundreds and stops being trustworthy signal.
Predecessor note: an earlier word-overlap (Jaccard) heuristic was
replaced entirely by this embedding-based approach — it produced 435
false positives on this same bank from shared templated section-header
vocabulary at a low threshold, a weaker signal than the local embedding
model already available in this project's own toolchain.

## Doctor (self-diagnostics)

    trm doctor
    trm doctor --quick          # skip the real embedder-load check
    trm doctor --fix            # also reclaim a stale lock, if any
    trm doctor --json

Checks the resolved bank's real failure surface: data root writable,
bank resolution + whether it's been created yet (not created is a warn,
not a fail — banks are lazily created on first write), lock staleness
(read-only unless `--fix`), model cache reachable, and whether the
embedder actually loads for real (skipped under `--quick` — this one
check covers both `recall` and `stage()`'s concept-split viability, so
it's reported once, not twice). Exit code 0 unless a real `Fail` shows
up; warnings don't fail the run. `--fix` is the only thing this command
ever mutates, and only reclaims a lock already confirmed stale (dead
pid) — never a live one.

## Ingest a Claude Code session

    trm ingest-session <transcript.jsonl> [--archive-after] [--trigger manual|precompact|sessionend]
    trm ingest-sessions --all

Rust port of `session_to_trm.py` — extraction+compression now live in
squishi (`squishi --session-digest`, which this shells out to);
`ingest-session` owns the actual `stage` call, keeping squishi's own
"never stores/retrieves" boundary intact. Stages into the bank resolved
from the SESSION's own cwd (parsed out of squishi's digest output), not
whatever directory `trm ingest-session` itself is invoked from —
`-p/--bank` still overrides both if given explicitly. Fails loudly if
`squishi` isn't on PATH or the session has nothing to digest (empty).

**Archival (ADR-0006 Phase 1, for a FINISHED session only).**
`--archive-after` gzip-archives the transcript into the resolved bank's
`sessions/<session-id>.jsonl.gz` and removes the source once staging
succeeds — the source is only ever deleted after the compressed copy is
read back and confirmed byte-identical. `trm ingest-sessions --all`
sweeps every transcript under `~/.claude/projects/`
(`MF_CLAUDE_PROJECTS_DIR` overrides) not yet archived and does the same;
resumable (a second sweep is a no-op over what's already archived), and
never touches the transcript belonging to `$CLAUDE_CODE_SESSION_ID` (the
invoking process's own live session) or anything modified in the last 10
minutes. `--trigger` (default `manual`, ungated) marks who's calling —
`precompact`/`sessionend` are checked against `trm.json`'s
`hooks.*.enabled` before staging anything; a disabled trigger exits 0
silently. `trm.json` also gates staging via `staging.min_turns`/
`staging.min_bytes` (both default to effectively "stage everything",
matching pre-`trm.json` behavior). See `<data_root>/trm.json` (global)
and `<repo_root>/.trm/trm.json` (project override, merged field-by-field
on top) for the full schema.

**Live, in-session staging (ADR-0006 Phase 2, for a STILL-RUNNING session
only — never combine with `--archive-after`).** `--since-checkpoint`
stages only the delta since this session's last saved checkpoint line,
instead of the whole transcript: resolves the bank from a cheap peek at
the transcript (no full digest needed just to find the checkpoint), asks
squishi for `--start-line <last_staged_line>`, and advances the
checkpoint to the new `total_lines` on success. "Nothing new since the
last checkpoint" is a normal `skipped:` outcome, not a failure — an
incremental call landing on an unchanged transcript is expected, not
exceptional. `scripts/claude-code-precompact-hook.sh` wires this to
Claude Code's own `PreCompact` hook (fires right before `/compact` would
summarize/discard detail): `trm ingest-session "$transcript_path"
--since-checkpoint --trigger precompact`, wrapped in a 30s timeout and
never failing the hook itself — a staging failure is logged, never blocks
compaction. Install globally in `~/.claude/settings.json`'s `hooks.PreCompact`
(matcher `"manual|auto"`, matching both trigger sources) to cover every
project, or per-project in that repo's own `.claude/settings.json`.

## Ingest a persona (build an advisor from YouTube/Wikipedia/git/web)

Persona ingestion lives in the standalone `persona` repo now (SQL-first
pipeline, no library dependency on trm) — see that repo's own README for
the actual ingestion commands. trm's only remaining role is the
receiving side of the handover contract: `persona`'s `stage-synthesis`
shells out to `trm stage-persona --manifest <path>`, which stages a
`PersonaBuild` handover exactly like any other — **never concept-split**
(that's the wrong shape for this job; see `handover.rs`'s own doc
comment on why, dated 2026-08-07). `trm pending-show <job-id>` prints
the real, embedded synthesis criteria (co-developed with a real
clinical-psychology advisor in this store) for the sub-agent completing
the job. Mechanical only — trm never calls an LLM itself, per ADR-0002;
synthesis happens when a sub-agent reads the pending prompt and writes
the actual wiki page via `trm complete-handover`.

## Where facts live

~/.trm/banks/<bank-id>/wiki/<slug>.md — one file per fact, plain
markdown, human-readable. index.md in the same bank directory lists every
entry with a one-line summary, plus a Pending section for anything staged
but not yet processed (empty until ingestion pathways exist).

## What trm does not do (yet)

Claude Code session ingestion is real now (`trm ingest-session`, above).
Repo scan, YouTube, and web scrape ingestion exist now too, but not in
trm itself — that's the standalone `persona` repo, which only touches
trm at the handover boundary (`stage-persona`/`pending-show`/
`complete-handover`); no ingestion logic for any of that lives in this
binary. `docs/chat-exports/`-style bulk sources have the mechanism
(stage + concept pre-split) but no dedicated ingestion script wired to
it yet. Kimi Code session ingestion is explicitly
deferred (Claude Code proven first, per the user's own sequencing call)
— real Kimi session data exists on disk, nothing reads it into trm yet.
"#;

/// Resolve content from the positional argument if given, otherwise
/// stdin — the same harness-agnostic pattern squishi's `read_input`
/// already proved out. Refuses to block reading from an interactive
/// terminal with neither: that's almost always a forgotten argument, not
/// someone about to type content, and hanging silently is the worst
/// failure mode for a tool meant to sit in an automated pipeline.
fn read_content_or_stdin(content: Option<String>) -> Result<String, String> {
    if let Some(content) = content {
        return Ok(content);
    }
    if std::io::IsTerminal::is_terminal(&io::stdin()) {
        return Err(
            "no content argument given and stdin is a terminal (not a pipe) — \
             pass content as an argument or pipe it in, e.g. `cat file | trm retain`"
                .to_string(),
        );
    }
    let mut buf = String::new();
    io::Read::read_to_string(&mut io::stdin(), &mut buf)
        .map_err(|e| format!("failed to read stdin: {e}"))?;
    Ok(buf)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let cwd = std::env::current_dir().expect("cwd must be readable");
    let data_root = bank::data_root();

    match cli.command {
        Commands::Retain { content } => {
            let content = match read_content_or_stdin(content) {
                Ok(content) => content,
                Err(e) => {
                    eprintln!("trm retain failed: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match retain(&data_root, &cwd, cli.bank.as_deref(), &content) {
                Ok(slug) => {
                    println!("retained: {slug}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("trm retain failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::Recall { query } => match recall(&data_root, &cwd, cli.bank.as_deref(), &query) {
            Ok(matches) if matches.is_empty() => {
                println!("no matches for {query:?}");
                ExitCode::SUCCESS
            }
            Ok(matches) => {
                for m in matches {
                    println!("{:.2}  {}: {}", m.score, m.slug, m.snippet);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("trm recall failed: {e}");
                ExitCode::FAILURE
            }
        },
        Commands::Stage {
            content,
            reason,
            source,
        } => {
            let content = match read_content_or_stdin(content) {
                Ok(content) => content,
                Err(e) => {
                    eprintln!("trm stage failed: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match stage(
                &data_root,
                &cwd,
                cli.bank.as_deref(),
                &content,
                &reason,
                &source,
            ) {
                Ok(job_id) => {
                    println!("staged: {job_id}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("trm stage failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::Pending { all: true } => match handover::list_pending_all_banks(&data_root) {
            Ok(jobs) if jobs.is_empty() => {
                println!("no pending handovers in any bank");
                ExitCode::SUCCESS
            }
            Ok(jobs) => {
                for (bank_id, job_id) in jobs {
                    println!("{bank_id}/{job_id}");
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("trm pending --all failed: {e}");
                ExitCode::FAILURE
            }
        },
        Commands::Pending { all: false } => match pending(&data_root, &cwd, cli.bank.as_deref()) {
            Ok(ids) if ids.is_empty() => {
                println!("no pending handovers");
                ExitCode::SUCCESS
            }
            Ok(ids) => {
                for id in ids {
                    println!("{id}");
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("trm pending failed: {e}");
                ExitCode::FAILURE
            }
        },
        Commands::PendingShow { job_id } => {
            match pending_show(&data_root, &cwd, cli.bank.as_deref(), &job_id) {
                Ok(prompt) => {
                    print!("{prompt}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("trm pending-show failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::CuratorScan { threshold } => {
            match curator_scan(&data_root, &cwd, cli.bank.as_deref(), threshold) {
                Ok(ids) if ids.is_empty() => {
                    println!("no curation candidates found");
                    ExitCode::SUCCESS
                }
                Ok(ids) => {
                    println!("staged {} curation candidate(s):", ids.len());
                    for id in ids {
                        println!("  {id}");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("trm curator-scan failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::CompleteHandover { job_id, result } => {
            let result = match read_content_or_stdin(result) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("trm complete-handover failed: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match complete_handover(&data_root, &cwd, cli.bank.as_deref(), &job_id, &result) {
                Ok(()) => {
                    println!("completed: {job_id}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("trm complete-handover failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::Import { source, bank } => match import(&data_root, &source, &bank) {
            Ok(count) => {
                println!("imported {count} entries into bank '{bank}'");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("trm import failed: {e}");
                ExitCode::FAILURE
            }
        },
        Commands::Doctor { quick, fix, json } => {
            let bank_id = bank::resolve_bank_id(cli.bank.as_deref(), &cwd);
            let paths = bank::paths_for(&data_root, &bank_id);

            if fix {
                match doctor::fix(&paths) {
                    Some(msg) => println!("fix: {msg}"),
                    None => println!("fix: nothing to fix"),
                }
            }

            let report = doctor::run(&data_root, &cwd, quick);
            if json {
                println!("{}", report.to_json());
            } else {
                for check in &report.checks {
                    let label = match check.status {
                        doctor::Status::Pass => "PASS",
                        doctor::Status::Warn => "WARN",
                        doctor::Status::Fail => "FAIL",
                        doctor::Status::Skipped => "SKIP",
                    };
                    println!("{label:>4}  {}: {}", check.name, check.message);
                }
            }

            if report.has_failures() {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Commands::IngestSessions { all } => {
            if !all {
                eprintln!("trm ingest-sessions: pass --all (the only mode today)");
                return ExitCode::FAILURE;
            }
            let projects_dir = ingest::claude_projects_dir();
            let current_session_id = std::env::var("CLAUDE_CODE_SESSION_ID").ok();
            let stats =
                ingest_sessions_all(&data_root, &projects_dir, current_session_id.as_deref());
            println!(
                "scanned {}: {} archived, {} already archived, {} current session, \
                 {} recently modified, {} below threshold, {} errors",
                projects_dir.display(),
                stats.archived,
                stats.skipped_already_archived,
                stats.skipped_current_session,
                stats.skipped_recent_mtime,
                stats.skipped_below_threshold,
                stats.errors
            );
            if stats.errors > 0 {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Commands::IngestSession {
            path,
            archive_after,
            since_checkpoint,
            trigger,
        } => {
            if archive_after && since_checkpoint {
                eprintln!(
                    "trm ingest-session failed: --archive-after and --since-checkpoint \
                     cannot be combined -- archival is for a finished session, \
                     since-checkpoint is for one that's still running"
                );
                return ExitCode::FAILURE;
            }
            match ingest_session(
                &data_root,
                &cwd,
                cli.bank.as_deref(),
                &path,
                archive_after,
                since_checkpoint,
                trigger,
            ) {
                Ok(IngestOutcome::Staged { job_id, archived }) => {
                    if archived {
                        println!("staged: {job_id} (archived)");
                    } else {
                        println!("staged: {job_id}");
                    }
                    ExitCode::SUCCESS
                }
                Ok(IngestOutcome::Skipped(reason)) => {
                    println!("skipped: {reason}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("trm ingest-session failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::StagePersona { manifest } => {
            match stage_persona(&data_root, cli.bank.as_deref(), &manifest) {
                Ok(job_id) => {
                    println!("staged: {job_id}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("trm stage-persona failed: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Commands::Skill { action } => match action {
            SkillAction::Get { topic } => match topic.as_str() {
                "core" => {
                    print!("{CORE_DOCS}");
                    ExitCode::SUCCESS
                }
                other => {
                    eprintln!("trm skill get: unknown topic '{other}' (only 'core' exists so far)");
                    ExitCode::FAILURE
                }
            },
        },
    }
}

/// Resolve the bank and search its wiki tier. No lock needed — reads
/// don't contend with the lease lock, which only guards writes.
/// Semantic recall: top 5 wiki entries scoring at or above 0.3 cosine
/// similarity to the query. Threshold chosen conservatively (lower than
/// curator-scan's 0.8 document-to-document default) since a short query
/// against a full document naturally scores lower than two documents
/// compared to each other — validate against real recall queries before
/// tightening further.
fn recall(
    data_root: &std::path::Path,
    cwd: &std::path::Path,
    explicit_bank: Option<&str>,
    query: &str,
) -> Result<Vec<wiki::RankedMatch>, String> {
    let bank_id = bank::resolve_bank_id(explicit_bank, cwd);
    let paths = bank::paths_for(data_root, &bank_id);
    let mut embedder = embeddings::Embedder::new(bank::data_root().join("models"))?;
    wiki::semantic_search(&paths.wiki, query, &mut embedder, 0.3, 5)
}

/// Recursively import every `*.md` file under `source` (except a
/// `MEMORY.md` index file, if present) into `bank_id`'s wiki tier,
/// preserving each file's stem as its slug. Returns the count imported.
fn import(
    data_root: &std::path::Path,
    source: &std::path::Path,
    bank_id: &str,
) -> Result<usize, String> {
    let paths = bank::paths_for(data_root, bank_id);
    std::fs::create_dir_all(&paths.root).map_err(|e| e.to_string())?;
    let _guard = lock::acquire(&paths.root).map_err(|e| e.to_string())?;

    let mut count = 0;
    for path in find_md_files(source).map_err(|e| e.to_string())? {
        let slug = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("non-utf8 filename: {}", path.display()))?
            .to_string();
        let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;

        wiki::write_named(&paths.wiki, &slug, &content).map_err(|e| e.to_string())?;
        let summary = bank::summarize(strip_frontmatter(&content));
        bank::append_index_entry(&paths.index, &slug, &summary).map_err(|e| e.to_string())?;
        count += 1;
    }
    Ok(count)
}

/// Walk `dir` recursively, collecting every `*.md` file except ones
/// literally named `MEMORY.md` (an index, not a fact to import).
fn find_md_files(dir: &std::path::Path) -> io::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in fs::read_dir(&current)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("md")
                && path.file_name().and_then(|n| n.to_str()) != Some("MEMORY.md")
            {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

fn stage(
    data_root: &std::path::Path,
    cwd: &std::path::Path,
    explicit_bank: Option<&str>,
    content: &str,
    reason: &str,
    source: &str,
) -> Result<String, String> {
    let bank_id = bank::resolve_bank_id(explicit_bank, cwd);
    let paths = bank::paths_for(data_root, &bank_id);
    fs::create_dir_all(&paths.root).map_err(|e| e.to_string())?;
    let _guard = lock::acquire(&paths.root).map_err(|e| e.to_string())?;
    handover::stage(&paths, content, reason, source).map_err(|e| e.to_string())
}

/// The persona payload manifest's required shape (ADR-0009) -- parsed
/// with plain `serde_json::Value` field access, matching this crate's
/// existing convention (no `serde` derive dependency, only
/// `serde_json`).
struct PersonaManifest {
    slug: String,
    bank: String,
    description: String,
    source_label: String,
    self_build: bool,
    artifacts: Vec<PathBuf>,
}

fn parse_persona_manifest(raw: &str) -> Result<PersonaManifest, String> {
    let json: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| format!("manifest is not valid JSON: {e}"))?;

    let field = |name: &str| -> Result<String, String> {
        json.get(name)
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| format!("manifest missing required string field {name:?}"))
    };

    let artifacts_json = json
        .get("artifacts")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "manifest missing required array field \"artifacts\"".to_string())?;
    let artifacts: Vec<PathBuf> = artifacts_json
        .iter()
        .map(|v| {
            v.as_str()
                .map(PathBuf::from)
                .ok_or_else(|| "manifest \"artifacts\" entries must all be strings".to_string())
        })
        .collect::<Result<_, _>>()?;
    if artifacts.is_empty() {
        return Err(
            "manifest \"artifacts\" must not be empty -- nothing for a sub-agent to read"
                .to_string(),
        );
    }

    Ok(PersonaManifest {
        slug: field("slug")?,
        bank: field("bank")?,
        description: field("description")?,
        source_label: field("source_label")?,
        self_build: json
            .get("self_build")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        artifacts,
    })
}

/// Stage a persona-build handover from an external caller's manifest
/// file. `explicit_bank` (the global `--bank` flag) overrides the
/// manifest's own `bank` field when given -- otherwise the manifest is
/// self-contained, since an external tool (e.g. `persona`) has no
/// cwd-based relationship to a bank the way an in-repo `trm` invocation
/// does.
fn stage_persona(
    data_root: &std::path::Path,
    explicit_bank: Option<&str>,
    manifest_path: &std::path::Path,
) -> Result<String, String> {
    let raw = fs::read_to_string(manifest_path)
        .map_err(|e| format!("failed to read manifest {}: {e}", manifest_path.display()))?;
    let manifest = parse_persona_manifest(&raw)?;

    let bank_id = explicit_bank.map(str::to_string).unwrap_or(manifest.bank);
    let paths = bank::paths_for(data_root, &bank_id);
    fs::create_dir_all(&paths.root).map_err(|e| e.to_string())?;
    let _guard = lock::acquire(&paths.root).map_err(|e| e.to_string())?;
    handover::stage_persona_sources(
        &paths,
        &manifest.slug,
        manifest.artifacts,
        &manifest.description,
        &manifest.source_label,
        manifest.self_build,
    )
    .map_err(|e| e.to_string())
}

/// What `ingest_session` actually did — `main()`'s match arm reports this
/// back to the user; a skip is not an error (a disabled hook or a
/// below-threshold session behaving exactly as configured).
enum IngestOutcome {
    Skipped(String),
    Staged { job_id: String, archived: bool },
}

/// Full `trm ingest-session` flow, pulled out of `main()`'s match arm so
/// it's independently testable against real fixtures — same reasoning
/// `stage`/`pending` already established for this file's other command
/// bodies. ADR-0006: whole-transcript digest + stage by default
/// (Phase 1), optionally followed by archiving the source transcript
/// (`archive_after`, finished sessions only) or scoped to just the delta
/// since a saved checkpoint (`since_checkpoint`, Phase 2, live sessions
/// only) — mutually exclusive, enforced both at the CLI layer and here.
#[allow(clippy::too_many_arguments)]
fn ingest_session(
    data_root: &std::path::Path,
    cwd: &std::path::Path,
    explicit_bank: Option<&str>,
    transcript_path: &std::path::Path,
    archive_after: bool,
    since_checkpoint: bool,
    trigger: Trigger,
) -> Result<IngestOutcome, String> {
    if archive_after && since_checkpoint {
        return Err("--archive-after and --since-checkpoint cannot be combined".to_string());
    }

    // Incremental start_line: resolved from a cheap peek (no squishi
    // subprocess) at the transcript's own sessionId/cwd, so the
    // checkpoint lookup doesn't need a full digest first. A peek failure
    // (unreadable/empty file) falls through to start_line 0 -- the
    // subsequent full digest call surfaces whatever the real problem is,
    // rather than this function guessing at one.
    let start_line = if since_checkpoint {
        match ingest::peek_session_meta(transcript_path) {
            Some((session_id, peeked_cwd)) => {
                let bank_id = bank::resolve_bank_id(explicit_bank, &peeked_cwd);
                let paths = bank::paths_for(data_root, &bank_id);
                let checkpoint = session_checkpoint::load(&paths.session_state);
                session_checkpoint::last_staged_line(&checkpoint, &session_id)
            }
            None => 0,
        }
    } else {
        0
    };

    let digest = ingest::run_squishi_session_digest_from(transcript_path, start_line)?;
    if digest.content.trim().is_empty() {
        // An incremental call landing on "nothing new since the last
        // checkpoint" is a normal outcome, not a failure -- squishi
        // itself already treats it this way for start_line > 0 (see
        // ADR-0006 Phase 2). A genuinely empty whole-file digest is
        // still worth failing loudly on, unchanged from Phase 1.
        if since_checkpoint {
            return Ok(IngestOutcome::Skipped(
                "nothing new since last checkpoint".to_string(),
            ));
        }
        return Err("nothing to digest (empty session)".to_string());
    }

    // The session's OWN cwd (parsed from squishi's output), not the
    // invoking process's cwd — same real behavior `stage`'s caller
    // already preserves for the non-checkpointed path.
    let session_cwd = digest
        .cwd
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| cwd.to_path_buf());

    let config = config::load(data_root, &session_cwd)?;

    let hook_enabled = match trigger {
        Trigger::Precompact => config.pre_compact_enabled,
        Trigger::Sessionend => config.session_end_enabled,
        Trigger::Manual => true, // never gated -- pre-existing, ungated behavior
    };
    if !hook_enabled {
        return Ok(IngestOutcome::Skipped(format!(
            "{trigger:?} hook disabled in trm.json"
        )));
    }

    if digest.turn_count < config.min_turns || digest.content.len() < config.min_bytes {
        return Ok(IngestOutcome::Skipped(format!(
            "below staging threshold (turn_count={} < min_turns={} or bytes={} < min_bytes={})",
            digest.turn_count,
            config.min_turns,
            digest.content.len(),
            config.min_bytes
        )));
    }

    let reason = ingest::build_reason(&digest);
    let job_id = stage(
        data_root,
        &session_cwd,
        explicit_bank,
        &digest.content,
        &reason,
        "direct",
    )?;

    if since_checkpoint {
        let bank_id = bank::resolve_bank_id(explicit_bank, &session_cwd);
        let paths = bank::paths_for(data_root, &bank_id);
        let session_id = digest.session_id.clone().unwrap_or_else(|| job_id.clone());
        let mut checkpoint = session_checkpoint::load(&paths.session_state);
        session_checkpoint::mark_staged(&mut checkpoint, &session_id, digest.total_lines);
        session_checkpoint::save(&paths.session_state, &checkpoint).map_err(|e| e.to_string())?;
    }

    if !archive_after {
        return Ok(IngestOutcome::Staged {
            job_id,
            archived: false,
        });
    }

    let bank_id = bank::resolve_bank_id(explicit_bank, &session_cwd);
    let paths = bank::paths_for(data_root, &bank_id);
    let session_id = digest.session_id.clone().unwrap_or_else(|| job_id.clone());
    let dest = paths.sessions.join(format!("{session_id}.jsonl.gz"));
    archive::archive_transcript(transcript_path, &dest).map_err(|e| e.to_string())?;

    let mut checkpoint = session_checkpoint::load(&paths.session_state);
    session_checkpoint::mark_archived(&mut checkpoint, &session_id);
    session_checkpoint::save(&paths.session_state, &checkpoint).map_err(|e| e.to_string())?;

    Ok(IngestOutcome::Staged {
        job_id,
        archived: true,
    })
}

/// How long a transcript must sit untouched before `ingest_sessions_all`
/// will consider archiving it — the secondary guard for when
/// `$CLAUDE_CODE_SESSION_ID` isn't set (`trm` run outside any live Claude
/// Code session). Not yet configurable via `trm.json` (see the plan's
/// Risks section) — a fixed constant until real usage shows it needs to be.
const RECENT_MTIME_GUARD_SECS: u64 = 600;

#[derive(Debug, Default, PartialEq)]
struct SweepStats {
    archived: usize,
    skipped_already_archived: usize,
    skipped_current_session: usize,
    skipped_recent_mtime: usize,
    skipped_below_threshold: usize,
    errors: usize,
}

/// `trm ingest-sessions --all`'s real work: walk every transcript under
/// `projects_dir`, skip what's clearly not eligible (already archived,
/// the caller's own live session, too recently modified), and run the
/// same stage-then-archive path `ingest_session --archive-after` uses for
/// everything else. One bad file never aborts the sweep — logged to
/// stderr, counted, moved on — same "not a versioned contract" discipline
/// every other transcript-reading path in this codebase already follows.
fn ingest_sessions_all(
    data_root: &std::path::Path,
    projects_dir: &std::path::Path,
    current_session_id: Option<&str>,
) -> SweepStats {
    let mut stats = SweepStats::default();

    for path in ingest::find_transcripts(projects_dir) {
        let Some((session_id, session_cwd)) = ingest::peek_session_meta(&path) else {
            eprintln!(
                "trm ingest-sessions: {} has no usable sessionId/cwd line, skipping",
                path.display()
            );
            stats.errors += 1;
            continue;
        };

        if current_session_id == Some(session_id.as_str()) {
            stats.skipped_current_session += 1;
            continue;
        }

        let bank_id = bank::resolve_bank_id(None, &session_cwd);
        let paths = bank::paths_for(data_root, &bank_id);
        let checkpoint = session_checkpoint::load(&paths.session_state);
        if session_checkpoint::is_archived(&checkpoint, &session_id) {
            stats.skipped_already_archived += 1;
            continue;
        }

        let recently_modified = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .and_then(|modified| {
                modified
                    .elapsed()
                    .map_err(|e| io::Error::other(e.to_string()))
            })
            .map(|elapsed| elapsed.as_secs() < RECENT_MTIME_GUARD_SECS)
            .unwrap_or(false);
        if recently_modified {
            stats.skipped_recent_mtime += 1;
            continue;
        }

        match ingest_session(
            data_root,
            &session_cwd,
            None,
            &path,
            true,
            false,
            Trigger::Manual,
        ) {
            Ok(IngestOutcome::Staged { archived: true, .. }) => stats.archived += 1,
            Ok(IngestOutcome::Staged {
                archived: false, ..
            }) => {
                // Can't happen with archive_after=true, but not a panic-
                // worthy invariant break either -- count it and move on.
                stats.errors += 1;
            }
            Ok(IngestOutcome::Skipped(_)) => stats.skipped_below_threshold += 1,
            Err(e) => {
                eprintln!("trm ingest-sessions: {} failed: {e}", path.display());
                stats.errors += 1;
            }
        }
    }

    stats
}

/// Resolve the bank and list its open handover jobs. No lock needed —
/// same as recall, this is a read.
fn pending(
    data_root: &std::path::Path,
    cwd: &std::path::Path,
    explicit_bank: Option<&str>,
) -> Result<Vec<String>, String> {
    let bank_id = bank::resolve_bank_id(explicit_bank, cwd);
    let paths = bank::paths_for(data_root, &bank_id);
    handover::list_pending(&paths).map_err(|e| e.to_string())
}

/// Resolve the bank and print a pending job's exact rendered prompt.
fn pending_show(
    data_root: &std::path::Path,
    cwd: &std::path::Path,
    explicit_bank: Option<&str>,
    job_id: &str,
) -> Result<String, String> {
    let bank_id = bank::resolve_bank_id(explicit_bank, cwd);
    let paths = bank::paths_for(data_root, &bank_id);
    handover::get_prompt(&paths, job_id).map_err(|e| e.to_string())
}

/// Resolve the bank, acquire its lock, and scan for curation candidates.
fn curator_scan(
    data_root: &std::path::Path,
    cwd: &std::path::Path,
    explicit_bank: Option<&str>,
    threshold: f64,
) -> Result<Vec<String>, String> {
    let bank_id = bank::resolve_bank_id(explicit_bank, cwd);
    let paths = bank::paths_for(data_root, &bank_id);
    let _guard = lock::acquire(&paths.root).map_err(|e| e.to_string())?;
    curator::scan(&paths, threshold)
}

/// Resolve the bank, acquire its lock, and commit a completed handover.
fn complete_handover(
    data_root: &std::path::Path,
    cwd: &std::path::Path,
    explicit_bank: Option<&str>,
    job_id: &str,
    result: &str,
) -> Result<(), String> {
    let bank_id = bank::resolve_bank_id(explicit_bank, cwd);
    let paths = bank::paths_for(data_root, &bank_id);
    let _guard = lock::acquire(&paths.root).map_err(|e| e.to_string())?;
    handover::complete(&paths, job_id, result).map_err(|e| e.to_string())
}

/// Strip a leading `---\n...\n---\n` YAML frontmatter block, if present,
/// so index summaries reflect the actual content, not frontmatter noise.
fn strip_frontmatter(content: &str) -> &str {
    let Some(rest) = content.strip_prefix("---\n") else {
        return content;
    };
    match rest.find("\n---\n") {
        Some(end) => rest[end + 5..].trim_start(),
        None => content,
    }
}

/// Resolve the bank, acquire its lease lock, write the fact into the wiki
/// tier, record it in index.md, and release the lock. Returns the slug.
fn retain(
    data_root: &std::path::Path,
    cwd: &std::path::Path,
    explicit_bank: Option<&str>,
    content: &str,
) -> Result<String, String> {
    let bank_id = bank::resolve_bank_id(explicit_bank, cwd);
    let paths = bank::paths_for(data_root, &bank_id);

    std::fs::create_dir_all(&paths.root).map_err(|e| e.to_string())?;
    let _guard = lock::acquire(&paths.root).map_err(|e| e.to_string())?;

    let slug = wiki::write(&paths.wiki, content).map_err(|e| e.to_string())?;
    let summary = bank::summarize(content);
    bank::append_index_entry(&paths.index, &slug, &summary).map_err(|e| e.to_string())?;

    Ok(slug)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn retain_writes_wiki_file_and_index_entry_in_resolved_bank() {
        let data_root_dir = tempfile::tempdir().unwrap();
        let cwd_dir = tempfile::tempdir().unwrap();

        let slug = retain(
            data_root_dir.path(),
            cwd_dir.path(),
            None,
            "user prefers terse commit messages",
        )
        .unwrap();

        let paths = bank::paths_for(data_root_dir.path(), "global");
        assert_eq!(
            fs::read_to_string(paths.wiki.join(format!("{slug}.md"))).unwrap(),
            "user prefers terse commit messages"
        );
        let index = fs::read_to_string(&paths.index).unwrap();
        assert!(index.contains(&slug));
    }

    #[test]
    fn retain_from_two_different_repos_lands_in_two_different_banks() {
        let data_root_dir = tempfile::tempdir().unwrap();
        let repo_a = tempfile::tempdir().unwrap();
        let repo_b = tempfile::tempdir().unwrap();
        fs::create_dir_all(repo_a.path().join(".git")).unwrap();
        fs::create_dir_all(repo_b.path().join(".git")).unwrap();

        retain(data_root_dir.path(), repo_a.path(), None, "fact in repo A").unwrap();
        retain(data_root_dir.path(), repo_b.path(), None, "fact in repo B").unwrap();

        let banks_dir = data_root_dir.path().join("banks");
        let mut bank_ids: Vec<_> = fs::read_dir(&banks_dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
            .collect();
        bank_ids.sort();
        assert_eq!(
            bank_ids.len(),
            2,
            "expected two distinct banks, got {bank_ids:?}"
        );
    }

    #[test]
    fn explicit_bank_flag_overrides_auto_resolution() {
        let data_root_dir = tempfile::tempdir().unwrap();
        let cwd_dir = tempfile::tempdir().unwrap();

        retain(
            data_root_dir.path(),
            cwd_dir.path(),
            Some("chosen-bank"),
            "explicit bank fact",
        )
        .unwrap();

        let paths = bank::paths_for(data_root_dir.path(), "chosen-bank");
        assert!(paths.wiki.exists());
    }

    #[test]
    fn core_docs_covers_retain_and_bank_resolution() {
        assert!(CORE_DOCS.contains("trm retain"));
        assert!(CORE_DOCS.contains("--bank"));
        assert!(CORE_DOCS.contains(".trm"));
    }

    #[test]
    fn core_docs_covers_recall() {
        assert!(CORE_DOCS.contains("trm recall"));
    }

    #[test]
    fn core_docs_covers_ingest_session() {
        assert!(CORE_DOCS.contains("trm ingest-session"));
        assert!(CORE_DOCS.contains("SESSION's own cwd"));
    }

    #[test]
    fn core_docs_covers_session_archival() {
        assert!(CORE_DOCS.contains("trm ingest-sessions --all"));
        assert!(CORE_DOCS.contains("--archive-after"));
        assert!(CORE_DOCS.contains("trm.json"));
        assert!(CORE_DOCS.contains("CLAUDE_CODE_SESSION_ID"));
    }

    #[test]
    fn core_docs_covers_precompact_incremental_staging() {
        assert!(CORE_DOCS.contains("--since-checkpoint"));
        assert!(CORE_DOCS.contains("claude-code-precompact-hook.sh"));
        assert!(CORE_DOCS.contains("PreCompact"));
        assert!(CORE_DOCS.contains("last_staged_line"));
    }

    #[test]
    fn core_docs_covers_persona_handover_boundary() {
        assert!(CORE_DOCS.contains("standalone `persona` repo"));
        assert!(CORE_DOCS.contains("trm stage-persona"));
        assert!(CORE_DOCS.contains("PersonaBuild"));
    }

    #[test]
    fn stage_pending_and_complete_handover_round_trip() {
        let data_root_dir = tempfile::tempdir().unwrap();
        let cwd_dir = tempfile::tempdir().unwrap();
        let data_root = data_root_dir.path();
        let cwd = cwd_dir.path();

        let job_id = stage(
            data_root,
            cwd,
            None,
            "raw transcript needing extraction",
            "extract key facts",
            "direct",
        )
        .unwrap();
        assert_eq!(pending(data_root, cwd, None).unwrap(), vec![job_id.clone()]);

        complete_handover(data_root, cwd, None, &job_id, "the extracted fact").unwrap();

        assert!(pending(data_root, cwd, None).unwrap().is_empty());
        let matches = recall(data_root, cwd, None, "extracted fact").unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].slug, job_id);
    }

    #[test]
    fn pending_show_returns_the_rendered_prompt_for_the_resolved_bank() {
        let data_root_dir = tempfile::tempdir().unwrap();
        let cwd_dir = tempfile::tempdir().unwrap();
        let data_root = data_root_dir.path();
        let cwd = cwd_dir.path();

        let job_id = stage(
            data_root,
            cwd,
            None,
            "raw content",
            "a distinctive reason",
            "direct",
        )
        .unwrap();
        let prompt = pending_show(data_root, cwd, None, &job_id).unwrap();
        assert!(prompt.contains("a distinctive reason"));
    }

    #[test]
    fn strip_frontmatter_removes_yaml_block() {
        let content =
            "---\nperson: Boris Cherny\nboard: technical\n---\n\n# Boris Cherny\n\nBody text.";
        assert_eq!(strip_frontmatter(content), "# Boris Cherny\n\nBody text.");
    }

    #[test]
    fn strip_frontmatter_leaves_content_without_frontmatter_untouched() {
        assert_eq!(
            strip_frontmatter("no frontmatter here"),
            "no frontmatter here"
        );
    }

    #[test]
    fn import_writes_named_entries_and_index() {
        let data_root_dir = tempfile::tempdir().unwrap();
        let source_dir = tempfile::tempdir().unwrap();
        fs::write(
            source_dir.path().join("boris-cherny.md"),
            "---\nperson: Boris Cherny\n---\n\nContext isolation is the mechanism.",
        )
        .unwrap();
        fs::write(
            source_dir.path().join("uncle-bob.md"),
            "---\nperson: Robert C. Martin\n---\n\nSmall functions, one responsibility.",
        )
        .unwrap();
        fs::write(
            source_dir.path().join("not-markdown.txt"),
            "should be skipped",
        )
        .unwrap();

        let count = import(data_root_dir.path(), source_dir.path(), "mindforge").unwrap();
        assert_eq!(count, 2);

        let paths = bank::paths_for(data_root_dir.path(), "mindforge");
        assert_eq!(
            fs::read_to_string(paths.wiki.join("boris-cherny.md")).unwrap(),
            "---\nperson: Boris Cherny\n---\n\nContext isolation is the mechanism."
        );
        assert!(paths.wiki.join("uncle-bob.md").exists());
        assert!(!paths.wiki.join("not-markdown.md").exists());

        let index = fs::read_to_string(&paths.index).unwrap();
        assert!(index.contains("`boris-cherny`"));
        assert!(index.contains("`uncle-bob`"));
    }

    #[test]
    fn import_walks_subdirectories_and_skips_memory_md_index() {
        let data_root_dir = tempfile::tempdir().unwrap();
        let source_dir = tempfile::tempdir().unwrap();
        fs::write(
            source_dir.path().join("top-level-fact.md"),
            "a top-level fact",
        )
        .unwrap();
        fs::write(
            source_dir.path().join("MEMORY.md"),
            "# Memory index\n- pointer only",
        )
        .unwrap();
        let nested = source_dir.path().join("wiki");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("nested-fact.md"), "a nested fact").unwrap();

        let count = import(data_root_dir.path(), source_dir.path(), "native-labs").unwrap();
        assert_eq!(count, 2);

        let paths = bank::paths_for(data_root_dir.path(), "native-labs");
        assert!(paths.wiki.join("top-level-fact.md").exists());
        assert!(paths.wiki.join("nested-fact.md").exists());
        assert!(!paths.wiki.join("MEMORY.md").exists());
    }
}
