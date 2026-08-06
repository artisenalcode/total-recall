mod atomic;
mod bank;
mod concepts;
mod curator;
mod doctor;
mod embeddings;
mod handover;
mod lock;
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
    CompleteHandover { job_id: String, result: String },
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

## Where facts live

~/.trm/banks/<bank-id>/wiki/<slug>.md — one file per fact, plain
markdown, human-readable. index.md in the same bank directory lists every
entry with a one-line summary, plus a Pending section for anything staged
but not yet processed (empty until ingestion pathways exist).

## What trm does not do (yet)

No ingestion pathways for repo scan, YouTube, or web scrape (`code
archaeology`/`youtube`/`web-scrape` per CONTEXT.md's resolved-but-unbuilt
pathway list). `docs/chat-exports/`-style bulk sources have the mechanism
(stage + concept pre-split) but no dedicated ingestion script wired to it
yet. Kimi Code session ingestion is explicitly deferred (Claude Code
proven first, per the user's own sequencing call) — real Kimi session data
exists on disk, nothing reads it into trm yet.
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

/// Resolve the bank, acquire its lock, and stage raw content for a
/// sub-agent handover. Returns the job id.
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
