# total-recall

Canonical cross-harness memory CLI. Binary name: `trm`.

Bank-scoped (per-project or global), plain markdown, no external LLM APIs —
any judgment work hands back to the calling agent, `trm` itself never
calls a model. Shared across harnesses (Claude Code, Kimi Code, anything
else) via `~/.trm/`, not tied to any one tool's own memory format.

Extracted from `mindforge/mf` into its own repo — was a subdirectory of a
larger suite, now standalone (full commit history preserved via
`git subtree split`). Formerly named `mf`; binary renamed to `trm` —
deliberately not `tr`, which is the POSIX coreutils translate-characters
command and would silently shadow it on PATH.

## Install

```bash
cargo install --path .
```

Builds from source — no published crate yet.

## Usage

Serves its own live usage docs, so instructions never go stale against
the installed version:

```bash
trm skill get core
```

Quick reference:

```bash
trm retain "<content>"                    # store a fact
trm -p global retain "<content>"          # force the global bank
trm recall "<query>"                      # semantic ranked search
trm stage "<raw>" --reason "<why>"        # queue for sub-agent judgment
trm pending [--all]
trm pending-show <job-id>
trm complete-handover <job-id> "<result>"
trm curator-scan [--threshold 0.8]        # find duplicate-candidate entries
trm import <source> --bank <bank>         # migrate markdown into a bank
```

Bank resolution precedence: explicit `-p/--bank` flag > `.trm-bank` file
at the repo root > git remote (owner/repo slug) > path hash > `global`.

## Persona ingestion

```bash
trm ingest-persona --person "Full Name" --slug the-slug \
  --channel https://www.youtube.com/@SomeChannel --max-videos 50 \
  --wikipedia "Full Name" \
  --git-repo https://github.com/owner/repo --git-author name@example.com --github-user handle \
  --website https://example.com/about
```

Source types: `--video`/`--channel` (YouTube, via `yt-dlp`), `--wikipedia`
(MediaWiki API), `--git-repo`+`--git-author` (commit messages, via a
blobless clone) and `--git-repo`+`--github-user` (issues/PRs authored by a
real GitHub login, via `gh api search/issues` — a different identifier
than `--git-author`'s email, since GitHub matches issues by account, not
commit-trailer email), `--website` (a real page fetch via the locally
installed `agent-browser` CLI — real browser rendering, so JS-heavy pages
work where a plain HTTP GET would return nothing), and `--session` (the
user's own Claude Code transcripts, always routed to the `self` bank —
see ADR-0007, can't be combined with the advisor sources above in one
call). Every pathway is mechanical fetch+clean+stage only — `trm` never
calls an LLM itself (ADR-0002); read the staged handover with
`trm pending-show <job-id>` and synthesize it yourself.

For many people at once (`scripts/bulk-ingest-personas.sh <roster-file>
[-p bank]`): a pipe-delimited roster file, one persona per line
(`<slug>|<Person Name>|<extra ingest-persona args>`), looped with
per-persona resilience (one failure doesn't abort the batch), followed
by one `trm curator-scan` pass across the whole bank for cross-persona
dedup. See the script's own header comment for the roster format and a
real limitation (`--video`'s `id|title` syntax collides with the
roster's own `|` delimiter — use `--channel` in a roster instead).

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

CI caches the embedding model (`~/.trm/models`, all-MiniLM-L6-v2 via
ONNX/fastembed) so it isn't re-downloaded every run.
