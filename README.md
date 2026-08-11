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

Persona ingestion (YouTube/Wikipedia/git/website fetching, dedup,
cross-file clustering, lexicon-scan) lives in the standalone
[`persona`](https://github.com/artisenalcode/persona) repo now — a
SQL-first pipeline with no library dependency on `trm`. `trm`'s only
remaining role in that pipeline is the receiving side of the handover
contract: `persona`'s own `stage-synthesis` command shells out to
`trm stage-persona --manifest <path>`, which stages a `PersonaBuild`
handover exactly like any other — read it with `trm pending-show
<job-id>` and synthesize it yourself, then `trm complete-handover
<job-id> "<result>"` same as any other handover kind. See `persona`'s
own README for the actual ingestion commands.

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

CI caches the embedding model (`~/.trm/models`, all-MiniLM-L6-v2 via
ONNX/fastembed) so it isn't re-downloaded every run.
