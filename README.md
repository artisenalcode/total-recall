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
trm ccr-put "<content>"                   # store bytes for later exact recovery, prints a handle
trm ccr-get <handle>                      # recover those exact bytes
trm ccr-gc [--max-age-days N] [--max-bytes N]  # evict old/oversized CCR entries
```

Bank resolution precedence: explicit `-p/--bank` flag > `.trm-bank` file
at the repo root > git remote (owner/repo slug) > path hash > `global`.

## CCR (content-addressed recovery)

A short-lived, content-addressed store for the exact bytes behind a lossy
compression — distinct from `retain`/`stage` above, which are durable and
judgment-gated. `ccr-put` writes bytes and prints a handle
(`ccr_<16 hex chars>`); `ccr-get <handle>` returns those exact bytes,
byte-for-byte. Identical content put twice returns the same handle and is
stored once — data-root scoped, not bank scoped, so the same repeated
blob across two different repos' sessions shares one object.

The intended pattern: a caller (e.g. [`governator`](https://github.com/artisenalcode/governator))
compresses a tool result lossily, calls `ccr-put` on the ORIGINAL bytes
first, and attaches the handle. If `ccr-put` itself fails, the caller
falls back to shipping the uncompressed original rather than a lossy
result with no way back — `trm` doesn't enforce this, it's a contract
callers are responsible for. `ccr-gc` evicts entries past a configurable
age (default 3 days) or once the store exceeds a size cap (default
500MB); `trm doctor` warns when either limit is exceeded. The PreCompact
hook below also runs `ccr-gc` opportunistically (throttled to at most
once per hour via a marker file) whenever it fires, so a real session
gets a sweep without a separate cron job.

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
`candle` — see `src/embeddings.rs`'s module doc comment for the
2026-08-18 port off `fastembed`/`ort`) so it isn't re-downloaded every
run.
