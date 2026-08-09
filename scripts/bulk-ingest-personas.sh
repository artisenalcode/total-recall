#!/usr/bin/env bash
# Bulk-runs `trm ingest-persona` over a roster of many people (the real
# case this exists for: mindforge's ~50-persona bank) — a thin loop, not
# a new trm subcommand, per this project's own philosophy (trm stays a
# single-persona-per-invocation mechanical fetcher; config-file parsing
# for a roster doesn't need to live inside the Rust binary when a shell
# loop over `ingest-persona`'s existing flags does the same job).
#
# Roster format: one persona per line, pipe-delimited:
#   <slug>|<Person Name>|<extra trm ingest-persona args>
# A line starting with `#` or blank is skipped. The third field is
# passed through to `ingest-persona` (real shell-word parsing via
# `eval`, so a quoted multi-word value works), so it can be any real
# combination of --channel/--video/--wikipedia/--git-repo/--git-author/
# --github-user/--website — whatever sources exist for that person.
# Example roster line:
#   coding-garden|CJ (Coding Garden)|--channel https://www.youtube.com/CodingGarden --max-videos 20
#
# Real limitation: don't use `--video <id>|<title>` in the third field —
# ingest-persona's own "id|title" syntax collides with this roster
# format's own `|` delimiter. Use `--channel` (enumerates real videos
# itself) instead; hand-picked `--video` entries still work fine run
# directly via `trm ingest-persona`, just not through this roster file.
#
# Usage:
#   bulk-ingest-personas.sh <roster-file> [-p bank]
#
# Resilient per-persona, same posture `ingest_videos`/`ingest_websites`
# already have internally: one persona's ingestion failing (rate limit,
# bad URL, no commits for that author) does not abort the batch — it's
# reported at the end, not silently swallowed or fatal to the rest.
#
# Runs `trm curator-scan` once at the end, across the whole bank — the
# real cross-persona dedup step (embedding cosine similarity, catches
# e.g. two advisors covering the same disagreement or shared podcast),
# not a per-persona concern; see `trm skill get core`'s Curator scan
# section for how to read its output.

set -uo pipefail

roster="${1:?usage: bulk-ingest-personas.sh <roster-file> [-p bank]}"
shift
bank_args=("$@")

if [[ ! -f "$roster" ]]; then
    echo "bulk-ingest-personas: roster file not found: $roster" >&2
    exit 1
fi

if ! command -v trm >/dev/null 2>&1; then
    echo "bulk-ingest-personas: trm not on PATH" >&2
    exit 1
fi

succeeded=0
failed=0
failed_slugs=()

while IFS='|' read -r slug person extra_args; do
    # Skip blank lines and comments.
    [[ -z "${slug// /}" ]] && continue
    [[ "$slug" =~ ^[[:space:]]*# ]] && continue

    echo "=== ingesting: $person ($slug) ==="
    # `eval`-parse extra_args as real shell words so a quoted multi-word
    # value (e.g. --wikipedia "Some Title") splits correctly, instead of
    # naive unquoted word-splitting breaking on the embedded space. Safe
    # here specifically because the roster is a local file the user
    # authors themselves, not untrusted/attacker-controlled input.
    eval "extra_arr=($extra_args)"
    if trm "${bank_args[@]}" ingest-persona --person "$person" --slug "$slug" "${extra_arr[@]}"; then
        succeeded=$((succeeded + 1))
    else
        failed=$((failed + 1))
        failed_slugs+=("$slug")
        echo "warning: ingestion failed for $slug (see above) — continuing with the rest" >&2
    fi
done < "$roster"

echo
echo "bulk-ingest-personas: $succeeded succeeded, $failed failed"
if [[ $failed -gt 0 ]]; then
    printf 'failed: %s\n' "${failed_slugs[*]}"
fi

echo
echo "=== running curator-scan for cross-persona dedup ==="
trm "${bank_args[@]}" curator-scan

exit 0
