#!/usr/bin/env bash
# Claude Code PreCompact hook: stages new transcript content into the bank before /compact discards it, and sweeps ccr-gc opportunistically.
# Every exit path is 0 -- a staging/gc failure must never block compaction, only get logged to stderr.

set -uo pipefail

input="$(cat)"
transcript_path="$(jq -r '.transcript_path // empty' <<<"$input" 2>/dev/null)"

if [[ -z "$transcript_path" ]]; then
    exit 0
fi

if ! command -v trm >/dev/null 2>&1; then
    exit 0
fi

# 30s ceiling: a hung squishi/trm subprocess must never hang compaction indefinitely.
if command -v timeout >/dev/null 2>&1; then
    timeout 30s trm ingest-session "$transcript_path" --since-checkpoint --trigger precompact \
        1>&2 || echo "claude-code-precompact-hook: trm ingest-session failed or timed out (exit $?)" >&2
else
    trm ingest-session "$transcript_path" --since-checkpoint --trigger precompact \
        1>&2 || echo "claude-code-precompact-hook: trm ingest-session failed (exit $?)" >&2
fi

# Throttled to once per hour via a marker file; MF_DATA_ROOT override matches bank.rs::data_root() so isolated tests don't race a real marker.
data_root="${MF_DATA_ROOT:-$HOME/.trm}"
gc_marker="$data_root/.last-ccr-gc"
gc_throttle_secs=3600
now="$(date +%s)"
last_gc=0
if [[ -f "$gc_marker" ]]; then
    last_gc="$(cat "$gc_marker" 2>/dev/null)"
    [[ "$last_gc" =~ ^[0-9]+$ ]] || last_gc=0
fi
if (( now - last_gc >= gc_throttle_secs )); then
    if command -v timeout >/dev/null 2>&1; then
        timeout 30s trm ccr-gc 1>&2 || echo "claude-code-precompact-hook: trm ccr-gc failed or timed out (exit $?)" >&2
    else
        trm ccr-gc 1>&2 || echo "claude-code-precompact-hook: trm ccr-gc failed (exit $?)" >&2
    fi
    mkdir -p "$data_root" 2>/dev/null
    echo "$now" >"$gc_marker" 2>/dev/null || true
fi

exit 0
