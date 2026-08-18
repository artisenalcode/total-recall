#!/usr/bin/env bash
# Claude Code PreCompact hook (ADR-0006 Phase 2): stages whatever's new
# in the current session's transcript into its resolved bank's raw tier
# BEFORE Claude Code's own /compact summarization discards detail. Live,
# in-session only — never archives (the session is still running; the
# transcript file may still be appended to). Incremental via `trm
# ingest-session --since-checkpoint`: each firing stages only the delta
# since the last one, not the whole transcript again.
#
# Also the one place `trm ccr-gc` gets called opportunistically — closes
# the "nothing runs ccr-gc on a schedule" gap noted in
# docs/ideation/governator-ccr/2026-08-18-ccr-recovery-spec.md. Throttled
# to at most once per hour via a marker file: PreCompact can fire often
# within a busy session, and a full CCR store walk on every firing would
# be wasted work for no benefit (entries are evicted on trm.json's own
# age/size thresholds regardless of how often the sweep runs).
#
# Side-effect-only hook: PreCompact has no `updatedToolOutput`-style
# mechanism to feed anything back into the harness, so this script's
# only job is "did the staging (and, throttled, the gc) happen or not" —
# it never blocks compaction over a staging or gc failure. Every exit
# path is 0.
#
# Requires: trm and jq on PATH. Missing either is a silent no-op, not a
# blocked compaction.

set -uo pipefail

input="$(cat)"
transcript_path="$(jq -r '.transcript_path // empty' <<<"$input" 2>/dev/null)"

if [[ -z "$transcript_path" ]]; then
    exit 0
fi

if ! command -v trm >/dev/null 2>&1; then
    # trm not installed — don't block compaction over a missing optional
    # dependency, just skip staging this time.
    exit 0
fi

# 30s ceiling: a hung squishi/trm subprocess must never hang Claude
# Code's own compaction indefinitely. A real staging failure (timeout,
# nonzero exit) is logged to stderr (visible in Claude Code's debug
# output) but is not this hook's problem to escalate — never a nonzero
# exit back to the harness over it.
if command -v timeout >/dev/null 2>&1; then
    timeout 30s trm ingest-session "$transcript_path" --since-checkpoint --trigger precompact \
        1>&2 || echo "claude-code-precompact-hook: trm ingest-session failed or timed out (exit $?)" >&2
else
    trm ingest-session "$transcript_path" --since-checkpoint --trigger precompact \
        1>&2 || echo "claude-code-precompact-hook: trm ingest-session failed (exit $?)" >&2
fi

# Opportunistic CCR sweep, throttled to at most once per hour via a
# marker file — uses trm's own configured thresholds (trm.json's
# ccr.max_age_days/ccr.max_bytes, or the built-in defaults); no flags
# passed here. Same `MF_DATA_ROOT` override this whole codebase already
# respects (see bank.rs::data_root()), so tests that isolate their own
# data root don't also race a real marker file.
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
