#!/usr/bin/env bash
# Claude Code PreCompact hook (ADR-0006 Phase 2): stages whatever's new
# in the current session's transcript into its resolved bank's raw tier
# BEFORE Claude Code's own /compact summarization discards detail. Live,
# in-session only — never archives (the session is still running; the
# transcript file may still be appended to). Incremental via `trm
# ingest-session --since-checkpoint`: each firing stages only the delta
# since the last one, not the whole transcript again.
#
# Side-effect-only hook: PreCompact has no `updatedToolOutput`-style
# mechanism to feed anything back into the harness, so this script's
# only job is "did the staging happen or not" — it never blocks
# compaction over a staging failure. Every exit path is 0.
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

exit 0
