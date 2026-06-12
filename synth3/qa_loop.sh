#!/usr/bin/env bash
# Periodic QA sampling: one report every 30 min until the pipeline log prints
# PIPELINE_ALL_DONE (then a final report + exit).
# Usage: qa_loop.sh <run_dir> <pipeline_log>
set -u
SYNTH3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SYNTH3_DIR/.." && pwd)"

RUN="${1:-$REPO_ROOT/runs/synth3_pipe}"
PIPE_LOG="${2:-$REPO_ROOT/runs/synth3_pipe_driver.log}"
cd "$SYNTH3_DIR"
while true; do
  python3 qa_sampler.py "$RUN" 2>&1 | grep "QA_REPORT" || echo "QA_SAMPLER_ERROR"
  if grep -q "PIPELINE_ALL_DONE" "$PIPE_LOG" 2>/dev/null; then
    python3 qa_sampler.py "$RUN" 2>&1 | grep "QA_REPORT"
    echo "QA_LOOP_FINISHED (pipeline complete)"
    break
  fi
  sleep 1800
done
