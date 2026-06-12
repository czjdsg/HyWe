#!/usr/bin/env bash
SYNTH3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SYNTH3_DIR/.." && pwd)"

RUN="${VIZ_RUN:-$REPO_ROOT/runs/synth3_pipe}"
VIZ="${VIZ_OUT:-$REPO_ROOT/runs/synth3_pipe_viz}"
PIPE_LOG="${VIZ_PIPE_LOG:-$REPO_ROOT/runs/synth3_pipe_driver.log}"
cd "$SYNTH3_DIR"
while true; do
  python3 build_pipe_viz.py --run "$RUN" --out "$VIZ" >/dev/null 2>&1
  # relink any new site dirs
  for d in "$RUN"/*/; do
    dn=$(basename "$d"); [ "$dn" = "_logs" -o "$dn" = "shims" ] && continue
    ln -snf "$d" "$VIZ/$dn" 2>/dev/null
  done
  grep -q "PIPELINE_ALL_DONE" "$PIPE_LOG" 2>/dev/null && { python3 build_pipe_viz.py --run "$RUN" --out "$VIZ" >/dev/null 2>&1; break; }
  sleep 30
done
