#!/usr/bin/env bash
# Full v8 pipeline launch (example). Override the path-y env vars below for
# your environment; everything else inherits sane defaults from
# synth3/pipeline_driver.py.
#
# Model wiring (the original run used a 2-endpoint topology):
#   A operator      : qwen3.7-plus   (vision, 0-1000 logical coords)
#   A fact extractor: kimi-k2.6      (vision)
#   B search probe  : qwen3.7-max
#   C task synth    : aws.claude-opus-4-7
#   R operator      : qwen3.7-plus
#   R search        : qwen3.7-max
set -euo pipefail

SYNTH3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SYNTH3_DIR/.." && pwd)"

# ----- inputs / outputs ---------------------------------------------------
# Point POOL at a JSON pool of seed sites (see synth3/README.md for schema).
POOL="${POOL:-$REPO_ROOT/benchmarks/data/hybrid_pool/hybrid_pool_combined.json}"
OUT="${OUT:-$REPO_ROOT/runs/synth3_v8_full1170}"
LOG="${LOG:-$OUT.log}"

mkdir -p "$OUT"
cd "$SYNTH3_DIR"
# Pulls in JINA / SERPER pool URLs + keys. Required.
[[ -f "$REPO_ROOT/env.local.sh" ]] && source "$REPO_ROOT/env.local.sh"
# Pull in the experiment topology (endpoints, models, timeouts). Override
# with EXPERIMENT_ENV=/path/to/file or skip by setting EXPERIMENT_ENV=NONE.
EXPERIMENT_ENV="${EXPERIMENT_ENV:-$SYNTH3_DIR/scripts/v8_experiment.env}"
if [[ -n "$EXPERIMENT_ENV" && "$EXPERIMENT_ENV" != "NONE" && -f "$EXPERIMENT_ENV" ]]; then
  source "$EXPERIMENT_ENV"
fi

# Pool endpoints (defaults match scripts/start_pools.sh).
export SERPER_BASE_URL="${SERPER_BASE_URL:-http://127.0.0.1:8765}"
export JINA_READER_BASE_URL="${JINA_READER_BASE_URL:-http://127.0.0.1:8766}"

# Playwright in explorer_runtime reads PROXY_URL. Without it, Chromium goes
# direct — many sites region-wall the IP. Leave empty to skip.
export PROXY_URL="${PROXY_URL:-${TASK_PROXY_URL:-}}"

# A-stage explorer settings
export EXPLORE_DRY="${EXPLORE_DRY:-20}"
export SYNTH3_AGENT="${SYNTH3_AGENT:-v8}"
export PIPE_T_A="${PIPE_T_A:-$((150 * 60))}"  # 150 min/site
export EXPLORER_MODEL="${EXPLORER_MODEL:-qwen3.7-plus}"
export EXPLORER_API_BASE="${EXPLORER_API_BASE:-http://localhost:8000/v1}"
export FACT_MODEL="${FACT_MODEL:-kimi-k2.6}"
export FACT_API_BASE="${FACT_API_BASE:-http://localhost:8000/v1}"

# C-stage synth model
export SYNTH_MODEL="${SYNTH_MODEL:-qwen3.7-max}"

# R-stage operator model (pipeline_driver reads this)
export R_OPERATION_MODEL="${R_OPERATION_MODEL:-qwen3.7-plus}"

nohup python3 pipeline_driver.py \
  --pool "$POOL" \
  --out "$OUT" \
  --max-steps 50 \
  --n-tasks-cap 5 \
  --skip-existing \
  --agent v8 \
  --sem-a 30 \
  --sem-b 12 \
  --sem-c 2 \
  --sem-r 15 \
  --r-workers 2 \
  > "$LOG" 2>&1 &
echo "PID: $!"
echo "log: $LOG"
echo "out: $OUT"
