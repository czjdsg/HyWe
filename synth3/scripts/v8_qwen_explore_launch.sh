#!/usr/bin/env bash
# v8 explore-only run (Stage A only, no B/C/R) — useful for eyeballing what
# a single operator model produces before committing the full pipeline.
# Defaults to qwen3.7-plus on both operator and fact extractor.
set -euo pipefail

SYNTH3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SYNTH3_DIR/.." && pwd)"

POOL="${POOL:-$REPO_ROOT/benchmarks/data/hybrid_pool/hybrid_pool_combined.json}"
OUT="${OUT:-$REPO_ROOT/runs/synth3_v8_qwen_explore}"
LOG="${LOG:-$OUT.log}"

mkdir -p "$OUT"
cd "$SYNTH3_DIR"
[[ -f "$REPO_ROOT/env.local.sh" ]] && source "$REPO_ROOT/env.local.sh"
# Pull in the experiment topology (endpoints, models, timeouts). Override
# with EXPERIMENT_ENV=/path/to/file or skip by setting EXPERIMENT_ENV=NONE.
EXPERIMENT_ENV="${EXPERIMENT_ENV:-$SYNTH3_DIR/scripts/v8_experiment.env}"
if [[ -n "$EXPERIMENT_ENV" && "$EXPERIMENT_ENV" != "NONE" && -f "$EXPERIMENT_ENV" ]]; then
  source "$EXPERIMENT_ENV"
fi
# In explore-only mode prefer a single-model topology unless the experiment env
# already pinned both roles.
export FACT_MODEL="${FACT_MODEL:-qwen3.7-plus}"
export FACT_API_BASE="${FACT_API_BASE:-$EXPLORER_API_BASE}"

export SERPER_BASE_URL="${SERPER_BASE_URL:-http://127.0.0.1:8765}"
export JINA_READER_BASE_URL="${JINA_READER_BASE_URL:-http://127.0.0.1:8766}"
export PROXY_URL="${PROXY_URL:-${TASK_PROXY_URL:-}}"

# Both operator and fact extractor default to qwen3.7-plus on a single endpoint.
export SYNTH3_AGENT="${SYNTH3_AGENT:-v8}"
export EXPLORER_MODEL="${EXPLORER_MODEL:-qwen3.7-plus}"
export EXPLORER_API_BASE="${EXPLORER_API_BASE:-http://localhost:8000/v1}"
export FACT_MODEL="${FACT_MODEL:-qwen3.7-plus}"
export FACT_API_BASE="${FACT_API_BASE:-http://localhost:8000/v1}"
export EXPLORE_DRY="${EXPLORE_DRY:-20}"

nohup python3 synth_driver.py \
  --pool "$POOL" \
  --out "$OUT" \
  --concurrency 30 \
  --explore-only \
  --max-steps 50 \
  --skip-existing \
  --agent v8 \
  > "$LOG" 2>&1 &
echo "PID: $!"
echo "log: $LOG"
echo "out: $OUT"
