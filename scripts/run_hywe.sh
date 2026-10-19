#!/usr/bin/env bash
# Run the HyWe agent over a task JSONL with the paper's inference settings.
#
#   bash scripts/run_hywe.sh <tasks.jsonl> <out_dir> [operation|search] [extra run_hybrid_web_agent args]
#
# Profile (3rd arg) only sets the step budget: operation = 100 (Online-Mind2Web,
# WebVoyager, HyWeBench), search = 300 (HLE, xbench-DeepSearch, BrowseComp).
# The start mode follows each task: a task with a `web` start URL starts in the
# operation-based mode, otherwise in the search-based mode.
#
# Requires `source configs/hywe.env`. The loop re-launches the run until every
# task has finished (unfinished results are cleared first, see resume_state.py).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IN="${1:?tasks.jsonl}"; OUT="${2:?output dir}"; PROFILE="${3:-operation}"; shift $(( $# >= 3 ? 3 : $# ))
PY="${PY:-python}"
case "$PROFILE" in
  operation) MAX_ITER="${MAX_ITER:-100}" ;;
  search)    MAX_ITER="${MAX_ITER:-300}" ;;
  *) echo "profile must be operation|search" >&2; exit 2 ;;
esac
: "${BASE_URL:?source configs/hywe.env first}"
: "${MODEL:?source configs/hywe.env first}"
[ -s "$IN" ] || { echo "Task file is empty or missing: $IN" >&2; exit 2; }
TOTAL=$(grep -c . "$IN")
for round in $(seq 1 "${MAX_ROUNDS:-10}"); do
  n=$("$PY" scripts/resume_state.py "$OUT" done 2>/dev/null || echo 0)
  echo "=== round $round: $n/$TOTAL finished ==="
  [ "$n" -ge "$TOTAL" ] && break
  "$PY" scripts/resume_state.py "$OUT" clean 2>/dev/null || true
  "$PY" inference/run_hybrid_web_agent.py \
    --input-path "$IN" --output-dir "$OUT" --agent-mode hybrid_react_v2 \
    --base-url "$BASE_URL" --model "$MODEL" --api-key "${API_KEY:-EMPTY}" \
    --summary-base-url "${SUMMARY_BASE_URL:-$BASE_URL}" --summary-model "${SUMMARY_MODEL:-$MODEL}" \
    --summary-api-key "${SUMMARY_API_KEY:-${API_KEY:-EMPTY}}" \
    --code-sandbox-base-url "${CODE_SANDBOX_BASE_URL:-http://127.0.0.1:8080}" \
    ${TASK_PROXY_URL:+--proxy-url "$TASK_PROXY_URL"} \
    --api-timeout 1800 --summary-timeout 1800 \
    --max-tokens 8192 --max-format-retries 5 --max-task-attempts 2 \
    --max-iter "$MAX_ITER" --max-iter-restarts 5 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --repetition-penalty 1.05 \
    --max-observation-chars 100000 --max-workers "${MAX_WORKERS:-8}" \
    --resume --headless "$@"
done
n=$("$PY" scripts/resume_state.py "$OUT" done)
echo "=== done: $n/$TOTAL ==="
[ "$n" -ge "$TOTAL" ] || { echo "Run incomplete after ${MAX_ROUNDS:-10} rounds; resume with the same command." >&2; exit 1; }
