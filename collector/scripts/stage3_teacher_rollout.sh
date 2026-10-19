#!/usr/bin/env bash
# HyWeCollector Stage 3b — coordinated execution (executor + phase controller).
#
# The executor acts through the HyWe framework (--agent-mode hybrid_react_v2) and
# sees only the public task, the current phase goal, verified bindings and
# sanitized controller feedback. The controller (--teacher-phase-control, verifier
# model PHASE_VERIFIER_MODEL) checks each phase's completion condition and gates
# every `switch` and `answer`.
#
# Settings below are the ones used for the paper's collection runs.
#
# Usage: source configs/collector.env && TAG=demo bash collector/scripts/stage3_teacher_rollout.sh [extra run_hybrid_web_agent args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
TAG="${TAG:?set TAG, e.g. TAG=demo}"
INPUT="${INPUT:-runs/hybrid_task_design_${TAG}/input_${TAG}.jsonl}"
OUT="${OUT:-runs/hybrid_rollout_${TAG}}"
BASE="${TEACHER_BASE_URL:?}"; KEY="${TEACHER_API_KEY:-EMPTY}"; MODEL="${TEACHER_MODEL:-kimi-k3}"
VERIFIER="${PHASE_VERIFIER_MODEL:-$MODEL}"

# JSON output mode for the executor (cot in {"cot": ...}, decoded with
# response_format=json_object); exported samples are re-rendered into the
# <think>/<tool_call> training format.
export HYBRID_JSON_OUTPUT="${HYBRID_JSON_OUTPUT:-1}"
# Teacher context budget (large-context frontier models).
export HYBRID_MAX_INPUT_TOKENS="${HYBRID_MAX_INPUT_TOKENS:-750000}"
export HYBRID_FOLD_HIGH="${HYBRID_FOLD_HIGH:-700000}"
export HYBRID_FOLD_LOW="${HYBRID_FOLD_LOW:-500000}"
# Collection-time screenshot transport only (the full frame history is kept on
# disk and re-rendered at export): 1000x1000 WEBP frames, last 10 in context.
export HYBRID_TEACHER_SCREENSHOT_SIZE="${HYBRID_TEACHER_SCREENSHOT_SIZE:-1000}"
export HYBRID_TEACHER_SCREENSHOT_FORMAT="${HYBRID_TEACHER_SCREENSHOT_FORMAT:-WEBP}"
export HYBRID_TEACHER_SCREENSHOT_QUALITY="${HYBRID_TEACHER_SCREENSHOT_QUALITY:-80}"
export HYBRID_MAX_O_SCREENSHOTS="${HYBRID_MAX_O_SCREENSHOTS:-10}"
export HYBRID_TEACHER_SOFT_MAX_STEPS="${HYBRID_TEACHER_SOFT_MAX_STEPS:-80}"
# Survive transient endpoint errors instead of killing long trajectories.
export HYBRID_LLM_MAX_ATTEMPTS="${HYBRID_LLM_MAX_ATTEMPTS:-20}"
export HYBRID_LLM_BACKOFF_CAP="${HYBRID_LLM_BACKOFF_CAP:-60}"
mkdir -p "$OUT"

"$PY" inference/run_hybrid_web_agent.py \
  --input-path "$INPUT" --output-dir "$OUT/rollouts" \
  --agent-mode hybrid_react_v2 \
  --base-url "$BASE" --model "$MODEL" --api-key "$KEY" \
  --routed-search-base-url "$BASE" --routed-search-model "$MODEL" \
  --routed-operation-base-url "$BASE" --routed-operation-model "$MODEL" \
  --teacher-phase-control \
  --teacher-phase-verifier-base-url "$BASE" --teacher-phase-verifier-model "$VERIFIER" \
  --teacher-phase-verifier-api-key "$KEY" \
  --teacher-phase-verifier-timeout 600 \
  --teacher-phase-check-interval "${PHASE_CHECK_INTERVAL:-8}" \
  --teacher-phase-max-actions "${PHASE_MAX_ACTIONS:-70}" \
  --teacher-deep-search-phase-max-actions "${DEEP_SEARCH_PHASE_MAX_ACTIONS:-120}" \
  --teacher-phase-no-progress-actions "${PHASE_NO_PROGRESS_ACTIONS:-3}" \
  --summary-base-url "${SUMMARY_BASE_URL:-$BASE}" --summary-model "${SUMMARY_MODEL:-$MODEL}" \
  --summary-api-key "${SUMMARY_API_KEY:-$KEY}" \
  --code-sandbox-base-url "${CODE_SANDBOX_BASE_URL:-http://127.0.0.1:8080}" \
  --api-timeout 600 --summary-timeout 600 --router-timeout 600 \
  --max-iter "${MAX_ITER:-300}" --max-workers "${MAX_WORKERS:-16}" --max-task-attempts "${MAX_TASK_ATTEMPTS:-1}" \
  --headless --no-css-som --resume \
  ${TASK_PROXY_URL:+--proxy-url "$TASK_PROXY_URL"} \
  "$@"
