#!/usr/bin/env bash
# WebVoyager auto-evaluator adapted to hybrid trajectories: screenshots and text
# observations merged into one timeline, last k=15 passed to the judge. Paper judge: GPT-4o.
#   OPENAI_API_KEY=... [OPENAI_BASE_URL=...] bash evaluation/webvoyager/judge.sh <run_dir> [out.jsonl]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
RUN="${1:?run dir}"; OUT="${2:-$RUN/wv_eval/results.jsonl}"
mkdir -p "$(dirname "$OUT")"
"$PY" evaluation/webvoyager/run_wv_eval.py --run-dir "$RUN" --out "$OUT" --model "${WV_JUDGE_MODEL:-gpt-4o}" \
  --base-url "${OPENAI_BASE_URL:-https://api.openai.com/v1}" --api-key "${OPENAI_API_KEY:?}" \
  --k 15 --workers "${WORKERS:-6}" --timeout 600
"$PY" evaluation/score_operation.py "$OUT" --run "$RUN" \
  ${TASKS:+--tasks "$TASKS"} --name WebVoyager
