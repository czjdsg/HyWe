#!/usr/bin/env bash
# Deep-search benchmarks: exact match, otherwise 5 judge votes at temperature 0.6,
# majority decides. Paper judge: kimi-k3.
#   bash evaluation/search/judge.sh <run_dir> [out_dir]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
RUN="${1:?run dir}"; OUT="${2:-$RUN/eval}"
"$PY" evaluation/search/evaluate_reference_answers.py --run-dir "$RUN" --output-dir "$OUT" \
  --base-url "${JUDGE_BASE_URL:?}" --api-key "${JUDGE_API_KEY:-EMPTY}" --model "${JUDGE_MODEL:-kimi-k3}" \
  --votes 5 --vote-temperature 0.6 --max-workers "${WORKERS:-8}" --resume
