#!/usr/bin/env bash
# Judge HyWe(Bench) trajectories against the human-written checklists and score them.
#   bash evaluation/hywebench/judge.sh <run_dir> [out_dir]
# Judge: JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL (paper: kimi-k3). Screenshots
# are rendered with the action coordinate marked and sent inline (base64).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
RUN="${1:?run dir with <task_id>/result.json}"; OUT="${2:-$RUN/hywebench_judge}"
TASKS="${TASKS:-benchmarks/hywebench/hywebench_100.jsonl}"
mkdir -p "$OUT"
# HJ_OP_SEARCH_RULE=0 matches the rules used for the paper's verdicts (the
# "operation item cited only a search step" auto-rejection was added later).
export HJ_OP_SEARCH_RULE="${HJ_OP_SEARCH_RULE:-0}"
# A verdict citing screenshots on an image-less trajectory is re-sampled; on the
# last round such items are rejected instead.
for round in 1 2 3; do
  [ "$round" = 3 ] && export HJ_FABRICATION_POLICY=reject
  "$PY" evaluation/hywebench/hybrid_traj_judge.py --run "$RUN" --tasks "$TASKS" \
    --out "$OUT/verdicts.jsonl" --jpg-dir "$OUT/jpg" --workers "${WORKERS:-6}" \
    --model "${JUDGE_MODEL:-kimi-k3}" --base-url "${JUDGE_BASE_URL:?}" --api-key "${JUDGE_API_KEY:-EMPTY}"
done
"$PY" evaluation/hywebench/score.py "$OUT/verdicts.jsonl" --tasks "$TASKS"
