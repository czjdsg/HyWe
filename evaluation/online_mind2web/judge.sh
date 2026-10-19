#!/usr/bin/env bash
# Online-Mind2Web: WebJudge adapted to hybrid trajectories (text observations are
# scored on the same 1-5 relevance scale as screenshots; threshold 3). Paper judge: o4-mini.
#   OPENAI_API_KEY=... [OPENAI_BASE_URL=...] bash evaluation/online_mind2web/judge.sh <run_dir> [out_dir]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
RUN="${1:?run dir}"; OUT="${2:-$RUN/webjudge}"
JUDGE="${M2W_JUDGE_MODEL:-o4-mini}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
RUN=$("$PY" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$RUN")
OUT=$("$PY" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$OUT")
mkdir -p "$OUT/out"
"$PY" evaluation/online_mind2web/adapt_trajectories.py "$RUN" "$OUT/trajectories"
# o-series judges spend output budget on reasoning: request a large max_completion_tokens.
export JUDGE_MIN_COMPLETION_TOKENS="${JUDGE_MIN_COMPLETION_TOKENS:-16000}" WEBJUDGE_JPEG_QUALITY=85
(cd evaluation/online_mind2web && "$PY" run.py --mode WebJudge_Online_Mind2Web_eval --model "$JUDGE" \
  --trajectories_dir "$OUT/trajectories" --api_key "$OPENAI_API_KEY" \
  --output_path "$OUT/out" --score_threshold 3 --num_worker "${WORKERS:-4}")
"$PY" evaluation/score_operation.py \
  "$OUT/out/WebJudge_Online_Mind2Web_eval_${JUDGE}_score_threshold_3_auto_eval_results.json" \
  --run "$RUN" ${TASKS:+--tasks "$TASKS"} --name Online-Mind2Web
