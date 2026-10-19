#!/usr/bin/env bash
# HyWeCollector Stage 1 — task prototypes from capability probing.
#
#   explore (operator + fact extractor, real browser)  ->  value filter
#   -> search probe (S agent, budget PROBE_MAX_ITER)   ->  refilter  ->  heuristic labels
#
# Output: $OUT/<domain>/{result.json, fact_tags.json, facts_refiltered.json, facts_labeled.json}
# The labelled fact pool is what the S/O task prototypes in
# collector/hybrid_task_design.seed.json were abstracted from.
#
# Usage: source configs/collector.env && bash collector/scripts/stage1_capability_probing.sh
#   POOL=collector/data/insta_screened.pool.json  OUT=runs/stage1  LIMIT=0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
POOL="${POOL:-collector/data/insta_screened.pool.json}"
OUT="${OUT:-${STAGE1_RUN_DIR:-runs/stage1}}"
export STAGE1_RUN_DIR="$OUT"
mkdir -p "$OUT"

"$PY" collector/stage1_probe.py --pool "$POOL" --out "$OUT" \
  --agent v8 --value-filter --skip-existing \
  --max-steps "${MAX_STEPS:-40}" --sem-a "${SEM_A:-16}" --sem-b "${SEM_B:-8}" --b-conc "${B_CONC:-4}" \
  --limit "${LIMIT:-0}"

# Second, stricter per-fact filter and stability labels over the tagged pool.
"$PY" collector/refilter_pool.py --run-dir "$OUT" --all --conc "${CONC:-8}"
"$PY" collector/heuristic_label.py --run-dir "$OUT" --all --conc "${CONC:-8}"
