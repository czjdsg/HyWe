#!/usr/bin/env bash
# HyWeCollector Stage 2/3a — switching patterns + task synthesis.
#
# hybrid_task_design_pipeline.py
#   catalog     embed the S/O prototypes (Jina embeddings) for semantic pairing
#   generate    sample blueprints b=(pattern, sites, entity, composition, difficulty)
#               deterministically from --seed; no LLM call
#   naturalize  LLM writes the public question x=N(b) + per-phase goals; 9 validators,
#               rejected drafts are re-sampled with the failure reason
# prepare_hybrid_rollout_inputs.py folds the private plan C(b) into
# `private_teacher_steps`; only O-start tasks get a start URL.
#
# Usage: source configs/collector.env && TAG=demo COUNT=200 bash collector/scripts/stage2_task_design.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
TAG="${TAG:?set TAG, e.g. TAG=demo}"
DESIGN="runs/hybrid_task_design_${TAG}"

"$PY" collector/hybrid_task_design_pipeline.py --mode all \
  --count "${COUNT:-5000}" --seed "${SEED:-29}" \
  --site-pool "${SITE_POOL:-collector/data/hybrid_pool_insta2k_expanded.json}" \
  --site-detail "${SITE_DETAIL:-collector/data/insta_screened.detail.jsonl}" \
  --out-dir "$DESIGN" \
  --text-base-url "$SYNTH_BASE" --api-key "$SYNTH_API_KEY" --text-model "$SYNTH_MODEL" \
  --workers "${WORKERS:-8}" --resume

# De-duplicate against earlier batches (OLD_TASKS, optional) and against HyWeBench.
"$PY" - "$DESIGN/tasks.jsonl" "${OLD_TASKS:-}" "$DESIGN/tasks_dedup.jsonl" <<'PYX'
import json, sys
new_p, old_p, out_p = sys.argv[1:4]
old_q, old_id = set(), set()
paths = [p for p in (old_p, "benchmarks/hywebench/hywebench_100.jsonl") if p]
for p in paths:
    for l in open(p, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            old_q.add((d.get("question") or "").strip().lower()); old_id.add(d.get("task_id"))
keep, drop, seen = [], 0, set()
for l in open(new_p, encoding="utf-8"):
    if not l.strip():
        continue
    d = json.loads(l)
    q = (d.get("question") or "").strip().lower()
    if q in old_q or d.get("task_id") in old_id or q in seen:
        drop += 1; continue
    seen.add(q); keep.append(d)
with open(out_p, "w", encoding="utf-8") as f:
    for d in keep:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"tasks {len(keep) + drop} -> {len(keep)} after de-duplication")
PYX

"$PY" collector/prepare_hybrid_rollout_inputs.py \
  --tasks "$DESIGN/tasks_dedup.jsonl" --out "$DESIGN/input_${TAG}.jsonl" --max-iter "${MAX_ITER:-300}"
