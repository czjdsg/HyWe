#!/usr/bin/env bash
# HyWeCollector Stage 3c — review & refine the executor's trajectories, then export.
#
# One pass of the production post-processing chain (re-run it as more rollouts
# finish; every step is --resume / incremental). All judge calls go to
# JUDGE_BASE_URL / JUDGE_MODEL (default kimi-k3).
#
#   1  audit      unified two-stage audit: per-phase factual progress + leakage steps,
#                 then task-success / route verdicts (quotes verified verbatim)
#   2  cot        rewrite leaked reasoning in place (facts and decision unchanged)
#   3  site       A1 site fidelity (answer values must come from the named site)
#   4  switch     A1/A2 switch rationale (on the rewritten reasoning)
#   5  coverage   per-field answer coverage
#   6  merge      merge fields obtained in earlier phases into the final answer
#   7  assemble   join all verdicts -> runs/final_pick_$TAG/{survivors.json,rollouts/}
#   8  residue    controller-log driven residual-leak detection + second rewrite
#   9  export     n_sw + 1 multi-turn samples per trajectory  -> runs/train_$TAG.jsonl
#  10  opurity    O-segment purity judge (needs screenshots) + filter -> runs/train_$TAG.opure.jsonl
#
# Usage: source configs/collector.env && TAG=demo bash collector/scripts/stage4_review_refine.sh [step ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
TAG="${TAG:?set TAG, e.g. TAG=demo}"
ROLLOUT="${ROLLOUT:-runs/hybrid_rollout_${TAG}/rollouts}"
INPUT="${INPUT:-runs/hybrid_task_design_${TAG}/input_${TAG}.jsonl}"
AUDIT="runs/unified_audit_${TAG}"
W="${W:-8}"
COMMON=(--rollout-dir "$ROLLOUT" --audit-dir "$AUDIT/tasks" --max-workers "$W" --resume)
: "${JUDGE_API_KEY:?set JUDGE_API_KEY (see configs/collector.env.example)}"
export JUDGE_API_KEY

d() { echo; echo "=== [$TAG] $* ==="; }

step_audit() {
  d "unified audit -> $AUDIT"
  "$PY" collector/audit_trajectory_unified.py --rollout-dir "$ROLLOUT" --input "$INPUT" --out "$AUDIT" \
    --terminations answer --base-url "$JUDGE_BASE_URL" --model "$JUDGE_MODEL" \
    --max-workers "$W" --api-attempts 5 --image-transport base64 --max-image-bytes 1500000 --resume
}
step_cot() {
  d "leaked-reasoning rewrite -> runs/cot_rewrite_$TAG"
  "$PY" collector/rewrite_cot_leaks.py "${COMMON[@]}" --out "runs/cot_rewrite_$TAG"
}
step_site() {
  d "site fidelity (A1) -> runs/site_fidelity_$TAG"
  "$PY" collector/check_site_fidelity.py "${COMMON[@]}" --input "$INPUT" --forms "${FORMS:-F1}" \
    --out "runs/site_fidelity_$TAG"
}
step_switch() {
  d "switch rationale (A1/A2) -> runs/switch_rationale_$TAG"
  "$PY" collector/check_switch_rationale.py "${COMMON[@]}" --input "$INPUT" \
    --exclude "runs/exclude_offsite_mixed_$TAG.json" --cot-rewrite-dir "runs/cot_rewrite_$TAG" \
    --out "runs/switch_rationale_$TAG"
}
step_coverage() {
  d "answer coverage -> runs/answer_coverage_$TAG"
  "$PY" collector/check_answer_coverage.py "${COMMON[@]}" --input "$INPUT" --out "runs/answer_coverage_$TAG"
}
step_merge() {
  "$PY" - "$TAG" <<'PY'
import glob, json, os, sys
T = sys.argv[1]; ids = []
for p in glob.glob(f"runs/answer_coverage_{T}/*.json"):
    try:
        if json.load(open(p)).get("missing_but_available"):
            ids.append(os.path.basename(p)[:-5])
    except Exception:
        pass
json.dump(ids, open(f"runs/merge_targets_{T}.json", "w"), ensure_ascii=False, indent=1)
print(f"merge targets: {len(ids)}")
PY
  d "cross-phase answer merge -> runs/answer_merged_$TAG"
  "$PY" collector/merge_final_answer.py "${COMMON[@]}" --input "$INPUT" \
    --only-ids "runs/merge_targets_$TAG.json" --out "runs/answer_merged_$TAG"
}
step_assemble() {
  # Benchmark tasks must never reach the training set.
  "$PY" -c "import json; json.dump([json.loads(l)['task_id'] for l in open('benchmarks/hywebench/hywebench_100.jsonl') if l.strip()], open('runs/bench_taskids.json','w'))"
  d "assemble survivors -> runs/final_pick_$TAG"
  "$PY" collector/assemble_final_pick.py --tag "$TAG" --rollout "$ROLLOUT" --input "$INPUT" \
    --exclude runs/bench_taskids.json --copy
}
step_residue() {
  d "residual leakage (controller-driven) -> runs/leak_detect_$TAG"
  "$PY" collector/detect_leaks_by_controller.py --rollout-dir "$ROLLOUT" --out "runs/leak_detect_$TAG" \
    --model "$JUDGE_MODEL" --window "${LEAK_WINDOW:-12}" --max-workers "$W" --resume
  "$PY" collector/rewrite_leak_residue.py --rollout-dir "$ROLLOUT" --cot-dir "runs/cot_rewrite_$TAG" \
    --leaks-dir "runs/leak_detect_$TAG" --max-workers "$W"
}
step_export() {
  d "export multi-turn samples -> runs/train_$TAG.jsonl"
  "$PY" training/build_hybrid_teacher_v2.py --rollout-dir "runs/final_pick_$TAG/rollouts" \
    --out "runs/train_$TAG.jsonl" --cot-rewrite-dir "runs/cot_rewrite_$TAG" \
    --answer-merge-dir "runs/answer_merged_$TAG" --image-mode abs
}
step_opurity() {
  d "O-segment purity -> runs/train_$TAG.opure.jsonl"
  "$PY" collector/check_hybrid_o_purity.py --train "runs/train_$TAG.jsonl" --out "runs/o_purity_$TAG" \
    --model "$JUDGE_MODEL" --input "$INPUT" --forms "${O_FORMS:-F2,F3,F4}" \
    --max-workers "$W" --resume
  "$PY" collector/report_hybrid_o_purity.py --judge "runs/o_purity_$TAG" --train "runs/train_$TAG.jsonl" \
    --out "runs/o_purity_${TAG}_report.json"
  "$PY" collector/apply_o_purity_filter.py --train "runs/train_$TAG.jsonl" --judge "runs/o_purity_$TAG" \
    --out "runs/train_$TAG.opure.jsonl" --input "$INPUT" --forms "${O_FORMS:-F2,F3,F4}" --drop-impure-tasks
}

STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(audit cot site switch coverage merge assemble residue export opurity)
for s in "${STEPS[@]}"; do "step_$s" || { echo "step $s failed" >&2; exit 1; }; done
