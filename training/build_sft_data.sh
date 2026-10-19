#!/usr/bin/env bash
# Build the SFT mixture from HyWeCollector exports (+ optional public search data).
#
#   1. concatenate the per-batch exports runs/train_<TAG>.opure.jsonl
#   2. loss masks (only the O side uses judges; S side uses mechanical de-dup rules)
#        O spin         make_hybrid_jpegs.py  -> judge_spin_hybrid.py
#        O coordinates  mark_coord_images.py  -> judge_coord_accuracy.py
#        merge          apply_hybrid_mask.py  (answer / switch turns are never masked)
#   3. align system prompts byte-for-byte with the current inference prompts
#   4. make image paths portable (images/<task_id>/<file>) + write an image manifest
#   5. (optional) convert XYZ-Aquila-SFT search trajectories into the same format
#
# Each sample is one segment: the context rendered by the HyWe renderer, targets =
# that segment's assistant turns (ms-swift reads the per-message `loss` field).
#
# Usage: source configs/collector.env && TAGS="demo" bash training/build_sft_data.sh
#   AQUILA_SRCS="/path/XYZ-Aquila-SFT/en.jsonl /path/zh.jsonl"   (optional)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python}"
OUT="${OUT:-runs/sft}"
mkdir -p "$OUT"
ALL=runs/train_hybrid_all.jsonl

: > "$ALL"
for t in ${TAGS:?set TAGS, e.g. TAGS=\"demo\"}; do cat "runs/train_${t}.opure.jsonl" >> "$ALL"; done
echo "hybrid samples: $(wc -l < "$ALL")"

"$PY" training/make_hybrid_jpegs.py --src "$ALL" --dst runs/hybrid_jpg --side O
"$PY" training/judge_spin_hybrid.py --src "$ALL" --out runs/hybrid_spin.jsonl --jpg-dir runs/hybrid_jpg --side O
"$PY" training/mark_coord_images.py --in "$ALL" --img-dir / --dst runs/coord_marked --index runs/coord_marked_index.jsonl
"$PY" training/judge_coord_accuracy.py --index runs/coord_marked_index.jsonl --image-dir runs/coord_marked \
  --out runs/hybrid_coord_verdicts.jsonl
"$PY" training/apply_hybrid_mask.py --src "$ALL" --coord runs/hybrid_coord_verdicts.jsonl \
  --spin runs/hybrid_spin.jsonl --out runs/train_hybrid_all.masked.jsonl

"$PY" training/align_system_prompt.py --in runs/train_hybrid_all.masked.jsonl --out runs/train_hybrid_all.aligned.jsonl
"$PY" training/normalize_hybrid_images.py --in runs/train_hybrid_all.aligned.jsonl \
  --out "$OUT/hybrid.jsonl" --manifest "$OUT/hybrid_images.jsonl" --copy-images

if [ -n "${AQUILA_SRCS:-}" ]; then
  "$PY" training/build_aquila_v2.py --srcs $AQUILA_SRCS --out "$OUT/search_aquila.jsonl"
fi
echo "done -> $OUT"
