#!/usr/bin/env bash
# Stand up an OpenAI-compatible endpoint serving Qwen3.6-35B-A3B via vllm.
#
# Defaults match the production wiring in configs/inference.env:
#   served-model-name = qwen3.6-plus   (the configs/inference.env MODEL value
#                                       is qwen3.7-plus by default; swap one
#                                       of them so they line up before you
#                                       point the agent at this endpoint).
#   --port 8000
#   tensor-parallel-size auto-detected from CUDA_VISIBLE_DEVICES (or all GPUs)
#
# Override everything with env vars, e.g.:
#   QWEN36_PORT=8001 QWEN36_TP=4 QWEN36_NAME=qwen3.6-plus bash scripts/serve_qwen36.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN36_DIR="${QWEN36_DIR:-$ROOT_DIR/models/Qwen3.6-35B-A3B}"

if [[ ! -d "$QWEN36_DIR" ]]; then
  echo "[serve_qwen36] weights not found at $QWEN36_DIR" >&2
  echo "[serve_qwen36] run scripts/download_qwen36.sh first, or" >&2
  echo "               QWEN36_DIR=/your/path bash scripts/serve_qwen36.sh" >&2
  exit 1
fi

QWEN36_NAME="${QWEN36_NAME:-qwen3.6-plus}"
QWEN36_PORT="${QWEN36_PORT:-8000}"
QWEN36_HOST="${QWEN36_HOST:-0.0.0.0}"

# Auto-pick TP from CUDA_VISIBLE_DEVICES when present, else from nvidia-smi.
if [[ -z "${QWEN36_TP:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # shellcheck disable=SC2206
    _gpus=(${CUDA_VISIBLE_DEVICES//,/ })
    QWEN36_TP="${#_gpus[@]}"
  else
    QWEN36_TP="$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)"
    [[ "$QWEN36_TP" -lt 1 ]] && QWEN36_TP=1
  fi
fi

# 35B-A3B is a 256-expert MoE with native 262K ctx. 64K is enough for
# anything HybridWebAgent feeds it and keeps the KV cache reasonable on
# 2xA100. Bump QWEN36_MAX_LEN if you need more.
QWEN36_MAX_LEN="${QWEN36_MAX_LEN:-65536}"
QWEN36_GPU_UTIL="${QWEN36_GPU_UTIL:-0.92}"

echo "[serve_qwen36] model dir       : $QWEN36_DIR"
echo "[serve_qwen36] served name     : $QWEN36_NAME"
echo "[serve_qwen36] endpoint        : http://$QWEN36_HOST:$QWEN36_PORT/v1"
echo "[serve_qwen36] tensor-parallel : $QWEN36_TP"
echo "[serve_qwen36] max-model-len   : $QWEN36_MAX_LEN"
echo "[serve_qwen36] gpu-util        : $QWEN36_GPU_UTIL"

exec vllm serve "$QWEN36_DIR" \
  --served-model-name "$QWEN36_NAME" \
  --host "$QWEN36_HOST" \
  --port "$QWEN36_PORT" \
  --tensor-parallel-size "$QWEN36_TP" \
  --max-model-len "$QWEN36_MAX_LEN" \
  --gpu-memory-utilization "$QWEN36_GPU_UTIL" \
  --trust-remote-code \
  --enable-prefix-caching
