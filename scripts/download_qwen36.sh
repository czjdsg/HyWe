#!/usr/bin/env bash
# Download Qwen3.6-35B-A3B weights via the hf-mirror.com CDN.
#
# About 72 GB, 26 safetensors shards. The direct huggingface.co route times
# out from most China-side hosts; hf-mirror.com is reliable but requires
# bypassing any HTTP proxy you might have set (the mirror itself does not
# expect proxy traffic).
#
# Override the destination with QWEN36_DIR=/path/to/weights bash <this>.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN36_DIR="${QWEN36_DIR:-$ROOT_DIR/models/Qwen3.6-35B-A3B}"
REPO_ID="${QWEN36_REPO:-Qwen/Qwen3.6-35B-A3B}"
WORKERS="${HF_WORKERS:-8}"

mkdir -p "$QWEN36_DIR"

# Strip every proxy env var for this call only — hf-mirror is reached
# directly. Keep them set in the parent shell otherwise.
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
    HF_ENDPOINT=https://hf-mirror.com \
    hf download "$REPO_ID" \
      --local-dir "$QWEN36_DIR" \
      --max-workers "$WORKERS"

echo
echo "Downloaded $REPO_ID -> $QWEN36_DIR"
du -sh "$QWEN36_DIR"
