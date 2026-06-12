#!/usr/bin/env bash
# One-shot installer for HybridWebAgent.
# Idempotent: re-running is safe.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[install] repo root: $ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "[install] creating virtualenv at .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[install] upgrading pip"
pip install --upgrade pip >/dev/null

echo "[install] installing python deps"
pip install -r requirements.txt

echo "[install] installing Playwright Chromium"
python -m playwright install chromium

echo "[install] done."
echo
echo "Next:"
echo "  bash scripts/start_pools.sh        # start jina + serper key pools"
echo "  python pools/jina_pool/manage_keys.py  add JINA_KEY"
echo "  python pools/serper_pool/manage_keys.py add SERPER_KEY"
echo "  see INSTALL.md for smoke-test commands."
