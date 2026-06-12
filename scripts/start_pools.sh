#!/usr/bin/env bash
# Start both jina_pool (127.0.0.1:8766) and serper_pool (127.0.0.1:8765).
# Pools default to *empty* SQLite DBs created next to their packages.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT_DIR/pools/jina_pool/jina_pool_ctl.sh" start
bash "$ROOT_DIR/pools/serper_pool/serper_pool_ctl.sh" start

echo
echo "Pools are up. To make the agent use them:"
echo "  export JINA_READER_BASE_URL=http://127.0.0.1:8766"
echo "  export SERPER_BASE_URL=http://127.0.0.1:8765"
