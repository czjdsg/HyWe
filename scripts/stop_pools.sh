#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT_DIR/pools/jina_pool/jina_pool_ctl.sh"   stop || true
bash "$ROOT_DIR/pools/serper_pool/serper_pool_ctl.sh" stop || true
