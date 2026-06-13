#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT_DIR/tools/local_sandbox/sandbox_ctl.sh" stop || true
bash "$ROOT_DIR/scripts/stop_pools.sh" || true
