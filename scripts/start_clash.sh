#!/usr/bin/env bash
# Start the vendored clash (USA topology) on 127.0.0.1:8990 (http) / :8991 (socks).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT_DIR/vendor/clash/clash_ctl.sh" start
echo
echo "Clash up. Point the browser proxy at it:"
echo "  export TASK_PROXY_URL=http://127.0.0.1:8990"
