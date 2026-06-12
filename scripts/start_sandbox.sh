#!/usr/bin/env bash
# Start the local code-sandbox server on 127.0.0.1:8080.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT_DIR/tools/local_sandbox/sandbox_ctl.sh" start
echo
echo "Sandbox up. Point the agent at it:"
echo "  export CODE_SANDBOX_BASE_URL=http://127.0.0.1:8080"
