#!/usr/bin/env bash
# Start every supporting service the inference framework expects:
#   jina_pool   :8766
#   serper_pool :8765
#   local_sandbox :8080
#   clash (USA) :8990 (http) / :8991 (socks) / :9090 (ctl)
#
# Idempotent — each ctl script reports "already running" and exits 0
# when its service is already up.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT_DIR/scripts/start_pools.sh"
bash "$ROOT_DIR/scripts/start_sandbox.sh"
bash "$ROOT_DIR/scripts/start_clash.sh"
echo
echo "All services up. Source configs/inference.env for the env wiring:"
echo "  source $ROOT_DIR/configs/inference.env"
