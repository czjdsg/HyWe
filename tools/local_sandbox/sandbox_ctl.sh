#!/usr/bin/env bash
# tmux-based start/stop/status for the local code sandbox.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$ROOT_DIR/sandbox.pid"
LOG_FILE="$LOG_DIR/sandbox.log"
TMUX_SESSION="hwa_local_sandbox"
DEFAULT_HOST="${SANDBOX_HOST:-127.0.0.1}"
DEFAULT_PORT="${SANDBOX_PORT:-8080}"

mkdir -p "$LOG_DIR"

usage() { echo "Usage: $0 {start|stop|restart|status}"; }

tmux_available() { command -v tmux >/dev/null 2>&1; }
session_running() {
  tmux_available && tmux has-session -t "$TMUX_SESSION" 2>/dev/null
}

is_running() {
  if session_running; then return 0; fi
  [[ -f "$PID_FILE" ]] || return 1
  local pid; pid="$(cat "$PID_FILE" || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_service() {
  if is_running; then echo "local_sandbox already running"; return 0; fi
  : > "$LOG_FILE"
  local env_str="env SANDBOX_HOST='$DEFAULT_HOST' SANDBOX_PORT='$DEFAULT_PORT'"
  if tmux_available; then
    rm -f "$PID_FILE"
    tmux new-session -d -s "$TMUX_SESSION" \
      "cd $ROOT_DIR && $env_str python3 server.py >>$LOG_FILE 2>&1"
    sleep 0.5
    if session_running; then
      echo "local_sandbox started on http://$DEFAULT_HOST:$DEFAULT_PORT (tmux $TMUX_SESSION)"
      echo "log: $LOG_FILE"; return 0
    fi
  else
    nohup env SANDBOX_HOST="$DEFAULT_HOST" SANDBOX_PORT="$DEFAULT_PORT" \
      python3 "$ROOT_DIR/server.py" >>"$LOG_FILE" 2>&1 </dev/null &
    local pid=$!; echo "$pid" > "$PID_FILE"; sleep 0.5
    if kill -0 "$pid" 2>/dev/null; then
      echo "local_sandbox started pid=$pid on http://$DEFAULT_HOST:$DEFAULT_PORT"
      echo "log: $LOG_FILE"; return 0
    fi
  fi
  echo "failed to start local_sandbox; check $LOG_FILE" >&2
  rm -f "$PID_FILE"; return 1
}

stop_service() {
  if session_running; then
    tmux kill-session -t "$TMUX_SESSION"
    rm -f "$PID_FILE"; echo "local_sandbox stopped."; return 0
  fi
  if ! is_running; then
    echo "local_sandbox not running."; rm -f "$PID_FILE"; return 0
  fi
  local pid; pid="$(cat "$PID_FILE")"; kill "$pid"
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then rm -f "$PID_FILE"; echo "stopped."; return 0; fi
    sleep 0.5
  done
  kill -9 "$pid" 2>/dev/null || true; rm -f "$PID_FILE"
}

status_service() {
  if is_running; then echo "running"; return 0; fi
  echo "not running"; return 1
}

case "${1:-}" in
  start)   start_service ;;
  stop)    stop_service ;;
  restart) stop_service || true; start_service ;;
  status)  status_service ;;
  *)       usage; exit 1 ;;
esac
