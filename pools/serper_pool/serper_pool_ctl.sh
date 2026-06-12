#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$ROOT_DIR/serper_pool.pid"
LOG_FILE="$LOG_DIR/serper_pool.log"
CTL_ERR_FILE="$LOG_DIR/serper_pool_ctl.err"
TMUX_SESSION="serper_pool"
DEFAULT_DB="$ROOT_DIR/serper_pool.db"
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="8765"

mkdir -p "$LOG_DIR"

usage() {
  echo "Usage: $0 {start|stop|restart|status}"
}

tmux_available() {
  command -v tmux >"$CTL_ERR_FILE" 2>&1
}

session_running() {
  if ! tmux_available; then
    return 1
  fi

  tmux has-session -t "$TMUX_SESSION" 2>"$CTL_ERR_FILE"
}

is_running() {
  if session_running; then
    return 0
  fi

  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi

  local pid
  pid="$(cat "$PID_FILE" || true)"
  if [[ -z "${pid}" ]]; then
    return 1
  fi

  kill -0 "$pid" 2>"$CTL_ERR_FILE"
}

start_service() {
  if is_running; then
    if session_running; then
      echo "serper_pool is already running in tmux session $TMUX_SESSION."
    else
      echo "serper_pool is already running with pid $(cat "$PID_FILE")."
    fi
    return 0
  fi

  local host="${SERPER_POOL_HOST:-$DEFAULT_HOST}"
  local port="${SERPER_POOL_PORT:-$DEFAULT_PORT}"
  local db="${SERPER_POOL_DB:-$DEFAULT_DB}"

  : > "$LOG_FILE"
  if tmux_available; then
    rm -f "$PID_FILE"
    tmux new-session -d -s "$TMUX_SESSION" \
      "cd $ROOT_DIR && env SERPER_POOL_HOST='$host' SERPER_POOL_PORT='$port' SERPER_POOL_DB='$db' python3 $ROOT_DIR/server.py >>$LOG_FILE 2>&1"
    sleep 1
    if session_running; then
      echo "serper_pool started on http://$host:$port in tmux session $TMUX_SESSION"
      echo "log: $LOG_FILE"
      return 0
    fi
  else
    nohup env \
      SERPER_POOL_HOST="$host" \
      SERPER_POOL_PORT="$port" \
      SERPER_POOL_DB="$db" \
      python3 "$ROOT_DIR/server.py" >>"$LOG_FILE" 2>&1 </dev/null &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 1

    if kill -0 "$pid" 2>"$CTL_ERR_FILE"; then
      echo "serper_pool started on http://$host:$port with pid $pid"
      echo "log: $LOG_FILE"
      return 0
    fi
  fi

  echo "failed to start serper_pool; check $LOG_FILE" >&2
  rm -f "$PID_FILE"
  return 1
}

stop_service() {
  if session_running; then
    tmux kill-session -t "$TMUX_SESSION"
    rm -f "$PID_FILE"
    echo "serper_pool stopped."
    return 0
  fi

  if ! is_running; then
    echo "serper_pool is not running."
    rm -f "$PID_FILE"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid"
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>"$CTL_ERR_FILE"; then
      rm -f "$PID_FILE"
      echo "serper_pool stopped."
      return 0
    fi
    sleep 0.5
  done

  echo "serper_pool did not stop gracefully; sending SIGKILL."
  kill -9 "$pid" 2>"$CTL_ERR_FILE" || true
  rm -f "$PID_FILE"
}

status_service() {
  if session_running; then
    echo "serper_pool is running in tmux session $TMUX_SESSION"
    echo "log file: $LOG_FILE"
    return 0
  fi

  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "serper_pool is running with pid $pid"
    echo "pid file: $PID_FILE"
    echo "log file: $LOG_FILE"
    return 0
  fi

  echo "serper_pool is not running."
  return 1
}

case "${1:-}" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service || true
    start_service
    ;;
  status)
    status_service
    ;;
  *)
    usage
    exit 1
    ;;
esac
