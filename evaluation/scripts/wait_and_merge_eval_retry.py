#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
MERGE_SCRIPT = SCRIPT_DIR / "merge_eval_retry_rows.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for retry rows, then merge them into an eval_results.jsonl file.")
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--expected-repeat-count", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    return parser.parse_args()


def load_matching_rows(path: Path, task_id: str, mode: str) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") == task_id and row.get("mode") == mode:
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    deadline = time.time() + max(1, args.timeout_seconds)
    last_count = -1
    while time.time() < deadline:
        rows = load_matching_rows(args.retry, args.task_id, args.mode)
        if len(rows) != last_count:
            print(f"retry rows: {len(rows)}/{args.expected_repeat_count}", flush=True)
            last_count = len(rows)
        repeat_indexes = set()
        for row in rows:
            try:
                repeat_indexes.add(int(row.get("repeat_index") or 1))
            except Exception:
                repeat_indexes.add(1)
        if all(index in repeat_indexes for index in range(1, args.expected_repeat_count + 1)):
            break
        time.sleep(max(1, args.poll_seconds))
    else:
        raise SystemExit(f"Timed out waiting for {args.expected_repeat_count} retry rows in {args.retry}")

    cmd = [
        sys.executable,
        str(MERGE_SCRIPT),
        "--main",
        str(args.main),
        "--retry",
        str(args.retry),
        "--backup",
        str(args.backup),
        "--summary",
        str(args.summary),
        "--task-id",
        args.task_id,
        "--mode",
        args.mode,
        "--expected-repeat-count",
        str(args.expected_repeat_count),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
