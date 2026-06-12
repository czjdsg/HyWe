#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for a retry eval output and copy it back into the main output dir.")
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--retry-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--done-log", type=Path, default=None)
    parser.add_argument("--backup-suffix", default="")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def log_done(path: Path | None) -> bool:
    if path is None:
        return True
    if not path.exists():
        return False
    return "Saved evaluation results to" in path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    args = parse_args()
    retry_results = args.retry_dir / "eval_results.jsonl"
    retry_summary = args.retry_dir / "summary.json"
    main_results = args.main_dir / "eval_results.jsonl"
    main_summary = args.main_dir / "summary.json"
    deadline = time.time() + max(1, args.timeout_seconds)
    last_count = -1

    while time.time() < deadline:
        count = count_jsonl(retry_results)
        if count != last_count:
            print(f"retry rows: {count}/{args.expected_rows}", flush=True)
            last_count = count
        if count >= args.expected_rows and log_done(args.done_log):
            break
        time.sleep(max(1, args.poll_seconds))
    else:
        raise SystemExit(f"Timed out waiting for retry output: {retry_results}")

    suffix = args.backup_suffix or datetime.now().strftime("before_retry_merge_%Y%m%d_%H%M%S")
    if main_results.exists():
        shutil.copy2(main_results, args.main_dir / f"eval_results.{suffix}.jsonl")
    if main_summary.exists():
        shutil.copy2(main_summary, args.main_dir / f"summary.{suffix}.json")

    shutil.copy2(retry_results, main_results)
    if retry_summary.exists():
        shutil.copy2(retry_summary, main_summary)
    print(f"merged retry output into {args.main_dir}", flush=True)


if __name__ == "__main__":
    main()
