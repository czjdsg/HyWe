#!/usr/bin/env python3
"""Score operation verdicts against all expected tasks, keeping failures visible.

Use --tasks for the full benchmark denominator. Otherwise all task results in
--run form the denominator. Missing/invalid judgments contribute zero successes
and are reported separately; they are never removed from the denominator.
"""
import argparse
import json
from pathlib import Path


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score(verdicts, expected):
    latest = {}
    for row in verdicts:
        if row.get("predicted_label") in (0, 1):
            latest[row["task_id"]] = row
    success = sum(latest.get(tid, {}).get("predicted_label") == 1 for tid in expected)
    return {"total": len(expected), "judged": len(set(latest) & set(expected)),
            "success": success, "rate": 100 * success / max(1, len(expected))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--name", default="Operation benchmark")
    args = parser.parse_args()
    if args.tasks:
        expected = {row["task_id"] for row in read_rows(args.tasks)}
    else:
        expected = {json.loads(p.read_text())["task_id"] for p in args.run.glob("*/result.json")}
    if not expected:
        parser.error("No expected tasks; provide a non-empty --tasks file or run directory.")
    rows = read_rows(args.verdicts) if args.verdicts.is_file() else []
    result = score(rows, expected)
    print(f"{args.name}: {result['success']}/{result['total']} = {result['rate']:.1f}% success; "
          f"judged {result['judged']}, missing/invalid {result['total'] - result['judged']}")


if __name__ == "__main__":
    main()
