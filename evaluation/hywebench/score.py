#!/usr/bin/env python3
"""Aggregate HyWeBench verdicts (output of hybrid_traj_judge.py) into the paper metrics.

  Success  tasks whose checklist items are all achieved (%)
  Info.    achieved information items / all information items (%)
  Op.      achieved operation items / all operation items (%)

Totals are taken over the full benchmark (every task in --tasks); a task without
a successful verdict counts as failed with no achieved items. When a verdict file
holds several rows for a task (re-judging), the last `ok` row wins.

Usage:
  python evaluation/hywebench/score.py runs/hywebench_judged/verdicts.jsonl
  python evaluation/hywebench/score.py evaluation/hywebench/paper_verdicts/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TASKS = os.path.join(HERE, "..", "..", "benchmarks", "hywebench", "hywebench_100.jsonl")


def score(path: str, tasks: list[dict]) -> dict:
    best = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("ok"):
                best[row["task_id"]] = row
    n_info = sum(1 for t in tasks for s in t["checklist"] if s["kind"] == "information")
    n_op = sum(1 for t in tasks for s in t["checklist"] if s["kind"] == "operation")
    done_info = sum(best.get(t["task_id"], {}).get("n_info_done", 0) for t in tasks)
    done_op = sum(best.get(t["task_id"], {}).get("n_op_done", 0) for t in tasks)
    success = sum(1 for t in tasks if best.get(t["task_id"], {}).get("complete"))
    return {
        "judged": sum(1 for t in tasks if t["task_id"] in best),
        "success": 100.0 * success / len(tasks),
        "info": 100.0 * done_info / max(1, n_info),
        "op": 100.0 * done_op / max(1, n_op),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verdicts", nargs="+")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    print(f"{'run':40s} {'judged':>6s} {'Success':>8s} {'Info.':>7s} {'Op.':>7s}")
    for p in a.verdicts:
        s = score(p, tasks)
        name = os.path.splitext(os.path.basename(p))[0]
        if name == "verdicts":
            name = os.path.basename(os.path.dirname(os.path.abspath(p)))
        print(f"{name:40s} {s['judged']:>6d} {s['success']:>8.1f} {s['info']:>7.1f} {s['op']:>7.1f}")


if __name__ == "__main__":
    main()
