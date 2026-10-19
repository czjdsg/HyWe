#!/usr/bin/env python3
"""Count finished tasks in a run directory / clear unfinished ones before resuming.

result.json is written while a task runs (termination="running"), and
run_hybrid_web_agent.py --resume skips every task that already has one, so
half-finished or infrastructure-failed tasks must be removed before re-running.

    python scripts/resume_state.py <run_dir> done    # print #finished tasks
    python scripts/resume_state.py <run_dir> clean   # delete running / task_error results
"""
import glob
import json
import os
import sys


def termination(path):
    try:
        return json.load(open(path, encoding="utf-8")).get("termination")
    except Exception:
        return None


def main():
    out, op = sys.argv[1], sys.argv[2]
    files = glob.glob(os.path.join(out, "*", "result.json"))
    unfinished = (None, "running", "task_error")
    if op == "done":
        print(sum(1 for f in files if termination(f) not in unfinished))
    elif op == "clean":
        k = 0
        for f in files:
            if termination(f) in unfinished:
                try:
                    os.remove(f); k += 1
                except OSError:
                    pass
        print(f"[clean] removed {k} unfinished results; --resume will re-run them")
    else:
        sys.exit(f"unknown op {op}")


if __name__ == "__main__":
    main()
