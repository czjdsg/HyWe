#!/usr/bin/env python3
"""Queue a Search partial-rerun after the current full Search process exits."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data/hybrid_search_deepseek_v4_flash_full.jsonl"
IDS_PATH = ROOT / "data/search_rerun_partial_after_full_ids.txt"
INPUTS = "data/openseeker_skeleton_correct.jsonl,data/redsearcher_skeleton_correct.jsonl"


def stage2_processes() -> list[str]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception as exc:
        print(f"[wait] ps failed: {type(exc).__name__}: {exc}", flush=True)
        return []

    mine = os.getpid()
    matches: list[str] = []
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, args = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == mine:
            continue
        if "stage2_rewrite.py" not in args:
            continue
        if str(OUT_PATH) not in args and OUT_PATH.name not in args:
            continue
        matches.append(stripped)
    return matches


def wait_for_current_search() -> None:
    print("[wait] waiting for current Search stage2 process to finish", flush=True)
    last_report = 0.0
    while True:
        matches = stage2_processes()
        if not matches:
            print("[wait] no current Search stage2 process found", flush=True)
            return
        now = time.time()
        if now - last_report >= 300:
            print(f"[wait] still running: {matches[0][:240]}", flush=True)
            last_report = now
        time.sleep(30)


def all_search_traj_ids() -> set[str]:
    sys.path.insert(0, str(ROOT))
    from stage2_rewrite import load_skeleton_grouped

    paths = [p.strip() for p in INPUTS.split(",") if p.strip()]
    by_traj = load_skeleton_grouped(paths)
    return set(by_traj)


def successful_final_ids() -> set[str]:
    done: set[str] = set()
    if not OUT_PATH.exists():
        return done
    with OUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("is_final") and rec.get("rewrite") is not None:
                tid = rec.get("source_traj_id")
                if tid:
                    done.add(tid)
    return done


def write_rerun_ids() -> list[str]:
    all_ids = all_search_traj_ids()
    done_ids = successful_final_ids()
    rerun_ids = sorted(all_ids - done_ids)
    IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDS_PATH.write_text("\n".join(rerun_ids) + ("\n" if rerun_ids else ""), encoding="utf-8")
    print(
        f"[plan] total={len(all_ids)} final_ok={len(done_ids)} rerun={len(rerun_ids)} ids={IDS_PATH}",
        flush=True,
    )
    return rerun_ids


def run_rerun() -> int:
    cmd = [
        sys.executable,
        "-u",
        "stage2_rewrite.py",
        "--inputs",
        INPUTS,
        "--base-url",
        "http://47.82.159.100:8002/v1",
        "--model",
        "deepseek-v4-flash",
        "--api-key",
        "qazxswedcvfrtgbnhy",
        "--out-path",
        str(OUT_PATH),
        "--concurrency",
        "48",
        "--max-retries",
        "5",
        "--temperature",
        "0.0",
        "--retry-temperature",
        "0.7",
        "--traj-ids-file",
        str(IDS_PATH),
    ]
    print("[run] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    wait_for_current_search()
    rerun_ids = write_rerun_ids()
    if not rerun_ids:
        print("[done] no trajectories need rerun", flush=True)
        return 0
    rc = run_rerun()
    print(f"[done] rerun exit_code={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
