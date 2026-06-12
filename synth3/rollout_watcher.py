#!/usr/bin/env python3
"""Periodic watcher: scan synth3 outputs, batch new candidates to rollout.

Every PERIOD seconds:
  1. Scan /tmp/synth3_full_v6/<domain>/result.json
  2. Pick candidates whose domain is not yet in seen.txt
  3. Convert to run_hybrid_web_agent input format
  4. Launch run_hybrid_web_agent.py on this batch (background)
  5. Mark domains as seen

Rollout jobs run concurrently — each batch gets its own output dir under
/data/czj/trajectory_quality_filter/outputs/synth3_rollout_v6/batch_<idx>/

Resource share:
  - explorer (synth3 driver) uses :8000 qwen3.7-plus for ACT and :8002 Claude for RECORD
  - rollout uses :8002 qwen3.7-plus, max-workers=32
  - Claude RECORD calls are sparse (~2/min total at conc=48), rollout is dense — RECORD
    won't be starved.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SYNTH3 = Path(__file__).resolve().parent
REPO_ROOT = SYNTH3.parent
SYNTH_OUT_DIR = Path(os.environ.get("WATCHER_SYNTH_OUT_DIR", "/tmp/synth3_full_v6"))
SEEN_FILE = Path(os.environ.get("WATCHER_SEEN_FILE", "/tmp/rollout_watcher_v6_seen.txt"))
BATCH_ROOT = Path(os.environ.get(
    "WATCHER_BATCH_ROOT", str(REPO_ROOT / "runs" / "synth3_rollout")))
HYBRID_AGENT = Path(os.environ.get(
    "HYBRID_AGENT_ENTRY",
    str(REPO_ROOT / "inference" / "run_hybrid_web_agent.py"),
))

# Rollout config — override via env vars when serving from your own endpoints.
ROLLOUT_BASE_URL = os.environ.get("ROLLOUT_BASE_URL", "http://localhost:8000/v1")
ROLLOUT_API_KEY = os.environ.get("ROLLOUT_API_KEY", "EMPTY")
ROLLOUT_MODEL = os.environ.get("ROLLOUT_MODEL", "qwen3.7-plus")
ROUTED_SEARCH_MODEL = os.environ.get("ROUTED_SEARCH_MODEL", "glm-5.1")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "deepseek-v4-flash")
ROLLOUT_TOKENIZER_PATH = os.environ.get("ROLLOUT_TOKENIZER_PATH", "")
ROLLOUT_PROXY_URL = os.environ.get("ROLLOUT_PROXY_URL", "")
ROLLOUT_SANDBOX_BASE_URL = os.environ.get(
    "ROLLOUT_SANDBOX_BASE_URL", "http://127.0.0.1:8080")


def _load_seen() -> set[str]:
    if not SEEN_FILE.exists():
        return set()
    return {line.strip() for line in SEEN_FILE.read_text().splitlines() if line.strip()}


def _append_seen(domains: list[str]) -> None:
    with SEEN_FILE.open("a") as f:
        for d in domains:
            f.write(d + "\n")


def _scan_new_candidates(seen: set[str]) -> list[dict]:
    """Walk synth3 output dirs, return list of candidates from sites whose
    `domain` is not in `seen` and whose result.json exists + has candidates."""
    if not SYNTH_OUT_DIR.is_dir():
        return []
    rows: list[dict] = []
    for site_dir in sorted(SYNTH_OUT_DIR.iterdir()):
        if not site_dir.is_dir():
            continue
        domain = site_dir.name  # e.g. "google.com_flights" with slashes mangled
        if domain in seen:
            continue
        rp = site_dir / "result.json"
        if not rp.is_file():
            continue
        try:
            r = json.load(rp.open())
        except Exception:
            continue
        cands = r.get("candidates") or []
        if not cands:
            # No candidates produced — still mark seen so we don't re-scan.
            seen.add(domain)
            continue
        for i, c in enumerate(cands):
            q = (c.get("question") or "").strip()
            rp_list = c.get("route_plan")
            if not q or not rp_list:
                continue
            row = {
                "task_id": f"synth3-{domain}-{i}",
                "subset": r.get("site", {}).get("category", "synth3"),
                "question": q,
                "web": r.get("site", {}).get("entry_url", "") or "",
                "dataset": "synth3",
                "split": "test",
                "record_id": f"synth3-{domain}-{i}",
                "answer": "",
                "start_mode": c.get("start_mode") or "O",
                "route_plan": json.dumps(rp_list, ensure_ascii=False),
                "_signal": json.dumps({
                    "switch_kind": c.get("switch_kind", ""),
                    "image_required": c.get("image_required", False),
                    "category": r.get("site", {}).get("category", ""),
                }, ensure_ascii=False),
                "_domain": domain,
            }
            rows.append(row)
    return rows


def _launch_rollout(batch_idx: int, rows: list[dict], dry_run: bool) -> tuple[int | None, Path]:
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    bd = BATCH_ROOT / f"batch_{batch_idx:04d}"
    bd.mkdir(exist_ok=True)
    input_path = bd / "input.jsonl"
    with input_path.open("w") as f:
        for r in rows:
            # Strip the _domain helper before writing
            r2 = {k: v for k, v in r.items() if not k.startswith("_") or k == "_signal"}
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    results_path = bd / "results.jsonl"
    log_path = bd / "rollout.log"

    cmd = [
        sys.executable, str(HYBRID_AGENT),
        "--input-path", str(input_path),
        "--output-dir", str(bd),
        "--output-path", str(results_path),
        "--agent-mode", "hybrid_guided_routed",
        "--max-workers", "32",
        "--max-task-attempts", "1",
        "--max-iter", "100",
        "--headless",
        "--window-width", "1280",
        "--window-height", "720",
        "--coord-scale", "1000",
        "--base-url", ROLLOUT_BASE_URL,
        "--model", ROLLOUT_MODEL,
        "--api-key", ROLLOUT_API_KEY,
        "--api-timeout", "900",
        "--routed-search-model", ROUTED_SEARCH_MODEL,
        "--routed-search-base-url", ROLLOUT_BASE_URL,
        "--routed-operation-model", ROLLOUT_MODEL,
        "--routed-operation-base-url", ROLLOUT_BASE_URL,
        "--router-model", ROLLOUT_MODEL,
        "--router-base-url", ROLLOUT_BASE_URL,
        "--router-api-key", ROLLOUT_API_KEY,
        "--router-timeout", "900",
        "--router-max-tokens", "16000",
        "--summary-base-url", ROLLOUT_BASE_URL,
        "--summary-model", SUMMARY_MODEL,
        "--summary-api-key", ROLLOUT_API_KEY,
        "--summary-timeout", "900",
        "--visit-summary-max-tokens", "16000",
        "--no-visit-images",
        "--visit-max-images", "0",
        "--max-tokens", "16000",
        "--code-sandbox-base-url", ROLLOUT_SANDBOX_BASE_URL,
    ]
    if ROLLOUT_TOKENIZER_PATH:
        cmd += ["--tokenizer-path", ROLLOUT_TOKENIZER_PATH,
                "--summary-tokenizer-path", ROLLOUT_TOKENIZER_PATH]
    if ROLLOUT_PROXY_URL:
        cmd += ["--proxy-url", ROLLOUT_PROXY_URL]
    if dry_run:
        print(f"[batch {batch_idx}] DRY RUN, would launch with {len(rows)} tasks → {bd}")
        return None, bd

    print(f"[batch {batch_idx}] launching rollout: {len(rows)} tasks → {bd}", flush=True)
    log_f = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f, stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT / "inference"),
    )
    return proc.pid, bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=900, help="scan period seconds")
    ap.add_argument("--min-batch", type=int, default=8,
                    help="don't launch rollout unless at least N new candidates")
    ap.add_argument("--max-batch", type=int, default=64,
                    help="max tasks per rollout batch (split larger)")
    ap.add_argument("--once", action="store_true",
                    help="run one scan + launch then exit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    batch_idx = 0
    # Resume: count existing batches
    for d in sorted(BATCH_ROOT.iterdir()) if BATCH_ROOT.exists() else []:
        if d.is_dir() and d.name.startswith("batch_"):
            try:
                n = int(d.name.split("_")[1])
                batch_idx = max(batch_idx, n + 1)
            except Exception:
                pass

    while True:
        seen = _load_seen()
        rows = _scan_new_candidates(seen)
        if rows:
            # Mark domains seen before launching (so a slow rollout doesn't
            # cause re-enqueue on next tick).
            new_domains = sorted({r["_domain"] for r in rows})
            _append_seen(new_domains)
            # Split into chunks of max-batch
            for i in range(0, len(rows), args.max_batch):
                chunk = rows[i : i + args.max_batch]
                if len(chunk) < args.min_batch and not args.once:
                    # Re-stash these — try again next cycle when more accumulate.
                    # Remove their domains from seen so they re-enter the pool.
                    rollback = sorted({r["_domain"] for r in chunk})
                    cur_seen = _load_seen()
                    cur_seen -= set(rollback)
                    SEEN_FILE.write_text("\n".join(sorted(cur_seen)) + "\n")
                    print(f"[batch {batch_idx}] only {len(chunk)} < min-batch={args.min_batch}, deferring", flush=True)
                    continue
                pid, bd = _launch_rollout(batch_idx, chunk, args.dry_run)
                if pid is not None:
                    print(f"[batch {batch_idx}] PID={pid}  out={bd}", flush=True)
                batch_idx += 1
        else:
            print(f"[scan @ {time.strftime('%H:%M:%S')}] no new candidates", flush=True)

        if args.once:
            break
        time.sleep(args.period)


if __name__ == "__main__":
    main()
