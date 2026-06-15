#!/usr/bin/env python3
"""Stage 2 — LLM rewrites skeleton steps into fully-filled hybrid search-mode samples.

Per-trajectory sequential (cumulative_progress carries forward), cross-trajectory
parallel via a ThreadPoolExecutor.

Usage:
  python stage2_rewrite.py \
      --inputs data/openseeker_skeleton_correct.jsonl,data/redsearcher_skeleton_correct.jsonl \
      --base-url http://47.82.159.100:8002/v1 \
      --model vertex_ai.claude-sonnet-4-6 \
      --api-key qazxswedcvfrtgbnhy \
      --out-path data/hybrid_search_samples.jsonl \
      --concurrency 16 \
      --max-trajectories 10        # for dry-run

Output JSONL — one line per rewritten step:
{
  "source":          "openseeker" | "redsearcher",
  "source_traj_id":  "...",
  "step_idx":        1,
  "total_steps":     23,
  "is_final":        false,
  "language":        "en",
  "task":            "...",
  "system_prompt":   "<the SEARCH_SYSTEM_PROMPT_TEMPLATE filled, for downstream training>",  // skipped to save space; downstream renders from agent_mode
  "agent_mode":      "search",
  "tool_call":       {"name":"search","arguments":{"query":["..."]}} or null,
  "answer":          null or "..."
  "rewrite":         {<full hybrid JSON schema fields>},
  "meta":            {<source meta passthrough>, "rewrite_attempts": N, "rewrite_ms": ...},
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib.stage2_rewriter import (  # noqa: E402
    RewriterConfig,
    StepRewriter,
    make_time_insensitive_inferred_date,
    normalize_date_evidence_carry_forward,
    validate_rewrite,
)


def load_skeleton_grouped(paths: list[str]) -> dict[str, list[dict]]:
    """Returns {source_traj_id: [step_dict, ...]} sorted by step_idx."""
    by_traj = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                by_traj[d["source_traj_id"]].append(d)
    for tid, steps in by_traj.items():
        steps.sort(key=lambda s: s["step_idx"])
    return by_traj


def load_already_done(out_path: Path) -> set[str]:
    """Trajectories whose FINAL step already appears and succeeded in out_path → done."""
    if not out_path.exists():
        return set()
    done = set()
    final_by_traj = {}
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            tid = d.get("source_traj_id")
            if tid is None:
                continue
            if d.get("is_final"):
                final_by_traj[tid] = d
    done.update(tid for tid, rec in final_by_traj.items() if rec.get("rewrite") is not None)
    return done


_TRANSIENT_ERROR_MARKERS = (
    "502", "503", "504",
    "Bad Gateway", "Service Unavailable", "Gateway Timeout",
    "InternalServerError",
    "APITimeoutError", "Request timed out", "timed out", "timeout",
    "APIConnectionError", "Connection error", "Connection reset",
    "RateLimitError", "rate limit", "429",
)


def _is_transient_error(info: dict) -> bool:
    if not info:
        return False
    err = (info.get("error") or "")
    return any(m in err for m in _TRANSIENT_ERROR_MARKERS)


def process_trajectory(traj_id: str, steps: list[dict], rewriter: StepRewriter, max_retries: int) -> list[dict]:
    """Sequentially rewrite all steps of one trajectory.

    Retry policy:
    - Transient errors (502 / 503 / 504 / timeout / connection / rate-limit):
      retry indefinitely with capped exponential backoff (5s, 10s, 20s, 40s, 60s cap),
      so a temporary endpoint outage does NOT discard the trajectory.
    - Hard errors (JSON parse / validation / 400 BadRequest):
      retry up to `max_retries` total attempts. The first attempt uses the base
      temperature; hard-error retries use `rewriter.cfg.retry_temperature`.
    """
    out_records: list[dict] = []
    prior_cp = ""
    prior_sg: list[str] = []
    prior_obs = ""
    recent_actions: list[dict] = []   # accumulated across steps for "Recent Actions" prompt
    date_evidence: list[str] = []     # YYYY-MM-DD lower bounds collected across steps
    time_insensitive_inferred_date = make_time_insensitive_inferred_date(traj_id)

    for s in steps:
        is_final = bool(s["is_final"])

        hard_attempts = 0      # counts only deterministic failures
        transient_attempts = 0 # counts transient retries (no upper bound, just for backoff)
        rewrite = None
        info = None
        verr = "rewrite never attempted"
        while True:
            attempt_temperature = (
                rewriter.cfg.temperature
                if hard_attempts == 0
                else rewriter.cfg.retry_temperature
            )
            rewrite, info = rewriter.rewrite_step(
                task=s["task"],
                step_idx=s["step_idx"],
                total_steps=s["total_steps"],
                is_final=is_final,
                prior_cumulative_progress=prior_cp,
                prior_remaining_subgoals=prior_sg,
                prior_observation=prior_obs,
                source_think=s["source_think"],
                mapped_tool_call=s.get("mapped_tool_call"),
                final_answer=s.get("mapped_answer"),
                recent_actions=recent_actions,
                prior_date_evidence=date_evidence,
                temperature=attempt_temperature,
                time_insensitive_inferred_date=time_insensitive_inferred_date,
            )
            is_initial = (s["step_idx"] == 1)
            if rewrite is not None:
                if not is_initial:
                    normalize_date_evidence_carry_forward(rewrite, date_evidence)
                verr = validate_rewrite(
                    rewrite,
                    is_final=is_final,
                    is_initial=is_initial,
                    time_insensitive_inferred_date=time_insensitive_inferred_date,
                )
                if verr is None:
                    break  # success
                # Validation failure → counts as hard error
                hard_attempts += 1
                if hard_attempts >= max_retries:
                    break
                continue

            # rewrite is None — distinguish transient vs hard
            if _is_transient_error(info):
                transient_attempts += 1
                # Capped exponential backoff: 5, 10, 20, 40, 60, 60, 60 ...
                backoff = min(60, 5 * (2 ** (transient_attempts - 1)))
                import time as _t
                _t.sleep(backoff)
                continue
            else:
                verr = (info or {}).get("error", "unknown rewrite failure")
                hard_attempts += 1
                if hard_attempts >= max_retries:
                    break
                continue

        rec = {
            "source": s["source"],
            "source_traj_id": traj_id,
            "step_idx": s["step_idx"],
            "total_steps": s["total_steps"],
            "is_final": is_final,
            "language": s["language"],
            "task": s["task"],
            "agent_mode": "search",
            "tool_call": s.get("mapped_tool_call"),
            "answer": s.get("mapped_answer"),
            "rewrite": rewrite if verr is None else None,
            "rewrite_error": verr,
            "meta": {
                **(s.get("meta") or {}),
                "rewrite_hard_attempts": hard_attempts,
                "rewrite_transient_retries": transient_attempts,
                "rewrite_ms": (info or {}).get("latency_ms"),
                "prompt_tokens": (info or {}).get("prompt_tokens"),
                "completion_tokens": (info or {}).get("completion_tokens"),
                "temperature": (info or {}).get("temperature"),
                "time_insensitive_inferred_date": time_insensitive_inferred_date,
            },
        }
        out_records.append(rec)

        if verr is not None:
            # Stop processing this trajectory on unrecoverable failure to avoid wasting
            # tokens on dependent downstream steps with bad cumulative_progress.
            return out_records

        if s["step_idx"] == 1:
            prior_cp = "No progress yet."
        else:
            prior_cp = rewrite["cumulative_progress"]
        prior_sg = rewrite["remaining_subgoals"]
        # next step's prior_observation = THIS step's source_observation_after
        prior_obs = s.get("source_observation_after", "") or ""
        recent_actions.append({
            "step_idx": s["step_idx"],
            "tool_call": s.get("mapped_tool_call"),
            "action_description": rewrite.get("action_description", "") or "",
        })
        de = rewrite.get("date_evidence")
        if isinstance(de, str) and de.strip() and (not date_evidence or date_evidence[-1] != de.strip()):
            date_evidence.append(de.strip())

    return out_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        default="data/openseeker_skeleton_correct.jsonl,data/redsearcher_skeleton_correct.jsonl",
        help="Comma-separated skeleton .jsonl paths.",
    )
    ap.add_argument(
        "--out-path",
        default="data/hybrid_search_samples.jsonl",
    )
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", ""))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retry-temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Cap on number of new trajectories to process (excludes already-done resumed).",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite out-path instead of skipping already-done trajectories.",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle trajectories before processing (useful for smoke runs).",
    )
    ap.add_argument(
        "--traj-ids-file",
        default=None,
        help="Optional path to a .txt file containing one source_traj_id per line. "
             "Only trajectories whose IDs appear in this file will be processed.",
    )
    args = ap.parse_args()

    if not (args.base_url and args.model):
        ap.error("--base-url and --model are required")

    paths = [p.strip() for p in args.inputs.split(",") if p.strip()]
    print(f"[load] reading {len(paths)} skeleton files")
    by_traj = load_skeleton_grouped(paths)
    print(f"[load] {len(by_traj)} trajectories total")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set()
    if not args.no_resume:
        already_done = load_already_done(out_path)
        if already_done:
            print(f"[resume] {len(already_done)} trajectories already finished in {out_path}")

    # Optional whitelist of trajectory IDs
    whitelist = None
    if args.traj_ids_file:
        with open(args.traj_ids_file, encoding="utf-8") as f:
            whitelist = {line.strip() for line in f if line.strip()}
        print(f"[whitelist] restricting to {len(whitelist)} trajectory ids from {args.traj_ids_file}")

    pending_ids = []
    for tid in by_traj:
        if tid in already_done:
            continue
        if whitelist is not None and tid not in whitelist:
            continue
        pending_ids.append(tid)
    if args.shuffle:
        import random
        random.seed(0)
        random.shuffle(pending_ids)
    if args.max_trajectories is not None:
        pending_ids = pending_ids[: args.max_trajectories]
    print(f"[plan] {len(pending_ids)} trajectories to rewrite")
    if not pending_ids:
        return

    rewriter = StepRewriter(
        RewriterConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            retry_temperature=args.retry_temperature,
        )
    )

    file_lock = threading.Lock()
    stats = {
        "trajs_done": 0,
        "trajs_partial": 0,
        "steps_written": 0,
        "steps_failed": 0,
        "tokens_in": 0,
        "tokens_out": 0,
    }
    stats_lock = threading.Lock()
    mode = "w" if args.no_resume else "a"
    fout = out_path.open(mode, encoding="utf-8")

    t0 = time.time()

    def worker(tid: str) -> tuple[str, list[dict]]:
        recs = process_trajectory(tid, by_traj[tid], rewriter, max_retries=args.max_retries)
        with file_lock:
            for r in recs:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
        return tid, recs

    print(f"[run] {args.concurrency} workers ...")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, tid) for tid in pending_ids]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                tid, recs = fut.result()
            except Exception as exc:
                with stats_lock:
                    stats["trajs_partial"] += 1
                print(f"  [{i}/{len(pending_ids)}] worker exception: {type(exc).__name__}: {exc}")
                continue
            with stats_lock:
                last = recs[-1] if recs else None
                fully_done = bool(last and last.get("rewrite") is not None and last.get("is_final"))
                if fully_done:
                    stats["trajs_done"] += 1
                else:
                    stats["trajs_partial"] += 1
                    fail_step = next((r for r in recs if r.get("rewrite") is None), None)
                    fail_err = (fail_step.get("meta",{}).get("rewrite_error","") or fail_step.get("rewrite_error","")) if fail_step else "no_recs"
                    print(f"  [PARTIAL] {tid}: step={fail_step.get('step_idx','?')}/{fail_step.get('total_steps','?')} err={str(fail_err)[:200]}" if fail_step else f"  [PARTIAL] {tid}: no_recs")
                for r in recs:
                    if r.get("rewrite") is None:
                        stats["steps_failed"] += 1
                    else:
                        stats["steps_written"] += 1
                    m = r.get("meta") or {}
                    if m.get("prompt_tokens"):
                        stats["tokens_in"] += m["prompt_tokens"]
                    if m.get("completion_tokens"):
                        stats["tokens_out"] += m["completion_tokens"]
            if i % 5 == 0 or i == len(pending_ids):
                elapsed = time.time() - t0
                rate = i / max(1e-6, elapsed)
                eta_s = (len(pending_ids) - i) / max(1e-6, rate)
                print(
                    f"  [{i}/{len(pending_ids)}] done={stats['trajs_done']} "
                    f"partial={stats['trajs_partial']} "
                    f"steps={stats['steps_written']}+{stats['steps_failed']}f "
                    f"tok_in={stats['tokens_in']} tok_out={stats['tokens_out']} "
                    f"rate={rate:.2f}/s eta={eta_s/60:.1f}m"
                )

    fout.close()
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    print()
    print("=" * 60)
    print("Final stats")
    print("=" * 60)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
