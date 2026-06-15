#!/usr/bin/env python3
"""Multimodal Stage 2 rewriter for MolmoWeb hybrid skeleton.

- Reads skeleton JSONL grouped by trajectory.
- Filters by --traj-ids-file (one traj_id per line) for load-balancing across models.
- Streams one parquet shard at a time, loading screenshots only for whitelisted trajs.
- Per traj: sequential steps so cumulative_progress carries forward.
- Cross traj (within a shard): parallel via ThreadPoolExecutor.
- Transient errors (502/5xx/timeout/...) retry forever with capped exponential backoff.
- Hard errors (JSON parse / validation) retry max_retries then write a failure record.
- Resume: skips trajs whose final step already exists in --out-path.
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
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.molmoweb_stage2_rewriter import (
    MMRewriterConfig,
    MMStepRewriter,
    make_time_insensitive_inferred_date,
    normalize_date_evidence_carry_forward,
    validate_rewrite,
)


_TRANSIENT_MARKERS = (
    "502", "503", "504",
    "Bad Gateway", "Service Unavailable", "Gateway Timeout",
    "InternalServerError",
    "APITimeoutError", "Request timed out", "timed out", "timeout",
    "APIConnectionError", "Connection error", "Connection reset",
    "RateLimitError", "rate limit", "429",
)


def is_transient(err: str) -> bool:
    return any(m in (err or "") for m in _TRANSIENT_MARKERS)


def load_skeleton_by_traj(skel_path: Path, whitelist: set[str] | None) -> dict[str, list[dict]]:
    by_traj: dict[str, list[dict]] = defaultdict(list)
    with skel_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            tid = d["source_traj_id"]
            if whitelist is not None and tid not in whitelist:
                continue
            by_traj[tid].append(d)
    for tid in by_traj:
        by_traj[tid].sort(key=lambda r: r["step_idx"])
    drop = []
    for tid, recs in by_traj.items():
        for r in recs:
            sref = r.get("screenshot_ref") or {}
            if not sref.get("sample_id") or not isinstance(sref.get("source_step_idx"), int):
                drop.append(tid)
                break
    for tid in drop:
        del by_traj[tid]
    if drop:
        print(f"[filter] dropped {len(drop)} trajectories with missing screenshot_ref")
    return by_traj


def load_already_done(out_path: Path) -> set[str]:
    """A traj is 'done' iff its final step record (is_final=True) is present AND rewrite != None."""
    if not out_path.exists():
        return set()
    final_by_traj: dict[str, dict] = {}
    with out_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("is_final"):
                # Keep last one (in case of duplicates from retries)
                final_by_traj[d["source_traj_id"]] = d
    return {tid for tid, rec in final_by_traj.items() if rec.get("rewrite") is not None}


def stream_shard_for_trajs(shard_path: str, sample_ids: set[str], batch_size: int = 50):
    """Yield mini-batches of {sample_id: (trajectory, images)} from a parquet shard.

    Uses duckdb fetchmany streaming to avoid materializing entire shard payload
    (which OOMs when ~700 trajs × ~16 images × ~500KB ≈ 5GB)."""
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='12GB';")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET threads=2;")
    if not sample_ids:
        return
    placeholders = ",".join(f"'{s}'" for s in sample_ids)
    cursor = con.execute(
        f"SELECT sample_id, trajectory, images FROM read_parquet('{shard_path}') "
        f"WHERE sample_id IN ({placeholders})"
    )
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            return
        batch: dict[str, tuple[dict, list[bytes]]] = {}
        for sid, tj, imgs in rows:
            try:
                traj = json.loads(tj)
            except Exception:
                continue
            imgs_list: list[bytes] = []
            for it in (imgs or []):
                if isinstance(it, (bytes, bytearray)):
                    imgs_list.append(bytes(it))
                elif isinstance(it, dict):
                    imgs_list.append(it.get("bytes", b""))
            batch[sid] = (traj, imgs_list)
        yield batch
        # Drop reference so GC can free image bytes from the previous batch
        del batch


def process_traj_disk(skel_recs: list[dict], sample_dir: Path,
                      rewriter: MMStepRewriter, max_hard_retries: int,
                      log_prefix: str) -> list[dict]:
    """Like process_traj but reads pre-extracted JPEGs from disk per-step."""
    out_records: list[dict] = []
    prior_cp = ""
    prior_sg: list[str] = []
    img_cache: dict[int, bytes] = {}
    recent_actions: list[dict] = []
    date_evidence: list[str] = []
    traj_id = skel_recs[0].get("source_traj_id", log_prefix) if skel_recs else log_prefix
    time_insensitive_inferred_date = make_time_insensitive_inferred_date(traj_id)
    for r in skel_recs:
        src_idx = (r.get("screenshot_ref") or {}).get("source_step_idx", 1)
        if src_idx not in img_cache:
            p = sample_dir / f"{src_idx}.jpg"
            if not p.exists():
                rec = dict(r)
                rec["rewrite"] = None
                rec["rewrite_error"] = f"image_file_missing: {p}"
                rec["rewrite_meta"] = {}
                out_records.append(rec)
                break
            img_cache[src_idx] = p.read_bytes()
        shot = img_cache[src_idx]

        is_initial = (r["step_idx"] == 1 and not r["is_final"])
        hard_attempts = 0
        transient_attempts = 0
        parsed = None
        info = None
        verr = "never attempted"
        while True:
            attempt_temperature = (
                rewriter.cfg.temperature
                if hard_attempts == 0
                else rewriter.cfg.retry_temperature
            )
            parsed, info = rewriter.rewrite_step(
                task=r["task"],
                initial_url=r["initial_url"],
                step_idx=r["step_idx"],
                total_steps=r["total_steps"],
                is_final=r["is_final"],
                prior_cumulative_progress=prior_cp,
                prior_remaining_subgoals=prior_sg,
                source_thought=r.get("source_thought", ""),
                tool_call=r.get("tool_call"),
                final_answer=r.get("answer"),
                screenshot_bytes=shot,
                current_url=r.get("current_url", ""),
                recent_actions=recent_actions,
                prior_date_evidence=date_evidence,
                temperature=attempt_temperature,
            )
            if parsed is not None:
                normalize_date_evidence_carry_forward(parsed, date_evidence)
                verr = validate_rewrite(
                    parsed,
                    is_final=r["is_final"],
                    is_initial=is_initial,
                    time_insensitive_inferred_date=time_insensitive_inferred_date,
                )
                if verr is None:
                    break
                hard_attempts += 1
                if hard_attempts >= max_hard_retries:
                    break
                continue
            err = (info or {}).get("error", "")
            if is_transient(err):
                transient_attempts += 1
                backoff = min(60, 5 * (2 ** (transient_attempts - 1)))
                time.sleep(backoff)
                continue
            verr = err or "rewrite returned None"
            hard_attempts += 1
            if hard_attempts >= max_hard_retries:
                break

        rec = dict(r)
        rec["rewrite"] = parsed if verr is None else None
        rec["rewrite_error"] = verr
        rec["rewrite_meta"] = {
            "hard_attempts": hard_attempts,
            "transient_retries": transient_attempts,
            "latency_ms": (info or {}).get("latency_ms"),
            "prompt_tokens": (info or {}).get("prompt_tokens"),
            "completion_tokens": (info or {}).get("completion_tokens"),
            "temperature": (info or {}).get("temperature"),
            "time_insensitive_inferred_date": time_insensitive_inferred_date,
        }
        out_records.append(rec)
        if verr is not None:
            break
        if is_initial:
            prior_cp = "No progress yet."
        else:
            prior_cp = parsed.get("cumulative_progress", "")
        prior_sg = parsed.get("remaining_subgoals", []) or []
        recent_actions.append({
            "step_idx": r["step_idx"],
            "tool_call": r.get("tool_call"),
            "action_description": parsed.get("action_description", "") or "",
        })
        de = parsed.get("date_evidence")
        if isinstance(de, str) and de.strip() and (not date_evidence or date_evidence[-1] != de.strip()):
            date_evidence.append(de.strip())
    return out_records


def _resize_to_target(image_bytes: bytes, target_w: int, target_h: int, jpeg_q: int) -> bytes:
    """Open, resize to (target_w, target_h), re-encode as JPEG. Returns raw bytes."""
    from PIL import Image
    import io as _io
    img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
    if (img.width, img.height) != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_q, optimize=True)
    return buf.getvalue()


def process_traj(skel_recs: list[dict], images: list[bytes], rewriter: MMStepRewriter,
                 max_hard_retries: int, log_prefix: str,
                 image_out_dir: Path | None,
                 target_w: int, target_h: int, jpeg_q: int) -> list[dict]:
    """Sequentially rewrite all steps of one traj. Returns list of output records.

    If image_out_dir is set, the 1280×720 JPEG used as model input is also persisted
    to disk at <image_out_dir>/<sample_id>/<source_step_idx>.jpg, and the path is
    recorded on each output record as `image_path`."""
    out_records: list[dict] = []
    prior_cp = ""
    prior_sg: list[str] = []
    saved_steps: dict[int, Path] = {}     # src_step_idx → saved path (avoid re-saving)
    resized_cache: dict[int, bytes] = {}  # src_step_idx → resized JPEG bytes
    sample_id = (skel_recs[0].get("screenshot_ref") or {}).get("sample_id", "") if skel_recs else ""
    traj_id = skel_recs[0].get("source_traj_id", log_prefix) if skel_recs else log_prefix
    time_insensitive_inferred_date = make_time_insensitive_inferred_date(traj_id)
    recent_actions: list[dict] = []
    date_evidence: list[str] = []
    for r in skel_recs:
        src_idx = (r.get("screenshot_ref") or {}).get("source_step_idx", 1)
        if not (1 <= src_idx <= len(images)):
            verr = f"missing_screenshot src_idx={src_idx} (have {len(images)})"
            rec = dict(r)
            rec["rewrite"] = None
            rec["rewrite_error"] = verr
            rec["rewrite_meta"] = {}
            out_records.append(rec)
            break
        # Resize + save once per source step
        if src_idx not in resized_cache:
            try:
                resized = _resize_to_target(images[src_idx - 1], target_w, target_h, jpeg_q)
                resized_cache[src_idx] = resized
                if image_out_dir is not None and sample_id:
                    save_dir = image_out_dir / sample_id
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{src_idx}.jpg"
                    if not save_path.exists():
                        save_path.write_bytes(resized)
                    saved_steps[src_idx] = save_path
            except Exception as exc:
                verr = f"image_resize_save: {type(exc).__name__}: {exc}"
                rec = dict(r)
                rec["rewrite"] = None
                rec["rewrite_error"] = verr
                rec["rewrite_meta"] = {}
                out_records.append(rec)
                break
        shot = resized_cache[src_idx]

        is_initial = (r["step_idx"] == 1 and not r["is_final"])
        hard_attempts = 0
        transient_attempts = 0
        parsed = None
        info = None
        verr = "never attempted"
        while True:
            attempt_temperature = (
                rewriter.cfg.temperature
                if hard_attempts == 0
                else rewriter.cfg.retry_temperature
            )
            parsed, info = rewriter.rewrite_step(
                task=r["task"],
                initial_url=r["initial_url"],
                step_idx=r["step_idx"],
                total_steps=r["total_steps"],
                is_final=r["is_final"],
                prior_cumulative_progress=prior_cp,
                prior_remaining_subgoals=prior_sg,
                source_thought=r.get("source_thought", ""),
                tool_call=r.get("tool_call"),
                final_answer=r.get("answer"),
                screenshot_bytes=shot,
                current_url=r.get("current_url", ""),
                recent_actions=recent_actions,
                prior_date_evidence=date_evidence,
                temperature=attempt_temperature,
            )
            if parsed is not None:
                normalize_date_evidence_carry_forward(parsed, date_evidence)
                verr = validate_rewrite(
                    parsed,
                    is_final=r["is_final"],
                    is_initial=is_initial,
                    time_insensitive_inferred_date=time_insensitive_inferred_date,
                )
                if verr is None:
                    break
                hard_attempts += 1
                if hard_attempts >= max_hard_retries:
                    break
                continue
            err = (info or {}).get("error", "")
            if is_transient(err):
                transient_attempts += 1
                backoff = min(60, 5 * (2 ** (transient_attempts - 1)))
                time.sleep(backoff)
                continue
            verr = err or "rewrite returned None"
            hard_attempts += 1
            if hard_attempts >= max_hard_retries:
                break

        rec = dict(r)
        rec["rewrite"] = parsed if verr is None else None
        rec["rewrite_error"] = verr
        rec["rewrite_meta"] = {
            "hard_attempts": hard_attempts,
            "transient_retries": transient_attempts,
            "latency_ms": (info or {}).get("latency_ms"),
            "prompt_tokens": (info or {}).get("prompt_tokens"),
            "completion_tokens": (info or {}).get("completion_tokens"),
            "temperature": (info or {}).get("temperature"),
            "time_insensitive_inferred_date": time_insensitive_inferred_date,
        }
        out_records.append(rec)
        if verr is not None:
            break  # truncate traj at first hard failure
        if is_initial:
            prior_cp = "No progress yet."
        else:
            prior_cp = parsed.get("cumulative_progress", "")
        prior_sg = parsed.get("remaining_subgoals", []) or []
        # Append this step to recent_actions history (used as input on subsequent steps)
        recent_actions.append({
            "step_idx": r["step_idx"],
            "tool_call": r.get("tool_call"),
            "action_description": parsed.get("action_description", "") or "",
        })
        de = parsed.get("date_evidence")
        if isinstance(de, str) and de.strip() and (not date_evidence or date_evidence[-1] != de.strip()):
            date_evidence.append(de.strip())
    return out_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton", default="data/hybrid_op_molmoweb_skeleton.jsonl")
    ap.add_argument("--data-glob", default="/data/czj/WebDatasets/allenai/MolmoWeb-HumanTrajs/data/train-*.parquet")
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--traj-ids-file", default="",
                    help="Path to a file with one traj_id per line; only these are processed.")
    ap.add_argument("--base-url", default="http://47.82.159.100:8002/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="qazxswedcvfrtgbnhy")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--max-hard-retries", "--max-retries", dest="max_hard_retries", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retry-temperature", type=float, default=0.7)
    ap.add_argument("--max-trajectories", type=int, default=0,
                    help="If >0, cap total trajs processed (for smoke runs).")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore existing output (start fresh).")
    ap.add_argument("--image-dir", default="data/molmoweb_images",
                    help="Read pre-extracted 1280×720 JPEGs from <dir>/<sample_id>/<src_step>.jpg")
    ap.add_argument("--shard-batch-size", type=int, default=50)
    args = ap.parse_args()

    skel_path = Path(args.skeleton)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Whitelist of traj ids to process (per-model bucket)
    whitelist: set[str] | None = None
    if args.traj_ids_file:
        with open(args.traj_ids_file) as f:
            whitelist = {ln.strip() for ln in f if ln.strip()}
        print(f"[whitelist] {len(whitelist)} traj ids from {args.traj_ids_file}")

    print(f"[load] reading skeleton {skel_path}")
    by_traj = load_skeleton_by_traj(skel_path, whitelist)
    print(f"[load] {len(by_traj)} trajectories in scope")

    already_done: set[str] = set()
    if not args.no_resume:
        already_done = load_already_done(out_path)
        print(f"[resume] {len(already_done)} trajectories already fully done in {out_path}")
    todo = [tid for tid in by_traj if tid not in already_done]
    if args.max_trajectories > 0:
        todo = todo[:args.max_trajectories]
    print(f"[plan] {len(todo)} trajectories to rewrite")

    # Disk-image mode (default): no parquet access at all.
    image_dir = Path(args.image_dir) if args.image_dir else None
    if image_dir is None:
        print("[error] --image-dir is required (pre-extract images first via molmoweb_extract_images.py)")
        sys.exit(1)
    if not image_dir.exists():
        print(f"[error] {image_dir} does not exist; run molmoweb_extract_images.py first")
        sys.exit(1)
    print(f"[image-dir] reading screenshots from {image_dir}")

    rewriter = MMStepRewriter(MMRewriterConfig(
        base_url=args.base_url, model=args.model, api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retry_temperature=args.retry_temperature,
    ))

    file_lock = threading.Lock()
    mode = "w" if args.no_resume else "a"
    fout = out_path.open(mode, encoding="utf-8")
    n_done_traj = 0
    n_done_steps = 0
    n_failed_traj = 0
    n_failed_steps = 0
    t0 = time.time()

    print(f"[run] {len(todo)} trajs · {args.concurrency} workers · model={args.model}")

    # Single long-lived worker pool over the whole todo list. Each worker reads its
    # own traj's screenshots from disk (no parquet access, no shard sharing required).
    pool = ThreadPoolExecutor(max_workers=args.concurrency)

    def _worker(tid: str):
        recs = by_traj[tid]
        sid = tid.split(":", 1)[-1]
        sample_dir = image_dir / sid
        if not sample_dir.exists():
            return tid, [], f"image_dir_missing: {sample_dir}"
        try:
            out_recs = process_traj_disk(recs, sample_dir, rewriter, args.max_hard_retries, tid)
        except Exception as exc:
            return tid, [], f"worker_exc: {type(exc).__name__}: {exc}"
        return tid, out_recs, None

    try:
        futures = {pool.submit(_worker, tid): tid for tid in todo}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                _, out_recs, err = fut.result()
            except Exception as exc:
                out_recs, err = [], f"future_exc: {exc}"
            with file_lock:
                for r in out_recs:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_done_steps += 1
                    if r.get("rewrite") is None:
                        n_failed_steps += 1
                fout.flush()
            if err:
                n_failed_traj += 1
                print(f"  [FAIL] {tid}: {err[:200]}")
            else:
                last = out_recs[-1] if out_recs else None
                if last and last.get("rewrite") is not None and last.get("is_final"):
                    n_done_traj += 1
                else:
                    n_failed_traj += 1
            if (n_done_traj + n_failed_traj) % 10 == 0:
                elapsed = time.time() - t0
                rate = (n_done_traj + n_failed_traj) / max(1e-6, elapsed)
                eta_min = (len(todo) - n_done_traj - n_failed_traj) / max(1e-6, rate) / 60
                print(f"  done={n_done_traj} fail={n_failed_traj} steps={n_done_steps}+{n_failed_steps}f "
                      f"rate={rate:.2f}/s eta={eta_min:.1f}min")
    finally:
        pool.shutdown(wait=True)

    fout.close()
    elapsed = time.time() - t0
    print()
    print(f"=== finished in {elapsed/60:.1f}m ===")
    print(f"  trajs done:    {n_done_traj}")
    print(f"  trajs failed:  {n_failed_traj}")
    print(f"  steps written: {n_done_steps}")
    print(f"  steps failed:  {n_failed_steps}")


if __name__ == "__main__":
    main()
