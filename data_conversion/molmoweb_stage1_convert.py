#!/usr/bin/env python3
"""MolmoWeb-HumanTrajs → hybrid (operation-mode) Stage 1 skeleton conversion.

For each KEPT trajectory (per Stage A v2 filter), produces one JSONL record per
hybrid step containing:
  - hybrid tool_call (or final `answer`)
  - reference to the source screenshot in the original parquet shard
  - source_thought (the agent's free-form reasoning, for Stage 2 LLM context)
  - traj-level metadata: task (high_level instruction), initial_url, total_steps

The actual screenshot bytes are NOT inlined; Stage 2 (LLM rewrite) will load
them on demand by (shard, sample_id, source_step_idx) and resize to 1280x720.

Usage:
  python3 molmoweb_stage1_convert.py \
    --per-traj molmoweb_quality/stage_a_v2/per_traj.jsonl \
    --data-glob "/data/czj/WebDatasets/allenai/MolmoWeb-HumanTrajs/data/train-*.parquet" \
    --out-path data/hybrid_op_molmoweb_skeleton.jsonl \
    --limit 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.molmoweb_to_hybrid import convert_trajectory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-traj", default="molmoweb_quality/stage_a_v2/per_traj.jsonl")
    p.add_argument("--data-glob",
                   default="/data/czj/WebDatasets/allenai/MolmoWeb-HumanTrajs/data/train-*.parquet")
    p.add_argument("--out-path", default="data/hybrid_op_molmoweb_skeleton.jsonl")
    p.add_argument("--target-width", type=int, default=1280)
    p.add_argument("--target-height", type=int, default=720)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only process this many kept trajectories (smoke run).")
    return p.parse_args()


def load_keep_ids(per_traj_path: Path) -> set[str]:
    keep = set()
    with per_traj_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("keep"):
                keep.add(d["sample_id"])
    return keep


def build_step_records(sid: str, instruction_json: str, trajectory_json: str,
                       shard_basename: str,
                       target_w: int, target_h: int) -> list[dict]:
    """Convert one trajectory into a list of per-step hybrid records."""
    try:
        instr = json.loads(instruction_json) if instruction_json else {}
    except Exception:
        instr = {}
    try:
        traj = json.loads(trajectory_json)
    except Exception:
        return []

    conv = convert_trajectory(traj, target_w=target_w, target_h=target_h)

    # Per-step image dims and thought come from source steps
    src_keys = sorted(traj.keys(), key=int)
    src_by_idx: dict[int, dict] = {int(k): traj[k] for k in src_keys}

    hybrid_steps = conv["hybrid_steps"]
    n_hybrid = len(hybrid_steps)
    if n_hybrid == 0:
        return []

    task = (instr.get("high_level") or "").strip()
    initial_url = conv["initial_url"]
    out_records: list[dict] = []

    # For scroll replicas with empty src_indices, use the previous emitted screenshot
    last_screenshot_ref: dict | None = None
    last_source_thought: str = ""
    last_current_url: str = initial_url

    for hi, h in enumerate(hybrid_steps):
        kind = h.get("kind", "tool")
        src_indices = h.get("src_indices") or []
        # Choose representative source step for screenshot/thought (the FIRST consumed
        # step is the agent's observation BEFORE the merged action — the right input).
        rep_src_idx = src_indices[0] if src_indices else None
        if rep_src_idx is not None and rep_src_idx in src_by_idx:
            src_step = src_by_idx[rep_src_idx]
            img_w = int(src_step.get("image_w") or target_w)
            img_h = int(src_step.get("image_h") or target_h)
            thought = ((src_step.get("action") or {}).get("action_output") or {}).get("thought", "") or ""
            current_url = (src_step.get("other_obs") or {}).get("url", "") or ""
            screenshot_ref = {
                "shard": shard_basename,
                "sample_id": sid,
                "source_step_idx": rep_src_idx,         # 1-based
                "image_w": img_w,
                "image_h": img_h,
            }
            last_screenshot_ref = screenshot_ref
            last_source_thought = thought
            last_current_url = current_url
        else:
            # Replica scroll with no own src — reuse previous
            screenshot_ref = last_screenshot_ref
            thought = last_source_thought
            current_url = last_current_url

        is_final = (hi == n_hybrid - 1) and (kind == "answer")
        rec = {
            "source": "molmoweb",
            "source_traj_id": f"molmoweb:{sid}",
            "step_idx": hi + 1,                # 1-based hybrid step index
            "total_steps": n_hybrid,
            "is_final": is_final,
            "language": "en",
            "task": task,
            "initial_url": initial_url,
            "current_url": current_url,
            "agent_mode": "operation",
            "tool_call": h.get("tool_call"),
            "answer": h.get("answer") if kind == "answer" else None,
            "screenshot_ref": screenshot_ref,
            "source_thought": thought.strip(),
            "src_merge_indices": src_indices,
            "merge_note": h.get("merge_note", "") or "",
        }
        out_records.append(rec)

    return out_records


def main():
    args = parse_args()
    per_traj_path = Path(args.per_traj)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] keep IDs from {per_traj_path}")
    keep_ids = load_keep_ids(per_traj_path)
    print(f"[load] {len(keep_ids)} kept trajectories to convert")
    if args.limit > 0:
        keep_ids = set(list(keep_ids)[:args.limit])
        print(f"[load] limited to first {len(keep_ids)} trajectories (smoke run)")

    files = sorted(glob(args.data_glob))
    if not files:
        print(f"[error] no parquet files match {args.data_glob}", file=sys.stderr)
        sys.exit(1)
    print(f"[load] {len(files)} parquet shards to scan")

    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB';")
    con.execute("SET preserve_insertion_order=false;")

    n_traj = 0
    n_steps = 0
    n_answer = 0
    n_tool_calls_by_name: dict[str, int] = defaultdict(int)
    pending = set(keep_ids)
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fout:
        for fi, f in enumerate(files, start=1):
            if not pending:
                break
            shard_basename = f.rsplit("/", 1)[-1]
            print(f"[file {fi}/{len(files)}] {shard_basename}  (pending={len(pending)})")
            # Single scan of shard
            cursor = con.execute(f"SELECT sample_id, instruction, trajectory FROM read_parquet('{f}')")
            while True:
                batch = cursor.fetchmany(64)
                if not batch:
                    break
                for sid, instr, traj in batch:
                    if sid not in pending:
                        continue
                    pending.discard(sid)
                    records = build_step_records(
                        sid, instr, traj, shard_basename,
                        target_w=args.target_width, target_h=args.target_height,
                    )
                    if not records:
                        continue
                    n_traj += 1
                    for r in records:
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                        n_steps += 1
                        if r["is_final"]:
                            n_answer += 1
                        elif r.get("tool_call"):
                            n_tool_calls_by_name[r["tool_call"]["name"]] += 1
                    if n_traj % 500 == 0:
                        elapsed = time.time() - t0
                        rate = n_traj / max(1e-6, elapsed)
                        print(f"  [{n_traj}] wrote {n_steps} steps, rate={rate:.1f} traj/s")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  trajectories converted: {n_traj}")
    print(f"  hybrid step records:    {n_steps}")
    print(f"  final answer steps:     {n_answer}")
    print(f"  avg steps / traj:       {n_steps/max(1,n_traj):.2f}")
    print(f"  tool_call distribution:")
    for name, cnt in sorted(n_tool_calls_by_name.items(), key=lambda x: -x[1]):
        print(f"    {name:12} {cnt}")
    if pending:
        print(f"  [warn] {len(pending)} kept IDs not found in any shard")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  output:  {out_path}")


if __name__ == "__main__":
    main()
