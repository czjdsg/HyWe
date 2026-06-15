#!/usr/bin/env python3
"""WebChain → hybrid (operation-mode) Stage 1 skeleton conversion.

For each KEPT trajectory (after dropping copy/paste/right_click and empty trajs),
emit one JSONL record per hybrid step:
  - hybrid tool_call
  - reference to source img URL (for Stage 2 / download)
  - source title/href/value as metadata
  - traj-level: task (= traj title), initial_url, total hybrid steps
"""

from __future__ import annotations
import argparse, glob, json, sys, time
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.webchain_to_hybrid import convert_trajectory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", default="/tmp/webchain_all_json")
    p.add_argument("--out-path", default="data/hybrid_op_webchain_skeleton.jsonl")
    p.add_argument("--target-width", type=int, default=1280)
    p.add_argument("--target-height", type=int, default=720)
    p.add_argument("--limit", type=int, default=0,
                   help=">0: smoke run, cap total trajs.")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f'{args.json_dir}/**/*.json', recursive=True))
    print(f'[load] {len(files)} json files')

    n_trajs = 0; n_kept = 0; n_dropped = 0; n_steps = 0; n_answer = 0
    drop_reasons = Counter(); tool_counts = Counter()
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fout:
        for fi, f in enumerate(files, start=1):
            try: data = json.load(open(f))
            except: continue
            if isinstance(data, dict): data = [data]
            if not isinstance(data, list): continue
            for traj in data:
                if not isinstance(traj, dict): continue
                n_trajs += 1
                if args.limit and n_trajs > args.limit:
                    break

                conv = convert_trajectory(traj, target_w=args.target_width, target_h=args.target_height)
                if conv["skip_reason"]:
                    n_dropped += 1
                    drop_reasons[conv["skip_reason"].split(":")[0]] += 1
                    continue
                if conv["n_hybrid_steps"] == 0:
                    n_dropped += 1
                    drop_reasons["zero_hybrid_steps"] += 1
                    continue
                n_kept += 1

                source_traj_id = f"webchain:{traj.get('id', '?')}"
                task = (traj.get('title') or '').lstrip('- ').strip('"').strip()
                src_steps = traj.get('steps') or []
                initial_url = conv["initial_url"]

                # WebChain doesn't have an explicit "answer" step — the trajectory IS the
                # operation. Final step is the last hybrid step (a tool action).
                # We emit no synthetic answer; downstream training treats last action as final.

                for hi, h in enumerate(conv["hybrid_steps"]):
                    # Source step idx for screenshot ref = first idx of src_indices
                    src_idx = h["src_indices"][0] if h["src_indices"] else None
                    rep_src = src_steps[src_idx - 1] if src_idx else {}
                    img_ref = rep_src.get("img", "") or ""
                    source_thought = rep_src.get("title", "") or ""  # WebChain "title" is Chinese description
                    is_final = (hi == conv["n_hybrid_steps"] - 1)
                    rec = {
                        "source": "webchain",
                        "source_traj_id": source_traj_id,
                        "step_idx": hi + 1,
                        "total_steps": conv["n_hybrid_steps"],
                        "is_final": is_final,
                        "language": "en",   # mostly English; titles vary
                        "task": task,
                        "initial_url": initial_url,
                        "current_url": rep_src.get("href", "") or "",
                        "agent_mode": "operation",
                        "tool_call": h.get("tool_call"),
                        "answer": None,
                        "screenshot_ref": {
                            "kind": "url" if img_ref.startswith("http") else "local",
                            "url": img_ref if img_ref.startswith("http") else "",
                            "local_id": img_ref if not img_ref.startswith("http") else "",
                            "source_step_idx": src_idx,
                            "viewport_w": (rep_src.get("viewport") or {}).get("width", args.target_width),
                            "viewport_h": (rep_src.get("viewport") or {}).get("height", args.target_height),
                        },
                        "source_thought": source_thought.strip(),
                        "src_merge_indices": h["src_indices"],
                        "merge_note": h.get("merge_note", "") or "",
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_steps += 1
                    tc = rec.get("tool_call") or {}
                    tool_counts[tc.get("name", "?")] += 1

                if n_kept % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f'  [kept={n_kept} dropped={n_dropped}] {n_steps} steps  rate={n_kept/max(1e-6,elapsed):.1f} traj/s')
            if args.limit and n_trajs > args.limit:
                break

    elapsed = time.time() - t0
    print()
    print(f"=== finished in {elapsed:.1f}s ===")
    print(f"  scanned trajs: {n_trajs:,}")
    print(f"  kept trajs:    {n_kept:,}")
    print(f"  dropped:       {n_dropped:,}")
    print(f"  drop reasons:  {dict(drop_reasons.most_common())}")
    print(f"  step records:  {n_steps:,}  (avg {n_steps/max(1,n_kept):.1f} step/traj)")
    print(f"  tool dist:")
    for k, v in tool_counts.most_common():
        print(f"    {k:<16} {v:>8,}  ({100*v/max(1,n_steps):.1f}%)")
    print(f"  output:        {out_path}")


if __name__ == '__main__':
    main()
