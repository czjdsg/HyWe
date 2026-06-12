#!/usr/bin/env python3
"""Build SPA viz for v7 pipeline runs.

Reads a Stage-A/B/C run dir and emits:
  viz/manifest.json   - slim site index (counts/kinds for sidebar)
  viz/data/<dom>.json - per-site bundle: full fact table with tags+probe
                        evidence, harvested facts, task_set with chain+phases.
The SPA fetches per-site JSON on demand. Screenshots resolve to
'<dom>/trajectory/...png' via a symlink the start script creates.

Usage:
  python3 build_v7_viz.py --run /tmp/synth7_base --out /tmp/synth7_base_viz
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

SHOT_RE = re.compile(r"(\d+)_full_screenshot\.png")


def _trim(s, n=2400):
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def build_site(site_dir: Path) -> dict | None:
    res_p = site_dir / "result.json"
    if not res_p.is_file():
        return None
    res = json.loads(res_p.read_text())
    domain = site_dir.name
    site = res.get("site") or {}

    # screenshots: step -> relative URL (rendered as ../<dom>/trajectory/...)
    shots: dict[int, str] = {}
    traj = site_dir / "trajectory"
    if traj.is_dir():
        for fn in os.listdir(traj):
            m = SHOT_RE.match(fn)
            if m:
                shots[int(m.group(1))] = f"../{domain}/trajectory/{fn}"

    # ---------- Stage A: KB + notes timeline ----------
    kb = res.get("kb") or {}
    facts_kb = kb.get("facts") or {}
    notes = res.get("notes") or []
    history = res.get("history") or []
    hist_by_step = {h.get("step"): h for h in history if isinstance(h, dict)}
    steps = []
    for n in notes:
        st = n.get("step")
        parsed = n.get("parsed") or {}
        h = hist_by_step.get(st, {})
        steps.append({
            "step": st,
            "mode": n.get("mode") or "?",
            "tool": (parsed.get("tool_call") or {}).get("name"),
            "thought": _trim(parsed.get("thought"), 600),
            "facts_learned": parsed.get("facts_learned") or [],
            "visual_note": parsed.get("visual_note") or "",
            "shot": shots.get(st, ""),
            "obs_status": (h.get("observation") or {}).get("status"),
            "obs_diagnostics": _trim(json.dumps(
                (h.get("observation") or {}).get("diagnostics") or {},
                ensure_ascii=False), 1200),
        })

    # ---------- Stage B: fact tags ----------
    tags = {}
    tag_index = {}
    tg_p = site_dir / "fact_tags.json"
    if tg_p.is_file():
        tg = json.loads(tg_p.read_text())
        for row in tg.get("facts") or []:
            k = row["key"]
            tags[k] = {
                "key": k,
                "value": row.get("value"),
                "step": row.get("step"),
                "mode": row.get("mode"),
                "probe": row.get("probe"),
                "tag": row.get("tag"),
                "reason": row.get("reason"),
                "question": row.get("question"),
                "s_answer": row.get("s_answer"),
                "s_found": row.get("s_found"),
                "s_steps": row.get("s_steps"),
                "harvested": row.get("harvested") or [],
                # full S-side verification trace: each entry has
                # {step, action: search/scholar/visit, args, result_head}
                "s_transcript": row.get("s_transcript") or [],
            }
            tag_index[k] = row.get("tag") or row.get("probe")
        site_counts = tg.get("counts") or {}
    else:
        site_counts = {}

    # Synthesize a unified fact list ordered by step then key
    unified = []
    seen = set()
    # tagged facts (preserve order from tags file)
    for k, t in tags.items():
        unified.append({**t, "row_kind": "real"})
        seen.add(k)
    # harvested facts as own rows, anchored to their parent
    for parent_k, t in tags.items():
        for h in t.get("harvested") or []:
            hkey = f"h__{parent_k[:24]}__{h.get('key','')}"[:80]
            unified.append({
                "key": hkey,
                "value": h.get("value"),
                "source": h.get("source"),
                "anchor": parent_k,
                "tag": "S_HARVESTED",
                "row_kind": "harvested",
            })
    # also catch any KB facts not in tags table (shouldn't happen but be safe)
    for k, v in facts_kb.items():
        if k in seen:
            continue
        unified.append({
            "key": k,
            "value": v.get("value"),
            "step": v.get("step"),
            "mode": v.get("mode"),
            "tag": None,
            "row_kind": "untagged",
        })

    # ---------- Stage C: task_set ----------
    task_p = site_dir / "task.json"
    task_set = None
    if task_p.is_file():
        try:
            task_set = json.loads(task_p.read_text())
        except Exception:
            task_set = None

    return {
        "domain": domain,
        "site": site,
        "steps_used": res.get("steps_used") or len(steps),
        "n_steps": len(steps),
        "elapsed_s": res.get("elapsed_s"),
        "kb_dynamic": kb.get("dynamic") or {},
        "facts": unified,
        "tags": tags,
        "stage_b_counts": site_counts,
        "task_set": task_set,
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    manifest = []
    site_dirs = sorted(d for d in run.iterdir()
                       if d.is_dir() and (d / "result.json").is_file())
    for sd in site_dirs:
        rec = build_site(sd)
        if not rec:
            continue
        dom = rec["domain"]
        (out / "data" / f"{dom}.json").write_text(
            json.dumps(rec, ensure_ascii=False))
        ts = rec.get("task_set") or {}
        manifest.append({
            "domain": dom,
            "category": (rec["site"] or {}).get("category"),
            "n_steps": rec["n_steps"],
            "n_facts": len(rec["facts"]),
            "n_tagged_O": sum(1 for f in rec["facts"]
                              if f.get("tag") == "O_EXCLUSIVE"),
            "n_tagged_S": sum(1 for f in rec["facts"]
                              if f.get("tag") == "S_REACHABLE"),
            "n_harv": sum(1 for f in rec["facts"]
                          if f.get("tag") == "S_HARVESTED"),
            "stage_c_kind": (ts or {}).get("kind") or "no_task",
            "n_tasks": len((ts or {}).get("tasks") or []),
            "difficulties": [t.get("difficulty") for t in
                             ((ts or {}).get("tasks") or [])],
        })
    (out / "manifest.json").write_text(
        json.dumps({"tasks": manifest}, ensure_ascii=False))
    print(f"wrote {len(manifest)} sites; manifest "
          f"{(out / 'manifest.json').stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
