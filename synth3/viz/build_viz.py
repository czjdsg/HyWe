#!/usr/bin/env python3
"""Build SPA-viz data for synth3 explorer runs.

Reads outputs/<run>/<domain>/result.json + trajectory/*.png and produces:
  viz/data/<domain>.json   slim per-task data the SPA consumes
  viz/manifest.json        list of all tasks
The SPA itself (index.html) is colocated and reads these files via fetch.
"""
import argparse, glob, json, os, re

SCREENSHOT_RE = re.compile(r"(\d+)_full_screenshot\.png")


def _truncate(s, n=4000):
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def build_one(task_dir: str, out_dir: str, run_root: str) -> dict:
    rp = os.path.join(task_dir, "result.json")
    if not os.path.isfile(rp):
        return {}
    r = json.load(open(rp))
    domain = os.path.basename(task_dir.rstrip("/"))

    # screenshots
    shots = {}
    traj = os.path.join(task_dir, "trajectory")
    if os.path.isdir(traj):
        for fn in os.listdir(traj):
            m = SCREENSHOT_RE.match(fn)
            if m:
                shots[int(m.group(1))] = f"../{domain}/trajectory/{fn}"

    # build interleaved step list: each step = router decide + LLM raw + parsed + observation
    notes = r.get("notes", [])
    history = r.get("history", [])
    history_by_step = {h["step"]: h for h in history}

    steps = []
    for n in notes:
        step = n.get("step")
        parsed = n.get("parsed") or {}
        h = history_by_step.get(step, {})
        steps.append({
            "step": step,
            "mode": n.get("mode") or "?",
            "raw_response": _truncate(n.get("raw", ""), 4000),
            "parsed_kind": parsed.get("kind"),
            "parsed": parsed,
            "observation": h.get("observation", {}),
            "screenshot": shots.get(step, ""),
            "error": _truncate(n.get("error", ""), 800),
        })

    return {
        "domain": domain,
        "site": r.get("site", {}),
        "candidates": r.get("candidates", []),
        "steps_used": r.get("steps_used", len(history)),
        "elapsed_s": r.get("elapsed_s"),
        "steps": steps,
        "n_steps": len(steps),
        "n_candidates": len(r.get("candidates", [])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="root dir containing per-domain task subdirs")
    ap.add_argument("--out", required=True, help="output dir to populate viz/data + manifest")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    task_dirs = sorted(
        os.path.dirname(p) for p in glob.glob(os.path.join(args.run, "*", "result.json"))
    )
    print(f"Found {len(task_dirs)} task dirs")

    manifest = []
    for td in task_dirs:
        rec = build_one(td, args.out, args.run)
        if not rec:
            continue
        domain = rec["domain"]
        with open(os.path.join(args.out, "data", domain + ".json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False)
        manifest.append({
            "domain": rec["domain"],
            "category": rec["site"].get("category"),
            "entry_url": rec["site"].get("entry_url"),
            "n_steps": rec["n_steps"],
            "n_candidates": rec["n_candidates"],
        })
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"tasks": manifest}, f, ensure_ascii=False, indent=2)
    print(f"manifest written ({len(manifest)} tasks) -> {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
