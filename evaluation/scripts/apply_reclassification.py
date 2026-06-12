#!/usr/bin/env python3
"""Merge dual-axis reclassification results back into the point_classification caches.

Input: outputs/reclassify_shards/_result.json  — the Workflow's structured output:
    {"shards": [{"shardId":0, "tasks":[{"task_id","dataset","classifications":[
        {"point_id","axes":["evidence","answer"],"reason","qc_corrected"?}]}], ...}], ...}

For each original cache row, joined on (dataset, task_id):
  - set row["axes_by_point"]  = [axes for each point, in point order]   (row-level, loader's 1st choice)
  - set classifications[i]["axes"] = the point's axes                    (per-classification, loader's 2nd choice)
  - set classifications[i]["kind"] = dominant_kind(axes)                 (evidence_bearing if evidence in axes)
  - set row["kinds"][i]            = dominant_kind(axes)                 (keep length-N invariant)
  - keep classifications[i]["reason"] = the new reason
Writes <cache>.dualaxis next to each original, then (on --replace) swaps in place with .bak backup.
"""
import json, sys, argparse
from pathlib import Path

DATASET_FILES = {
    "wv": "outputs/web_webvoyager_643_final.point_classification.claude-opus-4-8.jsonl",
    "m2w": "outputs/web_online_mind2web_300_temporal_preserve_256_rewrites.point_classification.claude-opus-4-8.jsonl",
}

def dominant_kind(axes):
    return "evidence_bearing" if "evidence" in axes else "answer_bearing"

def norm_axes(axes):
    """Ordered, distinct, evidence-first."""
    s = set(a for a in axes if a in ("answer", "evidence"))
    if not s:
        s = {"evidence"}
    return [a for a in ("evidence", "answer") if a in s]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="outputs/reclassify_shards/_result.json")
    ap.add_argument("--replace", action="store_true", help="swap caches in place (.bak backup)")
    args = ap.parse_args()

    result = json.loads(Path(args.result).read_text())
    # build lookup: (dataset, task_id) -> {point_id: axes}
    lut = {}
    qc_corrected = 0
    for shard in result["shards"]:
        for t in shard["tasks"]:
            key = (t["dataset"], t["task_id"])
            pm = {}
            for c in t["classifications"]:
                pm[c["point_id"]] = norm_axes(c["axes"])
                if c.get("qc_corrected"):
                    qc_corrected += 1
            lut[key] = pm
    print(f"loaded reclassification for {len(lut)} tasks; qc_corrected points={qc_corrected}")

    grand = {"rows": 0, "points": 0, "dual": 0, "answer_only": 0, "evidence_only": 0, "missing": 0}
    for tag, path in DATASET_FILES.items():
        path = Path(path)
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        out_lines = []
        for row in rows:
            key = (tag, row["task_id"])
            pm = lut.get(key)
            n = len(row["answer_key_points"])
            if pm is None:
                grand["missing"] += 1
                print(f"  WARN missing reclassification: {key}")
                # keep row unchanged
                out_lines.append(json.dumps(row, ensure_ascii=False))
                continue
            axes_by_point = []
            for i in range(n):
                axes = pm.get(i)
                if axes is None:
                    # fall back to existing kind -> single axis
                    old = (row.get("kinds") or ["evidence_bearing"] * n)[i] if i < len(row.get("kinds", [])) else "evidence_bearing"
                    axes = ["answer"] if old == "answer_bearing" else ["evidence"]
                axes_by_point.append(axes)
            # write back
            row["axes_by_point"] = axes_by_point
            row["kinds"] = [dominant_kind(a) for a in axes_by_point]
            # sync per-classification entries (sorted by point_id)
            cls = sorted(row.get("classifications", []), key=lambda c: c.get("point_id", 0))
            for i, c in enumerate(cls):
                if i < len(axes_by_point):
                    c["axes"] = axes_by_point[i]
                    c["kind"] = dominant_kind(axes_by_point[i])
            row["classifications"] = cls
            # stats
            for a in axes_by_point:
                grand["points"] += 1
                if len(a) == 2:
                    grand["dual"] += 1
                elif a == ["answer"]:
                    grand["answer_only"] += 1
                else:
                    grand["evidence_only"] += 1
            grand["rows"] += 1
            out_lines.append(json.dumps(row, ensure_ascii=False))
        new_path = path.with_suffix(path.suffix + ".dualaxis")
        new_path.write_text("\n".join(out_lines) + "\n")
        print(f"  wrote {new_path} ({len(out_lines)} rows)")
        if args.replace:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                bak.write_text(path.read_text())
                print(f"  backed up {path} -> {bak}")
            path.write_text("\n".join(out_lines) + "\n")
            new_path.unlink()
            print(f"  replaced {path} in place")
    print("SUMMARY:", grand)

if __name__ == "__main__":
    main()
