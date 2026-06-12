#!/usr/bin/env python3
"""Periodic QA sampler for the synth7 full-run chain.

Each invocation scans the run dir, samples fresh artifacts, and appends a
timestamped markdown report covering BOTH halves of the pipeline:

  1. Task generation (Stage C): machine stats over all task.json produced so
     far (shape / difficulty / score / start_mode / reject reasons) + an LLM
     spot-review of N sampled tasks (naturalness, stitched-together smell,
     answer leakage) via deepseek-v4-pro on :8000 (keeps :8002 free for the
     pipeline itself).
  2. hybrid_guided_routed rollout: termination distribution, switch usage,
     guided_step_quality verdict mix, guided_final_verdict solve rate, plus
     sampled verdict reasons verbatim.

Stdout protocol: prints one line "QA_REPORT <path> | <summary>" on success —
a tail -F Monitor picks this up for live notification.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

_default_run = os.environ.get("QA_RUN_DIR", str(
    Path(__file__).resolve().parent.parent / "runs" / "synth3_pipe"))
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else _default_run)
QA_DIR = RUN.parent / (RUN.name + "_qa")
QA_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_BASE = os.environ.get("QA_REVIEW_BASE", "http://localhost:8000/v1")
REVIEW_MODEL = os.environ.get("QA_REVIEW_MODEL", "deepseek-v4-pro")
API_KEY = os.environ.get("QA_API_KEY", "EMPTY")
N_TASK_SAMPLE = int(os.environ.get("QA_TASK_SAMPLE", "5"))
N_ROLLOUT_SAMPLE = int(os.environ.get("QA_ROLLOUT_SAMPLE", "5"))


def llm(messages, max_tokens=1500, temperature=0.1, timeout=180, retries=3):
    payload = json.dumps({"model": REVIEW_MODEL, "messages": messages,
                          "max_tokens": max_tokens,
                          "temperature": temperature}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            REVIEW_BASE.rstrip("/") + "/chat/completions",
            data=payload, method="POST")
        req.add_header("Authorization", f"Bearer {API_KEY}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8", errors="replace"))
                return d["choices"][0]["message"]["content"] or ""
        except Exception:
            time.sleep(5 * (attempt + 1))
    return ""


def extract_json(text):
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---------------- stage progress ----------------
def stage_progress():
    n_a = n_b = n_c_set = n_c_other = 0
    for d in RUN.iterdir():
        if not d.is_dir() or d.name.startswith("_") or d.name == "rollout":
            continue
        if (d / "result.json").is_file():
            n_a += 1
        if (d / "fact_tags.json").is_file():
            n_b += 1
        tp = d / "task.json"
        if tp.is_file():
            try:
                k = json.loads(tp.read_text()).get("kind")
                if k == "task_set":
                    n_c_set += 1
                else:
                    n_c_other += 1
            except Exception:
                n_c_other += 1
    n_roll = 0
    legacy = RUN / "rollout"
    if legacy.is_dir():
        n_roll += sum(1 for d in legacy.iterdir()
                      if d.is_dir() and (d / "result.json").is_file())
    n_roll += sum(1 for d in RUN.glob("*/rollout/*")
                  if d.is_dir() and (d / "result.json").is_file())
    return n_a, n_b, n_c_set, n_c_other, n_roll


# ---------------- task generation QA ----------------
def task_gen_stats():
    """Aggregate over every per-site task.json present so far."""
    difs, shapes, scores, starts = Counter(), Counter(), [], Counter()
    rejects, no_tasks = [], 0
    all_tasks = []
    for d in RUN.iterdir():
        if not d.is_dir():
            continue
        tp = d / "task.json"
        if not tp.is_file():
            continue
        try:
            obj = json.loads(tp.read_text())
        except Exception:
            continue
        kind = obj.get("kind")
        if kind == "no_task":
            no_tasks += 1
            continue
        if kind == "rejected":
            rejects.append((d.name, (obj.get("reason") or "")[:140]))
            continue
        for t in obj.get("tasks") or []:
            difs[t.get("difficulty") or "?"] += 1
            shapes[t.get("shape") or "?"] += 1
            if isinstance(t.get("score"), int):
                scores.append(t["score"])
            starts[t.get("start_mode") or "?"] += 1
            all_tasks.append({"domain": d.name, **t})
    return {"difs": difs, "shapes": shapes, "scores": scores,
            "starts": starts, "rejects": rejects, "no_tasks": no_tasks,
            "tasks": all_tasks}


def sample_tasks(tasks, n):
    """No external LLM review — emit raw samples verbatim into the report;
    the supervising agent reads each report and reviews inline."""
    sample = random.sample(tasks, min(n, len(tasks)))
    return [{k: t.get(k) for k in
             ("domain", "question", "shape", "difficulty", "score",
              "start_mode", "answer_sketch")} for t in sample]


# ---------------- rollout QA ----------------
def rollout_stats():
    # two layouts: barrier chain (RUN/rollout/<task>/) and per-site pipeline
    # (RUN/<domain>/rollout/<task>/)
    task_dirs = []
    legacy = RUN / "rollout"
    if legacy.is_dir():
        task_dirs += [d for d in legacy.iterdir() if d.is_dir()]
    task_dirs += [d for d in RUN.glob("*/rollout/*") if d.is_dir()]
    if not task_dirs:
        return None
    terms, fv_verdicts, q_verdicts = Counter(), Counter(), Counter()
    steps_list, switch_counts = [], []
    samples = []
    rows = []
    for d in sorted(task_dirs):
        rp = d / "result.json"
        if not rp.is_file():
            continue
        try:
            r = json.loads(rp.read_text())
        except Exception:
            continue
        recs = r.get("records") or []
        term = r.get("termination") or "?"
        # running checkpoints are in-flight tasks — count separately
        if term == "running":
            terms["(in-flight)"] += 1
            continue
        terms[term] += 1
        steps_list.append(len(recs))
        switch_counts.append(sum(1 for rec in recs
                                 if (rec.get("tool_call") or {}).get("name") == "switch"))
        for rec in recs:
            q = rec.get("guided_step_quality")
            if q:
                q_verdicts[q.get("verdict") or "?"] += 1
        fv = r.get("guided_final_verdict") or (recs[-1].get("guided_final_verdict") if recs else None)
        if fv:
            fv_verdicts[fv.get("verdict") or "?"] += 1
        rows.append({"task": d.name, "term": term, "steps": len(recs),
                     "switches": switch_counts[-1] if switch_counts else 0,
                     "fv": fv})
    if rows:
        samples = random.sample(rows, min(N_ROLLOUT_SAMPLE, len(rows)))
    return {"terms": terms, "fv": fv_verdicts, "q": q_verdicts,
            "steps": steps_list, "switches": switch_counts,
            "samples": samples, "n_done": len(rows)}


# ---------------- report ----------------
def fmt_counter(c):
    total = sum(c.values()) or 1
    return ", ".join(f"{k}: {v} ({v / total * 100:.0f}%)"
                     for k, v in c.most_common())


def main():
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    n_a, n_b, n_c_set, n_c_other, n_roll = stage_progress()
    lines = [f"# QA report — {ts}", "",
             f"## Pipeline progress",
             f"- Stage A explored: **{n_a}**/1170",
             f"- Stage B tagged:   **{n_b}**",
             f"- Stage C task_set: **{n_c_set}** (+{n_c_other} no_task/rejected)",
             f"- Rollout finished: **{n_roll}**", ""]
    summary_bits = [f"A:{n_a} B:{n_b} C:{n_c_set} R:{n_roll}"]

    tg = task_gen_stats()
    if tg["tasks"]:
        scores = tg["scores"]
        lines += ["## Task generation",
                  f"- tasks so far: **{len(tg['tasks'])}** "
                  f"(no_task: {tg['no_tasks']}, rejected: {len(tg['rejects'])})",
                  f"- difficulty: {fmt_counter(tg['difs'])}",
                  f"- shape: {fmt_counter(tg['shapes'])}",
                  f"- start_mode: {fmt_counter(tg['starts'])}",
                  f"- score: min {min(scores) if scores else '-'} / "
                  f"median {sorted(scores)[len(scores)//2] if scores else '-'} / "
                  f"max {max(scores) if scores else '-'}", ""]
        if tg["rejects"]:
            lines.append("### Reject reasons (latest 5)")
            for dom, why in tg["rejects"][-5:]:
                lines.append(f"- `{dom}`: {why}")
            lines.append("")
        samples = sample_tasks(tg["tasks"], N_TASK_SAMPLE)
        if samples:
            lines.append(f"### Sampled tasks ({len(samples)} — for inline "
                         f"review by the supervising agent)")
            for s in samples:
                lines.append(f"- ({s['domain']}) [{s.get('shape')}/"
                             f"{s.get('difficulty')}/score={s.get('score')}/"
                             f"start={s.get('start_mode')}]")
                lines.append(f"  - Q: {(s.get('question') or '')}")
                lines.append(f"  - A: {(s.get('answer_sketch') or '')[:220]}")
            lines.append("")
            summary_bits.append(f"sampled:{len(samples)}")

    rs = rollout_stats()
    if rs and rs["n_done"]:
        avg_steps = sum(rs["steps"]) / max(len(rs["steps"]), 1)
        sw_rate = sum(1 for s in rs["switches"] if s > 0) / max(len(rs["switches"]), 1)
        solved = rs["fv"].get("solved", 0)
        n_fv = sum(rs["fv"].values()) or 1
        lines += ["## Rollout (hybrid_guided_routed)",
                  f"- finished tasks: **{rs['n_done']}**",
                  f"- termination: {fmt_counter(rs['terms'])}",
                  f"- avg steps: {avg_steps:.1f} · tasks with >=1 switch: {sw_rate*100:.0f}%",
                  f"- step quality verdicts: {fmt_counter(rs['q'])}",
                  f"- final verdicts: {fmt_counter(rs['fv'])} "
                  f"→ solve rate **{solved/n_fv*100:.0f}%**", "",
                  "### Sampled final verdicts"]
        for s in rs["samples"]:
            fv = s.get("fv") or {}
            lines.append(f"- `{s['task']}` term={s['term']} steps={s['steps']} "
                         f"switches={s['switches']} → "
                         f"**{fv.get('verdict', 'n/a')}**")
            if fv.get("reason"):
                lines.append(f"  - {fv['reason'][:220]}")
        lines.append("")
        summary_bits.append(
            f"rollout:{rs['n_done']} solve:{solved/n_fv*100:.0f}%")

    stamp = time.strftime("%m%d_%H%M")
    out = QA_DIR / f"qa_{stamp}.md"
    out.write_text("\n".join(lines))
    latest = QA_DIR / "latest.md"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out.name)
    except Exception:
        pass
    print(f"QA_REPORT {out} | {' | '.join(summary_bits)}", flush=True)


if __name__ == "__main__":
    main()
