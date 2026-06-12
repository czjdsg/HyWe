#!/usr/bin/env python3
"""Aggregate the per-site streaming pipeline into a viz dataset.

Walks the run dir and emits:
  <viz>/manifest.json   - one slim row per site (stage reached, counts,
                          skip reason, timing) for the sidebar + filters
  <viz>/data/<dom>.json - per-site bundle: A trajectory + facts, B tags +
                          probe transcripts, C tasks (typed plans),
                          R rollout results (steps, quality verdicts, final
                          verdict). Loaded on demand by the SPA.

Idempotent: safe to re-run on a live run dir (skips half-written json).
The SPA polls manifest.json so finished sites appear incrementally; SKIP
sites are included with their stop stage + reason.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

SHOT_RE = re.compile(r"(\d+)_full_screenshot\.png")


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _trim(s, n=2000):
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def stage_state(run: Path) -> dict:
    """domain -> {stage -> last event dict} from pipeline_state.jsonl."""
    st: dict[str, dict] = {}
    sp = run / "pipeline_state.jsonl"
    if sp.is_file():
        for line in sp.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            st.setdefault(e["domain"], {})[e["stage"]] = e
    return st


def build_site(run: Path, domain: str, state: dict) -> dict | None:
    d = run / domain
    res = _load(d / "result.json")
    if res is None:
        return None
    site = res.get("site") or {}
    kb = res.get("kb") or {}

    # ---- B: load first (A-step rendering folds B verdicts in by key) ----
    tags = _load(d / "fact_tags.json")
    facts, b_counts = [], {}
    b_by_key: dict[str, dict] = {}
    if tags:
        b_counts = tags.get("counts") or {}
        for f in tags.get("facts") or []:
            s_steps = []
            for st in (f.get("s_transcript") or []):
                tool = st.get("tool")
                if not tool:
                    continue
                args = st.get("args") or {}
                qy = args.get("query") or args.get("url") or ""
                if isinstance(qy, list):
                    qy = "; ".join(str(x) for x in qy)
                s_steps.append({
                    "step": st.get("step"), "tool": tool,
                    "query": _trim(str(qy), 200),
                    "obs": _trim(st.get("observation_head")
                                 or st.get("observation") or "", 300),
                })
            row = {
                "key": f.get("key"), "value": f.get("value") or "",
                "mode": f.get("mode"), "tag": f.get("tag"),
                "probe": f.get("probe"),
                "question": f.get("question"),
                "s_answer": f.get("s_answer") or "",
                "s_found": f.get("s_found"), "s_steps": f.get("s_steps"),
                "s_transcript": s_steps,
                "harvested": [{"key": h.get("key"),
                               "value": h.get("value") or ""}
                              for h in (f.get("harvested") or [])],
            }
            facts.append(row)
            if row["key"]:
                b_by_key[row["key"]] = row

    # ---- A: trajectory ----
    records = res.get("records") or res.get("history") or []
    shots = {}
    traj = d / "trajectory"
    if traj.is_dir():
        for fn in os.listdir(traj):
            m = SHOT_RE.match(fn)
            if m:
                shots[int(m.group(1))] = f"../{domain}/trajectory/{fn}"

    def _shot_for(rec):
        """Prefer the explicit shot_path recorded by explorer_agent (the
        EXACT pre-action screenshot the operator saw). Fall back to the
        naive step-number lookup only when the field is missing — that
        fallback is unreliable when runtime.state() is called >once per
        step (e.g. v8's pre+post extraction pattern doubles the counter).
        """
        sp = rec.get("shot_path") or rec.get("img_path")
        if sp:
            # convert absolute /tmp/.../site/trajectory/N.png to relative
            # ../<domain>/trajectory/N.png so the viz can serve it.
            fn = os.path.basename(sp)
            return f"../{domain}/trajectory/{fn}"
        return shots.get(rec.get("step"))
    # ---- v8 fact-extractor notes, keyed by step. Each step may have one
    # fact_raw/fact_parsed pair (the extractor was called after that
    # operator action). We splice these in as their own pseudo-steps right
    # after the matching operator step so the viz shows O→E→O→E alternating.
    fact_by_step: dict[int, dict] = {}
    for n in res.get("notes") or []:
        if "fact_raw" not in n and "fact_parsed" not in n:
            continue
        s = n.get("step")
        if s is None:
            continue
        fact_by_step[int(s)] = {
            "raw": n.get("fact_raw") or "",
            "parsed": n.get("fact_parsed") or None,
            "updates_applied": n.get("fact_updates_applied", 0),
            "shot_path": n.get("shot_path") or "",
        }
    a_steps = []
    for r in records:
        resp = r.get("response_json") or {}
        tc = resp.get("tool_call") or {}
        # facts_learned: include key + value, plus the B-stage verdict
        # (tag / question / S-search outcome) folded in by key so each
        # fact is shown right under the explore step that captured it.
        flearned = []
        for f in (resp.get("facts_learned") or []):
            k_raw = f.get("key") or ""
            if not k_raw:
                continue
            # mirror absorb_facts._norm_key — lowercase alnum + underscores,
            # cap 60 — so we can find the B row even if the explorer used
            # mixed case or punctuation.
            k_norm = "".join(c.lower() if c.isalnum() else "_"
                              for c in k_raw.strip())[:60]
            b = b_by_key.get(k_norm) or b_by_key.get(k_raw)
            entry = {"key": k_raw, "value": f.get("value") or ""}
            if b:
                entry["tag"] = b.get("tag")
                entry["probe"] = b.get("probe")
                entry["question"] = b.get("question")
                entry["s_answer"] = b.get("s_answer")
                entry["s_found"] = b.get("s_found")
                entry["s_steps"] = b.get("s_steps")
                entry["s_transcript"] = b.get("s_transcript") or []
                entry["harvested"] = b.get("harvested") or []
            flearned.append(entry)
        obs_raw = r.get("observation")
        obs_str = obs_raw if isinstance(obs_raw, str) else \
                  json.dumps(obs_raw, ensure_ascii=False)
        op_step = r.get("step")
        # Full tool_call arguments (so the viz can show the exact click
        # coordinates / fill text / scroll direction the operator emitted).
        # Pretty-printed inline JSON keeps it compact in the timeline.
        tc_args = tc.get("arguments") or {}
        try:
            tc_args_str = json.dumps(tc_args, ensure_ascii=False)
        except Exception:
            tc_args_str = str(tc_args)
        # Observation diagnostics: surface the actually-hit element so the
        # reader can see "clicked nav" vs "clicked button 'Accept all'".
        obs_d = r.get("observation") if isinstance(r.get("observation"), dict) else {}
        diag = (obs_d.get("diagnostics") or {}) if obs_d else {}
        ts = diag.get("target_state") or {}
        hit_tag = ts.get("tag", "") if isinstance(ts, dict) else ""
        hit_text = (ts.get("text", "") or "")[:80] if isinstance(ts, dict) else ""
        hit_aria = (ts.get("aria_label", "") or "")[:80] if isinstance(ts, dict) else ""
        sticky = ts.get("sticky_overlay") if isinstance(ts, dict) else None
        sticky_tag = sticky.get("tag", "") if isinstance(sticky, dict) else ""
        obs_status = obs_d.get("status", "") if isinstance(obs_d, dict) else ""
        a_steps.append({
            "step": op_step,
            "mode": r.get("mode") or r.get("active_mode_before"),
            "tool": tc.get("name"),
            "tool_args": tc_args_str,
            "action": tc.get("action_description") or "",
            "thought": resp.get("thought") or "",
            "facts_learned": flearned,
            "obs": obs_str or "",
            "obs_status": obs_status,
            "hit_tag": hit_tag,
            "hit_text": hit_text,
            "hit_aria": hit_aria,
            "hit_sticky": sticky_tag,
            "shot": _shot_for(r),
            # role tag: viewer renders OPERATOR vs EXTRACTOR distinctly
            "role": "operator",
        })
        # v8: splice the matching fact-extractor turn (if any) right after
        # the operator turn that produced the observation it analyzed.
        fx = fact_by_step.get(op_step)
        if fx is not None:
            parsed = fx.get("parsed") or {}
            fx_facts = []
            for f in (parsed.get("facts_learned") or []):
                if isinstance(f, dict) and f.get("key"):
                    k_raw = f.get("key") or ""
                    k_norm = "".join(c.lower() if c.isalnum() else "_"
                                      for c in k_raw.strip())[:60]
                    b = b_by_key.get(k_norm) or b_by_key.get(k_raw)
                    entry = {"key": k_raw, "value": f.get("value") or ""}
                    if b:
                        entry["tag"] = b.get("tag")
                        entry["probe"] = b.get("probe")
                        entry["question"] = b.get("question")
                        entry["s_answer"] = b.get("s_answer")
                        entry["s_found"] = b.get("s_found")
                        entry["s_steps"] = b.get("s_steps")
                        entry["s_transcript"] = b.get("s_transcript") or []
                        entry["harvested"] = b.get("harvested") or []
                    fx_facts.append(entry)
            fx_updates = []
            for u in (parsed.get("facts_updated") or []):
                if isinstance(u, dict) and u.get("key"):
                    fx_updates.append({
                        "key": u.get("key") or "",
                        "value": u.get("value") or "",
                        "reason": u.get("reason") or "",
                    })
            fx_pursue = parsed.get("facts_to_pursue") or []
            fx_note = parsed.get("note") or ""
            a_steps.append({
                "step": op_step,
                "mode": "EXTRACTOR",
                "tool": "fact_agent",
                "action": fx_note or (
                    f"+{len(fx_facts)} fact" + ("s" if len(fx_facts) != 1 else "")
                    + (f", ~{len(fx_updates)} upd" if fx_updates else "")
                    + (f", →{len(fx_pursue)} q" if fx_pursue else "")
                    if (fx_facts or fx_updates or fx_pursue)
                    else "(no fact this turn)"),
                "thought": fx_note,
                "facts_learned": fx_facts,
                "facts_updated": fx_updates,
                "facts_to_pursue": fx_pursue,
                "obs": "",
                # Use the EXACT post-action shot the extractor saw, recorded
                # in fact_by_step.shot_path. Fall back to filename-based
                # lookup only when missing (older runs lacked the field).
                "shot": (f"../{domain}/trajectory/{os.path.basename(fx.get('shot_path'))}"
                          if fx.get('shot_path')
                          else shots.get(op_step)),
                "role": "extractor",
                "extractor_raw": (fx.get("raw") or "")[:600],
                "updates_applied": fx.get("updates_applied", 0),
            })

    # (B loaded above; the loop below is no longer used — kept disabled)
    if False and tags:
        b_counts = tags.get("counts") or {}
        for f in tags.get("facts") or []:
            # search-agent blind-answer transcript (the S-side probe steps)
            s_steps = []
            for st in (f.get("s_transcript") or []):
                tool = st.get("tool")
                if not tool:
                    continue
                args = st.get("args") or {}
                qy = args.get("query") or args.get("url") or ""
                if isinstance(qy, list):
                    qy = "; ".join(str(x) for x in qy)
                s_steps.append({
                    "step": st.get("step"), "tool": tool,
                    "query": _trim(str(qy), 200),
                    "obs": _trim(st.get("observation_head")
                                 or st.get("observation") or "", 300),
                })
            row = {
                # No truncation — show the FULL fact value.
                "key": f.get("key"), "value": f.get("value") or "",
                "mode": f.get("mode"), "tag": f.get("tag"),
                "probe": f.get("probe"),
                "question": f.get("question"),
                "s_answer": f.get("s_answer") or "",
                "s_found": f.get("s_found"), "s_steps": f.get("s_steps"),
                "s_transcript": s_steps,
                "harvested": [{"key": h.get("key"),
                               "value": h.get("value") or ""}
                              for h in (f.get("harvested") or [])],
            }
            facts.append(row)
            if row["key"]:
                b_by_key[row["key"]] = row

    # ---- C: tasks (typed plans) ----
    taskj = _load(d / "task.json")
    c_kind, tasks, c_reject = "pending", [], None
    if taskj:
        c_kind = taskj.get("kind")
        c_reject = taskj.get("reason")
        for t in taskj.get("tasks") or []:
            tasks.append({
                "question": t.get("question"),
                "shape": t.get("shape"), "difficulty": t.get("difficulty"),
                "score": t.get("score"), "start_mode": t.get("start_mode"),
                "n_crossings": t.get("n_crossings"),
                "answer_sketch": _trim(t.get("answer_sketch"), 400),
                "route_plan": t.get("route_plan_typed") or [],
                "o_fact_keys": t.get("o_fact_keys") or [],
                "s_fact_keys": t.get("s_fact_keys") or [],
            })

    # ---- R: rollout ----
    rollouts = []
    rdir = d / "rollout"
    rp = rdir / "results.jsonl"
    # count tasks the rollout subprocess was started with, so the viz can
    # report `n_finished / n_planned` (per-site subprocesses queue multiple
    # tasks and append to results.jsonl per task; an in-flight subprocess
    # will legitimately show fewer results than inputs until it exits).
    n_planned = 0
    ip = rdir / "input.jsonl"
    if ip.is_file():
        for _ in ip.open():
            n_planned += 1
    if rp.is_file():
        # input map task_id -> task meta (for question)
        inputs = {}
        ip = rdir / "input.jsonl"
        if ip.is_file():
            for line in ip.read_text().splitlines():
                try:
                    row = json.loads(line)
                    inputs[row.get("task_id")] = row
                except Exception:
                    pass
        for line in rp.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            recs = r.get("records") or []
            steps = []
            for rec in recs:
                tc = rec.get("tool_call") or {}
                q = rec.get("guided_step_quality") or {}
                args = tc.get("arguments") or {}
                rp = rec.get("report_parsed") or {}
                # surface the search query / visited url so the search path
                # is legible step by step
                qy = args.get("query") or args.get("url") or ""
                if isinstance(qy, list):
                    qy = "; ".join(str(x) for x in qy)
                # O-step browser screenshot — rewrite the absolute path into
                # a viz-relative URL so the SPA can <img src=...> it.
                shot = None
                ip = rec.get("img_path")
                if ip and rec.get("active_mode_before") == "O":
                    # path looks like <RUN>/<dom>/rollout/<task>/trajectory/N.png
                    parts = ip.split("/")
                    if "rollout" in parts:
                        i = parts.index("rollout")
                        if i + 2 < len(parts):
                            shot = (f"../{parts[i-1]}/rollout/{parts[i+1]}"
                                    f"/trajectory/{parts[-1]}")
                # serialize tool args; clip huge code_sandbox payloads
                try:
                    args_pretty = json.dumps(args, ensure_ascii=False, indent=1)
                except Exception:
                    args_pretty = str(args)
                steps.append({
                    "step": rec.get("step"),
                    "mode": rec.get("active_mode_before"),
                    "tool": tc.get("name") or ("answer" if rec.get("answer") else None),
                    "query": _trim(str(qy), 220),
                    "tool_args": _trim(args_pretty, 1600),
                    "obs": _trim(rec.get("observation") or "", 320),
                    "obs_full": _trim(rec.get("observation") or "", 4000),
                    "action": rec.get("action_description") or "",
                    "verdict": q.get("verdict"),
                    "verdict_reason": _trim(q.get("reason") or "", 400),
                    "is_answer": bool(rec.get("answer")),
                    "answer_text": _trim(rec.get("answer") or "", 1200),
                    "shot": shot,
                    "current_url": rec.get("current_url") or "",
                    "current_title": rec.get("current_title") or "",
                    "task_analysis": _trim(rp.get("task_analysis") or "", 1200),
                    "this_turn": _trim(rp.get("this_turn") or "", 800),
                    "cumulative_progress": _trim(rp.get("cumulative_progress") or "", 1400),
                    "rationale": _trim(rp.get("next_action_rationale") or "", 600),
                    "current_subgoal": _trim(rp.get("current_subgoal") or "", 600),
                    "remaining": rp.get("remaining_subgoals") or [],
                })
            inp_row = inputs.get(r.get("task_id")) or {}
            plan = inp_row.get("route_plan")
            if isinstance(plan, str):
                try:
                    plan = json.loads(plan)
                except Exception:
                    plan = []
            sig = inp_row.get("_signal")
            if isinstance(sig, str):
                try:
                    sig = json.loads(sig)
                except Exception:
                    sig = {}
            rollouts.append({
                "task_id": r.get("task_id"),
                "question": inp_row.get("question"),
                "route_plan": plan or [],
                "shape": (sig or {}).get("shape"),
                "difficulty": (sig or {}).get("difficulty"),
                "start_mode": inp_row.get("start_mode"),
                "termination": r.get("termination"),
                "prediction": _trim(r.get("prediction") or "", 500),
                "n_steps": len(recs),
                "n_switch": sum(1 for s in steps if s["tool"] == "switch"),
                "final_verdict": r.get("guided_final_verdict"),
                "steps": steps,
            })

    return {
        "domain": domain, "site": site,
        "state": state.get(domain, {}),
        "a": {"n_steps": len(a_steps), "n_facts": len(kb.get("facts") or {}),
              "elapsed_s": res.get("elapsed_s"), "steps": a_steps},
        "b": {"counts": b_counts, "facts": facts},
        "c": {"kind": c_kind, "reject": c_reject, "tasks": tasks},
        "r": {"rollouts": rollouts, "n_planned": n_planned,
              "in_flight": max(0, n_planned - len(rollouts))},
    }


def stop_stage(rec: dict, state: dict) -> tuple[str, str]:
    """Furthest stage reached + skip reason if any."""
    a = rec["a"]; b = rec["b"]; c = rec["c"]; r = rec["r"]
    st = state
    if r["rollouts"]:
        return "R", ""
    if c["kind"] == "task_set" and c["tasks"]:
        return "R_pending", ""
    if c["kind"] in ("no_task", "rejected"):
        return "C", c["reject"] or c["kind"]
    if b["counts"]:
        return "C_pending", ""
    if a["n_facts"] == 0:
        return "A", "no facts from exploration"
    # B running or failed
    bev = st.get("B")
    if bev and not bev.get("ok"):
        return "B", f"probe failed rc={bev.get('rc')}"
    return "B_pending", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    state = stage_state(run)
    domains = sorted(d.name for d in run.iterdir()
                     if d.is_dir() and d.name not in ("_logs", "shims", "rollout")
                     and (d / "result.json").is_file())

    manifest = []
    agg = {"A": 0, "B": 0, "C": 0, "R": 0, "skip": 0, "tasks": 0, "rollouts": 0,
           "solved": 0, "verdicts": 0}
    def last_ts(dom):
        """Most recent mtime across A/B/C/R artifacts for this site."""
        d = run / dom
        ts = 0
        for name in ("result.json", "fact_tags.json", "task.json"):
            p = d / name
            if p.is_file():
                ts = max(ts, int(p.stat().st_mtime))
        roll = d / "rollout"
        if roll.is_dir():
            for sub in roll.iterdir():
                rp = sub / "result.json" if sub.is_dir() else None
                if rp and rp.is_file():
                    ts = max(ts, int(rp.stat().st_mtime))
            rj = roll / "results.jsonl"
            if rj.is_file():
                ts = max(ts, int(rj.stat().st_mtime))
        return ts
    for dom in domains:
        rec = build_site(run, dom, state)
        if rec is None:
            continue
        stage, reason = stop_stage(rec, state)
        n_tasks = len(rec["c"]["tasks"])
        n_roll = len(rec["r"]["rollouts"])
        solved = sum(1 for ro in rec["r"]["rollouts"]
                     if (ro.get("final_verdict") or {}).get("task_solved"))
        verdicts = sum(1 for ro in rec["r"]["rollouts"]
                       if ro.get("final_verdict"))
        (out / "data" / f"{dom}.json").write_text(
            json.dumps(rec, ensure_ascii=False))
        manifest.append({
            "domain": dom,
            "category": (rec["site"] or {}).get("category"),
            "stage": stage, "reason": reason,
            "a_facts": rec["a"]["n_facts"], "a_steps": rec["a"]["n_steps"],
            "b": rec["b"]["counts"],
            "c_kind": rec["c"]["kind"], "n_tasks": n_tasks,
            "n_rollouts": n_roll, "n_solved": solved,
            "n_planned": rec["r"]["n_planned"],
            "in_flight": rec["r"]["in_flight"],
            "ts": last_ts(dom),
        })
        if rec["a"]["n_facts"]:
            agg["A"] += 1
        if rec["b"]["counts"]:
            agg["B"] += 1
        if rec["c"]["kind"] in ("task_set", "no_task", "rejected"):
            agg["C"] += 1
        if n_roll:
            agg["R"] += 1
        if stage in ("A", "B", "C") and reason:
            agg["skip"] += 1
        agg["tasks"] += n_tasks
        agg["rollouts"] += n_roll
        agg["solved"] += solved
        agg["verdicts"] += verdicts

    (out / "manifest.json").write_text(json.dumps(
        {"agg": agg, "n_sites": len(manifest), "sites": manifest},
        ensure_ascii=False))
    print(f"wrote {len(manifest)} sites | {agg}")


if __name__ == "__main__":
    main()
