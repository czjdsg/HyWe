#!/usr/bin/env python3
"""逐字段判定最终答案是否覆盖了 answer_schema,以及缺的字段能不能从前序阶段补。

为什么要这么判:先前我按依赖边分类 —— 前段产出被后段消费就算"中间量、不必进
答案" —— 抽样立刻发现是错的。OSM+GoogleMaps 那条 F4,依赖确实是 o1→s1→o2 的链,
但 answer_schema 七个字段里前三个只有 o1 能给、后四个只有 o2 能给:**被消费不等于
不是交付物**。所以判据必须落在 answer_schema 上:答案缺不缺字段、缺的那个字段是
哪一段产出的。

三种结局要分开,处理方式完全不同:
  * 字段齐 —— 什么都不用做;
  * 缺字段,但值在**前序阶段**的 values_obtained 里 —— 这才是需要补写的那批;
  * 缺字段,且轨迹里压根没有 —— 是任务没做完,补写只会让模型编。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

SYSTEM = """
You decide, field by field, whether an agent's final answer delivers what the task
asked for.

You are given: the task question, the answer_schema, the objective per-phase
progress accounts (including the values obtained in each phase, numbered by
phase), and the final answer.

For EVERY field in answer_schema decide:
- present: does the final answer give this field's actual content? (naming the
  field is not enough — it must give the value)
- available_in: if present is false, which phase's values_obtained /
  what_happened contains this value? Give the phase number; give null if no phase
  has it.
- note: one line of justification, quoting the source material verbatim.

Judging rules:
- Equivalent content counts as present; the wording need not match verbatim
  (e.g. "US$455/night" and "455 dollars per night").
- If the answer only says the value could not be obtained, present is false.
- Fill available_in only for a phase where the value is ACTUALLY VISIBLE. When
  unsure, use null — a wrong phase number makes the downstream merge step splice
  in a value that does not exist, which is worse than a missing field.

Output JSON only:
{"fields": [{"field": "...", "present": true, "available_in": null, "note": "..."}],
 "missing_but_available": ["<missing but recoverable from some phase>", ...],
 "missing_everywhere": ["<missing and absent from the trajectory too>", ...]}
"""


def build_prompt(task: dict, audit: dict, result: dict) -> str:
    pts = task.get("private_teacher_steps") or {}
    schema = task.get("answer_schema") or pts.get("answer_schema") or []
    parts = [f"## 任务\n{result.get('question')}",
             f"\n## answer_schema(必须交付的字段)\n{json.dumps(schema, ensure_ascii=False)}",
             "\n## 各阶段客观进度"]
    for i, ph in enumerate(audit.get("phases") or []):
        prog = ph.get("progress") or {}
        parts.append(f"### 阶段 {i} ({ph.get('mode')} 侧)")
        parts.append(f"what_happened: {prog.get('what_happened')}")
        vo = prog.get("values_obtained")
        if vo:
            parts.append(f"values_obtained: {json.dumps(vo, ensure_ascii=False)}")
    parts.append(f"\n## 最终答案\n{result.get('prediction') or '(空)'}")
    return "\n".join(parts)


def one(client, model, task, audit, result, timeout, attempts):
    pts = task.get("private_teacher_steps") or {}
    schema = task.get("answer_schema") or pts.get("answer_schema") or []
    if not schema:
        return {"error": "该任务没有 answer_schema"}
    prompt = build_prompt(task, audit, result)
    convo = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    last = None
    for i in range(attempts):
        try:
            r = client.chat.completions.create(model=model, messages=convo, timeout=timeout,
                                               max_tokens=2500, temperature=0.0)
            txt = r.choices[0].message.content or ""
            s, e = txt.find("{"), txt.rfind("}")
            out = json.loads(txt[s:e + 1])
            got = [f.get("field") for f in (out.get("fields") or [])]
            if len(got) != len(schema):
                raise ValueError(f"必须逐一覆盖 {len(schema)} 个字段,实际 {len(got)}")
            out["n_phases"] = len(audit.get("phases") or [])
            return out
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:160]
            time.sleep(min(8.0, 1.5 * (2 ** i)))
    return {"error": last}


def _second_dim_ok(j: dict) -> bool:
    """审计的第二个判据。2026-09-02 从 answer.reasonable(答案文本是否合格)改成
    trajectory.succeeded(整条轨迹是否完成任务);新旧产物都要认。"""
    t = (j.get("trajectory") or j.get("answer") or {})
    return t.get("succeeded", t.get("reasonable")) is True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", default="runs/hybrid_rollout_phase_control_qwen38_"
                                             "square1000_full_budget50_100_20260806/rollouts")
    ap.add_argument("--audit-dir", default="runs/unified_audit_full/tasks")
    ap.add_argument("--input", default="runs/hybrid_rollout_v2_teacher_kimi_oss_full/input_all.jsonl")
    ap.add_argument("--out", default="runs/answer_coverage")
    ap.add_argument("--min-phases", type=int, default=2)
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("API_KEY")
    if not key:
        sys.exit("Set JUDGE_API_KEY (或 API_KEY)")

    tasks = {}
    for line in Path(args.input).open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            tasks[str(r["task_id"])] = r

    good, audits = set(), {}
    for f in Path(args.audit_dir).glob("*/*.json"):
        try:
            o = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = str(o.get("task_id") or f.parent.name)
        if o.get("phases"):
            audits[t] = o
        j = o.get("judgement") or {}
        # 版本号用前缀判,别写死具体版本 —— 2026-09-02 升到 v3 时这五处字面相等
        # 全部静默失配,四道过滤空跑,而日志只显示「待判定 0 条」看不出原因。
        if (str(j.get("schema_version") or "").startswith("hybrid_unified_audit_")
                and j.get("task_success") is True
                and _second_dim_ok(j)):
            good.add(t)
    for f in Path("runs/unified_audit_v2/tasks").glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        v = r.get("v2") or {}
        if v.get("task_success") is True and _second_dim_ok(v):
            good.add(str(r["task_id"]))

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t in sorted(good):
        au = audits.get(t)
        if not au or len(au.get("phases") or []) < args.min_phases:
            continue
        if args.resume and (outdir / f"{t}.json").is_file():
            continue
        rp = Path(args.rollout_dir) / t / "result.json"
        if not rp.is_file() or t not in tasks:
            continue
        try:
            res = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        jobs.append((t, tasks[t], au, res))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"待判定 {len(jobs)} 条 → {outdir}", flush=True)

    client = OpenAI(base_url=args.base_url, api_key=key)
    stats = Counter()
    with cf.ThreadPoolExecutor(args.max_workers) as ex:
        futs = {ex.submit(one, client, args.model, task, au, res,
                          args.timeout, args.attempts): t
                for t, task, au, res in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            t = futs[fut]; r = fut.result()
            (outdir / f"{t}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
            if "error" in r:
                stats["失败"] += 1
            elif r.get("missing_but_available"):
                stats["缺字段·可从前序阶段补"] += 1
            elif r.get("missing_everywhere"):
                stats["缺字段·轨迹里也没有"] += 1
            else:
                stats["字段齐全"] += 1
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}  {dict(stats)}", flush=True)
    print(f"\n结果: {dict(stats)}")


if __name__ == "__main__":
    main()
