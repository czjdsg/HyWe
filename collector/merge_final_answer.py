#!/usr/bin/env python3
"""把散落在各阶段的产出合并进最终答案。

问题:阶段控制器是双盲的 —— 教师只看得到**当前阶段**的目标,看不到全局契约。
跑到最后一段拿到作答许可时,它按"当前目标"作答,于是最终答案往往只覆盖最后一个
组件。实测 730 条多阶段双达标轨迹里,前序阶段获取的值出现在最终答案里的比例
中位只有 14%,其中 241 条(33%)一个都没有。

合并有两路素材,按可得性择优:
  A. 中途被驳回的 answer —— 教师在前一阶段尝试作答、被控制器拦下的那一版。
     它已经是组织好的、面向用户的成稿,合并质量最高。340 条轨迹有。
  B. 各阶段 progress 里的 values_obtained —— 结构化的值,没有成稿,但覆盖全。

无论走哪路,回验口径一致:合并后的每个具体值都要能在轨迹的观测里找到出处。
这条是整件事的价值所在 —— 没有它,"合并"就退化成让模型把几段话拼一起,顺带
编出一些看着像样的细节,比原来那个不完整的答案更难发现问题。

产物写旁路目录,不动 result.json。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

SYSTEM = """
You are merging findings scattered across the phases of an agent trajectory into
one complete final answer.

Background: this trajectory ran in several phases, each solving part of the task.
When the agent wrote its answer in the last phase, it often reported only that
phase's result and dropped what earlier phases had worked to find.

You are given: the task question, the answer_schema, the objective per-phase
progress accounts (including the values already obtained), any mid-run answer
drafts, and the current final answer.

Requirements:
1. USE ONLY VALUES THAT APPEAR IN THE GIVEN MATERIAL. Add no facts, do not infer,
   do not bring in outside knowledge.
2. Cover EVERYTHING the task asked for: findings from earlier phases must be
   written in, not just the last stretch.
3. Keep whatever is already correct in the current final answer — you are
   COMPLETING it, not rewriting it.
4. Organise it by answer_schema: itemise what should be itemised, carry units
   where they belong, and keep each value in the form it appears in the material.
5. For a field genuinely absent from the material, say plainly that it was not
   obtained. Do not invent it.

Output JSON only:
{"answer": "<the merged, complete answer>",
 "used_values": ["<every concrete value used in the answer, verbatim>", ...],
 "added_from_earlier": ["<each point brought in from an earlier phase>", ...]}
"""

ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
NUM = re.compile(r"\d")


def flat_values(v, out: list) -> None:
    if isinstance(v, dict):
        for x in v.values():
            flat_values(x, out)
    elif isinstance(v, list):
        for x in v:
            flat_values(x, out)
    elif v is not None:
        s = str(v).strip()
        if 2 <= len(s) <= 200:
            out.append(s)


def traj_blob(recs: list) -> str:
    """回验语料:观测 + agent 自述 + 页面标题。合并后的值必须落在这里面。"""
    out = []
    for r in recs:
        for k in ("observation", "tool_result", "report", "action_description",
                  "current_url", "current_title", "raw_response"):
            v = r.get(k)
            if v:
                out.append(str(v))
    return re.sub(r"[\s,，]", "", " ".join(out)).lower()


def verify(used: list, blob: str) -> tuple[list, list]:
    ok, bad = [], []
    for v in used:
        s = str(v).strip()
        if not s:
            continue
        # 只对含数字或较长的专名较真:纯描述性短语("价格""可用")无从校验,
        # 硬验只会把噪音刷满、淹掉真正的编造。
        if not (NUM.search(s) or len(s) >= 8):
            continue
        (ok if re.sub(r"[\s,，]", "", s).lower() in blob else bad).append(s)
    return ok, bad


def build_prompt(task: dict, audit: dict, result: dict, mids: list) -> str:
    pts = task.get("private_teacher_steps") or {}
    schema = task.get("answer_schema") or pts.get("answer_schema") or []
    parts = [f"## 任务\n{result.get('question')}",
             f"\n## answer_schema\n{json.dumps(schema, ensure_ascii=False)}"]
    parts.append("\n## 各阶段客观进度")
    for i, ph in enumerate(audit.get("phases") or []):
        prog = ph.get("progress") or {}
        parts.append(f"### 阶段 {i} ({ph.get('mode')})")
        parts.append(f"what_happened: {prog.get('what_happened')}")
        vo = prog.get("values_obtained")
        if vo:
            parts.append(f"values_obtained: {json.dumps(vo, ensure_ascii=False)}")
    if mids:
        parts.append("\n## 中途答案草稿(前序阶段写的,被拦下但内容有效)")
        for st, txt in mids:
            parts.append(f"[step {st}]\n{txt}")
    parts.append(f"\n## 当前最终答案(往往只覆盖最后一个阶段)\n{result.get('prediction') or '(空)'}")
    return "\n".join(parts)


def one(client, model, task, audit, result, mids, recs, timeout, attempts):
    prompt = build_prompt(task, audit, result, mids)
    blob = traj_blob(recs)
    convo = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    last = None
    for i in range(attempts):
        try:
            r = client.chat.completions.create(model=model, messages=convo, timeout=timeout,
                                               max_tokens=4000, temperature=0.0)
            txt = r.choices[0].message.content or ""
            s, e = txt.find("{"), txt.rfind("}")
            out = json.loads(txt[s:e + 1])
            if not out.get("answer"):
                raise ValueError("answer 为空")
            ok, bad = verify(out.get("used_values") or [], blob)
            if not bad:
                out["_verified"] = len(ok)
                out["_source"] = "mid_answer" if mids else "values_obtained"
                return out
            last = f"以下值在轨迹里找不到出处: {bad[:5]}"
            convo = convo[:2] + [{"role": "assistant", "content": txt},
                                 {"role": "user", "content": f"{last}。请去掉或改正后重新输出 JSON。"}]
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:160]
            time.sleep(min(8.0, 1.5 * (2 ** i)))
    return {"error": str(last)[:220]}


def _second_dim_ok(j: dict) -> bool:
    """审计的第二个判据。2026-09-02 从 answer.reasonable(答案文本是否合格)
    改成 trajectory.succeeded(整条轨迹是否完成任务) —— 答案文本的完备性
    由 check_answer_coverage 逐字段判,放两处是重复评判。
    新旧产物都要认:改名当天漏了这些下游消费者,四道过滤全部空跑。"""
    t = (j.get("trajectory") or j.get("answer") or {})
    v = t.get("succeeded", t.get("reasonable"))
    return v is True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", default="runs/hybrid_rollout_phase_control_qwen38_"
                                             "square1000_full_budget50_100_20260806/rollouts")
    ap.add_argument("--audit-dir", default="runs/unified_audit_full/tasks")
    ap.add_argument("--input", default="runs/hybrid_rollout_v2_teacher_kimi_oss_full/input_all.jsonl")
    ap.add_argument("--out", default="runs/answer_merged")
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--only-ids", type=Path, default=None,
                    help="只处理这个 JSON 清单里的 task_id")
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

    # 双达标 + 多阶段才需要合并:单段轨迹没有"前序阶段"可言
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

    only = None
    if args.only_ids and args.only_ids.is_file():
        only = set(json.loads(args.only_ids.read_text(encoding="utf-8")))
        print(f"限定 {len(only)} 个 task_id", flush=True)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t in sorted(good):
        if only is not None and t not in only:
            continue
        au = audits.get(t)
        if not au or len(au.get("phases") or []) < 2:
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
        recs = res.get("records") or []
        mids = []
        for r in recs:
            if not r.get("answer_rejected_by_phase_controller"):
                continue
            m = ANS_RE.search(str(r.get("raw_response") or ""))
            txt = (m.group(1).strip() if m else str(r.get("answer") or "")).strip()
            if txt:
                mids.append((r.get("step"), txt))
        jobs.append((t, tasks[t], au, res, mids, recs))
    if args.limit:
        jobs = jobs[:args.limit]
    with_mid = sum(1 for j in jobs if j[4])
    print(f"待合并 {len(jobs)} 条(其中 {with_mid} 条有中途答案草稿)→ {outdir}", flush=True)

    client = OpenAI(base_url=args.base_url, api_key=key)
    stats = Counter()
    with cf.ThreadPoolExecutor(args.max_workers) as ex:
        futs = {ex.submit(one, client, args.model, task, au, res, mids, recs,
                          args.timeout, args.attempts): t
                for t, task, au, res, mids, recs in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            t = futs[fut]; r = fut.result()
            (outdir / f"{t}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
            stats["失败" if "error" in r else f"通过·{r.get('_source')}"] += 1
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}  {dict(stats)}", flush=True)
    print(f"\n结果: {dict(stats)}")


if __name__ == "__main__":
    main()
