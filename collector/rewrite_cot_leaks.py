#!/usr/bin/env python3
"""重写思维链里泄露私有信道的句子。

病因:采集时教师被一个双盲的阶段控制器驱动 —— 它下发目标、驳回不合时宜的
switch/answer、给出反馈。这些是学生**看不到**的私有信道,但教师常在 <think> 里
直说"按照控制器反馈…""如当前 rollout 目标所述""我现在只能调用 switch"。照原样
训练,等于教模型引用一个推理时不存在的东西。

改写的边界:只动泄露的那部分表述,**事实、结论、这一步采取的动作都不能变**。
所以:
  * rewrite_scope=quote_only 的只替换那一句,其余原样保留;
  * whole_think 的整段重写,但必须保留原有的每一个事实与决定;
  * 逐条回验 —— 泄露词消失、没有引入原文没有的数值、动作描述仍然吻合。
改写产物写到旁路目录,不碰 result.json:口径变更必须可回溯,而且导出时才决定用哪版。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

SYSTEM = """
You are revising a fragment of a web agent's chain of thought.

Background: while this trajectory was collected, a private phase controller drove
it — issuing the current objective, rejecting premature switch/answer attempts,
and sending feedback. None of that exists at inference time, so the chain of
thought must not refer to it.

Your task: rewrite the flagged leaking statements so they read as conclusions the
agent reached from PUBLIC information only — the task itself, the pages and search
results it has seen, and its own earlier reasoning.

Hard requirements:
1. FACTS UNCHANGED. Every number, name, date, URL and conclusion in the original
   stays exactly as it is; introduce no fact that was not already there.
2. DECISION UNCHANGED. Whatever this step decided to do (click / search / switch /
   answer) must remain the same decision, and the new reasoning must genuinely
   support it — replace "because the controller told me to switch" with the real
   observation-based reason, e.g. "live pricing is not reachable from this path, so
   I need to act on the site".
3. COHERENT VOICE. The rewritten sentences must join the surrounding text smoothly,
   leaving no abrupt seam or empty filler transition.
4. KEEP THE ORIGINAL LANGUAGE (English stays English, Chinese stays Chinese).
5. Must not contain: controller, objective, feedback, rollout, hidden instruction,
   current objective, private plan, "the system told me to", or any other phrasing
   that points at a private channel.

Output JSON only:
{"think": "<the full rewritten think text>", "changed": ["<each original sentence you changed>", ...]}
"""


LEAK_PAT = re.compile(
    r"(controller|rollout objective|current objective|hidden instruction|"
    r"private plan|feedback (?:said|confirms|indicates|requires)|"
    r"as (?:specified|instructed) in the|I can only use|I am only allowed to|"
    r"控制器|私有|隐藏指令|当前目标要求|系统(?:要求|指示)我)", re.I)
NUM = re.compile(r"\d[\d.,:/%$-]*")


def context_for(recs: list, step: int) -> str:
    """给前一步的动作和观测摘要,让改写后的句子能接上上文。"""
    prev = [r for r in recs if isinstance(r.get("step"), int) and r["step"] < step]
    if not prev:
        return "(这是本段第一步)"
    r = prev[-1]
    act = json.dumps(r.get("tool_call") or {}, ensure_ascii=False)[:200]
    obs = str(r.get("observation") or "")[:400]
    return f"上一步动作: {act}\n上一步观测(截断): {obs}"


def verify(orig: str, new: str, quotes: list[str]) -> list[str]:
    """回验。返回问题列表,空列表表示通过。"""
    bad = []
    if not new.strip():
        return ["改写结果为空"]
    if LEAK_PAT.search(new):
        bad.append(f"仍含泄露词: {LEAK_PAT.search(new).group(0)!r}")
    for q in quotes:
        q = str(q or "").strip()
        if len(q) >= 12 and q.lower() in new.lower():
            bad.append(f"原泄露句仍在: {q[:50]!r}")
    # 不得引入原文没有的数值 —— 这是"事实不变"最容易机械检查的一面
    on = {re.sub(r"[^\d.]", "", x) for x in NUM.findall(orig)}
    nn = {re.sub(r"[^\d.]", "", x) for x in NUM.findall(new)}
    extra = {x for x in nn - on if len(x.strip(".")) >= 2}
    if extra:
        bad.append(f"引入了原文没有的数值: {sorted(extra)[:5]}")
    if len(new) < len(orig) * 0.35:
        bad.append(f"内容大幅缩水({len(orig)}→{len(new)} 字符),可能丢了事实")
    return bad


def one(client, model, task_q, rec, leaks, ctx, timeout, attempts):
    think = ((rec.get("report_parsed") or {}).get("think") or "")
    if not think:
        m = re.search(r"<think>(.*?)</think>", str(rec.get("raw_response") or ""), re.S)
        think = m.group(1).strip() if m else ""
    if not think:
        return {"error": "该步没有 think 文本"}
    action = json.dumps(rec.get("tool_call") or {}, ensure_ascii=False)[:300]
    quotes = [str(x.get("exact_quote") or "") for x in leaks]
    scope = "whole_think" if any(x.get("rewrite_scope") == "whole_think" for x in leaks) else "quote_only"
    prompt = "\n".join([
        f"## 任务\n{task_q}",
        f"\n## 上文\n{ctx}",
        f"\n## 这一步实际采取的动作(不可更改)\n{action or '(无工具调用,是作答步)'}",
        f"\n## 当前 think 原文\n{think}",
        f"\n## 需要消除的泄露表述({scope})\n" +
        "\n".join(f"- {json.dumps(x, ensure_ascii=False)}" for x in leaks),
        "\n改写范围:" + ("整段重写,但保留全部事实与决定。" if scope == "whole_think"
                        else "只替换列出的句子,其余原样保留。"),
    ])
    last = None
    convo = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    for i in range(attempts):
        try:
            r = client.chat.completions.create(model=model, messages=convo, timeout=timeout,
                                               max_tokens=3000, temperature=0.0)
            txt = r.choices[0].message.content or ""
            s, e = txt.find("{"), txt.rfind("}")
            out = json.loads(txt[s:e + 1])
            new = str(out.get("think") or "")
            bad = verify(think, new, quotes)
            if not bad:
                return {"think": new, "changed": out.get("changed") or [],
                        "orig_len": len(think), "new_len": len(new)}
            last = "; ".join(bad)
            # 把问题回灌让它自纠,比直接重来省一轮
            convo = convo[:2] + [{"role": "assistant", "content": txt},
                                 {"role": "user", "content": f"不合格:{last}。请重新输出修正后的 JSON。"}]
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:150]
            time.sleep(min(8.0, 1.5 * (2 ** i)))
    return {"error": last}


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
    ap.add_argument("--out", default="runs/cot_rewrite")
    ap.add_argument("--only-qualified", action="store_true", default=True)
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

    # 双达标集合(v2 口径,两个来源)
    good = set()
    for f in Path(args.audit_dir).glob("*/*.json"):
        try:
            o = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        j = o.get("judgement") or {}
        # 版本号用前缀判,别写死具体版本 —— 2026-09-02 升到 v3 时这五处字面相等
        # 全部静默失配,四道过滤空跑,而日志只显示「待判定 0 条」看不出原因。
        if (str(j.get("schema_version") or "").startswith("hybrid_unified_audit_")
                and j.get("task_success") is True
                and _second_dim_ok(j)):
            good.add(str(o.get("task_id")))
    for f in Path("runs/unified_audit_v2/tasks").glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        v = r.get("v2") or {}
        if v.get("task_success") is True and _second_dim_ok(v):
            good.add(str(r["task_id"]))

    # 收集泄露:task -> step -> [leak]
    leaks: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for f in Path(args.audit_dir).glob("*/*.json"):
        try:
            o = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = str(o.get("task_id") or f.parent.name)
        if args.only_qualified and t not in good:
            continue
        for ph in (o.get("phases") or []):
            for x in (ph.get("cot_leaks") or []):
                st = x.get("step")
                if isinstance(st, int):
                    leaks[t][st].append(x)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t, per_step in leaks.items():
        dst = outdir / f"{t}.json"
        if args.resume and dst.is_file():
            continue
        rp = Path(args.rollout_dir) / t / "result.json"
        if not rp.is_file():
            continue
        try:
            res = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        recs = res.get("records") or []
        q = str(res.get("question") or "")
        by_step = {r.get("step"): r for r in recs if isinstance(r.get("step"), int)}
        for st, ls in per_step.items():
            rec = by_step.get(st)
            if rec:
                jobs.append((t, st, q, rec, ls, context_for(recs, st)))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"待改写 {len(jobs)} 步,涉及 {len({j[0] for j in jobs})} 条轨迹 → {outdir}", flush=True)

    client = OpenAI(base_url=args.base_url, api_key=key)
    results: dict[str, dict] = defaultdict(dict)
    from collections import Counter
    stats = Counter()
    with cf.ThreadPoolExecutor(args.max_workers) as ex:
        futs = {ex.submit(one, client, args.model, q, rec, ls, ctx,
                          args.timeout, args.attempts): (t, st)
                for t, st, q, rec, ls, ctx in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            t, st = futs[fut]
            r = fut.result()
            results[t][str(st)] = r
            stats["失败" if "error" in r else "通过"] += 1
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}  {dict(stats)}", flush=True)

    for t, per in results.items():
        (outdir / f"{t}.json").write_text(json.dumps(per, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"\n结果: {dict(stats)}")
    print(f"写出 {len(results)} 个文件 → {outdir}")


if __name__ == "__main__":
    main()
