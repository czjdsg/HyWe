#!/usr/bin/env python3
"""判定 switch 那一步给出的理由,是不是该 form 想教的那个理由。

训练数据里 switch 步是最关键的一轮 —— 模型要学的正是"什么时候换路径、凭什么换"。
理由写歪了,学到的就是错的触发条件:

  F1(O→S):要教的是**认出这是检索型任务、浏览器路径更慢**。
      合格:"这是事实查询,搜索能直接拿到结构化结果,不必在预订站里点"
      不合格:"页面坏了/点不动"(那是站点故障,不是路径选择)、"控制器让我切"

  F2(S→O):要教的是**搜索确实拿不到、必须去站上实时操作**。
      合格:"两次不同查询都只返回单条记录页,拿不到按条件筛选的完整列表"
      不合格:"visit 工具在这条路径上没有了"(那是工具可用性)、"控制器说该切"、
              以及只说"我要去操作"却不说搜索为什么不行

抽样里两种不合格都真实出现过,所以这一项必须判,不能假定 switch_reason 天然合格。
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

SYS_F1 = """
You judge the reason a web agent gave when switching from the OPERATION path (O)
to the SEARCH path (S).

Design intent of this task family: the run starts on some site, and the agent is
supposed to RECOGNISE that this is a lookup-style question that search answers
faster, and switch away early. That judgement is exactly what we want to teach.

verdict:
- "recognizes_search_faster" — the reason expresses "this is a factual/lookup
  question, search gets it faster and more completely, no page interaction
  needed". Wording may differ; what matters is that it lands on PATH CHOICE.
- "site_broken" — the reason is that the page is broken, unclickable, or will not
  load. That is a site failure, not "search is faster".
- "controller" — the reason mentions the controller / current objective /
  feedback, i.e. the private channel.
- "vague" — it only says "let me search", with no reason why search is right.
- "other" — none of the above; explain in `why`.

Output JSON only: {"verdict": "...", "why": "<at most 25 words>"}
"""

SYS_F2 = """
You judge the reason a web agent gave when switching from the SEARCH path (S) to
the OPERATION path (O).

Design intent of this task family: the agent should first genuinely try search and
ESTABLISH that search cannot get it (live data needed, filtering by criteria
needed, a logged-in state needed, and so on), and only then switch to acting on
the site. The "verified insufficiency" judgement is what we want to teach.

verdict:
- "search_insufficient_verified" — the reason shows search was TRIED AND FOUND
  WANTING: e.g. the query only returns a single record page and not the filtered
  list, the static page carries no live price/stock, results only appear after a
  form submission.
- "tool_unavailable" — the reason is a TOOL AVAILABILITY problem ("there is no
  visit tool on this path"), not a limitation of search itself.
- "controller" — the reason mentions the controller / current objective /
  feedback, i.e. the private channel.
- "vague" — it only says "I need to act on the site", without saying why search
  will not do.
- "other" — none of the above; explain in `why`.

Output JSON only: {"verdict": "...", "why": "<at most 25 words>"}
"""


def one(client, model, form, question, reason, think, timeout, attempts):
    sysmsg = SYS_F1 if form == "F1" else SYS_F2
    prompt = (f"## 任务\n{question}\n\n"
              f"## switch 步给出的 switch_reason\n{reason or '(空)'}\n\n"
              f"## 该步的思维链\n{think or '(空)'}")
    last = None
    for i in range(attempts):
        try:
            r = client.chat.completions.create(
                model=model, timeout=timeout, max_tokens=400, temperature=0.0,
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": prompt}])
            txt = r.choices[0].message.content or ""
            s, e = txt.find("{"), txt.rfind("}")
            out = json.loads(txt[s:e + 1])
            if not out.get("verdict"):
                raise ValueError("缺 verdict")
            return out
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:150]
            time.sleep(min(6.0, 1.5 * (2 ** i)))
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
    ap.add_argument("--input", default="runs/hybrid_rollout_v2_teacher_kimi_oss_full/input_all.jsonl")
    ap.add_argument("--exclude", default="runs/exclude_offsite_mixed.json")
    ap.add_argument("--out", default="runs/switch_rationale")
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--cot-rewrite-dir", type=Path, default=Path("runs/cot_rewrite"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("API_KEY")
    if not key:
        sys.exit("Set JUDGE_API_KEY (或 API_KEY)")

    excl = set(json.loads(Path(args.exclude).read_text())) if Path(args.exclude).is_file() else set()
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
    good -= excl

    tasks = {}
    for line in Path(args.input).open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            tasks[str(r["task_id"])] = r

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t in sorted(good):
        task = tasks.get(t)
        if not task:
            continue
        form = (task.get("private_teacher_steps") or {}).get("switch_form")
        if form not in ("F1", "F2"):
            continue
        if args.resume and (outdir / f"{t}.json").is_file():
            continue
        rp = Path(args.rollout_dir) / t / "result.json"
        if not rp.is_file():
            continue
        try:
            recs = json.loads(rp.read_text(encoding="utf-8")).get("records") or []
        except Exception:
            continue
        sw = [r for r in recs if r.get("switch_executed")]
        if not sw:
            continue
        r = sw[0]
        think = ((r.get("report_parsed") or {}).get("think") or "")
        # 优先用改写后的 think:原文里"按控制器反馈…"这类表述会把判定带偏,而那部分
        # 已经在 CoT 改写里消掉了。注意 switch_reason 字段**没有**被改写覆盖 ——
        # 它是独立字段,里面若仍提控制器,判成 controller 是对的,不是误判。
        if args.cot_rewrite_dir:
            cf_ = args.cot_rewrite_dir / f"{t}.json"
            if cf_.is_file():
                try:
                    m = json.loads(cf_.read_text(encoding="utf-8"))
                    fixed = m.get(str(r.get("step")))
                    if isinstance(fixed, dict) and fixed.get("think"):
                        think = fixed["think"]
                except Exception:
                    pass
        jobs.append((t, form, str(task.get("question") or ""),
                     str(r.get("switch_reason") or ""), think))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"待判定 {len(jobs)} 条(F1 {sum(1 for j in jobs if j[1]=='F1')} / "
          f"F2 {sum(1 for j in jobs if j[1]=='F2')})→ {outdir}", flush=True)

    client = OpenAI(base_url=args.base_url, api_key=key)
    stats = Counter()
    with cf.ThreadPoolExecutor(args.max_workers) as ex:
        futs = {ex.submit(one, client, args.model, form, q, reason, think,
                          args.timeout, args.attempts): (t, form)
                for t, form, q, reason, think in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            t, form = futs[fut]
            r = fut.result(); r["form"] = form
            (outdir / f"{t}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
            stats[(form, r.get("verdict", "失败"))] += 1
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    for k in sorted(stats, key=str):
        print(f"  {k[0]} {k[1]:32}{stats[k]}")


if __name__ == "__main__":
    main()
