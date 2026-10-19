#!/usr/bin/env python3
"""判定最终答案的关键值是否真的取自任务点名的站点。

动机:F1 的结构是"起手在指定站点、一步认出该搜、切到 S"。切走之后没有任何东西
把 S 侧拉回那个站点,于是常见的失败形态是 —— 契约点名 furniturerow.com,S 段跑去
ashleyfurniture / bedbathandbeyond 抓了一圈,最后交出一份"主流家具零售商价格基准"。
任务问的是那个站,答的是别处,**违背用户意图**;这类不能靠补写救,只能剔除。

机械判据不够用:搜索结果里出现过该域名的 URL,不等于答案里的值取自那个站。实测
708 条 F1 里只有 9% 是"S 段从未访问契约站",另有 58% 属于"访问过但答案没提",
灰色地带比黑白两端都大。所以这里逐值判来源。

判据只有一条:**answer_schema 要求的值,是不是从契约点名的站点上得到的**。
用搜索去解析标识符、找入口、确认术语,都不算跳站 —— 那是 S 为 O 提速,是设计意图。
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
from urllib.parse import urlparse

from openai import OpenAI

SYSTEM = """
You decide whether the key content of an agent's final answer was taken from the
site(s) the task named.

Background: the task names one or more sites, and the user's intent is "get the
answer ON these sites". The agent may use search to move faster (resolve an
identifier, find the entry page, confirm terminology, check a spelling), but the
DELIVERED VALUES must come from the named site(s). Going to some other website and
returning a plausible-looking answer defeats the user's intent.

You are given: the task question, the named site(s), the answer_schema, the
objective per-phase progress accounts, and the final answer.

For every field in answer_schema decide:
- from_target_site: was this field's value taken from a named site? (true/false/unclear)
- actual_source: where it actually came from (a site domain, or "search snippet" /
  "not obtained")
- note: one line of justification, quoting the progress accounts verbatim.

Then give the overall verdict:
- verdict:
    "on_site"  — the key values all come from the named site(s); incidental
                 supporting detail from search does not change this;
    "off_site" — key values came from elsewhere and were never confirmed on the
                 named site → the user's intent was not met;
    "mixed"    — some values from the named site(s), some from elsewhere;
- why: at most two sentences.

Judging notes:
- Using search to RESOLVE a value and then returning to the named site to confirm
  it counts as on_site.
- Seeing the value only in a search snippet, never confirmed on the named site,
  is off_site.
- If the named site was structurally unusable (tool retired, page 404) and the
  agent switched sources, still record off_site, but say in `why` that the site
  was unavailable — that distinction decides whether the trajectory should be
  dropped or the task should go back to the pool to be rebuilt.

Output JSON only:
{"fields": [{"field": "...", "from_target_site": true, "actual_source": "...", "note": "..."}],
 "verdict": "on_site|off_site|mixed", "why": "..."}
"""


def build_prompt(task: dict, audit: dict, result: dict, sites: list[str]) -> str:
    pts = task.get("private_teacher_steps") or {}
    schema = task.get("answer_schema") or pts.get("answer_schema") or []
    parts = [f"## 任务\n{result.get('question')}",
             f"\n## 任务点名的站点\n{json.dumps(sites, ensure_ascii=False)}",
             f"\n## answer_schema\n{json.dumps(schema, ensure_ascii=False)}",
             "\n## 各阶段客观进度"]
    for i, ph in enumerate(audit.get("phases") or []):
        prog = ph.get("progress") or {}
        parts.append(f"### 阶段 {i} ({ph.get('mode')} 侧)")
        parts.append(f"what_happened: {prog.get('what_happened')}")
        st = prog.get("sites_touched")
        if st:
            parts.append(f"sites_touched: {json.dumps(st, ensure_ascii=False)}")
        vo = prog.get("values_obtained")
        if vo:
            parts.append(f"values_obtained: {json.dumps(vo, ensure_ascii=False)}")
    parts.append(f"\n## 最终答案\n{result.get('prediction') or '(空)'}")
    return "\n".join(parts)


def one(client, model, task, audit, result, sites, timeout, attempts):
    pts = task.get("private_teacher_steps") or {}
    schema = task.get("answer_schema") or pts.get("answer_schema") or []
    if not schema or not sites:
        return {"error": "缺 answer_schema 或站点"}
    prompt = build_prompt(task, audit, result, sites)
    convo = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    last = None
    for i in range(attempts):
        try:
            r = client.chat.completions.create(model=model, messages=convo, timeout=timeout,
                                               max_tokens=2500, temperature=0.0)
            txt = r.choices[0].message.content or ""
            s, e = txt.find("{"), txt.rfind("}")
            out = json.loads(txt[s:e + 1])
            if out.get("verdict") not in ("on_site", "off_site", "mixed"):
                raise ValueError("verdict 非法")
            if len(out.get("fields") or []) != len(schema):
                raise ValueError(f"必须逐一覆盖 {len(schema)} 个字段")
            return out
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:160]
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
    ap.add_argument("--input", default="runs/hybrid_rollout_v2_teacher_kimi_oss_full/input_all.jsonl")
    ap.add_argument("--out", default="runs/site_fidelity")
    ap.add_argument("--forms", default="F1", help="逗号分隔,如 F1 或 F1,F2")
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
    forms = {x.strip() for x in args.forms.split(",") if x.strip()}

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
        task = tasks.get(t); au = audits.get(t)
        if not task or not au:
            continue
        pts = task.get("private_teacher_steps") or {}
        if pts.get("switch_form") not in forms:
            continue
        if args.resume and (outdir / f"{t}.json").is_file():
            continue
        sites = []
        for st in (pts.get("steps") or []):
            for c in (st.get("components") or []):
                for s in (c.get("sites") or []):
                    if isinstance(s, dict) and s.get("domain"):
                        sites.append(str(s["domain"]))
        su = task.get("start_url") or ""
        if su:
            sites.append(urlparse(su).netloc)
        sites = sorted({s for s in sites if s})
        rp = Path(args.rollout_dir) / t / "result.json"
        if not rp.is_file():
            continue
        try:
            res = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        jobs.append((t, task, au, res, sites))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"待判定 {len(jobs)} 条(form={sorted(forms)})→ {outdir}", flush=True)

    client = OpenAI(base_url=args.base_url, api_key=key)
    stats = Counter()
    with cf.ThreadPoolExecutor(args.max_workers) as ex:
        futs = {ex.submit(one, client, args.model, task, au, res, sites,
                          args.timeout, args.attempts): t
                for t, task, au, res, sites in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            t = futs[fut]; r = fut.result()
            (outdir / f"{t}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
            stats[r.get("verdict", "失败")] += 1
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}  {dict(stats)}", flush=True)
    print(f"\n结果: {dict(stats)}")


if __name__ == "__main__":
    main()
