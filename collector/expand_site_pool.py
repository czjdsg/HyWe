#!/usr/bin/env python3
"""扩充 hybrid 站点池:把纯 O 那条线实测可达的新域名归类后并入。

为什么需要:v2_8600 那批 8451 道任务覆盖了池内**全部 1170 个域名** —— 池子已经
抽干了,再换种子也只是重排组合,不会带来新站点。

新域名来自 runs/op_tasks_v5/tasks_reachable.jsonl(molmoweb + webchain 的真实任务),
且经过 probe_site_reachability.py 用生产同款浏览器配置实测过,只收 verdict=ok 的。
这比凭空列域名可靠 —— 它们是真打开过、真跑过任务的。

设计管线实际只读 5 个字段(load_sites 的白名单):
    domain / category / entry_url / image_required / suggested_switch_kind
外加 capability_hint(来自 switch_trigger_observation 或 evidence,喂给自然化模型)。
**category 是硬约束**:compatible_operations 按它匹配 O 原型,填错整个任务就配歪。
所以 category 交给模型判,并把该域名真实跑过的任务和原型作为依据一起给它。
score / s_can_cover / url_constructible / probe / provenance 压根没被读,按来源如实填。
"""
from __future__ import annotations

import argparse, json, os, sys, time
import concurrent.futures as cf
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI

POOL = Path("benchmarks/data/hybrid_pool/hybrid_pool_combined.json")

SYSTEM = """You assign a website to exactly one category from a fixed taxonomy.

You are given the domain, its entry URL, and a few REAL tasks that agents actually
ran on that site (with the interaction prototype each task used). Use those tasks
as the primary evidence for what the site is for — they describe observed
behaviour, not marketing copy.

Rules:
- Pick exactly one id from the taxonomy. Never invent a new one.
- Judge by what a user DOES on the site, not by the company's industry. A bank's
  mortgage calculator page is a calculator; an airline's booking widget is
  booking-travel-air.
- The "-v2" variants are the same domain of activity as their base name; prefer
  the base name unless the tasks clearly match the v2 description you are shown.
- If several fit, pick the one matching the MOST FREQUENT prototype in the tasks.

Also decide:
- image_required: does acting on this site require seeing the rendered page
  (visual layout, canvas, map, media)? true/false
- suggested_switch_kind: "S_then_O" if an agent would typically research first and
  then act on the site; "O_then_S" if it would start on the site and then need
  search to finish; "" if neither is clearly better.
- capability_hint: one sentence, in English, on what interaction this site affords
  — the thing an agent must DO here that a search engine cannot do for it.

Output JSON only:
{"category": "<taxonomy id>", "image_required": true,
 "suggested_switch_kind": "S_then_O|O_then_S|",
 "capability_hint": "...", "why": "<one line>"}"""


def dom_of(u: str) -> str:
    try:
        h = urlparse(u if "://" in u else "http://" + u).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def one(cli, model, d, taxonomy, timeout, attempts):
    tasks = "\n".join(f"- [{p}] {q[:180]}" for q, p in d["samples"][:6])
    prompt = (f"## Taxonomy (pick exactly one id)\n{taxonomy}\n"
              f"\n## Site\ndomain: {d['domain']}\nentry_url: {d['entry_url']}\n"
              f"\n## Real tasks run on this site ({d['n']} total)\n{tasks}\n"
              f"\n## Prototype distribution\n"
              + ", ".join(f"{k} x{v}" for k, v in d["protos"].most_common(5)))
    msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    valid = {x.split(" —")[0].strip() for x in taxonomy.split("\n") if x.strip()}
    err = "?"
    for _ in range(attempts):
        try:
            r = cli.chat.completions.create(model=model, messages=msg, timeout=timeout,
                                            temperature=0, response_format={"type": "json_object"})
            o = json.loads(r.choices[0].message.content)
            cat = str(o.get("category") or "").strip()
            if cat not in valid:
                err = f"category {cat!r} 不在分类表里"
                continue
            return {"domain": d["domain"], "entry_url": d["entry_url"], "category": cat,
                    "image_required": bool(o.get("image_required")),
                    "suggested_switch_kind": str(o.get("suggested_switch_kind") or ""),
                    "switch_trigger_observation": str(o.get("capability_hint") or "")[:600],
                    "evidence": str(o.get("why") or "")[:300],
                    "score": 4,
                    "s_can_cover": "no",
                    "url_constructible": False,
                    "probe": "reachable (probe_site_reachability, verdict=ok)",
                    "provenance": "op_tasks_v5 reachable + expand_site_pool",
                    "weight": 1}
        except Exception as e:
            err = str(e)[:110]
    return {"domain": d["domain"], "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="runs/op_tasks_v5/tasks_reachable.jsonl")
    ap.add_argument("--reach", default="runs/site_reachability.json")
    ap.add_argument("--out", default="benchmarks/data/hybrid_pool/hybrid_pool_expanded.json")
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    raw = json.loads(POOL.read_text(encoding="utf-8"))
    pool = raw["all_usable"]
    cur = {str(x.get("domain", "")).split("/")[0].lower().lstrip("www.") for x in pool}
    cats = sorted({x["category"] for x in pool if x.get("category")})
    # 分类表带一个真实例站,让模型知道每类长什么样
    sample_of = {}
    for x in pool:
        sample_of.setdefault(x.get("category"), x.get("domain"))
    taxonomy = "\n".join(f"{c} — e.g. {sample_of.get(c,'')}" for c in cats)

    reach = {}
    if os.path.isfile(a.reach):
        for r in json.loads(Path(a.reach).read_text(encoding="utf-8")):
            h = dom_of(r.get("url", ""))
            if h and h not in reach:
                reach[h] = r.get("verdict")

    cand = {}
    for l in open(a.tasks, encoding="utf-8"):
        if not l.strip():
            continue
        x = json.loads(l)
        h = dom_of(x.get("web") or "")
        if not h or h in cur:
            continue
        if reach.get(h) != "ok":          # 只收实测可达的
            continue
        sp = x.get("operation_spec") or {}
        c = cand.setdefault(h, {"domain": h, "entry_url": x.get("web") or "",
                                "n": 0, "protos": Counter(), "samples": []})
        c["n"] += 1
        c["protos"][sp.get("prototype")] += 1
        if len(c["samples"]) < 6:
            c["samples"].append((x.get("question") or "", sp.get("prototype") or ""))
    items = sorted(cand.values(), key=lambda x: -x["n"])
    if a.limit:
        items = items[:a.limit]
    print(f"现有池 {len(pool)} 条 / {len(cur)} 域名,分类 {len(cats)} 种", flush=True)
    print(f"候选新域名 {len(items)} 个(仅 verdict=ok),模型 {a.model}", flush=True)

    cli = OpenAI(base_url=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
                 api_key=os.environ.get("JUDGE_API_KEY") or os.environ["API_KEY"])
    out, bad = [], []
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.max_workers) as ex:
        futs = [ex.submit(one, cli, a.model, d, taxonomy, a.timeout, a.attempts) for d in items]
        for n, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            (bad if "error" in r else out).append(r)
            if n % 40 == 0:
                print(f"  {n}/{len(items)}  成功 {len(out)} 失败 {len(bad)}  "
                      f"{(time.time()-t0)/n:.1f}s/个", flush=True)
    print(f"\n归类成功 {len(out)},失败 {len(bad)}")
    print("新增 category 分布:", dict(Counter(x["category"] for x in out).most_common(10)))
    merged = pool + out
    Path(a.out).write_text(json.dumps({"all_usable": merged, "total": len(merged)},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n扩充后 {len(merged)} 条 → {a.out}")
    print("  (原池未改动;要启用请把设计管线的 --site-pool 指向新文件)")


if __name__ == "__main__":
    main()
