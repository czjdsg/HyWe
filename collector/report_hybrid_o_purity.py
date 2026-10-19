#!/usr/bin/env python3
"""汇总 O 侧纯净度审计,并产出可用/该剔的段清单。

口径说明:
- **按段算**是主口径:训练样本本来就是按段切的(n 个 switch → n+1 个样本),
  一条轨迹里 S 段干净、O 段退化,该剔的是那个 O 段而不是整条轨迹。
- **按轨迹算**另报一份:一条轨迹只要有一个 O 段是真操作,就算这条轨迹的 O 侧站得住。
"""
from __future__ import annotations

import argparse, json, glob, os
from collections import Counter, defaultdict
from pathlib import Path

ORDER = ["operation_required", "static_readable", "search_coverable", "degenerate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="runs/hybrid_o_purity")
    ap.add_argument("--train", default="runs/train_hybrid_all.jsonl")
    ap.add_argument("--out", default="runs/hybrid_o_purity_report.json")
    args = ap.parse_args()

    V = []
    for p in glob.glob(f"{args.judge}/*.json"):
        try:
            V.append(json.load(open(p, encoding="utf-8")))
        except Exception:
            pass
    ok = [v for v in V if "error" not in v]
    n = len(ok)
    rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    O_all = [r for r in rows if r.get("side") == "O"]
    S_all = [r for r in rows if r.get("side") == "S"]

    print(f"O 段总数 {len(O_all)} | 已判 {len(V)} | 有效 {n} | 失败 {len(V)-n}\n")
    c = Counter(v["verdict"] for v in ok)
    print("=== 按段(主口径)===")
    for k in ORDER:
        print(f"  {k:20}{c[k]:5}  {100*c[k]/max(n,1):5.1f}%")
    pure = c["operation_required"]
    print(f"\n  → O 侧真正纯净:{pure}/{n} = {100*pure/max(n,1):.1f}%")

    # 按轨迹
    bytask = defaultdict(list)
    for v in ok:
        bytask[v["task_id"]].append(v["verdict"])
    t_pure = sum(1 for t, vs in bytask.items() if "operation_required" in vs)
    print(f"\n=== 按轨迹 ===")
    print(f"  有判决的轨迹 {len(bytask)} 条")
    print(f"  至少一个 O 段是真操作:{t_pure} ({100*t_pure/max(len(bytask),1):.1f}%)")
    print(f"  O 段全部不合格:      {len(bytask)-t_pure} ({100*(len(bytask)-t_pure)/max(len(bytask),1):.1f}%)")

    print("\n=== 按段长 ===")
    for lo, hi, name in [(1, 1, "1步"), (2, 4, "2-4步"), (5, 10, "5-10步"),
                         (11, 20, "11-20步"), (21, 9999, "21步+")]:
        g = [v for v in ok if lo <= v.get("seg_steps", 0) <= hi]
        if not g:
            continue
        cc = Counter(v["verdict"] for v in g)
        print(f"  {name:8} n={len(g):5}  真操作 {100*cc['operation_required']/len(g):5.1f}%"
              f"  静态 {100*cc['static_readable']/len(g):5.1f}%"
              f"  可搜 {100*cc['search_coverable']/len(g):5.1f}%"
              f"  退化 {100*cc['degenerate']/len(g):5.1f}%")

    print("\n=== 真操作段的原型分布 ===")
    for k, v in Counter(x.get("prototype") for x in ok
                        if x["verdict"] == "operation_required").most_common():
        print(f"  {v:5}  {k}")

    print("\n=== 结束方式 ===")
    ew = {(r["task_id"], r["segment"]): r.get("ends_with") for r in O_all}
    for k in ORDER:
        g = [v for v in ok if v["verdict"] == k]
        cc = Counter(ew.get((v["task_id"], v["segment"])) for v in g)
        if g:
            print(f"  {k:20} {dict(cc)}")

    keep = sorted(f"{v['task_id']}#{v['segment']}" for v in ok
                  if v["verdict"] == "operation_required")
    drop = sorted(f"{v['task_id']}#{v['segment']}" for v in ok
                  if v["verdict"] != "operation_required")
    rep = {"n_o_segments": len(O_all), "n_judged": n,
           "by_verdict": {k: c[k] for k in ORDER},
           "purity_by_segment": round(100 * pure / max(n, 1), 2),
           "n_trajectories": len(bytask),
           "trajectories_with_pure_o": t_pure,
           "keep_segments": keep, "drop_segments": drop}
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n清单 -> {args.out}  (保留 {len(keep)} / 剔除 {len(drop)})")

    # 若全部剔除 O 段,训练集还剩多少
    print(f"\n=== 对训练集的影响 ===")
    print(f"  现有样本 {len(rows)}(S {len(S_all)} + O {len(O_all)})")
    if not rows:
        # 新起的一路在攒够第一批合格轨迹之前,训练集就是空的 —— 这不是错误,
        # 但除零会让循环日志里每轮都吐一段 traceback。
        print("  (训练集为空,跳过占比)")
        return
    print(f"  只保留真操作 O 段:{len(S_all)} + {pure} = {len(S_all)+pure} 条"
          f"({100*(len(S_all)+pure)/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
