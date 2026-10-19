#!/usr/bin/env python3
"""按 O 纯净度判决过滤训练样本。

只动 O 侧样本:S 侧全留(S 段的合法性不由 O 判官管)。O 段判为 operation_required
的留下,其余(static_readable / search_coverable / degenerate)剔除。

为什么不整条轨迹一起剔:训练样本本来就是按段切的(n 次 switch → n+1 条样本),
一条轨迹里 S 段干净、O 段退化时,该剔的是那个 O 段。

注意 **末段被剔会让这条轨迹失去 answer 样本**,中间段被剔则会在段序列里留下断口。
默认只做逐段过滤并把这两种情况统计出来;--drop-broken-tasks 可以把"末段被剔"的
轨迹整条丢掉,避免训练集里出现有头无尾的轨迹。
"""
from __future__ import annotations

import argparse, json, glob
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="导出的训练样本 jsonl")
    ap.add_argument("--judge", required=True, help="check_hybrid_o_purity.py 的产物目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default="operation_required",
                    help="逗号分隔的保留判决;默认只留 operation_required")
    ap.add_argument("--input", default="", help="任务表,用来取 switch_form")
    ap.add_argument("--forms", default="F2,F3,F4",
                    help="只对这些 form 应用 O 纯净度判决。F1 的 O 段按设计是 1 步 switch"
                         "(认出该走搜索),不是操作监督,判官必然判 degenerate —— "
                         "盘上可能还留着改口径之前判的 F1 判决,所以这里也要按 form 兜一道。")
    ap.add_argument("--drop-broken-tasks", action="store_true",
                    help="末段(answer 段)被剔的轨迹整条丢弃")
    ap.add_argument("--drop-impure-tasks", action="store_true",
                    help="任一受判 O 段不合格,就把这条轨迹的**全部**样本(含 S 段)丢掉。"
                         "老批 1802 条用的就是这个口径;不加则只逐段剔 O。"
                         "只看 --forms 命中的受判段,F1 的不判段不算不合格。")
    ap.add_argument("--unjudged", choices=("keep", "drop"), default="keep",
                    help="没有判决的 O 段怎么办。默认保留 —— F1 的 O 段按设计不送判"
                         "(它是 switch 监督不是操作监督),剔了就把整类样本丢光了。")
    args = ap.parse_args()

    keep_set = {x.strip() for x in args.keep.split(",") if x.strip()}
    V = {}
    for p in glob.glob(f"{args.judge}/*.json"):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if "error" in d:
            continue
        V[(d["task_id"], d["segment"])] = d["verdict"]

    forms = {}
    if args.input:
        for l in open(args.input, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                forms[d["task_id"]] = ((d.get("source") or {}).get("switch_form") or "?")
    apply_to = {x.strip() for x in args.forms.split(",") if x.strip()}

    rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    stat = Counter()

    # 整条丢弃:先扫一遍,标出"有任一受判 O 段不合格"的轨迹。
    # 判据要和逐段过滤完全一致 —— 同样先过 form 过滤,同样只看有判决的段,
    # 否则 F1 的不判段会被当成不合格,把整类样本连根拔掉。
    impure = set()
    if args.drop_impure_tasks:
        for r in rows:
            if r.get("side") != "O":
                continue
            if forms and apply_to and forms.get(r["task_id"]) not in apply_to:
                continue
            v = V.get((r["task_id"], r["segment"]))
            if v is not None and v not in keep_set:
                impure.add(r["task_id"])
        n_before = len(rows)
        rows = [r for r in rows if r["task_id"] not in impure]
        stat["因 O 段不合格整条丢弃的样本"] = n_before - len(rows)
        print(f"整条丢弃模式:{len(impure)} 条轨迹含不合格 O 段,"
              f"连带丢掉 {n_before - len(rows)} 条样本")
    keep_rows = []
    dropped_last = set()
    for r in rows:
        if r.get("side") != "O":
            stat["S 侧保留"] += 1
            keep_rows.append(r)
            continue
        if forms and apply_to and forms.get(r["task_id"]) not in apply_to:
            stat[f"O 保留(form={forms.get(r['task_id'])},不判)"] += 1
            keep_rows.append(r)
            continue
        v = V.get((r["task_id"], r["segment"]))
        if v is None:
            stat[f"O 无判决({args.unjudged})"] += 1
            if args.unjudged == "keep":
                keep_rows.append(r)
            elif r.get("ends_with") == "answer":
                dropped_last.add(r["task_id"])
            continue
        if v in keep_set:
            stat[f"O 保留 {v}"] += 1
            keep_rows.append(r)
        else:
            stat[f"O 剔除 {v}"] += 1
            if r.get("ends_with") == "answer":
                dropped_last.add(r["task_id"])

    if args.drop_broken_tasks and dropped_last:
        before = len(keep_rows)
        keep_rows = [r for r in keep_rows if r["task_id"] not in dropped_last]
        stat["因末段被剔而整条丢弃的样本"] = before - len(keep_rows)

    # 段序列断口统计:同一轨迹保留下来的 segment 号是否还连续
    seg = defaultdict(list)
    for r in keep_rows:
        seg[r["task_id"]].append(r["segment"])
    gaps = sum(1 for t, ss in seg.items()
               if sorted(ss) != list(range(min(ss), min(ss) + len(ss))))
    no_answer = sum(1 for t, ss in seg.items()
                    if not any(r.get("ends_with") == "answer"
                               for r in keep_rows if r["task_id"] == t))

    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep_rows),
        encoding="utf-8")

    print(f"输入 {len(rows)} 条 → 保留 {len(keep_rows)} 条 ({100*len(keep_rows)/max(len(rows),1):.1f}%)")
    for k, v in sorted(stat.items()):
        print(f"  {k:34}{v:6}")
    print(f"\n  轨迹 {len(seg)} 条;段号不连续的 {gaps} 条;没有 answer 段的 {no_answer} 条")
    if no_answer and not args.drop_broken_tasks:
        print("  ↑ 这些轨迹只剩中间段(全是 switch 结尾),要整条丢就加 --drop-broken-tasks")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
