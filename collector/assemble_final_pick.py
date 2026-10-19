#!/usr/bin/env python3
"""把各阶段判决汇总成最终合格集合,并打印端到端漏斗。

漏斗口径照搬 TRAJECTORY_POSTPROCESSING.md §0(历史两轮 29.4% / 30.4%),
本版在末尾**新增 O 侧纯净度**这一道 —— 之前审 1802 条时发现 O 段只有 49.7%
是真操作(其余是退化 / 静态可读 / 搜索可覆盖),那部分水以前一直混在训练集里。

每一道的判据都写在下面的 keep_* 函数里,便于单独调口径。
"""
from __future__ import annotations

import argparse, json, glob, os, shutil
from collections import Counter, OrderedDict
from pathlib import Path


def load_dir(d: str, name: str = None) -> dict:
    """读 <dir>/<task_id>.json 或 <dir>/<task_id>/<name>。"""
    out = {}
    if not os.path.isdir(d):
        return out
    if name:
        for p in glob.glob(os.path.join(d, "*", name)):
            try: out[os.path.basename(os.path.dirname(p))] = json.load(open(p, encoding="utf-8"))
            except Exception: pass
    else:
        for p in glob.glob(os.path.join(d, "*.json")):
            try: out[os.path.basename(p)[:-5]] = json.load(open(p, encoding="utf-8"))
            except Exception: pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2_8600")
    ap.add_argument("--rollout", default=None)
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", default="runs/bench_taskids.json",

                    help="排除清单(json 数组)。评测集必须从训练产物里剔除,"

                         "否则每轮 assemble 会按 survivors 把它们拷回 final_pick。")
    ap.add_argument("--copy", action="store_true", help="把幸存者拷进 final_pick 目录")
    a = ap.parse_args()
    T = a.tag
    rollout = a.rollout or f"runs/hybrid_rollout_{T}/rollouts"
    inp = a.input or f"runs/hybrid_task_design_{T}/input_{T}.jsonl"
    out = a.out or f"runs/final_pick_{T}/rollouts"

    forms = {}
    for l in open(inp, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            forms[d["task_id"]] = ((d.get("source") or {}).get("switch_form") or "?")

    audit = load_dir(f"runs/unified_audit_{T}/tasks", "audit.json")
    site = load_dir(f"runs/site_fidelity_{T}")
    switch = load_dir(f"runs/switch_rationale_{T}")
    cover = load_dir(f"runs/answer_coverage_{T}")
    merged = load_dir(f"runs/answer_merged_{T}")
    purity = load_dir(f"runs/o_purity_{T}")

    # 起点:termination=answer 的轨迹
    answers = []
    for p in glob.glob(os.path.join(rollout, "*", "result.json")):
        try:
            if json.load(open(p, encoding="utf-8")).get("termination") == "answer":
                answers.append(os.path.basename(os.path.dirname(p)))
        except Exception:
            pass
    answers.sort()

    steps = OrderedDict()
    cur = list(answers)
    steps["answer 轨迹(全部)"] = len(cur)
    # 审计是快照式的:启动之后新完成的 answer 轨迹没被判到。
    # 不先收窄分母的话,这些"没判过"的会被算成"没过双达标",把漏斗第一道压低。
    if audit:
        cur = [t for t in cur if t in audit]
        steps["其中已审计(漏斗分母)"] = len(cur)

    # ① v2 中性口径双达标:task_success 且 answer.reasonable
    _excl = set()
    try:
        import json as _j, os as _o
        if a.exclude and _o.path.isfile(a.exclude):
            _excl = set(_j.load(open(a.exclude, encoding="utf-8")))
    except Exception:
        _excl = set()

    def dual_ok(t):
        """双达标 = 检查点全满足 ∧ 整条轨迹完成了任务。

        2026-09-02:第二项从「答案文本是否合格」改成「轨迹是否完成任务」。
        答案文本的完备性在下面「答案没解决问题」那道由 check_answer_coverage
        逐字段判 —— 放两处是重复评判,而且会把「活干完了但答案写得潦草」和
        「答案漂亮但任务没做完」混成同一个布尔。旧产物用 answer.reasonable,
        这里两个键都认。
        """
        j = (audit.get(t) or {}).get("judgement") or {}
        tj = j.get("trajectory") or j.get("answer") or {}
        ok2 = tj.get("succeeded", tj.get("reasonable"))
        return bool(j.get("task_success")) and bool(ok2)
    cur = [t for t in cur if t in audit and dual_ok(t)]
    steps["检查点全满足 ∧ 轨迹完成任务"] = len(cur)

    # ② 站点跳出(只卡 F1:契约点名了站点,答案的值却取自别处)
    cur = [t for t in cur if not (forms.get(t) == "F1"
                                  and (site.get(t) or {}).get("verdict") == "off_site")]
    steps["− 站点跳出(F1)"] = len(cur)

    # ③ switch 理由不合格(F1/F2):理由必须是"认出该换路径",
    #    不能是站点坏了/工具没了/控制器让我切/含糊
    BAD = {"site_broken", "tool_unavailable", "controller", "vague"}
    cur = [t for t in cur if not (forms.get(t) in ("F1", "F2")
                                  and (switch.get(t) or {}).get("verdict") in BAD)]
    steps["− switch 理由不合格"] = len(cur)

    # ④ 答案没解决问题:有字段在全轨迹里都拿不到
    cur = [t for t in cur if not ((cover.get(t) or {}).get("missing_everywhere"))]
    steps["− 答案没解决问题"] = len(cur)

    # ⑤ 补写失败:字段在更早阶段拿到过但没写进答案,且补写没成功
    def merge_ok(t):
        need = (cover.get(t) or {}).get("missing_but_available")
        if not need:
            return True
        return bool((merged.get(t) or {}).get("answer"))
    cur = [t for t in cur if merge_ok(t)]
    steps["− 补写失败"] = len(cur)

    # ⑥ O 侧纯净度(本版新增):这条轨迹至少要有一个 O 段是真操作。
    #    没有 O 段的(纯 S 路线)不卡这一道。
    def o_ok(t):
        # F1 不参与:它的 O 段按设计就是 1 步 switch(认出该走搜索),
        # 用「有没有做 O 独有的事」判必然 degenerate,但样本价值在 switch 决策。
        if forms.get(t) == "F1":
            return True
        segs = [v for k, v in purity.items() if v.get("task_id") == t]
        if not segs:
            return True          # 没判到 / 没有 O 段
        return any(s.get("verdict") == "operation_required" for s in segs)
    if purity:
        cur = [t for t in cur if o_ok(t)]
        steps["− O 侧无真操作(新增,F1 除外)"] = len(cur)

    # 百分比以"已审计"为分母(没有审计时退回全部 answer)
    n0 = max(steps.get("其中已审计(漏斗分母)") or steps["answer 轨迹(全部)"], 1)
    print(f"{'阶段':26}{'剩余':>7}{'占 answer':>11}")
    for k, v in steps.items():
        print(f"  {k:24}{v:7}{100*v/n0:10.1f}%")
    print(f"\n端到端合格率 {100*len(cur)/n0:.1f}%  (历史两轮 29.4% / 30.4%,但那两轮没有 O 纯净度这道)")
    print("\nform 分布:", dict(Counter(forms.get(t, "?") for t in cur).most_common()))
    miss = [k for k, d in [("audit", audit), ("site", site), ("switch", switch),
                           ("coverage", cover), ("purity", purity)] if not d]
    if miss:
        print(f"\n注意:这些阶段还没有产物,对应的过滤被跳过了 -> {miss}")

    # 评测集必须在写 survivors 之前排除:survivors 是 --copy 的依据,
    # 名单里留着就会每轮把它们拷回 final_pick,事后搬文件堵不住(踩过)。
    if _excl:
        _b = [t for t in cur if t in _excl]
        if _b:
            cur = [t for t in cur if t not in _excl]
            print(f"\n排除评测集 {len(_b)} 条(--exclude)")

    Path(f"runs/final_pick_{T}").mkdir(parents=True, exist_ok=True)
    Path(f"runs/final_pick_{T}/survivors.json").write_text(
        json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n幸存 {len(cur)} 条 -> runs/final_pick_{T}/survivors.json")
    if a.copy:
        Path(out).mkdir(parents=True, exist_ok=True)
        n = 0
        for t in cur:
            src, dst = os.path.join(rollout, t), os.path.join(out, t)
            if not os.path.isdir(src):
                continue
            # 幂等 + 容忍竞态:后处理循环和手动重跑可能同时在拷同一个目录,
            # 先 exists 再 copytree 会撞 FileExistsError 把整轮打断。
            try:
                if not os.path.exists(dst):
                    shutil.copytree(src, dst, dirs_exist_ok=True); n += 1
            except FileExistsError:
                pass
        print(f"已拷 {n} 条到 {out}")


if __name__ == "__main__":
    main()
