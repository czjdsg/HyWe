#!/usr/bin/env python3
"""给这批带 switch 的轨迹打 loss mask,S/O 两侧都做。

## S 侧只认「机械可验证的重复」

S 侧不用判官的语义判断。原因是判官会漂移:9.6% 的 mask 理由其实是「这一步失败了/
没搜到东西」,而不是「重复了前面做过的事」—— 首发的搜索没搜到是探索失败,不是原地打转,
按判据不该 mask;另有一半「重复类」判词在机械比对下匹配不上,真假各半无从分辨。

改成两条硬规则,零误判:
    visit   本段内已经访问过完全相同的 URL(去 fragment、去尾斜杠)
    search  查询词归一化后与本段内某次查询完全相同,或词集合 Jaccard >= 0.85

0.85 这条线是看实例定的:
    1.00  "…cyberpunk 2077 pc global cd key" vs "…cyberpunk 2077 global pc cd key"   只是词序
    0.88  "site imgur com 1992 speedy 30…"   vs "imgur com 1992 speedy 30…"          只去掉 site:
    0.78  "exercism track concept exercises…" vs "exercism ruby track concept…"      加了 ruby,是真收窄
    0.67  "bairro alto hostel rooftop bar…"   vs "yes lisbon hostel rooftop bar…"    换了实体
0.70~0.85 那一档已经在杀正当的收窄,所以卡在 0.85。

## mask 的两个来源

  坐标不准(仅 O 侧带坐标的步)—— judge_coord_accuracy.py
      看画了落点的截图 + CoT + 动作,判动作有没有忠实执行 CoT 说的那件事。
      miss:coord   落点压错了元素
      miss:action  动作类型/参数与 CoT 不符
      这批的 CoT 是 rollout 时模型自己写的(不是按坐标反推生成的),
      所以拿 CoT 当参照没有循环问题。

  明显空转(仅 O 侧)—— judge_spin_hybrid.py
      看画面变没变。宽口径:探索一律保留,只剔「重复了前面做过的事且没带来任何新东西」。
      S 侧改用上面的机械规则,不走判官。

## 缺判一律按 keep

两个判官的口径都是"拿不准就 keep",缺判等价于没判准,给 keep 保持一致。
缺判量会打印出来,不让它悄悄溜过去。

## 收口步(answer / switch)

不 mask。answer 是这一段的结论,switch 是路径切换的决策 —— 两者都是要学的行为,
不在"点得准不准""是不是空转"的判据范围内。
"""
from __future__ import annotations
import argparse, collections, json, re

JAC = 0.85          # search 查询词集合重合到这个程度就算重复,依据见文件头


def norm_q(a):
    q = a.get("query")
    qs = [q] if isinstance(q, str) else [str(x) for x in (q or [])]
    return [re.sub(r"\W+", " ", x.lower()).strip() for x in qs if x]


def norm_u(a):
    u = a.get("url")
    us = [u] if isinstance(u, str) else [str(x) for x in (u or [])]
    out = []
    for x in us:
        x = x.strip().strip('"').split("#")[0].rstrip("/")
        if x: out.append(x)
    return out

TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
AN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def mt(m):
    c = m.get("content")
    if isinstance(c, list):
        return "".join(z.get("text", "") for z in c
                       if isinstance(z, dict) and z.get("type") == "text")
    return str(c or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/train_hybrid_all.jsonl")
    ap.add_argument("--coord", default="runs/hybrid_coord_verdicts.jsonl")
    ap.add_argument("--spin", default="runs/hybrid_spin.jsonl")
    ap.add_argument("--out", default="runs/train_hybrid_all.masked.jsonl")
    a = ap.parse_args()

    C = {}
    for l in open(a.coord):
        try: x = json.loads(l)
        except Exception: continue
        if x.get("v") == "err" and x["key"] in C: continue
        C[x["key"]] = x
    S = {}
    for l in open(a.spin):
        try: x = json.loads(l)
        except Exception: continue
        p = S.get(x["si"])
        # 续跑会写多行,取判得最全的那次
        if p and len(p.get("verdicts") or {}) > len(x.get("verdicts") or {}): continue
        S[x["si"]] = x
    print(f"载入 坐标判读 {len(C):,} 步 · 空转判读 {len(S):,} 条样本")

    st = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as fo:
        for si, l in enumerate(open(a.src)):
            d = json.loads(l)
            side = d.get("side")
            sv = (S.get(si) or {}).get("verdicts") or {}
            has_spin = si in S
            seen_q, seen_u = [], set()      # S 侧机械查重用,按段独立
            ai = -1
            for m in d["messages"]:
                if m.get("role") != "assistant": continue
                ai += 1
                t = mt(m)
                x = TC.search(t)
                name = ""
                if x:
                    try: name = (json.loads(x.group(1)) or {}).get("name") or ""
                    except Exception: name = ""
                if AN.search(t) or name == "switch":
                    m["loss"] = 1
                    st[f"{side}:收口/切换:keep"] += 1
                    continue
                cv = C.get(f"{si}_{ai}")
                coord_bad = bool(cv and (cv.get("v") or "").startswith("miss"))
                ck = cv.get("k") if cv else ""
                if side == "S":
                    # 机械查重:只认完全相同的 URL、以及归一化后相同或高度重合的查询词
                    args = {}
                    if x:
                        try: args = (json.loads(x.group(1)) or {}).get("arguments") or {}
                        except Exception: args = {}
                    spin_bad = False
                    if name == "search":
                        qs = norm_q(args)
                        for q in qs:
                            w = set(q.split())
                            for p2 in seen_q:
                                w2 = set(p2.split())
                                if len(w & w2) / max(len(w | w2), 1) >= JAC:
                                    spin_bad = True; break
                            if spin_bad: break
                        seen_q.extend(qs)
                    elif name == "visit":
                        us = norm_u(args)
                        if us and all(u in seen_u for u in us): spin_bad = True
                        seen_u.update(us)
                else:
                    spin_bad = (sv.get(str(ai)) or {}).get("v") == "spin"
                if side == "O":
                    if not has_spin: st[f"{side}:(附注)整条缺空转判读"] += 1
                    elif str(ai) not in sv: st[f"{side}:(附注)缺空转判按keep"] += 1
                if coord_bad and spin_bad: st[f"{side}:坐标+空转"] += 1
                elif coord_bad: st[f"{side}:仅{'坐标歪' if ck == 'coord' else '动作不符'}"] += 1
                elif spin_bad: st[f"{side}:仅{'机械重复' if side == 'S' else '空转'}"] += 1
                else: st[f"{side}:keep"] += 1
                m["loss"] = 0 if (coord_bad or spin_bad) else 1
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n写出 {a.out}\n")
    for side in ("O", "S"):
        ks = [k for k in st if k.startswith(side + ":")]
        # 「(附注)」开头的是同一步的旁注,不能计入合计,否则那些步被算两遍
        tot = sum(st[k] for k in ks if not k.startswith(side + ":(附注)"))
        if not tot: continue
        mask = sum(st[k] for k in ks if "keep" not in k and not k.startswith(side + ":(附注)"))
        print(f"  == {side} 侧 {tot:,} 步 · mask {mask:,} ({100*mask/tot:.1f}%) ==")
        for k in sorted(ks, key=lambda x: -st[x]):
            print(f"      {k.split(':', 1)[1]:<20} {st[k]:>7,}")
        print()


if __name__ == "__main__":
    main()
