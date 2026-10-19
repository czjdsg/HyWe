#!/usr/bin/env python3
"""对 swift 多轮训练文件打「机械重复」mask —— 不用判官,零误判。

## 为什么不用 LLM 判官

在 hybrid 的 S 侧试过语义判官,它会漂移:9.6% 的判词理由其实是「这一步失败了/
没搜到东西」而不是「重复了」;另有约一半的「重复类」判词在机械比对下匹配不上,
真假各半无从分辨。搜索侧「换个说法再搜」本身是正当技能,判错的代价是教出过早放弃,
所以这里只认硬证据。

## 两条规则

    visit   本样本内已访问过完全相同的 URL(去 fragment、去尾斜杠)
    search  查询词归一化后与本样本内某次查询完全相同,或词集合 Jaccard >= --jaccard

## 阈值 0.85 是看实例定的

    1.00  "dastagird khosrow shad kavadh"      / "khosrow shad kavadh dastagird"     纯词序
    0.86  "kingdom destroyed by huns 454 455"  / "…huns 453 454 455"                 只加了个年份
    0.78  "carl von ossietzky university oldenburg renamed 1991"
          "…oldenburg founded 1973 renamed 1991"                                     加了新线索,是真收窄
0.70~0.85 那一档已经在杀正当的收窄,所以卡在 0.85。降到 0.70 会多 mask 一倍以上。

## 作用域是「样本」不是「轨迹」

128k 水位线折叠会把一条长轨迹切成多条样本,跨切点的重复看不见。
文件里没有 traj_id,也就无从跨样本聚合 —— 这是已知的覆盖缺口,不是 bug。

## 收口步不 mask

answer / switch 是要学的行为,不在重复的判据范围内。
"""
from __future__ import annotations
import argparse, collections, json, re, sys

TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
AN = re.compile(r"<answer>", re.S)


def mt(m):
    c = m.get("content")
    if isinstance(c, list):
        return "".join(z.get("text", "") for z in c
                       if isinstance(z, dict) and z.get("type") == "text")
    return str(c or "")


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
        if x:
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jaccard", type=float, default=0.85)
    ap.add_argument("--keep-existing-mask", action="store_true",
                    help="已有 loss=0 的步保持屏蔽(默认就是这样,只加不减)")
    a = ap.parse_args()

    st = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as fo:
        for li, l in enumerate(open(a.src)):
            d = json.loads(l)
            seen_q, seen_u = [], set()
            for m in d.get("messages") or []:
                if m.get("role") != "assistant":
                    continue
                t = mt(m)
                x = TC.search(t)
                name = ""
                if x:
                    try:
                        name = (json.loads(x.group(1)) or {}).get("name") or ""
                    except Exception:
                        name = ""
                if AN.search(t) or name in ("answer", "switch"):
                    st["收口/切换:keep"] += 1
                    if m.get("loss") is None:
                        m["loss"] = 1
                    continue
                args = {}
                if x:
                    try:
                        args = (json.loads(x.group(1)) or {}).get("arguments") or {}
                    except Exception:
                        args = {}
                rep = False
                if name == "search":
                    qs = norm_q(args)
                    for q in qs:
                        w = set(q.split())
                        for p2 in seen_q:
                            w2 = set(p2.split())
                            if len(w & w2) / max(len(w | w2), 1) >= a.jaccard:
                                rep = True
                                break
                        if rep:
                            break
                    seen_q.extend(qs)
                elif name == "visit":
                    us = norm_u(args)
                    if us and all(u in seen_u for u in us):
                        rep = True
                    seen_u.update(us)
                old = m.get("loss", 1)
                # 只加不减:原本就屏蔽的保持屏蔽
                m["loss"] = 0 if (rep or old == 0) else 1
                if rep:
                    st[f"重复:{name or '?'}"] += 1
                    if old == 0:
                        st["(其中)原本已屏蔽"] += 1
                elif old == 0:
                    st["沿用原有屏蔽"] += 1
                else:
                    st["keep"] += 1
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            if (li + 1) % 2000 == 0:
                print(f"  {li+1:,}", file=sys.stderr, flush=True)

    print(f"\n写出 {a.out}")
    rep = sum(v for k, v in st.items() if k.startswith("重复:"))
    tot = sum(v for k, v in st.items() if not k.startswith("(其中)"))
    for k, v in st.most_common():
        print(f"  {k:<18} {v:>9,}")
    print(f"\n  新增重复屏蔽 {rep:,} / 总步 {tot:,} = {100*rep/max(tot,1):.1f}%")


if __name__ == "__main__":
    main()
