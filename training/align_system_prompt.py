#!/usr/bin/env python3
"""把训练数据的 system 提示词重建成**推理端此刻会发出的那一份**,逐字一致。

## 为什么要重建

推理端的配方(inference/hybrid_agent.py:4076)是:

    system_prompt(active_mode, current_date=…, expose_switch=True)
      + "\\n\\n# Tool Schemas\\n" + _v2_tool_schema_text()

三份数据实测下来:
    hybrid            400/400 逐字一致          —— 按当前提示词造的
    纯 search(Aquila) 全部不一致,少 93 字
    纯 operation(v8)  全部不一致,少 604 字

后两份是按**旧版** react_v2_prompts.py 造的,之后提示词改过(多了
"It takes no arguments — the reason belongs in `<think>`" 之类的句子)。
该文件在 git 里是未跟踪状态,拿不到旧版,也不需要 —— 要对齐的是**当前**推理端。

## 只动 system,别的一律不碰

user / assistant 的格式三份本来就一致(`<think>` + `<tool_call>` / `<answer>`,
观测走 `<tool_response>`),不需要改。改多了反而引入新的不一致。

## 两个保留项

  * **日期行**按样本原样保留。system_prompt 的 current_date 在推理时由 args 注入,
    训练数据里那个日期是任务的时间上下文(纯 search 是 2026-06-27~29),
    重建时必须原样带回去,否则任务里"今天"的指代就断了。
  * **expose_switch 恒为 True**,因为推理端写死是 True。两份旧数据实测也都暴露了
    switch,所以这一项本来就没有分歧。
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))
from react_v2_prompts import system_prompt                      # noqa: E402
from hybrid_prompts import get_allowed_tool_names, TOOLS, SWITCH_TOOL   # noqa: E402

DATE = re.compile(r"\nToday is: ([^\n]+)\n")


def schema_text(side: str) -> str:
    """与 hybrid_agent._v2_tool_schema_text 同构:同样的取舍、同样的 json.dumps 参数。"""
    allowed = get_allowed_tool_names("hybrid_react_v2", side)
    out = []
    for t in TOOLS + [SWITCH_TOOL]:
        fn = t.get("function") or {}
        if fn.get("name") in allowed:
            out.append({"name": fn.get("name"), "description": fn.get("description"),
                        "parameters": fn.get("parameters")})
    return json.dumps(out, ensure_ascii=False, indent=2)


SCH = {s: schema_text(s) for s in ("S", "O")}


def build(side: str, date: str) -> str:
    return (system_prompt(side, current_date=date, expose_switch=True)
            + "\n\n# Tool Schemas\n" + SCH[side])


def text_of(m):
    c = m.get("content")
    if isinstance(c, list):
        return "".join(z.get("text", "") for z in c
                       if isinstance(z, dict) and z.get("type") == "text")
    return str(c or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", default="auto",
                    help="S / O / auto(读每条的 side 字段,hybrid 用这个)")
    # 采集时的日期在 rollout 的 system_prompt 里,导出器默认没带出来(--current-date
    # 默认空串),于是训练样本里那行整条消失 —— 而推理端的日期行是无条件生成的。
    # 传一个 {task_id: "YYYY-MM-DD"} 映射就能把采集当天的日期补回去,外部映射优先。
    ap.add_argument("--dates", default="",
                    help="task_id -> 采集日期 的 JSON 映射;优先于样本内抠到的日期")
    a = ap.parse_args()

    st = collections.Counter()
    DATES = json.load(open(a.dates, encoding="utf-8")) if a.dates else {}
    if DATES:
        print(f"外部日期映射 {len(DATES)} 条")
    with open(a.out, "w", encoding="utf-8") as fo:
        for li, l in enumerate(open(a.src)):
            d = json.loads(l)
            ms = d.get("messages") or []
            if not ms or ms[0].get("role") != "system":
                st["无首条system(跳过)"] += 1
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
                continue
            side = a.side if a.side in ("S", "O") else (d.get("side") or "S")
            old = text_of(ms[0])
            dm = DATE.search(old)
            date = DATES.get(d.get("task_id") or "") or (dm.group(1) if dm else "")
            if DATES and not DATES.get(d.get("task_id") or ""):
                st["无采集日期"] += 1
            new = build(side, date)
            st["已一致" if new == old else "重建"] += 1
            st[f"side={side}"] += 1
            c = ms[0].get("content")
            if isinstance(c, list):
                done = False
                for z in c:
                    if isinstance(z, dict) and z.get("type") == "text" and not done:
                        z["text"] = new; done = True
                if not done:
                    c.append({"type": "text", "text": new})
            else:
                ms[0]["content"] = new
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            if (li + 1) % 2000 == 0:
                print(f"  {li+1:,}", file=sys.stderr, flush=True)
    print(f"\n写出 {a.out}")
    for k, v in st.most_common():
        print(f"  {k:<18} {v:>8,}")


if __name__ == "__main__":
    main()
