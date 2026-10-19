#!/usr/bin/env python3
"""审 hybrid 轨迹的 O 侧段:有多少是真正需要浏览器操作的。

背景:hybrid 采集时相位控制器把任务切成 S/O 交替的段,但"进了 O 相位"不等于
"这段真的在做 O 独有的事" —— 教师可能只是 goto 一个静态页读一行字,那种活
搜索侧就能干,留在训练集里会教模型在不需要操作的时候去操作。

判据用纯 O 那条线的 12 个原型 + 三类否定(静态可读 / 搜索可覆盖 / 退化)。

证据口径(纯 O 那轮踩过的坑):
  截图      地面真值
  <think>   教师自述,可能编造,只当线索
observation 字段在 O 侧只有动作执行结果(status/tool/url),不含页面内容,
所以**必须给截图**,否则判官等于瞎判 —— 那一轮 v1 判官的 phi 是 -0.039。
"""
from __future__ import annotations

import argparse, base64, io, json, os, re, sys, time
import concurrent.futures as cf
from collections import Counter
from pathlib import Path

from openai import OpenAI
from PIL import Image

PROTOS = [
    ("O_COLLECTION_FILTERED_AGGREGATE", "对集合施加筛选/分面后,读出结果的计数、区间、极值或汇总"),
    ("O_COLLECTION_RANKED_ITEM", "在排序或分页的结果列表里定位某一项"),
    ("O_BOOKING_AVAILABILITY_RESULT", "输入日期/人数等条件后读出预订或可用性结果"),
    ("O_FORM_CALCULATED_RESULT", "填表或提交计算器后读出算出来的结果"),
    ("O_PRODUCT_CONFIGURATOR_VALUE", "配置产品选项后读出随之变化的值(价格/规格/可用性)"),
    ("O_REVEALED_UI_VALUE", "值藏在需要展开/切换/悬停的有状态控件后面"),
    ("O_STATEFUL_MULTI_TURN_PLAY", "多轮交互推进到终局(网页游戏、测验、逐轮猜测),读最终状态"),
    ("O_INTERACTIVE_CHART_TABLE_READING", "在交互式图表/仪表盘/透视表上调整维度后读数"),
    ("O_MAP_ROUTE_SPATIAL_STATE", "地图上的路径、距离、通勤时间或空间位置查询"),
    ("O_SESSION_STATE_MUTATION", "改变会话状态并验证(加购物车、设为默认门店、切换语言、收藏)"),
    ("O_SESSION_GENERATED_ARTIFACT", "会话中生成的产物(导出、分享链接、生成结果)"),
    ("O_VISUAL_MEDIA_READING", "答案只能从渲染出来的视觉证据里读到(图像/画布/媒体)"),
]

SYSTEM = """
You audit one "operation phase" segment of a web-agent trajectory and decide
whether it genuinely required browser operation.

## O prototypes (the 12 shapes of real operation)
""" + "\n".join(f"  {i}. {k} —— {v}" for i, (k, v) in enumerate(PROTOS, 1)) + """

## Verdict
operation_required  This segment really lands on one of the O prototypes: the value
                    was CREATED OR REVEALED by this sequence of interactions, and
                    is unreachable without them.
static_readable     The value was already printed on a static page and the agent
                    merely did goto / scrolled / read. Paging through, or clicking
                    an ordinary link to another static page, is still
                    static_readable.
search_coverable    A search engine could have returned this directly; visiting the
                    site was unnecessary.
degenerate          The segment accomplished nothing: only goto/wait/scroll with no
                    substantive interaction, the page errored, an anti-bot wall
                    blocked it, or it switched away immediately on entry.

## Evidence rules (important)
  screenshots  ground truth — what the page actually displayed.
  <think>      the teacher's own narration; it MAY CONTRADICT the page. Treat it as
               a hint only, never as sole grounds.
When deciding "did the interaction actually change page state", compare the
before/after screenshots first.

## Output (JSON only)
{"prototype": "<one of the 12 prototype ids, or null>",
 "verdict": "operation_required|static_readable|search_coverable|degenerate",
 "state_changed": true,
 "why": "<one line, saying whether you relied on a screenshot (which one) or on narration>"}"""

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
NOOP = {"goto", "wait", "scroll", "switch", "go_back", "screenshot"}


def seg_actions(msgs: list) -> list:
    out = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        for j in TOOL_RE.findall(c):
            try:
                d = json.loads(j)
            except Exception:
                continue
            a = {k: v for k, v in (d.get("arguments") or {}).items() if k not in ("x", "y")}
            out.append((d.get("name"), a))
    return out


def seg_thinks(msgs: list) -> list:
    out = []
    for m in msgs:
        if m.get("role") != "assistant" or not isinstance(m.get("content"), str):
            continue
        t = re.search(r"<think>(.*?)</think>", m["content"], re.S)
        if t:
            out.append(t.group(1).strip())
    return out


def seg_images(msgs: list) -> list:
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for x in c:
                if x.get("type") == "image_url":
                    p = (x.get("image_url") or {}).get("url") or ""
                    if p and os.path.isfile(p):
                        out.append(p)
    return out


def pick(imgs: list, n: int) -> list:
    """首帧 + 末尾若干帧:开头给起点,结尾是交互结果最可能显形的地方。"""
    if len(imgs) <= n:
        return imgs
    return [imgs[0]] + imgs[-(n - 1):]


def shrink(p: str, w=1000, q=70) -> str:
    im = Image.open(p).convert("RGB")
    if im.width > w:
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return base64.b64encode(b.getvalue()).decode()


def task_text(msgs: list) -> str:
    for m in msgs:
        if m.get("role") == "user":
            c = m.get("content")
            s = c if isinstance(c, str) else ""
            g = re.search(r"##\s*Task\s*\n(.+?)(?:\n##|\Z)", s, re.S)
            if g:
                return g.group(1).strip()
            return s[:600]
    return ""


def build(row: dict, nimg: int):
    msgs = row["messages"]
    acts = seg_actions(msgs)
    thinks = seg_thinks(msgs)
    imgs = pick(seg_images(msgs), nimg)
    real = [a for a in acts if a[0] not in NOOP]
    txt = (f"## 任务(整条轨迹的目标)\n{task_text(msgs)[:900]}\n"
           f"\n## 这一段是第 {row['segment']+1}/{row['segment_of']} 段,{row['seg_steps']} 步,"
           f"结束于 {row.get('ends_with')}\n"
           f"\n## 本段动作序列({len(acts)} 个,其中非导航类 {len(real)} 个)\n"
           + "\n".join(f"{i}. {n} {json.dumps(a, ensure_ascii=False)[:110]}"
                       for i, (n, a) in enumerate(acts, 1))
           + "\n\n## 教师自述(仅线索,可能与页面不符)\n"
           + "\n---\n".join(t[:400] for t in thinks[:8]))
    return txt, imgs, len(acts), len(real)


def one(cli, model, row, nimg, timeout, attempts):
    key = f"{row['task_id']}#{row['segment']}"
    txt, imgs, n_act, n_real = build(row, nimg)
    content = [{"type": "text", "text": txt}]
    for i, p in enumerate(imgs, 1):
        try:
            content.append({"type": "text", "text": f"[截图 {i}/{len(imgs)}]"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{shrink(p)}"}})
        except Exception:
            pass
    msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    err = "?"
    for _ in range(attempts):
        try:
            r = cli.chat.completions.create(model=model, messages=msg, timeout=timeout,
                                            temperature=0, response_format={"type": "json_object"})
            o = json.loads(r.choices[0].message.content)
            if o.get("verdict") in ("operation_required", "static_readable",
                                    "search_coverable", "degenerate"):
                o.update(task_id=row["task_id"], segment=row["segment"],
                         seg_steps=row["seg_steps"], n_act=n_act, n_real=n_real,
                         n_img=len(imgs))
                return key, o
        except Exception as e:
            err = str(e)[:110]
    return key, {"error": err, "task_id": row["task_id"], "segment": row["segment"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="runs/train_hybrid_all.jsonl")
    ap.add_argument("--input", default="", help="任务表,用来取 switch_form")
    ap.add_argument("--forms", default="F2,F3,F4",
                    help="只判这些 form 的 O 段。F1 不在其中 —— F1 的定义就是"
                         "「起手在 O,认出这是检索型问题、立刻 switch 到 S」,"
                         "它的 O 段本来就该是 1 步 switch,按「有没有做 O 独有的事」"
                         "判必然是 degenerate,但那条样本的训练价值是 switch 决策。"
                         "留空表示不按 form 过滤。")
    ap.add_argument("--out", default="runs/hybrid_o_purity")
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nimg", type=int, default=5)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    O = [r for r in rows if r.get("side") == "O"]
    if args.forms and args.input:
        keep = {x.strip() for x in args.forms.split(",") if x.strip()}
        forms = {}
        for l in open(args.input, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                forms[d["task_id"]] = ((d.get("source") or {}).get("switch_form") or "?")
        before = len(O)
        O = [r for r in O if forms.get(r["task_id"]) in keep]
        print(f"按 form 过滤({','.join(sorted(keep))}):O 段 {before} → {len(O)}", flush=True)
    if args.resume:
        # 产出文件名是 <task_id>_seg<N>.json(见写出处 k.replace('#','_seg')),
        # 早先这里拿 "<task_id>#<N>" 去比,永远匹配不上 —— --resume 形同虚设,
        # 每轮把全部 O 段从头重判。实测 483 个已判文件一个没复用,每轮多烧 45 分钟。
        have = {p.stem for p in outd.glob("*.json")}
        _before = len(O)
        O = [r for r in O if f"{r['task_id']}_seg{r['segment']}" not in have]
        print(f"resume: 已判 {len(have)} 段,跳过 {_before-len(O)},待判 {len(O)}", flush=True)
    if args.limit:
        O = O[:args.limit]
    print(f"待判 O 段 {len(O)} 条,模型 {args.model},每条最多 {args.nimg} 张图", flush=True)

    cli = OpenAI(base_url=os.environ.get("JUDGE_BASE_URL") or os.environ.get("BASE_URL", "http://localhost:8000/v1"),
                 api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("API_KEY", "EMPTY"))
    t0 = time.time(); done = 0
    with cf.ThreadPoolExecutor(args.max_workers) as ex:
        futs = [ex.submit(one, cli, args.model, r, args.nimg, args.timeout, args.attempts)
                for r in O]
        for f in cf.as_completed(futs):
            k, o = f.result()
            (outd / f"{k.replace('#','_seg')}.json").write_text(
                json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(O)}  {el/done:.1f}s/条  剩 {(len(O)-done)*el/done/60:.0f}min",
                      flush=True)
    print("完成 ->", outd, flush=True)


if __name__ == "__main__":
    main()
