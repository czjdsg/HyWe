#!/usr/bin/env python3
"""判「这一步点准了没有」:给判官看标了坐标的截图 + 本步 think + 动作。

和之前那版 mask 判官的区别:那一版看的是**动作前后两帧**,靠页面有没有变化推断成败,
点歪但页面恰好有反应的会被放过。这一版把坐标**直接画在图上**,判官看的是
标记底下压着什么控件,能直接回答「想点的东西和实际落点是不是同一个」。

判定三挡:
    hit      标记落在 think 想要的那个控件上(或其可点区域内)
    miss     落在别处 —— 要说出实际压着什么
    unclear  截图看不清、think 没说清目标、或元素本身有歧义

scroll 带的坐标只是滚动锚点,不要求压在具体控件上,落在正确的滚动区域内即算 hit;
判官提示词里单独交代了这一条,否则 scroll 会被大面积误判成 miss。

用法(先用 mark_coord_images.py 画好标记图;图片默认 base64 内联,
也可用 --url-base 指向该目录的 HTTPS 镜像):
    python training/mark_coord_images.py --in runs/train_hybrid_all.jsonl \
        --img-dir <images root> --dst runs/coord_marked --index runs/coord_marked.jsonl
    python training/judge_coord_accuracy.py \
         --index runs/coord_marked.jsonl --image-dir runs/coord_marked \
         --out runs/coord_verdicts.jsonl
"""
from __future__ import annotations
import argparse, collections, concurrent.futures as cf, json, os, re, sys, threading, time

SYS = """你在检查一个网页 agent 的一步操作**执行得准不准**。

判定基准是 **CoT + 动作** 这一对:
  * CoT 说明了这一步**打算做什么**(操作什么元素、想达成什么效果)
  * 动作是**实际做了什么**(动作类型 + 参数 + 坐标)
判的是:动作有没有忠实地执行 CoT 说的那件事。

**不要判 CoT 本身的决策对不对**(该不该点这里、对完成任务有没有帮助),
那不是这一关的事。CoT 说要点 A,动作就该点在 A 上 —— 只判这个。

给你的截图上**已经画好了这一步动作的坐标**:
  * 红色空心圆 + 十字准星 = 点击/滚动锚点,或拖拽起点
  * 蓝色方框 + 箭头       = 拖拽终点
  标记中心那个小点是精确落点,圆环只为不挡住底下的控件。

要查两件事,**两件都对才算 hit**:

  A. 动作类型和参数与 CoT 一致
     CoT 说要输入文字,动作却是 click -> 不一致
     CoT 说向下滚,动作却向上滚 -> 不一致
     CoT 说拖动滑块调最大值,动作却把起点放在最小值手柄上 -> 不一致
  B. 坐标落点压在 CoT 所述的那个元素上
     按钮/链接/输入框:落在它的框内都算
     小图标(放大镜、X、箭头):落在图标外接框内都算
     日历日期、下拉菜单项这类密集元素:必须压在**正确的那一项**上,压到相邻项算不准
     **scroll 的坐标只是滚动锚点**,不要求压在某个具体控件上,
       落在 CoT 所指的那块可滚动区域内就算对

判定三挡:
  hit      A、B 都对
  miss     A 或 B 不对 —— 用 k 字段说明是哪一类,并说出落点实际压着什么
  unclear  截图糊/被遮挡看不清、CoT 没说清要操作什么、页面上有多个同名元素分不出

只依据截图判断,不要脑补页面上没有的东西。

严格输出 JSON,不要额外文字:
{"v":"hit|miss|unclear","k":"coord|action|","want":"CoT 要操作的元素(≤12字)","on":"落点实际压着什么(≤12字)","why":"一句话"}
k 只在 v=miss 时填:coord = 坐标压错了元素;action = 动作类型或参数与 CoT 不符。
"""


def parse(t):
    r"""从可能夹带大段分析散文的回复里挖出判词。

    贪婪的 `\{.*\}` 会从第一个左括号匹到最后一个右括号,把散文一起吞进去必然解析失败;
    模型又常在 JSON 前先写一段推理,所以扫描所有**配平**的花括号块,取最后一个含 "v" 的。
    """
    cands = []
    depth = start = 0
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0: cands.append(t[start:i + 1])
    # 散文里出现落单的 "{"(比如描述控件名时)会让配平扫描一路吃到结尾都不闭合,
    # 于是补一条兜底:直接抓不含嵌套、且带 "v" 字段的最小花括号块
    cands += re.findall(r'\{[^{}]*"v"\s*:[^{}]*\}', t)
    for c in reversed(cands):
        if '"v"' not in c: continue
        try: return json.loads(c, strict=False)
        except Exception: pass
    return None



def _img_url(base: str, image_dir: str, name: str) -> str:
    """Image reference for the judge: <base>/<name> when an HTTPS mirror of the
    image directory is configured (--url-base), else an inline base64 data URL."""
    if base:
        return f"{base}/{name}"
    import base64
    with open(os.path.join(image_dir, name), "rb") as fh:
        return "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()


def one(args):
    from openai import OpenAI
    r, a, base = args
    cli = OpenAI(base_url=a.base_url, api_key=a.api_key, timeout=a.timeout, max_retries=0)
    tc = r["tc"]; th = (r.get("think") or "").strip()
    txt = (f"## 这一步的思考\n{th[:1500]}\n\n"
           f"## 实际执行的动作\n{json.dumps(tc, ensure_ascii=False)}\n\n"
           f"## 截图(坐标已标注)")
    content = [{"type": "text", "text": txt},
               {"type": "image_url", "image_url": {"url": _img_url(base, a.image_dir, f"{r['key']}.jpg")}}]
    for att in range(a.attempts):
        try:
            resp = cli.chat.completions.create(
                model=a.model, temperature=0.0 if att == 0 else 0.3, max_tokens=a.max_tokens,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": content}])
            o = parse(resp.choices[0].message.content or "")
            if o and o.get("v") in ("hit", "miss", "unclear"):
                return {**{k: r[k] for k in ("key", "traj_id", "seg", "si", "ai", "loss")},
                        "name": tc.get("name"), "args": tc.get("arguments"),
                        "v": o["v"], "k": o.get("k", ""),
                        "want": o.get("want", ""), "on": o.get("on", ""),
                        "why": o.get("why", "")}
        except Exception as e:
            err = f"{type(e).__name__}"
            time.sleep(3 + 4 * att)
            continue
        time.sleep(1)
    return {**{k: r[k] for k in ("key", "traj_id", "seg", "si", "ai", "loss")},
            "name": tc.get("name"), "args": tc.get("arguments"), "v": "err"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--attempts", type=int, default=3)
    # 模型常先写一段分析再给 JSON,700 会把 JSON 截掉(finish_reason=length)
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY") or "EMPTY")
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--url-base", default=os.environ.get("HYBRID_IMAGE_URL_BASE", ""),
                    help="optional HTTPS mirror of --image-dir; default = inline base64")
    ap.add_argument("--image-dir", default="", help="directory written by mark_coord_images.py --dst")
    a = ap.parse_args()
    if not a.url_base and not a.image_dir:
        sys.exit("need --image-dir (or --url-base)")

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                d = json.loads(l)
                if d.get("v") != "err": done.add(d["key"])
            except Exception: pass
    rows = []
    for l in open(a.index):
        d = json.loads(l)
        if d["key"] not in done: rows.append(d)
    print(f"待判 {len(rows):,}(缓存 {len(done):,});模型 {a.model}", file=sys.stderr, flush=True)

    lock = threading.Lock(); fo = open(a.out, "a", encoding="utf-8")
    st = collections.Counter(); t0 = time.time()
    base = a.url_base.rstrip("/")
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, (r, a, base)) for r in rows]
        for k, fu in enumerate(cf.as_completed(futs), 1):
            try: r = fu.result()
            except Exception as e:
                st[f"exc:{type(e).__name__}"] += 1; continue
            st[r["v"] + (":" + r["k"] if r.get("k") else "")] += 1
            with lock:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(rows)} | {(time.time()-t0)/60:.1f} 分 | {dict(st)}",
                      file=sys.stderr, flush=True)
    fo.close()
    n = sum(v for k, v in st.items() if not k.startswith(("err", "exc")))
    st = collections.Counter({k: v for k, v in st.items()})
    hit = st["hit"]; miss = sum(v for k, v in st.items() if k.startswith("miss"))
    unc = sum(v for k, v in st.items() if k.startswith("unclear"))
    print(f"\n{dict(st)}", file=sys.stderr)
    if n:
        print(f"hit {100*hit/n:.1f}% · miss {100*miss/n:.1f}% · "
              f"unclear {100*unc/n:.1f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
