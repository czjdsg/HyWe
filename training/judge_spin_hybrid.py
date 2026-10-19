#!/usr/bin/env python3
"""宽口径空转判读,S 侧和 O 侧都做。只剔明显原地打转的步,探索一律保留。

## 两侧的空转形态不同,判据要分开写

  O 侧(操作路径):反复点同一处而页面纹丝不动、在同一屏来回上下滚、
                  goto 当前 URL。判据看**画面变没变**,所以给判官看截图。

  S 侧(搜索路径):换个措辞把同一个问题搜第 N 遍、重复 visit 同一个 URL、
                  搜索返回的结果和上一次几乎一样却接着再搜。判据看**信息增量**,
                  给判官看查询词和返回结果的摘要,不需要图。

  实测这批 S 侧确实有连着 4 次近义搜索、第 2、3 次返回同一条结果的情形。

## 默认 keep

为找内容而滚动、换个角度再搜一次、打开菜单看一眼再关掉、点进详情页再返回 ——
这些都是必要的探索,一律保留。只有「重复了前面做过的事**且**没带来任何新东西」才判 spin。
判不准一律 keep。

## 与坐标判读的分工

坐标判官管「这一下点在哪」(事实性),这里管「这一步有没有意义」(语义)。
两者取并集做 mask。
"""
from __future__ import annotations
import argparse, collections, concurrent.futures as cf, json, os, re, sys, threading, time

TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
AN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
STATUS = re.compile(r"-\s*status:\s*(\S+)")
SU = re.compile(r"start_url:\s*(\S+)"); EU = re.compile(r"end_url:\s*(\S+)")
TITLE = re.compile(r"^\s*\d+\.\s*(.+?)\s*URL:\s*(\S+)", re.M)

SYS_O = """你在审一条网页 agent 的**操作**轨迹,只挑出明显**空转**的步。

给你:任务、每一步的动作与执行观测,以及该步动作**执行之前**的页面截图。
**第 i 步配的图是动作前的画面;要看第 i 步的效果,请看第 i+1 步的图。**

spin(空转)= 同时满足两条:
  1. 这一步做的事,**前面已经做过**
  2. 并且**没有带来任何新画面、新信息、新状态**

典型:反复点同一处而页面纹丝不动;在同一屏来回上下滚动;goto 当前 URL;
连着好几步做同一件毫无反应的事。

下面这些**一律 keep**,是真实操作里必要的探索:
  * 为找内容而滚动 —— **哪怕滚了半天什么也没找到**
  * 打开菜单/筛选面板看一眼再关掉;点进详情页再返回
  * 试了个筛选条件发现不对;第一次尝试失败换个方式;等页面加载

单步"对完成任务没有直接推进"**不是**空转的理由。拿不准一律 keep。"""

SYS_S = """你在审一条网页 agent 的**搜索**轨迹,只挑出明显**空转**的步。

给你:任务、每一步的动作(search 的查询词 / visit 的 URL),以及返回结果的摘要。

spin(空转)= 同时满足两条:
  1. 这一步实质上重复了前面某一步
  2. 并且**没有带来任何新信息**

典型:
  * 把同一个问题换个措辞再搜一遍,返回的结果和上一次高度重合
  * 重复 visit 同一个 URL
  * 连着搜好几次,结果集几乎不变却还在搜

下面这些**一律 keep**:
  * 换一个**实质不同**的角度再搜(加了新的限定词、换了实体、换了语言)
  * 搜出来发现不对,改用别的关键词
  * visit 一个结果发现没用,再 visit 另一个
  * 为了交叉验证同一事实去看第二个来源

判断重点是**信息增量**:哪怕查询词看着像,只要返回了前面没见过的结果,就算 keep。
拿不准一律 keep。"""

OUT_FMT = """

严格输出 JSON,不要额外文字:
{"steps":[{"i":1,"v":"keep"},{"i":5,"v":"spin","why":"不超过15字"}]}
v=keep 时不要 why;v=spin 时必须给 why,说明它重复了第几步、哪里没有增量。
steps 必须覆盖给到的每一个步号。"""



def _img_url(base: str, image_dir: str, name: str) -> str:
    """Image reference for the judge: <base>/<name> when an HTTPS mirror of the
    image directory is configured (--url-base), else an inline base64 data URL."""
    if base:
        return f"{base}/{name}"
    import base64
    with open(os.path.join(image_dir, name), "rb") as fh:
        return "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()

def mtext(m):
    c = m.get("content")
    if isinstance(c, list):
        return "".join(z.get("text", "") for z in c
                       if isinstance(z, dict) and z.get("type") == "text")
    return str(c or "")


def obs_o(t):
    bits = []
    m = STATUS.search(t)
    if m: bits.append(f"status={m.group(1)}")
    su, eu = SU.search(t), EU.search(t)
    if su and eu:
        bits.append("URL 未变" if su.group(1) == eu.group(1) else f"URL -> {eu.group(1)[:60]}")
    return " | ".join(bits) or " ".join(t.split())[:150]


def obs_s(t, k=4):
    """S 侧:抽出前 k 条结果的标题+URL —— 判重复搜索靠的就是结果集重不重合。"""
    hits = TITLE.findall(t)
    if hits:
        return "\n".join(f"      - {a[:70]} | {b[:70]}" for a, b in hits[:k])
    return "      " + " ".join(t.split())[:260]


def parse(t):
    cands, depth, start = [], 0, 0
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0: cands.append(t[start:i + 1])
    for c in reversed(cands):
        if '"steps"' not in c: continue
        try: return json.loads(c, strict=False)
        except Exception: pass
    steps = []
    for mm in re.finditer(r'\{[^{}]*"i"\s*:\s*\d+[^{}]*\}', t):
        try: steps.append(json.loads(mm.group(0), strict=False))
        except Exception: pass
    return {"steps": steps} if steps else None


def one(args):
    from openai import OpenAI
    d, si, a, base = args
    cli = OpenAI(base_url=a.base_url, api_key=a.api_key, timeout=a.timeout, max_retries=0)
    side = d.get("side")
    msgs = d["messages"]
    task = ""
    for m in msgs:
        if m["role"] == "user":
            mm = re.search(r"##\s*Task\s*\n(.*?)(?=\n##|\Z)", mtext(m), re.S)
            task = (mm.group(1).strip() if mm else "")[:400]; break
    steps, pend = [], ""
    for m in msgs:
        if m["role"] == "user":
            pend = mtext(m); continue
        if m["role"] != "assistant": continue
        t = mtext(m)
        x = TC.search(t)
        try: tc = json.loads(x.group(1)) if x else {}
        except Exception: tc = {}
        steps.append({"tc": tc, "obs": pend, "is_ans": bool(AN.search(t))})
        pend = ""

    sysmsg = SYS_O + OUT_FMT if side == "O" else SYS_S + OUT_FMT
    out, errs = {}, []
    i = 0
    while i < len(steps):
        j = min(i + a.window, len(steps))
        content = [{"type": "text", "text": f"## 任务\n{task}\n"}]
        for k in range(i, j):
            s = steps[k]
            act = json.dumps(s["tc"], ensure_ascii=False)[:220] if s["tc"] else "answer(收口)"
            if side == "O":
                head = (f"\n### 第 {k+1} 步\n动作: {act}\n执行观测: {obs_o(s['obs'])}\n"
                        f"动作**执行前**的画面:")
                content.append({"type": "text", "text": head})
                p = os.path.join(a.jpg_dir, f"{si}_{k}.jpg")
                if os.path.exists(p):
                    content.append({"type": "image_url",
                                    "image_url": {"url": _img_url(base, a.jpg_dir, f"{si}_{k}.jpg")}})
            else:
                head = (f"\n### 第 {k+1} 步\n动作: {act}\n返回结果:\n{obs_s(s['obs'])}")
                content.append({"type": "text", "text": head})
        obj = None
        for att in range(a.attempts):
            try:
                r = cli.chat.completions.create(
                    model=a.model, temperature=0.0 if att == 0 else 0.3,
                    max_tokens=a.max_tokens,
                    messages=[{"role": "system", "content": sysmsg},
                              {"role": "user", "content": content}])
                obj = parse(r.choices[0].message.content or "")
            except Exception as e:
                # 只记类名会让 BadRequest 这类「请求本身不合法」的问题无从下手,把消息也留下
                errs.append(f"api:{type(e).__name__}:{str(e)[:160]}")
                time.sleep(3 + 4 * att); continue
            if obj: break
            errs.append("bad_json")
        if obj:
            for s in obj.get("steps") or []:
                try: k = int(s.get("i")) - 1
                except Exception: continue
                if not (0 <= k < len(steps)): continue
                v = str(s.get("v") or "").lower()
                if v not in ("keep", "spin"): continue
                out[str(k)] = {"v": v, "why": (s.get("why") or "")[:60]} if v == "spin" else {"v": "keep"}
        if j >= len(steps): break
        i = max(i + 1, j - a.overlap)
    return {"si": si, "task_id": d.get("task_id"), "segment": d.get("segment"),
            "side": side, "n_steps": len(steps), "verdicts": out, "errs": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/train_hybrid_all.jsonl")
    ap.add_argument("--out", default="runs/hybrid_spin.jsonl")
    ap.add_argument("--jpg-dir", default="runs/hybrid_jpg")
    ap.add_argument("--side", default="", help="只判某一侧;空则两侧")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--overlap", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY") or "EMPTY")
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--url-base", default=os.environ.get("HYBRID_IMAGE_URL_BASE", ""))
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                x = json.loads(l)
                if not x.get("errs") and len(x.get("verdicts") or {}) >= x.get("n_steps", 0):
                    done.add(x["si"])
            except Exception: pass
    todo = []
    for si, l in enumerate(open(a.src)):
        d = json.loads(l)
        if a.side and d.get("side") != a.side: continue
        if si in done: continue
        todo.append((d, si))
        if a.limit and len(todo) >= a.limit: break
    need_img = any(d.get("side") == "O" for d, _ in todo)
    if need_img and not (a.url_base or a.jpg_dir):
        sys.exit("O-side samples need --jpg-dir (from make_hybrid_jpegs.py) or --url-base")
    print(f"待判 {len(todo):,}(缓存 {len(done):,});模型 {a.model}", file=sys.stderr, flush=True)

    lock = threading.Lock(); fo = open(a.out, "a", encoding="utf-8")
    st = collections.Counter(); t0 = time.time()
    base = (a.url_base or "").rstrip("/")
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, (d, si, a, base)) for d, si in todo]
        for k, fu in enumerate(cf.as_completed(futs), 1):
            try: r = fu.result()
            except Exception as e:
                st[f"exc:{type(e).__name__}"] += 1; continue
            st["steps"] += r["n_steps"]
            for v in r["verdicts"].values(): st[f"{r['side']}:{v['v']}"] += 1
            st["缺判"] += r["n_steps"] - len(r["verdicts"])
            for e in r["errs"]: st[e] += 1
            with lock:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(todo)} | {(time.time()-t0)/60:.1f} 分 | {dict(st.most_common(7))}",
                      file=sys.stderr, flush=True)
    fo.close()
    print(f"\n{dict(st.most_common())}", file=sys.stderr)
    for s in ("S", "O"):
        k, p = st[f"{s}:keep"], st[f"{s}:spin"]
        if k + p: print(f"{s} 侧 spin {100*p/(k+p):.1f}% ({p:,}/{k+p:,})", file=sys.stderr)


if __name__ == "__main__":
    main()
