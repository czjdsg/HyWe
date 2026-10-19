#!/usr/bin/env python3
"""渲染层:把统一格式的轨迹变成判官能吃的 (文本, 图片URL) 交替内容。

## 坐标为什么要画上去

判官光看 {"x":543,"y":428} 判断不了点没点对 —— 它得知道那个位置上是什么控件。
把准心画到图上,判官就能直接看"红圈套住的是不是 think 里说的那个东西"。
上一轮 base-vs-op 审计里 `grounding` 这一类之所以能判出来,靠的就是观测里的
target html;训练数据没有 target,所以必须靠画。

准心画在**动作前**那张图上:那是模型做决策时看到的画面,判"点得对不对"要在这张上判。
换算 px = x*W/1000,与 inference/hybrid_agent.py::_scale_xy 一致(0-1000 归一化)。
drag 额外画起点→终点的箭头。

## 提示词里必须说明这是标注

不说的话判官会把红圈当成页面本身的元素,判出"页面上有个红色圆圈"这种废话 ——
实测发生过(见 gen_and_judge.py 的 marker_leak 校验)。
"""
from __future__ import annotations
import os
import io, json, math, os, re
from pathlib import Path

RED = (230, 0, 0)
BLUE = (0, 90, 230)

MARK_NOTE = ("Note: the **red crosshair / circle** on the screenshots is an annotation we "
             "drew afterwards; it marks the coordinate this step's action targets. The page "
             "itself has no such circle. It is only there to help you tell **what control "
             "actually sits at the clicked / dragged position** — do not treat it as a page "
             "element and do not mention it in your verdict.")


def _ring(dr, px, py, color=RED, r=20, w=3):
    dr.ellipse([px - r, py - r, px + r, py + r], outline=color, width=w)
    dr.line([px - r - 10, py, px - 6, py], fill=color, width=w)
    dr.line([px + 6, py, px + r + 10, py], fill=color, width=w)
    dr.line([px, py - r - 10, px, py - 6], fill=color, width=w)
    dr.line([px, py + 6, px, py + r + 10], fill=color, width=w)
    dr.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)


# 远端判官经隧道取图时, 单张 400KB+ x 24 张会打满网关的下载预算(400 Download multimodal
# file timed out)。用环境变量压尺寸/画质, 默认值不变, 不影响别人的跑。
_IMG_MAX_W = int(os.environ.get("HJ_IMG_MAX_W", "1100"))
_IMG_QUALITY = int(os.environ.get("HJ_IMG_QUALITY", "80"))


def render_step_image(src, call, out_path, max_w=None, mark=True):
    max_w = _IMG_MAX_W if max_w is None else max_w
    """读源图,按需画准心,写成 jpg。已存在就跳过。返回是否成功。

    mark=False 出**不带准心**的原图。判官要准心(它得核实动作落在哪);
    CoT 重写不要 —— 教师必须像学生一样自己在页面上找到目标元素再推落点,
    喂了准心等于让教师用了学生没有的信息,那条推理链学生复现不了。
    """
    if os.path.exists(out_path): return True
    if not src or not os.path.exists(src): return False
    try:
        from PIL import Image, ImageDraw
        im = Image.open(src).convert("RGB")
        W, H = im.size
        args = (call or {}).get("arguments") or {}
        name = (call or {}).get("name") or ""
        if mark and isinstance(args.get("x"), (int, float)) and isinstance(args.get("y"), (int, float)):
            dr = ImageDraw.Draw(im)
            px, py = float(args["x"]) * W / 1000.0, float(args["y"]) * H / 1000.0
            if name == "drag" and isinstance(args.get("x2"), (int, float)):
                qx, qy = float(args["x2"]) * W / 1000.0, float(args["y2"]) * H / 1000.0
                dr.line([px, py, qx, qy], fill=BLUE, width=3)
                ang = math.atan2(qy - py, qx - px)
                for s in (2.6, -2.6):
                    dr.line([qx, qy, qx + 16 * math.cos(ang + s), qy + 16 * math.sin(ang + s)],
                            fill=BLUE, width=3)
                _ring(dr, px, py, RED, r=14, w=3)
                _ring(dr, qx, qy, BLUE, r=14, w=3)
            else:
                _ring(dr, px, py)
        if im.width > max_w:
            im = im.resize((max_w, int(im.height * max_w / im.width)))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, "JPEG", quality=_IMG_QUALITY)
        return True
    except Exception:
        return False


def key_of(traj, i):
    return f"{traj['ds']}_{re.sub(r'[^A-Za-z0-9]', '_', str(traj['tid']))[-44:]}_{i}"


def render_traj(traj, jpg_dir, max_w=1100, mark=True):
    """导出该条轨迹所有步的 jpg,回填 traj['keys'] = {step_i: key} 和 traj['vers'] = {step_i: mtime}。

    带/不带准心的图 key 相同,必须放**不同目录**,否则会互相覆盖。

    vers 是**缓存版本号**,必须有 —— 图片服务发的是
    `Cache-Control: public, max-age=31536000, immutable`,Cloudflare 边缘一旦缓存住
    就一年不回源。文件重渲染但 URL 不变时,判官那一侧的 PoP 仍然拿到**旧图**,
    而本机 curl 命中的是本地 PoP(已经是新的),md5 一对就以为没问题。
    实测代价:答案步补图那次(09-03 重写了 4,377 张末屏)判官一张都没看到,
    于是满屏"最后一张截图显示点击没生效" —— 全是冤判。URL 带上 mtime 就自愈。
    """
    keys, vers = {}, {}
    for s in traj["steps"]:
        k = key_of(traj, s["i"])
        p = os.path.join(jpg_dir, f"{k}.jpg")
        if render_step_image(s.get("img"), s.get("call"), p, max_w, mark):
            keys[s["i"]] = k
            try: vers[s["i"]] = int(os.path.getmtime(p))
            except OSError: pass
    traj["keys"] = keys
    traj["vers"] = vers
    traj["jpg_dir"] = jpg_dir
    return keys


def img_url(traj, base, i):
    """图片 URL,带缓存版本号。见 render_traj 里为什么必须带。

    base 为空时不走图床, 直接把渲染好的 jpg 以 base64 内联(默认)。"""
    if not base:
        import base64
        with open(os.path.join(traj["jpg_dir"], f"{traj['keys'][i]}.jpg"), "rb") as fh:
            return "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()
    u = f"{base.rstrip('/')}/{traj['keys'][i]}.jpg"
    v = (traj.get("vers") or {}).get(i)
    return f"{u}?v={v}" if v else u


def prewarm(traj, url_base, workers=8, timeout=20):
    """先从本机把图拉一遍,让 Cloudflare 边缘缓存住,再让判官去取。

    图多的请求(30+ 张)网关侧常报 `Failed to download multimodal content` ——
    冷缓存下回源太慢。命名隧道的边缘缓存是真生效的(cf-cache-status: HIT,
    见 collector/scripts/tunnel_supervisor.sh),预热一遍能把这类失败基本消掉。
    失败不抛错:预热只是加速,拉不到也让判官自己去试。
    """
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    base = url_base.rstrip("/")

    def one(u):
        # 必须伪装 UA:Cloudflare 对默认的 `Python-urllib/3.x` 直接回 403,
        # 表现成"预热瞬间完成但一张没缓存",查了一轮才发现。
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (prewarm)"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                r.read(1)
        except Exception:
            pass
    us = [img_url(traj, base, i) for i in (traj.get("keys") or {})]
    if not us: return
    with ThreadPoolExecutor(min(workers, len(us))) as ex:
        list(ex.map(one, us))


def build_content(traj, url_base, head_note="", show_url=False, final_note=False,
                  repeat_final=False, lead_final=False):
    """轨迹 → OpenAI content 数组。文本与截图严格按时间序交替,不截断。

    show_url:把每步的地址栏也给判官。molmoweb 这类数据集 obs 是空的,判官唯一能
    依据的就是截图 —— 而演示录制的截图常常在页面加载完成**之前**就抓了,于是判官
    看到"点了没反应"其实是快门快了一拍。地址栏是"这一步到底生没生效"最硬的证据,
    数据里一直有(step['url']),以前没送进去。默认关,由调用方显式打开。"""
    base = url_base.rstrip("/")
    c = [{"type": "text",
          "text": (f"## Task\n{traj['task']}\n"
                   + (f"Starting site: {traj['site']}\n" if traj.get("site") else "")
                   + f"\n{MARK_NOTE}\n"
                   + ("\nEach step also carries the **address bar** as it stood at that moment. "
                      "Read it together with the screenshots: a screenshot may have been captured "
                      "before the new page finished rendering, so a page that looks unchanged but "
                      "whose address already moved on did take effect.\n" if show_url else "")
                   + (f"\n{head_note}\n" if head_note else "")
                   + f"\n## Trajectory ({len(traj['steps'])} steps)")}]
    last_i = traj["steps"][-1]["i"] if traj["steps"] else None
    if lead_final and last_i in (traj.get("keys") or {}) and traj.get("has_answer", True):
        c.append({"type": "text", "text":
                  "\n## Where the trajectory ended up\nBefore reading the steps, look at this: it is "
                  "the page **after every action in the trajectory had been carried out** — the state "
                  "the agent was looking at when it wrote its final answer. Judge the answer against "
                  "this picture. It also appears again in its place at the end of the step list."})
        c.append({"type": "image_url", "image_url": {"url": img_url(traj, base, last_i)}})
    for s in traj["steps"]:
        if s["i"] in traj.get("keys", {}):
            cap = (f"\n[{s['i']}] **The FINAL state of the page**, after every action above "
                   f"has been carried out:"
                   if (final_note and s["i"] == last_i and traj.get("has_answer", True))
                   else f"\n[{s['i']}] The page as seen before this step:")
            c.append({"type": "text", "text": cap})
            c.append({"type": "image_url", "image_url": {"url": img_url(traj, base, s["i"])}})
            head = ""
        else:
            head = f"\n[{s['i']}]"
        buf = ([head] if head else []) + [
            f"  action: {json.dumps(s.get('call') or {}, ensure_ascii=False)}"]
        if show_url and s.get("url"): buf.append(f"  address bar at this point: {s['url']}")
        if s.get("xy"): buf.append(f"  (the red crosshair above marks this coordinate {tuple(s['xy'])})")
        if s.get("think"): buf.append(f"  thought: {s['think']}")
        if s.get("obs"): buf.append(f"  result: {s['obs']}")
        c.append({"type": "text", "text": "\n".join(buf)})
    if traj.get("has_answer", True):
        if repeat_final and last_i in traj.get("keys", {}):
            c.append({"type": "text", "text":
                      "\n## The page at the end of the trajectory\nThis is the same screenshot as "
                      "step %d, repeated here so you judge the answer against it: it is the page "
                      "**after** every action above, i.e. the state the agent was looking at when "
                      "it wrote the answer." % last_i})
            c.append({"type": "image_url", "image_url": {"url": img_url(traj, base, last_i)}})
        c.append({"type": "text", "text": f"\n## Final answer text\n{traj.get('answer') or '(none)'}"})
    else:
        c.append({"type": "text",
                  "text": "\n## (this dataset has no closing step)\nThe trajectory ends right after the "
                          "last action above; the page that resulted from it was never captured."})
    return c
