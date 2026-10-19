#!/usr/bin/env python3
"""把动作坐标画到截图上,供判官检查「这一步点没点准」。

坐标口径:0~1000 的相对值,逐轴归一化(见 inference/hybrid_agent.py 的 _scale_xy)。
截图统一 1280x720,所以 px = x*1280/1000, py = y*720/1000。

## 标记怎么画才不挡住目标

判官要看的恰恰是标记**底下**是什么控件,所以中心只留一个 2px 的点,
外面套空心圆和十字准星 —— 圆环半径 20px,把目标围起来而不是盖住。
文字标签甩到离画面中心较远的一侧,避免压住待判元素。

    click / scroll   空心圆 + 十字 + 中心点
    drag             起点空心圆、终点空心方、之间一条带箭头的线

## 输出

每步一张 JPEG,文件名 `<样本序>_<步序>.jpg`,和判官请求里的 step_key 一一对应。
"""
from __future__ import annotations
import argparse, json, math, os, re, sys
from PIL import Image, ImageDraw

TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
TH = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)
RED = (255, 40, 40)
BLUE = (40, 120, 255)


def mt(m):
    c = m.get("content")
    if isinstance(c, list):
        return "".join(z.get("text", "") for z in c
                       if isinstance(z, dict) and z.get("type") == "text")
    return str(c or "")


def imgs_of(m):
    c = m.get("content")
    if not isinstance(c, list): return []
    return [z["image_url"]["url"] for z in c
            if isinstance(z, dict) and z.get("type") == "image_url"]


def ring(dr, px, py, color, r=20, w=3):
    dr.ellipse([px - r, py - r, px + r, py + r], outline=color, width=w)
    dr.line([px - r - 10, py, px - 6, py], fill=color, width=w)
    dr.line([px + 6, py, px + r + 10, py], fill=color, width=w)
    dr.line([px, py - r - 10, px, py - 6], fill=color, width=w)
    dr.line([px, py + 6, px, py + r + 10], fill=color, width=w)
    dr.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)


def label(dr, px, py, text, color, W, H):
    # 甩向离画面中心远的一侧,别压住待判元素
    dx = 30 if px < W / 2 else -30 - 8 * len(text)
    dy = 30 if py < H / 2 else -42
    x, y = max(2, min(W - 8 * len(text) - 6, px + dx)), max(2, min(H - 20, py + dy))
    dr.rectangle([x - 3, y - 2, x + 7 * len(text) + 3, y + 15], fill=(0, 0, 0))
    dr.text((x, y), text, fill=color)


def draw(img, tc):
    a = (tc or {}).get("arguments") or {}
    W, H = img.size
    dr = ImageDraw.Draw(img)
    def to_px(kx, ky):
        return a[kx] * W / 1000.0, a[ky] * H / 1000.0
    if isinstance(a.get("from_x"), (int, float)) and isinstance(a.get("to_x"), (int, float)):
        x1, y1 = to_px("from_x", "from_y"); x2, y2 = to_px("to_x", "to_y")
        ring(dr, x1, y1, RED); label(dr, x1, y1, f"FROM {a['from_x']},{a['from_y']}", RED, W, H)
        r = 18
        dr.rectangle([x2 - r, y2 - r, x2 + r, y2 + r], outline=BLUE, width=3)
        dr.ellipse([x2 - 2, y2 - 2, x2 + 2, y2 + 2], fill=BLUE)
        label(dr, x2, y2, f"TO {a['to_x']},{a['to_y']}", BLUE, W, H)
        dr.line([x1, y1, x2, y2], fill=BLUE, width=3)
        ang = math.atan2(y2 - y1, x2 - x1)
        for s in (2.6, -2.6):
            dr.line([x2, y2, x2 + 16 * math.cos(ang + s), y2 + 16 * math.sin(ang + s)],
                    fill=BLUE, width=3)
        return True
    if isinstance(a.get("x"), (int, float)) and isinstance(a.get("y"), (int, float)):
        px, py = to_px("x", "y")
        ring(dr, px, py, RED)
        label(dr, px, py, f"{(tc or {}).get('name','?').upper()} {a['x']},{a['y']}", RED, W, H)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
                    default="runs/aquila_v2/train_molmoweb_v2_o_only_swift.v5.jsonl")
    # jsonl 里写的是逻辑前缀 images/<sid>/N.png,实际盘上是重建过的 train_operation_images_fixed
    ap.add_argument("--img-dir", default="runs/train_operation_images_fixed")
    ap.add_argument("--dst", default="runs/coord_marked")
    ap.add_argument("--index", default="runs/coord_marked_index.jsonl")
    ap.add_argument("--sample", type=int, default=0, help="0 = 全部;否则按步等距抽样")
    # 只重渲染某份索引里的步(查看器只要那 150 条轨迹的图,不必再铺全量 3.5G)
    ap.add_argument("--keys", default="", help="已有索引 jsonl,只渲染其中的 key")
    ap.add_argument("--quality", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    cand = []
    for si, l in enumerate(open(a.src)):
        d = json.loads(l)
        pend = None
        ai = -1
        for m in d["messages"]:
            if m.get("role") == "user":
                u = imgs_of(m)
                if u: pend = u[-1]
                continue
            if m.get("role") != "assistant": continue
            ai += 1
            t = mt(m); x = TC.search(t)
            img, pend = pend, None
            if not x or not img: continue
            try: tc = json.loads(x.group(1))
            except Exception: continue
            aa = tc.get("arguments") or {}
            if not any(isinstance(aa.get(k), (int, float)) for k in ("x", "from_x")): continue
            th = TH.search(t)
            cand.append({"si": si, "ai": ai, "traj_id": d.get("traj_id"), "seg": d.get("seg"),
                         "img": img, "tc": tc, "think": th.group(1) if th else "",
                         "loss": m.get("loss", 1)})
    print(f"带坐标的步 {len(cand):,}")
    if a.keys:
        want = {json.loads(l)["key"] for l in open(a.keys)}
        cand = [c for c in cand if f"{c['si']}_{c['ai']}" in want]
        print(f"按 --keys 过滤 -> {len(cand):,}")
    if a.sample and a.sample < len(cand):
        step = len(cand) / a.sample
        cand = [cand[int(i * step)] for i in range(a.sample)]
        print(f"等距抽样 -> {len(cand):,}")

    fo = open(a.index, "w", encoding="utf-8")
    ok = miss = 0
    for c in cand:
        p = os.path.join(a.img_dir, c["img"].replace("images/", "", 1))
        if not os.path.exists(p):
            miss += 1; continue
        im = Image.open(p).convert("RGB")
        if not draw(im, c["tc"]):
            miss += 1; continue
        key = f"{c['si']}_{c['ai']}"
        im.save(os.path.join(a.dst, key + ".jpg"), "JPEG", quality=a.quality)
        c["key"] = key
        fo.write(json.dumps(c, ensure_ascii=False) + "\n")
        ok += 1
        if ok % 500 == 0: print(f"  {ok:,}")
    fo.close()
    print(f"标注好 {ok:,} 张 -> {a.dst}(跳过 {miss:,})\n索引 {a.index}")


if __name__ == "__main__":
    main()
