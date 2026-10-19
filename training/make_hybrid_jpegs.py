#!/usr/bin/env python3
"""把这批 switch 轨迹里 O 侧每一步的前置截图,转成扁平的 JPEG 镜像。

原图散在 runs/<rollout>/rollouts/<task_id>/trajectory/ 下,文件名是 post_{k}_full_screenshot.png,
判官要按「第几条样本的第几步」取图,所以拍平成 <行号>_<步序>.jpg。

转 JPEG 不只是省空间:隧道传图按字节收时间,PNG 截图动辄 1~2MB,
一个 24 张图的窗口就是 30MB,判官那端经常超时。q80 的 JPEG 一张 100KB 左右,
对「画面变没变」这个判断绰绰有余。
"""
from __future__ import annotations
import argparse, json, os, sys
from PIL import Image


def imgs(m):
    c = m.get("content")
    if not isinstance(c, list): return []
    return [z["image_url"]["url"] for z in c
            if isinstance(z, dict) and z.get("type") == "image_url"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/train_hybrid_all.jsonl")
    ap.add_argument("--dst", default="runs/hybrid_jpg")
    ap.add_argument("--side", default="O", help="只转某一侧;给空字符串则两侧都转")
    ap.add_argument("--quality", type=int, default=75)
    # 缩宽不只是省盘:判官走隧道取图,一个窗口十几张,单张越大越容易有一张取失败,
    # 而只要有一张失败整个请求就 400。判「画面变没变」用 800 宽绰绰有余。
    ap.add_argument("--max-width", type=int, default=800)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    ok = miss = skip = 0
    for si, l in enumerate(open(a.src)):
        d = json.loads(l)
        if a.side and d.get("side") != a.side: continue
        pend, ai = None, -1
        for m in d["messages"]:
            if m.get("role") == "user":
                u = imgs(m)
                if u: pend = u[-1]
                continue
            if m.get("role") != "assistant": continue
            ai += 1
            p, pend = pend, None
            if not p: continue
            out = os.path.join(a.dst, f"{si}_{ai}.jpg")
            if os.path.exists(out): skip += 1; continue
            if not os.path.exists(p): miss += 1; continue
            try:
                im = Image.open(p).convert("RGB")
                if a.max_width and im.width > a.max_width:
                    im = im.resize((a.max_width, int(im.height * a.max_width / im.width)))
                im.save(out, "JPEG", quality=a.quality)
                ok += 1
            except Exception:
                miss += 1
            if ok % 5000 == 0 and ok:
                print(f"  {ok:,}", file=sys.stderr, flush=True)
    print(f"转好 {ok:,} 张 -> {a.dst}(已存在跳过 {skip:,},缺图/坏图 {miss:,})")


if __name__ == "__main__":
    main()
