#!/usr/bin/env python3
"""把 hybrid 数据里的绝对图片路径改成可移植的相对路径。

原样是本机绝对路径:
    /abs/path/to/runs/<rollout>/rollouts/<task_id>/trajectory/post_10_full_screenshot.png
换成和 operation 数据一致的相对形式:
    images/<task_id>/post_10_full_screenshot.png

不这么做的话,文件一离开这台机器就全是死链 —— operation 那份用的就是相对路径,
两份统一之后,配套图片按 images/<id>/<file> 布好目录即可,不必改数据。

同时导出一份「引用到的图片清单」,方便之后单独打包截图。
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys, shutil
from pathlib import Path

ABS = re.compile(r"^/.*/rollouts/([^/]+)/trajectory/(.+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--copy-images", action="store_true",
                    help="Copy referenced screenshots next to the output JSONL for a self-contained dataset")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    st = collections.Counter(); files = {}
    with open(a.out, "w", encoding="utf-8") as fo:
        for d_i, l in enumerate(open(a.src)):
            d = json.loads(l)
            for m in d.get("messages") or []:
                c = m.get("content")
                if not isinstance(c, list): continue
                for z in c:
                    if not (isinstance(z, dict) and z.get("type") == "image_url"): continue
                    u = z["image_url"]["url"]
                    if not u.startswith("/"):
                        st["本来就是相对路径"] += 1; continue
                    mm = ABS.match(u)
                    if not mm:
                        st["路径形态不认识(原样保留)"] += 1; continue
                    rel = f"images/{mm.group(1)}/{mm.group(2)}"
                    if a.copy_images:
                        dst = Path(a.out).parent / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not Path(u).is_file():
                            raise FileNotFoundError(f"Referenced training image is missing: {u}")
                        if Path(u).resolve() != dst.resolve():
                            shutil.copy2(u, dst)
                    files[rel] = u
                    z["image_url"]["url"] = rel
                    st["已改写"] += 1
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    if a.manifest:
        with open(a.manifest, "w", encoding="utf-8") as fo:
            for rel, src in sorted(files.items()):
                fo.write(json.dumps({"rel": rel, "src": src,
                                     "exists": os.path.exists(src)}, ensure_ascii=False) + "\n")
    print(f"写出 {a.out}")
    for k, v in st.most_common(): print(f"  {k:<22} {v:>8,}")
    print(f"  去重后的图片文件 {len(files):,}")
    if a.manifest:
        miss = sum(1 for rel, src in files.items() if not os.path.exists(src))
        print(f"  清单 -> {a.manifest}(其中源文件缺失 {miss:,})")


if __name__ == "__main__":
    main()
