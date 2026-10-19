#!/usr/bin/env python3
"""打包可发布数据集。

图片保持采集时的**原始 1280x720 PNG**(无损)—— 训练/推理端吃的就是这个尺寸,
配 0-1000 归一化坐标;1000x1000 方图只是教师采集当时的输入变换,不进数据集。

入库用硬链接(同一文件系统零额外占用),按内容 sha1 命名做全局去重。
输出:
  <out>/data/train.jsonl   image_url 改为 images/xx/<sha1>.png 相对路径
  <out>/images/<2位>/<sha1>.png
  <out>/meta.json
"""
import argparse, collections, hashlib, json, os, shutil, time
import concurrent.futures as cf

FILES = ["runs/train_recheck1802.strict.jsonl", "runs/train_v2_8600.opure.jsonl",
         "runs/train_qwen.opure.jsonl", "runs/train_insta_kimi.opure.jsonl",
         "runs/train_insta_qwen.opure.jsonl"]

def ingest(src, outdir):
    try:
        h = hashlib.sha1()
        with open(src, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
        d = h.hexdigest()
        rel = f"images/{d[:2]}/{d}.png"
        dst = os.path.join(outdir, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try: os.link(src, dst)           # 同一 fs:零拷贝
            except OSError: shutil.copyfile(src, dst)
        return src, rel, None
    except Exception as e:
        return src, None, f"{type(e).__name__}: {e}"[:100]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=FILES)
    ap.add_argument("--out", default="runs/release/HybridWebTraj")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)

    rows, srcs = [], set()
    for f in a.files:
        tag = os.path.basename(f).replace("train_", "").split(".")[0]
        for l in open(f, encoding="utf-8"):
            o = json.loads(l); o["batch"] = tag; rows.append(o)
            for m in o["messages"]:
                c = m.get("content")
                if isinstance(c, list):
                    for x in c:
                        if x.get("type") == "image_url": srcs.add(x["image_url"]["url"])
    grp = collections.defaultdict(list)
    for r in rows: grp[(r["task_id"], r["batch"])].append(r)
    bytid = collections.defaultdict(list)
    for k in grp: bytid[k[0]].append(k)
    dropped = []
    for tid, ks in bytid.items():
        if len(ks) < 2: continue
        best = max(ks, key=lambda k: (sum(r["seg_steps"] for r in grp[k] if r["side"] == "O"),
                                      sum(r["seg_steps"] for r in grp[k])))
        for k in ks:
            if k != best: dropped.append(k)
    if dropped:
        drop = set(dropped)
        rows = [r for r in rows if (r["task_id"], r["batch"]) not in drop]
        print(f"跨批次重复任务 {len(dropped)} 条(同任务采了两次),按 O 步数留优: "
              + ", ".join(f"{t}@{b}" for t, b in dropped[:5]), flush=True)
        srcs = set()
        for o in rows:
            for m in o["messages"]:
                c = m.get("content")
                if isinstance(c, list):
                    for x in c:
                        if x.get("type") == "image_url": srcs.add(x["image_url"]["url"])
    seen = {}
    for r in rows:
        k = (r["task_id"], r["segment"])
        assert k not in seen, f"段号重复: {k}"
        seen[k] = 1
    print(f"样本 {len(rows)} | 唯一图片 {len(srcs)}", flush=True)

    cache_p = os.path.join(a.out, ".hash_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = sorted(srcs - set(cache))
    mapping = {s_: cache[s_] for s_ in srcs if s_ in cache}
    errs = []
    t0 = time.time(); n = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for src, rel, err in ex.map(lambda p: ingest(p, a.out), todo):
            n += 1
            if err: errs.append((src, err))
            else: mapping[src] = rel
            if n % 10000 == 0: print(f"  入库 {n}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(mapping, open(cache_p, "w"))
    print(f"入库完成 {len(mapping)}(新增 {n - len(errs)})| 失败 {len(errs)}", flush=True)
    for s, e in errs[:5]: print("   ", s, e)

    miss = 0
    for o in rows:
        for m in o["messages"]:
            c = m.get("content")
            if not isinstance(c, list): continue
            for x in c:
                if x.get("type") != "image_url": continue
                u = x["image_url"]["url"]
                if u in mapping: x["image_url"]["url"] = mapping[u]
                else: miss += 1
    out_jsonl = os.path.join(a.out, "data", "train.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as w:
        for o in rows: w.write(json.dumps(o, ensure_ascii=False) + "\n")

    uniq = set(mapping.values())
    isz = sum(os.path.getsize(os.path.join(a.out, r)) for r in uniq)
    meta = {"samples": len(rows), "trajectories": len({r["task_id"] for r in rows}),
            "images": len(uniq), "image_format": "原始 1280x720 PNG(无损)",
            "coord_space": "0-1000 归一化整数; px = x*1280/1000, py = y*720/1000",
            "unresolved_image_refs": miss, "ingest_errors": len(errs),
            "images_bytes": isz, "jsonl_bytes": os.path.getsize(out_jsonl)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
