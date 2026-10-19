#!/usr/bin/env python3
"""用与生产一致的浏览器配置,逐个探任务表里的起始 URL 能不能打开。

为什么必须用真浏览器而不是 curl:反爬判定看的是 TLS/HTTP2 指纹、JS 执行、
以及 navigator/WebGL 那些运行时特征。curl 探通了不代表 Playwright 也能通,
反过来也一样 —— 只有用采集时那套完全相同的配置探,结论才可迁移。

判定分四类:
  ok         页面正常加载,标题/正文不含挑战特征
  blocked    Cloudflare / 人机验证 / 403 / Access Denied
  error      网络层失败(ERR_*、超时)
  empty      加载成功但正文几乎为空(常见于需要 JS 但被拦的软墙)

产出按 URL 记,不是按域名 —— 同一站点的深链和首页可达性经常不同。
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, os, re, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
from playwright.async_api import async_playwright          # noqa: E402
from hybrid_runtime import _fp_pick, _stealth_script       # noqa: E402

BLOCK_PAT = re.compile(
    r"verify you are human|checking your browser|just a moment|cloudflare|"
    r"access denied|403 forbidden|attention required|are you a robot|"
    r"unusual traffic|bot detection|请完成安全验证|人机验证|访问被拒绝",
    re.I)


def pick_proxy(url: str, pool: list[int], fallback: str | None = None) -> str | None:
    # 池子为空时退回 --proxy-url(与 hybrid_runtime 的生产行为一致)。
    # 不加这条兜底就是"不走代理"探测 —— 结论不可迁移到采集现场。
    if not pool:
        return fallback
    h = int(hashlib.sha256(url.encode("utf-8")).hexdigest()[:12], 16)
    return f"http://127.0.0.1:{pool[h % len(pool)]}"


async def probe_one(pw, url: str, pool: list[int], timeout: int,
                    fallback: str | None = None) -> dict:
    ua, tz, gpu, cores, mem = _fp_pick(url)
    proxy = pick_proxy(url, pool, fallback)
    args = [
        "--disable-blink-features=AutomationControlled",
        "--disable-features=IsolateOrigins,site-per-process,AutomationControlled",
        "--disable-infobars", "--no-first-run", "--no-default-browser-check",
        "--disable-dev-shm-usage", "--lang=en-US", "--no-sandbox",
    ]
    if proxy:
        args.append(f"--proxy-server={proxy}")
    b = ctx = None
    t0 = time.time()
    try:
        b = await pw.chromium.launch(headless=True, args=args)
        ctx = await b.new_context(viewport={"width": 1280, "height": 720},
                                  user_agent=ua, locale="en-US", timezone_id=tz)
        await ctx.add_init_script(_stealth_script(ua, gpu, cores, mem))
        page = await ctx.new_page()
        resp = await page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
        status = resp.status if resp else 0
        # SPA 在 domcontentloaded 时正文还是空的(标题常是 "Loading …")。
        # 第一版只等 1.5s,把 89 个正常站点误判成 empty。先等网络静默,
        # 再补一段固定等待,拿不到静默也不算失败 —— 有些站长期有轮询请求。
        try:
            await page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass
        try:
            await page.wait_for_timeout(2500)
        except Exception:
            pass
        title = (await page.title()) or ""
        body = ""
        try:
            body = (await page.inner_text("body"))[:3000]
        except Exception:
            pass
        blob = f"{title}\n{body}"
        if status >= 400 or BLOCK_PAT.search(blob):
            v = "blocked"
        elif len(body.strip()) < 120:
            v = "empty"
        else:
            v = "ok"
        return {"url": url, "verdict": v, "status": status, "title": title[:90],
                "chars": len(body), "sec": round(time.time() - t0, 1), "proxy": proxy}
    except Exception as e:
        msg = str(e)[:130]
        return {"url": url, "verdict": "error", "status": 0, "title": "",
                "chars": 0, "sec": round(time.time() - t0, 1), "err": msg, "proxy": proxy}
    finally:
        for x in (ctx, b):
            try:
                if x:
                    await x.close()
            except Exception:
                pass


async def worker(name, q, pw, pool, timeout, out, lock, total, done, fallback=None):
    while True:
        try:
            url = q.get_nowait()
        except asyncio.QueueEmpty:
            return
        r = await probe_one(pw, url, pool, timeout, fallback)
        async with lock:
            out.append(r)
            done[0] += 1
            if done[0] % 50 == 0:
                c = Counter(x["verdict"] for x in out)
                print(f"  {done[0]}/{total}  {dict(c)}", flush=True)


async def main_async(args):
    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8") if l.strip()]
    urls = sorted({t["web"] for t in tasks if t.get("web")})
    done_before = {}
    outp = Path(args.out)
    if args.resume and outp.is_file():
        for r in json.loads(outp.read_text(encoding="utf-8")):
            done_before[r["url"]] = r
        urls = [u for u in urls if u not in done_before]
    pool = []
    pf = Path(args.pool)
    if pf.is_file():
        pool = [r["port"] for r in json.loads(pf.read_text(encoding="utf-8")) if r.get("port")]
    print(f"待探 {len(urls)} 个 URL(已有 {len(done_before)}),代理池 {len(pool)} 个出口,"
          f"兜底出口 {args.proxy_url or '无(直连)'},并发 {args.concurrency}", flush=True)

    q = asyncio.Queue()
    for u in urls:
        q.put_nowait(u)
    out = list(done_before.values())
    fresh = []
    lock = asyncio.Lock()
    done = [0]
    async with async_playwright() as pw:
        ws = [asyncio.create_task(worker(i, q, pw, pool, args.timeout, fresh, lock,
                                         len(urls), done, args.proxy_url))
              for i in range(args.concurrency)]
        await asyncio.gather(*ws)
    out += fresh
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    c = Counter(x["verdict"] for x in out)
    print(f"\n=== 探测结果({len(out)} 个 URL)===")
    for k, v in c.most_common():
        print(f"  {k:10}{v:5}  {100*v/max(len(out),1):5.1f}%")
    bad = {x["url"] for x in out if x["verdict"] in ("blocked", "error", "empty")}
    keep = [t for t in tasks if t.get("web") not in bad]
    print(f"\n任务表 {len(tasks)} → 可达 {len(keep)} ({100*len(keep)/max(len(tasks),1):.1f}%)")
    if args.write_tasks:
        Path(args.write_tasks).write_text(
            "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in keep), encoding="utf-8")
        print(f"  → {args.write_tasks}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="runs/op_tasks_v3/tasks.jsonl")
    ap.add_argument("--out", default="runs/site_reachability.json")
    ap.add_argument("--write-tasks", default="runs/op_tasks_v3/tasks_reachable.jsonl")
    ap.add_argument("--pool", default="runs/proxy_pool.json")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=35)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--proxy-url", default=os.environ.get("TASK_PROXY_URL", ""),
                    help="代理池为空时的兜底出口;设空串则真的不走代理")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
