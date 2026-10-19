#!/usr/bin/env python3
"""对导出后仍残留私有信道措辞的 <think> 做定向二次改写。

为什么需要这一道:第一遍 `rewrite_cot_leaks.py` 的活儿来自审计的 `cot_leaks`,
审计漏掉的它就看不见。实测 8500 批导出后仍有约 20 条轨迹在 think 里直说
"The controller feedback says…""The controller is telling me…" —— 审计没标,
所以从没进过改写队列。

与第一遍的关键差别:**在生效的那版 think 上改**(有第一遍产物就用它),而不是回到
result.json 的原文。否则会把第一遍的成果覆盖掉。

"controller" 这个词有无辜义项(游戏手柄 Full Controller Support、GDPR 的 data
controller、MIDI controller),所以命中词表之外还要求出现私有信道的动词搭配。
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, json, os, re, sys, time
from collections import Counter
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rewrite_cot_leaks import SYSTEM  # 同一套改写口径,不另起炉灶

# 私有信道的判据:controller/控制器 + 指令性动词,或直白的"被驳回/反复交卷"自述。
LEAK = re.compile(
    r"(?:\bcontroller\b|控制器)\s*(?:feedback|instruction|is|was|says?|said|keeps?|wants?|"
    r"tells?|told|asks?|asked|requires?|confirms?|now|earlier|反馈|说|要求|让我|告诉)"
    r"|per the controller|according to the controller|the controller'?s"
    r"|驳回|rejected my answer|not accepting my answer|keeps sending back"
    r"|already (?:ended|answered|submitted)[^.]{0,40}(?:twice|multiple times|many times)"
    r"|<answer> tags (?:many|multiple) times"
    r"|phase goal|current phase objective|rollout 目标|当前阶段目标",
    re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)*")


def sentences(t: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?。!?\n])\s+", t) if s.strip()]


def verify(old: str, new: str) -> str | None:
    if not new.strip():
        return "改写为空"
    if LEAK.search(new):
        return "改写后仍含私有信道措辞"
    if len(new) < 0.35 * len(old):
        return "内容大幅缩水"
    fresh = set(NUM.findall(new)) - set(NUM.findall(old))
    if fresh:
        return f"引入了原文没有的数值 {sorted(fresh)[:4]}"
    return None


def one(cli, model, question, think, quotes, ctx, timeout, attempts):
    prompt = (f"## 任务\n{question}\n\n## 上下文(上一步的动作与观测)\n{ctx}\n\n"
              f"## 需要改写的思维链全文\n{think}\n\n"
              f"## 其中泄露私有信道的句子\n" + "\n".join(f"- {q}" for q in quotes) +
              '\n\n只输出 JSON:{"think": "<改写后的完整思维链>"}')
    convo = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    last = None
    for i in range(attempts):
        try:
            r = cli.chat.completions.create(model=model, messages=convo, timeout=timeout,
                                            max_tokens=4000, temperature=0.0)
            txt = r.choices[0].message.content or ""
            out = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
            new = str(out.get("think") or "")
            bad = verify(think, new)
            if not bad:
                return new
            last = bad
            convo = convo[:2] + [{"role": "assistant", "content": txt},
                                 {"role": "user", "content": f"{bad}。请修正后重新输出 JSON。"}]
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:160]
            time.sleep(min(8.0, 1.5 * (2 ** i)))
    return None if last else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=Path, default=None,
                    help="要扫描的 task_id 清单 JSON。**不给就扫 rollout-dir 下全部轨迹** —— "
                         "清单本身是漏洞来源:它要么来自审计(会漏标)、要么来自 survivors 快照"
                         "(与只增不减的导出集合漂移)。2026-09-02 连修三层都是这个模式,"
                         "所以默认全扫:正则是纯本地的,8541 步的成本可以忽略。")
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--cot-dir", type=Path, required=True, help="第一遍改写产物,就地更新")
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--leaks-dir", type=Path, default=None,
                    help="detect_leaks_by_controller.py 的产物目录。给了就用它定位泄露步,"
                         "**不再跑正则** —— 正则会漏(没有真值可对)也会误报(实测 335 步里"
                         "3 步是 `驳回|rejected my answer` 这类非控制器专属模式触发的)。"
                         "控制器自己记着反馈原文和送达步号,是可对的真值。")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("API_KEY")
    if not key:
        sys.exit("Set JUDGE_API_KEY")

    if args.ids and args.ids.is_file():
        ids = json.loads(args.ids.read_text(encoding="utf-8"))
    else:
        ids = sorted(d.name for d in Path(args.rollout_dir).iterdir()
                     if d.is_dir() and (d / "result.json").is_file())
        print(f"未给清单 → 全扫 {len(ids)} 条轨迹", flush=True)
    jobs = []
    for t in ids:
        rp = Path(args.rollout_dir) / t / "result.json"
        cp = args.cot_dir / f"{t}.json"
        if not rp.is_file():
            continue
        try:
            recs = json.loads(rp.read_text(encoding="utf-8")).get("records") or []
        except Exception:
            continue
        cot = {}
        if cp.is_file():
            try:
                cot = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                cot = {}
        q = ""
        try:
            q = str(json.loads(rp.read_text(encoding="utf-8")).get("question") or "")
        except Exception:
            pass
        # 给了 --leaks-dir 就用控制器驱动的检测结果定位,不再跑正则。
        det = {}
        if args.leaks_dir:
            f = args.leaks_dir / f"{t}.json"
            if f.is_file():
                try:
                    for x in (json.loads(f.read_text(encoding="utf-8")).get("leaks") or []):
                        det.setdefault(int(x["step"]), []).append(str(x.get("quote") or ""))
                except Exception:
                    det = {}
        for i, rec in enumerate(recs):
            st = rec.get("step")
            fixed = cot.get(str(st))
            think = (fixed.get("think") if isinstance(fixed, dict) and fixed.get("think")
                     else ((rec.get("report_parsed") or {}).get("think") or ""))
            if not think:
                continue
            if args.leaks_dir:
                if st not in det:
                    continue
                # 判官给的引用已经逐字校验过在原文里;取不到就退回整句扫描
                quotes = [q for q in det[st] if q and q in think][:6] or \
                         [x for x in sentences(think) if LEAK.search(x)][:6]
            else:
                if not LEAK.search(think):
                    continue
                quotes = [x for x in sentences(think) if LEAK.search(x)][:6]
            prev = recs[i - 1] if i else {}
            ctx = (f"上一步动作: {json.dumps(prev.get('tool_call'), ensure_ascii=False)[:300]}\n"
                   f"上一步观测: {str(prev.get('observation') or '')[:600]}")
            jobs.append((t, st, q, think, quotes, ctx))
    print(f"命中 {len(jobs)} 步,涉及 {len({j[0] for j in jobs})} 条轨迹", flush=True)
    if args.dry_run:
        for t, st, _, _, qs, _ in jobs[:15]:
            print(f"  {t[:22]} step {st}: {qs[0][:110] if qs else ''}")
        return

    cli = OpenAI(base_url=args.base_url, api_key=key)
    stats = Counter(); updates = {}
    with cf.ThreadPoolExecutor(args.max_workers) as ex:
        futs = {ex.submit(one, cli, args.model, q, th, qs, ctx, args.timeout, args.attempts): (t, st)
                for t, st, q, th, qs, ctx in jobs}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            t, st = futs[fut]; new = fut.result()
            stats["通过" if new else "失败"] += 1
            if new:
                updates.setdefault(t, {})[str(st)] = new
            if n % 10 == 0:
                print(f"  {n}/{len(jobs)} {dict(stats)}", flush=True)

    for t, m in updates.items():
        cp = args.cot_dir / f"{t}.json"
        cur = {}
        if cp.is_file():
            try:
                cur = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                cur = {}
        for st, th in m.items():
            cur[st] = {"think": th, "source": "leak_residue_pass2"}
        cp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果: {dict(stats)},更新 {len(updates)} 个文件")


if __name__ == "__main__":
    main()
