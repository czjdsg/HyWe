#!/usr/bin/env python3
"""控制器驱动的 CoT 泄露检测 —— 不用正则找泄露,用控制器自己的记录定位并比对。

## 为什么不用正则

正则总会漏,而且会误报。实测本批:
  * 正则(rewrite_leak_residue.LEAK)命中 335 步;
  * 其中 3 步是**误报** —— 原文里根本没有控制器内容,是 `驳回|rejected my answer`
    这类非控制器专属的模式触发的;
  * 漏的那部分无从知晓 —— 这正是正则方案的根本问题:没有真值可对。

## 控制器知道什么

`teacher_phase_control.json` 的每个 event 里有:
  feedback                    送给教师的**原文**
  feedback_delivered_to_step  送到了哪一步
所以不必猜"这段像不像泄露",可以直接问"**这段是不是在复述这句话**"。

## 候选集怎么定

一步只有在**收到过反馈之后**才可能泄露反馈内容。实测泄露离送达点的距离:
  +0 步 70.1% | 5 步内 92.2% | 12 步内 99.1%
默认窗口 --window 12(覆盖 99%),设 0 表示送达点之后全收(最保守但最贵)。

## 判官怎么问

**按轨迹批量**问,一条轨迹一次调用:把该轨迹送达过的全部 feedback 原文 +
候选步的 think 一起给出,让它逐步判"有没有复述控制器",并给出原句。
批量而非逐步是为了成本 —— 候选集比正则大 20 倍,逐步调用扛不住。
"""
from __future__ import annotations

import argparse, json, os, sys, time
import concurrent.futures as cf
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
import react_v2 as rv2  # noqa: E402

SYSTEM = """
You decide, for each reasoning step, whether it reveals the existence or content
of a private "phase controller" channel.

Background: during collection a controller ran in the background and sent the
agent feedback — whether the current phase was complete, whether it should switch,
that an answer had been rejected, and so on. At inference time that controller
DOES NOT EXIST. So if the agent's reasoning refers to it, a student trained on
this data will come to rely on something that is not there.

You are given:
  * every piece of feedback the controller actually sent in this trajectory;
  * the reasoning text of a number of steps.

For EACH step decide: does this reasoning cite, restate, or depend on that feedback?

Counts as a leak:
  "the controller says the current objective is complete" / "The controller
  feedback confirms…" / "as the controller instructed, switch now" /
  "my answer was rejected, so…" / "the system is telling me to switch"
  — the point is treating that external channel as a reason or an instruction.

Does NOT count as a leak:
  a conclusion the agent reached on its own, even if it happens to agree with the
  feedback ("the evidence is in, time to move to the search path");
  the word "controller" meaning something else (a game controller, a GDPR data
  controller, a job title);
  simply restating the task goal.

Output JSON only:
{"steps": [{"step": 12, "leak": true, "quote": "<the exact sentence from that step's text>",
            "why": "<one line>"}]}
List only steps where leak is true. Return {"steps": []} if there are none.
"""


def load(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def candidates(recs: list, deliveries: list[int], window: int) -> list[dict]:
    """送达点及其之后 window 步内的、有 think 的步。window<=0 表示之后全收。"""
    out = []
    for r in recs:
        st = r.get("step")
        if st is None:
            continue
        th = ((r.get("report_parsed") or {}).get("think")
              or rv2.extract_think(str(r.get("raw_response") or "")) or "")
        if not th.strip():
            continue
        prior = [d for d in deliveries if d is not None and d <= st]
        if not prior:
            continue
        if window > 0 and st - prior[-1] > window:
            continue
        out.append({"step": st, "think": th})
    return out


def one(cli, model, tid, feedbacks, cands, timeout, attempts):
    fb = "\n".join(f"- [送到第 {s} 步] {t}" for s, t in feedbacks) or "(无)"
    body = "\n\n".join(f"### step {c['step']}\n{c['think'][:2400]}" for c in cands)
    prompt = (f"## 控制器实际发出的反馈\n{fb}\n\n"
              f"## 待判的推理步({len(cands)} 步)\n{body}")
    msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    valid = {c["step"]: c["think"] for c in cands}
    err = "?"
    for _ in range(attempts):
        try:
            r = cli.chat.completions.create(model=model, messages=msg, timeout=timeout,
                                            temperature=0, response_format={"type": "json_object"})
            o = json.loads(r.choices[0].message.content)
            steps = o.get("steps")
            if not isinstance(steps, list):
                err = "steps 不是列表"
                continue
            out = []
            bad = None
            for x in steps:
                st = x.get("step")
                q = str(x.get("quote") or "")
                if st not in valid:
                    bad = f"step {st} 不在候选集里"
                    break
                # 反幻觉:引用必须逐字出现在该步文本里。审计那边踩过 —— 判官会把
                # 提示词里的字段名当引用填回来,凭空造出泄露步。
                if q and q not in valid[st]:
                    bad = f"step {st} 的引用不在原文里"
                    break
                out.append({"step": st, "quote": q, "why": str(x.get("why") or "")})
            if bad:
                err = bad
                continue
            return {"task_id": tid, "n_candidates": len(cands), "leaks": out}
        except Exception as e:
            err = str(e)[:120]
    return {"task_id": tid, "n_candidates": len(cands), "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--window", type=int, default=12,
                    help="送达点之后多少步内算候选(0=之后全收)。默认 12,覆盖实测 99.1%%")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=240)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in sorted(Path(a.rollout_dir).iterdir()):
        if not d.is_dir():
            continue
        if a.resume and (outd / f"{d.name}.json").is_file():
            continue
        res = load(str(d / "result.json"))
        if not res or res.get("termination") != "answer":
            continue
        side = load(str(d / "teacher_phase_control.json")) or {}
        evs = side.get("events") or []
        fbs = [(e.get("feedback_delivered_to_step"), str(e.get("feedback") or "").strip())
               for e in evs if e.get("feedback")]
        fbs = [(s, t) for s, t in fbs if t]
        deliveries = sorted({s for s, _ in fbs if s is not None})
        if not deliveries:
            continue                      # 从没发过反馈 -> 不可能泄露控制器内容
        cands = candidates(res.get("records") or [], deliveries, a.window)
        if cands:
            jobs.append((d.name, fbs, cands))
        if a.limit and len(jobs) >= a.limit:
            break

    tot_c = sum(len(c) for _, _, c in jobs)
    print(f"待判 {len(jobs)} 条轨迹,候选步 {tot_c}(窗口 {a.window}),模型 {a.model}", flush=True)
    cli = OpenAI(base_url=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
                 api_key=os.environ.get("JUDGE_API_KEY") or os.environ["API_KEY"])
    n_leak = 0; done = 0; t0 = time.time()
    with cf.ThreadPoolExecutor(a.max_workers) as ex:
        futs = [ex.submit(one, cli, a.model, tid, fbs, cands, a.timeout, a.attempts)
                for tid, fbs, cands in jobs]
        for f in cf.as_completed(futs):
            r = f.result()
            (outd / f"{r['task_id']}.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
            n_leak += len(r.get("leaks") or [])
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}  累计泄露步 {n_leak}  "
                      f"{(time.time()-t0)/done:.1f}s/条", flush=True)
    print(f"完成:{done} 条轨迹,泄露步 {n_leak} -> {outd}", flush=True)


if __name__ == "__main__":
    main()
