#!/usr/bin/env python3
"""把 phase-controlled teacher rollout 导出成 harness v2 的多轮训练数据。

一条 n 次 switch 的 hybrid 轨迹产出 **n+1 条样本**(不考虑折叠时)。

段的定义与推理端 `ReactHistory.segment_start()` 完全一致:段 = 两次**已执行**的
switch 之间的连续同侧区间。切端必须换系统提示词(S/O 两侧差异极大,19 个工具只
共享 2 个),而 chat_template 禁止 system 出现在非首位,所以只能在切端处重建 ——
这正是"n 次 switch → n+1 条样本"的由来。switch 那一步记在**来源侧**,所以它是
所属段的最后一个待预测 assistant;末段的最后一个 assistant 是 `<answer>`。

对齐三件事(与 build_aquila_v2 / build_molmoweb_v2 同一套规矩):
  1. 上下文 —— 直接调推理端同一个 `ReactHistory.render_multiturn`,不自己拼消息;
  2. 系统提示词 —— `react_v2_prompts.system_prompt(side)` + **完整**工具 schema。
     采集时相位控制器会在切换边界把 schema 缩成只剩 `switch`
     (`hybrid_agent._teacher_phase_allowed_tools`),但推理时没有控制器,所以
     导出必须用该侧的完整工具集,否则训推不一致;
  3. 截图 —— 采集时窗口被压到 10 帧只是为了活过 OSS 传输
     (`HYBRID_MAX_O_SCREENSHOTS`);原图全都落盘了,导出按 `--max-o-screenshots`
     重渲染,默认 -1(全量),与推理端默认一致。

用法:
    python training/build_hybrid_teacher_v2.py \
      --rollout-dir runs/<run>/rollouts \
      --out runs/<run>/train/hybrid_teacher_v2.jsonl \
      --judge-dir runs/<run>/judge          # 可选,按判官结论过滤
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

import react_v2 as rv2                       # noqa: E402
import react_v2_prompts as rv2p              # noqa: E402
from hybrid_prompts import TOOLS, SWITCH_TOOL, get_allowed_tool_names  # noqa: E402

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


# --------------------------------------------------------------- system prompt

_SYS_CACHE: dict[tuple[str, str], str] = {}


_COLLECT_DATE_RE = re.compile(r"\nToday is: ([^\n]+)\n")


def system_text(side: str, current_date: str) -> str:
    """该侧的 v2 系统提示词 + **完整**工具 schema —— 推理时模型看到的那一份。

    不复用采集期的 schema:相位控制器在切换边界只放行 `switch`,那是采集期的门控,
    推理端不存在。用受限 schema 训练会让模型在两种系统提示词下工作。
    """
    key = (side, current_date)
    if key in _SYS_CACHE:
        return _SYS_CACHE[key]
    allowed = set(get_allowed_tool_names("hybrid_react_v2", side))
    schema = [
        {"name": (t.get("function") or {}).get("name"),
         "description": (t.get("function") or {}).get("description"),
         "parameters": (t.get("function") or {}).get("parameters")}
        for t in TOOLS + [SWITCH_TOOL]
        if (t.get("function") or {}).get("name") in allowed
    ]
    _SYS_CACHE[key] = (
        rv2p.system_prompt(side, current_date, expose_switch=True)
        + "\n\n# Tool Schemas\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )
    return _SYS_CACHE[key]


def flatten(msgs: list[dict]) -> list[dict]:
    """纯文本的 content 列表压成字符串;带图的原样保留(同 build_aquila_v2)。"""
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            if all(isinstance(x, dict) and x.get("type") == "text" for x in c):
                c = "".join(x.get("text", "") for x in c)
        out.append({"role": m["role"], "content": c})
    return out


# ------------------------------------------------------------------ 轨迹重建

def image_ref(task_dir: Path, path_value: str, task_id: str, mode: str,
              out_dir: Path) -> str:
    """把一张落盘截图变成样本里引用的地址。

    abs      : 直接用绝对路径,零额外磁盘(默认 —— 输出盘紧张时唯一安全的选择)
    symlink  : images/<task_id>/<name> 软链,给出可移植的相对布局且几乎不占空间
    copy     : 真拷贝,自包含但每任务约 8MB
    """
    if not path_value:
        return ""
    src = Path(path_value)
    if not src.is_absolute():
        src = (task_dir / src).resolve()
    if not src.is_file():
        return ""
    if mode == "abs":
        return str(src)
    rel = Path("images") / task_id / src.name
    dst = out_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        if mode == "symlink":
            os.symlink(src, dst)
        else:
            import shutil
            shutil.copy2(src, dst)
    return str(rel)


def build_history(result: dict, task_dir: Path, task_id: str, args,
                  out_dir: Path, cot_map: dict | None = None,
                  merged_answer: str | None = None) -> tuple[rv2.ReactHistory, list[dict], list[dict], Counter]:
    """从落盘 records 重建 ReactHistory,返回 (hist, steps, step_records, stats)。

    `step_records` 与 `steps` 一一对齐 —— 渲染入场截图时要回查原始 record 的
    `img_path` / `current_url`,而 steps 里只留了渲染用的字段。

    与 `hybrid_agent` 里那四处 `add_step` 一一对应:
      * 普通工具步        -> action = json.dumps(tool_call),带执行后截图
      * 已执行的 switch    -> switch_executed=True(段边界)
      * 被控制器驳回的 switch -> switch_executed=False(**不**开新段)
      * 答案步            -> action_kind="answer"
    """
    hist = rv2.ReactHistory(max_o_screenshots=args.max_o_screenshots,
                            max_input_tokens=args.max_input_tokens)
    stats = Counter()
    step_records: list[dict] = []
    for rec in result.get("records") or []:
        side = rec.get("active_mode_before") or "S"
        raw = str(rec.get("raw_response") or "")
        think = ((rec.get("report_parsed") or {}).get("think")
                 or rv2.extract_think(raw) or "")
        # CoT 泄露改写:采集期教师常在 <think> 里直说"按控制器反馈…",而学生
        # 推理时没有那条私有信道。改写产物按 (task_id, step) 覆盖,原 result.json
        # 不动 —— 想比对或回退,原文始终在。
        if cot_map:
            fixed = cot_map.get(str(rec.get("step")))
            if isinstance(fixed, dict) and fixed.get("think"):
                think = fixed["think"]
                stats["cot_rewritten"] += 1
        tool_call = rec.get("tool_call")
        observation = str(rec.get("observation") or "")

        if isinstance(tool_call, dict) and tool_call:
            # switch 的 reason 参数不进训练数据:它和 <think> 里的理由是同一段话的
            # 两份拷贝,而且 CoT 改写覆盖不到它 —— 历史轨迹里 F1 有 70 条 reason 仍在
            # 提控制器,原样导出等于把改写掉的泄露又从工具参数里放回去。
            if str(tool_call.get("name") or "") == "switch" and tool_call.get("arguments"):
                tool_call = {**tool_call, "arguments": {}}
                stats["stripped_switch_reason"] += 1
            action = json.dumps(tool_call, ensure_ascii=False)
            kind = "tool"
        else:
            # 末步:接受的答案没有 tool_call,原始输出里是 <answer>…</answer>。
            # 未闭合的 <answer>(模型写完答案没写闭合标签)要一并接住 ——
            # 运行时解析器 2026-09-01 已经这么改了,导出侧当时漏了同步,结果
            # 151 条样本里 14 条末段停在 visit/click/search 这类普通工具上,
            # 变成"有动作没收口"的残段。判据同 react_v2:开标签之后没有闭标签时,
            # 后面全部就是答案,没有歧义。
            # JSON 输出模式(HYBRID_JSON_OUTPUT=1)下答案在 {"cot":…,"answer":…}
            # 里,**没有 <answer> 标签**,只按标签找会整步跳过 —— 表现为末段停在
            # visit/click 上、整条轨迹没有 answer 段。2026-09-05 实测:新两批
            # 606/608 条轨迹的 answer 段就是这么丢的(老批是标签模式采的,没中招)。
            # 先试 JSON,再退回标签。
            action = ""
            try:
                _j = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
                if isinstance(_j, dict) and str(_j.get("answer") or "").strip():
                    action = str(_j["answer"]).strip()
                    stats["answer_from_json"] += 1
            except Exception:
                pass
            if not action:
                m = ANSWER_RE.search(raw) or rv2.OPEN_ANSWER_RE.search(raw)
                if not m:
                    stats["skipped_unparsable_step"] += 1
                    continue
                action = m.group(1).strip()
            # 跨阶段补写:阶段控制器是双盲的,教师在最后一段作答时只看得到当前
            # 阶段目标,答案往往只覆盖最后一个组件。补写产物把前序阶段的发现并
            # 回来(每个值都回验过出处)。不接这里的话,补写就白做了 —— 上一轮
            # 53 条补写答案就是因为导出侧没有这个入口,从没进过训练数据。
            if merged_answer:
                action = merged_answer.strip()
                stats["answer_merged"] += 1
            kind = "answer"

        # 观测里的截图:O 侧动作执行后的那张(与实跑时 image_url 的来源一致)。
        shot = ""
        if side == "O":
            shot = image_ref(task_dir, rec.get("post_shot_path") or rec.get("img_path") or "",
                             task_id, args.image_mode, out_dir)

        switch_executed = bool(rec.get("switch_executed", True))
        if rec.get("is_switch_step") and not switch_executed:
            stats["rejected_switch"] += 1
        if rec.get("answer_rejected_by_phase_controller"):
            # 整步丢弃。教师在任务中途交了答案、被阶段控制器驳回,紧接着才去 switch。
            # 把这一轮留作训练目标等于教模型"做到一半可以先交卷" —— 而推理时没有
            # 控制器驳回它,直接就答错收工。留在历史里也不行:那是个坏示范。
            # 删掉不破坏因果链,被驳回的那步本来也没改变页面状态。
            stats["dropped_rejected_answer"] += 1
            continue
        if rec.get("received_teacher_phase_feedback"):
            stats["got_controller_feedback"] += 1

        hist.add_step(side, action, observation, think=think, image_url=shot,
                      switch_executed=switch_executed, action_kind=kind)
        step_records.append(rec)

    steps = [t for t in hist.turns if t.get("role") == "step"]
    return hist, steps, step_records, stats


def entry_shot_for(step_records: list[dict], seg0: int, task_dir: Path,
                   task_id: str, args, out_dir: Path) -> str:
    """一个 O 段刚接手时模型看到的那张图。

    实跑时(`hybrid_agent` 渲染分支)只在"本段还没有任何动作"时把**当前页面**
    截图递给渲染器,渲染器把它钉在段内不变。那张图就是该段第一步的动作**前**
    截图,即 `records[seg0]["img_path"]`。页面停在 about:blank 时不传 —— 纯白图
    只会骗模型去点空白,渲染器改发一行"先 goto"的文字提示。
    """
    if seg0 >= len(step_records):
        return ""
    rec = step_records[seg0]
    url = str(rec.get("current_url") or "").strip()
    path_value = rec.get("img_path") or ""
    if not path_value or url == "about:blank":
        return ""
    return image_ref(task_dir, path_value, task_id, args.image_mode, out_dir)


def segment_bounds(steps: list[dict]) -> list[tuple[int, int]]:
    """段边界 [(start, end), ...]。与 `ReactHistory.segment_start()` 同一判据。

    只有**已执行**的 switch 才开新段;被驳回的 switch 留在原段里(它没有换侧,
    系统提示词也没变)。所以 n 次已执行 switch -> n+1 段。
    """
    cuts = [0]
    for i, t in enumerate(steps):
        if (rv2.ReactHistory._tool_name_of(t.get("action") or "") == "switch"
                and t.get("switch_executed", True)):
            cuts.append(i + 1)
        elif i and t.get("side") != steps[i - 1].get("side"):
            if cuts[-1] != i:
                cuts.append(i)
    cuts = sorted(set(c for c in cuts if c < len(steps)))
    return [(c, cuts[k + 1] if k + 1 < len(cuts) else len(steps))
            for k, c in enumerate(cuts)]


def build_samples(task_id: str, result: dict, task_dir: Path, args,
                  out_dir: Path, cot_map: dict | None = None,
                  merged_answer: str | None = None) -> tuple[list[dict], Counter]:
    hist, steps, step_records, stats = build_history(
        result, task_dir, task_id, args, out_dir, cot_map, merged_answer)
    if not steps:
        stats["no_steps"] += 1
        return [], stats

    question = str(result.get("question") or "").strip()
    # 采集当天的日期在 rollout 自己的 system_prompt 里(运行时无条件写 Today is:)。
    # 以前这里只用 args.current_date,而它默认空串、也没人传 —— 于是导出的样本
    # 整体丢掉日期行,而推理端的日期行是必然生成的,训推就此不一致;题面里
    # today/this weekend 这类相对时间也失去了锚点。默认改成从 rollout 里取,
    # --current-date 显式给值时仍然覆盖它。
    cur_date = args.current_date
    if not cur_date:
        for _r in step_records:
            _m = _COLLECT_DATE_RE.search(str(_r.get("system_prompt") or ""))
            if _m:
                cur_date = _m.group(1).strip()
                break
    bounds = segment_bounds(steps)
    samples = []
    for seg_index, (seg0, seg_end) in enumerate(bounds):
        if seg_end <= seg0:
            continue
        side = steps[seg0].get("side") or "S"
        # 单独渲染一段:构造一个只含 steps[:seg_end] 的历史,再指定段起点。
        # 这样 prior = steps[:seg0](历史块),cur = steps[seg0:seg_end](本段)。
        sub = rv2.ReactHistory(max_o_screenshots=args.max_o_screenshots,
                               max_input_tokens=args.max_input_tokens)
        sub.turns = steps[:seg_end]
        if side == "O":
            # 钉住入场图。渲染器只在"本段还没有动作"时自己去取(`if not cur`),
            # 而导出时整段的步骤都在,那个分支永远不进 —— 不显式钉的话,推理端
            # 每个 O 段第一轮都有的这块在训练数据里会整体缺失。
            sub._seg_shot = (seg0, entry_shot_for(step_records, seg0, task_dir,
                                                  task_id, args, out_dir))
        msgs = sub.render_multiturn(
            system_text(side, cur_date), side, question,
            seg_start=seg0, drop_last_observation=True,
        )
        if not msgs or msgs[-1].get("role") != "assistant":
            stats["bad_tail"] += 1
            continue
        last = steps[seg_end - 1]
        samples.append({
            "messages": flatten(msgs),
            "task_id": task_id,
            "segment": seg_index,
            "segment_of": len(bounds),
            "side": side,
            "seg_steps": seg_end - seg0,
            "ends_with": ("answer" if last.get("action_kind") == "answer"
                          else rv2.ReactHistory._tool_name_of(last.get("action") or "") or "tool"),
        })
    stats["switches"] += len(bounds) - 1
    stats["samples"] += len(samples)
    return samples, stats


# ------------------------------------------------------------------- 判官过滤

def load_judgments(judge_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not judge_dir:
        return out
    for p in judge_dir.glob("tasks/*/judgment.json"):
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if o.get("status") != "ok":
            continue
        j = o.get("judgment") or {}
        out[str(o.get("task_id"))] = {
            "trajectory_success": bool(j.get("trajectory_success")),
            "cot_leakage": bool(j.get("cot_leakage")),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--judge-dir", type=Path, default=None,
                    help="判官输出目录;给了就按 --require-success / --drop-leakage 过滤")
    ap.add_argument("--require-success", action="store_true",
                    help="只导出 trajectory_success=true 的轨迹(需要 --judge-dir)")
    ap.add_argument("--drop-leakage", action="store_true",
                    help="丢掉 cot_leakage=true 的轨迹(需要 --judge-dir)")
    ap.add_argument("--terminations", default="answer",
                    help="逗号分隔;默认只要跑出答案的。设 all 则不筛")
    ap.add_argument("--current-date", default="",  # 留空=从 rollout 的 system_prompt 里取采集当天
                    help="写进系统提示词的日期,留空则不写(与采集时一致最好)")
    ap.add_argument("--max-o-screenshots", type=int, default=-1,
                    help="-1=全量保留(默认,与推理端一致)。采集期的 10 帧窗口只是"
                         "传输限制,原图都在盘上")
    ap.add_argument("--max-input-tokens", type=int, default=10 ** 9,
                    help="默认极大 = 不折叠。要按窗口切段请另行处理")
    ap.add_argument("--image-mode", choices=["abs", "symlink", "copy"], default="abs")
    ap.add_argument("--cot-rewrite-dir", type=Path, default=None,
                    help="CoT 泄露改写产物目录(每任务一个 <task_id>.json,键是步号)")
    ap.add_argument("--answer-merge-dir", type=Path, default=None,
                    help="跨阶段答案补写产物目录(每任务一个 <task_id>.json,取 answer 字段)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    judgments = load_judgments(args.judge_dir) if args.judge_dir else {}
    if (args.require_success or args.drop_leakage) and not judgments:
        ap.error("--require-success / --drop-leakage 需要有效的 --judge-dir")

    wanted = None if args.terminations.strip() == "all" else {
        s.strip() for s in args.terminations.split(",") if s.strip()
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_dir = args.out.parent

    task_dirs = sorted(d for d in args.rollout_dir.iterdir()
                       if d.is_dir() and (d / "result.json").is_file())
    if args.limit:
        task_dirs = task_dirs[:args.limit]

    tot = Counter()
    per_switch = Counter()
    n_written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for d in task_dirs:
            tot["tasks_seen"] += 1
            try:
                result = json.loads((d / "result.json").read_text(encoding="utf-8"))
            except Exception:
                tot["unreadable"] += 1
                continue
            term = result.get("termination")
            if wanted is not None and term not in wanted:
                tot[f"skip_termination_{term}"] += 1
                continue
            task_id = str(result.get("task_id") or d.name)
            verdict = judgments.get(task_id)
            if judgments and verdict is None:
                tot["skip_unjudged"] += 1
                continue
            if args.require_success and not verdict["trajectory_success"]:
                tot["skip_judge_failed"] += 1
                continue
            if args.drop_leakage and verdict["cot_leakage"]:
                tot["skip_leakage"] += 1
                continue

            cot_map = None
            if args.cot_rewrite_dir:
                f = args.cot_rewrite_dir / f"{task_id}.json"
                if f.is_file():
                    try:
                        cot_map = json.loads(f.read_text(encoding="utf-8"))
                    except Exception:
                        cot_map = None
            merged_answer = None
            if args.answer_merge_dir:
                f = args.answer_merge_dir / f"{task_id}.json"
                if f.is_file():
                    try:
                        mo = json.loads(f.read_text(encoding="utf-8"))
                        # 补写失败的记录里只有 error,没有 answer —— 那种就用原答案
                        a = mo.get("answer") if isinstance(mo, dict) else None
                        if isinstance(a, str) and a.strip():
                            merged_answer = a
                        elif isinstance(a, (dict, list)) and a:
                            # 模型偶尔按 answer_schema 直接返回结构体而不是成稿。
                            # str() 会变成 Python repr(单引号),序列化成 JSON 才可读。
                            merged_answer = json.dumps(a, ensure_ascii=False, indent=2)
                    except Exception:
                        merged_answer = None
            samples, stats = build_samples(task_id, result, d, args, out_dir,
                                           cot_map, merged_answer)
            tot.update(stats)
            if not samples:
                tot["tasks_no_sample"] += 1
                continue
            tot["tasks_exported"] += 1
            per_switch[len(samples) - 1] += 1
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"导出 {n_written} 条样本 → {args.out}")
    print(f"  轨迹: 扫描 {tot['tasks_seen']}  导出 {tot['tasks_exported']}  "
          f"无样本 {tot['tasks_no_sample']}")
    print(f"  switch 总数 {tot['switches']}  "
          f"→ 期望样本 {tot['switches'] + tot['tasks_exported']}  实际 {n_written}  "
          f"{'✓ 一致' if n_written == tot['switches'] + tot['tasks_exported'] else '✗ 不一致'}")
    if per_switch:
        print("  按 switch 次数的轨迹分布:")
        for k in sorted(per_switch):
            print(f"    {k} 次 switch → {k+1} 条/轨迹 : {per_switch[k]} 条轨迹")
    noteworthy = {k: v for k, v in tot.items()
                  if k.startswith("skip_") or k in
                  ("rejected_switch", "rejected_answer", "got_controller_feedback",
                   "bad_tail", "skipped_unparsable_step", "unreadable")}
    if noteworthy:
        print("  其他计数:", dict(sorted(noteworthy.items())))
    if tot["got_controller_feedback"]:
        print(f"  ⚠ {tot['got_controller_feedback']} 步收到过相位控制器反馈,其观测文本"
              f"已原样保留;推理端没有控制器,后续清洗时需处理")


if __name__ == "__main__":
    main()
