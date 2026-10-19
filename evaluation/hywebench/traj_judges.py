#!/usr/bin/env python3
"""三个判官的提示词与调用。全量输入(完整 think + 完整观测 + 每步截图 + 答案正文)。

判官分三个而不是一个大而全的:
  exclusive  这条轨迹对应的任务,是不是**必须开浏览器交互**才能做
  outcome    这条轨迹**有没有把任务做成**
  process    哪些步**不该被学**

分开的理由是它们的判据互相独立,合在一起会互相污染:问"做成没有"时提到过程问题,
判官会倾向说没做成(实测过程审计给的 premature_answer 里,只有 30% 在看到答案和
页面后仍然成立)。三个独立判决交叉,才能看出一致性 —— 上一轮实测:过程零问题的
1264 条,结果判官独立地判 100% 成功。
"""
from __future__ import annotations
import json, time

# ---------------------------------------------------------------- 1. O 独有性

# 提示词语言:2026-09-02 起判据与正文骨架统一为英文。
# **例外**:现存的 exclusive 判决(molmoweb 8,311 条 / webchain 19,257 条)是中文提示词
# 跑出来的,按用户决定**不重跑** —— 阶段一判的是"这题要不要交互"、四选一、判据简单,
# 抽查过且分布与数据集特性吻合(molmoweb 52.9% / webchain 80% operation_required),
# 换语言重跑大概率只是噪声级变动,不值 7 小时。review / decide / CoT 重写全部用英文。

EXCLUSIVE = """You decide whether the task a web-agent trajectory solves **genuinely requires
browser interaction** to complete.

You are given: the task, and for each step its action, the agent's thought, the execution
result and the screenshot taken before that step, plus the final answer.

**Read the URL trail together with the actions** — the two together are the clearest signal:
  * The URL only ever navigates (home -> section -> detail), the actions are just
    goto / scroll / clicking links, and the answer is printed on one of those pages —
    that is **pure navigation plus shallow reading**, not an operation task.
  * The URL keeps changing query/params on the same address, or does not change at all
    while the page state does (expanding, filtering, paging, switching tabs), and the
    actions include type / select_option / drag / checking boxes — that is real interaction.

Pick exactly one:
- "operation_required"  The key value in the answer only appears on the page **after**
  applying a filter, submitting input, or advancing state. Without those interactions you
  cannot get it. Session state changes (add to cart, set default, switch language,
  bookmark, place an order) always belong here — the product is changed state, which no
  search engine can produce.
- "shallow_read"  Pure navigation plus reading shallow page content. Opening the target
  page (or clicking a few links to reach it) is enough to read the answer; the browser is
  merely a reader. **Even if the agent clicked a lot**, if those clicks were not necessary
  to obtain the answer, it belongs here.
- "search_coverable"  A search engine answers this fact directly; there is no need to open
  this site at all (common knowledge, encyclopedic facts, a company's public information,
  paper titles and authors, and the like).
- "unclear"  Not enough information to decide.

Rules of thumb:
- Judge whether **the question itself** needs interaction, not how many times the agent
  clicked. A roundabout route does not make a task interactive; a shortcut does not make
  it non-interactive.
- shallow_read vs search_coverable: the former's answer lives on this site's pages and a
  search engine may not have indexed it; the latter a search engine gives you outright.
  When unsure, choose shallow_read.
- When torn between operation_required and shallow_read, choose shallow_read (the
  conservative call).

Output strict JSON and nothing else:
{"verdict":"operation_required|shallow_read|search_coverable|unclear",
 "proto":"the closest O-prototype id, or an empty string if this is not an O task",
 "url_pattern":"one phrase describing the shape of the URL trail (<=12 words)",
 "why":"reason for the verdict (<=30 words)"}"""

# ---------------------------------------------------------------- 2. 任务结果

OUTCOME = """你判定一条网页 agent 轨迹**有没有把任务完成**。

给你:任务、每一步的动作 + 想法 + 执行结果 + 执行前的页面截图、最终答案正文。
截图上的红色准心是我们画的标注,页面本身没有,判定时别提它。

做法:
1. 先从任务描述里拆出**必须满足的要点**(要查的字段、要满足的筛选条件、要完成的操作)。
2. 逐条对照:该要点的依据在**截图或执行结果里实际出现过**吗?
   —— 只出现在想法或答案文字里、页面上找不到的,不算满足。
3. 给出判决。

verdict:
  success   所有要点都满足,答案有页面证据支撑
  partial   主要目标达成但有要点没满足(漏了一个筛选/只答了一半/某个数字无出处)
  fail      核心目标没达成(答错对象/答案与页面矛盾/根本没找到/页面报错就收口)

严格输出 JSON,不要额外文字:
{"verdict":"success|partial|fail",
 "key_points":["要点1","要点2"],
 "unmet":["没满足的要点,全满足就空数组"],
 "why":"判决理由(60字内)",
 "evidence":"支撑判决的截图/执行结果出处(40字内)"}"""

# ---------------------------------------------------------------- 3. 步级诊断

PROCESS = """你在审计一条**网页 agent 的训练轨迹**。这条轨迹会被当作正样本,模型逐步模仿它的
每一个动作和每一段想法。找出**不该被学**的步骤。

给你:任务、每一步的动作 + 想法 + 执行结果 + 执行前的页面截图、最终答案正文。
**截图上的红色准心/圆圈是我们事后画的标注**,标的就是该步动作的坐标落点,页面本身没有。
它是给你判断「这个坐标落在什么控件上」用的 —— 判 grounding 全靠它。结论里不要提这个圆圈。

**以截图和执行结果里实际存在的内容为准**,不要只信想法的自述:
想法说"我看到 82 首"而截图上确实有,那就是有证据;截图上没有,才是编的。

类别(可多选,最多报 3 个最严重的):
  premature_answer   证据不足就收口(答案里的关键事实在截图/执行结果里找不到依据)
  missed_constraint  收口时任务里的某个显式约束没满足
  wrong_target       操作对象错(点进错条目/错按钮/搜索词跑偏),且后面没纠正
  no_recovery        某步明显没生效,却原地重复同一招,不换别的办法
  misread            想法或答案把页面上的值读错了(看错行/看错列)
  overexplore        在与任务无关的页面上兜圈子
  worse_route        明显有更短的路(直接搜索/直接改 URL)却一路点过去
  grounding          红色准心落在空白处,或落在想法所说元素之外的东西上

重要豁免 —— 不算问题,不要报:
  * 连续滚屏找内容(哪怕连滚十几次),只要页面在往下走
  * 正常的探索性点击:点进去看一眼发现不对再退回来
  * 想法写得啰嗦

严格输出 JSON,没问题就报空数组:
{"issues":[{"cat":"类别","step":步号(整数),"why":"坏在哪(60字内)",
            "evidence":"你在哪一步的截图/执行结果里看到的依据(40字内)"}]}"""


def _parse(t, need):
    d, st, cands = 0, 0, []
    for i, ch in enumerate(t):
        if ch == "{":
            if d == 0: st = i
            d += 1
        elif ch == "}" and d:
            d -= 1
            if d == 0: cands.append(t[st:i + 1])
    for c in reversed(cands):
        if f'"{need}"' not in c: continue
        try: return json.loads(c, strict=False)
        except Exception: pass
    return None


def call(client, model, sys_prompt, content, need, attempts=3, max_tokens=800, err=None):
    """err: 传一个 list 进来,失败时把最后一次的原因写进去 —— 静默失败查不出原因,
    上一轮 39% 失败率就是因为这里把异常全吞了。"""
    # temperature 恒为 0:kimi-k3 只接受 0,给别的值会直接 400
    # (`Parameter 'temperature'=0.3 is not supported for kimi-k3 model.`)。
    # 之前重试时抬温度,导致**所有重试都必然失败** —— 首次抖动就等于永久失败,
    # 表现为"图多的轨迹 100% 判不出来",查了两轮才发现根因不在图片。
    last = ""
    for att in range(attempts):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": content}])
            txt = r.choices[0].message.content or ""
            o = _parse(txt, need)
            if o is not None: return o
            last = f"parse_fail: {' '.join(txt.split())[:160]}"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(3 + 4 * att)
    if err is not None: err.append(last)
    return None


# ---------------------------------------------------- 无 answer 步的数据集专用口径
#
# WebChain 这类数据集**一条 answer 步都没有**,而且截图全是"动作执行前"的观测 ——
# 最后一个动作做完页面变成什么样,数据里根本没采集。所以:
#   * 不能问"答案对不对"(没有答案);
#   * 也不能问"末屏是不是目标状态"(末屏是末动作执行**前**的画面)。
# 能问的只有:到最后一个动作为止,这条轨迹把任务推进到了哪一步。
OUTCOME_NOANS = """你判定一条网页 agent 轨迹**把任务推进到了什么程度**。

给你:任务、每一步的动作 + 想法 + 执行结果 + 执行该步之前的页面截图。
截图上的红色准心是我们画的标注,页面本身没有,判定时别提它。

**这个数据集有两个特点,判定时必须考虑**:
  1. 轨迹**没有收口步**,做完最后一个 GUI 动作就结束,不存在"给出答案"这一步。
     所以不要因为"没有作答"就判失败。
  2. 每张截图是执行该步**之前**的画面。**最后一个动作的结果没有被采集** ——
     你看不到它做完之后页面变成什么样,只能从任务目标和该动作本身推断。

做法:先从任务里拆出必须达成的要点,再逐条对照截图/执行结果里实际出现过的证据。

verdict:
  success   到最后一个动作为止,任务要点已经全部达成,或**只差这最后一个动作**
            (且该动作明显就是完成任务的那一步)
  partial   走在正确路径上,但离目标还差若干步(要点只满足了一部分)
  fail      走偏了:操作对象错、被困在无关页面、反复无效操作、或路径与任务目标无关

严格输出 JSON,不要额外文字:
{"verdict":"success|partial|fail",
 "key_points":["要点1","要点2"],
 "unmet":["到末动作为止仍未达成的要点,全达成就空数组"],
 "why":"判决理由(60字内)",
 "evidence":"支撑判决的截图/执行结果出处(40字内)"}"""

# 不要加"半路停下"这类类别:这个数据集**按构造**就是做完最后一个 GUI 动作即止,
# 加了会 100% 假阳(实测 3/3 全中)。轨迹推进到什么程度是 outcome 判官的事,
# process 只管步级缺陷。
PROCESS_NOANS = PROCESS.replace(
    """  premature_answer   证据不足就收口(答案里的关键事实在截图/执行结果里找不到依据)
  missed_constraint  收口时任务里的某个显式约束没满足""",
    """  missed_constraint  任务里的某个显式约束从头到尾没被满足(少设一个筛选/少选一个选项)""").replace(
    "、以及最终答案正文。", "。**这个数据集没有收口步**:轨迹做完最后一个 GUI 动作就结束,"
                          "最后一个动作的结果也没有被采集。这是数据集的采集方式,不是缺陷 —— "
                          "不要因为「没有作答」「在末步就停了」去报问题,只报步级缺陷。")

# ==================================================================== 合并判官
#
# 原来 outcome / process 是两个独立调用,结果**互相打架**:同一条轨迹
# outcome 判 success、process 在收口步报 premature_answer,实测 453 条。
# 抽读四条,两个判官对同一张截图读出了不同内容 —— 这不是尺度差异,是不自洽。
#
# 合成一个调用后:
#   * "任务完没完成"只有一个说法,没有 partial 这一档 —— 没完成就整条丢;
#   * `premature_answer`(证据不足就收口)和 `missed_constraint`(约束没满足)
#     不再是步级问题:它们本来就等于"任务没完成",归到 completed=false 里去;
#   * 步级 issues 只保留**不该被模仿的操作缺陷**,且只在 completed=true 时有意义。
REVIEW = """You review a web-agent trajectory and do two things at once: first decide
whether the task was completed, and if it was, point out which steps should not be imitated.

You are given: the task, and for each step its action, the agent's thought, the execution
result and the screenshot taken before that step, plus the final answer text.

## Part one: completed — was the task finished?

First break the task into the **points that must be met** (fields to look up, constraints
to satisfy, inputs to submit), then check each one against evidence that **actually appears**
in the screenshots or execution results.

Answer true only if all three hold:
  1. Every point is backed by a screenshot or execution result somewhere in the trajectory;
  2. The final answer agrees with that evidence — nothing invented, nothing borrowed from
     a different item and attributed to the target;
  3. Every explicit constraint in the task (count, time, ordering, filters) is satisfied.

Answer false when: only some points were done; a key fact in the answer cannot be found in
any screenshot; the agent operated on the wrong target; a click never took effect yet the
agent answered anyway; an explicitly required constraint was skipped.

**Reasonable differences in wording are not failures**: sensible rounding or truncation of
numbers, omitted units, an answer that adds background beyond what was asked, phrasing that
does not match the page verbatim — all of these are still true.

## Part two: issues — if it was completed, which steps should not be imitated?

Fill this only when completed=true (give an empty array when false — the whole trajectory
is dropped, so picking at it is pointless). Report **operation-level** defects only; do not
re-report "insufficient evidence" or "constraint not met" — those belong to part one.

**Report only intermediate steps that carry an action. The final action-less step (thought
plus answer text) is never the subject of an issue** — judging the task complete means the
answer stands, so faulting the closing step contradicts that. If you think the answer is
wrong, that is a false in part one, not an issue here.

  wrong_target   acted on the wrong thing (wrong entry, wrong button, search query drifted)
                 and never corrected it afterwards
  no_recovery    a step plainly had no effect, yet the agent repeated the same move in place
                 instead of trying something else
  overexplore    circling around pages unrelated to the task
  worse_route    an obviously shorter path existed (search directly, edit the URL) but the
                 agent clicked its way through instead
  misread        the thought misread a value on the page (wrong row, wrong column)
  leak           the thought mentions the annotation itself — crosshair, marker, red circle,
                 box, overlay. That annotation is ours; the real page has none and the model
                 being trained never sees it, so any thought that describes it is unusable
                 and must be rewritten. Flag it whenever you see such wording.

Do **not** judge whether the coordinate lands on the right element — that check has moved to
the rewrite stage, which sees the same crosshair and reports it there. Ignore it here.

Report at most 3, only the ones you are sure of; an empty array if there are none. Use the
step number from the screenshot caption.

Output strict JSON and nothing else:
{"completed":true,
 "key_points":["point 1","point 2"],
 "unmet":["points still unmet; empty array if all met"],
 "why":"reason for the completion verdict (<=30 words)",
 "evidence":"which screenshot / execution result supports it (<=20 words)",
 "issues":[{"step":3,"cat":"wrong_target","why":"<=12 words"}]}"""


# 无 answer 步的数据集(WebChain):没有收口步,末动作的结果也没采集。
REVIEW_NOANS = REVIEW.replace(
    """result and the screenshot taken before that step, plus the final answer text.""",
    """result and the screenshot taken before that step.

**This dataset has no closing step**: the trajectory ends after its last GUI action, there
is no "give the answer" turn, and **the result of that last action was never captured** —
you cannot see what the page looked like afterwards. That is how the data was collected,
not a defect: never mark it incomplete, and never raise an issue, because "no answer was
given" or "it stopped at the last step".""").replace(
    """Answer true only if all three hold:
  1. Every point is backed by a screenshot or execution result somewhere in the trajectory;
  2. The final answer agrees with that evidence — nothing invented, nothing borrowed from
     a different item and attributed to the target;
  3. Every explicit constraint in the task (count, time, ordering, filters) is satisfied.

Answer false when: only some points were done; a key fact in the answer cannot be found in
any screenshot; the agent operated on the wrong target; a click never took effect yet the
agent answered anyway; an explicitly required constraint was skipped.

**Reasonable differences in wording are not failures**: sensible rounding or truncation of
numbers, omitted units, an answer that adds background beyond what was asked, phrasing that
does not match the page verbatim — all of these are still true.""",
    """Answer true when, as of the last action, every point has been met and is backed by a
screenshot or execution result — **or when only that last action is left** and it is plainly
the step that completes the task (the points are all in place and the last action is exactly
the submit / confirm / open-the-target move).

Answer false when: only some points were done; the trajectory is still several steps short
of the goal; the agent operated on the wrong target; a click never took effect yet the agent
carried on; an explicitly required constraint was skipped.""")

STAGES = {
    "exclusive": (EXCLUSIVE, "verdict"),
    "review": (REVIEW, "completed"),
    # 下面两个是合并前的旧两段式,保留只为复现历史判决,新数据别用
    "outcome": (OUTCOME, "verdict"),
    "process": (PROCESS, "issues"),
}

# has_answer=False 时用这套;exclusive 两边通用(它判的是任务本身要不要交互)
STAGES_NOANS = {
    "exclusive": (EXCLUSIVE, "verdict"),
    "review": (REVIEW_NOANS, "completed"),
    "outcome": (OUTCOME_NOANS, "verdict"),
    "process": (PROCESS_NOANS, "issues"),
}


def stages_for(has_answer):
    return STAGES if has_answer else STAGES_NOANS
