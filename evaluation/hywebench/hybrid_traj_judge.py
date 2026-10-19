#!/usr/bin/env python3
"""通用 hybrid 轨迹判官:两阶段、数据集无关。MMInA / HWB / 其它 hybrid 轨迹共用。

被判的是一个**通用网页 agent**,不是网页导航 agent —— 它同时握着两条路:
  * 搜索侧:search / scholar / visit / code_sandbox,拿到的是文本结果;
  * 操作侧:真实浏览器,click / type / goto / scroll …,拿到的是页面截图和地址栏。
两条路都是正当手段,判官不该偏向任何一条,只看子目标有没有真达成。

## 为什么拆成两阶段

上一版让判官一边读轨迹一边自己从任务里拆子目标,有两个后果:
  1. **同一道题、不同轨迹拆出来的子目标不一样**,跨模型没法横比 —— 甲模型被拆出 2 个
     子目标全做完判 success,乙模型被拆出 4 个做完 3 个判 partial,差异一半来自拆法。
  2. 判官容易顺着轨迹"事后合理化":agent 做了什么就把什么当成子目标。

所以:
  阶段 A  只看**任务文本**拆一次子目标,每道题拆一次、固定下来,所有模型共用同一份清单。
  阶段 B  把这份清单**作为输入**交给判官,逐条到轨迹里找证据。

## 阶段 B 的核心判据

每个子目标要回答的不是"像不像做了",而是"证据在哪一步、是什么模态"。
(2026-09-22 起判官不再吐 evidence_kind / claimed_only 两个字段, 只留 done/first_step/evidence/correct/blocked)
这是 hybrid agent 最常见的失真(用 visit 读一读就当成在站上操作过了),
必须单独计量,不能混进 done 里。

总判决**不问判官**,由代码从逐条结果算出来 —— 判官只报事实,聚合口径由我们定,
这样换判官模型时聚合规则不变。
"""
from __future__ import annotations
import argparse, collections, concurrent.futures as cf, glob, hashlib, importlib.util, json, os, re, threading, time

_H = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(_H, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
JG = _load("traj_judges", "traj_judges.py")
BC = _load("blocked_criterion", "blocked_criterion.py")
TR = _load("traj_render", "traj_render.py")


# ─────────────────────────────────────────────────────────── 阶段 A:拆子目标
EXTRACT_SYS = """You break a task down into the sub-goals an agent must accomplish to complete it.

The task will be carried out by a general-purpose web agent that has two ways of working:
a search side (web search, scholarly search, opening a URL to read its content, running code)
which yields text, and an operation side (a real browser it can click, type into, scroll and
navigate) which yields rendered pages. Both are legitimate. You are not deciding which one
should be used — you are only naming what has to end up true.

## What counts as one sub-goal

One sub-goal is one thing the task asks for that could independently succeed or fail. Split
on that, not on how many steps it might take. Ignore boilerplate wrapped around the request
(reference links, output-format instructions, encouragement); those are not sub-goals.

Classify each one by **what has to be verified**, never by what it takes to get there:

  information  — verifying it means checking a fact or an answer. Mark it information even
                 when obtaining that fact could only have come from driving a site: filtering
                 a catalogue to find the cheapest item, submitting a query to read a value,
                 opening an item page to see its colour. How the agent had to reach the fact
                 is irrelevant; what gets checked is whether the fact is right.

  operation    — verifying it means checking that something was actually carried out: a
                 sequence of actions performed to achieve a function, a particular page state
                 reached, a submission made, an item placed somewhere, a setting changed.
                 There is no question and no answer here; what gets checked is that the end
                 state exists.

A useful test: write down what a verifier would compare against. If the comparison is against
a value ("is it black?", "which city?", "how many?"), it is information. If the comparison is
against a state of the world ("is that page showing results for these inputs?", "is the item
in the cart?"), it is operation.

## Fields

goal        one sentence naming what must end up true. Concrete and checkable.
kind        "information" or "operation"
site        context only: the site the task associates with this sub-goal, if it names one.
            Leave "" when the task does not. This is a hint for the verifier, not a
            requirement — a sub-goal is not failed merely for happening somewhere else.
acceptance  what a verifier should look for to call it done — the observable end state, in
            <=25 words. For an operation sub-goal, say what the site must be showing.
depends_on  ids of sub-goals whose answer this one needs (e.g. an action whose target is
            decided by an earlier question). [] if independent.

Output strict JSON and nothing else:
{"subgoals":[{"id":1,"goal":"...","kind":"information|operation","site":"","acceptance":"...","depends_on":[]}]}

Keep the list minimal and non-overlapping. Do not invent requirements the task does not state."""


# ─────────────────────────────────────────────────────── 阶段 B:对轨迹逐条核验
VERIFY_SYS = """You verify, against a recorded trajectory, which of a task's sub-goals the agent
actually accomplished.

The agent is a general-purpose web agent working on two sides. Treat both as equally valid:

  search side     search / scholarly search / opening a URL for its content / running code.
                  Its evidence is the returned text. Screenshots are not expected here.
  operation side  a real browser. Its evidence is the page state after each action: the
                  screenshot of the resulting page plus the address bar and page title.

Steps that only hand control between the two sides carry no evidence of their own.

## Only cite what the record contains

Many records contain NO images at all — a search-only agent has no browser. Whether a step
carries a screenshot is visible to you: the step either includes an image or it does not.

  * Never describe a step as showing a screenshot when that step carries no image.
  * Never describe what "the screenshot shows" unless you are looking at that image.
  * Quote evidence from the record verbatim. If a value, label or UI string does not appear
    in the record, it is not evidence — no matter how plausible it is for that site.

A sub-goal whose only support would be an imagined screenshot is NOT done.

For an **operation** sub-goal, a step that merely runs a search is never sufficient evidence:
finding an article about a feature is not performing it. Cite the step that actually performed
the action — a browser action, an HTTP request that submits the form/API call, or a
parameterized URL that applies the setting. Those alternative routes count as done; a search
result describing them does not.

## How to read the evidence

Read the address bar together with the screenshot. A screenshot can be captured before the
new page finished painting, so a page that still looks unchanged but whose address already
moved did take effect. Conversely, an address that never left the entry page means nothing
was submitted, however confident the agent's reasoning sounds.

Screenshots and returned text are authentic records. The agent's own reasoning and its final
answer are claims written by the agent — where a claim and the record disagree, the record
wins. Where the record is simply silent on something the answer states, you may accept the
answer for an information sub-goal, but never for an operation one.

An **operation** sub-goal is done only when the trajectory shows the designated site actually
driven to the required end state — the inputs entered, the submission made, the resulting page
reached. Reading about it from search results, opening a constructed URL merely to look at it,
or asserting the outcome in the answer does not satisfy it. If a `site` is listed with the
sub-goal, treat it as a hint about where this was meant to happen, not as a requirement:
the same end state reached elsewhere still counts. If the agent claims the end state and the
record does not show it anywhere, mark done=false.

An **information** sub-goal is done when the agent committed to the correct fact. Here the route
does not matter: the fact may have come from a search result, from a page the agent read, or
from a value it had to filter a catalogue to expose — any of those count equally. Judge
correctness from the record and from your own knowledge of the world; if you genuinely cannot
tell whether the stated fact is right, set correct=null rather than guessing.

Partial credit is not yours to give: each sub-goal is done or not.

{CRITERION}

## Output

For every sub-goal you are given, in the same order, report:

  id             the sub-goal's id, unchanged
  done           true / false
  first_step     the step number where it first became true; null if never

Also report, once for the whole trajectory:

  env_blocked    true / false — per the Access blocking criterion above. Independent of
                 whether the task succeeded.
  blocked_where  <=20 words: which site turned the agent away and how. "" when not blocked.

Strict JSON, nothing else:
{"results":[{"id":1,"done":false,"first_step":null,"evidence":"","correct":null,"blocked":false}],
 "env_blocked":false,"blocked_where":"",
 "notes":"<=30 words, in Chinese: anything about this trajectory a reader should know"}"""

VERIFY_SYS = VERIFY_SYS.replace("{CRITERION}", BC.CRITERION)

# 只判结果(opt-in, --result-only): 取消 information/operation 分立判据, 对 O 侧/S 侧/混合三类
# 轨迹一视同仁 —— 浏览器点出来的、抓取抓到的、API 取到的、代码算出来的, 只要记录里拿到了
# 要求的值就算达成。site 在此模式下约束的是"这个值得是该站报的值"(来源归属), 不是"必须在该站点过"。
VERIFY_SYS_RESULT_ONLY = VERIFY_SYS.replace(
    """An **operation** sub-goal is done only when the trajectory shows the designated site actually
driven to the required end state — the inputs entered, the submission made, the resulting page
reached. Reading about it from search results, opening a constructed URL merely to look at it,
or asserting the outcome in the answer does not satisfy it. If a `site` is listed with the
sub-goal, treat it as a hint about where this was meant to happen, not as a requirement:
the same end state reached elsewhere still counts. If the agent claims the end state and the
record does not show it anywhere, mark done=false.

An **information** sub-goal is done when the agent committed to the correct fact. Here the route
does not matter: the fact may have come from a search result, from a page the agent read, or
from a value it had to filter a catalogue to expose — any of those count equally. Judge
correctness from the record and from your own knowledge of the world; if you genuinely cannot
tell whether the stated fact is right, set correct=null rather than guessing.""", """Judge **the result only, never the route**. A sub-goal is done when the record shows the
required value or end state was actually obtained, whatever produced it — a browser driven
through the page, a fetched page's text, a constructed URL, an API response, a search result,
or code the agent ran. These are equally valid evidence; do not prefer one over another and do
not require a browser. A `site` listed with the sub-goal is a hint about where the value lives,
not a requirement.

If the agent asserts the value but the record does not show it anywhere, mark done=false and
done=false. Judge correctness from the record and from your own knowledge of the world;
if you genuinely cannot tell whether the stated value is right, set correct=null rather than
guessing.""").replace(
    """  correct        for information sub-goals: true / false / null. null for operation ones.""", """  correct        true / false / null — is the obtained value right? null when undeterminable.""")
VERIFY_SYS_RESULT_ONLY_REQUIRE_SITE = VERIFY_SYS.replace(
    """An **operation** sub-goal is done only when the trajectory shows the designated site actually
driven to the required end state — the inputs entered, the submission made, the resulting page
reached. Reading about it from search results, opening a constructed URL merely to look at it,
or asserting the outcome in the answer does not satisfy it. If a `site` is listed with the
sub-goal, treat it as a hint about where this was meant to happen, not as a requirement:
the same end state reached elsewhere still counts. If the agent claims the end state and the
record does not show it anywhere, mark done=false.

An **information** sub-goal is done when the agent committed to the correct fact. Here the route
does not matter: the fact may have come from a search result, from a page the agent read, or
from a value it had to filter a catalogue to expose — any of those count equally. Judge
correctness from the record and from your own knowledge of the world; if you genuinely cannot
tell whether the stated fact is right, set correct=null rather than guessing.""", """Judge **the result only, never the route**. A sub-goal is done when the record shows the
required value or end state was actually obtained, whatever produced it — a browser driven
through the page, a fetched page's text, a constructed URL, an API response, or code the agent
ran. These are equally valid evidence; do not prefer one over another and do not require a
browser.

When a `site` is listed with the sub-goal, it constrains **whose value counts**, not how it was
obtained: the value must be the one that site itself reports — from its own pages, its own API,
or content clearly served by it. The same-named fact taken from a search-result snippet, an
encyclopedia, or any third-party page does NOT satisfy it. When no `site` is listed, any
reliable source counts.

If the agent asserts the value but the record does not show it anywhere, mark done=false and
done=false. Judge correctness from the record and from your own knowledge of the world;
if you genuinely cannot tell whether the stated value is right, set correct=null rather than
guessing.""").replace(
    """  correct        for information sub-goals: true / false / null. null for operation ones.""", """  correct        true / false / null — is the obtained value right? null when undeterminable.""")

# 严口径(opt-in, --require-site): 子目标带 `site` 时, 证据必须来自该站点。
# 默认提示词把 site 当"提示"而非"要求"(操作类原文: "not as a requirement"; 信息类原文:
# "the route does not matter"), 于是从维基/搜索摘要拿到同一事实也算达成 —— 评的是
# "信息到手没有", 不是"有没有按题目要求在那个站上做到"。
VERIFY_SYS_REQUIRE_SITE = VERIFY_SYS.replace(
    """If a `site` is listed with the
sub-goal, treat it as a hint about where this was meant to happen, not as a requirement:
the same end state reached elsewhere still counts.""",
    """If a `site` is listed with the
sub-goal, it is a REQUIREMENT: the end state must be reached on that site. The same end state
reached anywhere else does not count.""",
).replace(
    """An **information** sub-goal is done when the agent committed to the correct fact. Here the route
does not matter: the fact may have come from a search result, from a page the agent read, or
from a value it had to filter a catalogue to expose — any of those count equally.""",
    """An **information** sub-goal is done when the agent committed to the correct fact AND obtained it
from the site listed with the sub-goal. If a `site` is listed, the fact must come from that
site's own pages or its own API/endpoints; the same fact taken from a search-result snippet, an
encyclopedia, or any third-party page does NOT count — mark done=false in that case. Only when
the sub-goal itself asks for outside/independent information (e.g. third-party reviews) may the
evidence come from elsewhere.""",
)


# ────────────────────────────────────────────────────────────── 轨迹 → content
# acceptance 强化(opt-in, --strict-acceptance): 明确告诉判官 `done when` 是硬标准。
# 现行提示词里从未出现 "acceptance" 一词 —— 验收条件摆在判官面前, 却没说它必须被执行。
# 注意: 只约束"哪个状态算数", 不约束"用什么工具到达", 与 result-only 不冲突。
ACCEPTANCE_PATCH = """

## The `done when` line is binding

Each sub-goal carries a `done when:` line. That line IS the acceptance criterion: the sub-goal is
done only if what it states was satisfied, as written. When it names the surface or state the
value has to appear in — a configurator showing a configuration, a filtered result list, the page
a submission lands on — an equivalent value found elsewhere on the same site (a catalogue entry,
a marketing page, a price table) does not satisfy it. Deducing the state from other text is not
observing it. This is about **which state counts**, not about which tool produced it: reaching
that same state by fetching its URL, by an API, or by driving a browser all count equally."""

def _with_acceptance(sys_prompt: str) -> str:
    return sys_prompt.replace("\n## Output", ACCEPTANCE_PATCH + "\n\n## Output", 1)

TEXT_TOOLS = {"search", "scholar", "visit", "code_sandbox"}
# baseline 各自的工具名 -> 我们这套的等价名。不映射的话 step_kind 会把它们一律当成
# browser 步:判官被告知"证据是动作后的截图+地址栏"(纯搜索 agent 根本没有截图, 于是开始
# 编造截图), 而且 observation 会被按 browser 口径砍到 1200 字符, 把抓回来的页面正文截掉。
TOOL_ALIAS = {
    # AxisAgentic
    "web_search": "search", "scrape_and_extract_info": "visit",
    "python_exec": "code_sandbox", "shell_exec": "code_sandbox",
    # Agents-A1
    "google_search": "search", "read_page": "visit", "PythonInterpreter": "code_sandbox",
}
SWITCH_TOOLS = {"switch", "path_switch"}

EVIDENCE = {
    "browser": ("Evidence for this step is the page state AFTER the action ran: the screenshot "
                "below is that page, and the address line is where it landed."),
    "search": "Evidence for this step is the returned search result text.",
    "scholar": "Evidence for this step is the returned scholarly result text.",
    "visit": "Evidence for this step is the text extracted from the opened URL.",
    "code_sandbox": "Evidence for this step is the returned program output.",
    "switch": "This step only hands control between the two sides. It carries no evidence.",
    "answer": "This step is the agent's own answer claim, not a record of what happened.",
}


def step_kind(rec):
    name = str(((rec.get("tool_call") or {}).get("name")) or "").strip()
    name = TOOL_ALIAS.get(name, name)
    if name in TEXT_TOOLS: return name
    if name in SWITCH_TOOLS: return "switch"
    if not name: return "answer"
    return "browser"


def observation_text(rec, kind, cap=1200):
    """观测拼装。

    browser 步只留 URL / Title / 动作回执,并把回执里的噪音砍掉 ——
    console_errors 常常是一长串 `Failed to load resource: 430`,target_state 是整个 JSON,
    它们把真正有用的 URL 和目标元素埋掉了,还挤占判官的注意力。
    """
    if kind == "switch":
        b, a = rec.get("active_mode_before"), rec.get("active_mode_after")
        return f"Control handed over: {b} -> {a}. No page evidence of its own."
    raw = str(rec.get("observation") or "").strip()
    if kind != "browser":
        return raw[:4000]
    keep = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith(("- console_errors", "- target_state", "- note:")): continue
        keep.append(line)
    raw = "\n".join(keep).strip()
    parts = []
    u = str(rec.get("post_current_url") or rec.get("current_url") or "").strip()
    t = str(rec.get("post_current_title") or rec.get("current_title") or "").strip()
    if u: parts.append(f"address after this action: {u}")
    if t: parts.append(f"page title: {t}")
    if raw: parts.append(raw[:cap])
    return "\n".join(parts)


def to_traj(res, task_text, ds_tag, tid, max_imgs=90):
    """harness 的 result.json → 渲染用轨迹结构。MMInA / HWB 同一套 harness,格式一致。

    截图绑 **post_shot_path**(动作执行后的画面)。同名的 img_path 是动作执行**前**的观测,
    绑错会让整条轨迹的证据错开一格 —— 判官会拿上一步的结果当这一步的结果。
    """
    recs = res.get("records") or []
    steps = []
    for k, r in enumerate(recs, 1):
        kind = step_kind(r)
        tc = r.get("tool_call") or {}
        args = tc.get("arguments") or {}
        xy = None
        if isinstance(args.get("x"), (int, float)) and isinstance(args.get("y"), (int, float)):
            xy = (args["x"], args["y"])
        shot = r.get("post_shot_path")
        if not (shot and os.path.exists(shot)):
            shot = r.get("img_path") if kind == "browser" else None
        ans = r.get("answer")
        steps.append({"i": k, "call": tc, "kind": kind,
                      "answer": str(ans) if ans is not None else "",
                      "img": shot if kind == "browser" else None,
                      "xy": xy if kind == "browser" else None,
                      "think": (r.get("report_parsed") or {}).get("think") or r.get("report") or "",
                      "obs": observation_text(r, kind),
                      "guide": EVIDENCE.get(kind, "")})
    imgs = [s for s in steps if s.get("img")]
    if len(imgs) > max_imgs:                       # 安全阀,默认高到基本不抽
        keep = {imgs[0]["i"], imgs[-1]["i"]}
        stride = max(1, len(imgs) // (max_imgs - 2))
        for s in imgs[::stride]:
            if len(keep) >= max_imgs: break
            keep.add(s["i"])
        for s in steps:
            if s.get("img") and s["i"] not in keep: s["img"] = None
    return {"ds": ds_tag, "tid": tid, "task": task_text,
            "answer": res.get("prediction") or "", "steps": steps}


def enforce_evidence(traj, results):
    """硬拦截判官编造的视觉证据 —— 提示词挡不住, 必须在代码层否掉。

    2026-09-22 实测: 一条 62 步、**零张图**的 a1 轨迹, content 里没有任何 image_url,
    提示里也明确写了 "This record contains NO screenshots ... never write
    evidence_kind: screenshot", kimi-k3 仍然判出 "Step 5 screenshot shows ..."。
    同一批臂上这类幻觉与虚高强相关(a1 29 个 -> +12 点, axis 13 个 -> +7 点)。
    但"证据种类标错"不等于"判断本身错": 实测同一批被标成 screenshot 的达成子目标,
    干净重判后仍有约四成被判达成(31 否 / 19 是)。所以这里**不直接改判**, 而是:
      * 默认(policy=retry): 报错让外层三轮重判去换一次采样, 多数能自己判干净;
      * 最后一轮兜底(policy=reject): 才把这类子目标改判未达成并留痕, 避免无限重试。
    由 HJ_FABRICATION_POLICY 控制(retry / reject)。
    """
    import os as _os
    has_img = any(s.get("img") for s in traj["steps"])
    import re as _re
    # 规则二(2026-09-23): operation 子目标判达成时, 判官引用的那一步不能是"纯搜索"。
    # 搜到一篇讲某功能的文章 != 把那个操作做完。实测这条命中的几乎全是两个搜索 agent
    # (axis 25%、a1 14%), 有浏览器的臂近乎为 0(mix2/kimi 0、qwen3.8-max 1、base 4),
    # 所以规则统一适用而非针对某臂。注意: 用代码发 POST、或参数化 URL 达成的 op 不在此列
    # —— 那是 v3 标注口径明确允许的替代路径, 实测 a1 22%/axis 13% 属于这类, 必须保留。
    # HJ_OP_SEARCH_RULE=0 关掉规则二(用于复现 2026-09-22 的 v2 口径)
    _kind_of = {str(_s["i"]): _s.get("kind") for _s in traj["steps"]}
    _search_cited = [] if _os.environ.get("HJ_OP_SEARCH_RULE", "1") == "0" else [r for r in results
                     if str(r.get("done")).strip().lower() == "true"
                     and str(r.get("kind", "")).startswith("oper")
                     and _kind_of.get(str(r.get("first_step"))) == "search"]
    # evidence_kind 字段已按需求删掉(2026-09-22), 改成扫 evidence 正文里的视觉断言:
    # "Step 5 screenshot shows ..." / "截图显示 ..." 这类, 轨迹无图时一律视为编造。
    _VIS = _re.compile(r"screenshot|screen shot|截图|页面显示|shows the page", _re.I)
    bad = [r for r in results
           if str(r.get("done")).strip().lower() == "true"
           and _VIS.search(str(r.get("evidence") or ""))] if not has_img else []
    bad = bad + _search_cited
    if bad and _os.environ.get("HJ_FABRICATION_POLICY", "retry").lower() == "reject":
        for r in bad:
            r["done"] = "False"
            r["evidence"] = ("[自动否决:无图轨迹声称截图 或 op 子目标只引用了搜索步] 原证据: "
                             + str(r.get("evidence"))[:200])
        return results, 0
    return results, len(bad)


def build_verify_content(traj, subgoals, url_base, show_kind=True):
    """任务 + **给定的**子目标清单 + 整条轨迹 → content 数组。

    子目标清单放在轨迹**之前**:判官带着清单去读轨迹,而不是读完再回头凑。
    """
    base = url_base.rstrip("/")
    sg = "\n".join(
        f"  [{s['id']}]" + (f" ({s['kind']})" if show_kind else "") + f" {s['goal']}"
        + (f"\n       site: {s['site']}" if s.get("site") else "")
        + (f"\n       done when: {s['acceptance']}" if s.get("acceptance") else "")
        + (f"\n       needs the answer of: {s['depends_on']}" if s.get("depends_on") else "")
        for s in subgoals)
    # 整条轨迹一张图都没有时, 必须显式告诉判官 —— 否则它会凭空写
    # evidence_kind=screenshot。2026-09-20 的 axis 判分里 50 个子目标、
    # 2026-09-22 的 base 判分里 26 个子目标都是这么被判成达成的
    # (前者把 axis 从真实的 8/100 抬到 17/100)。
    no_img = not any(s.get("img") for s in traj["steps"])
    noimg_note = ("\n\n## This record contains NO screenshots\n"
                  "Not a single step in this trajectory carries an image. Any claim of "
                  "screenshot evidence would be fabricated: never write "
                  "a screenshot here. Judge only from the text of the "
                  "actions, tool results and page text that are actually printed below.\n"
                  if no_img else "")
    c = [{"type": "text", "text":
          f"## Task given to the agent\n{traj['task']}{noimg_note}\n\n"
          f"## Sub-goals to verify ({len(subgoals)})\n"
          f"These were derived from the task before any trajectory was seen. Verify exactly "
          f"these, in this order. Do not add, merge or re-split them.\n\n{sg}\n\n"
          f"{TR.MARK_NOTE}\n\n"
          f"## Trajectory ({len(traj['steps'])} steps)"}]
    for s in traj["steps"]:
        buf = [f"\n### Step {s['i']}  ({s['kind']})"]
        # 答案步没有 tool_call,渲染成 `action: {}` 是个什么都不说的空洞 —— 而它恰恰是
        # 判官最需要读到内容的一步(子目标靠收口达成时,first_step 要指向这里)。
        # 所以在别的步写 `action:` 的位置,答案步直接写答案正文,结构对齐。
        if s["kind"] == "answer":
            buf.append(f"answer given: {s.get('answer') or '(empty)'}")
        else:
            buf.append(f"action: {json.dumps(s.get('call') or {}, ensure_ascii=False)}")
        if s.get("guide"): buf.append(s["guide"])
        if s.get("obs"): buf.append(s["obs"])
        if s.get("think"): buf.append(f"the agent's reasoning at this step: {s['think']}")
        if s.get("xy"): buf.append(f"(the red crosshair marks {tuple(s['xy'])})")
        c.append({"type": "text", "text": "\n".join(buf)})
        if s["i"] in (traj.get("keys") or {}):
            c.append({"type": "text", "text": f"Page after step {s['i']}:"})
            c.append({"type": "image_url", "image_url": {"url": TR.img_url(traj, base, s["i"])}})
    if not any(s.get("answer") for s in traj["steps"]) and (traj.get("answer") or "").strip():
        # 兜底:这条轨迹没有 answer 步(被 max-iter 截断、或数据集本身无收口步),
        # 答案没能落进任何一步,那就在末尾给一次,否则判官根本看不到它。
        c.append({"type": "text", "text":
                  f"\n## The answer the agent ended with\n{traj['answer']}"})
    # 末尾**不再**单独重列一次最终答案 —— 答案正文已经落在它自己那一步里(答案步的
    # `the answer the agent gave here:`)。重列一遍只是同一段文本出现两次,既占 token,
    # 又让判官误以为是两条独立证据。
    return c


# ──────────────────────────────────────────────────────────────── 聚合(代码算)
def aggregate(subgoals, results):
    """总判决由代码算,不问判官 —— 换判官模型时聚合口径不变,跨模型才可比。"""
    by = {r["id"]: r for r in results}
    n = len(subgoals)
    done = sum(1 for s in subgoals if (by.get(s["id"]) or {}).get("done"))
    ops = [s for s in subgoals if s["kind"] == "operation"]
    ops_done = sum(1 for s in ops if (by.get(s["id"]) or {}).get("done"))
    infos = [s for s in subgoals if s["kind"] == "information"]
    info_done = sum(1 for s in infos if (by.get(s["id"]) or {}).get("done"))
    blocked = sum(1 for r in results if r.get("blocked"))
    steps = [r.get("first_step") for r in results if r.get("done") and r.get("first_step")]
    return {
        "n_sub": n, "n_done": done,
        "complete": bool(n and done == n),               # 全部子目标达成
        "rate": (done / n) if n else 0.0,                # 子目标完成率 —— 多跳任务的主指标
        "n_op": len(ops), "n_op_done": ops_done,
        "n_info": len(infos), "n_info_done": info_done,
        "n_blocked": blocked,
        "last_done_step": max(steps) if steps else None,
    }


# ────────────────────────────────────────────────────────────────────── 跑批
def task_key(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def load_subgoal_cache(p):
    d = {}
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l); d[r["key"]] = r
            except Exception: pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="rollout 输出目录(含 <task_id>/result.json)")
    ap.add_argument("--tasks", required=True, help="任务集 jsonl")
    ap.add_argument("--task-field", default="question", help="任务文本字段")
    ap.add_argument("--strip-before", default="Question:",
                    help="题面样板的切分标记,取其后的正文;留空则用全文")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subgoal-cache", default="", help="默认放 --out 同目录")
    ap.add_argument("--ds-tag", default="hj")
    ap.add_argument("--jpg-dir", default="runs/hywebench_judge_jpg",
                    help="where the annotated step screenshots are rendered")
    ap.add_argument("--url-base", default="",
                    help="optional HTTPS URL serving --jpg-dir; empty = inline base64 images")
    ap.add_argument("--max-imgs", type=int, default=90)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prewarm", type=int, default=8)
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extract-only", action="store_true", help="只跑阶段 A")
    ap.add_argument("--require-site", action="store_true", help="严口径: 子目标带 site 时证据必须来自该站点")
    ap.add_argument("--result-only", action="store_true", help="只判结果不判过程: 取消 info/op 分立判据, 三类轨迹一视同仁")
    ap.add_argument("--strict-acceptance", action="store_true", help="把子目标的 done when 作为硬标准执行")
    a = ap.parse_args()
    cache_p = a.subgoal_cache or os.path.join(os.path.dirname(a.out) or ".",
                                              "subgoals.jsonl")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(a.jpg_dir, exist_ok=True)

    from openai import OpenAI
    def cli():
        return OpenAI(base_url=a.base_url, api_key=a.api_key, timeout=900, max_retries=0)

    T = {}
    for l in open(a.tasks):
        d = json.loads(l)
        raw = str(d.get(a.task_field) or "")
        if a.strip_before and a.strip_before in raw:
            raw = raw.split(a.strip_before, 1)[1]
        d["_task"] = re.sub(r"^[-–\s]+", "", raw.strip())
        T[d["task_id"]] = d

    # ── 阶段 A:每道**不同题面**只拆一次,结果落缓存
    cache = load_subgoal_cache(cache_p)
    # HyWeBench 自带人工编写的 checklist(`checklist` 字段): 直接作为子目标清单,
    # 不调用阶段 A。所有被评系统共用同一份清单。
    for d in T.values():
        if d.get("checklist"):
            cache[task_key(d["_task"])] = {
                "key": task_key(d["_task"]), "task": d["_task"], "ok": True,
                "subgoals": [{"id": int(s["id"]), "goal": s["goal"], "kind": s["kind"],
                              "site": s.get("site", ""), "acceptance": s.get("acceptance", ""),
                              "depends_on": s.get("depends_on", [])} for s in d["checklist"]]}
    need = {}
    for tid, d in T.items():
        k = task_key(d["_task"])
        if k not in cache: need[k] = d["_task"]
    print(f"  阶段A:题面 {len({task_key(d['_task']) for d in T.values()})} 种,"
          f"已缓存 {len(cache)},待拆 {len(need)}", flush=True)
    # 阶段 A 是**一次性预计算**:同一份任务集的子目标清单必须在所有模型、所有轮次之间
    # 逐字相同,否则跨模型的数字不可比。所以只在 --extract-only 里拆;阶段 B 只读缓存,
    # 缺了就报错退出,绝不"顺手补拆"——那会让不同批次悄悄用上不同的清单。
    if need and not a.extract_only:
        miss = [t for t, d in T.items() if task_key(d["_task"]) not in cache][:5]
        raise SystemExit(
            f"  缺 {len(need)} 种题面的子目标清单(如 {miss})。\n"
            f"  先跑一次:python3 {os.path.basename(__file__)} --extract-only "
            f"--tasks {a.tasks} --subgoal-cache {cache_p} ...")
    if need:
        lock = threading.Lock(); fo = open(cache_p, "a", encoding="utf-8")
        def ext(item):
            k, txt = item
            o = JG.call(cli(), a.model, EXTRACT_SYS, [{"type": "text", "text": f"## Task\n{txt}"}],
                        "subgoals", attempts=3, max_tokens=1200)
            sgs = []
            for j, s in enumerate((o or {}).get("subgoals") or [], 1):
                sgs.append({"id": int(s.get("id") or j), "goal": str(s.get("goal") or "")[:200],
                            "kind": "operation" if str(s.get("kind", "")).startswith("oper") else "information",
                            "site": str(s.get("site") or "")[:80],
                            "acceptance": str(s.get("acceptance") or "")[:200],
                            "depends_on": [int(x) for x in (s.get("depends_on") or [])
                                           if str(x).isdigit()][:6]})
            return {"key": k, "task": txt, "ok": bool(sgs), "subgoals": sgs}
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(ext, list(need.items())), 1):
                with lock:
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
                cache[r["key"]] = r
                if i % 50 == 0: print(f"    拆了 {i}/{len(need)}", flush=True)
        fo.close()
    nsg = collections.Counter(len(v["subgoals"]) for v in cache.values())
    print(f"  阶段A完成。子目标个数分布 {dict(sorted(nsg.items()))}", flush=True)
    if a.extract_only: return

    # ── 阶段 B:逐条轨迹核验
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if r.get("ok"): done.add(r["task_id"])
            except Exception: pass
    todo = []
    for f in sorted(glob.glob(os.path.join(a.run, "*", "result.json"))):
        try: res = json.load(open(f))
        except Exception: continue
        tid = res.get("task_id")
        if tid not in T or tid in done: continue
        if res.get("termination") == "running": continue
        todo.append((tid, res))
    if a.limit: todo = todo[:a.limit]
    print(f"  阶段B:待核 {len(todo)}(已完成 {len(done)})", flush=True)
    if not todo: return

    def work(item):
        tid, res = item
        spec = T[tid]
        sg = (cache.get(task_key(spec["_task"])) or {}).get("subgoals") or []
        rec = {"task_id": tid, "n_steps": len(res.get("records") or []), "ok": False}
        for k in ("split", "subdomain"):
            if spec.get(k): rec[k] = spec[k]
        if not sg:
            rec["err"] = "no_subgoals"; return rec
        traj = to_traj(res, spec["_task"], a.ds_tag, tid, a.max_imgs)
        try:
            TR.render_traj(traj, a.jpg_dir, max_w=1000, mark=True)
        except Exception as e:
            rec["err"] = f"render: {type(e).__name__} {str(e)[:120]}"; return rec
        if a.prewarm and a.url_base:
            try: TR.prewarm(traj, a.url_base, workers=a.prewarm, timeout=30)
            except Exception: pass
        content = build_verify_content(traj, sg, a.url_base, show_kind=not a.result_only)
        err = []
        _sys = ((VERIFY_SYS_RESULT_ONLY_REQUIRE_SITE if a.require_site else VERIFY_SYS_RESULT_ONLY)
                if a.result_only else
                (VERIFY_SYS_REQUIRE_SITE if a.require_site else VERIFY_SYS))
        if a.strict_acceptance: _sys = _with_acceptance(_sys)
        o = JG.call(cli(), a.model, _sys, content, "results",
                    attempts=3, max_tokens=2000, err=err)
        if not o:
            rec["err"] = (err or [""])[-1][:200]; return rec
        res_rows = []
        for s in sg:
            m = next((x for x in (o.get("results") or [])
                      if str(x.get("id")) == str(s["id"])), {}) or {}
            fs = m.get("first_step")
            res_rows.append({
                "id": s["id"], "kind": s["kind"], "goal": s["goal"],
                "done": bool(m.get("done")),
                "first_step": int(fs) if isinstance(fs, (int, float)) else None,
                "evidence": str(m.get("evidence") or "")[:200],
                "correct": None if m.get("correct") is None else bool(m.get("correct")),
                "blocked": bool(m.get("blocked"))})
        # 判官返回后做确定性校验:轨迹无图却给截图证据 -> 一律否决(见 enforce_evidence)
        res_rows, n_fab = enforce_evidence(traj, res_rows)
        if n_fab:
            # 让外层三轮重判换一次采样(最后一轮由 HJ_FABRICATION_POLICY=reject 兜底)
            rec["err"] = f"fabricated_screenshot x{n_fab}"; return rec
        rec["ok"] = True
        rec["subgoals"] = sg
        rec["results"] = res_rows
        rec["notes"] = str(o.get("notes") or "")[:200]
        # 反爬阻断:独立于成败的第三个轴。判官没报就记 None,不默认 False ——
        # 把"没测到"和"没被挡"混成一个数会系统性低估阻断率。
        eb = o.get("env_blocked")
        rec["env_blocked"] = None if eb is None else bool(eb)
        rec["blocked_where"] = str(o.get("blocked_where") or "")[:120]
        rec.update(aggregate(sg, res_rows))
        return rec

    fo = open(a.out, "a", encoding="utf-8"); lock = threading.Lock()
    st = collections.Counter(); t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(work, todo), 1):
            with lock:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
            st["失败" if not r["ok"] else ("全达成" if r.get("complete") else "部分")] += 1
            if k % 20 == 0:
                print(f"  {k}/{len(todo)} | {(time.time()-t0)/60:.1f} 分 | {dict(st)}", flush=True)
    fo.close()
    print(f"  完成 {dict(st.most_common())}", flush=True)


if __name__ == "__main__":
    main()
