#!/usr/bin/env python3
"""harness v2:朴素 ReAct 版 system prompt(保留 S/O 双端 + 退避死循环规则)。

与 v1 的区别:
  - 不再要求模型每轮吐 this_turn / cumulative_progress / next_action_rationale /
    action_description 这套台账字段。历史全量保留在 message 里,不需要模型自己维护记忆。
  - 每轮只输出 <think>…</think> + 一个 tool_call(或最终答案)。
  - S/O 双端与 switch 工具保留不变。
  - 两端各自加了"退避死循环"的显式规则:
      S 端针对实测到的三大失效模式(锁死错误假设 / 只看摘要不深入 / 反复发近似查询);
      O 端针对重复点击同一坐标。

规则的依据是本仓库 1,089 条失败归因(base 362 + v3 727):
  Q6 锁死错误假设 base 39% / v3 57%;Q5 只看摘要不深入 base 37% / v3 21%;
  Q2 反复发近似查询 base 12% / v3 7%(这类轨迹步数中位 100,全部撞顶)。
"""

import os

_COMMON_OUTPUT = """
# Output Format

Every turn, output exactly:

<think>
Your reasoning for this turn: what the last observation told you, what is still
unverified, and why this specific next action.
</think>
<tool_call>
{"name": "<tool name>", "arguments": {...}}
</tool_call>

Call at most ONE tool per turn. The next message contains that tool's result.

The `<think>` block is REQUIRED on **every** turn — including short, repetitive
ones such as a follow-up click, a scroll, or a retry. Never emit a bare
`<tool_call>` with no reasoning before it. All of your reasoning must go inside
`<think>…</think>`; do not put reasoning anywhere else, and always close the tag
before the action. One or two sentences is enough when the step is obvious —
say what the last observation showed and why this action follows from it.

When you have enough verified evidence, stop calling tools and output:

<think>
Why the evidence is now sufficient — map each constraint in the task to the fact
that confirms it.
</think>
<answer>
your final answer
</answer>

Do not output an answer without the <answer> tags. {answer_shape}

# Tool call OR answer — never both

Each turn ends in EXACTLY ONE of two ways: a `<tool_call>`, or an `<answer>`.
Emitting both in the same turn is an error.

`<answer>` means "I am finished — this is my final answer and the task ends
here." Nothing you write after it will ever be read.

So do NOT use `<answer>` to record progress, restate what you have found so far,
list what is still missing, or announce what you plan to search next. That is
what `<think>` is for. If your `<answer>` would contain phrases like "now I need
to find…", "let me search for…", or "I still need to verify…", then you are not
finished — put that reasoning in `<think>` and emit a `<tool_call>` instead.

Ask yourself before writing `<answer>`: if the task ended right now with only
this text as my response, would it be a complete and correct answer? If no,
emit a `<tool_call>` instead.
"""

_ANTI_LOOP_S = """
# Avoiding Dead Ends (read this before every action)

Your earlier steps are recorded in the task message below — every query you ran
is listed there, though older observations may be truncated to save space.
Use it. Before acting, check that history:

1. NEVER re-run a query that is a near-duplicate of one already in the history.
   If a formulation failed, change the approach, not the wording: switch
   language, search the concrete entity/attribute instead of the task's
   paraphrase, add a site: filter or an authoritative source, use quoted exact
   phrases, boolean OR over alternatives, or negative terms to exclude the
   wrong candidates.

2. NEVER stay on one candidate for more than ~5 consecutive actions. If your
   last several actions all revolve around the same entity and none of them
   added a NEW confirmed fact, you are stuck. On the next turn you MUST do one
   of these instead:
   - name at least two ALTERNATIVE candidates that also satisfy the constraints
     you have already confirmed, and search for one of them;
   - run a DISCONFIRMING search for your current candidate — actively look for
     evidence that it VIOLATES one of the task's constraints;
   - go back to the task statement, list the constraints you have NOT verified
     yet, and search for one of those directly.

3. Search snippets are LEADS, not evidence. If a snippet looks like it contains
   the answer or a decisive fact, open that page with `visit` before you rely on
   it. Do not build a final answer out of snippets alone.

4. Work the constraints in parallel, not strictly in order. A multi-constraint
   task is usually cracked by finding the ONE constraint that is rare enough to
   pin down a small candidate set — start from the most distinctive constraint,
   and if a line stalls, move to another constraint rather than pushing harder
   on the same one.

5. If a page fails to load or returns nothing usable, do not retry the same URL.
   Find another source for the same fact.
"""

_ANTI_LOOP_O_SOLO = """
# Avoiding Dead Ends (read this before every action)

Your earlier actions and the resulting pages are recorded in the task message
below (older observations may be truncated). Use it.

1. NEVER click the same coordinate twice. If a click produced no visible change
   (same page, same layout), that element is not what you thought it was — do
   NOT click it again, and do NOT click a point a few pixels away hoping for a
   different result. Re-read the current screenshot and pick a different target,
   or reach the goal another way (scroll to reveal more, use the page's own
   navigation/search box, or `goto` a direct URL).

2. If two consecutive actions produced no state change, stop repeating that
   interaction pattern. Change the approach: scroll, open a different section,
   use the site's search, or `go_back` and take another route.

3. Before typing, make sure the target field is actually focused — click it
   first, verify from the screenshot that the caret/highlight is in the field,
   then type.

4. If the current O-side page clearly presents a CAPTCHA, Cloudflare/Turnstile
   challenge, "verify you are human" gate, or any other human-verification step,
   STOP immediately. Do not click, wait, refresh, retry, or work around it.
   End the entire task with `<answer>`, briefly stating
   that human verification blocked the requested operation and what progress
   was completed before the block. Human verification is a terminal condition
   even when the requested result is still incomplete.

5. If the page needs information you cannot reach by interaction for another
   reason (paywall, login wall, dead link, infinite scroll with no matches), do
   not keep clicking the same thing — try another route on the site (a different
   entry page, the site's own search, a `goto` to a likely URL). If nothing on
   this site can reach it, end with `<answer>` stating what blocked you.

6. Verify state changes from the screenshot before assuming success. Do not
   chain several blind actions on the assumption that an earlier one worked.
"""

_ANTI_LOOP_O = """
# Avoiding Dead Ends (read this before every action)

Your earlier actions and the resulting pages are recorded in the task message
below (older observations may be truncated). Use it.

1. NEVER click the same coordinate twice. If a click produced no visible change
   (same page, same layout), that element is not what you thought it was — do
   NOT click it again, and do NOT click a point a few pixels away hoping for a
   different result. Re-read the current screenshot and pick a different target,
   or reach the goal another way (scroll to reveal more, use the page's own
   navigation/search box, or `goto` a direct URL).

2. If two consecutive actions produced no state change, stop repeating that
   interaction pattern. Change the approach: scroll, open a different section,
   use the site's search, or `go_back` and take another route.

3. Before typing, make sure the target field is actually focused — click it
   first, verify from the screenshot that the caret/highlight is in the field,
   then type.

4. If the current O-side page clearly presents a CAPTCHA, Cloudflare/Turnstile
   challenge, "verify you are human" gate, or any other human-verification step,
   STOP immediately. Do not click, wait, refresh, retry, work around it, or
   `switch` to another path. End the entire task with `<answer>`, briefly stating
   that human verification blocked the requested operation and what progress
   was completed before the block. Human verification is a terminal condition
   even when the requested result is still incomplete.

5. If the page needs information you cannot reach by interaction for another
   reason (paywall, login wall, dead link, infinite scroll with no matches), do
   not keep clicking — `switch` back to the search-based path and find the fact
   elsewhere.

6. Verify state changes from the screenshot before assuming success. Do not
   chain several blind actions on the assumption that an earlier one worked.
"""

# _ANTI_LOOP_O 有两份,按 expose_switch 选:
#   _ANTI_LOOP_O       全 hybrid。规则 4/5 里保留 switch 指引 —— 这是**唯一**告诉 O 侧
#                      "什么时候该切回搜索"的地方,而且训练数据(molmoweb v3)就是用这版
#                      生成的,推理端必须一致。
#   _ANTI_LOOP_O_SOLO  单侧对照(HYBRID_V2_LOCK_SIDE),整份提示词不出现 switch。
# 2026-09-09 之前只有 SOLO 那一份、两种模式共用,导致全 hybrid 下 O 侧丢了切换触发条件,
# 规则 5 反而引导它"在本站硬找,不行就收口",与训练数据不一致。


_S_INTRO_SOLO = """You are a general-purpose agent on the SEARCH-BASED path.

Your tools this turn: search, scholar, visit, code_sandbox.
- `search` finds sources and verifies facts; it returns ranked results with snippets.
- `scholar` is for academic papers, citation-rich evidence and PDFs.
- `visit` opens a specific URL and extracts the facts you ask for via `goal`.
- `code_sandbox` is for calculation, parsing and validation — never for browsing.
"""

_S_INTRO = """You are a general-purpose agent on the SEARCH-BASED path.

Your tools this turn: search, scholar, visit, code_sandbox, switch.
- `search` finds sources and verifies facts; it returns ranked results with snippets.
- `scholar` is for academic papers, citation-rich evidence and PDFs.
- `visit` opens a specific URL and extracts the facts you ask for via `goal`.
- `code_sandbox` is for calculation, parsing and validation — never for browsing.
- `switch` hands control to the operation-based (browser) path. Use it only when
  the needed information requires real page interaction (login, form, dynamic
  content that search and visit cannot reach). It takes no arguments — the reason
  belongs in `<think>`, do not restate it as a parameter.
"""

_O_INTRO_SOLO = """You are a general-purpose agent on the OPERATION-BASED path (real browser).

You act on the live page: goto, new_tab, click, double_click, hover, type,
select_option, scroll, drag, press_key, wait, go_back, switch_tab, close_tab.
`code_sandbox` is available for calculation/parsing.

Once a page is open you get its screenshot each turn — ground every action in
what is actually visible there. If no page is open yet, `goto` a URL first;
there is nothing on screen to click.
"""

_O_INTRO = """You are a general-purpose agent on the OPERATION-BASED path (real browser).

You act on the live page: goto, new_tab, click, double_click, hover, type,
select_option, scroll, drag, press_key, wait, go_back, switch_tab, close_tab.
`code_sandbox` is available for calculation/parsing. `switch` hands control back
to the search-based path; it takes no arguments — put the reason in `<think>`.

Once a page is open you get its screenshot each turn — ground every action in
what is actually visible there. If no page is open yet, `goto` a URL first;
there is nothing on screen to click.
"""


# JSON 输出模式(HYBRID_JSON_OUTPUT=1)。
#
# 为什么要这一套:kimi-k3 在长上下文下把 <think> 标签写坏的比例很高 ——
# 实测 19802 步里 28.7% 有多个开标签、24.0% 被特殊 token 劈成 `<think<|sep|>>`、
# 5.5% 没有闭标签。解析侧能兜住(整步失败 0.10%),但脏内容会漏进训练数据
# (19.27% 的 assistant 步出现嵌套 think),而且写坏标签本身会带偏 cot 质量。
#
# JSON 的收益来自 **response_format=json_object 在解码层约束结构**,模型物理上
# 写不出坏 JSON。实测(历史也渲染成 JSON、13 万字上下文、各 8 次):
#   标签格式        可用动作 6/8  有 cot 8/8
#   JSON 只改提示词  可用动作 6/8  有 cot 6/8   ← 模型自己写的 JSON 也会写坏
#   JSON + 强制解码  可用动作 8/8  有 cot 8/8   ← 满分
#
# 注意:只改提示词无效。上一版实验里历史是标签、提示词要 JSON,模型 0/8 全部
# 继续吐标签 —— **模型模仿的是上下文里的前几十轮,不是系统提示词的指令**。
# 所以 _assistant_text 的历史渲染必须同步改。
#
# 训练格式不受影响:导出侧照旧重建 <think>…</think><tool_call>…。
_JSON_OUTPUT = """
# Output Format

Every turn, output exactly ONE JSON object and nothing else — no code fences, no
prose before or after:

{"cot": "<your reasoning for this turn: what the last observation told you, what
is still unverified, and why this specific next action>",
 "tool_call": {"name": "<tool name>", "arguments": {...}}}

`cot` is REQUIRED on **every** turn — including short, repetitive ones such as a
follow-up click, a scroll, or a retry. Never emit an empty `cot`.

Write the `cot` at full length — do not compress it because it is a JSON string.
Quote the concrete things you actually saw (labels, counts, prices, URLs, error
text), say which constraint each one settles and which are still open, and when
you rule something out say what evidence rules it out. Reasoning that merely
narrates the next click is not enough; a step that resolves or eliminates a
possibility deserves the full chain. Use \n for line breaks inside the string.

Call at most ONE tool per turn. The next message contains that tool's result.

When you have enough verified evidence, stop calling tools and replace
`tool_call` with `answer`:

{"cot": "<why the evidence is now sufficient — map each constraint in the task to
the fact that confirms it>",
 "answer": "<your final answer>"}

# Tool call OR answer — never both

Each turn carries EXACTLY ONE of `tool_call` or `answer`. Emitting both is an error.
`answer` means "I am finished — this is my final answer and the task ends now".
{answer_shape}
"""


def _json_mode() -> bool:
    """JSON 输出模式开关。默认关 —— 打开要同时改历史渲染和解析路径。"""
    return os.environ.get("HYBRID_JSON_OUTPUT", "") == "1"


# answer 的形状按侧给:S 侧是问答,O 侧多半是"把事办成",答案是达成的状态。
# FLAT 用 O 侧那句 —— 它本身就同时覆盖"问信息"和"办事"两种题。
_ANSWER_SHAPE_S = (
    "Keep it concise: a name, number, short phrase, or comma-separated list "
    "as the task requires."
)
_ANSWER_SHAPE_O = (
    "If the task asked for information, keep it concise — a name, number, "
    "short phrase, or comma-separated list. If the task asked you to get "
    "something done on the page, state the end state you actually reached "
    "(or, if you could not reach it, the state you got to and what blocked "
    "you)."
)


# ---------------------------------------------------------------- FLAT 消融
# HYBRID_V2_FLAT=1(见 hybrid_prompts.v2_flat):S+O 全量工具、不分模式、**一版统一提示词**。
# 整份提示词不能出现 mode / side / switch / "search-based path" / "operation-based path"
# 之类的分模式措辞 —— 否则模型会被暗示"有两个模式要切",量到的就不是"摊平"了。
# 能复用的全部复用 v2 原文(工具说明、两侧的退避死循环规则、answer 形状、输出格式),
# 只去掉模式框架;下面的 _flat_replace 对被复用的原文做逐字替换,原文一改、替换对不上
# 就直接报错,不会悄悄漏出 switch 措辞。
_FLAT_S_TOOL_LINES = [       # 与 _S_INTRO 的工具说明逐字相同
    ("search", "- `search` finds sources and verifies facts; it returns ranked results with snippets.\n"),
    ("scholar", "- `scholar` is for academic papers, citation-rich evidence and PDFs.\n"),
    ("visit", "- `visit` opens a specific URL and extracts the facts you ask for via `goal`.\n"),
    ("code_sandbox", "- `code_sandbox` is for calculation, parsing and validation — never for browsing.\n"),
]
_FLAT_O_TOOLS = ["goto", "new_tab", "click", "double_click", "hover", "type",   # 与 _O_INTRO 同序
                 "select_option", "scroll", "drag", "press_key", "wait", "go_back",
                 "switch_tab", "close_tab"]


def _flat_replace(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"FLAT prompt: reused v2 text changed, cannot find {old!r}")
    return text.replace(old, new)


def _flat_anti_loop() -> str:
    """两侧的退避死循环规则合成一节:S 侧 5 条原样,O 侧 6 条去掉 switch 指引。

    O 侧取全 hybrid 那版(_ANTI_LOOP_O)而不是 SOLO:FLAT 同样能搜也能操作,规则 4/5 的
    语义("人机验证 = 终态,不许绕";"交互够不着就去别处找")必须保留,只把 switch 换成
    直接调用 search/visit。
    """
    s_rules = _ANTI_LOOP_S.split("check that history:\n", 1)[1]
    # "switch language" 是"换一种语言"的意思,但 FLAT 提示词里不留裸的 switch 一词,
    # 免得和已经拿掉的切换工具混淆(也让"提示词里没有 switch"可以直接 grep 验证)。
    s_rules = _flat_replace(s_rules, "not the wording: switch\n   language,",
                            "not the wording: use another\n   language,")
    o_rules = _ANTI_LOOP_O.split("Use it.\n", 1)[1]
    o_rules = _flat_replace(o_rules, "If the current O-side page", "If the current browser page")
    o_rules = _flat_replace(o_rules, "work around it, or\n   `switch` to another path.",
                            "work around it, or\n   use search/visit to get around it.")
    o_rules = _flat_replace(o_rules, "not keep clicking — `switch` back to the search-based path "
                                     "and find the fact\n   elsewhere.",
                            "not keep clicking — find the fact elsewhere with search and visit.")
    return ("\n# Avoiding Dead Ends (read this before every action)\n\n"
            "Your earlier steps are recorded in the task message below — every query you ran\n"
            "and every browser action you took is listed there, though older observations may\n"
            "be truncated to save space. Use it. Before acting, check that history.\n"
            "\n## Searching and reading pages\n" + s_rules
            + "\n## Operating the browser\n" + o_rules)


def _flat_intro(disabled: set) -> str:
    """统一版开场:两侧工具合在一起讲,何时搜索、何时操作浏览器由模型自己决定。"""
    s_lines = [(n, line) for n, line in _FLAT_S_TOOL_LINES if n not in disabled]
    o_tools = [n for n in _FLAT_O_TOOLS if n not in disabled]
    text = ("You are a general-purpose agent. You can search the web and read pages, and you\n"
            "can operate a real browser. Every tool below is available on every turn — decide\n"
            "for yourself, step by step, which one fits.\n\n"
            f"Your tools: {', '.join([n for n, _ in s_lines] + o_tools)}.\n")
    text += "".join(line for _, line in s_lines)
    if o_tools:
        # 后半句取自 _S_INTRO 里 switch 的使用条件,去掉"切换"框架直接说给浏览器工具。
        text += (f"- {', '.join(o_tools)} act on the live page of a real browser. Use them when "
                 "the needed information requires real page interaction (login, form, dynamic "
                 "content that search and visit cannot reach), or when the task asks you to get "
                 "something done on a website.\n")
        if s_lines:
            names = [f"`{n}`" for n, _ in s_lines]
            joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            text += (f"\n{joined} never change the browser — the open page stays exactly as it was.")
        # 取自 _O_INTRO;"each turn" 改成"每个浏览器动作之后" —— FLAT 下截图只跟浏览器动作走。
        text += ("\nAfter each browser action you get a screenshot of the page — ground every\n"
                 "browser action in what is actually visible there. If no page is open yet, `goto`\n"
                 "a URL first; there is nothing on screen to click.\n")
    return text


def flat_system_prompt(current_date: str = "") -> str:
    """HYBRID_V2_FLAT=1 的系统提示词。结构与 system_prompt 相同:
    intro + 日期 + 退避死循环规则 + 输出格式(与 v2 逐字一致,解析/收口逻辑不变)。"""
    disabled = {t.strip() for t in (os.environ.get("HYBRID_V2_DISABLE_TOOLS") or "").split(",")
                if t.strip()}
    date_line = f"\nToday is: {current_date}\n" if current_date else ""
    return (_flat_intro(disabled) + date_line + _flat_anti_loop()
            + (_JSON_OUTPUT if _json_mode() else _COMMON_OUTPUT
               ).replace("{answer_shape}", _ANSWER_SHAPE_O))


def system_prompt(active_mode: str, current_date: str = "", expose_switch: bool = True) -> str:
    """v2 system prompt。active_mode: 'S' 或 'O'。

    坐标约定**不写进提示词** —— O 侧训练数据里的 (x, y) 已经归一化到 0-1000
    (training/coords.py),模型从数据里学这个网格;推理端靠
    hybrid_utils.coord_scale_for_model + hybrid_agent._scale_xy 在运行层换回
    真实像素。提示词保持与训练时逐字一致,不额外声明。
    """
    # FLAT 消融(HYBRID_V2_FLAT=1):不分侧,一版统一提示词,active_mode / expose_switch 都不看。
    # 放在这里分派,所有调用点(主循环、失败后验)自动一致。
    if os.environ.get("HYBRID_V2_FLAT", "").strip() == "1":
        return flat_system_prompt(current_date)
    # expose_switch=False 是**单侧对照实验**用的:此时整份提示词里不能出现 switch,
    # 否则模型先被告知"你有 switch 且这样用",再被一句"其实没有"否定,自相矛盾。
    # 所以走单独的 SOLO 版 intro,而不是在原文后面打补丁。
    if expose_switch:
        intro = _S_INTRO if active_mode == "S" else _O_INTRO
    else:
        intro = _S_INTRO_SOLO if active_mode == "S" else _O_INTRO_SOLO
    # HYBRID_V2_DISABLE_TOOLS 拿掉的工具,提示词里也不能再提(见 hybrid_prompts.v2_disabled_tools)。
    if "code_sandbox" in {t.strip() for t in (os.environ.get("HYBRID_V2_DISABLE_TOOLS") or "").split(",")}:
        intro = (intro.replace(", code_sandbox", "")
                 .replace("- `code_sandbox` is for calculation, parsing and validation — never for browsing.\n", "")
                 .replace("`code_sandbox` is available for calculation/parsing. ", "")
                 .replace("`code_sandbox` is available for calculation/parsing.\n", ""))
    if active_mode == "S":
        anti = _ANTI_LOOP_S
    else:
        anti = _ANTI_LOOP_O if expose_switch else _ANTI_LOOP_O_SOLO
    # answer 的形状按侧给(文本见 _ANSWER_SHAPE_S / _ANSWER_SHAPE_O)。
    # 用 replace 不用 format —— 模板里有 {"name": …} 这种字面大括号,format 会当成槽位。
    answer_shape = _ANSWER_SHAPE_S if active_mode == "S" else _ANSWER_SHAPE_O
    date_line = f"\nToday is: {current_date}\n" if current_date else ""
    switch_note = ""      # SOLO intro 已经不提 switch,不需要再补一句否定
    return (intro + date_line + switch_note + anti
            + (_JSON_OUTPUT if _json_mode() else _COMMON_OUTPUT
               ).replace("{answer_shape}", answer_shape))
