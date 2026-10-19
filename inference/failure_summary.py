"""retry 保活:撞顶重开前,让 agent 模型自己写一份失败后验带进下一次尝试。

移植自 Iris(arXiv 2609.04304)的 `retry` —— 他们的消融里这一项单独就有
BC-ZH +10.7 / DeepSearchQA +8.1(Iris-mini,相对不做上下文管理),比 discard-all
还高。我们原来的 `--max-iter-restarts 5` 只有"重开"这个壳,前几次尝试的发现
全部丢掉,等于每次都从零开始。

三处实测结论(runs/probe_failure_summary.py,8 道真实撞顶轨迹):
  * 只给纯文本指令、不传 tool schema —— **挡不住**。3/3 全部把 <tool_call>{…}
    当文本吐出来。不传 schema 只能让它发不出*真*的工具调用。
  * 预填 assistant 的 <think> 前缀是决定性的那一步,加上之后不再吐工具调用。
  * prompt 必须适配我们 v2 的标签口径(把三小节放进 <answer>)。硬要模型输出
    裸文本是逆着训练分布走:Iris 原文 prompt 只有 2/3,适配后 7/8。
    那 1 道失败是采样抖动 —— 单独重抽 5 次 5/5 全可用。
"""
from __future__ import annotations

import os
import re

# ── 提示词(逻辑与 Iris prompt_utils.py 对齐,外壳换成我们的标签)──────────────
FAILURE_SUMMARY_PROMPT = """The task attempt has ended without a final answer. Write a brief post-mortem so the NEXT attempt can try a better strategy.

STRICT OUTPUT RULES (read carefully):
- This is NOT a chance to answer the question. Do NOT output a final answer here.
- You MUST NOT call any tool. Do NOT emit <tool_call> or any JSON tool arguments. Emitting a tool call here is a mistake and the reply will be discarded.
- Write ONLY from the conversation above and your own knowledge.
- Put the three sections below inside <answer></answer>, as plain prose, starting with the literal text "Failure type:".

<answer>
Failure type: [incomplete / blocked / misdirected / format_missed]
  - incomplete: ran out of steps before finishing
  - blocked: got stuck due to tool failure or missing information
  - misdirected: went down the wrong path
  - format_missed: found the answer but never emitted it
What happened: [1-3 sentences: the approach taken and why a final answer was not reached]
Useful findings: [concrete facts, URLs, candidate answers, or dead-ends discovered that the next attempt should reuse or avoid; write "None" if nothing useful]
</answer>"""

FAILURE_SUMMARY_THINK = """I must write a structured post-mortem now, not an answer and not a tool call. I will put exactly these three sections inside <answer></answer>, starting with "Failure type:":

* Failure type: one of incomplete / blocked / misdirected / format_missed
* What happened: the approach taken and why it did not reach a final answer
* Useful findings: concrete facts, URLs, candidate answers, or dead-ends to reuse or avoid"""

# 第一次不合格时**只**纠正重问一次 —— 有界,绝不循环(保护评测吞吐)。
FAILURE_SUMMARY_RETRY_PROMPT = """Your previous reply was invalid: it was a tool call or a final answer, not a post-mortem. You CANNOT call tools now and must NOT give a final answer.

Reply with ONLY these three sections inside <answer></answer>, as plain prose, beginning with "Failure type:":
Failure type: [incomplete / blocked / misdirected / format_missed]
What happened: [1-3 sentences]
Useful findings: [concrete facts / URLs / candidate answers / dead-ends, or "None"]"""

# 两次都写不出来时的确定性兜底。保证下一次的任务描述**绝不**收到工具调用原文。
FAILURE_SUMMARY_FALLBACK = (
    "Failure type: incomplete\n"
    "What happened: The previous attempt ended without a usable post-mortem.\n"
    "Useful findings: None recorded. Try a substantially different search strategy "
    "and different query phrasings this time."
)

# ── 拼进下一次任务描述的外壳(Iris common_benchmark.py 同构)──────────────────
FAILURE_EXPERIENCE_HEADER = (
    "\n\n=== Previous Attempts Analysis ===\n"
    "The following summarizes what was tried before and why it didn't work. "
    "Use this to guide a NEW approach.\n\n"
)
FAILURE_EXPERIENCE_ITEM = "[Attempt {n}]\n{summary}\n\n"
FAILURE_EXPERIENCE_FOOTER = (
    "=== End of Analysis ===\n\n"
    "Based on the above, you should try a different strategy this time.\n"
)

_TOOLCALL_PAT = re.compile(
    r"<tool_call>|</tool_call>|<function\s*=|<parameter\s*=|<use_mcp_tool>",
    re.I)
_SECTIONS = (r"Failure type\s*:", r"What happened\s*:", r"Useful findings\s*:")


def enabled() -> bool:
    return os.environ.get("HYBRID_RETRY_SUMMARY", "") == "1"


def extract(text: str) -> str:
    """剥掉 <think> / <answer> 外壳,拿到后验正文。"""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", t, re.S)
    if m:
        t = m.group(1)
    # 兜底:即使模型仍吐了工具调用文本,也绝不让它流进下一次任务描述
    t = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", t, flags=re.S)
    return t.strip()


def is_usable(text: str) -> bool:
    body = extract(text)
    if not body:
        return False
    if _TOOLCALL_PAT.search(text or ""):
        return False
    return all(re.search(p, body, re.I) for p in _SECTIONS)


def compose_task_text(base_task: str, summaries: list[str]) -> str:
    """原题 + 累积的历次后验。每次都从原题重拼(与 Iris 一致),不做增量追加。"""
    if not summaries:
        return base_task
    out = base_task + FAILURE_EXPERIENCE_HEADER
    for i, s in enumerate(summaries, 1):
        out += FAILURE_EXPERIENCE_ITEM.format(n=i, summary=(s or "").strip())
    return out + FAILURE_EXPERIENCE_FOOTER
