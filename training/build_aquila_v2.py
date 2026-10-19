#!/usr/bin/env python3
"""把 XYZ-Aquila-SFT(7000 条)改写成本仓库 harness v2 的训练数据。

对齐三件事:
  1. 工具名 / 参数 / 观测文本 —— 全部换成我们框架里工具真实的输出格式;
  2. 上下文 —— 用推理时同一个 ReactHistory.render_multiturn_capped 渲染,
     token 上限沿用推理配置(HYBRID_MAX_INPUT_TOKENS,默认 224000),
     超了就先折叠切端前的旧历史、再折本段观测,和线上行为逐字一致;
  3. 系统提示词 —— 用 react_v2_prompts.system_prompt() + 当前侧工具 schema,
     不含任何 v1 台账字段。

段内多轮(方案 A):切端要换系统提示词,而 chat_template 禁止 system 出现在
非首位,所以只能在切端处重建;段内是标准多轮。Aquila 全是纯 S 路径、零 switch,
因此每条轨迹恰好一段 = 一条样本(相比每步一条省约 58x 训练 token)。

Aquila 侧形态:
  system / user(题目) / assistant(<think>+<tool_calls_begin><tool_call>…) /
  user(<tool_response>{…}</tool_response>) / … / assistant(末步 \\boxed{答案})
输出形态(每段一条多轮样本,loss 打在所有 assistant 轮上):
  {"messages":[system, user, assistant, user, assistant, …, assistant(答案)]}
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))

import react_v2 as rv2                      # noqa: E402
import react_v2_prompts as rv2p             # noqa: E402
from hybrid_prompts import TOOLS, SWITCH_TOOL, get_allowed_tool_names  # noqa: E402

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
TOOLRESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
# `\boxed{…}` 里的最终答案。
#
# 原来写的是 `\\boxed\{(.*)\}` + re.S —— 贪婪且跨行,会从**第一个** \boxed{ 一直吃到
# 全文**最后一个** }。Aquila 的推理里 \boxed{} 常出现多次(中途的候选答案),于是
# "答案"里混进了 </think>、后续整段推理、别的 \boxed{},末尾那个 } 还被当成收尾吃掉,
# 产出形如:
#     <answer>Splenda}.  However, let's also consider… </think> \boxed{Splenda</answer>
# 7,000 条末段样本里 3,440 条(49.1%)是这么坏的,直接教会模型两件事:
# `</think>` 可以提前吐、花括号可以不闭合。
#
# 改成不跨括号的非贪婪匹配,并且取**最后一个** —— 推理中途的候选不算数,最终结论才算。
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

# Aquila 工具 -> 我们的工具
TOOL_MAP = {
    "web_search": "search",
    "scrape_and_extract_info": "visit",
    "run_python_code": "code_sandbox",
}


def map_arguments(src_name: str, args: dict) -> dict:
    """把 Aquila 的参数改写成我们工具的 schema。"""
    if src_name == "web_search":
        q = args.get("query")
        if isinstance(q, str):
            q = [q]
        elif not isinstance(q, list):
            q = [str(q)]
        return {"query": [str(x) for x in q if str(x).strip()]}
    if src_name == "scrape_and_extract_info":
        out = {"url": str(args.get("url", ""))}
        goal = args.get("info_to_extract") or args.get("goal") or ""
        out["goal"] = str(goal)
        return out
    if src_name == "run_python_code":
        return {"language": "python", "code": str(args.get("code", ""))}
    return dict(args)


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


def render_observation(src_name: str, our_args: dict, payload: str) -> str:
    """把 Aquila 的 tool_response 渲染成我们工具真实的观测文本格式。"""
    try:
        obj = json.loads(payload)
    except Exception:
        obj = None

    if src_name == "web_search":
        queries = our_args.get("query") or [""]
        organic = (obj or {}).get("organic")
        if not isinstance(organic, list):
            return f"Search results for '{queries[0]}':\n\n{str(payload).strip()}"
        blocks = []
        for q in queries:
            lines = [f"Search results for '{q}':", ""]
            if not organic:
                return f"No search results found for '{q}'."
            for i, it in enumerate(organic, 1):
                if not isinstance(it, dict):
                    continue
                url = str(it.get("link") or it.get("url") or "")
                lines.append(f"{i}. {it.get('title', '')}")
                lines.append(f"URL: {url}")
                src = it.get("source") or _host(url)
                if src:
                    lines.append(f"Source: {src}")
                if it.get("date"):
                    lines.append(f"Date: {it['date']}")
                if it.get("snippet"):
                    lines.append(f"Snippet: {it['snippet']}")
                lines.append("")
            blocks.append("\n".join(lines).rstrip())
        return "\n\n".join(blocks)

    if src_name == "scrape_and_extract_info":
        content = (obj or {}).get("content") if isinstance(obj, dict) else None
        if content is None:
            content = str(payload)
        url = our_args.get("url", "")
        goal = our_args.get("goal", "")
        return (f"The useful information in '{url}' for user goal '{goal}':\n\n"
                f"Evidence in page:\n{str(content).strip()}")

    if src_name == "run_python_code":
        if isinstance(obj, dict):
            out = obj.get("stdout") or obj.get("output") or obj.get("result") or ""
            err = obj.get("stderr") or obj.get("error") or ""
        else:
            out, err = str(payload), ""
        body = [
            "Code sandbox execution result:",
            "- status: Success",
            "- message: ",
            "Compile result: [not applicable]",
            "Run result:",
            "- status: Finished",
            "- return_code: 0",
            "- stdout:",
            str(out).rstrip(),
        ]
        if str(err).strip():
            body += ["- stderr:", str(err).rstrip()]
        return "\n".join(body)

    return str(payload)


# Aquila 题面末尾统一带这句,和我们的 <answer> 收口格式冲突,必须去掉。
BOXED_INSTR_RE = re.compile(
    r"\s*Follow the request'?s format instructions strictly and wrap the final "
    r"answer in \\boxed\{\}\.\s*$", re.I)


def clean_question(q: str) -> str:
    q = BOXED_INSTR_RE.sub("", q or "")
    # 兜底:任何残留的 boxed 指令句也一并删掉
    q = re.sub(r"[^.\n]*wrap[^.\n]*\\boxed\{\}[^.\n]*\.\s*", "", q, flags=re.I)
    return q.strip()


def system_text(current_date: str) -> str:
    """v2 的 S 侧系统提示词 + 当前侧工具 schema —— 与推理时逐字相同的那一份。

    switch 照常暴露(expose_switch=True、schema 里保留),尽管 Aquila 的 7000 条
    轨迹里一次都没调过它。理由是训推必须一致:推理侧是暴露的,训练里说"没有这个
    工具"会让模型在两种提示词下工作。代价是模型学到"有 switch 但从不用",这在
    纯检索任务上无所谓(实测 211 道题只调了 6 次)。
    """
    allowed = set(get_allowed_tool_names("hybrid_react_v2", "S"))
    schema = [
        {"name": (t.get("function") or {}).get("name"),
         "description": (t.get("function") or {}).get("description"),
         "parameters": (t.get("function") or {}).get("parameters")}
        for t in TOOLS + [SWITCH_TOOL]
        if (t.get("function") or {}).get("name") in allowed
    ]
    return (rv2p.system_prompt("S", current_date, expose_switch=True)
            + "\n\n# Tool Schemas\n"
            + json.dumps(schema, ensure_ascii=False, indent=2))


def flatten(msgs: list[dict]) -> list[dict]:
    """render_rebuilt 产出的是多模态 content 列表;纯文本样本压成普通字符串,
    和 noidle-v7 的训练数据格式保持一致。带图片的原样保留。"""
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            if all(isinstance(x, dict) and x.get("type") == "text" for x in c):
                c = "".join(x.get("text", "") for x in c)
        out.append({"role": m["role"], "content": c})
    return out


def parse_traj(d: dict):
    """拆出 (题目, [(think, our_tool_call, 观测文本), …], 末步 think, 最终答案)。"""
    tr = d.get("trajectory") or []
    sys_msg = next((m for m in tr if m["role"] == "system"), None)
    date = ""
    if sys_msg:
        m = re.search(r"Today is:\s*(\S+)", sys_msg["content"])
        if m:
            date = m.group(1)
    users = [m for m in tr if m["role"] == "user"]
    if not users:
        return None
    question = clean_question(users[0]["content"])
    if not question:
        return None

    steps, final_think, final_answer = [], "", ""
    idx = [i for i, m in enumerate(tr) if m["role"] == "assistant"]
    for i in idx:
        content = tr[i]["content"]
        think = (THINK_RE.search(content).group(1).strip()
                 if THINK_RE.search(content) else "")
        tcm = TOOLCALL_RE.search(content)
        if not tcm:
            # 取**最后一个** \boxed{} —— 推理中途会出现候选答案,最终结论在末尾。
            hits = BOXED_RE.findall(content)
            final_think, final_answer = think, (hits[-1].strip() if hits else "")
            continue
        try:
            call = json.loads(tcm.group(1))
        except Exception:
            return None
        src_name = call.get("name")
        if src_name not in TOOL_MAP:
            return None
        our_args = map_arguments(src_name, call.get("arguments") or {})
        obs = ""
        if i + 1 < len(tr) and tr[i + 1]["role"] == "user":
            rm = TOOLRESP_RE.search(tr[i + 1]["content"])
            if rm:
                obs = render_observation(src_name, our_args, rm.group(1))
        steps.append((think, {"name": TOOL_MAP[src_name], "arguments": our_args}, obs))
    if not final_answer:
        return None
    return question, steps, final_think, final_answer, date


# 模型窗口。整条样本的完整序列不能超它;末步 prompt 还要留出输出预算。
# 由 --max-model-len 覆盖 —— 训练窗口改了,切分结果整个都要重算。
MAX_MODEL_LEN = 262144
# chat template 每条消息约加 5 token(<|im_start|>role\n … <|im_end|>\n)。
# 整条多轮样本动辄上百条消息,不能当成固定常数 —— 取 8 留余量。
TEMPLATE_PER_MSG = 8
# 估算值逼近上限时改用真实模板精确核算(估算有分段计数误差,不能只靠它把关)。
EXACT_CHECK_RATIO = 0.9

_TPL_TOK = {}


def _exact_tokens(msgs, model_path, generation_prompt=True):
    """用真实 chat template 渲染后计数。仅在逼近上限时调用。"""
    tok = _TPL_TOK.get(model_path)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _TPL_TOK[model_path] = tok
    plain = [{"role": m["role"], "content": m["content"]} for m in flatten(msgs)]
    s = tok.apply_chat_template(plain, tokenize=False,
                                add_generation_prompt=generation_prompt)
    return len(tok.encode(s, add_special_tokens=False))


def build_multiturn(d: dict, max_input_tokens: int, model_path: str) -> tuple:
    """整条轨迹一条多轮样本。不折叠 —— 超出窗口的直接跳过,交给调用方计数。

    返回 (samples, skip_reason)。skip_reason 为 None 表示成功。

    为什么不折叠:折叠是"随步数推进逐条发生"的,一条样本里每个观测只能存一个
    版本。选最终折叠态,就等于让早期那些 assistant 轮"在看不到全文的历史上做
    决策",而推理时它们当时看到的是全文 —— 训推对不上。忠实做法是在每次折叠处
    切一段,但实测 299 步那条从第 202 步起折了 92 次,切完就退化成单轮了。
    """
    parsed = parse_traj(d)
    if parsed is None:
        return [], "parse_failed"
    question, steps, final_think, final_answer, date = parsed
    sys_text = system_text(date)

    hist = rv2.ReactHistory(max_o_screenshots=2, max_input_tokens=10 ** 9,
                            fold_keep_chars=10 ** 9, model_path=model_path)
    for think, call, obs in steps:
        hist.add_step("S", json.dumps(call, ensure_ascii=False), obs, think)

    msgs = hist.render_multiturn(sys_text, "S", question)
    final = f"<think>\n{final_think}\n</think>\n<answer>\n{final_answer}\n</answer>"

    # 唯一约束:整条序列 ≤ 模型窗口(256k)。
    # 不套推理时"末步 prompt ≤ 224k 以留出输出预算"那条 —— 那是给在线生成留的,
    # 训练样本里最后一轮的答案本来就是给定的,不需要再留生成空间。
    total_tok = (hist._tokens_of(msgs) + rv2.count_tokens(final, model_path)
                 + TEMPLATE_PER_MSG * (len(msgs) + 1))
    if total_tok > MAX_MODEL_LEN * EXACT_CHECK_RATIO:
        total_tok = _exact_tokens(msgs + [{"role": "assistant",
                                           "content": [{"type": "text", "text": final}]}],
                                  model_path, generation_prompt=False)
    if total_tok > MAX_MODEL_LEN:
        return [], "over_model_len"

    return [{"messages": flatten(msgs) + [{"role": "assistant", "content": final}]}], None


def build_multiturn_split(d: dict, model_path: str,
                          prior_fold_chars: int = rv2.PRIOR_FOLD_CHARS,
                          fold_high: int = rv2.FOLD_HIGH,
                          fold_low: int = rv2.FOLD_LOW) -> tuple:
    """超窗口的轨迹按**水位线**切成多段,与推理端逐字同构。

    推理端 (ReactHistory.render_multiturn_capped) 的规则:
      * 上下文涨到 fold_high 才动手,一动就折到 fold_low 以下;
      * 折的时候把切点推到只剩末步,历史块里从最早的观测起依次折
        (丢思考 + 观测截到 prior_fold_chars),直到降到 fold_low;
      * 折过的步永久保持折叠态。
    每一次折叠在训练里就是一个切点:折之前那段是模型在满上下文下做的决策,
    折之后是它在压缩历史上做的决策,两种形态都要学到。

    所以每段的构造是:
      1. 段起点处先定折叠量 —— 折到该点的重建上下文 ≤ fold_low;
      2. 折叠量固定不动,段往后长到 ≤ fold_high 为止。
    (老做法是"二分找最远 + 回头求最小折叠量",目标只有一个上限;水位线有两个,
     且折叠量由**切点**决定而不是由段尾决定,否则训练里的折叠时机和线上对不上。)

    非末段丢掉本段最后一条观测,让样本以 assistant 收尾(训练需要待预测的目标)。
    """
    parsed = parse_traj(d)
    if parsed is None:
        return [], "parse_failed"
    question, steps, final_think, final_answer, date = parsed
    sys_text = system_text(date)
    final = f"<think>\n{final_think}\n</think>\n<answer>\n{final_answer}\n</answer>"
    n = len(steps)

    def render(seg_from: int, seg_to: int, n_folded: int, with_final: bool):
        """渲染 [seg_from, seg_to);n_folded = 历史块里从最早起折掉几步。

        只折历史块(下标 < seg_from),**当前段一律不折** —— 让当前段参与折叠会
        把本来要切两段的轨迹压成一段,等于改了切点。
        """
        h = rv2.ReactHistory(max_o_screenshots=2, max_input_tokens=10 ** 9,
                             fold_keep_chars=prior_fold_chars, model_path=model_path)
        h.prior_fold_chars = prior_fold_chars
        for think, call, obs in steps[:seg_to]:
            h.add_step("S", json.dumps(call, ensure_ascii=False), obs, think)
        tl = [t for t in h.turns if t.get("role") == "step"]
        for t in tl[:min(n_folded, seg_from)]:
            h.folded.add(id(t))
        msgs = h.render_multiturn(sys_text, "S", question, seg_start=seg_from,
                                  drop_last_observation=not with_final)
        tail = rv2.count_tokens(final, model_path) if with_final else 0
        est = h._tokens_of(msgs) + tail + TEMPLATE_PER_MSG * (len(msgs) + 2)
        if est > fold_high * EXACT_CHECK_RATIO:
            extra = ([{"role": "assistant", "content": [{"type": "text", "text": final}]}]
                     if with_final else [])
            est = _exact_tokens(msgs + extra, model_path, generation_prompt=not with_final)
        return est, msgs

    out, s, n_folded = [], 0, 0
    while s < n:
        # 1) 切点处定折叠量:折到"历史块 + 本段第一步"降到低水位以下。
        #    折叠是永久的,所以只增不减。s == 0 时没有历史块,自然是 0。
        if s > 0:
            lo, hi, k = n_folded, s, s
            while lo <= hi:
                mid = (lo + hi) // 2
                est, _ = render(s, s + 1, mid, with_final=(s + 1 == n))
                if est <= fold_low:
                    k = mid; hi = mid - 1
                else:
                    lo = mid + 1
            n_folded = max(n_folded, k)      # 全折也够不到低水位时就是全折
        # 2) 折叠量固定,本段往后长到高水位为止
        lo, hi, best_e, best_msgs = s + 1, n, None, None
        while lo <= hi:
            mid = (lo + hi) // 2
            est, msgs = render(s, mid, n_folded, with_final=(mid == n))
            if est <= fold_high:
                best_e, best_msgs = mid, msgs; lo = mid + 1
            else:
                hi = mid - 1
        if best_e is None:
            return [], "single_step_too_big"
        tail = ([{"role": "assistant", "content": final}] if best_e == n else [])
        out.append({"messages": flatten(best_msgs) + tail})
        s = best_e
        if len(out) > 64:
            return [], "too_many_segments"
    return out, None


def build_samples(d: dict, max_input_tokens: int, model_path: str,
                  fold_keep_chars: int) -> list[dict]:
    parsed = parse_traj(d)
    if parsed is None:
        return []
    question, steps, final_think, final_answer, date = parsed
    sys_text = system_text(date)

    hist = rv2.ReactHistory(max_o_screenshots=2, max_input_tokens=max_input_tokens,
                            fold_keep_chars=fold_keep_chars, model_path=model_path)
    # 单轮:每步一条样本,上下文用推理时同一个 render_rebuilt 渲染。
    # (段内多轮方案已实现在 ReactHistory.render_multiturn_capped,但长轨迹一旦
    #  触发折叠,前缀几乎每步都被改写 —— 实测 299 步那条从第 202 步起触发了 92
    #  次折叠,忠实切分要切成 93 段,等于退回单轮,所以直接用单轮。)
    tln = int(os.environ.get("HYBRID_THINK_LAST_N", "-1"))
    out = []
    for think, call, obs in steps:
        msgs = hist.render_rebuilt(sys_text, "S", question, think_last_n=tln)
        assistant = (f"<think>\n{think}\n</think>\n"
                     f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>")
        out.append({"messages": flatten(msgs) + [{"role": "assistant", "content": assistant}]})
        hist.add_step("S", json.dumps(call, ensure_ascii=False), obs, think)

    msgs = hist.render_rebuilt(sys_text, "S", question, think_last_n=tln)
    final = f"<think>\n{final_think}\n</think>\n<answer>\n{final_answer}\n</answer>"
    out.append({"messages": flatten(msgs) + [{"role": "assistant", "content": final}]})
    return out


def main():
    global MAX_MODEL_LEN            # --max-model-len 会改写它,声明须早于任何引用
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-input-tokens", type=int,
                    default=int(os.environ.get("HYBRID_MAX_INPUT_TOKENS", "224000")))
    ap.add_argument("--fold-keep-chars", type=int, default=600)
    ap.add_argument("--model-path", default=os.environ.get(
        "HYBRID_TOKENIZER_PATH", "yzxjb/HyWe-35B"))
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--prior-fold-chars", type=int, default=rv2.PRIOR_FOLD_CHARS,
                    help="multiturn 切分时,切点之前每条观测折到多少字符")
    ap.add_argument("--no-split", action="store_true",
                    help="超窗口的直接跳过,不切分")
    ap.add_argument("--fold-high", type=int, default=rv2.FOLD_HIGH,
                    help="折叠触发水位。整条超过它才切分/折叠;也是样本长度上限")
    ap.add_argument("--fold-low", type=int, default=rv2.FOLD_LOW,
                    help="折叠目标水位。切点处一次折到它以下,与推理端同值")
    ap.add_argument("--mode", choices=("perstep", "multiturn"), default="perstep",
                    help="perstep=每步一条样本(会折叠);multiturn=整条一条样本(超窗口的跳过)")
    a = ap.parse_args()

    n_traj = n_skip = n_samp = n_split = 0
    skip_reasons = {}
    seg_hist = {}
    # 每步样本都完整重复一遍历史,明文体积极大(全量约 106 GB);
    # .gz 输出信息无损,观测文本重复度高,实测能压到 1/5 以下。
    if a.stats_only:
        fout = None
    elif a.out.endswith(".gz"):
        fout = gzip.open(a.out, "wt", encoding="utf-8", compresslevel=6)
    else:
        fout = open(a.out, "w")
    tok_lens, step_counts = [], []
    # build_multiturn 直接读全局常量做"整条装不装得下"的判定,和切分用同一个水位。
    MAX_MODEL_LEN = a.fold_high

    for src in a.srcs:
        for line in open(src):
            if a.limit and n_traj >= a.limit:
                break
            d = json.loads(line)
            if a.mode == "multiturn":
                samples, why = build_multiturn(d, a.max_input_tokens, a.model_path)
                if why == "over_model_len" and not a.no_split:
                    # 整条装不下 -> 切成尽可能少的多轮段(切点处激进折叠)
                    samples, why = build_multiturn_split(
                        d, a.model_path, prior_fold_chars=a.prior_fold_chars,
                        fold_high=a.fold_high, fold_low=a.fold_low)
                    if samples:
                        n_split += 1
                        seg_hist[len(samples)] = seg_hist.get(len(samples), 0) + 1
            else:
                samples, why = build_samples(d, a.max_input_tokens, a.model_path,
                                             a.fold_keep_chars), "parse_failed"
            n_traj += 1
            if not samples:
                n_skip += 1
                skip_reasons[why] = skip_reasons.get(why, 0) + 1
                continue
            step_counts.append(len(samples))
            for s in samples:
                n_samp += 1
                if fout:
                    fout.write(json.dumps(s, ensure_ascii=False) + "\n")
            if a.stats_only:
                last = samples[-1]["messages"]
                tok_lens.append(rv2.count_tokens(
                    "".join(str(m.get("content")) for m in last), a.model_path))
            if n_traj % 100 == 0:
                print(f"  {n_traj} 条轨迹 / {n_samp} 条样本", file=sys.stderr, flush=True)
    if fout:
        fout.close()
    print(f"\n轨迹 {n_traj}(跳过 {n_skip}) -> 样本 {n_samp}")
    if skip_reasons:
        print("跳过原因: " + ", ".join(f"{k}={v}" for k, v in sorted(skip_reasons.items())))
    if n_split:
        print(f"切分的轨迹: {n_split} 条 -> 段数分布 "
              + ", ".join(f"{k}段×{v}" for k, v in sorted(seg_hist.items())))
    if step_counts:
        step_counts.sort()
        print(f"每条轨迹样本数: 中位 {step_counts[len(step_counts)//2]} "
              f"| P90 {step_counts[int(len(step_counts)*.9)]} | max {step_counts[-1]}")
    if tok_lens:
        tok_lens.sort()
        print(f"末步 prompt token: 中位 {tok_lens[len(tok_lens)//2]:,} "
              f"| P90 {tok_lens[int(len(tok_lens)*.9)]:,} | max {tok_lens[-1]:,}")


if __name__ == "__main__":
    main()
