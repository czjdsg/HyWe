#!/usr/bin/env python3
"""harness v2 运行时:append-only 消息历史 + 截图裁剪 + 输出解析。

设计要点
  1. 历史全量保留:每一步的 assistant 输出和 tool 观测都留在 message 列表里,
     不再靠模型自写 cumulative_progress 当记忆(v1 那样每步压缩会丢信息:
     实测"答案出现在观测里 -> 进入累计进度"只有 44-50% 的留存率)。
  2. O 端截图只保留最近两张:更早的图片替换成一行文字占位,避免图片撑爆上下文;
     文字观测不裁剪。
  3. 解析:<think>...</think> + <tool_call>{...}</tool_call>,或 <answer>...</answer>。
"""
from __future__ import annotations

import json
import os
import re

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
# 开思考时 chat_template 已经在 prompt 末尾吐出了 `<think>\n`,所以 response body 里
# 只剩闭合的 `</think>`——此时 `</think>` 之前的全部内容就是思考。
CLOSE_THINK_RE = re.compile(r"</think>", re.I)
# 只有开标签的兜底,见 extract_think
OPEN_THINK_RE = re.compile(r"<think\s*>(.+)$", re.S | re.I)
# 特殊 token 泄漏进正文:实测 kimi-k3 有 10% 的步出现 `<|sep|>`(还见过 `<|close|>`),
# 典型形态是把开标签劈成 `<think<|sep|>>`,于是 think 抽不出来、答案步整步解析失败,
# 段就断在普通工具上(151 条样本里 14 条末段停在 visit/click/search)。
# 这类 token 不可能是合法内容,解析前一律剥掉。
SPECIAL_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>", re.I)
TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S | re.I)
# 严格式要求有右括号才匹配;模型把括号漏光时用这条兜底,交给 balance_braces 补。
LOOSE_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.+?)\s*(?:</tool_call>|$)", re.S | re.I)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
# 未闭合的 <answer>:模型写到一半被截断,尾部没有 </answer>。
# 2026-09-01 实测 kimi 会这样,占重启后 format_error 的 7/8 —— 一整段写好的答案
# (2249 字)因为少一个闭合标签被判"没找到 answer",整条轨迹作废。
# 开标签之后没有闭标签时,后面全部就是答案,没有歧义;预填收口那条路早就在做
# 同样的补全("模型被 max_tokens 截断、没写出 </answer> 时补上"),这里只是把
# 同一个判断挪到通用路径上。
OPEN_ANSWER_RE = re.compile(r"<answer\s*>(?!.*</answer\s*>)(.+)$", re.S | re.I)
# Kimi's OpenAI-compatible gateway may serialize native tool calls into the
# message content instead of populating ``message.tool_calls``.  Example:
#   <|tool_call_begin|>functions.search:2
#   <|tool_call_argument_begin|>{"query": [...]}
#   <|tool_call_end|>
# The optional numeric suffix is a gateway call index, not part of the name.
# 2026-09-01:网关上的 qwen3.8-max 开始吐另外两种工具调用写法,提示词没变、
# 模型名没变,是服务端行为漂移。实测 8451 道新批跑到 162 条时 format_error 占 35%
# (8500 那轮全程只有 0.1%)。两种都只出现在 content 里,`message.tool_calls` 始终为空,
# 所以只能在文本层归一化 —— 和上面 Kimi 那条同一个思路。
#
#   ① XML 参数式(Hermes 风格):
#      <tool_call name="goto"><parameter name="url">http://a</parameter></tool_call>
#   ② Qwen 原生标记泄漏成正文:
#      <tool_call_tool_name>goto</tool_call_tool_name>
#      <tool_call_args>{"url": "http://a"}</tool_call_args>
#
# 注意闭合标签里可能带空格(`</parameter >`),所以每个标签都写成 \s*>。
# 引号前可能带反斜杠(模型把整段又转义了一遍:`<tool_call name=\\"search\\">`),
# 开标签前也可能粘上杂字符(实测见过 `<tool_call]>`),所以引号写成 \\\\?["\'] 。
QWEN_XML_TOOLCALL_RE = re.compile(
    r'<tool_call\s+name\s*=\s*\\?["\']([^"\'\\]+)\\?["\'][^>]*>(.*?)(?:</tool_call\s*>|$)',
    re.S | re.I)
QWEN_XML_PARAM_RE = re.compile(
    r'<parameter\s+name\s*=\s*\\?["\']([^"\'\\]+)\\?["\'][^>]*>(.*?)</parameter\s*>',
    re.S | re.I)
QWEN_NATIVE_LEAK_RE = re.compile(
    r'<tool_call_tool_name\s*>\s*(.+?)\s*</tool_call_tool_name\s*>\s*'
    r'<tool_call_args\s*>\s*(.*?)\s*</tool_call_args\s*>', re.S | re.I)


_XML_BODY_ARGS_RE = re.compile(r"\barguments\s*[:=]\s*(\{)", re.I)


def _qwen_xml_to_json(match: "re.Match") -> str:
    """XML 参数式 → 标准 <tool_call>{json}</tool_call>。

    参数值一律先试 json.loads(数字/布尔/对象/数组要保型),失败就当字符串。
    """
    name = match.group(1)
    body = match.group(2) or ""
    args = {}
    for k, v in QWEN_XML_PARAM_RE.findall(body):
        v = v.strip()
        try:
            args[k] = json.loads(v)
        except Exception:
            args[k] = v
    if not args:
        # 没有 <parameter> 标签时,体里常常是裸的 `arguments: {...}` 或 `arguments={...}`
        # ——qwen3.6-flash 用的就是这一种。原来只认 <parameter>,于是把参数整个吞掉,
        # 返回 arguments={} ;下游校验缺必填参数 → 格式重试 → 5 次耗尽判 format_error。
        # 实测 flash 轮 19 条 format_error 里 17 条是这个形态(不是 max_tokens 截断:
        # 回复中位才 535 字符)。允许 `=` 或 `:`,也允许直接就是一个 JSON 对象。
        m = _XML_BODY_ARGS_RE.search(body)
        start = m.start(1) if m else (body.find("{") if body.lstrip().startswith("{") else -1)
        if start >= 0:
            obj, _ = _balanced_json(body, start)
            if isinstance(obj, dict):
                args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj
    return "<tool_call>" + json.dumps({"name": name, "arguments": args},
                                      ensure_ascii=False) + "</tool_call>"


def _qwen_native_to_json(match: "re.Match") -> str:
    name = (match.group(1) or "").strip()
    raw = (match.group(2) or "").strip()
    try:
        args = json.loads(raw) if raw else {}
    except Exception:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return "<tool_call>" + json.dumps({"name": name, "arguments": args},
                                      ensure_ascii=False) + "</tool_call>"


KIMI_TOOLCALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?:functions\.)?([A-Za-z_]\w*)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*"
    r"<\|tool_call_end\|>",
    re.S | re.I,
)
# 容错:有些模型会吐 ```json 包裹或直接裸 JSON
BARE_JSON_RE = re.compile(r"\{\s*\"name\"\s*:\s*\"[a-z_]+\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\}", re.S)


def _parse_loose_argument_value(value: str):
    text = re.sub(r"</?arg_(?:key|value)>", "", value or "", flags=re.I).strip()
    text = text.strip().rstrip(",").strip()
    if not text:
        return ""
    try:
        return json.loads(balance_braces(text))
    except Exception:
        return text.strip('"\'')


def parse_named_arguments_toolcall(
    raw: str,
    tool_params: dict | None = None,
) -> dict | None:
    """Parse gateway-style ``tool_name\n{arguments}`` tool calls.

    Some OpenAI-compatible model templates serialize a tool call as the tool
    name followed by its arguments object, rather than the unified
    ``{"name": ..., "arguments": ...}`` object requested by the prompt.  The
    mapping is unambiguous, so normalize it instead of terminating the rollout.
    """
    value = (raw or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    match = re.match(r"^([A-Za-z_]\w*)\s*(.*)$", value, re.S)
    if not match:
        return None
    name = match.group(1)
    payload = match.group(2).strip()

    # Standard arguments object after the tool name.
    if payload.startswith("{"):
        try:
            arguments = json.loads(balance_braces(payload))
        except Exception:
            arguments = None
        if isinstance(arguments, dict):
            return {"name": name, "arguments": arguments}

    # GLM/Qwen gateways may expose their native template as either
    #   <arg_key>query</arg_key><arg_value>[...]</arg_value>
    # or the partially stripped form
    #   query: [...]</arg_value>
    # Normalize complete XML pairs into one-key-per-line text first.
    payload = re.sub(
        r"<arg_key>\s*([^<]+?)\s*</arg_key>\s*"
        r"<arg_value>\s*(.*?)\s*</arg_value>",
        lambda m: f"\n{m.group(1).strip()}: {m.group(2).strip()}\n",
        payload,
        flags=re.I | re.S,
    )
    schema = (tool_params or {}).get(name) or {}
    property_names = list((schema.get("properties") or {}).keys())
    key_pattern = (
        "|".join(re.escape(key) for key in sorted(property_names, key=len, reverse=True))
        if property_names
        else r"[A-Za-z_]\w*"
    )
    matches = list(
        re.finditer(
            rf"(?m)(?:^|\n)\s*({key_pattern})\s*:\s*",
            payload,
        )
    )
    if matches:
        arguments = {}
        for index, item in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(payload)
            arguments[item.group(1)] = _parse_loose_argument_value(
                payload[item.end():end]
            )
        return {"name": name, "arguments": arguments}

    # A single unnamed payload can still be mapped unambiguously when the
    # schema has exactly one required parameter.
    required = schema.get("required") or []
    if len(required) == 1 and payload:
        return {
            "name": name,
            "arguments": {required[0]: _parse_loose_argument_value(payload)},
        }
    return None


def balance_braces(s: str) -> str:
    """补齐未闭合的 {} [] 和字符串。模型偶尔漏最后一个右括号(实测 0.7% 的步),
    而一次解析失败就会烧掉一整道题——这里兜住,不必靠重试。"""
    out = []
    stack = []
    instr = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            out.append(ch)
            continue
        if instr and ch == "\\":
            esc = True
            out.append(ch)
            continue
        if ch == '"':
            instr = not instr
        elif not instr:
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    continue  # 多余的右括号,丢掉
                stack.pop()
        out.append(ch)
    if instr:
        out.append('"')
    out.extend("}" if ch == "{" else "]" for ch in reversed(stack))
    return "".join(out)


_MISSING_ARGS_RE = re.compile(
    r'("name"\s*:\s*"[A-Za-z_][\w]*"\s*,\s*)(?=\{)', re.S)


_POSITIONAL_Y_RE = re.compile(
    r'("x"\s*:\s*-?\d+(?:\.\d+)?\s*,\s*)(-?\d+(?:\.\d+)?)(\s*[},])')


def repair_positional_y(s: str) -> str:
    """`{"x": 782, 60}` -> `{"x": 782, "y": 60}` —— 唯一一个真去修的坏法。

    其余坏法一律只诊断不改写(改写会掩盖模型的错误,也有猜错意图的风险)。
    这一条例外的理由是它既确定又高频:

      * **确定**:click / double_click / hover 的 required 就是有序的 ["x","y"],
        模型写的正好是两个数,第二个只可能是 y,没有别的解释;
      * **高频**:实测 base 模型在 O 侧 44.9% 的 click 栽在这里,而不带坐标的
        工具(goto/search/type/answer/switch…)失败率只有 0.6% —— 格式本身它会写,
        垮的只有这一个参数形状。单步 45% 出错,一条轨迹点十几次,活下来的概率
        是 0.55ⁿ,几步归零;不修的话评测量到的是"会不会补键名",不是操作能力。

    修完必须能 json.loads 成功才采用,失败就退回报格式错误。
    """
    return _POSITIONAL_Y_RE.sub(r'\1"y": \2\3', s)


def repair_missing_keys(s: str) -> str:
    """补回模型漏写的键名。实测两种:
      {"name":"search", {"query":[…]}}   -> 漏了 "arguments":
      {"x": 167, 416}                    -> 漏了 "y":(click 的坐标)
    都是结构性非法 JSON,补括号救不了,json5 也解析不了。
    """
    s = _MISSING_ARGS_RE.sub(r'\1"arguments": ', s)
    # click 少写 "y":  {"x": 167, 416}
    s = re.sub(r'("x"\s*:\s*-?\d+(?:\.\d+)?\s*,\s*)(-?\d+(?:\.\d+)?\s*\})',
               r'\1"y": \2', s)
    return s


# 函数调用语法。第一版要求 `name(ident=`,只认出 `search(query=[…])` 一种写法,
# 漏掉了实测里更常见的另外三种:
#     search("query": ["…"])      括号里直接是 JSON 片段
#     search("a" "b")             位置参数、没有 =
#     =search("…")                前面莫名多个 =
# 判据可以放得很宽:合法 tool_call 必然以 `{` 开头,所以"标识符 + 左括号"开头的
# 一律是函数调用,不会误伤。漏判的代价很实在 —— 模型收到的反馈说不出错在哪,
# 会一直往同一个方向改(实测有轨迹自述"上次格式错了,这次改对"然后又写成函数调用)。
PYCALL_RE = re.compile(r"^\s*=?\s*([A-Za-z_]\w*)\s*\(", re.S)

_ARGS_BRACKET_RE = re.compile(r'("arguments"\s*:\s*)\[')


def repair_bracket_args(s: str) -> str:
    """`"arguments": ["query": […]]` —— 该用 {} 的地方用了 []。

    补括号和补键名都救不了它(语法层面是"列表里出现了 key:"),而默认的报错
    "每个键必须是带引号的字符串"是误导 —— 键确实带引号,错的是括号类型。
    这里把 arguments 那一对 [] 换成 {},只用于诊断,不改模型输出。
    """
    m = _ARGS_BRACKET_RE.search(s)
    if not m:
        return s
    i = m.end() - 1                       # 指向那个 [
    depth, instr, esc = 0, False, False
    for j in range(i, len(s)):
        ch = s[j]
        if esc:
            esc = False
            continue
        if instr:
            if ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return s[:i] + "{" + s[i + 1:j] + "}" + s[j + 1:]
    return s


def diagnose_json(raw: str) -> str:
    """判断这段坏 JSON 属于哪种已知坏法,给模型一句能照着改的话。

    只诊断、不改写——改写会掩盖模型的错误,也会有猜错意图的风险。
    每条都用"能否被对应修复解析成功"来验证,避免误报。
    """
    # 函数调用语法:search(query=[...])。实测模型收到"格式不对"的反馈后,会误以为
    # 是 JSON 里有坏字符,干脆换成 Python 写法 —— 不点破的话它会一直往这个方向改。
    m = PYCALL_RE.match(raw)
    if m:
        return (f"You wrote a function-call expression `{m.group(1)}(...)`, not JSON. "
                "Tool calls are JSON objects, never Python/JS call syntax. "
                f'Write {{"name": "{m.group(1)}", "arguments": {{"<param>": <value>}}}}. ')
    hints = []
    for repair, hint in (
        (repair_bracket_args,
         'You wrapped `arguments` in square brackets — `"arguments": ["key": …]`. '
         "`arguments` is an object, so it takes curly braces: "
         '`"arguments": {"key": …}`. Square brackets are only for list values '
         "such as `search`'s `query`. "),
        (repair_missing_keys,
         'You omitted a key name — write `"arguments": {...}` in full, and give '
         "every value its parameter name (e.g. both `x` and `y` for a click). "),
        (balance_braces,
         "The braces/brackets are unbalanced — count them and close every one. "),
        (lambda s: balance_braces(repair_missing_keys(s)),
         "A key name is missing and the braces are unbalanced. "),
    ):
        try:
            fixed = repair(raw)
        except Exception:
            continue
        if fixed == raw:
            continue
        try:
            json.loads(fixed)
        except Exception:
            continue
        hints.append(hint)
        break
    return hints[0] if hints else ""


# think 正文里混进来的标签碎片。模型会写着写着**中途重起一个 <think> 块**
# (前后两段都是真实推理),提取出来就是 "推理A<think>\n推理B";导出侧再套一层
# 规范标签,训练样本里就出现嵌套 —— 实测 19.27% 的 assistant 步中招,
# 学生会学到吐嵌套 think。think 内容里永远不该有 think 标签,一律剥掉。
STRAY_THINK_RE = re.compile(r"</?think\s*>", re.I)


def extract_think(text: str) -> str:
    """取出思考内容,兼容 chat_template 已吐出开标签、body 里只剩 </think> 的情况。"""
    text = SPECIAL_TOKEN_RE.sub("", text or "")
    m = THINK_RE.search(text)
    if m:
        return STRAY_THINK_RE.sub("", m.group(1)).strip()
    m = CLOSE_THINK_RE.search(text)
    if m:
        return STRAY_THINK_RE.sub("", text[: m.start()]).strip()
    # 只有开标签、没有闭标签:模型写着写着直接转去发工具调用了。
    # 2026-09-01 实测 kimi-k3 有 4.6% 的步是这形态,而两条老路径都返回空 ——
    # 于是 build_hybrid_teacher_v2 拿到 think="",整个 <think> 块不进训练样本,
    # 一段真实存在的推理被丢掉;泄露改写那边也因为"该步没有 think 文本"跳过。
    # 开标签之后到第一个动作标记之间就是思考,没有歧义。
    m = OPEN_THINK_RE.search(text)
    if m:
        body = m.group(1)
        cut = min([i for i in (body.find("<tool_call"), body.find("<answer"))
                   if i >= 0] or [len(body)])
        return STRAY_THINK_RE.sub("", body[:cut]).strip()
    return ""


def json_output_mode() -> bool:
    """JSON 输出模式开关,与 react_v2_prompts._json_mode 同一个 env。"""
    return os.environ.get("HYBRID_JSON_OUTPUT", "") == "1"


def parse_json_output(text: str, tool_params: dict | None = None) -> dict:
    """解析 JSON 模式的输出:{"cot":…, "tool_call":{…}} 或 {"cot":…, "answer":…}。

    配合 response_format=json_object 使用时,结构由解码层保证,这里只做取值和
    最基本的形状校验。仍留一层容错(代码块包裹、前后有杂字符),因为强制解码
    不是所有网关都支持。
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        i = t.find("{")
        if i >= 0:
            t = t[i:]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return {"think": "", "tool_call": None, "answer": "", "ok": False,
                "error": 'Output must be one JSON object: {"cot": "...", '
                         '"tool_call": {"name": ..., "arguments": {...}}} '
                         'or {"cot": "...", "answer": "..."}.'}
    try:
        d = json.loads(t[i:j + 1])
    except Exception as exc:
        return {"think": "", "tool_call": None, "answer": "", "ok": False,
                "error": f"Invalid JSON: {exc}"}
    if not isinstance(d, dict):
        return {"think": "", "tool_call": None, "answer": "", "ok": False,
                "error": "Top level must be a JSON object."}
    cot = str(d.get("cot") or d.get("think") or "").strip()
    ans = d.get("answer")
    tc = d.get("tool_call")
    if isinstance(tc, dict) and tc.get("name"):
        args = tc.get("arguments")
        if not isinstance(args, dict):
            args = {} if args in (None, "") else {"_raw": args}
        return {"think": cot, "tool_call": {"name": str(tc["name"]), "arguments": args},
                "answer": "", "ok": True, "error": ""}
    if isinstance(ans, str) and ans.strip():
        return {"think": cot, "tool_call": None, "answer": ans.strip(),
                "ok": True, "error": ""}
    return {"think": cot, "tool_call": None, "answer": "", "ok": False,
            "error": 'JSON must carry exactly one of "tool_call" '
                     '(with a "name") or a non-empty "answer".'}


def parse_output(text: str, tool_params: dict | None = None) -> dict:
    """返回 {think, tool_call, answer, ok, error}。

    tool_params: 可选的 {工具名: JSON Schema parameters}。给了就能修
    `"arguments"` 被写成裸列表/裸标量的情况(模型漏掉了参数名那层包装)。
    """
    text = text or ""
    # Normalize Kimi's native markers BEFORE stripping special tokens: the
    # markers are themselves <|...|> tokens, so stripping first would erase them.
    native_kimi_call = bool(KIMI_TOOLCALL_RE.search(text))
    if native_kimi_call:
        def _normalize_kimi_call(match: re.Match) -> str:
            return (
                '<tool_call>{"name": '
                + json.dumps(match.group(1))
                + ', "arguments": '
                + match.group(2)
                + "}</tool_call>"
            )

        text = KIMI_TOOLCALL_RE.sub(_normalize_kimi_call, text)
    text = SPECIAL_TOKEN_RE.sub("", text)
    # qwen 的两种漂移写法同样先归一化成标准形态,后面所有逻辑都不用动
    if QWEN_XML_TOOLCALL_RE.search(text):
        text = QWEN_XML_TOOLCALL_RE.sub(_qwen_xml_to_json, text)
    if QWEN_NATIVE_LEAK_RE.search(text):
        text = QWEN_NATIVE_LEAK_RE.sub(_qwen_native_to_json, text)
    think = extract_think(text)

    # 只在思考之后的正文里找 tool_call / answer:模型常在思考里草拟候选 JSON,
    # 从全文搜会把草稿当成真正的调用。找不到闭合标签时退回全文。
    close = CLOSE_THINK_RE.search(text)
    body = text[close.end():] if close else text
    if not (TOOLCALL_RE.search(body) or ANSWER_RE.search(body) or BARE_JSON_RE.search(body)):
        body = text

    # 同时吐 <answer> 和 <tool_call> 时以 tool_call 为准:实测这种情况下 answer
    # 全是「我目前找到了…接下来要搜…」的中间草稿,按 answer 收口会拿一段半成品
    # 当最终预测、整道题作废。让它继续跑,下一轮还能正常收口。
    ans_m = ANSWER_RE.search(body) or OPEN_ANSWER_RE.search(body)
    has_tc = bool(TOOLCALL_RE.search(body) or BARE_JSON_RE.search(body))
    draft = ans_m.group(1).strip() if ans_m else ""
    if ans_m and not has_tc:
        if draft:
            return {"think": think, "tool_call": None, "answer": draft, "ok": True, "error": ""}

    tc_m = (TOOLCALL_RE.search(body) or BARE_JSON_RE.search(body)
            or LOOSE_TOOLCALL_RE.search(body))
    if tc_m:
        raw = tc_m.group(0) if tc_m.re is BARE_JSON_RE else tc_m.group(1)
        repaired = native_kimi_call
        try:
            tc = json.loads(raw)
        except Exception as json_exc:
            tc = parse_named_arguments_toolcall(raw, tool_params=tool_params)
            if tc is not None:
                repaired = True
            # 只对两种**纯结构性**坏法真去修,其余一律只诊断:
            #   1) repair_positional_y —— 坐标漏 "y" 键(operation ckpt 44.9% 的 click 栽在这)
            #   2) balance_braces      —— 只缺闭合的 } ] (search ckpt 实测漏 2 个右花括号,
            #                             占其格式失败的 63%;JSON 只是没写完,补什么位置
            #                             是唯一确定的,不存在猜意图)
            # 两者都自带校验:补完必须能 json.loads 成功才采用,否则照常报格式错误。
            # 顺序上先试 y 键(更窄),再试补括号。
            if tc is None:
                for fix in (repair_positional_y, balance_braces):
                    fixed = fix(raw)
                    if fixed == raw:
                        continue
                    try:
                        tc = json.loads(fixed)
                        repaired = True
                        break
                    except Exception:
                        tc = None
            if tc is None:
                # 其余坏法一律不改写:改写会掩盖模型的错误,也有猜错意图的风险。
                # 修复函数只当诊断器用——定位到具体哪种坏法,把错误信息说准。
                # (真正的语法错误必须留住:json5 没装时 `import json5` 抛的
                #  ImportError 曾经把它顶掉,模型收到 "No module named 'json5'"
                #  这种看不懂的反馈,5 次重试全废、整道题作废。)
                return {"think": think, "tool_call": None, "answer": None, "ok": False,
                        "error": (f"tool_call is not valid JSON ({str(json_exc)[:90]}). "
                                  f"{diagnose_json(raw)}"
                                  "It must be exactly "
                                  '{"name": "<tool>", "arguments": {...}} — every key '
                                  "must be a quoted string. Re-emit the whole tool_call.")}
        if not isinstance(tc, dict) or not tc.get("name"):
            return {"think": think, "tool_call": None, "answer": None, "ok": False,
                    "error": "tool_call must be an object with a \"name\" field."}
        tc.setdefault("arguments", {})
        if not isinstance(tc["arguments"], dict):
            # 模型漏掉参数名那层包装:{"name":"search","arguments":["查询词"]}
            req = ((tool_params or {}).get(tc.get("name")) or {}).get("required") or []
            want = (f'For `{tc.get("name")}` it must be '
                    '{"' + req[0] + '": <value>}. ') if len(req) == 1 else ""
            return {"think": think, "tool_call": None, "answer": None, "ok": False,
                    "error": ('tool_call.arguments must be an object mapping parameter '
                              "names to values, not a bare list or scalar. " + want +
                              "Re-emit the whole tool_call with named arguments.")}
        # answer_draft:同轮里被压下的 <answer> 文本。正常轮忽略它(继续调工具),
        # 但最终强制作答轮没有别的选择时可以拿它兜底,避免白白 format_error。
        return {"think": think, "tool_call": tc, "answer": None, "ok": True, "error": "",
                "answer_draft": draft, "repaired": repaired}

    return {"think": think, "tool_call": None, "answer": None, "ok": False,
            "error": ("No <tool_call> and no <answer> found. Emit exactly one "
                      "<tool_call>{\"name\":...,\"arguments\":{...}}</tool_call> "
                      "or a final <answer>...</answer>.")}


# 折叠时每条工具观测保留多少字符。训练数据(training/build_aquila_v2.py
# 的 --prior-fold-chars)和推理必须用同一个值,否则模型在训练里见到的折叠形态
# 和线上不一样。
PRIOR_FOLD_CHARS = 600
# 被压缩进 user 消息的历史里,每步思考保留多少字符。
# -1 = 全留(默认):历史块里每一步都带 think。
# 0 = 整段丢掉;N = 截到 N 字符。
# 注意:被折叠的那些步无论这个值是多少,think 都会被丢掉 —— 折叠的定义就是
# "丢思考 + 观测截到 prior_fold_chars"。
# 注意:这只管"被压缩的旧历史";当前段里 assistant 消息带的 <think> 是真实
# 对话轮的内容,永远原样保留。
PRIOR_THINK_CHARS = -1

# 折叠水位线。上线触发折叠,下线是折叠的目标 —— 一次折到 LOW 以下,而不是
# 折到"刚好装下"。训练数据用同一对数值切段(build_aquila_v2 / build_molmoweb_v2
# 的 --fold-high / --fold-low),改这里必须同步改那两个。
# chat_template 每条消息的固定开销(<|im_start|>role\n … <|im_end|>\n)。
# 实测斜率 4.0/消息、截距 3;取 5 和 16 是刻意留在偏大一侧,见 _tokens_of。
TEMPLATE_PER_MSG = 5
TEMPLATE_TAIL = 16

# 历史块里连续同一调用达到几次就折成 `×N`。3 = 出现 3 份相同就折。
LOOP_COLLAPSE_MIN = int(os.environ.get("HYBRID_LOOP_COLLAPSE_MIN", "3"))

class ContextLimitExceeded(Exception):
    """上下文越过触发线且不允许折叠(HYBRID_CONTEXT_RESTART=1)。由 hybrid_agent 捕获后整题重开。"""

    def __init__(self, tokens: int, limit: int):
        super().__init__(f"context {tokens} > limit {limit}")
        self.tokens, self.limit = tokens, limit


FOLD_HIGH = 131072
FOLD_LOW = 65536

_TOKENIZER = None


def _get_tokenizer(model_path: str = ""):
    """惰性加载 tokenizer 用于精确计数;失败则回退到字符估算。"""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    path = model_path or os.environ.get("HYBRID_TOKENIZER_PATH", "")
    if path:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            return _TOKENIZER
        except Exception:
            pass
    _TOKENIZER = False          # 标记为不可用
    return _TOKENIZER


def count_tokens(text: str, model_path: str = "") -> int:
    tok = _get_tokenizer(model_path)
    if tok:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return int(len(text) / 3.0)      # 中英混合保守估算


class ReactHistory:
    """append-only 消息历史。切段保留图片;超长折叠时才移除对应步骤的图片。"""

    def __init__(self, max_o_screenshots: int = 0, max_input_tokens: int = 250000,
                 fold_keep_chars: int = 0, model_path: str = ""):
        # 0 = 取环境变量(默认 -1 表示全留)。训练数据放的是全量截图,推理必须
        # 一致 —— 只留最近 2 张会让模型在训练里学到的"回看历史截图"在线上失效。
        # 实测全量截图的开销可接受:中位 11 步 ≈ 9.7k token,P90 31 步 ≈ 27k。
        self.max_o_screenshots = (max_o_screenshots or
                                  int(os.environ.get("HYBRID_MAX_O_SCREENSHOTS", "-1")))
        self.max_input_tokens = max_input_tokens
        # 0 = 用与训练一致的默认值(PRIOR_FOLD_CHARS)。两边必须相同。
        self.fold_keep_chars = fold_keep_chars or PRIOR_FOLD_CHARS
        self.model_path = model_path
        self.folded: set = set()             # 已折叠的 turn 下标
        self.turns: list[dict] = []          # 每项: {role, text, image_url?, kind}
        self._tok_cache: dict[str, int] = {}  # 分段分词缓存,见 _tokens_of
        self._img_tok: int | None = None
        # 被压缩的旧历史里每步思考保留多少字符,见 PRIOR_THINK_CHARS。
        self.think_chars = int(os.environ.get("HYBRID_THINK_CHARS",
                                              str(PRIOR_THINK_CHARS)))
        # 超长切出来的段起点,跨轮保持;None = 按 side 变化(switch)自动判定。
        self.seg_start_override: int | None = None
        # O 段入场那张图,(段起点, data_url);钉住不动,保证段内前缀 append-only。
        self._seg_shot: tuple[int, str] | None = None
        # 段起点之前的历史,每条观测折到多少字符。
        # 切点处一次性把所有旧观测折掉,前缀最小 -> 本段能放下最多轮次。
        # 训练数据用同一个默认值生成,改这里必须同步改 build_aquila_v2.py。
        self.prior_fold_chars = int(os.environ.get("HYBRID_PRIOR_FOLD_CHARS",
                                                   str(PRIOR_FOLD_CHARS)))
        # 水位线折叠:碰到 HIGH 才折,一折就折到 LOW 以下。
        # 之所以不是"折到刚好装下":那样每步都会再触发一次折叠,而折叠改的是
        # 前缀(历史块在第一条 user 消息里),前缀一动 KV cache 就整段失效。
        # 留出 HIGH-LOW 的余量,折完之后能一路 append 到下次触线,中间全是命中。
        self.fold_high = int(os.environ.get("HYBRID_FOLD_HIGH", str(FOLD_HIGH)))
        self.fold_low = int(os.environ.get("HYBRID_FOLD_LOW", str(FOLD_LOW)))
        # 给生成留的预算。服务端的硬约束是 **输入 + max_tokens ≤ max_model_len**,
        # 而 fold_high 是按整条序列定的(训练侧 n_tokens 统计的就是含末条 assistant
        # 的全序列)。不减掉这块,折叠触发点就等于整个窗口 —— 输入涨到 98K 出头时
        # 服务端已经 400 了,而 131,072 那条线永远够不到,折叠形同虚设。
        #   实测:DeepSearch-2510-test-111 跑到第 88 步吃 400 ——
        #   "requested 32768 output tokens and your prompt contains at least
        #    98305 input tokens, for a total of at least 131073"
        #   98,304 = 131,072 - 32,768,就差 1 个 token。
        self.reserve_output = int(os.environ.get("HYBRID_RESERVE_OUTPUT_TOKENS", "0"))
        # discard-all(移植自 Iris,arXiv 2609.04304):触线不折也不重跑整题,
        # **原地把已积累的步骤全部丢掉、只留原题,在同一条 episode 里继续跑**。
        # 丢弃点之前的步骤不从 self.turns 删除(轨迹要完整落盘供事后分析),
        # 只是渲染时跳过 —— 等价于 Iris 的 discard_all_to_first_user()。
        self.discard_start: int = 0
        self.discard_events: list[dict] = []
        # 两次丢弃之间至少要跑这么多步。防止丢完仍然超线(system+任务本身就很大)
        # 时每步都丢一次,把这道题卡成活锁。
        self.discard_min_gap = int(os.environ.get("HYBRID_DISCARD_MIN_GAP", "3"))
        self.discard_all_enabled = os.environ.get("HYBRID_DISCARD_ALL", "") == "1"
        # FLAT 消融(HYBRID_V2_FLAT=1,见 hybrid_prompts.v2_flat):没有模式,也就没有"切端"。
        #   * 不再按 side 变化切段 —— 整条轨迹一段,只有折叠 / loop guard 的切点;
        #   * 渲染里去掉所有分模式措辞("You are now on the … path"、历史块的 S/O 图例和
        #     [step N · S] 标签、"as you took over this path");
        #   * 段入场图 / "浏览器还是空白" 提示不再只给 O 侧,段起点一律给(有页就给图)。
        # side 仍记在每一步上:截图只挂浏览器动作的观测(与默认 v2 同一条规则)。
        self.flat = os.environ.get("HYBRID_V2_FLAT", "").strip() == "1"

    # ---------------------------------------------------------------- 写入

    def add_user_task(self, text: str, input_images: list[str] | None = None):
        self.turns.append({"role": "user", "text": text, "kind": "task",
                           "images": list(input_images or [])})

    def add_assistant(self, text: str):
        self.turns.append({"role": "assistant", "text": text, "kind": "act"})

    def add_observation(self, text: str, image_url: str = "", label: str = "observation"):
        self.turns.append({"role": "user", "text": text, "kind": label,
                           "images": [image_url] if image_url else []})

    # ---------------------------------------------------------------- 渲染

    def _fold_text(self, text: str) -> str:
        """折叠一条工具输出:保留开头一段 + 标注被折叠的规模。"""
        t = text or ""
        if len(t) <= self.fold_keep_chars:
            return t
        return (t[:self.fold_keep_chars].rstrip() +
                f"\n…[tool output folded — {len(t) - self.fold_keep_chars} more characters "
                f"omitted to fit the context window; re-run the tool if you need the rest]")

    def render(self, system_text: str) -> list[dict]:
        """渲染成 OpenAI messages。
        1) 只有最近 max_o_screenshots 张截图保留图片,更早的降级成一行文字;
        2) 若总输入超过 max_input_tokens,**从最早的工具输出开始依次折叠**,
           每折一条重新计数,一旦不超限立即停止。"""
        img_turns = [i for i, t in enumerate(self.turns)
                     if t.get("images") and t.get("kind") in ("observation", "screenshot")]
        keep = (set(range(len(self.turns))) if self.max_o_screenshots < 0
                else set(img_turns[-self.max_o_screenshots:]) if self.max_o_screenshots > 0
                else set())

        msgs = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
        for i, t in enumerate(self.turns):
            content = []
            if t.get("text"):
                content.append({"type": "text", "text": t["text"]})
            imgs = t.get("images") or []
            if imgs:
                if t.get("kind") == "task" or i in keep:
                    for u in imgs:
                        if u:
                            content.append({"type": "image_url", "image_url": {"url": u}})
                else:
                    content.append({"type": "text",
                                    "text": "[screenshot from this step omitted — only the two most "
                                            "recent screenshots are kept in context]"})
            if not content:
                continue
            msgs.append({"role": t["role"], "content": content})

        # ---- 超长则从头折叠工具输出 ----
        def total_tokens(ms):
            buf = []
            for m in ms:
                for c in (m.get("content") or []):
                    if c.get("type") == "text":
                        buf.append(c.get("text") or "")
                    else:
                        buf.append(" " * 4000)      # 图片按约 1.3K token 粗算
            return count_tokens("\n".join(buf), self.model_path)

        if total_tokens(msgs) > self.max_input_tokens:
            obs_idx = [i for i, t in enumerate(self.turns)
                       if t.get("kind") in ("observation", "screenshot") and i not in self.folded]
            for i in obs_idx:                        # 从最早的开始
                if self.turns[i].get("protect_observation"):
                    continue
                self.folded.add(i)
                self.turns[i]["text"] = self._fold_text(self.turns[i].get("text") or "")
                msgs = self._render_raw(system_text, keep)
                if total_tokens(msgs) <= self.max_input_tokens:
                    break
        return msgs

    def _render_raw(self, system_text: str, keep: set) -> list[dict]:
        msgs = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
        for i, t in enumerate(self.turns):
            content = []
            if t.get("text"):
                content.append({"type": "text", "text": t["text"]})
            imgs = t.get("images") or []
            if imgs:
                if t.get("kind") == "task" or i in keep:
                    for u in imgs:
                        if u:
                            content.append({"type": "image_url", "image_url": {"url": u}})
                else:
                    content.append({"type": "text",
                                    "text": "[screenshot from this step omitted — only the two most "
                                            "recent screenshots are kept in context]"})
            if content:
                msgs.append({"role": t["role"], "content": content})
        return msgs


    # ------------------------------------------------- 轻度重建(每轮重新组织)

    def add_step(self, side: str, action_text: str, observation: str,
                 think: str = "", image_url: str = "", *,
                 switch_executed: bool = True, action_kind: str = "tool"):
        """记录一步:哪一侧、动作、工具观测、可选截图。"""
        self.turns.append({"role": "step", "side": (side or "S"), "action": action_text,
                           "text": observation or "", "original_text": observation or "",
                           "think": think or "",
                           "kind": "observation", "images": [image_url] if image_url else [],
                           "switch_executed": bool(switch_executed),
                           "action_kind": action_kind})

    def protect_steps(self, step_numbers) -> None:
        """Keep verifier-cited source tool results verbatim during folding."""
        protected = {
            int(step) for step in (step_numbers or [])
            if isinstance(step, int) or str(step).isdigit()
        }
        if not protected:
            return
        steps = [turn for turn in self.turns if turn.get("role") == "step"]
        for index, turn in enumerate(steps, 1):
            if index in protected:
                turn["protect_observation"] = True
                turn["text"] = turn.get("original_text", turn.get("text") or "")
                self.folded.discard(id(turn))
                self.folded.discard(index - 1)

    # -------------------------------------------------- 段内多轮(方案 A)

    @staticmethod
    def _tool_name_of(action: str) -> str:
        """从一步的 action 串里取工具名 —— **解析,不做字面量搜索**。

        原来判段边界用的是 `'"name": "switch"' in action`。子串匹配有两个问题:
          * 参数里出现同样的字面量就会误判。JSON 转义挡掉了大部分场景
            (code_sandbox 的代码、search 的查询、type 的文本里写这串,序列化后是
            `\\"name\\": \\"switch\\"`,不匹配),但**嵌套对象**挡不住 ——
            `{"name":"code_sandbox","arguments":{"files":{"name":"switch"}}}`
            里那段是没转义的,实测会误判成 switch。
          * 今天把 action 从 `name(arg)` 简写改回完整 JSON 之后,这个暴露面还变大了
            (简写形态压根不可能含这串)。

        兼容老记录:action 不是合法 JSON 时退回子串判断,别让历史轨迹重放时错段。
        """
        s = action or ""
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                n = d.get("name")
                return n if isinstance(n, str) else ""
        except Exception:
            pass
        if '"name": "switch"' in s or '"name":"switch"' in s:
            return "switch"
        return ""

    def segment_start(self) -> int:
        """当前段的起始步下标。段 = 两次 switch 之间的连续同侧区间。

        切端必须换系统提示词(S/O 两侧提示词差异 71%,19 个工具只共享 2 个),
        而 chat_template 明确禁止 system 出现在非首位:
            'System message must be at the beginning.'
        所以只能在切端处重建;段内则是标准多轮,前缀只增不改,KV cache 全命中。
        """
        if self.flat:
            # FLAT:一版提示词、不切端,side 交替不是段边界(否则每次换工具类型都重建上下文)。
            return 0
        steps = self._visible_steps()
        seg0 = 0
        for i, t in enumerate(steps):
            # switch 那一步记的是**来源侧**(它是在旧侧上做的决定),所以只看 side
            # 变化的话,边界要等新一侧第一步落下才出现 —— 中间那一轮模型已经拿到
            # 新侧的 system 提示词,当前段里却还摆着旧侧的调用(O 侧提示词甚至不
            # 暴露 search)。切换的语义是"从下一步开始换人",所以边界就在它之后。
            if (
                self._tool_name_of(t.get("action") or "") == "switch"
                and t.get("switch_executed", True)
            ):
                seg0 = i + 1
            elif i and t.get("side") != steps[i - 1].get("side"):
                seg0 = i
        return seg0

    def _visible_steps(self) -> list[dict]:
        """渲染时可见的步骤 = discard-all 丢弃点之后的部分。

        丢弃点之前的步骤仍留在 self.turns 里(轨迹要完整落盘),只是不进上下文。
        discard_start == 0(默认)时就是全部步骤,行为与移植前逐字相同。
        所有以"步下标"为参数的接口(seg_start / seg_start_override)在丢弃之后
        都改用**相对可见列表**的下标,丢弃时会把 seg_start_override 一起清掉。
        """
        steps = [t for t in self.turns if t.get("role") == "step"]
        return steps[self.discard_start:] if self.discard_start else steps

    def has_current_segment_steps(self) -> bool:
        """本段里已经有动作了吗。没有就说明是 O 打头或刚 switch 过来 ——
        此刻上下文里一张图都没有,渲染时要把当前页面补进去。"""
        return len(self._visible_steps()) > self.effective_seg_start()

    def effective_seg_start(self) -> int:
        """真正生效的段起点 = max(折叠推出来的切点, 最近一次 switch 的边界)。

        两个来源都会产生切点,必须取靠后的那个:
          * 折叠把切点往后推,写进 seg_start_override,跨轮粘住;
          * switch 换了 side,segment_start() 会给出新的边界。
        只看 override 的话,折叠之后再 switch,新段里会混进**上一侧**的步骤 ——
        system 提示词已经换成 O 侧,当前段里却还摆着 S 侧的 search 调用,而 O 侧
        提示词根本不暴露 search。反过来只看 side 变化,则会丢掉折叠的效果、
        上下文重新涨回去。
        """
        by_side = self.segment_start()
        ov = self.seg_start_override
        if ov is None:
            return by_side
        if by_side > ov:
            # switch 发生在折叠切点之后:切点前移到 switch 边界,并把 override 同步,
            # 否则下一轮又会退回旧值。
            self.seg_start_override = by_side
            return by_side
        return ov

    def _assistant_text(self, t: dict) -> str:
        """把一步还原成模型该输出的样子。训练与推理共用同一个函数,保证格式逐字一致。

        JSON 模式(HYBRID_JSON_OUTPUT=1)下渲染成 {"cot":…, "tool_call"/"answer":…}。
        **必须和系统提示词同时切**:实测只改提示词、历史仍是标签时,模型 0/8 全部
        继续吐标签 —— 它模仿的是上下文里的前几十轮,不是系统提示词的指令。
        导出侧不走这条路(build_hybrid_teacher_v2 自己拼标签),所以训练格式不变。
        """
        think = t.get("think") or ""
        act = t.get("action") or ""
        if json_output_mode():
            body = {"cot": think}
            if t.get("action_kind") == "answer":
                body["answer"] = act
            else:
                try:
                    body["tool_call"] = json.loads(act) if act else {}
                except Exception:
                    body["tool_call"] = {"raw": act}
            return json.dumps(body, ensure_ascii=False)
        head = f"<think>\n{think}\n</think>\n" if think else ""
        if t.get("action_kind") == "answer":
            return f"{head}<answer>\n{act}\n</answer>"
        return f"{head}<tool_call>\n{act}\n</tool_call>"

    @staticmethod
    def _action_summary(t: dict) -> str:
        action = t.get("action") or ""
        if t.get("action_kind") == "answer":
            return f"<answer>{action}</answer>"
        return action

    def render_multiturn(self, system_text: str, active_side: str, task_text: str,
                         task_images: list[str] | None = None,
                         seg_start: int | None = None,
                         drop_last_observation: bool = False,
                         current_shot: str = "") -> list[dict]:
        """段内多轮渲染。

        [system(当前侧)]
        [user  : 任务 + (若非首段)之前各段重建出来的历史 + 本段起点标记]
        [assistant/user 交替]  ← 本段内每一步的动作与观测,append-only

        seg_start: 显式指定段起点。不给则按 side 变化(switch)自动判定。
                   超长轨迹要切成多段时用它指定切点。
        drop_last_observation: 丢掉本段最后一条观测,让消息以 assistant 收尾
                   (训练样本必须以待预测的 assistant 结束)。
        """
        steps = self._visible_steps()
        seg0 = self.segment_start() if seg_start is None else max(0, min(seg_start, len(steps)))
        prior, cur = steps[:seg0], steps[seg0:]

        # 截图额度按整条轨迹算,不能在切段后只看 cur。切段只是把旧步骤搬进
        # Earlier history,不是压缩动作;未折叠的 O 步骤仍应带着原截图。真正折叠
        # 时才由 self.folded 去掉图片。显式配置 max_o_screenshots 时仍尊重这层
        # 独立的总量限制。
        img_pos = [i for i, t in enumerate(steps)
                   if t.get("images") and t.get("side") == "O"]
        keep_images = (set(img_pos) if self.max_o_screenshots < 0
                       else set(img_pos[-self.max_o_screenshots:])
                       if self.max_o_screenshots > 0 else set())

        head = [{"type": "text", "text": f"## Task\n{task_text}"}]
        for u in (task_images or []):
            if u:
                head.append({"type": "image_url", "image_url": {"url": u}})
        if prior:
            head.append({"type": "text",
                         "text": (f"\n## Earlier history — {len(prior)} steps before this "
                                  f"context rebuild"
                                  # FLAT 下没有 S/O 之分,图例不给(步标签也不带侧,见 _prior_step_text)
                                  + ("" if self.flat else
                                     " (S = search-based, O = operation-based)"))})
            # 重复的同一个调用归并成一条。**按指纹分组,不要求连续** ——
            # 实测打转的轨迹多是 A→B→C→A→B 交替(test-20 就是),要求连续的话
            # run-length 一次都抓不到。
            #
            # 归并规则:
            #   * 锚在**首次出现的位置**,保持时间序 —— 训练数据的历史块就是纯
            #     时间序的 [step N · S] 行,这样偏离最小,也保住因果链。
            #   * **保留一次真实的工具结果**(那次搜索是真跑过的,结果是真信息),
            #     取第一条有实质内容的:跳过 "No results found" 和 loop-guard 那种
            #     合成消息 —— 同样的查询多次搜结果会抖动,第一次可能恰好是空的。
            #   * 条目的步号 = 被采用的那条观测所在的步,不然会出现
            #     "[step 15] 底下挂着第 31 步结果"这种对不上的情况。
            groups: dict = {}
            for idx, t in enumerate(prior):
                groups.setdefault((t.get("action") or "", t.get("side")), []).append(idx)

            def _substantive(t) -> bool:
                o = t.get("text") or ""
                return bool(o.strip()) and "[loop guard]" not in o and "No results found" not in o

            anchor_of = {}
            for key, idxs in groups.items():
                pick = next((i for i in idxs if _substantive(prior[i])), idxs[0])
                anchor_of[pick] = idxs

            for idx, t in enumerate(prior):
                if idx in anchor_of:
                    idxs = anchor_of[idx]
                    note = ""
                    if len(idxs) > 1:
                        others = [i + 1 for i in idxs if i != idx]
                        if len(others) <= 8:
                            where = ", ".join(str(x) for x in others)
                        else:
                            where = ", ".join(str(x) for x in others[:8]) + f", … ({len(others)} times total)"
                        note = ("\n  (you made this exact call {n} more time{s} — at step{s2} {w} — "
                                "and got nothing new; that avenue is exhausted)").format(
                                    n=len(others), s="" if len(others) == 1 else "s",
                                    s2="" if len(others) == 1 else "s", w=where)
                    head.append({"type": "text",
                                 "text": self._prior_step_text(idx, t, dup_note=note)})

                # 重复调用的文本可以归并,截图不跟着归并:每张图仍在它原本的时间
                # 位置出现。只有该步真的进入 folded 后才移除,所以单纯推进 seg0
                # 不会再悄悄改变模型可见的视觉历史。
                if (idx in keep_images and id(t) not in self.folded
                        and t.get("side") == "O"):
                    for u in t.get("images") or []:
                        if u:
                            head.append({"type": "text",
                                         "text": f"  screenshot after earlier step {idx+1}:"})
                            head.append({"type": "image_url", "image_url": {"url": u}})
        if not self.flat:              # FLAT 没有"当前在哪条路径"这回事,这行整个不给
            head.append({"type": "text",
                         "text": f"\n## You are now on the "
                                 f"{'SEARCH-BASED (S)' if active_side == 'S' else 'OPERATION-BASED (O)'} path"})
        # 首图:O 侧全靠看图决定点哪里,而截图只会作为"上一步的观测"进上下文 ——
        # 本段还没有任何动作时(O 打头,或刚 switch 过来),模型手上一张图都没有,
        # 第一次 click 是闭眼点的(实测切换后上下文截图数为 0)。
        #
        # 放在 user 消息的**最末尾**:紧贴模型要生成的位置,和后续每轮"观测在前、
        # 决策在后"的节奏一致;历史块在前面,不会把图挤到中间。
        # 来源优先当前实拍(current_shot),没有就退回交接那步带的图。
        #
        # **段内钉死**:这块一旦渲染出来就在整段里保持逐字节不变,不能"第一步有、
        # 第二步没了" —— 它位于 user 消息(=前缀),一消失整段 KV cache 全废。
        # 钉住也顺带更省:待在稳定前缀里的图第一步之后就是命中缓存,不再重算。
        # 段起点一变(switch / 折叠推进切点)前缀本来就重排,此时作废重钉。
        # FLAT:浏览器工具每一轮都可用,段起点一律交代浏览器现状(有页给图,空白给一行字),
        # 与默认 v2 进入 O 侧时同一套钉图逻辑,只是措辞去掉 "path"。
        if active_side == "O" or self.flat:
            if self._seg_shot is None or self._seg_shot[0] != seg0:
                # 只在段还空着时钉:此刻模型手上一张图都没有。
                # 钉空串 = "这段确认没有入场图",同样跨轮保持 —— 否则那行提示
                # 会在第二步凭空消失,又是一次前缀重排。
                self._seg_shot = None
                if not cur:
                    shot = current_shot or next(
                        (t["images"][0] for t in reversed(prior)
                         if t.get("images") and t["images"][0]), "")
                    self._seg_shot = (seg0, shot)
            if self._seg_shot is not None:
                if self._seg_shot[1]:
                    head.append({"type": "text", "text":
                                 ("\n## The browser page at this point (before the actions that follow)"
                                  if self.flat else
                                  "\n## The browser page as you took over this path")})
                    head.append({"type": "image_url",
                                 "image_url": {"url": self._seg_shot[1]}})
                elif self.flat:
                    # FLAT 的空白提示:不能说"这条路径的第一步必须 goto" —— 模型也可以先搜索。
                    head.append({"type": "text", "text":
                                 "\n## No page is open in the browser yet — it is still blank. "
                                 "Before any browser action, `goto` a URL (or `new_tab`); "
                                 "there is nothing on screen to click."})
                else:
                    # 没图的唯一成因:浏览器还停在 about:blank(纯搜索类任务没有起始 URL)。
                    # 与其塞一张纯白截图冒充"当前页面",不如直说 —— 白图既骗模型
                    # 去点空白,又白烧 ~281 个视觉 token。
                    head.append({"type": "text", "text":
                                 "\n## No page is open yet — the browser is still blank. "
                                 "Your first action on this path must be `goto` with a URL "
                                 "(or `new_tab`); there is nothing on screen to click."})

        msgs = [{"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": head}]

        # 截图只在 O 侧的观测里出现:S 侧的工具(search/visit)结果本来就是纯文本,
        # 训练数据的 S 侧样本也一张图都没有。所以这里按 side 过滤,哪怕上游误挂了图
        # 也不会渲染进去 —— 免得 S 侧凭空多出视觉 token 且和训练分布对不上。
        for i, t in enumerate(cur):
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": self._assistant_text(t)}]})
            _o = t.get('text') or ''
            if id(t) in self.folded and len(_o) > self.fold_keep_chars:
                _o = (_o[:self.fold_keep_chars].rstrip() +
                      f"\n…[tool output folded — {len(_o) - self.fold_keep_chars} more characters "
                      f"omitted to fit the context window; re-run the tool if you need the rest]")
            obs = [{"type": "text", "text": f"<tool_response>\n{_o}\n</tool_response>"}]
            if t.get("images") and t.get("side") == "O":
                step_i = seg0 + i
                if id(t) in self.folded:
                    pass
                elif step_i in keep_images:
                    for u in t["images"]:
                        if u:
                            obs.append({"type": "image_url", "image_url": {"url": u}})
                else:
                    obs.append({"type": "text",
                                "text": f"[screenshot omitted — only the {self.max_o_screenshots} "
                                        f"most recent are kept]"})
            # 训练样本必须以待预测的 assistant 收尾;最后一条观测丢掉不损失信息,
            # 它会出现在下一段重建出来的历史里。
            if drop_last_observation and i == len(cur) - 1:
                continue
            msgs.append({"role": "user", "content": obs})
        return msgs

    def _prior_step_text(self, i: int, t: dict, dup_note: str = "") -> str:
        """历史块里的一步的文本部分。

        没被折叠的步:完整思考 + 完整观测。
        被折叠的步(id 在 self.folded 里):丢掉思考和截图、观测截到 prior_fold_chars。
        折叠是**非破坏性**的 —— 只标记不改写 t["text"],所以任意一步当时看到的
        上下文都能事后精确重建。图片由 render_multiturn 紧跟在这段文本后附加，
        不受当前 active side 限制。
        """
        side = "S" if t.get("side", "S") == "S" else "O"
        tag = "" if self.flat else f" · {side}"        # FLAT:步标签不带侧
        out = f"\n[step {i+1}{tag}] {self._action_summary(t)}" + (dup_note or "")
        folded = id(t) in self.folded
        think = t.get("think") or ""
        if think and not folded and self.think_chars != 0:
            if self.think_chars > 0:
                think = think[:self.think_chars]
            out += f"\n  (your reasoning that turn: {think})"
        obs_text = t.get("text") or ""
        if folded and len(obs_text) > self.prior_fold_chars:
            obs_text = (obs_text[:self.prior_fold_chars].rstrip() +
                        f"\n…[folded — {len(obs_text) - self.prior_fold_chars} more characters omitted]")
        out += f"\n  Observation:\n<tool_response>\n{obs_text}\n</tool_response>"
        return out

    def render_multiturn_capped(self, system_text: str, active_side: str, task_text: str,
                                task_images: list[str] | None = None,
                                current_shot: str = "", allow_fold: bool = True) -> list[dict]:
        """render_multiturn + 水位线折叠。

        **只在越过 fold_high 时才动手,一动就折到 fold_low 以下。**
        逐条折到"刚好装下"的老做法每步都会再触发一次,而折叠改的是第一条 user
        消息里的历史块 —— 前缀一变 KV cache 就整段失效。留出 high-low 的余量,
        折完能一路 append 到下次触线。

        压缩顺序:**先榨干历史块,再动当前段**。
          1. 折历史块里的步(丢思考和截图 + 观测截到 prior_fold_chars),最早的优先;
          2. 整个历史块都折完还超,才把切点往后推 —— 当前段最早的步进历史块,
             进去时直接标记为已折叠。

        顺序不能反。当前段是模型正在working的上下文(刚拿到的观测、刚看的截图),
        历史块是已经压过一轮的旧账;先推切点等于**先拆working区、留着老账不动**。
        实测反着来的后果:13 步的轨迹触线时段起点从 7 直接跳到 12,当前段只剩 1 步,
        而历史块里的老步骤反倒一条没折。

        切点(seg_start_override)跨轮保持。折叠只在 self.folded 打标记 +
        记 folded_at_step,不改写 t["text"],任意一步当时的上下文都能事后重建。
        """
        def render(seg):
            return self.render_multiturn(system_text, active_side, task_text,
                                         task_images, seg_start=seg,
                                         current_shot=current_shot)

        # 触发点 = min(水位线, 硬顶) **再减掉生成预算**。
        # reserve_output 见 __init__ 里的说明:服务端约束是 输入+max_tokens ≤ 窗口,
        # 不减这块的话折叠线够不到,轨迹长了直接吃 400。
        high = min(self.fold_high, self.max_input_tokens) - self.reserve_output
        low = min(self.fold_low, high)

        seg = self.effective_seg_start()
        msgs = render(seg)
        if self._tokens_of(msgs) <= high:
            return msgs                      # 没触线,原样 append
        if self.discard_all_enabled:
            # discard-all(Iris 口径):触线既不折也不重跑整题,**原地把已积累的步骤
            # 全部丢掉、只留原题,在同一条 episode 里接着跑**。相比"抛
            # ContextLimitExceeded 让 runner 整题重跑",省掉了已经烧掉的那几百步。
            #
            # 两道护栏:
            #   * discard_min_gap —— 丢完仍然超线(system+任务本身就大)时不要每步都丢;
            #   * 丢完还超 -> 落回折叠(若允许)或抛异常,绝不空转。
            n_vis = len(self._visible_steps())
            last_at = self.discard_events[-1]["at_step"] if self.discard_events else None
            total = len([t for t in self.turns if t.get("role") == "step"])
            # keep_last=0 是 Iris 的原口径(只留原题,连刚拿到的那条观测也丢)。
            # 留几步能省掉"丢完立刻把同一个查询再搜一遍"的浪费,留作 A/B 开关。
            keep_last = max(0, int(os.environ.get("HYBRID_DISCARD_KEEP_LAST", "0")))
            new_start = max(0, total - keep_last)
            if n_vis > keep_last and (last_at is None or total - last_at >= self.discard_min_gap):
                before = self._tokens_of(msgs)
                self.discard_events.append({
                    "at_step": total, "discarded": new_start - self.discard_start,
                    "kept_last": keep_last, "tokens_before": before, "limit": high,
                })
                self.discard_start = new_start
                self.seg_start_override = None   # 下标口径重置为相对可见列表
                self._seg_shot = None            # 段入场图失效,由调用方重新补
                msgs = render(self.effective_seg_start())
                if self._tokens_of(msgs) <= high:
                    return msgs
                # 丢完还超:只剩 system+任务本身就装不下,交给下面的折叠/异常处理。
        if not allow_fold:
            # HYBRID_CONTEXT_RESTART=1:触线不折,抛给上层 —— 整题清空重开(discard-all)。
            # 折叠会把整条轨迹七成的观测截成 600 字符、丢掉思考,模型之后几十上百步都在这份
            # 残缺上下文里跑:BC-EN 实测 37% 的题触发折叠,同等步数下比未折叠的低十几个点,
            # 而且这种形态训练时没见过。Iris(arXiv 2609.04304)同样选择不裁历史而是重开,
            # 其 35B 模型上 discard-all 相对不做上下文管理 +17.5(BrowseComp)/+12.5(BC-ZH)。
            raise ContextLimitExceeded(self._tokens_of(msgs), high)

        steps = self._visible_steps()
        # 1) 先折历史块,最早的优先
        for i in range(seg):
            t = steps[i]
            if id(t) in self.folded or t.get("protect_observation"):
                continue
            self.folded.add(id(t))
            t["folded_at_step"] = len(steps)
            msgs = render(seg)
            if self._tokens_of(msgs) <= low:
                return msgs
        # 2) 历史块榨干了还超 -> 把切点往后推,当前段最早的步进历史块并直接折掉。
        #    末步永远留在当前段:它带着刚拿到的观测/截图,是这一轮决策的依据。
        for new_seg in range(seg + 1, len(steps)):
            t = steps[new_seg - 1]
            if not t.get("protect_observation"):
                self.folded.add(id(t))
                t["folded_at_step"] = len(steps)
            self.seg_start_override = new_seg
            msgs = render(new_seg)
            if self._tokens_of(msgs) <= low:
                return msgs
        return msgs

    def render_full_for_summary(self, system_text: str, active_side: str, task_text: str,
                                task_images: list[str] | None = None) -> list[dict]:
        """渲染**整条**轨迹(含已被 discard-all 丢弃的部分),供失败后验使用。

        discard-all 之后 self.turns 里仍保留全部步骤(只是渲染时跳过),所以这里
        临时把丢弃点归零、允许折叠,渲染完再还原。

        为什么后验要看全轨迹而不是像 Iris 那样只看可见段:丢掉的那部分正是最该带给
        下一次尝试的 —— 一道题丢弃三次才撞顶,只看最后一小段等于 discard-all 把发现
        扔了一次、retry 再也捡不回来,而丢弃次数多的恰恰是 retry 最该救的长难题。
        Iris 只能看可见段是实现所限(他们把历史真删了),不是有意的设计。
        折叠在这里是可接受的:后验只生成一次,不是让 agent 在残缺上下文里continue跑。
        """
        saved = (self.discard_start, self.seg_start_override,
                 set(self.folded), self._seg_shot, self.fold_low,
                 self.discard_all_enabled)
        try:
            # **必须先关掉 discard-all**:render_multiturn_capped 里丢弃分支排在折叠
            # 前面,不关的话这里会把整条轨迹直接丢光,只剩 system+原题(实测 300 步
            # 的轨迹渲染出来只有 1195 token),后验等于什么都没看见。
            self.discard_all_enabled = False
            self.discard_start = 0
            self.seg_start_override = None
            self._seg_shot = None
            # **只压到刚好装得下**,不要用主循环那个 fold_low。
            # fold_low 低是为了折完能一路 append 几十步不再触线(保 KV cache),而后验
            # 是一次性调用,不需要留余量。用主循环口径的实测后果:23 万 token 的 300 步
            # 轨迹被压成 1,200 token —— 折叠会一路把切点推到当前段只剩最后一步,
            # 加上历史块按指纹归并,模型几乎什么都看不见,后验也就写不出有用的东西。
            high = min(self.fold_high, self.max_input_tokens) - self.reserve_output
            self.fold_low = max(0, high - 2048)
            return self.render_multiturn_capped(system_text, active_side, task_text,
                                                task_images, allow_fold=True)
        finally:
            (self.discard_start, self.seg_start_override,
             self.folded, self._seg_shot, self.fold_low,
             self.discard_all_enabled) = saved

    def _build_rebuilt(self, system_text: str, active_side: str, task_text: str,
                       task_images, think_last_n: int) -> list[dict]:
        steps = self._visible_steps()
        img_pos = [i for i, t in enumerate(steps) if t.get("images")]
        keep = (set(range(len(steps))) if self.max_o_screenshots < 0
                else set(img_pos[-self.max_o_screenshots:]) if self.max_o_screenshots > 0
                else set())

        content = [{"type": "text", "text": f"## Task\n{task_text}"}]
        for u in (task_images or []):
            if u:
                content.append({"type": "image_url", "image_url": {"url": u}})
        if steps:
            # 不要把步数写进这行 —— 它排在所有 step 块之前,每轮 N→N+1 就让后面
            # 全部内容位移,prefix cache 从这里开始整段失效(实测命中率只有 10%)。
            # FLAT(HYBRID_V2_FLAT=1)下不给 S/O 图例、步标签不带侧、末行不说"在哪条路径"。
            content.append({"type": "text", "text": "\n## History" if self.flat else
                            "\n## History (S = search-based path, O = operation-based path)"})
        for i, t in enumerate(steps):
            side = "S" if t.get("side", "S") == "S" else "O"
            head = (f"\n[step {i+1}] {self._action_summary(t)}" if self.flat else
                    f"\n[step {i+1} · {side}] {self._action_summary(t)}")
            # think_last_n: -1 = 全部保留;0 = 一条不留;N = 只留最近 N 步。
            # think_chars: 0 = 不截断。
            keep_think = (think_last_n < 0) or (i >= len(steps) - think_last_n)
            # think_chars 语义与 _prior_step_text 一致:0 = 整段丢掉,-1 = 全留。
            if t.get("think") and keep_think and think_last_n != 0 and self.think_chars != 0:
                tx = t["think"]
                if self.think_chars > 0:
                    tx = tx[:self.think_chars]
                head += f"\n  (your reasoning that turn: {tx})"
            # 标签必须对每一步恒定:写成 "Latest observation" 的话,这一步在下一轮
            # 会变回 "Observation",又是一处会让前缀失效的改写。最新一条本来就紧挨
            # 着末尾的 "## Your turn",位置已经足够表明它是最新的。
            content.append({"type": "text",
                            "text": f"{head}\n  Observation:\n<tool_response>\n{t.get('text') or ''}\n</tool_response>"})
            if t.get("images"):
                if i in keep:
                    for u in t["images"]:
                        if u:
                            content.append({"type": "text", "text": f"  screenshot after step {i+1}:"})
                            content.append({"type": "image_url", "image_url": {"url": u}})
                else:
                    content.append({"type": "text",
                                    "text": f"  [screenshot after step {i+1} omitted — only the "
                                            f"{self.max_o_screenshots} most recent are kept]"})
        content.append({"type": "text",
                        "text": "\n## Your turn" if self.flat else
                                f"\n## Your turn — you are on the "
                                f"{'SEARCH-BASED (S)' if active_side == 'S' else 'OPERATION-BASED (O)'} path"})
        return [{"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": content}]

    def _tokens_of(self, msgs) -> int:
        """分段计数 + 按文本缓存 + **补上 chat template 的每条消息开销**。

        原来每次都把整个 prompt 拼起来重 tokenize 一遍,而 render_rebuilt 每折叠
        一条观测就要再算一次 —— 长轨迹上是 O(n²) 次全量分词。分段之后每段只在
        内容变化时算一次,后续步骤直接命中缓存。

        **只数正文是不够的**:chat_template 给每条消息套 `<|im_start|>role\n …
        <|im_end|>\n`,这部分不在任何一段正文里。实测(12→322 条消息)漏掉的量
        就是 4.0 token/消息 + 截距 3,相对误差 2.4%→3.5% 且随消息数单调上升 ——
        也就是说旧注释"只会偏大不会偏小"是反的,它**偏小**,而偏小的后果是
        服务端直接 400(输入+max_tokens 超窗口),不是"晚折一点"那么轻。

        这里按 5/消息 + 16 收尾算,刻意往大了取:估大只是早折一点,估小要吃 400。
        """
        total = TEMPLATE_PER_MSG * len(msgs) + TEMPLATE_TAIL
        for m in msgs:
            for c in (m.get("content") or []):
                if c.get("type") != "text":
                    total += self._image_tokens()
                    continue
                text = c.get("text") or ""
                n = self._tok_cache.get(text)
                if n is None:
                    n = count_tokens(text, self.model_path)
                    if len(self._tok_cache) < 4096:
                        self._tok_cache[text] = n
                total += n + 1                      # 段间换行
        return total

    def _image_tokens(self) -> int:
        if self._img_tok is None:
            self._img_tok = count_tokens(" " * 4000, self.model_path)
        return self._img_tok

    def render_rebuilt(self, system_text: str, active_side: str, task_text: str,
                       task_images: list[str] | None = None, think_last_n: int = 1) -> list[dict]:
        """每轮轻度重建上下文:system(当前侧) + 一条结构化 user 消息。

        两端共享同一份混排历史,每步打 S/O 标签;工具观测全文保留,
        超过 max_input_tokens 时从最早的观测开始逐条折叠,第一次不超限即停。
        截图只保留最近 max_o_screenshots 张(附在对应步骤后)。
        """
        msgs = self._build_rebuilt(system_text, active_side, task_text, task_images, think_last_n)
        if self._tokens_of(msgs) <= self.max_input_tokens:
            return msgs
        steps = [t for t in self.turns if t.get("role") == "step"]
        for t in steps[:-1]:                      # 最新一条永远全文
            if id(t) in self.folded or t.get("protect_observation"):
                continue
            self.folded.add(id(t))
            t["text"] = self._fold_text(t.get("text") or "")
            msgs = self._build_rebuilt(system_text, active_side, task_text, task_images, think_last_n)
            if self._tokens_of(msgs) <= self.max_input_tokens:
                break
        return msgs

    # ---------------------------------------------------------------- 统计

    def stats(self) -> dict:
        chars = sum(len(t.get("text") or "") for t in self.turns)
        imgs = sum(len(t.get("images") or []) for t in self.turns)
        images_kept = (
            imgs if self.max_o_screenshots < 0
            else min(self.max_o_screenshots, imgs)
        )
        return {"turns": len(self.turns), "chars": chars, "images_total": imgs,
                "images_kept": images_kept,
                "folded_observations": len(self.folded),
                "discard_all_count": len(self.discard_events),
                "discarded_steps": sum(e["discarded"] for e in self.discard_events),
                "discard_events": list(self.discard_events),
                "protected_observations": sum(
                    bool(turn.get("protect_observation"))
                    for turn in self.turns
                    if turn.get("role") == "step"
                ),
                "max_input_tokens": self.max_input_tokens}


# --------------------------------------------------------------- 坏标签打捞
# 2026-09-01:网关上的 qwen3.8-max 会把 <tool_call> 标签写坏,而且**形态每次不同**:
#   <tool_call name="search">{...}</tool_call>            属性名 + JSON 体
#   <tool_call name="search" arguments={...}>             参数塞进属性
#   <tool_call name="s"><arg_key>q</arg_key><arg_value>[...]}}</arg_value></tool_call_name>
#   <tool_call name="s"><parameter name="q">[...]</parameter}</tool_call_>
#   <tool_call  /  <t                                      吐一半就停(不是 max_tokens,
#                                                          16384 上限而回复最长才 1082 字)
# 共同点是 **name 和参数一直都在且合理**,坏的只有标签语法。所以在严格解析失败后
# 做一次尽力打捞:抠出 name + arguments,再用调用方给的 allowed 卡一遍 ——
# 名字不在允许集里就不认,避免从垃圾里"造"出一个调用。
_SALVAGE_NAME_RES = [
    re.compile(r'<tool_call[^>]*?\bname\s*=\s*\\?["\']([A-Za-z_][A-Za-z0-9_]*)', re.I),
    re.compile(r'<tool_call_tool_name\s*>\s*([A-Za-z_][A-Za-z0-9_]*)', re.I),
    re.compile(r'"name"\s*:\s*\\?["\']([A-Za-z_][A-Za-z0-9_]*)', re.I),
]
_SALVAGE_ARGS_ATTR_RE = re.compile(r'\barguments\s*=\s*(\{)', re.I)
_SALVAGE_PARAM_RE = re.compile(
    r'<parameter\s+name\s*=\s*\\?["\']([^"\'\\]+)\\?["\'][^>]*>(.*?)(?:</parameter[^>]*>|<parameter|$)',
    re.S | re.I)
_SALVAGE_ARGKV_RE = re.compile(
    r'<arg_key\s*>\s*(.*?)\s*</arg_key\s*>\s*<arg_value\s*>\s*(.*?)\s*(?:</arg_value[^>]*>|$)',
    re.S | re.I)


def _balanced_json(text: str, start: int):
    """从 text[start] 的 '{' 开始扫平衡括号,返回 (对象, 结束位置) 或 (None, -1)。

    字符串内的括号不计数,反斜杠转义要跳过 —— 参数里带 JSON 片段时这点必须对。
    """
    if start < 0 or start >= len(text) or text[start] != "{":
        return None, -1
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1]), i + 1
                except Exception:
                    return None, -1
    return None, -1


def _loose_value(raw: str):
    """把参数值转成 Python 值。模型常在结尾粘垃圾(`[...]}}`),逐步削尾再试。"""
    raw = (raw or "").strip()
    if not raw:
        return ""
    for end in range(len(raw), max(len(raw) - 12, 0), -1):
        try:
            return json.loads(raw[:end])
        except Exception:
            continue
    return raw


def salvage_tool_call(text: str, allowed: set | None = None) -> dict | None:
    """严格解析失败后的兜底。allowed 给了就只认里面的工具名。"""
    text = text or ""
    if "<tool_call" not in text and "tool_call_tool_name" not in text:
        return None
    name = ""
    for rx in _SALVAGE_NAME_RES:
        m = rx.search(text)
        if m:
            name = m.group(1)
            break
    if not name or (allowed is not None and name not in allowed):
        return None

    args = None
    # ① arguments={...} 当属性
    m = _SALVAGE_ARGS_ATTR_RE.search(text)
    if m:
        obj, _ = _balanced_json(text, m.start(1))
        if isinstance(obj, dict):
            args = obj
    # ② 标签之后的第一个平衡 JSON 对象
    if args is None:
        tag = text.find("<tool_call")
        brace = text.find("{", tag if tag >= 0 else 0)
        obj, _ = _balanced_json(text, brace)
        if isinstance(obj, dict):
            args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj
            if "name" in args and len(args) == 1:
                args = None
    # ③ <parameter name=..>值</parameter>
    if args is None:
        pairs = _SALVAGE_PARAM_RE.findall(text)
        if pairs:
            args = {k: _loose_value(v) for k, v in pairs}
    # ④ <arg_key>k</arg_key><arg_value>v</arg_value>
    if args is None:
        pairs = _SALVAGE_ARGKV_RE.findall(text)
        if pairs:
            args = {k.strip(): _loose_value(v) for k, v in pairs}
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}
