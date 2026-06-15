"""Source-format parsers shared by Stage 1.

These extract per-turn payload from raw OpenSeeker / REDSearcher assistant text:
  - the <think> reasoning block (used as raw material for the LLM rewriter)
  - all <tool_call>{json}</tool_call> entries inside the turn
  - any sibling <code>...</code> block (REDSearcher PythonInterpreter pattern)
  - the <answer>...</answer> string when the agent terminates this turn
"""

from __future__ import annotations

import json
import re
from typing import Iterator


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>(.*?)(?=<tool_call>|<answer>|$)",
    re.DOTALL,
)
CODE_BLOCK_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)


def extract_think(text: str) -> str:
    if not text:
        return ""
    matches = THINK_RE.findall(text)
    if not matches:
        return ""
    return matches[-1].strip()  # last <think> if multiple (rare)


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def extract_tool_calls(text: str) -> list[tuple[dict, str]]:
    """Return [(tool_call_dict, attached_code_or_empty)] in order of appearance.

    `attached_code` is the text from any <code>...</code> block that immediately
    follows this <tool_call> block (REDSearcher's PythonInterpreter pattern).
    Empty string when no code block is attached.

    Skips malformed JSON inside tool_call blocks.
    """
    if not text:
        return []
    out: list[tuple[dict, str]] = []
    for m in TOOL_CALL_BLOCK_RE.finditer(text):
        raw_json = m.group(1)
        tail = m.group(2) or ""
        try:
            tc = json.loads(raw_json)
        except Exception:
            continue
        if not isinstance(tc, dict):
            continue
        code_m = CODE_BLOCK_RE.search(tail)
        attached_code = code_m.group(1) if code_m else ""
        out.append((tc, attached_code))
    return out


# -----------------------------------------------------------------------------
# Source iterators
# -----------------------------------------------------------------------------


def iter_openseeker_trajectories(path: str) -> Iterator[dict]:
    """Yield one parsed dict per line from the OpenSeeker jsonl.

    Each yielded dict has the same shape as the source line:
      {
        "question": str,
        "answer": str,
        "number of tool calls": int,
        "trajectory": list[dict[role, content]],
        "trajectory correctness": "Correct" | "Incorrect",
      }
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_redsearcher_rows(parquet_glob: str) -> Iterator[dict]:
    """Yield rows from REDSearcher parquet shards via duckdb.

    Each yielded dict:
      {
        "meta": {"question": str, "answer": str, "language": str},
        "messages": list[dict],   # role, content
        "system_prompt": str,
      }
    """
    import duckdb

    con = duckdb.connect()
    cursor = con.execute(
        f"SELECT meta, messages, system_prompt FROM read_parquet('{parquet_glob}')"
    )
    while True:
        chunk = cursor.fetchmany(64)
        if not chunk:
            break
        for meta, messages, system_prompt in chunk:
            yield {
                "meta": meta or {},
                "messages": messages or [],
                "system_prompt": system_prompt or "",
            }
