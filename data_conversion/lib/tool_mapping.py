"""Source-tool → hybrid-tool deterministic mapping.

Source vocabularies analyzed in the dataset survey:
  - OpenSeeker: search({query: array<str>}), visit({url, goal})
  - REDSearcher: search({query: str, topn?, source?}),
                 visit({url, goal}),
                 google_scholar({query: str}),
                 PythonInterpreter({}) + outer <code>...</code>,
                 google_maps({q, page?})    <-- unsupported; drop the trajectory.

All target tools are hybrid search-path tools:
  - search({query: array<str>})
  - scholar({query: array<str>})
  - visit({url, goal, parse_type?})
  - code_sandbox({code, language?, ...})
"""

from __future__ import annotations

from typing import Any


class UnsupportedToolError(Exception):
    """Raised when a source tool has no hybrid equivalent (e.g. google_maps)."""


def _ensure_query_array(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    return [str(value)]


def map_tool_call(name: str, arguments: dict, *, attached_code: str | None = None) -> dict:
    """Translate a source tool call into a hybrid tool call.

    `attached_code` is only used for REDSearcher's PythonInterpreter pattern
    where the actual code lives in a sibling <code>...</code> block outside the
    tool_call JSON.

    Returns a dict {"name": <hybrid_name>, "arguments": {...}}.
    Raises UnsupportedToolError if the source tool cannot be mapped.
    """
    arguments = arguments or {}

    if name == "search":
        return {
            "name": "search",
            "arguments": {"query": _ensure_query_array(arguments.get("query"))},
        }

    if name == "visit":
        out = {"goal": str(arguments.get("goal", "")).strip()}
        url = arguments.get("url")
        if url is not None:
            out["url"] = url
        out["parse_type"] = arguments.get("parse_type", "auto")
        return {"name": "visit", "arguments": out}

    if name == "google_scholar":
        return {
            "name": "scholar",
            "arguments": {"query": _ensure_query_array(arguments.get("query"))},
        }

    if name == "PythonInterpreter":
        code = attached_code if attached_code is not None else ""
        if not code.strip():
            # Some samples invoke PythonInterpreter with arguments.code populated
            # directly instead of an external <code> block — accept that path too.
            code = str(arguments.get("code", ""))
        out = {
            "code": code,
            "language": "python",
        }
        # forward optional stdin / timeouts if the source happened to include them
        for k in ("stdin", "run_timeout", "compile_timeout"):
            if k in arguments and arguments[k] is not None:
                out[k] = arguments[k]
        return {"name": "code_sandbox", "arguments": out}

    if name == "google_maps":
        raise UnsupportedToolError("google_maps has no hybrid equivalent")

    # Defensive: if the source ever emits a tool we haven't seen, surface it.
    raise UnsupportedToolError(f"Unknown source tool: {name!r}")


SUPPORTED_SOURCE_TOOLS = {"search", "visit", "google_scholar", "PythonInterpreter"}
DROP_SOURCE_TOOLS = {"google_maps"}
