import json
import re
from pathlib import Path


OPERATION_KEYWORDS = {
    "click",
    "hover",
    "open",
    "select",
    "choose",
    "fill",
    "type",
    "enter",
    "login",
    "sign in",
    "submit",
    "book",
    "add to cart",
    "apply",
    "filter",
    "sort",
    "navigate",
    "go to",
    "find the page",
}

INFORMATION_KEYWORDS = {
    "what",
    "who",
    "when",
    "where",
    "which",
    "how much",
    "price",
    "cost",
    "list",
    "summarize",
    "identify",
    "find out",
    "compare",
}

URL_PATTERN = re.compile(r"https?://[^\s)>\"]+")
DOMAIN_PATTERN = re.compile(
    r"\b(?:in|on|at|from)\s+((?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:/[\w\-./?%&=]*)?",
    re.IGNORECASE,
)
GENERIC_DOMAIN_PATTERN = re.compile(
    r"\b((?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:/[\w\-./?%&=]*)?\b"
)
SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")


def strip_reasoning(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text.strip()


def extract_tag(text: str, tag: str) -> str:
    if not isinstance(text, str):
        return ""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def repair_missing_closing_tags(text: str, tags: list[str] | tuple[str, ...]) -> str:
    if not isinstance(text, str):
        return ""

    repaired = text
    for tag in tags:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        if open_tag in repaired and close_tag not in repaired:
            repaired = repaired.rstrip() + close_tag
    return repaired


def infer_task_mode(task: str) -> str:
    lowered = (task or "").lower()
    op_hits = sum(1 for item in OPERATION_KEYWORDS if item in lowered)
    info_hits = sum(1 for item in INFORMATION_KEYWORDS if item in lowered)

    if op_hits and info_hits:
        return "hybrid"
    if op_hits:
        return "operation"
    if info_hits:
        return "information"
    return "hybrid"


def extract_start_url(task: str, explicit_web: str = "") -> str:
    if explicit_web:
        return normalize_url(explicit_web)

    task = task or ""

    match = URL_PATTERN.search(task)
    if match:
        return normalize_url(match.group(0))

    match = DOMAIN_PATTERN.search(task)
    if match:
        return normalize_url(match.group(1))

    match = GENERIC_DOMAIN_PATTERN.search(task)
    if match:
        return normalize_url(match.group(1))

    return ""


def normalize_url(url: str) -> str:
    url = (url or "").strip().rstrip(".,);]")
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def compact_som_text(text: str, max_chars: int = 0) -> str:
    text = (text or "").replace("\t", "\n").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def compact_observation_text(text: str, max_chars: int = 0) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def sanitize_surrogates(value):
    if isinstance(value, str):
        return SURROGATE_PATTERN.sub("\ufffd", value)
    if isinstance(value, list):
        return [sanitize_surrogates(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_surrogates(item) for item in value)
    if isinstance(value, dict):
        return {
            sanitize_surrogates(key) if isinstance(key, str) else key: sanitize_surrogates(item)
            for key, item in value.items()
        }
    return value


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def append_jsonl(path: str, obj: dict) -> None:
    obj = sanitize_surrogates(obj)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_task_id(raw: str, fallback: str) -> str:
    value = (raw or "").strip()
    if not value:
        value = fallback
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:120] or fallback


def save_json(path: str | Path, obj: dict) -> None:
    obj = sanitize_surrogates(obj)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def coord_scale_for_model(model: str) -> int:
    """Return the coord-scale convention used by the given vision model.

    Different VLM families train on different coordinate spaces:
      kimi*                            -> 1     (relative 0.0..1.0 floats)
      qwen* (qwen-VL, qwen3.7-plus/max) -> 1000 (0..1000 integer grid)
      claude / sonnet / opus / haiku   -> 0     (raw viewport pixels)
      gpt-4v / gpt-4o                  -> 0     (raw viewport pixels)
      unknown                          -> 1     (safest fallback — out-of-range
                                                 pixel coords would loudly error
                                                 instead of silently misclicking)

    Used by `--coord-scale -1` (the default) to pick a scale based on
    `--model` so users don't have to remember per-family conventions.
    Explicit `--coord-scale` always wins.
    """
    m = (model or "").lower()
    if "kimi" in m:
        return 1
    if "qwen" in m:
        return 1000
    if any(x in m for x in ("claude", "sonnet", "opus", "haiku")):
        return 0
    if any(x in m for x in ("gpt-4v", "gpt-4o", "gpt4v", "gpt4o")):
        return 0
    return 1
