#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any
from urllib.parse import quote, urlparse

from openai import OpenAI
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = SCRIPT_DIR.parent

DEFAULT_WEBVOYAGER_RUN_DIR = Path(
    "/data/czj/HybridWebAgentBenchmarks/run_v2/webvoyager_643_final_kimi_k25_summary8001_w4_max100_20260421"
)
DEFAULT_WEBVOYAGER_ANNOTATIONS = Path(
    "/data/czj/HybridWebAgentBenchmarks/data_v2/annotation_service/web_webvoyager_643_annotations.json"
)
DEFAULT_WEBVOYAGER_CLASSIFICATIONS = Path(
    "/data/czj/answer_eval_framework/outputs/online_m2w_webvoyager.validation_type_classification.kimi-k2.5.jsonl"
)
DEFAULT_CLASSIFICATION_DATASET_FILE = "web_webvoyager_643_final.jsonl"
DEFAULT_JINA_READER_BASE_URL = "http://127.0.0.1:8766"
KNOWN_JINA_PREFIXES = (
    "https://r.jina.ai/",
    "http://r.jina.ai/",
    "http://127.0.0.1:8766/",
    "http://localhost:8766/",
)

PROMPT_PATHS = {
    "static_answer": FRAMEWORK_ROOT / "prompts" / "judge_static_answer.txt",
    "answer_with_evidence": FRAMEWORK_ROOT / "prompts" / "judge_answer_with_evidence.txt",
    "defer_stage1": FRAMEWORK_ROOT / "prompts" / "judge_defer_stage1.txt",
    "select_relevant_steps": FRAMEWORK_ROOT / "prompts" / "select_relevant_steps.txt",
    "defer_truthfulness": FRAMEWORK_ROOT / "prompts" / "judge_defer_truthfulness.txt",
    "operation_key_point": FRAMEWORK_ROOT / "prompts" / "judge_operation_key_point.txt",
    # Stateful defer_framework judge (replaces the three-stage flow).
    "defer_classify_points": FRAMEWORK_ROOT / "prompts" / "defer_classify_points.txt",
    "defer_update_state": FRAMEWORK_ROOT / "prompts" / "defer_update_state.txt",
    "defer_finalize": FRAMEWORK_ROOT / "prompts" / "defer_finalize.txt",
    # Stateful operation_key_points judge (timing-aware page-state tracking).
    "operation_update_state": FRAMEWORK_ROOT / "prompts" / "operation_update_state.txt",
    "operation_finalize": FRAMEWORK_ROOT / "prompts" / "operation_finalize.txt",
}
VALIDATION_TYPES = {
    "static_answer",
    "target_urls",
    "reveal_script",
    "defer_framework",
    "operation_key_points",
}
LEGACY_TYPE_MAP = {
    "1": "static_answer",
    "2": "target_urls",
    "3": "defer_framework",
    "4": "operation_key_points",
    "ua": "UA",
    "UA": "UA",
    "unannotatable": "UA",
    "direct_answer": "static_answer",
    "webpage_content": "target_urls",
    "intermediate_steps": "operation_key_points",
    "custom_method": "defer_framework",
}


@dataclass
class AnnotationRecord:
    task_id: str
    annotation_type: str
    reference_answers: list[str]
    target_urls: list[str]
    reveal_script: str
    defer_reason: str
    operation_key_points: list[str]
    updated_at: str
    answer_key_points: list[str] = field(default_factory=list)
    annotation_types: list[str] = field(default_factory=list)
    notes: str = ""
    page_state_timing: str = ""
    page_states: list[dict[str, str]] = field(default_factory=list)
    revision_enabled: bool = False
    revised_question: str = ""
    revision_reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate web-agent trajectories using the manual annotation taxonomy "
            "(static_answer / target_urls / reveal_script / defer_framework / operation_key_points)."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_WEBVOYAGER_RUN_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_WEBVOYAGER_ANNOTATIONS)
    parser.add_argument("--classification", type=Path, default=DEFAULT_WEBVOYAGER_CLASSIFICATIONS)
    parser.add_argument("--classification-dataset-file", default=DEFAULT_CLASSIFICATION_DATASET_FILE)
    parser.add_argument(
        "--answer-key-points",
        type=Path,
        default=None,
        help=(
            "Optional JSONL file keyed by task_id with answer_key_points. "
            "When present, defer_framework stage 1/2/3 reuse these key points instead of generating new ones."
        ),
    )
    parser.add_argument(
        "--point-classification",
        type=Path,
        default=None,
        help=(
            "Optional JSONL produced by scripts/classify_defer_answer_points.py. "
            "When present and a task_id is found inside, the defer_framework Phase A "
            "(answer_bearing vs evidence_bearing) is read from the cache and the LLM is NOT "
            "called for that task. Missing task_ids fall back to live classification."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "kimi-k2.5"))
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument(
        "--defer-temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature for the stateful defer_framework judge (classify / per-step "
            "update / finalize LLM calls). Defaults to 0 for reproducible single-pass scoring. "
            "Raise it (e.g. 0.6) together with --repeat-count to draw independent samples for "
            "majority voting. Other evaluation types always stay at 0."
        ),
    )
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-retry-sleep", type=float, default=2.0)
    parser.add_argument("--browser-snapshot-command", default="browser_snapshot")
    parser.add_argument("--browser-snapshot-timeout", type=int, default=120)
    parser.add_argument("--jina-reader-base-url", default=os.getenv("JINA_READER_BASE_URL", DEFAULT_JINA_READER_BASE_URL))
    parser.add_argument("--jina-api-key", default="")
    parser.add_argument("--jina-api-key-env", default="JINA_API_KEY")
    parser.add_argument("--webpage-fetch-timeout", type=int, default=90)
    parser.add_argument("--webpage-screenshot-timeout", type=int, default=90000)
    parser.add_argument("--webpage-screenshot-wait-ms", type=int, default=1500)
    parser.add_argument("--webpage-screenshot-viewport-width", type=int, default=1365)
    parser.add_argument("--webpage-screenshot-viewport-height", type=int, default=768)
    parser.add_argument("--webpage-screenshot-browser-executable", default=os.getenv("WEBPAGE_SCREENSHOT_BROWSER_EXECUTABLE", ""))
    parser.add_argument("--reveal-script-timeout", type=int, default=180)
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=1, help="Evaluate each discovered task/mode pair this many times.")
    parser.add_argument("--resume-existing", action="store_true", help="Keep existing eval_results.jsonl rows and only evaluate missing task/mode pairs.")
    return parser.parse_args()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def normalize_page_states(value: Any, default_timing: str = "") -> list[dict[str, str]]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    page_states: list[dict[str, str]] = []
    for raw_item in raw_items:
        timing = default_timing if default_timing in {"final", "intermediate"} else ""
        state = ""
        if isinstance(raw_item, dict):
            state_value = raw_item.get("state") or raw_item.get("page_state") or raw_item.get("description") or raw_item.get("text")
            state = str(state_value or "").strip()
            item_timing = str(raw_item.get("timing") or raw_item.get("page_state_timing") or raw_item.get("position") or "").strip().lower()
            if item_timing in {"final", "intermediate"}:
                timing = item_timing
        elif isinstance(raw_item, str):
            state = raw_item.strip()
        elif raw_item is not None:
            state = str(raw_item).strip()
        if state:
            page_states.append({"timing": timing, "state": state})
    return page_states


def normalize_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            out.append(item)
            continue
        text = str(item or "").strip()
        if text.isdigit():
            out.append(int(text))
    return out


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def normalize_annotation_type(value: Any) -> str:
    raw = str(value or "").strip()
    if raw.upper() == "UA":
        return "UA"
    if raw in LEGACY_TYPE_MAP:
        return LEGACY_TYPE_MAP[raw]
    return raw


def normalize_annotation_types(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    for raw_item in raw_items:
        annotation_type = normalize_annotation_type(raw_item)
        if annotation_type and annotation_type not in normalized:
            normalized.append(annotation_type)
    return normalized


def annotation_type_source(annotation: AnnotationRecord | None, classification: dict[str, Any] | None) -> str:
    if annotation and (annotation.annotation_type or annotation.annotation_types):
        return "manual"
    if classification:
        return "classification"
    return "missing"


def load_annotations(path: Path) -> dict[str, AnnotationRecord]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict):
        items = [item for item in raw.values() if isinstance(item, dict)]
    elif isinstance(raw, list):
        items = [item for item in raw if isinstance(item, dict)]
    else:
        items = []

    annotations: dict[str, AnnotationRecord] = {}
    for item in items:
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        annotation_types = normalize_annotation_types(item.get("annotation_types"))
        if not annotation_types:
            annotation_types = normalize_annotation_types(item.get("types"))
        annotation_type = normalize_annotation_type(item.get("annotation_type"))
        if not annotation_type:
            annotation_type = normalize_annotation_type(item.get("type"))
        if not annotation_types and annotation_type:
            annotation_types = [annotation_type]
        if not annotation_type and annotation_types:
            annotation_type = annotation_types[0]
        reference_answers = normalize_text_list(item.get("reference_answers"))
        if not reference_answers:
            reference_answers = normalize_text_list(item.get("answer_facts"))
        if not reference_answers:
            reference_answers = normalize_text_list(item.get("reference_facts"))
        if not reference_answers:
            reference_answers = normalize_text_list(item.get("static_answers"))
        if not reference_answers:
            reference_answers = normalize_text_list(item.get("answer_keywords"))
        if not reference_answers and "static_answer" in annotation_types and "defer_framework" not in annotation_types:
            reference_answers = normalize_text_list(item.get("answer_points"))
        target_urls = normalize_text_list(item.get("target_urls"))
        if not target_urls:
            target_urls = normalize_text_list(item.get("urls"))
        if not target_urls:
            target_url = item.get("target_url")
            if isinstance(target_url, str) and target_url.strip():
                target_urls = [target_url.strip()]
        reveal_script = str(item.get("reveal_script") or "").strip()
        page_state_timing = str(item.get("page_state_timing") or item.get("page_state_position") or "").strip().lower()
        if page_state_timing not in {"final", "intermediate"}:
            page_state_timing = ""
        page_states = normalize_page_states(item.get("page_states"), page_state_timing)
        if not page_states:
            page_states = normalize_page_states(item.get("required_page_states"), page_state_timing)
        operation_key_points = [page_state["state"] for page_state in page_states]
        if not operation_key_points:
            operation_key_points = normalize_text_list(item.get("required_page_states"))
        if not operation_key_points:
            operation_key_points = normalize_text_list(item.get("operation_key_points"))
        if not operation_key_points:
            operation_key_points = normalize_text_list(item.get("step_key_points"))
        if not operation_key_points:
            operation_key_points = normalize_text_list(item.get("operation_points"))
        answer_key_points = normalize_text_list(item.get("answer_key_points"))
        if not answer_key_points:
            answer_key_points = normalize_text_list(item.get("key_points"))
        if not answer_key_points:
            answer_key_points = normalize_text_list(item.get("final_answer_key_points"))
        if not answer_key_points and "defer_framework" in annotation_types:
            answer_key_points = normalize_text_list(item.get("answer_points"))
        defer_reason = str(item.get("defer_reason") or "").strip()
        if not defer_reason:
            defer_reason = str(item.get("reason") or "").strip()
        if not defer_reason:
            parts = [str(item.get("custom_annotation_method") or "").strip(), str(item.get("custom_method_reason") or "").strip()]
            defer_reason = "\n\n".join(part for part in parts if part)
        notes = str(item.get("notes") or item.get("remark") or item.get("remarks") or "").strip()
        if not page_states and operation_key_points:
            page_states = [{"timing": page_state_timing, "state": state} for state in operation_key_points]

        task_revision = item.get("task_revision")
        revision_enabled = False
        revised_question = ""
        revision_reason = ""
        if isinstance(task_revision, dict):
            revision_enabled = coerce_bool(task_revision.get("enabled"))
            revised_question = str(task_revision.get("question") or "").strip()
            revision_reason = str(task_revision.get("reason") or "").strip()
        # Only treat the revision as active when it is enabled AND actually rewrites the question.
        if not (revision_enabled and revised_question):
            revision_enabled = False
            revised_question = ""

        annotations[task_id] = AnnotationRecord(
            task_id=task_id,
            annotation_type=annotation_type,
            reference_answers=reference_answers,
            target_urls=target_urls,
            reveal_script=reveal_script,
            defer_reason=defer_reason,
            operation_key_points=operation_key_points,
            updated_at=str(item.get("updated_at") or ""),
            answer_key_points=answer_key_points,
            annotation_types=annotation_types,
            notes=notes,
            page_state_timing=page_state_timing,
            page_states=page_states,
            revision_enabled=revision_enabled,
            revised_question=revised_question,
            revision_reason=revision_reason,
        )
    return annotations


def load_classifications(path: Path, dataset_file_filter: str) -> dict[str, dict[str, Any]]:
    rows = jsonl_load(path)
    data: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        dataset_file = row.get("dataset_file")
        if not isinstance(task_id, str) or not task_id:
            continue
        if dataset_file_filter and isinstance(dataset_file, str) and dataset_file.strip() and dataset_file.strip() != dataset_file_filter:
            continue
        cloned = dict(row)
        cloned["validation_type"] = normalize_annotation_type(cloned.get("validation_type"))
        data[task_id] = cloned
    return data


def load_answer_key_points(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    rows = jsonl_load(path)
    key_points_by_task: dict[str, list[str]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        key_points = normalize_text_list(row.get("answer_key_points"))
        if not key_points:
            key_points = normalize_text_list(row.get("key_points"))
        if not key_points:
            key_points = normalize_text_list(row.get("final_answer_key_points"))
        if key_points:
            key_points_by_task[task_id] = key_points
    return key_points_by_task


# Module-level registry populated in main() from --point-classification. Read by
# evaluate_defer_framework() to short-circuit Phase A without bloating the per-row
# `classification` dict that gets persisted into eval_results.jsonl.
_POINT_CLASSIFICATION_CACHE: dict[str, dict[str, Any]] = {}


def load_point_classification(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load the point-classification cache for defer_framework Phase A.

    Returns a mapping task_id -> {"answer_key_points", "kinds", "classifications",
    "model", "prompt_path", "generated_at", "raw_response", "user_prompt",
    "request_attempt"}. Only rows with status=="ok" and non-empty kinds are returned;
    missing files, malformed lines, and status=="error" rows are silently dropped
    (treated as a cache miss so the caller falls back to live classification).
    """
    if path is None or not Path(path).exists():
        return {}
    try:
        rows = jsonl_load(Path(path))
    except Exception as exc:
        sys.stderr.write(f"[point-classification] failed to load {path}: {exc}\n")
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        status = str(row.get("status") or "").strip().lower()
        if status and status != "ok":
            continue
        kinds_raw = row.get("kinds")
        if not isinstance(kinds_raw, list) or not kinds_raw:
            continue
        kinds = [normalize_kind(k) or "evidence_bearing" for k in kinds_raw]
        akp = row.get("answer_key_points")
        if not isinstance(akp, list):
            akp = []
        akp = [str(p) for p in akp]
        classifications = row.get("classifications")
        if not isinstance(classifications, list):
            classifications = []
        # Dual-axis: prefer per-point `axes` (a list of axis-lists) when present, either at
        # the row level (`axes_by_point`) or inside each classification entry (`axes`).
        # Fall back to deriving single axes from `kinds` for legacy single-kind caches.
        axes_by_point: list[list[str]] = []
        row_axes = row.get("axes_by_point")
        if isinstance(row_axes, list) and len(row_axes) == len(kinds):
            axes_by_point = [normalize_axes(a) for a in row_axes]
        elif classifications and all(isinstance(c, dict) and c.get("axes") is not None for c in classifications) and len(classifications) == len(kinds):
            axes_by_point = [normalize_axes(c.get("axes")) for c in sorted(classifications, key=lambda c: c.get("point_id", 0))]
        else:
            axes_by_point = [normalize_axes(k) for k in kinds]
        cache[task_id] = {
            "answer_key_points": akp,
            "kinds": kinds,
            "axes_by_point": axes_by_point,
            "classifications": classifications,
            "model": row.get("model", ""),
            "prompt_path": row.get("prompt_path", ""),
            "generated_at": row.get("generated_at", ""),
            "raw_response": row.get("raw_response", ""),
            "user_prompt": row.get("user_prompt", ""),
            "request_attempt": row.get("request_attempt", 0),
        }
    return cache


def detect_modes(run_dir: Path, requested_modes: list[str] | None) -> list[str]:
    preferred = requested_modes or ["search", "operation", "hybrid"]
    return [mode for mode in preferred if (run_dir / mode).is_dir()]


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty response")
    if raw.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
        if fenced:
            raw = fenced.group(1).strip()
    if not raw.startswith("{"):
        match = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("no JSON object found")
        raw = match.group(1).strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("response is not a JSON object")
    return obj


def build_client(args: argparse.Namespace) -> OpenAI:
    if not args.api_key:
        raise SystemExit("Missing --api-key or OPENAI_API_KEY")
    kwargs: dict[str, Any] = {"api_key": args.api_key}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.request_timeout and args.request_timeout > 0:
        kwargs["timeout"] = args.request_timeout
    return OpenAI(**kwargs)


def call_llm_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    max_tokens: int,
    request_max_retries: int,
    request_retry_sleep: float,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], str, int]:
    last_error = ""
    for attempt in range(1, max(1, request_max_retries) + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = completion.choices[0].message.content or ""
            return extract_json_object(raw_text), raw_text, attempt
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < request_max_retries:
                time.sleep(request_retry_sleep * attempt)
                continue
    raise RuntimeError(last_error or "LLM request failed")


def get_step_tool_name(record: dict[str, Any]) -> str:
    tool_call = record.get("tool_call") or {}
    name = str(tool_call.get("name") or "").strip()
    if name:
        return name
    if record.get("answer") is not None:
        return "answer"
    return "browser"


def get_step_kind(record: dict[str, Any]) -> str:
    tool_name = get_step_tool_name(record)
    if tool_name in {"search", "scholar", "visit", "code_sandbox", "answer"}:
        return tool_name
    # hybrid_routed mode-switch actions (S<->O) carry no page evidence of their own.
    # Treat them as a distinct, evidence-free kind so they are never given a screenshot,
    # never deduped as a browser state, and never judged as page state.
    if tool_name in {"switch"}:
        return "switch"
    return "browser"


def get_step_evidence_guidance(record: dict[str, Any]) -> str:
    step_kind = get_step_kind(record)
    if step_kind == "search":
        return (
            "Primary evidence for this step is the textual search result content. "
            "Judge whether it identifies candidate pages, sites, facts, or constraints relevant to verifying the final answer."
        )
    if step_kind == "scholar":
        return (
            "Primary evidence for this step is the textual scholarly result content. "
            "Judge whether it identifies relevant papers, metadata, citation counts, source URLs, or PDF URLs "
            "that matter for verifying the final answer."
        )
    if step_kind == "visit":
        return (
            "Primary evidence for this step is the textual visit observation, including extracted page or PDF content. "
            "Any attached visit screenshot would only be auxiliary."
        )
    if step_kind == "code_sandbox":
        return (
            "Primary evidence for this step is the textual sandbox output or computed result. "
            "Judge whether the computation, parsing, extraction, or validation materially supports or contradicts the final answer."
        )
    if step_kind == "answer":
        return (
            "This step is an answer claim. Treat it as auxiliary consistency evidence only. "
            "It is relevant only if it directly exposes a claim or contradiction tied to the key point."
        )
    if step_kind == "switch":
        return (
            "This step only switches the agent's mode (search <-> operation). It carries no page "
            "evidence of its own. Do not change any key point's status from this step."
        )
    return (
        "Primary evidence for this step is the resulting browser state AFTER the action ran. The "
        "Current Screenshot attached to this step is that after-action page and is a primary source "
        "of visible page facts (ratings, prices, counts, in-stock, labels, list contents) for THIS "
        "browser step. (The same facts are equally valid when established on the search side from a "
        "search/visit/scholar text result; a browser screenshot is not more authoritative than text "
        "evidence — it is just this step's modality.) The URL and Title are the page reached. The "
        "Observation here is the action-execution receipt (status, start_url/end_url, target) — use it "
        "for what the action did and where it landed, NOT as page content; for this browser step read "
        "ratings/prices from the screenshot rather than expecting them in the Observation text."
    )


def get_pre_answer_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pre_answer: list[dict[str, Any]] = []
    for record in records:
        if get_step_kind(record) == "answer":
            break
        pre_answer.append(record)
    return pre_answer


def compose_observation_text(record: dict[str, Any]) -> str:
    step_kind = get_step_kind(record)
    current_url = str(record.get("current_url") or "").strip()
    current_title = str(record.get("current_title") or "").strip()
    som_text = str(record.get("som_text") or "").strip()
    raw_observation = str(record.get("observation") or "").strip()

    if step_kind == "browser":
        parts = []
        if current_url:
            parts.append(f"URL: {current_url}")
        if current_title:
            parts.append(f"Title: {current_title}")
        if som_text:
            parts.append(f"SoM: {som_text}")
        if raw_observation:
            parts.append(f"State: {raw_observation}")
        return "\n".join(parts).strip()

    if step_kind == "switch":
        # Mode-switch action: report only that a switch happened. Do not surface any
        # leftover page text in `observation`, which would be stale pre-switch state.
        mode_after = str(record.get("active_mode_after") or "").strip()
        mode_before = str(record.get("active_mode_before") or "").strip()
        if mode_before and mode_after:
            return f"Agent switched mode: {mode_before} -> {mode_after}. No page evidence."
        if mode_after:
            return f"Agent switched mode to {mode_after}. No page evidence."
        return "Agent switched mode (search <-> operation). No page evidence."

    return raw_observation


def get_current_screenshot_path(task: dict[str, Any], record: dict[str, Any]) -> str:
    """Resolve THIS browser step's after-action screenshot.

    Convention (verified on both HybridWebAgentBenchmarks run_v2 and hybrid_routed
    trajectories): step N's own `img_path` / same-numbered `N_full_screenshot.*` is the
    page state AFTER step N's action executed. So we bind a browser step to its OWN
    screenshot, not the next step's — this fixes an off-by-one (the old code judged a step
    against the page reached after the FOLLOWING action) and lets the LAST browser step
    (often the landing/answer page) carry a screenshot.

    Resolution order:
      1. the step's own `img_path` field (absolute, or relative to task_dir / its trajectory)
      2. fallback: `<task_dir>/trajectory/<step>_full_screenshot.<ext>`
    """
    if get_step_kind(record) != "browser":
        return ""
    task_dir_raw = str(task.get("task_dir") or "").strip()
    trajectory_dir = (Path(task_dir_raw) / "trajectory") if task_dir_raw else None

    # 1. Prefer the step's own img_path field — it points at the after-action frame and
    #    is naming-agnostic (handles both `screenshotN.png` and `N_full_screenshot.png`).
    img_path_raw = str(record.get("img_path") or "").strip()
    if img_path_raw:
        candidates = [Path(img_path_raw)]
        if not Path(img_path_raw).is_absolute() and trajectory_dir is not None:
            name = Path(img_path_raw).name
            candidates.append(trajectory_dir / img_path_raw)
            candidates.append(trajectory_dir / name)
            candidates.append(Path(task_dir_raw) / img_path_raw)
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)

    # 2. Fallback: this step's own number as `<step>_full_screenshot.<ext>`.
    if trajectory_dir is None:
        return ""
    step_value = record.get("step")
    if isinstance(step_value, bool):
        return ""
    try:
        step_int = int(step_value)
    except Exception:
        return ""
    if step_int < 0:
        return ""
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = trajectory_dir / f"{step_int}_full_screenshot{suffix}"
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return ""


def build_image_content_item(image_path: str) -> dict[str, Any]:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
            "detail": "high",
        },
    }


def build_step_block_text(task: dict[str, Any], record: dict[str, Any]) -> tuple[str, str]:
    current_screenshot_path = get_current_screenshot_path(task, record)
    step_text = format_step(record, current_screenshot_path)
    return step_text, current_screenshot_path


def build_step_user_content(prefix_text: str, task: dict[str, Any], record: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    step_block_text, current_screenshot_path = build_step_block_text(task, record)
    prompt_text = "\n\n".join([prefix_text, step_block_text])
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    if current_screenshot_path:
        content.append(build_image_content_item(current_screenshot_path))
    return content, prompt_text, current_screenshot_path


def build_trajectory_user_content(
    intro_text: str,
    task: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    if not records:
        return [{"type": "text", "text": intro_text + "\n\n[no related steps selected]"}], intro_text + "\n\n[no related steps selected]", []

    content: list[dict[str, Any]] = [{"type": "text", "text": intro_text}]
    text_blocks: list[str] = [intro_text]
    current_screenshot_paths: list[str] = []
    for record in records:
        step_block_text, current_screenshot_path = build_step_block_text(task, record)
        content.append({"type": "text", "text": step_block_text})
        text_blocks.append(step_block_text)
        if current_screenshot_path:
            content.append(build_image_content_item(current_screenshot_path))
            current_screenshot_paths.append(current_screenshot_path)
    return content, "\n\n".join(text_blocks), current_screenshot_paths


def format_step(record: dict[str, Any], current_screenshot_path: str = "") -> str:
    tool_call = record.get("tool_call") or {}
    tool_name = get_step_tool_name(record)
    tool_args = tool_call.get("arguments") or {}
    step = record.get("step", "")
    observation = compose_observation_text(record)
    if current_screenshot_path:
        observation = "\n".join(
            part
            for part in [
                observation,
                "Current Screenshot: attached immediately after this step block.",
            ]
            if part
        )
    report = str(record.get("report") or "").strip()
    return (
        f"Step {step}\n"
        f"Tool: {tool_name}\n"
        f"Arguments: {json.dumps(tool_args, ensure_ascii=False)}\n"
        f"Observation: {observation}\n"
        f"Report: {report}"
    )


def format_trajectory(records: list[dict[str, Any]]) -> str:
    return "\n\n".join(format_step(record) for record in records)


def parse_snapshot_output(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    try:
        obj = json.loads(text)
    except Exception:
        return text
    if isinstance(obj, dict):
        for key in ("text", "content", "snapshot", "markdown", "body"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return text


def normalize_url(url: str) -> str:
    normalized = str(url or "").strip()
    if normalized and not normalized.startswith(("http://", "https://")):
        normalized = "https://" + normalized
    return normalized


def _coerce_entity(raw: Any) -> dict[str, Any] | None:
    """Normalize an entity tag from a step LLM update into {url,title,name} or None.

    Returns None if the input is not a dict or every field is empty/whitespace, so
    that downstream code can treat absence as "no entity bound for this satisfaction"
    rather than "bound to an unknown entity"."""
    if not isinstance(raw, dict):
        return None
    url = str(raw.get("url") or "").strip()
    title = str(raw.get("title") or "").strip()
    name = str(raw.get("name") or "").strip()
    if not url and not title and not name:
        return None
    return {
        "url": normalize_url(url) if url else "",
        "title": title,
        "name": name,
    }


# Generic non-discriminating tokens stripped before comparing entity names.
# Kept small and brand-agnostic — we add category words that are too generic to
# distinguish two items, not real brand names. The list can be tuned per corpus.
_BRAND_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "with", "by",
    "pro", "plus", "max", "mini", "lite", "ultra", "premium",
    "new", "best", "official", "site",
    "recipe", "recipes", "paper", "papers", "article",
    "product", "item", "page",
})


def _name_tokens(name: str) -> set[str]:
    """Lowercased alphanumeric tokens of a name minus generic/category stopwords."""
    if not name:
        return set()
    s = name.lower()
    # Split on any non-alphanumeric character.
    raw: list[str] = []
    cur = []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                raw.append("".join(cur)); cur = []
    if cur:
        raw.append("".join(cur))
    return {t for t in raw if t and t not in _BRAND_STOPWORDS}


def _same_entity(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Conservative same-entity test. Returns True when entities cannot be told apart
    (treat unknown as NOT-a-conflict), False only when they are clearly different.

    Decision order:
      1. If either side is missing, return True (unknown ≠ conflict).
      2. Compare NAMES first when both sides have a non-empty name. If both name
         token-sets reduce to only stopwords, fall through to URL.
      3. Otherwise use URL equality (after normalize_url) as the decisive signal.
    The aim is to FALSE-NEGATIVE (abstain) more often than FALSE-POSITIVE (wrongly
    flag a conflict). The harness elsewhere downgrades unprovable axes to PARTIAL
    when entities are unknown — that fail-safe path is preferred to silently passing
    a wrong entity match."""
    if not a or not b:
        return True
    na = str(a.get("name") or "").strip().lower()
    nb = str(b.get("name") or "").strip().lower()
    if na and nb:
        ta = _name_tokens(na)
        tb = _name_tokens(nb)
        if ta and tb:
            if na == nb:
                return True
            inter = len(ta & tb)
            union = len(ta | tb) or 1
            if inter / union >= 0.6:
                return True
            # Distinct, non-empty token sets → clearly different entities.
            return False
        # One side reduces to all-stopwords; fall through to URL.
    ua = normalize_url(str(a.get("url") or "").strip())
    ub = normalize_url(str(b.get("url") or "").strip())
    if ua and ub:
        return ua == ub
    # No usable URL and at least one name missing → cannot decide, return True.
    return True


def build_jina_headers(args: argparse.Namespace) -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = args.jina_api_key or os.environ.get(args.jina_api_key_env, "") or os.environ.get("JINA_API_KEYS", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    pool_token = os.environ.get("JINA_POOL_TOKEN", "")
    if pool_token:
        headers["X-Pool-Token"] = pool_token
    return headers


def build_jina_reader_url(url: str, args: argparse.Namespace) -> str:
    normalized = normalize_url(url)
    for prefix in KNOWN_JINA_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return f"{args.jina_reader_base_url.rstrip('/')}/{normalized}"


def parse_jina_response(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except Exception:
        return text
    if not isinstance(payload, dict):
        return text
    content = payload.get("content")
    if not isinstance(content, str):
        data = payload.get("data")
        if isinstance(data, dict):
            content = data.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return text


_JINA_RETRY_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})


def fetch_jina_page_content(url: str, args: argparse.Namespace) -> dict[str, Any]:
    normalized_url = normalize_url(url)
    reader_url = build_jina_reader_url(normalized_url, args)
    headers = build_jina_headers(args)
    max_attempts = max(1, int(args.request_max_retries))
    base_sleep = max(0.0, float(args.request_retry_sleep))
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(reader_url, headers=headers, timeout=args.webpage_fetch_timeout)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            time.sleep(base_sleep * attempt)
            continue
        if response.status_code in _JINA_RETRY_STATUS and attempt < max_attempts:
            time.sleep(base_sleep * attempt)
            continue
        response.raise_for_status()
        return {
            "url": normalized_url,
            "jina_reader_url": reader_url,
            "status_code": response.status_code,
            "text": parse_jina_response(response.text),
            "raw_output": response.text,
        }
    # Loop exited without returning: re-raise the last transient error if any,
    # otherwise raise on the last response's status.
    if last_exc is not None:
        raise last_exc
    response.raise_for_status()
    return {
        "url": normalized_url,
        "jina_reader_url": reader_url,
        "status_code": response.status_code,
        "text": parse_jina_response(response.text),
        "raw_output": response.text,
    }


def safe_url_slug(url: str) -> str:
    parsed = urlparse(normalize_url(url))
    seed = "_".join(part for part in [parsed.netloc, parsed.path.strip("/")] if part)
    if not seed:
        seed = quote(url, safe="")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", seed).strip("._-")
    return slug[:120] or "page"


def webpage_evidence_dir(task: dict[str, Any], args: argparse.Namespace) -> Path:
    path = args.output_dir / "webpage_evidence" / str(task["mode"]) / str(task["task_id"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_browser_executable(args: argparse.Namespace) -> str:
    configured = str(args.webpage_screenshot_browser_executable or "").strip()
    if configured:
        return configured
    for candidate in (
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ):
        if Path(candidate).exists():
            return candidate
    return ""


def capture_full_page_screenshot(url: str, task: dict[str, Any], index: int, args: argparse.Namespace) -> dict[str, Any]:
    normalized_url = normalize_url(url)
    output_dir = webpage_evidence_dir(task, args)
    screenshot_path = output_dir / f"{index:02d}_{safe_url_slug(normalized_url)}.png"

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError("Playwright is not installed in this Python environment.") from exc

    timeout = max(1, int(args.webpage_screenshot_timeout))
    viewport_width = max(1, int(args.webpage_screenshot_viewport_width))
    viewport_height = max(1, int(args.webpage_screenshot_viewport_height))
    with sync_playwright() as p:
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        }
        browser_executable = find_browser_executable(args)
        if browser_executable:
            launch_kwargs["executable_path"] = browser_executable
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            device_scale_factor=1,
        )
        page = context.new_page()
        try:
            page.goto(normalized_url, timeout=timeout, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout, 15000))
            except Exception:
                try:
                    page.wait_for_load_state("load", timeout=min(timeout, 5000))
                except Exception:
                    pass
            wait_ms = max(0, int(args.webpage_screenshot_wait_ms))
            if wait_ms:
                page.wait_for_timeout(wait_ms)

            dimensions = page.evaluate(
                """() => {
                    const doc = document.documentElement || {};
                    const body = document.body || {};
                    const width = Math.ceil(Math.max(
                        doc.scrollWidth || 0,
                        doc.offsetWidth || 0,
                        body.scrollWidth || 0,
                        body.offsetWidth || 0,
                        window.innerWidth || 0
                    ));
                    const height = Math.ceil(Math.max(
                        doc.scrollHeight || 0,
                        doc.offsetHeight || 0,
                        body.scrollHeight || 0,
                        body.offsetHeight || 0,
                        window.innerHeight || 0
                    ));
                    return {
                        scroll_width: width,
                        scroll_height: height,
                        viewport_width: window.innerWidth || 0,
                        viewport_height: window.innerHeight || 0,
                        device_pixel_ratio: window.devicePixelRatio || 1
                    };
                }"""
            )
            if isinstance(dimensions, dict) and int(dimensions.get("scroll_width") or 0) > viewport_width:
                page.set_viewport_size(
                    {
                        "width": int(dimensions["scroll_width"]),
                        "height": viewport_height,
                    }
                )
                page.wait_for_timeout(200)
                dimensions = page.evaluate(
                    """() => {
                        const doc = document.documentElement || {};
                        const body = document.body || {};
                        return {
                            scroll_width: Math.ceil(Math.max(doc.scrollWidth || 0, doc.offsetWidth || 0, body.scrollWidth || 0, body.offsetWidth || 0, window.innerWidth || 0)),
                            scroll_height: Math.ceil(Math.max(doc.scrollHeight || 0, doc.offsetHeight || 0, body.scrollHeight || 0, body.offsetHeight || 0, window.innerHeight || 0)),
                            viewport_width: window.innerWidth || 0,
                            viewport_height: window.innerHeight || 0,
                            device_pixel_ratio: window.devicePixelRatio || 1
                        };
                    }"""
                )

            page.screenshot(path=str(screenshot_path), full_page=True, timeout=timeout)
            final_url = page.url
            title = page.title()
            return {
                "url": normalized_url,
                "final_url": final_url,
                "title": title,
                "screenshot_path": str(screenshot_path),
                "page_dimensions": dimensions if isinstance(dimensions, dict) else {},
                "full_page": True,
            }
        finally:
            context.close()
            browser.close()


def fetch_target_url_evidence(url: str, task: dict[str, Any], index: int, args: argparse.Namespace) -> dict[str, Any]:
    normalized_url = normalize_url(url)
    block: dict[str, Any] = {
        "url": normalized_url,
        "text": "",
        "jina_reader_url": "",
        "jina_status_code": None,
        "screenshot_path": "",
        "final_url": "",
        "title": "",
        "page_dimensions": {},
        "errors": [],
    }

    try:
        jina = fetch_jina_page_content(normalized_url, args)
        block["jina_reader_url"] = jina["jina_reader_url"]
        block["jina_status_code"] = jina["status_code"]
        block["text"] = jina["text"]
        block["raw_jina_output"] = jina["raw_output"]
    except Exception as exc:
        block["errors"].append(f"jina_fetch: {type(exc).__name__}: {exc}")

    try:
        screenshot = capture_full_page_screenshot(normalized_url, task, index, args)
        block["screenshot_path"] = screenshot["screenshot_path"]
        block["final_url"] = screenshot["final_url"]
        block["title"] = screenshot["title"]
        block["page_dimensions"] = screenshot["page_dimensions"]
        block["full_page_screenshot"] = screenshot["full_page"]
    except Exception as exc:
        block["errors"].append(f"browser_screenshot: {type(exc).__name__}: {exc}")

    if not block["text"] and not block["screenshot_path"]:
        raise RuntimeError("; ".join(block["errors"]) or "No webpage evidence fetched.")
    return block


def run_browser_snapshot(url: str, args: argparse.Namespace) -> dict[str, Any]:
    command_template = args.browser_snapshot_command
    if "{url}" in command_template:
        cmd = shlex.split(command_template.format(url=url))
    else:
        cmd = shlex.split(command_template) + [url]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.browser_snapshot_timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "browser_snapshot command not found. Set --browser-snapshot-command to your local Playwright/MCP snapshot wrapper."
        ) from exc
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise RuntimeError(f"browser_snapshot failed ({proc.returncode}): {stderr.strip() or stdout.strip()}")
    return {
        "url": url,
        "command": cmd,
        "raw_output": stdout,
        "text": parse_snapshot_output(stdout),
    }


def run_reveal_script(script_text: str, task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="reveal_script_") as tmp_dir:
        script_path = Path(tmp_dir) / "reveal.py"
        script_path.write_text(script_text, encoding="utf-8")
        env = os.environ.copy()
        env["TASK_ID"] = str(task.get("task_id", ""))
        env["QUESTION"] = str(task.get("question", ""))
        env["MODE"] = str(task.get("mode", ""))
        env["RUN_DIR"] = str(task.get("run_dir", ""))
        env["BROWSER_SNAPSHOT_COMMAND"] = str(args.browser_snapshot_command)
        proc = subprocess.run(
            ["python3", str(script_path)],
            capture_output=True,
            text=True,
            timeout=args.reveal_script_timeout,
            check=False,
            env=env,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        if proc.returncode != 0:
            raise RuntimeError(f"reveal_script failed ({proc.returncode}): {stderr.strip() or stdout.strip()}")
        return {
            "raw_output": stdout,
            "text": parse_snapshot_output(stdout),
        }


def build_answer_with_evidence_prompt(task: dict[str, Any], prediction: str, evidence_blocks: list[dict[str, Any]]) -> str:
    evidence_text = []
    for idx, block in enumerate(evidence_blocks, start=1):
        evidence_text.append(
            f"Evidence {idx}\nURL: {block.get('url', '')}\nContent:\n{block.get('text', '')}"
        )
    return "\n\n".join(
        [
            "## Task",
            str(task.get("question", "")),
            "",
            "## Agent Final Answer",
            prediction,
            "",
            "## External Evidence",
            "\n\n".join(evidence_text) if evidence_text else "[no evidence]",
        ]
    )


def build_answer_with_webpage_evidence_user_content(
    task: dict[str, Any],
    prediction: str,
    evidence_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    intro_text = "\n\n".join(
        [
            "## Task",
            str(task.get("question", "")),
            "",
            "## Agent Final Answer",
            prediction,
            "",
            "## External Webpage Evidence",
            (
                "Each evidence block may include Jina reader text and a browser full-page screenshot. "
                "When a screenshot is attached, it belongs only to the evidence block immediately before it."
            ),
        ]
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": intro_text}]
    text_blocks: list[str] = [intro_text]
    for idx, block in enumerate(evidence_blocks, start=1):
        dimensions = block.get("page_dimensions")
        dimensions_text = json.dumps(dimensions, ensure_ascii=False) if isinstance(dimensions, dict) else "{}"
        screenshot_path = str(block.get("screenshot_path") or "")
        screenshot_line = (
            "Browser Full-Page Screenshot: attached immediately after this evidence block."
            if screenshot_path
            else "Browser Full-Page Screenshot: unavailable."
        )
        errors = block.get("errors")
        error_text = "\n".join(str(item) for item in errors) if isinstance(errors, list) and errors else ""
        jina_text = str(block.get("text") or "").strip()
        if jina_text:
            jina_section = ["Jina Reader Content:", jina_text]
        elif screenshot_path:
            jina_section = [
                "Jina Reader Content: (unavailable — Jina fetch returned no text for this URL; "
                "rely entirely on the attached full-page screenshot for this evidence block)",
            ]
        else:
            jina_section = [
                "Jina Reader Content: (unavailable — no Jina text AND no screenshot for this URL; "
                "this evidence block carries no usable content)",
            ]
        block_text = "\n".join(
            [
                f"Evidence {idx}",
                f"URL: {block.get('url', '')}",
                f"Final URL: {block.get('final_url', '')}",
                f"Title: {block.get('title', '')}",
                f"Jina Reader URL: {block.get('jina_reader_url', '')}",
                f"Jina Status Code: {block.get('jina_status_code', '')}",
                f"Page Dimensions: {dimensions_text}",
                f"Screenshot Path: {screenshot_path}",
                screenshot_line,
                *jina_section,
                "Evidence Fetch Errors:",
                error_text,
            ]
        )
        content.append({"type": "text", "text": block_text})
        text_blocks.append(block_text)
        if screenshot_path:
            content.append(build_image_content_item(screenshot_path))
    return content, "\n\n".join(text_blocks)


def discover_tasks(run_dir: Path, modes: list[str], task_filter: set[str] | None, limit: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for mode in modes:
        mode_tasks: dict[str, dict[str, Any]] = {}
        for result_path in sorted((run_dir / mode).glob("*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            task_id = str(result.get("task_id") or result_path.parent.name)
            if task_filter and task_id not in task_filter:
                continue
            task = {
                "run_dir": str(run_dir),
                "benchmark_id": run_dir.name,
                "mode": mode,
                "task_id": task_id,
                "question": str(result.get("question") or ""),
                "prediction": str(result.get("prediction") or "").strip(),
                "termination": str(result.get("termination") or ""),
                "task_dir": str(result_path.parent),
                "result_path": str(result_path),
                "records": result.get("records") or [],
            }
            existing = mode_tasks.get(task_id)
            if existing is None:
                mode_tasks[task_id] = task
                continue
            existing_is_canonical = Path(str(existing.get("task_dir") or "")).name == task_id
            candidate_is_canonical = result_path.parent.name == task_id
            if candidate_is_canonical and not existing_is_canonical:
                mode_tasks[task_id] = task
        tasks.extend(mode_tasks[task_id] for task_id in sorted(mode_tasks))
    if limit > 0:
        selected_task_ids: set[str] = set()
        filtered_tasks: list[dict[str, Any]] = []
        for task in tasks:
            task_id = str(task["task_id"])
            if task_id not in selected_task_ids:
                if len(selected_task_ids) >= limit:
                    continue
                selected_task_ids.add(task_id)
            if task_id in selected_task_ids:
                filtered_tasks.append(task)
        tasks = filtered_tasks
    return tasks


def make_missing_result(task: dict[str, Any], evaluation_type: str, status: str, reason: str, annotation: AnnotationRecord | None, classification: dict[str, Any] | None) -> dict[str, Any]:
    predicted_label: int | None = None
    if status in {"no_prediction", "not_solved"}:
        predicted_label = 0
    return {
        "benchmark_id": task["benchmark_id"],
        "mode": task["mode"],
        "task_id": task["task_id"],
        "question": task["question"],
        "evaluation_type": evaluation_type,
        "evaluation_status": status,
        "predicted_label": predicted_label,
        "reason": reason,
        "annotation_type_source": annotation_type_source(annotation, classification),
        "annotation": annotation.__dict__ if annotation else None,
        "classification": classification,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def make_error_result(task: dict[str, Any], evaluation_type: str, exc: Exception, annotation: AnnotationRecord | None, classification: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "benchmark_id": task["benchmark_id"],
        "mode": task["mode"],
        "task_id": task["task_id"],
        "question": task["question"],
        "evaluation_type": evaluation_type,
        "evaluation_status": "exception",
        "predicted_label": None,
        "reason": f"{type(exc).__name__}: {exc}",
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
        "annotation_type_source": annotation_type_source(annotation, classification),
        "annotation": annotation.__dict__ if annotation else None,
        "classification": classification,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def make_unannotatable_result(task: dict[str, Any], annotation: AnnotationRecord | None, classification: dict[str, Any] | None) -> dict[str, Any]:
    reason = ""
    if annotation:
        reason = annotation.defer_reason or ""
    return {
        "benchmark_id": task["benchmark_id"],
        "mode": task["mode"],
        "task_id": task["task_id"],
        "question": task["question"],
        "evaluation_type": "UA",
        "evaluation_status": "unannotatable",
        "predicted_label": None,
        "reason": reason or "Task was marked unannotatable.",
        "annotation_type_source": annotation_type_source(annotation, classification),
        "annotation": annotation.__dict__ if annotation else None,
        "classification": classification,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def evaluate_static_answer(task: dict[str, Any], annotation: AnnotationRecord, classification: dict[str, Any] | None, prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    if not annotation.reference_answers:
        return make_missing_result(task, "static_answer", "missing_annotation_fields", "Missing reference_answers.", annotation, classification)
    if not task["prediction"]:
        return make_missing_result(task, "static_answer", "no_prediction", "Agent did not provide a final answer.", annotation, classification)
    prompt = "\n\n".join(
        [
            "## Task",
            task["question"],
            "",
            "## Reference Answer Facts",
            "\n".join(f"- {item}" for item in annotation.reference_answers),
            "",
            "## Annotation Notes",
            annotation.notes or "(none)",
            "",
            "## Agent Final Answer",
            task["prediction"],
        ]
    )
    parsed, raw_text, attempt = call_llm_json(client, args.model, prompts["static_answer"], prompt, args.max_tokens, args.request_max_retries, args.request_retry_sleep)
    return {
        "benchmark_id": task["benchmark_id"],
        "mode": task["mode"],
        "task_id": task["task_id"],
        "question": task["question"],
        "evaluation_type": "static_answer",
        "evaluation_status": "completed",
        "predicted_label": 1 if int(parsed.get("predicted_label", 0)) == 1 else 0,
        "reason": str(parsed.get("reason") or ""),
        "annotation_type_source": annotation_type_source(annotation, classification),
        "annotation": annotation.__dict__,
        "classification": classification,
        "judge_prompt": prompt,
        "raw_judge_response": raw_text,
        "request_attempt": attempt,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def evaluate_target_urls(task: dict[str, Any], annotation: AnnotationRecord, classification: dict[str, Any] | None, prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    if not annotation.target_urls:
        return make_missing_result(task, "target_urls", "missing_annotation_fields", "Missing target_urls.", annotation, classification)
    if not task["prediction"]:
        return make_missing_result(task, "target_urls", "no_prediction", "Agent did not provide a final answer.", annotation, classification)

    evidence_blocks: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, url in enumerate(annotation.target_urls, start=1):
        try:
            evidence_blocks.append(fetch_target_url_evidence(url, task, index, args))
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    if not evidence_blocks:
        return make_missing_result(task, "target_urls", "evidence_fetch_failed", "; ".join(errors) or "No webpage evidence fetched.", annotation, classification)

    user_content, prompt = build_answer_with_webpage_evidence_user_content(task, task["prediction"], evidence_blocks)
    parsed, raw_text, attempt = call_llm_json(client, args.model, prompts["answer_with_evidence"], user_content, args.max_tokens, args.request_max_retries, args.request_retry_sleep)
    return {
        "benchmark_id": task["benchmark_id"],
        "mode": task["mode"],
        "task_id": task["task_id"],
        "question": task["question"],
        "evaluation_type": "target_urls",
        "evaluation_status": "completed",
        "predicted_label": 1 if int(parsed.get("predicted_label", 0)) == 1 else 0,
        "reason": str(parsed.get("reason") or ""),
        "evidence_errors": errors + [
            f"{block.get('url', '')}: {error}"
            for block in evidence_blocks
            for error in (block.get("errors") if isinstance(block.get("errors"), list) else [])
        ],
        "evidence_blocks": [
            {
                "url": block.get("url", ""),
                "final_url": block.get("final_url", ""),
                "title": block.get("title", ""),
                "jina_reader_url": block.get("jina_reader_url", ""),
                "jina_status_code": block.get("jina_status_code"),
                "text": block.get("text", ""),
                "screenshot_path": block.get("screenshot_path", ""),
                "page_dimensions": block.get("page_dimensions", {}),
                "full_page_screenshot": block.get("full_page_screenshot", False),
                "errors": block.get("errors", []),
            }
            for block in evidence_blocks
        ],
        "annotation_type_source": annotation_type_source(annotation, classification),
        "annotation": annotation.__dict__,
        "classification": classification,
        "judge_prompt": prompt,
        "raw_judge_response": raw_text,
        "request_attempt": attempt,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def evaluate_reveal_script(task: dict[str, Any], annotation: AnnotationRecord, classification: dict[str, Any] | None, prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    if not annotation.reveal_script:
        return make_missing_result(task, "reveal_script", "missing_annotation_fields", "Missing reveal_script.", annotation, classification)
    if not task["prediction"]:
        return make_missing_result(task, "reveal_script", "no_prediction", "Agent did not provide a final answer.", annotation, classification)
    try:
        evidence = run_reveal_script(annotation.reveal_script, task, args)
    except Exception as exc:
        return make_missing_result(task, "reveal_script", "evidence_fetch_failed", f"{type(exc).__name__}: {exc}", annotation, classification)

    prompt = build_answer_with_evidence_prompt(task, task["prediction"], [{"url": "reveal_script://local", "text": evidence["text"]}])
    parsed, raw_text, attempt = call_llm_json(client, args.model, prompts["answer_with_evidence"], prompt, args.max_tokens, args.request_max_retries, args.request_retry_sleep)
    return {
        "benchmark_id": task["benchmark_id"],
        "mode": task["mode"],
        "task_id": task["task_id"],
        "question": task["question"],
        "evaluation_type": "reveal_script",
        "evaluation_status": "completed",
        "predicted_label": 1 if int(parsed.get("predicted_label", 0)) == 1 else 0,
        "reason": str(parsed.get("reason") or ""),
        "reveal_script_output": evidence["text"],
        "annotation_type_source": annotation_type_source(annotation, classification),
        "annotation": annotation.__dict__,
        "classification": classification,
        "judge_prompt": prompt,
        "raw_judge_response": raw_text,
        "request_attempt": attempt,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def format_key_points_block(key_points: list[str], empty_text: str = "(none provided)") -> str:
    return "\n".join(f"- {point}" for point in key_points if point.strip()) or empty_text


def get_defer_answer_key_points(annotation: AnnotationRecord | None, classification: dict[str, Any] | None) -> list[str]:
    if annotation and annotation.answer_key_points:
        return annotation.answer_key_points
    if classification:
        for field_name in ("answer_key_points", "key_points", "final_answer_key_points"):
            key_points = normalize_text_list(classification.get(field_name))
            if key_points:
                return key_points
    return []


def normalize_kind(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"answer_bearing", "answer", "answer-bearing", "answerbearing"}:
        return "answer_bearing"
    if text in {"evidence_bearing", "evidence", "evidence-bearing", "evidencebearing"}:
        return "evidence_bearing"
    return ""


def normalize_status(value: Any, default: str = "pending") -> str:
    text = str(value or "").strip().lower()
    if text in {"satisfied", "contradicted", "partial", "pending"}:
        return text
    return default


def classify_answer_points(task: dict[str, Any], answer_key_points: list[str], prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> tuple[list[list[str]], dict[str, Any]]:
    """Classify each answer key point into its axes (answer / evidence / both).

    Returns (axes_by_index, raw_meta) where each entry is a list like ["evidence"],
    ["answer"], or ["evidence","answer"]. ["evidence"] is the safe default for any point
    the model fails to classify, because that forces trajectory verification. The return
    type is accepted directly by init_point_states."""
    points_block = "\n".join(f"{idx}. {point}" for idx, point in enumerate(answer_key_points))
    prompt = "\n\n".join(["## Task", task["question"], "", "## Answer Key Points (ordered, 0-indexed)", points_block])
    axes_by_point: list[list[str]] = [[AXIS_EVIDENCE] for _ in answer_key_points]
    raw_meta: dict[str, Any] = {"requested": len(answer_key_points)}
    try:
        parsed, raw_text, attempt = call_llm_json(
            client, args.model, prompts["defer_classify_points"], prompt,
            args.max_tokens, args.request_max_retries, args.request_retry_sleep,
            temperature=args.defer_temperature,
        )
        raw_meta.update({"raw_judge_response": raw_text, "request_attempt": attempt, "judge_prompt": prompt})
        for item in parsed.get("classifications") or []:
            if not isinstance(item, dict):
                continue
            try:
                pid = int(item.get("point_id"))
            except Exception:
                continue
            if 0 <= pid < len(axes_by_point):
                # Prefer the new `axes` field; fall back to legacy single `kind`.
                raw = item.get("axes") if item.get("axes") is not None else item.get("kind")
                axes_by_point[pid] = normalize_axes(raw)
    except Exception as exc:
        raw_meta["error"] = f"{type(exc).__name__}: {exc}"
    return axes_by_point, raw_meta


AXIS_ANSWER = "answer"
AXIS_EVIDENCE = "evidence"


def normalize_axes(value: Any) -> list[str]:
    """Coerce a point's classification into an ordered list of distinct axes.

    Accepts the legacy single-kind string ("answer_bearing"/"evidence_bearing"),
    an axis name ("answer"/"evidence"), or a list of either. Returns axes in a
    stable order with evidence first (it is verified during the walk, answer at
    finalize). Empty/unknown input defaults to ["evidence"] (the stronger, safer
    requirement: force trajectory verification)."""
    raw_items = value if isinstance(value, list) else [value]
    axes: list[str] = []
    for raw in raw_items:
        text = str(raw or "").strip().lower()
        if text in {"answer", "answer_bearing", "answer-bearing", "answerbearing"}:
            axis = AXIS_ANSWER
        elif text in {"evidence", "evidence_bearing", "evidence-bearing", "evidencebearing"}:
            axis = AXIS_EVIDENCE
        else:
            continue
        if axis not in axes:
            axes.append(axis)
    if not axes:
        axes = [AXIS_EVIDENCE]
    # Stable order: evidence before answer.
    return [a for a in (AXIS_EVIDENCE, AXIS_ANSWER) if a in axes]


def dominant_kind(axes: list[str]) -> str:
    """The single legacy `kind` value for a point: evidence dominates when present
    (it is the stronger requirement). Keeps operation route + stage projections intact."""
    return "evidence_bearing" if AXIS_EVIDENCE in axes else "answer_bearing"


def recompute_aggregate(state: dict[str, Any]) -> None:
    """Recompute a point's top-level aggregate fields from its per-axis sub-judgments.

    A point is satisfied only when every axis is satisfied; contradicted if any axis
    is contradicted; partial if some (but not all) axes are satisfied; else pending.
    Top-level supporting_step_ids / evidence_log / satisfied_at_step / last_note are the
    union/merge across axes so existing readers keep working."""
    axes = state.get("axes") or []
    statuses = [ax.get("status", "pending") for ax in axes]
    if statuses and all(s == "satisfied" for s in statuses):
        agg = "satisfied"
    elif any(s == "contradicted" for s in statuses):
        agg = "contradicted"
    elif any(s == "satisfied" for s in statuses):
        agg = "partial"
    else:
        agg = "pending"
    state["status"] = agg
    support: list[int] = []
    log: list[dict[str, Any]] = []
    sat_steps: list[int] = []
    last_note = ""
    for ax in axes:
        for sid in ax.get("supporting_step_ids", []):
            if sid not in support:
                support.append(sid)
        for entry in ax.get("evidence_log", []):
            log.append({**entry, "axis": ax.get("axis")})
        if ax.get("satisfied_at_step") is not None:
            sat_steps.append(ax["satisfied_at_step"])
        if ax.get("last_note"):
            last_note = ax["last_note"]
    state["supporting_step_ids"] = sorted(support)
    state["evidence_log"] = log
    state["satisfied_at_step"] = max(sat_steps) if sat_steps else None
    state["last_note"] = last_note
    # Derive satisfying_entities from every satisfied evidence_log entry across axes.
    # Dedup by normalize_url(url) if non-empty else lowercased(name); skip empty entities.
    sat_entities: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for ax in axes:
        for entry in ax.get("evidence_log", []):
            if entry.get("status") != "satisfied":
                continue
            ent = entry.get("entity")
            if not isinstance(ent, dict):
                continue
            url = str(ent.get("url") or "").strip()
            name = str(ent.get("name") or "").strip().lower()
            if not url and not name:
                continue
            key = normalize_url(url) if url else f"name::{name}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            sat_entities.append({
                "url": url,
                "title": str(ent.get("title") or "").strip(),
                "name": str(ent.get("name") or "").strip(),
            })
    state["satisfying_entities"] = sat_entities


def init_point_states(answer_key_points: list[str], kinds: list[Any]) -> list[dict[str, Any]]:
    """Build the per-point evaluation memory.

    `kinds[idx]` may be a legacy single string ("answer_bearing"/"evidence_bearing")
    or a list of axes (["evidence","answer"]) for dual-axis points. Each point carries
    one or two independent axis sub-judgments plus aggregate fields derived from them.
    Single-axis points are byte-identical in behaviour to the pre-multiaxis version."""
    states: list[dict[str, Any]] = []
    for idx, point in enumerate(answer_key_points):
        raw_kind = kinds[idx] if idx < len(kinds) else "evidence_bearing"
        axes_names = normalize_axes(raw_kind)
        axes = [
            {
                "axis": name,
                "status": "pending",
                "satisfied_at_step": None,
                "supporting_step_ids": [],
                "evidence_log": [],
                "last_note": "",
                # Informational only — the first entity an axis was satisfied on. The
                # authoritative cross-entity check uses the SET of entities across the
                # whole evidence_log (so attack-2 compare-then-pick doesn't false-flag).
                "satisfying_entity": None,
            }
            for name in axes_names
        ]
        # The `kind` field stays a single value. For answer/evidence classification it is
        # the dominant axis. For the operation route, `raw_kind` is a timing token
        # ("final"/"intermediate") that is NOT an answer/evidence axis — preserve it verbatim
        # so evaluate_operation_key_points keeps reading timing from state["kind"].
        kind_value = str(raw_kind) if (isinstance(raw_kind, str) and str(raw_kind).strip().lower() in {"final", "intermediate"}) else dominant_kind(axes_names)
        state = {
            "point_id": idx,
            "text": point,
            "kind": kind_value,
            "axes": axes,
            "status": "pending",
            "satisfied_at_step": None,
            "supporting_step_ids": [],
            "evidence_log": [],
            "last_note": "",
            # Pure projection of the per-axis evidence_log: every entity (deduped) that
            # ever satisfied any axis of this point. Drives the finalize same-entity check.
            "satisfying_entities": [],
        }
        recompute_aggregate(state)
        states.append(state)
    return states


def steps_equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two browser steps are equivalent (collapsible) if they land on the same URL with
    the same tool and carry no materially different observation text."""
    if get_step_kind(a) != "browser" or get_step_kind(b) != "browser":
        return False
    if get_step_tool_name(a) != get_step_tool_name(b):
        return False
    if str(a.get("current_url") or "").strip() != str(b.get("current_url") or "").strip():
        return False
    norm = lambda r: (str(r.get("som_text") or "").strip(), str(r.get("observation") or "").strip())
    return norm(a) == norm(b)


def browser_state_signature(record: dict[str, Any]) -> str:
    """Exact signature of a browser step's resulting page state. Two steps with the same
    signature land on an identical visible state and cannot yield new evidence."""
    return "".join([
        get_step_tool_name(record),
        str(record.get("current_url") or ""),
        str(record.get("current_title") or ""),
        str(record.get("som_text") or ""),
        str(record.get("observation") or ""),
    ])


def compress_navigation(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop redundant browser steps to cut LLM calls without losing information.

    Two passes, both lossless for evidence purposes:
    1. collapse consecutive equivalent steps;
    2. global dedup of already-seen exact browser page states (handles periodic
       click->scroll->go_back loops that revisit the same pages many times).
    Non-browser steps (search/scholar/visit/code_sandbox/answer) are never deduped —
    their textual output is the evidence and each call can differ. Keeps first
    occurrence of each state; returns (kept, collapsed_count)."""
    kept: list[dict[str, Any]] = []
    seen_browser_states: set[str] = set()
    collapsed = 0
    for record in records:
        if kept and steps_equivalent(kept[-1], record):
            collapsed += 1
            continue
        if get_step_kind(record) == "browser":
            signature = browser_state_signature(record)
            if signature in seen_browser_states:
                collapsed += 1
                continue
            seen_browser_states.add(signature)
        kept.append(record)
    return kept, collapsed


def render_memory_block(point_states: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for state in point_states:
        note = state.get("last_note") or ""
        support = ",".join(str(s) for s in state.get("supporting_step_ids") or [])
        axes = state.get("axes") or []
        axes_desc = " ".join(f"{ax['axis']}={ax['status']}" for ax in axes)
        dual = " [DUAL: must hold in BOTH the answer text AND the trajectory]" if len(axes) > 1 else ""
        # Surface the SET of entities that ever satisfied this point (across axes), so
        # the finalize LLM can compare them against the entity the final answer commits to.
        sat_ents = state.get("satisfying_entities") or []
        if sat_ents:
            def _label(e: dict[str, Any]) -> str:
                nm = (e.get("name") or "").strip()
                u = (e.get("url") or "").strip()
                if nm and u:
                    return f"{nm} <{u}>"
                return nm or u or "?"
            entity_str = " satisfying_entities=[" + " | ".join(_label(e) for e in sat_ents) + "]"
        else:
            entity_str = ""
        lines.append(
            f"[{state['point_id']}] kind={state['kind']} status={state['status']} "
            f"axes({axes_desc}) supporting=[{support}] note={note}{dual}{entity_str}"
            f"\n     point: {state['text']}"
        )
    return "\n".join(lines) if lines else "(no key points)"


def _update_axis(axis_state: dict[str, Any], item: dict[str, Any], step_id: int) -> None:
    """Apply one LLM per-point update to a single axis sub-judgment in place."""
    new_status = normalize_status(item.get("status"), axis_state["status"])
    new_evidence = item.get("new_evidence")
    changed = isinstance(new_evidence, dict) and str(new_evidence.get("note") or "").strip()
    # Never silently downgrade a satisfied axis to pending on a later silent step.
    if axis_state["status"] == "satisfied" and new_status == "pending" and not changed:
        new_status = "satisfied"
    entity = None
    if changed:
        note = str(new_evidence.get("note")).strip()
        relation = str(new_evidence.get("relation") or "").strip().lower() or "support"
        entity = _coerce_entity(new_evidence.get("entity"))
        axis_state["evidence_log"].append({
            "step": step_id,
            "status": new_status,
            "relation": relation,
            "note": note,
            "entity": entity,
        })
        axis_state["last_note"] = note
        if step_id not in axis_state["supporting_step_ids"]:
            axis_state["supporting_step_ids"].append(step_id)
    if new_status == "satisfied" and axis_state["status"] != "satisfied" and axis_state["satisfied_at_step"] is None:
        axis_state["satisfied_at_step"] = step_id
        # Informational pin: which entity this axis was FIRST satisfied on. The
        # cross-entity finalize check uses the full evidence_log set (not this pin),
        # so it remains correct even when an axis is later re-supported on Y after X.
        if entity is not None and axis_state.get("satisfying_entity") is None:
            axis_state["satisfying_entity"] = entity
    axis_state["status"] = new_status
    incoming_support = item.get("supporting_step_ids")
    if isinstance(incoming_support, list):
        for raw in incoming_support:
            try:
                sid = int(raw)
            except Exception:
                continue
            if sid not in axis_state["supporting_step_ids"]:
                axis_state["supporting_step_ids"].append(sid)
    axis_state["supporting_step_ids"] = sorted(axis_state["supporting_step_ids"])


def apply_step_update(point_states: list[dict[str, Any]], parsed: dict[str, Any], step_id: int, deferred_axes: frozenset[str] = frozenset({AXIS_ANSWER})) -> None:
    """Merge one step's LLM update into the evaluation memory in place, per axis.

    During the trajectory walk only the non-deferred axes are advanced. defer defers the
    `answer` axis (settled at finalize against the answer text) and advances the `evidence`
    axis here; operation passes deferred_axes=frozenset() so its single page-state axis
    advances live. A point with only deferred axes just records a note and stays pending."""
    by_id = {state["point_id"]: state for state in point_states}
    for item in parsed.get("points") or []:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("point_id"))
        except Exception:
            continue
        state = by_id.get(pid)
        if state is None:
            continue
        live_axes = [ax for ax in state.get("axes", []) if ax.get("axis") not in deferred_axes]
        if not live_axes:
            # All axes for this point are deferred to finalize; keep a note only.
            new_evidence = item.get("new_evidence")
            if isinstance(new_evidence, dict) and str(new_evidence.get("note") or "").strip():
                note = str(new_evidence.get("note")).strip()
                for ax in state.get("axes", []):
                    ax["last_note"] = note
            recompute_aggregate(state)
            continue
        for ax in live_axes:
            _update_axis(ax, item, step_id)
        recompute_aggregate(state)


def update_state_with_step(task: dict[str, Any], point_states: list[dict[str, Any]], step_record: dict[str, Any], notes: str, system_prompt: str, deferred_axes: frozenset[str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    """Feed one step + the current evaluation memory to `system_prompt`, merge the result.
    Shared by defer and operation; only the system prompt and deferred_axes differ."""
    step_id = int(step_record.get("step", -1))
    step_block_text, current_screenshot_path = build_step_block_text(task, step_record)
    prompt = "\n\n".join([
        "## Task", task["question"],
        "", "## Annotation Notes", notes or "(none)",
        "", "## Current Evaluation Memory", render_memory_block(point_states),
        "", "## New Trajectory Step", step_block_text,
    ])
    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if current_screenshot_path:
        user_content.append(build_image_content_item(current_screenshot_path))
    parsed, raw_text, attempt = call_llm_json(
        client, args.model, system_prompt, user_content,
        args.max_tokens, args.request_max_retries, args.request_retry_sleep,
        temperature=args.defer_temperature,
    )
    apply_step_update(point_states, parsed, step_id, deferred_axes)
    # Snapshot the memory after this step for step-by-step replay / visualization.
    return {
        "step_id": step_id,
        "current_screenshot_path": current_screenshot_path,
        "statuses": {state["point_id"]: state["status"] for state in point_states},
        "changed_points": [
            int(item.get("point_id"))
            for item in (parsed.get("points") or [])
            if isinstance(item, dict) and isinstance(item.get("new_evidence"), dict)
            and str(item["new_evidence"].get("note") or "").strip()
        ],
        "raw_judge_response": raw_text,
        "request_attempt": attempt,
        "request_temperature": args.defer_temperature,
    }


def walk_trajectory(task: dict[str, Any], point_states: list[dict[str, Any]], records: list[dict[str, Any]], notes: str, system_prompt: str, deferred_axes: frozenset[str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    """Generic stateful walk: compress redundant steps, then update the evaluation memory
    one step at a time. Returns the per-step snapshots and compression accounting.
    point_states is mutated in place. Shared by defer and operation routes."""
    compressed_records, collapsed_count = compress_navigation(records)
    step_updates: list[dict[str, Any]] = []
    for step_record in compressed_records:
        step_updates.append(
            update_state_with_step(task, point_states, step_record, notes, system_prompt, deferred_axes, client, args)
        )
    return {
        "step_updates": step_updates,
        "collapsed_steps": collapsed_count,
        "candidate_step_count": len(records),
        "evaluated_step_count": len(compressed_records),
    }


def finalize_defer_state(task: dict[str, Any], point_states: list[dict[str, Any]], notes: str, prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    prompt = "\n\n".join([
        "## Task", task["question"],
        "", "## Annotation Notes", notes or "(none)",
        "", "## Agent Final Answer", task["prediction"],
        "", "## Final Evaluation Memory", render_memory_block(point_states),
    ])
    parsed, raw_text, attempt = call_llm_json(
        client, args.model, prompts["defer_finalize"], prompt,
        args.max_tokens, args.request_max_retries, args.request_retry_sleep,
        temperature=args.defer_temperature,
    )
    by_id = {state["point_id"]: state for state in point_states}
    for item in parsed.get("points") or []:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("point_id"))
        except Exception:
            continue
        state = by_id.get(pid)
        if state is None:
            continue
        final_status = normalize_status(item.get("final_status"), state["status"])
        # finalize settles the ANSWER axis (verified against the final answer text). The
        # evidence axis keeps its walk verdict. For a point with no answer axis this is a
        # no-op on axes; the verdict can still flag the whole point via a consistency note.
        answer_axis = next((ax for ax in state.get("axes", []) if ax["axis"] == AXIS_ANSWER), None)
        if answer_axis is not None:
            if final_status != answer_axis["status"]:
                answer_axis["evidence_log"].append({"step": "final", "status": final_status, "relation": "finalize", "note": str(item.get("reason") or "")})
            answer_axis["status"] = final_status
        elif final_status == "contradicted":
            # Answer-only verdict can still contradict an evidence-only point (e.g. the
            # answer commits to a different entity than the trajectory supported).
            for ax in state.get("axes", []):
                if ax["status"] == "satisfied":
                    ax["evidence_log"].append({"step": "final", "status": "contradicted", "relation": "finalize", "note": str(item.get("reason") or "")})
                    ax["status"] = "contradicted"
        state["final_reason"] = str(item.get("reason") or "")
        recompute_aggregate(state)
    # ------------------------------------------------------------------
    # Same-entity post-pass: structural enforcement that every satisfied evidence
    # axis was satisfied on the entity the final answer commits to. The LLM is told
    # to do this in the prompt; we re-do it here in code so the verdict cannot rest
    # on LLM goodwill alone. Pure-Python, zero extra LLM calls.
    # ------------------------------------------------------------------
    answer_entity = _coerce_entity(parsed.get("answer_entity"))
    if answer_entity is None:
        # Fallback: try to resolve the answered entity from the prediction text by
        # token overlap against the union of satisfying_entities. Pick the entity
        # with the strictly highest non-zero overlap; tie / no overlap → leave None
        # and skip the post-pass (non-entity tasks: numbers, yes/no, lists).
        pred_tokens = _name_tokens(str(task.get("prediction") or ""))
        if pred_tokens:
            best: tuple[int, dict[str, Any] | None] = (0, None)
            seen_keys: set[str] = set()
            for state in point_states:
                for ent in state.get("satisfying_entities") or []:
                    key = (normalize_url(ent.get("url") or "")) or f"name::{(ent.get('name') or '').lower()}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    ov = len(_name_tokens(ent.get("name") or "") & pred_tokens)
                    if ov > best[0]:
                        best = (ov, ent)
            if best[1] is not None:
                answer_entity = {
                    "url": normalize_url(best[1].get("url") or ""),
                    "title": best[1].get("title") or "",
                    "name": best[1].get("name") or "",
                }

    entity_conflicts: list[dict[str, Any]] = []
    if answer_entity is not None:
        for state in point_states:
            for ax in state.get("axes", []):
                # Only the EVIDENCE axis is bound to a trajectory entity. The answer
                # axis is settled against the answer text itself (no entity provenance),
                # and operation route's timing axes (final/intermediate) are out of scope.
                if ax.get("axis") != AXIS_EVIDENCE:
                    continue
                if ax.get("status") != "satisfied":
                    continue
                # Collect entities of every supporting (non-contradicting) satisfied
                # log entry on this axis. Empty entries (entity is None) are tracked
                # separately so we can downgrade to PARTIAL when nothing is bindable.
                sat_ents: list[dict[str, Any]] = []
                had_unbound = False
                for entry in ax.get("evidence_log", []):
                    if entry.get("status") != "satisfied":
                        continue
                    if entry.get("relation") == "contradict":
                        continue
                    ent = entry.get("entity")
                    if not isinstance(ent, dict) or not (ent.get("url") or ent.get("name")):
                        had_unbound = True
                        continue
                    sat_ents.append(ent)

                if any(_same_entity(answer_entity, e) for e in sat_ents):
                    # D3a — match. Force satisfied and veto any LLM contradiction.
                    if ax["status"] != "satisfied":
                        ax["status"] = "satisfied"
                    continue
                if sat_ents and all(not _same_entity(answer_entity, e) for e in sat_ents):
                    # D3b — clearly different on every satisfying entry. Contradict.
                    ax["status"] = "contradicted"
                    ax["evidence_log"].append({
                        "step": "final",
                        "status": "contradicted",
                        "relation": "entity_mismatch",
                        "note": f"axis satisfied only on {[e.get('name') or e.get('url') for e in sat_ents]}, "
                                f"answer commits to {answer_entity.get('name') or answer_entity.get('url')}",
                        "entity": answer_entity,
                    })
                    entity_conflicts.append({
                        "point_id": state["point_id"],
                        "axis": ax.get("axis"),
                        "expected_entity": answer_entity,
                        "found_entities": sat_ents,
                    })
                elif had_unbound and not sat_ents:
                    # D3c — every satisfying entry had no usable entity tag. Abstain
                    # by downgrading to partial: do not silently pass a bare textual
                    # claim, but do not fabricate a contradiction either.
                    ax["status"] = "partial"
                    ax["evidence_log"].append({
                        "step": "final",
                        "status": "partial",
                        "relation": "entity_unbound",
                        "note": "axis satisfied without any entity tag; cannot confirm it was the answered entity",
                        "entity": None,
                    })
            recompute_aggregate(state)

    solved = coerce_bool(parsed.get("solved"))
    truthful = coerce_bool(parsed.get("truthful"))
    # Derive label from the settled memory, per axis, so it cannot disagree with the
    # per-axis statuses. solved = every answer axis satisfied; truthful = every evidence axis.
    answer_axes = [ax for s in point_states for ax in s.get("axes", []) if ax["axis"] == AXIS_ANSWER]
    evidence_axes = [ax for s in point_states for ax in s.get("axes", []) if ax["axis"] == AXIS_EVIDENCE]
    derived_solved = all(ax["status"] == "satisfied" for ax in answer_axes) if answer_axes else True
    derived_truthful = all(ax["status"] == "satisfied" for ax in evidence_axes) if evidence_axes else True
    solved = solved and derived_solved
    truthful = truthful and derived_truthful
    predicted_label = 1 if (solved and truthful) else 0
    return {
        "predicted_label": predicted_label,
        "solved": solved,
        "truthful": truthful,
        "reason": str(parsed.get("reason") or ""),
        "judge_prompt": prompt,
        "raw_judge_response": raw_text,
        "request_attempt": attempt,
        "request_temperature": args.defer_temperature,
        # Entity-binding audit fields (additive, ignored by old readers).
        "answer_entity": answer_entity,
        "entity_conflicts": entity_conflicts,
    }


def _summarize_points(point_states: list[dict[str, Any]], kind: str, ok: bool) -> str:
    """Human-readable one-liner for the compat stage reason fields, counted per axis.

    `kind` is "answer_bearing" or "evidence_bearing"; we report on the matching axis so
    a dual-axis point appears in both the answer and the evidence summary."""
    axis = AXIS_ANSWER if kind == "answer_bearing" else AXIS_EVIDENCE
    pairs = [(s, ax) for s in point_states for ax in s.get("axes", []) if ax["axis"] == axis]
    if not pairs:
        return "No answer-bearing key points." if kind == "answer_bearing" else "No evidence-bearing key points."
    failing = [(s, ax) for s, ax in pairs if ax["status"] != "satisfied"]
    if ok and not failing:
        return f"All {len(pairs)} {kind} requirement(s) satisfied."
    parts = [f"[{s['point_id']}/{axis}] {ax['status']}" for s, ax in failing]
    return f"Unsatisfied {kind} requirement(s): " + "; ".join(parts)


def evaluate_defer_framework(task: dict[str, Any], annotation: AnnotationRecord | None, classification: dict[str, Any] | None, prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    """Stateful defer judge: classify key points, then read the trajectory step by step
    maintaining an evaluation memory of each point's status, then settle against the final
    answer. Replaces the former three-stage (solved / relevance / truthfulness) flow.
    See STATEFUL_DEFER_DESIGN.md.
    """
    if not task["prediction"]:
        return make_missing_result(task, "defer_framework", "no_prediction", "Agent did not provide a final answer.", annotation, classification)

    answer_key_points = get_defer_answer_key_points(annotation, classification)
    notes = annotation.notes if annotation else ""

    # Phase A: classify points into answer_bearing / evidence_bearing.
    # First try the module-level _POINT_CLASSIFICATION_CACHE populated in main() from
    # --point-classification. On a clean length+text match we skip the LLM call
    # entirely; otherwise we fall back to live classification.
    if answer_key_points:
        cache_entry = _POINT_CLASSIFICATION_CACHE.get(task.get("task_id") or "")
        cache_hit = False
        cache_mismatch_reason = ""
        if cache_entry and isinstance(cache_entry, dict):
            cached_kinds = cache_entry.get("kinds") or []
            cached_akp = cache_entry.get("answer_key_points") or []
            if len(cached_kinds) != len(answer_key_points):
                cache_mismatch_reason = "length"
                sys.stderr.write(
                    f"[point-classification] task={task.get('task_id')!r} cache miss: "
                    f"cached kinds={len(cached_kinds)} vs answer_key_points={len(answer_key_points)}\n"
                )
            elif cached_akp and len(cached_akp) == len(answer_key_points) and any(
                str(cached_akp[i]).strip() != str(answer_key_points[i]).strip()
                for i in range(len(answer_key_points))
            ):
                cache_mismatch_reason = "text"
                sys.stderr.write(
                    f"[point-classification] task={task.get('task_id')!r} cache miss: "
                    f"key-point text drift between cache and annotation\n"
                )
            else:
                cache_hit = True
        if cache_hit:
            # Prefer per-point axes (dual-axis aware) when the cache carries them;
            # fall back to the legacy single `kinds` list for older caches.
            cached_axes = cache_entry.get("axes_by_point")
            init_kinds = cached_axes if cached_axes else list(cache_entry["kinds"])
            classify_meta = {
                "requested": len(answer_key_points),
                "source": "cache",
                "cache_model": cache_entry.get("model", ""),
                "cache_generated_at": cache_entry.get("generated_at", ""),
                "cache_prompt_path": cache_entry.get("prompt_path", ""),
                "classifications": cache_entry.get("classifications", []),
                "raw_judge_response": cache_entry.get("raw_response", ""),
                "judge_prompt": cache_entry.get("user_prompt", ""),
                "request_attempt": cache_entry.get("request_attempt", 0),
            }
        else:
            kinds, classify_meta = classify_answer_points(task, answer_key_points, prompts, client, args)
            init_kinds = kinds
            if not isinstance(classify_meta, dict):
                classify_meta = {"requested": len(answer_key_points)}
            classify_meta["source"] = "live"
            if cache_mismatch_reason:
                classify_meta["cache_mismatch_reason"] = cache_mismatch_reason
    else:
        init_kinds, classify_meta = [], {"requested": 0, "source": "n/a", "note": "no answer key points provided; finalize judges task directly"}
    point_states = init_point_states(answer_key_points, init_kinds)

    # Phase B+C: walk the trajectory (pre-answer steps), updating memory one step at a time.
    # answer_bearing points stay pending here; they are settled at finalize against the answer.
    candidate_records = get_pre_answer_records(task["records"])
    walk = walk_trajectory(
        task, point_states, candidate_records, notes,
        prompts["defer_update_state"], frozenset({AXIS_ANSWER}), client, args,
    )
    step_updates = walk["step_updates"]
    collapsed_count = walk["collapsed_steps"]

    # Phase D: settle answer_bearing points + overall verdict against the final answer.
    verdict = finalize_defer_state(task, point_states, notes, prompts, client, args)

    selected_step_ids = sorted({sid for state in point_states for sid in state["supporting_step_ids"]})
    return {
        "benchmark_id": task["benchmark_id"],
        "mode": task["mode"],
        "task_id": task["task_id"],
        "question": task["question"],
        "evaluation_type": "defer_framework",
        "evaluation_status": "completed",
        "predicted_label": verdict["predicted_label"],
        "reason": verdict["reason"],
        "stateful": True,
        # Evaluation memory: per-point status, supporting steps, and the running evidence log.
        "point_states": point_states,
        "classification_of_points": classify_meta,
        "step_updates": step_updates,
        "collapsed_steps": collapsed_count,
        "candidate_step_count": walk["candidate_step_count"],
        "evaluated_step_count": walk["evaluated_step_count"],
        # Backward-compatible projection onto the old stage shape so existing readers
        # (e.g. trajectory_viewer/experiment_compare_server.py) keep working unchanged.
        "stage1": {
            "solved": verdict["solved"],
            "reason": _summarize_points(point_states, "answer_bearing", verdict["solved"]),
            "answer_key_points": answer_key_points,
        },
        "stage2": {
            "answer_key_points": answer_key_points,
            "candidate_step_ids": [int(record.get("step", -1)) for record in candidate_records],
            "selected_step_ids": selected_step_ids,
        },
        "stage3": {
            "truthful": verdict["truthful"],
            "reason": verdict["reason"],
            "answer_key_points": answer_key_points,
        },
        "annotation_type_source": annotation_type_source(annotation, classification),
        "annotation": annotation.__dict__ if annotation else None,
        "classification": classification,
        "finalize_judge_prompt": verdict["judge_prompt"],
        "finalize_raw_judge_response": verdict["raw_judge_response"],
        "request_temperature": args.defer_temperature,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def describe_end_state(task: dict[str, Any]) -> str:
    """Best-effort description of the end-of-trajectory browser state, for final-state
    recheck of `final`-timed page states. Uses the last non-answer browser step."""
    records = task.get("records") or []
    for record in reversed(records):
        if get_step_kind(record) == "answer":
            continue
        if get_step_kind(record) == "browser":
            text = compose_observation_text(record)
            step = record.get("step", "")
            return f"(from step {step})\n{text}" if text else f"(from step {step}) [no observation]"
    # No browser step; fall back to the last record's observation.
    if records:
        return compose_observation_text(records[-1]) or "[no end-of-trajectory browser state available]"
    return "[empty trajectory]"


def finalize_operation_state(task: dict[str, Any], point_states: list[dict[str, Any]], notes: str, prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    """Settle the verdict: intermediate states keep their walk result; final states are
    rechecked against the end-of-trajectory state. Label derived from settled memory."""
    end_state_text = describe_end_state(task)
    prompt = "\n\n".join([
        "## Task", task["question"],
        "", "## Annotation Notes", notes or "(none)",
        "", "## Final Evaluation Memory", render_memory_block(point_states),
        "", "## End-of-Trajectory Page State", end_state_text,
    ])
    parsed, raw_text, attempt = call_llm_json(
        client, args.model, prompts["operation_finalize"], prompt,
        args.max_tokens, args.request_max_retries, args.request_retry_sleep,
        temperature=args.defer_temperature,
    )
    by_id = {state["point_id"]: state for state in point_states}
    for item in parsed.get("points") or []:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("point_id"))
        except Exception:
            continue
        state = by_id.get(pid)
        if state is None:
            continue
        final_status = normalize_status(item.get("final_status"), state["status"])
        if final_status != state["status"]:
            state["evidence_log"].append({"step": "final", "status": final_status, "relation": "finalize", "note": str(item.get("reason") or "")})
        state["status"] = final_status
        state["final_reason"] = str(item.get("reason") or "")
    # Derive label from settled memory so it cannot disagree with the per-point statuses.
    derived_achieved = all(s["status"] == "satisfied" for s in point_states) if point_states else False
    achieved = coerce_bool(parsed.get("achieved")) and derived_achieved
    return {
        "predicted_label": 1 if achieved else 0,
        "achieved": achieved,
        "reason": str(parsed.get("reason") or ""),
        "end_state_text": end_state_text,
        "judge_prompt": prompt,
        "raw_judge_response": raw_text,
        "request_attempt": attempt,
        "request_temperature": args.defer_temperature,
    }


def evaluate_operation_key_points(task: dict[str, Any], annotation: AnnotationRecord, classification: dict[str, Any] | None, prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    """Stateful operation judge: track each required page state across the trajectory,
    respecting timing (intermediate = passed through once; final = holds at the end).
    Shares the per-step engine with defer_framework. See STATEFUL_OPERATION_DESIGN.md."""
    if not annotation.operation_key_points:
        return make_missing_result(task, "operation_key_points", "missing_annotation_fields", "Missing page_states.", annotation, classification)

    notes = annotation.notes if annotation else ""
    page_states = annotation.page_states or [{"timing": annotation.page_state_timing, "state": state} for state in annotation.operation_key_points]
    points = [page_state["state"] for page_state in page_states]
    # The page state's timing IS its kind. Default unspecified timing to "final" (stricter:
    # must hold at the end) to match the legacy "achieved at end" reading.
    kinds = [(str(page_state.get("timing") or "").strip().lower() or "final") for page_state in page_states]
    point_states = init_point_states(points, kinds)

    # Walk the full trajectory (operation cares about page states throughout, including
    # the final answer step's browser state). No kinds are deferred — all update live.
    walk = walk_trajectory(
        task, point_states, task["records"], notes,
        prompts["operation_update_state"], frozenset(), client, args,
    )

    # Finalize: recheck final-timed states against the end-of-trajectory state.
    verdict = finalize_operation_state(task, point_states, notes, prompts, client, args)

    # Backward-compatible projection onto the old key_point_results shape.
    key_point_results = [
        {
            "key_point": state["text"],
            "page_state": state["text"],
            "timing": state["kind"],
            "achieved": state["status"] == "satisfied",
            "status": state["status"],
            "supporting_step_ids": state["supporting_step_ids"],
            "reason": state.get("final_reason") or state.get("last_note") or "",
        }
        for state in point_states
    ]
    return {
        "benchmark_id": task["benchmark_id"],
        "mode": task["mode"],
        "task_id": task["task_id"],
        "question": task["question"],
        "evaluation_type": "operation_key_points",
        "evaluation_status": "completed",
        "predicted_label": verdict["predicted_label"],
        "reason": verdict["reason"] or ("All required page states were achieved." if verdict["achieved"] else "At least one required page state was not achieved."),
        "stateful": True,
        # Evaluation memory: per page-state status, supporting steps, and evidence log.
        "page_state_results": point_states,
        "step_updates": walk["step_updates"],
        "collapsed_steps": walk["collapsed_steps"],
        "candidate_step_count": walk["candidate_step_count"],
        "evaluated_step_count": walk["evaluated_step_count"],
        # Backward-compatible field for existing readers.
        "key_point_results": key_point_results,
        "annotation_type_source": annotation_type_source(annotation, classification),
        "annotation": annotation.__dict__,
        "classification": classification,
        "finalize_judge_prompt": verdict["judge_prompt"],
        "finalize_raw_judge_response": verdict["raw_judge_response"],
        "request_temperature": args.defer_temperature,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def evaluate_by_type(effective_type: str, task: dict[str, Any], annotation: AnnotationRecord | None, classification: dict[str, Any] | None, prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    empty_annotation = AnnotationRecord(task["task_id"], "", [], [], "", "", [], "")
    if effective_type == "static_answer":
        return evaluate_static_answer(task, annotation or empty_annotation, classification, prompts, client, args)
    if effective_type == "target_urls":
        return evaluate_target_urls(task, annotation or empty_annotation, classification, prompts, client, args)
    if effective_type == "reveal_script":
        return evaluate_reveal_script(task, annotation or empty_annotation, classification, prompts, client, args)
    if effective_type == "defer_framework":
        return evaluate_defer_framework(task, annotation, classification, prompts, client, args)
    if effective_type == "operation_key_points":
        return evaluate_operation_key_points(task, annotation or empty_annotation, classification, prompts, client, args)
    return make_missing_result(task, effective_type, "unsupported_type", f"Unsupported evaluation type: {effective_type}", annotation, classification)


def combine_evaluation_results(task: dict[str, Any], effective_types: list[str], sub_results: list[dict[str, Any]], annotation: AnnotationRecord | None, classification: dict[str, Any] | None) -> dict[str, Any]:
    labels = [item.get("predicted_label") for item in sub_results]
    if labels and all(label == 1 for label in labels):
        predicted_label: int | None = 1
        reason = "All required evaluation dimensions passed."
    elif any(label == 0 for label in labels):
        predicted_label = 0
        failed = [
            f"{item.get('evaluation_type')}: {item.get('reason') or item.get('evaluation_status')}"
            for item in sub_results
            if item.get("predicted_label") == 0
        ]
        reason = " | ".join(failed) or "At least one required evaluation dimension failed."
    else:
        predicted_label = None
        reason = "At least one required evaluation dimension did not produce a usable judgment."

    if all(item.get("evaluation_status") == "completed" for item in sub_results):
        evaluation_status = "completed"
    elif any(item.get("evaluation_status") == "exception" for item in sub_results):
        evaluation_status = "exception"
    else:
        evaluation_status = "partial"

    return {
        "benchmark_id": task["benchmark_id"],
        "mode": task["mode"],
        "task_id": task["task_id"],
        "question": task["question"],
        "evaluation_type": "combined",
        "evaluation_types": effective_types,
        "evaluation_status": evaluation_status,
        "predicted_label": predicted_label,
        "reason": reason,
        "sub_results": sub_results,
        "annotation_type_source": annotation_type_source(annotation, classification),
        "annotation": annotation.__dict__ if annotation else None,
        "classification": classification,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def apply_task_revision(task: dict[str, Any], annotation: AnnotationRecord | None) -> dict[str, Any]:
    """Return a task whose question reflects the annotator's revision, preserving the original.

    The annotation taxonomy lets annotators rewrite a task's question (task_revision)
    when the original prompt cannot be fairly evaluated. The reference answers / key
    points are written against the revised question, so the judge must see the revised
    question. We keep the agent's original question for traceability and tag the row so
    revised tasks can be inspected or excluded during aggregation.
    """
    if not (annotation and annotation.revision_enabled and annotation.revised_question):
        return task
    revised = dict(task)
    revised["original_question"] = task.get("question", "")
    revised["question"] = annotation.revised_question
    revised["task_revised"] = True
    revised["task_revision_reason"] = annotation.revision_reason
    return revised


def tag_revision(row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    if task.get("task_revised"):
        row["task_revised"] = True
        row["original_question"] = task.get("original_question", "")
        row["task_revision_reason"] = task.get("task_revision_reason", "")
    else:
        row.setdefault("task_revised", False)
    return row


def evaluate_one(task: dict[str, Any], annotations: dict[str, AnnotationRecord], classifications: dict[str, dict[str, Any]], prompts: dict[str, str], client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    annotation = annotations.get(task["task_id"])
    classification = classifications.get(task["task_id"])
    # Evaluate against the annotator-revised question when present; reference answers
    # and key points were authored against it. Original question is kept for traceability.
    task = apply_task_revision(task, annotation)
    effective_types: list[str] = []
    try:
        if annotation and annotation.annotation_types:
            effective_types = annotation.annotation_types
        elif annotation and annotation.annotation_type:
            effective_types = [annotation.annotation_type]
        elif classification and isinstance(classification.get("validation_type"), str):
            classification_type = normalize_annotation_type(classification.get("validation_type"))
            if classification_type:
                effective_types = [classification_type]

        if effective_types == ["UA"]:
            return tag_revision(make_unannotatable_result(task, annotation, classification), task)
        if "UA" in effective_types:
            return tag_revision(make_missing_result(task, "+".join(effective_types), "invalid_annotation_type", "UA must be used alone and cannot be combined with other evaluation types.", annotation, classification), task)

        invalid_types = [item for item in effective_types if item not in VALIDATION_TYPES]
        if not effective_types or invalid_types:
            effective_type = "+".join(effective_types) if effective_types else "unknown"
            reason = "No usable annotation_type or classification." if not effective_types else f"Unsupported evaluation type(s): {', '.join(invalid_types)}"
            return tag_revision(make_missing_result(task, effective_type, "missing_annotation_type", reason, annotation, classification), task)

        if len(effective_types) == 1:
            return tag_revision(evaluate_by_type(effective_types[0], task, annotation, classification, prompts, client, args), task)

        sub_results = [
            evaluate_by_type(effective_type, task, annotation, classification, prompts, client, args)
            for effective_type in effective_types
        ]
        return tag_revision(combine_evaluation_results(task, effective_types, sub_results, annotation, classification), task)
    except Exception as exc:
        return tag_revision(make_error_result(task, "+".join(effective_types) if effective_types else "unknown", exc, annotation, classification), task)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def result_key(row: dict[str, Any]) -> tuple[str, str, int]:
    repeat_index = row.get("repeat_index", 1)
    try:
        normalized_repeat_index = int(repeat_index)
    except Exception:
        normalized_repeat_index = 1
    return str(row.get("task_id", "")), str(row.get("mode", "")), normalized_repeat_index


def expand_repeated_tasks(tasks: list[dict[str, Any]], repeat_count: int) -> list[dict[str, Any]]:
    normalized_repeat_count = max(1, int(repeat_count))
    if normalized_repeat_count == 1:
        for task in tasks:
            task["repeat_index"] = 1
            task["repeat_count"] = 1
        return tasks

    repeated: list[dict[str, Any]] = []
    for repeat_index in range(1, normalized_repeat_count + 1):
        for task in tasks:
            cloned = dict(task)
            cloned["repeat_index"] = repeat_index
            cloned["repeat_count"] = normalized_repeat_count
            repeated.append(cloned)
    return repeated


def build_summary(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    overall = {
        "total": len(rows),
        "completed": 0,
        "failed": 0,
        "unannotatable": 0,
        "success": 0,
        "failure": 0,
        "per_status": {},
    }
    per_mode: dict[str, dict[str, Any]] = {}
    per_type: dict[str, dict[str, Any]] = {}
    # Tasks whose question was rewritten by the annotator (task_revision). Tracked
    # separately so aggregation can inspect or exclude them; the agent ran against the
    # original question, so these rows judge the original trajectory under the new prompt.
    revised = {"total": 0, "completed": 0, "success": 0, "failure": 0, "failed": 0, "unannotatable": 0}
    for row in rows:
        status = row.get("evaluation_status", "")
        mode = str(row.get("mode", ""))
        eval_type = str(row.get("evaluation_type", ""))
        predicted = row.get("predicted_label")
        overall["per_status"][status] = int(overall["per_status"].get(status, 0)) + 1
        if status == "completed":
            overall["completed"] += 1
            if predicted == 1:
                overall["success"] += 1
            elif predicted == 0:
                overall["failure"] += 1
        elif status == "unannotatable":
            overall["unannotatable"] += 1
        else:
            overall["failed"] += 1

        if row.get("task_revised"):
            revised["total"] += 1
            if status == "completed":
                revised["completed"] += 1
                if predicted == 1:
                    revised["success"] += 1
                elif predicted == 0:
                    revised["failure"] += 1
            elif status == "unannotatable":
                revised["unannotatable"] += 1
            else:
                revised["failed"] += 1

        mode_stats = per_mode.setdefault(mode, {"total": 0, "completed": 0, "failed": 0, "unannotatable": 0, "success": 0, "failure": 0, "per_status": {}})
        mode_stats["total"] += 1
        mode_stats["per_status"][status] = int(mode_stats["per_status"].get(status, 0)) + 1
        if status == "completed":
            mode_stats["completed"] += 1
            if predicted == 1:
                mode_stats["success"] += 1
            elif predicted == 0:
                mode_stats["failure"] += 1
        elif status == "unannotatable":
            mode_stats["unannotatable"] += 1
        else:
            mode_stats["failed"] += 1

        type_stats = per_type.setdefault(eval_type, {"total": 0, "completed": 0, "failed": 0, "unannotatable": 0, "success": 0, "failure": 0, "per_status": {}})
        type_stats["total"] += 1
        type_stats["per_status"][status] = int(type_stats["per_status"].get(status, 0)) + 1
        if status == "completed":
            type_stats["completed"] += 1
            if predicted == 1:
                type_stats["success"] += 1
            elif predicted == 0:
                type_stats["failure"] += 1
        elif status == "unannotatable":
            type_stats["unannotatable"] += 1
        else:
            type_stats["failed"] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(args.run_dir),
        "annotations": str(args.annotations),
        "classification": str(args.classification),
        "classification_dataset_file": args.classification_dataset_file,
        "answer_key_points": str(args.answer_key_points) if args.answer_key_points else "",
        "model": args.model,
        "defer_temperature": args.defer_temperature,
        "repeat_count": max(1, int(args.repeat_count)),
        "browser_snapshot_command": args.browser_snapshot_command,
        "jina_reader_base_url": args.jina_reader_base_url,
        "webpage_fetch_timeout": args.webpage_fetch_timeout,
        "webpage_screenshot_timeout": args.webpage_screenshot_timeout,
        "webpage_screenshot_wait_ms": args.webpage_screenshot_wait_ms,
        "webpage_screenshot_viewport_width": args.webpage_screenshot_viewport_width,
        "webpage_screenshot_viewport_height": args.webpage_screenshot_viewport_height,
        "webpage_screenshot_browser_executable": args.webpage_screenshot_browser_executable or find_browser_executable(args),
        "output_dir": str(output_dir),
        "overall": overall,
        "revised_tasks": revised,
        "per_mode": per_mode,
        "per_type": per_type,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_results_path = output_dir / "eval_results.jsonl"
    prompts = {name: load_text(path) for name, path in PROMPT_PATHS.items()}
    client = build_client(args)
    annotations = load_annotations(args.annotations)
    classifications = load_classifications(args.classification, args.classification_dataset_file)
    answer_key_points_by_task = load_answer_key_points(args.answer_key_points)
    for task_id, answer_key_points in answer_key_points_by_task.items():
        classification = classifications.setdefault(task_id, {"task_id": task_id})
        classification["answer_key_points"] = answer_key_points
    point_classification_by_task = load_point_classification(args.point_classification)
    # Inject via the module-level registry so evaluate_defer_framework can read the
    # cache without changing its (already-long) function signature, and without
    # leaking the bulky cache entries into the per-row `classification` payload that
    # gets serialised into eval_results.jsonl.
    _POINT_CLASSIFICATION_CACHE.clear()
    _POINT_CLASSIFICATION_CACHE.update(point_classification_by_task)
    task_filter = set(args.task_id) if args.task_id else None
    modes = detect_modes(args.run_dir, args.modes)
    tasks = discover_tasks(args.run_dir, modes, task_filter, args.limit)
    tasks = expand_repeated_tasks(tasks, args.repeat_count)
    existing_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if args.resume_existing and eval_results_path.exists():
        existing_rows = {
            result_key(row): row
            for row in jsonl_load(eval_results_path)
            if row.get("task_id") and row.get("mode")
        }
        tasks = [task for task in tasks if result_key(task) not in existing_rows]
    else:
        eval_results_path.write_text("", encoding="utf-8")

    manifest = {
        "run_dir": str(args.run_dir),
        "annotations": str(args.annotations),
        "classification": str(args.classification),
        "classification_dataset_file": args.classification_dataset_file,
        "answer_key_points": str(args.answer_key_points) if args.answer_key_points else "",
        "answer_key_point_task_count": len(answer_key_points_by_task),
        "point_classification": str(args.point_classification) if args.point_classification else "",
        "point_classification_task_count": len(point_classification_by_task),
        "model": args.model,
        "defer_temperature": args.defer_temperature,
        "modes": modes,
        "repeat_count": max(1, int(args.repeat_count)),
        "task_count": len(tasks) + len(existing_rows),
        "pending_task_count": len(tasks),
        "existing_task_count": len(existing_rows),
        "resume_existing": bool(args.resume_existing),
        "streaming_write": True,
        "jina_reader_base_url": args.jina_reader_base_url,
        "webpage_fetch_timeout": args.webpage_fetch_timeout,
        "webpage_screenshot_timeout": args.webpage_screenshot_timeout,
        "webpage_screenshot_wait_ms": args.webpage_screenshot_wait_ms,
        "webpage_screenshot_viewport_width": args.webpage_screenshot_viewport_width,
        "webpage_screenshot_viewport_height": args.webpage_screenshot_viewport_height,
        "webpage_screenshot_browser_executable": args.webpage_screenshot_browser_executable or find_browser_executable(args),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = list(existing_rows.values())
    if existing_rows:
        rows.sort(key=lambda row: (str(row.get("mode", "")), str(row.get("task_id", ""))))
        write_jsonl(eval_results_path, rows)
        partial_summary = build_summary(rows, output_dir, args)
        (output_dir / "summary.json").write_text(
            json.dumps(partial_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_to_task = {
            executor.submit(evaluate_one, task, annotations, classifications, prompts, client, args): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                row = future.result()
            except Exception as exc:
                row = make_error_result(task, "unknown", exc, annotations.get(task["task_id"]), classifications.get(task["task_id"]))
            row["repeat_index"] = int(task.get("repeat_index") or 1)
            row["repeat_count"] = int(task.get("repeat_count") or 1)
            rows.append(row)
            append_jsonl_row(eval_results_path, row)
            partial_summary = build_summary(rows, output_dir, args)
            (output_dir / "summary.json").write_text(
                json.dumps(partial_summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    rows.sort(key=lambda row: (int(row.get("repeat_index") or 1), str(row.get("mode", "")), str(row.get("task_id", ""))))
    write_jsonl(eval_results_path, rows)
    summary = build_summary(rows, output_dir, args)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved evaluation results to {output_dir} (tasks={len(tasks)}, completed={summary['overall']['completed']}, failed={summary['overall']['failed']})")


if __name__ == "__main__":
    main()
