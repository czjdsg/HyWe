#!/usr/bin/env python3
"""Preprocess defer_framework answer-key-point classifications into a cache JSONL.

Reads annotations.json, keeps only entries whose annotation_types include
``defer_framework`` and whose ``answer_key_points`` (or legacy ``answer_points``)
list is non-empty, and runs the same Phase A classification the evaluator runs
inline. The result is one JSONL row per task, written to a stable cache file
the evaluator can later load via ``--point-classification`` to skip the per-task
classification LLM call entirely.

Reuses helpers from ``evaluate_classified_web_tasks.py`` so the cached path and
the live path are byte-identical (same prompt template, same client builder,
same JSON-extraction logic, same kind normalisation).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = FRAMEWORK_ROOT / "outputs"

# Make sure the evaluator module is importable regardless of how this script is
# launched (`python scripts/classify_defer_answer_points.py` vs module form).
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_classified_web_tasks import (  # noqa: E402  (sys.path setup above)
    PROMPT_PATHS,
    AnnotationRecord,
    apply_task_revision,
    extract_json_object as _eval_extract_json_object,
    jsonl_load,
    load_annotations,
    load_text,
    normalize_kind as _eval_normalize_kind,
    normalize_text_list,
)

DEFAULT_PROMPT_PATH = PROMPT_PATHS["defer_classify_points"]
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "kimi-k2.5")
PROMPT_NAME = "defer_classify_points_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-classify defer_framework answer key points into answer_bearing / "
            "evidence_bearing and cache the result as JSONL so the evaluator can skip "
            "the Phase A LLM call."
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to annotations.json (same file consumed by evaluate_classified_web_tasks.py).",
    )
    parser.add_argument(
        "--dataset-file",
        required=True,
        help=(
            "Benchmark dataset filename recorded in each output row (e.g. "
            "web_insta_v3_test50_nohuman_seed42.jsonl). Also used to derive the "
            "default output stem."
        ),
    )
    parser.add_argument(
        "--input-tasks",
        type=Path,
        default=None,
        help=(
            "Optional JSONL of tasks providing canonical 'question' text. When "
            "omitted, the question is read from the annotation entry's 'question' "
            "field; when missing there too, the row is skipped with "
            "status='skipped_no_question'."
        ),
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the classification system prompt. Defaults to prompts/defer_classify_points.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Explicit output JSONL path. Defaults to "
            "outputs/<dataset_stem>.point_classification.<model_slug>.jsonl."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON summary file. Defaults to <output>.summary.json.",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id used for classification. The slug derived from this distinguishes per-model cache files.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Load existing rows from --output and skip task_ids whose row has "
            "status='ok'. Failed rows are re-attempted and overwritten."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "When set without --resume, allow truncation of an existing output file. "
            "If the output exists and neither --resume nor --overwrite is given, the "
            "script exits to prevent silent data loss."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N tasks after annotation filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help="Repeatable. Process only these task_ids.",
    )
    return parser.parse_args()


def load_prompt(path: Path) -> str:
    """Read the defer_classify_points.txt system prompt."""
    return load_text(path)


def load_input_questions(path: Path | None) -> dict[str, str]:
    """Optional source of canonical task questions from a JSONL of benchmark tasks."""
    if path is None:
        return {}
    if not path.exists():
        return {}
    rows = jsonl_load(path)
    mapping: dict[str, str] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        question = str(row.get("question") or "").strip()
        if question:
            mapping[task_id] = question
    return mapping


def load_annotation_defer_tasks(
    annotations_path: Path,
    dataset_file: str,
    task_id_filter: set[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Load defer_framework tasks (with non-empty answer key points) from annotations.json.

    Returns a list of dicts of the shape:
        {task_id, answer_key_points, annotation_types, annotation, dataset_file}
    The ``annotation`` value is the populated ``AnnotationRecord`` so callers can
    apply task revision later. The ``question`` is intentionally NOT populated here:
    annotations.json has no question field; the caller pairs the task_id with
    --input-tasks or falls back to the (usually empty) revised_question.
    """
    annotations = load_annotations(annotations_path)
    items: list[dict[str, Any]] = []
    for task_id, annotation in annotations.items():
        if task_id_filter is not None and task_id not in task_id_filter:
            continue
        if "defer_framework" not in annotation.annotation_types:
            continue
        if not annotation.answer_key_points:
            continue
        items.append(
            {
                "task_id": task_id,
                "answer_key_points": list(annotation.answer_key_points),
                "annotation_types": list(annotation.annotation_types),
                "annotation": annotation,
                "dataset_file": dataset_file,
            }
        )
    items.sort(key=lambda row: str(row.get("task_id", "")))
    if limit and limit > 0:
        items = items[:limit]
    return items


def resolve_question(
    annotation: AnnotationRecord,
    input_question: str,
) -> str:
    """Return the question the classifier should see for this task.

    Mirrors evaluate_classified_web_tasks.apply_task_revision: when the annotator
    enabled a task_revision and supplied a revised_question, that overrides the
    benchmark question; otherwise the input-tasks question is used as-is. We
    build a minimal task dict and call apply_task_revision so the live path and
    cached path produce byte-identical prompts.
    """
    task: dict[str, Any] = {"question": input_question or ""}
    revised = apply_task_revision(task, annotation)
    return str(revised.get("question") or "").strip()


def build_user_prompt(task_id: str, question: str, answer_key_points: list[str]) -> str:
    """Build the exact same user prompt classify_answer_points constructs in the evaluator.

    The evaluator uses:
        "\\n\\n".join(["## Task", task["question"], "",
                       "## Answer Key Points (ordered, 0-indexed)", points_block])
    The ``task_id`` is unused by the prompt itself but is kept in the signature
    for symmetry with the contract.
    """
    _ = task_id  # explicitly unused; the prompt does not name the task id.
    points_block = "\n".join(
        f"{idx}. {point}" for idx, point in enumerate(answer_key_points)
    )
    return "\n\n".join(
        [
            "## Task",
            question,
            "",
            "## Answer Key Points (ordered, 0-indexed)",
            points_block,
        ]
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """JSON extraction helper. Delegates to the evaluator's implementation."""
    return _eval_extract_json_object(text)


def normalize_kind(value: Any) -> str:
    """Same normalisation used in the evaluator: '' if unrecognised."""
    return _eval_normalize_kind(value)


def build_record(
    task_id: str,
    question: str,
    dataset_file: str,
    annotation_types: list[str],
    answer_key_points: list[str],
    kinds: list[str],
    classifications: list[dict[str, Any]],
    model: str,
    prompt_path: Path,
    user_prompt: str,
    raw_response: str,
    request_attempt: int,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    """Build a single cache JSONL row matching the agreed contract schema."""
    return {
        "task_id": task_id,
        "question": question,
        "dataset_file": dataset_file,
        "annotation_types": list(annotation_types),
        "answer_key_points": list(answer_key_points),
        "kinds": list(kinds),
        "classifications": list(classifications),
        "model": model,
        "prompt_path": str(prompt_path),
        "prompt_name": PROMPT_NAME,
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "request_attempt": request_attempt,
        "status": status,
        "error": error,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def classify_one_task(
    client: OpenAI,
    item: dict[str, Any],
    system_prompt: str,
    prompt_path: Path,
    dataset_file: str,
    model: str,
    max_tokens: int,
    max_retries: int,
    request_retry_sleep: float,
) -> dict[str, Any]:
    """Run one LLM call and emit a cache row.

    Failure modes:
    - status='skipped_no_question': the resolved question is empty (no
      input-tasks entry for this task_id and annotation has no question/revision).
    - status='error': LLM call failed every retry or response could not be parsed.
    - status='ok': classifications align with the input key points.
    """
    task_id = str(item["task_id"])
    answer_key_points: list[str] = list(item["answer_key_points"])
    annotation_types: list[str] = list(item.get("annotation_types") or [])
    question: str = str(item.get("question") or "").strip()

    if not question:
        return build_record(
            task_id=task_id,
            question="",
            dataset_file=dataset_file,
            annotation_types=annotation_types,
            answer_key_points=answer_key_points,
            kinds=[],
            classifications=[],
            model=model,
            prompt_path=prompt_path,
            user_prompt="",
            raw_response="",
            request_attempt=0,
            status="skipped_no_question",
            error="No question available from annotation or --input-tasks.",
        )

    user_prompt = build_user_prompt(task_id, question, answer_key_points)
    kinds: list[str] = ["evidence_bearing"] * len(answer_key_points)
    classifications_out: list[dict[str, Any]] = [
        {
            "point_id": idx,
            "text": text,
            "kind": "evidence_bearing",
            "reason": "",
        }
        for idx, text in enumerate(answer_key_points)
    ]

    last_error = ""
    raw_text = ""
    attempt_used = 0
    for attempt in range(1, max(1, max_retries) + 1):
        attempt_used = attempt
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            raw_text = completion.choices[0].message.content or ""
            parsed = extract_json_object(raw_text)
            # Map model output back onto the input key-point indices.
            for item_obj in parsed.get("classifications") or []:
                if not isinstance(item_obj, dict):
                    continue
                try:
                    pid = int(item_obj.get("point_id"))
                except Exception:
                    continue
                if not (0 <= pid < len(kinds)):
                    continue
                kind = normalize_kind(item_obj.get("kind"))
                if kind:
                    kinds[pid] = kind
                    classifications_out[pid]["kind"] = kind
                reason = str(item_obj.get("reason") or "").strip()
                if reason:
                    classifications_out[pid]["reason"] = reason
            last_error = ""
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max(1, max_retries):
                time.sleep(max(0.0, request_retry_sleep) * attempt)
                continue

    status = "ok" if not last_error else "error"
    return build_record(
        task_id=task_id,
        question=question,
        dataset_file=dataset_file,
        annotation_types=annotation_types,
        answer_key_points=answer_key_points,
        kinds=kinds,
        classifications=classifications_out,
        model=model,
        prompt_path=prompt_path,
        user_prompt=user_prompt,
        raw_response=raw_text,
        request_attempt=attempt_used,
        status=status,
        error=last_error,
    )


def load_existing_cache_for_resume(path: Path) -> dict[str, dict[str, Any]]:
    """Read prior cache rows; returns task_id -> row for status=='ok' rows only."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("status") != "ok":
                continue
            task_id = row.get("task_id")
            if isinstance(task_id, str) and task_id:
                out[task_id] = row
    return out


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the cache via tmp-file + rename so partial runs cannot corrupt prior output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def derive_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Derive (output_jsonl, summary_json) per the outputs/<stem>.point_classification.<slug>.jsonl convention."""
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.model).strip("-") or "model"
    if args.output is None:
        stem = Path(args.dataset_file).stem
        output_path = DEFAULT_OUTPUT_DIR / f"{stem}.point_classification.{model_slug}.jsonl"
    else:
        output_path = args.output
    if args.summary_output is None:
        summary_path = output_path.with_suffix(".summary.json")
    else:
        summary_path = args.summary_output
    return output_path, summary_path


def build_client(api_key: str, base_url: str, request_timeout: float) -> OpenAI:
    """Build an OpenAI-compatible client. Matches the evaluator's build_client behaviour."""
    if not api_key:
        raise SystemExit("Missing --api-key or OPENAI_API_KEY")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if request_timeout and request_timeout > 0:
        kwargs["timeout"] = request_timeout
    return OpenAI(**kwargs)


def build_summary(
    rows: list[dict[str, Any]],
    output_path: Path,
    args: argparse.Namespace,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    """Summarise status counts, kind distribution and wall-clock for the run."""
    status_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    points_total = 0
    for row in rows:
        status_counts[row.get("status", "")] = status_counts.get(row.get("status", ""), 0) + 1
        for kind in row.get("kinds") or []:
            kind_counts[str(kind)] = kind_counts.get(str(kind), 0) + 1
            points_total += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotations": str(args.annotations),
        "dataset_file": args.dataset_file,
        "input_tasks": str(args.input_tasks) if args.input_tasks else "",
        "model": args.model,
        "prompt_path": str(args.prompt_path),
        "output_path": str(output_path),
        "row_count": len(rows),
        "status_counts": status_counts,
        "kind_counts": kind_counts,
        "points_total": points_total,
        "wall_clock_seconds": round(wall_clock_seconds, 3),
        "workers": args.workers,
    }


def main() -> None:
    args = parse_args()

    output_path, summary_path = derive_output_paths(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not (args.resume or args.overwrite):
        raise SystemExit(
            f"Output {output_path} already exists. Pass --resume to keep ok rows "
            f"and re-run failed tasks, or --overwrite to truncate."
        )

    system_prompt = load_prompt(args.prompt_path)
    client = build_client(args.api_key, args.base_url, args.request_timeout)

    task_filter = set(args.task_id) if args.task_id else None
    annotation_items = load_annotation_defer_tasks(
        args.annotations, args.dataset_file, task_filter, args.limit
    )

    input_questions = load_input_questions(args.input_tasks)

    # Stitch the question into each item via apply_task_revision so revised
    # questions take precedence over input-tasks questions.
    for item in annotation_items:
        annotation: AnnotationRecord = item["annotation"]
        input_q = input_questions.get(item["task_id"], "")
        item["question"] = resolve_question(annotation, input_q)

    existing_ok: dict[str, dict[str, Any]] = {}
    if args.resume:
        existing_ok = load_existing_cache_for_resume(output_path)

    pending = [it for it in annotation_items if it["task_id"] not in existing_ok]
    rows_by_task: dict[str, dict[str, Any]] = dict(existing_ok)
    lock = threading.Lock()
    start_time = time.time()

    def flush() -> None:
        ordered = sorted(rows_by_task.values(), key=lambda r: str(r.get("task_id", "")))
        write_jsonl_atomic(output_path, ordered)

    if not pending and not existing_ok:
        # Nothing to do; still emit empty file + summary so downstream is deterministic.
        flush()
        summary = build_summary([], output_path, args, time.time() - start_time)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"No defer_framework tasks with answer key points found. "
            f"Wrote empty cache to {output_path}",
            file=sys.stderr,
        )
        return

    # Persist existing rows immediately so a crash mid-run doesn't lose --resume state.
    if existing_ok:
        flush()

    completed = 0
    total = len(pending)
    print(
        f"Classifying {total} defer_framework tasks (resumed {len(existing_ok)}). "
        f"Model={args.model}, workers={args.workers}, output={output_path}",
        file=sys.stderr,
    )

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_to_task = {
                executor.submit(
                    classify_one_task,
                    client,
                    item,
                    system_prompt,
                    args.prompt_path,
                    args.dataset_file,
                    args.model,
                    args.max_tokens,
                    max(1, args.request_max_retries),
                    max(0.0, args.request_retry_sleep),
                ): item
                for item in pending
            }
            for future in as_completed(future_to_task):
                item = future_to_task[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = build_record(
                        task_id=str(item["task_id"]),
                        question=str(item.get("question") or ""),
                        dataset_file=args.dataset_file,
                        annotation_types=list(item.get("annotation_types") or []),
                        answer_key_points=list(item.get("answer_key_points") or []),
                        kinds=[],
                        classifications=[],
                        model=args.model,
                        prompt_path=args.prompt_path,
                        user_prompt="",
                        raw_response="",
                        request_attempt=0,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                with lock:
                    rows_by_task[row["task_id"]] = row
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(
                            f"  [{completed}/{total}] last={row['task_id']} "
                            f"status={row.get('status')}",
                            file=sys.stderr,
                        )
                    flush()

    summary = build_summary(
        list(rows_by_task.values()), output_path, args, time.time() - start_time
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Saved point-classification cache to {output_path} "
        f"(rows={summary['row_count']}, status={summary['status_counts']}, "
        f"kinds={summary['kind_counts']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
