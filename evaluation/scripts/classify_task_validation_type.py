#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = SCRIPT_DIR.parent
DEFAULT_PROMPT_PATH = FRAMEWORK_ROOT / "prompts" / "classify_validation_type.txt"
DEFAULT_OUTPUT_DIR = FRAMEWORK_ROOT / "outputs"
DEFAULT_INPUTS = [
    Path("/data/czj/HybridWebAgentBenchmarks/data_v2/web_online_mind2web_256_nohuman_final.jsonl"),
    Path("/data/czj/HybridWebAgentBenchmarks/data_v2/web_webvoyager_643_final.jsonl"),
    Path("/data/czj/HybridWebAgentBenchmarks/data_samples_v2/web_insta_v3_test50_nohuman_seed42.jsonl"),
]
VALIDATION_TYPES = {"static_answer", "target_urls", "defer_framework", "operation_key_points"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify web benchmark tasks into static-answer, target-urls, "
            "trajectory-evidence-verification, or operation-key-points validation types with an OpenAI-compatible model."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Input JSONL task file. Repeatable. Defaults to the three no-human benchmark files.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "kimi-k2.5"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-human-verification",
        action="store_true",
        help="Do not filter out needs_human_verification=true rows.",
    )
    return parser.parse_args()


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def load_tasks(path: Path, include_human_verification: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            if not include_human_verification and bool(obj.get("needs_human_verification")):
                continue
            rows.append(obj)
    return rows


def build_client(api_key: str, base_url: str) -> OpenAI:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def derive_output_paths(args: argparse.Namespace, input_paths: list[Path]) -> tuple[Path, Path]:
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.model).strip("-") or "model"
    if args.output is None:
        if len(input_paths) == 1:
            stem = input_paths[0].stem
        else:
            stem = "multi_benchmark_nohuman"
        output_path = DEFAULT_OUTPUT_DIR / f"{stem}.validation_type_classification.{model_slug}.jsonl"
    else:
        output_path = args.output
    if args.summary_output is None:
        summary_path = output_path.with_suffix(".summary.json")
    else:
        summary_path = args.summary_output
    return output_path, summary_path


def build_user_prompt(item: dict[str, Any], source_path: Path) -> str:
    extra = item.get("extra") or {}
    dataset = str(item.get("dataset") or source_path.stem).strip()
    question = str(item.get("question") or "").strip()
    website = str(
        extra.get("original_website")
        or item.get("web")
        or item.get("domain")
        or item.get("subdomain")
        or ""
    ).strip()
    criteria = normalize_text_list(extra.get("criteria"))

    lines = [
        f"Source dataset file: {source_path.name}",
        f"Dataset: {dataset}",
        f"Task ID: {item.get('task_id', '')}",
        f"Question: {question}",
        f"Needs human verification: {bool(item.get('needs_human_verification', False))}",
    ]
    if website:
        lines.append(f"Target website/domain: {website}")
    if criteria:
        lines.append("Native criteria:")
        lines.extend(f"- {criterion}" for criterion in criteria)
    return "\n".join(lines)


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
            raise ValueError("no JSON object found in response")
        raw = match.group(1).strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("response JSON is not an object")
    return obj


def normalize_result_object(obj: dict[str, Any]) -> dict[str, str]:
    validation_type = str(obj.get("validation_type") or "").strip()
    confidence = str(obj.get("confidence") or "").strip().lower()
    reason = str(obj.get("reason") or "").strip()
    annotation_focus = str(obj.get("annotation_focus") or "").strip()

    if validation_type not in VALIDATION_TYPES:
        raise ValueError(f"invalid validation_type: {validation_type!r}")
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"invalid confidence: {confidence!r}")
    if not reason:
        raise ValueError("missing reason")
    if not annotation_focus:
        raise ValueError("missing annotation_focus")

    return {
        "validation_type": validation_type,
        "confidence": confidence,
        "reason": reason,
        "annotation_focus": annotation_focus,
    }


def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def load_existing_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = (str(obj.get("dataset_file", "")), str(obj.get("task_id", "")))
            if key[0] and key[1]:
                existing[key] = obj
    return existing


def build_record(
    item: dict[str, Any],
    source_path: Path,
    model: str,
    prompt_path: Path,
    user_prompt: str,
    raw_response: str,
    parsed: dict[str, str],
    attempt: int,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    extra = item.get("extra") or {}
    website = str(
        extra.get("original_website")
        or item.get("web")
        or item.get("domain")
        or item.get("subdomain")
        or ""
    ).strip()
    return {
        "dataset_file": source_path.name,
        "dataset": item.get("dataset") or source_path.stem,
        "task_id": item.get("task_id"),
        "question": item.get("question"),
        "website": website,
        "needs_human_verification": bool(item.get("needs_human_verification", False)),
        "criteria": normalize_text_list(extra.get("criteria")),
        "model": model,
        "prompt_path": str(prompt_path),
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "validation_type": parsed.get("validation_type", ""),
        "confidence": parsed.get("confidence", ""),
        "reason": parsed.get("reason", ""),
        "annotation_focus": parsed.get("annotation_focus", ""),
        "attempt": attempt,
        "status": status,
        "error": error,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def classify_one(
    client: OpenAI,
    item: dict[str, Any],
    source_path: Path,
    model: str,
    system_prompt: str,
    prompt_path: Path,
    max_tokens: int,
    max_retries: int,
    request_retry_sleep: float,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(item, source_path)
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_model(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            parsed = normalize_result_object(extract_json_object(raw_response))
            return build_record(
                item=item,
                source_path=source_path,
                model=model,
                prompt_path=prompt_path,
                user_prompt=user_prompt,
                raw_response=raw_response,
                parsed=parsed,
                attempt=attempt,
                status="completed",
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(request_retry_sleep * attempt)
                continue

    return build_record(
        item=item,
        source_path=source_path,
        model=model,
        prompt_path=prompt_path,
        user_prompt=user_prompt,
        raw_response="",
        parsed={},
        attempt=max_retries,
        status="failed",
        error=last_error,
    )


def write_records(path: Path, rows_map: dict[tuple[str, str], dict[str, Any]]) -> None:
    ordered_rows = sorted(
        rows_map.values(),
        key=lambda row: (
            str(row.get("dataset_file", "")),
            str(row.get("task_id", "")),
        ),
    )
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(
    records: list[dict[str, Any]],
    input_paths: list[Path],
    output_path: Path,
    model: str,
    prompt_path: Path,
) -> dict[str, Any]:
    per_dataset: dict[str, dict[str, Any]] = {}
    total_counts = {
        "completed": 0,
        "failed": 0,
        "static_answer": 0,
        "target_urls": 0,
        "defer_framework": 0,
        "operation_key_points": 0,
    }
    for record in records:
        dataset_file = str(record.get("dataset_file", ""))
        stats = per_dataset.setdefault(
            dataset_file,
            {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "static_answer": 0,
                "target_urls": 0,
                "defer_framework": 0,
                "operation_key_points": 0,
            },
        )
        stats["total"] += 1
        total_counts["completed" if record.get("status") == "completed" else "failed"] += 1
        stats["completed" if record.get("status") == "completed" else "failed"] += 1
        validation_type = str(record.get("validation_type", ""))
        if validation_type in VALIDATION_TYPES:
            stats[validation_type] += 1
            total_counts[validation_type] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "prompt_path": str(prompt_path),
        "inputs": [str(path) for path in input_paths],
        "output_path": str(output_path),
        "overall": {
            "total": len(records),
            **total_counts,
        },
        "per_dataset": per_dataset,
    }


def main() -> None:
    args = parse_args()
    input_paths = args.input or DEFAULT_INPUTS
    if not args.api_key:
        raise SystemExit("Missing --api-key or OPENAI_API_KEY")

    output_path, summary_path = derive_output_paths(args, input_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    system_prompt = load_prompt(args.prompt_path)
    client = build_client(args.api_key, args.base_url)

    existing = {} if args.overwrite else load_existing_records(output_path)

    work_items: list[tuple[Path, dict[str, Any]]] = []
    for input_path in input_paths:
        rows = load_tasks(input_path, include_human_verification=args.include_human_verification)
        if args.limit > 0:
            rows = rows[: args.limit]
        for item in rows:
            key = (input_path.name, str(item.get("task_id", "")))
            if key in existing and existing[key].get("status") == "completed":
                continue
            work_items.append((input_path, item))

    records_map = dict(existing)
    lock = threading.Lock()

    def save_record(record: dict[str, Any]) -> None:
        key = (str(record.get("dataset_file", "")), str(record.get("task_id", "")))
        with lock:
            records_map[key] = record
            write_records(output_path, records_map)

    if work_items:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    classify_one,
                    client,
                    item,
                    source_path,
                    args.model,
                    system_prompt,
                    args.prompt_path,
                    args.max_tokens,
                    max(1, args.max_retries),
                    max(0.0, args.request_retry_sleep),
                )
                for source_path, item in work_items
            ]
            for future in as_completed(futures):
                save_record(future.result())
    else:
        write_records(output_path, records_map)

    summary = build_summary(
        records=list(records_map.values()),
        input_paths=input_paths,
        output_path=output_path,
        model=args.model,
        prompt_path=args.prompt_path,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = summary["overall"]
    print(
        "Saved classifications to",
        output_path,
        f"(total={overall['total']}, completed={overall['completed']}, failed={overall['failed']}, "
        f"static_answer={overall['static_answer']}, target_urls={overall['target_urls']}, "
        f"defer_framework={overall['defer_framework']}, "
        f"operation_key_points={overall['operation_key_points']})",
    )


if __name__ == "__main__":
    main()
