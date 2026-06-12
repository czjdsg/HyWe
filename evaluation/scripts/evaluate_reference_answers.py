#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = SCRIPT_DIR.parent
DEFAULT_PROMPT_PATH = FRAMEWORK_ROOT / "prompts" / "judge_reference_answer.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate agent answers against embedded reference answers across multiple benchmark run dirs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Run directory containing search/operation/hybrid subdirs. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retry-failed-from", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-retry-sleep", type=float, default=2.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows_map: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    ordered_rows = sorted(
        rows_map.values(),
        key=lambda row: (
            str(row.get("benchmark_id", "")),
            str(row.get("mode", "")),
            str(row.get("task_id", "")),
        ),
    )
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("benchmark_id", "")),
        str(row.get("mode", "")),
        str(row.get("task_id", "")),
    )


def detect_modes(run_dir: Path, requested_modes: list[str] | None) -> list[str]:
    preferred = requested_modes or ["search", "operation", "hybrid"]
    modes = [mode for mode in preferred if (run_dir / mode).is_dir()]
    return modes or [run_dir.name]


def get_mode_dir(run_dir: Path, mode: str) -> Path:
    candidate = run_dir / mode
    return candidate if candidate.is_dir() else run_dir


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def extract_reference_answer(result: dict[str, Any]) -> str:
    raw = result.get("raw") or {}
    raw_raw = raw.get("raw") or {}
    extra = raw.get("extra") or {}
    candidates = [
        extra.get("reference_answer"),
        raw.get("answer"),
        raw_raw.get("Final answer"),
        raw_raw.get("answer"),
        result.get("answer"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return ""


def extract_question(result: dict[str, Any]) -> str:
    raw = result.get("raw") or {}
    raw_raw = raw.get("raw") or {}
    candidates = [
        result.get("question"),
        raw.get("question"),
        raw_raw.get("Question"),
        raw_raw.get("question"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return ""


def extract_dataset(result: dict[str, Any], run_dir: Path) -> str:
    raw = result.get("raw") or {}
    for candidate in [raw.get("dataset"), raw.get("domain"), run_dir.name]:
        if candidate:
            return str(candidate)
    return run_dir.name


def discover_tasks(run_dir: Path, requested_modes: list[str] | None) -> list[dict[str, Any]]:
    benchmark_id = run_dir.name
    work_items: list[dict[str, Any]] = []
    for mode in detect_modes(run_dir, requested_modes):
        mode_dir = get_mode_dir(run_dir, mode)
        for result_path in sorted(mode_dir.glob("*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            task_id = str(result.get("task_id") or result_path.parent.name)
            prediction = str(result.get("prediction") or "").strip()
            work_items.append(
                {
                    "benchmark_id": benchmark_id,
                    "run_dir": str(run_dir),
                    "mode": mode,
                    "task_id": task_id,
                    "result_path": str(result_path),
                    "question": extract_question(result),
                    "reference_answer": extract_reference_answer(result),
                    "prediction": prediction,
                    "termination": result.get("termination", ""),
                    "dataset": extract_dataset(result, run_dir),
                    "split": str((result.get("raw") or {}).get("split", "")),
                }
            )
    return work_items


def build_task_prompt(task: dict[str, Any]) -> str:
    lines = [
        "## Benchmark",
        f"Benchmark ID: {task.get('benchmark_id', '')}",
        f"Dataset: {task.get('dataset', '')}",
        f"Mode: {task.get('mode', '')}",
        "",
        "## Question",
        str(task.get("question", "")),
        "",
        "## Reference Answer",
        str(task.get("reference_answer", "")),
        "",
        "## Agent Output",
        f"Termination: {task.get('termination', '')}",
        str(task.get("prediction", "")),
    ]
    return "\n".join(lines)


def parse_judgment(response_text: str) -> dict[str, Any]:
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    payload = match.group(1) if match else response_text.strip()
    data = json.loads(payload)

    predicted_label = data.get("predicted_label")
    if predicted_label not in (0, 1):
        correct = str(data.get("correct", "")).strip().lower()
        if correct in {"yes", "no"}:
            predicted_label = 1 if correct == "yes" else 0
            data["predicted_label"] = predicted_label
    if data.get("predicted_label") not in (0, 1):
        raise ValueError("predicted_label must be 0 or 1")

    if "extracted_final_answer" not in data:
        data["extracted_final_answer"] = None
    if "reasoning" not in data:
        data["reasoning"] = ""
    return data


def evaluate_one(
    task: dict[str, Any],
    system_prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = dict(task)
    reference_answer = str(task.get("reference_answer", "")).strip()
    prediction = str(task.get("prediction", "")).strip()

    if not reference_answer:
        row.update(
            {
                "evaluation_status": "missing_reference_answer",
                "predicted_label": None,
                "reasoning": "Reference answer is missing.",
                "extracted_final_answer": None,
            }
        )
        return row

    if not prediction:
        row.update(
            {
                "evaluation_status": "no_prediction",
                "predicted_label": 0,
                "reasoning": "Agent did not provide a final answer.",
                "extracted_final_answer": None,
            }
        )
        return row

    prompt = build_task_prompt(task)
    row["judge_prompt"] = prompt

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    last_error = ""
    for attempt in range(1, max(1, args.request_max_retries) + 1):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=args.max_tokens,
            )
            response_text = completion.choices[0].message.content or ""
            parsed = parse_judgment(response_text)
            row.update(parsed)
            row["raw_judge_response"] = response_text
            row["evaluation_status"] = "completed"
            row["request_attempt"] = attempt
            return row
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            row["request_attempt"] = attempt
            if attempt < max(1, args.request_max_retries):
                time.sleep(max(0.0, args.request_retry_sleep) * attempt)

    row.update(
        {
            "evaluation_status": "evaluation_failed",
            "predicted_label": None,
            "reasoning": "",
            "extracted_final_answer": None,
            "error": last_error,
        }
    )
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("predicted_label") in (0, 1)]
    success = sum(1 for row in completed if row.get("predicted_label") == 1)
    failure = sum(1 for row in completed if row.get("predicted_label") == 0)
    return {
        "total": len(rows),
        "completed": len(completed),
        "success": success,
        "failure": failure,
        "evaluation_failed": sum(1 for row in rows if row.get("evaluation_status") == "evaluation_failed"),
        "missing_reference_answer": sum(
            1 for row in rows if row.get("evaluation_status") == "missing_reference_answer"
        ),
        "no_prediction": sum(1 for row in rows if row.get("evaluation_status") == "no_prediction"),
        "judged_by_model": sum(1 for row in rows if row.get("evaluation_status") == "completed"),
        "success_rate_completed_only": (success / len(completed)) if completed else 0.0,
        "success_rate_all": (success / len(rows)) if rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_prompt(args.prompt_path)
    work_items: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        work_items.extend(discover_tasks(run_dir, args.modes))

    existing_rows_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    retry_keys: set[tuple[str, str, str]] | None = None
    if args.retry_failed_from:
        previous_rows = load_jsonl(args.retry_failed_from)
        existing_rows_map = {row_key(row): row for row in previous_rows if row.get("task_id")}
        retry_keys = {
            row_key(row)
            for row in previous_rows
            if row.get("evaluation_status") == "evaluation_failed"
        }

    manifest = {
        "run_dirs": [str(path) for path in args.run_dir],
        "output_dir": str(args.output_dir),
        "retry_failed_from": str(args.retry_failed_from) if args.retry_failed_from else None,
        "prompt_path": str(args.prompt_path),
        "base_url": args.base_url,
        "model": args.model,
        "work_item_count": len(work_items),
        "request_max_retries": args.request_max_retries,
        "request_retry_sleep": args.request_retry_sleep,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows_map: dict[tuple[str, str, str], dict[str, Any]] = dict(existing_rows_map)
    submit_items = []
    for task in work_items:
        key = row_key(task)
        if retry_keys is not None and key not in retry_keys:
            continue
        submit_items.append(task)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(evaluate_one, task, system_prompt, args)
            for task in submit_items
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows_map[row_key(row)] = row
            write_jsonl(args.output_dir / "eval_results.jsonl", rows_map)

    final_rows = sorted(
        rows_map.values(),
        key=lambda row: (
            str(row.get("benchmark_id", "")),
            str(row.get("mode", "")),
            str(row.get("task_id", "")),
        ),
    )

    benchmark_ids = sorted({str(row.get("benchmark_id", "")) for row in final_rows})
    modes = sorted({str(row.get("mode", "")) for row in final_rows})

    summary = {
        "overall": summarize_rows(final_rows),
        "by_benchmark": [],
        "by_mode": [],
        "by_benchmark_mode": [],
    }

    for benchmark_id in benchmark_ids:
        subset = [row for row in final_rows if row.get("benchmark_id") == benchmark_id]
        item = {"benchmark_id": benchmark_id, **summarize_rows(subset)}
        datasets = sorted({str(row.get("dataset", "")) for row in subset if row.get("dataset")})
        if datasets:
            item["datasets"] = datasets
        summary["by_benchmark"].append(item)

    for mode in modes:
        subset = [row for row in final_rows if row.get("mode") == mode]
        summary["by_mode"].append({"mode": mode, **summarize_rows(subset)})

    for benchmark_id in benchmark_ids:
        for mode in modes:
            subset = [
                row
                for row in final_rows
                if row.get("benchmark_id") == benchmark_id and row.get("mode") == mode
            ]
            if not subset:
                continue
            summary["by_benchmark_mode"].append(
                {"benchmark_id": benchmark_id, "mode": mode, **summarize_rows(subset)}
            )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
