#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = SCRIPT_DIR.parent
DEFAULT_PROMPT_PATH = FRAMEWORK_ROOT / "prompts" / "judge_final_answer_only.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final answers only against rewritten answer-only key points."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--keypoints-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retry-failed-from", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=900)
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


def detect_modes(run_dir: Path) -> list[str]:
    modes = []
    for mode in ["search", "operation", "hybrid"]:
        if (run_dir / mode).is_dir():
            modes.append(mode)
    return modes or [run_dir.name]


def get_mode_dir(run_dir: Path, mode: str) -> Path:
    candidate = run_dir / mode
    return candidate if candidate.is_dir() else run_dir


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_system_prompt(prompt_path: Path) -> str:
    return load_prompt(prompt_path)


def build_task_prompt(task: dict[str, Any], result: dict[str, Any]) -> str:
    key_points = task.get("answer_key_points") or []
    website = task.get("website") or ""
    lines = [
        "## Task",
        f"Instruction: {task.get('question', '')}",
        f"Website: {website}",
        "",
        "## Answer-Only Key Points",
        json.dumps(key_points, ensure_ascii=False, indent=2),
        "",
        "## Agent Output",
        f"Termination: {result.get('termination', '')}",
        f"Prediction: {result.get('prediction', '')}",
    ]
    return "\n".join(lines)


def parse_judgment(response_text: str) -> dict[str, Any]:
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    payload = match.group(1) if match else response_text.strip()
    data = json.loads(payload)
    predicted_label = data.get("predicted_label")
    if predicted_label not in (0, 1):
        raise ValueError("predicted_label must be 0 or 1")
    return data


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("task_id", "")), str(row.get("mode", ""))


def write_rows(path: Path, rows_map: dict[tuple[str, str], dict[str, Any]]) -> None:
    ordered_rows = sorted(
        rows_map.values(),
        key=lambda row: (str(row.get("mode", "")), str(row.get("task_id", ""))),
    )
    with path.open("w", encoding="utf-8") as handle:
        for item in ordered_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def evaluate_one(
    task: dict[str, Any],
    mode: str,
    mode_dir: Path,
    system_prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = task["task_id"]
    result_path = mode_dir / task_id / "result.json"
    row: dict[str, Any] = {
        "task_id": task_id,
        "mode": mode,
        "result_path": str(result_path),
        "question": task.get("question", ""),
        "website": task.get("website", ""),
        "answer_key_points": task.get("answer_key_points", []),
        "rewrites": task.get("rewrites", []),
        "dropped_criteria": task.get("dropped_criteria", []),
    }

    if not result_path.exists():
        row.update(
            {
                "evaluation_status": "missing_result",
                "predicted_label": None,
            }
        )
        return row

    result = json.loads(result_path.read_text(encoding="utf-8"))
    row["termination"] = result.get("termination")
    row["prediction"] = result.get("prediction", "")

    prompt = build_task_prompt(task, result)
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
            row["evaluation_status"] = parsed.get("evaluation_status", "completed")
            row["request_attempt"] = attempt
            return row
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            row["request_attempt"] = attempt
            if attempt < max(1, args.request_max_retries):
                import time
                time.sleep(max(0.0, args.request_retry_sleep) * attempt)

    row.update(
        {
            "evaluation_status": "evaluation_failed",
            "predicted_label": None,
            "error": last_error,
        }
    )
    return row


def summarize(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    mode_rows = [row for row in rows if row["mode"] == mode]
    completed = [row for row in mode_rows if row.get("predicted_label") in (0, 1)]
    success_count = sum(1 for row in completed if row.get("predicted_label") == 1)
    failure_count = sum(1 for row in completed if row.get("predicted_label") == 0)
    return {
        "mode": mode,
        "total": len(mode_rows),
        "completed": len(completed),
        "success": success_count,
        "failure": failure_count,
        "missing_result": sum(1 for row in mode_rows if row.get("evaluation_status") == "missing_result"),
        "evaluation_failed": sum(1 for row in mode_rows if row.get("evaluation_status") == "evaluation_failed"),
        "success_rate_completed_only": (success_count / len(completed)) if completed else 0.0,
        "success_rate_all": (success_count / len(mode_rows)) if mode_rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(args.keypoints_path)
    tasks = [
        task for task in tasks
        if isinstance(task.get("task_id"), str)
        and isinstance(task.get("question"), str)
        and isinstance(task.get("answer_key_points"), list)
    ]

    modes = detect_modes(args.run_dir)
    system_prompt = build_system_prompt(args.prompt_path)
    task_by_id = {str(task["task_id"]): task for task in tasks}
    existing_rows_map: dict[tuple[str, str], dict[str, Any]] = {}
    retry_keys: set[tuple[str, str]] | None = None

    if args.retry_failed_from:
        previous_rows = load_jsonl(args.retry_failed_from)
        existing_rows_map = {row_key(row): row for row in previous_rows if row.get("task_id") and row.get("mode")}
        retry_keys = {
            row_key(row)
            for row in previous_rows
            if row.get("evaluation_status") == "evaluation_failed"
        }

    manifest = {
        "dataset": "Insta-v3",
        "run_dir": str(args.run_dir),
        "keypoints_path": str(args.keypoints_path),
        "output_dir": str(args.output_dir),
        "retry_failed_from": str(args.retry_failed_from) if args.retry_failed_from else None,
        "prompt_path": str(args.prompt_path),
        "base_url": args.base_url,
        "model": args.model,
        "modes": modes,
        "task_count": len(tasks),
        "evaluation_type": "final_answer_only",
        "request_max_retries": args.request_max_retries,
        "request_retry_sleep": args.request_retry_sleep,
        "retry_failed_count": len(retry_keys or []),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows_map: dict[tuple[str, str], dict[str, Any]] = dict(existing_rows_map)
    work_items = []
    for mode in modes:
        current_mode_dir = get_mode_dir(args.run_dir, mode)
        for task in tasks:
            key = (str(task["task_id"]), mode)
            if retry_keys is not None and key not in retry_keys:
                continue
            work_items.append((task, mode, current_mode_dir))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(evaluate_one, task, mode, mode_dir, system_prompt, args)
            for task, mode, mode_dir in work_items
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows_map[row_key(row)] = row
            write_rows(args.output_dir / "eval_results.jsonl", rows_map)

    final_rows = sorted(rows_map.values(), key=lambda row: (str(row.get("mode", "")), str(row.get("task_id", ""))))
    summaries = [summarize(final_rows, mode) for mode in modes]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
