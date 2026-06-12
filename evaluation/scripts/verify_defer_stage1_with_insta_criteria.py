#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from openai import OpenAI


DEFAULT_RUN_DIR = Path(
    "/data/czj/HybridWebAgentBenchmarks/run_v2/"
    "insta_v3_test50_nohuman_kimi_k26_summary_qwen35plus_w4_max100_20260427"
)
DEFAULT_DATASET = Path(
    "/data/czj/HybridWebAgentBenchmarks/data_samples_v2/web_insta_v3_test50_nohuman_seed42.jsonl"
)
DEFAULT_ANNOTATED_EVAL = Path(
    "/data/czj/answer_eval_framework/outputs/"
    "insta_v3_test50_nohuman_k26run_annotated20_static_url_once_kimi-k2.6_20260429/eval_results.jsonl"
)
DEFAULT_DEFER_EVAL = Path(
    "/data/czj/answer_eval_framework/outputs/"
    "insta_v3_test50_nohuman_k26run_force_defer_repeat3_kimi-k2.6_20260428/eval_results.jsonl"
)
DEFAULT_STAGE1_PROMPT = Path("/data/czj/answer_eval_framework/prompts/judge_defer_stage1.txt")
DEFAULT_OUTPUT_DIR = Path(
    "/data/czj/answer_eval_framework/outputs/"
    "insta_k26_annotated_wrong_defer_correct_stage1_with_criteria_20260430"
)


CRITERIA_REWRITE_SYSTEM_PROMPT = """You rewrite Insta task criteria into answer-only key points for final-answer completeness checking.

Rules:
1. Convert criteria into requirements that the agent's final answer must satisfy.
2. Remove or rewrite navigation/operation/process wording such as clicking, accepting cookies, browsing pages, or using a specific route.
3. Preserve answer constraints such as required entities, number of items, attributes, relationships, scope, source/site affiliation, formatting requirements, and disallowed ambiguity.
4. Do not invent specific factual answer values that are not present in the task or criteria.
5. Keep the key points solution-agnostic and usable for search, browser operation, or hybrid trajectories.
6. If a criterion is only about the route and has no final-answer implication, put it in dropped_criteria instead of answer_key_points.

Return JSON only:
{
  "answer_key_points": ["..."],
  "dropped_criteria": ["..."],
  "reason": "short explanation"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find Insta task-mode pairs where annotated static/url evaluation failed but defer_framework passed, "
            "rewrite native Insta criteria into answer key points, then rerun defer stage-1 with those key points."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--annotated-eval", type=Path, default=DEFAULT_ANNOTATED_EVAL)
    parser.add_argument("--defer-eval", type=Path, default=DEFAULT_DEFER_EVAL)
    parser.add_argument("--stage1-system-prompt", type=Path, default=DEFAULT_STAGE1_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "kimi-k2.6"))
    parser.add_argument("--rewrite-model", default="")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--request-max-retries", type=int, default=5)
    parser.add_argument("--request-retry-sleep", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse-existing-rewrites", action="store_true")
    return parser.parse_args()


def jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
        if fenced:
            raw = fenced.group(1).strip()
    if not raw.startswith("{"):
        match = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("no JSON object found")
        raw = match.group(1).strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def call_llm_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any], str, int]:
    last_error = ""
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            raw_text = completion.choices[0].message.content or ""
            return extract_json_object(raw_text), raw_text, attempt
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(max(1.0, retry_sleep) * attempt)
    raise RuntimeError(last_error or "LLM request failed")


def normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for item in jsonl_load(path):
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        extra = item.get("extra") if isinstance(item.get("extra"), dict) else {}
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
        criteria = normalize_text_list(extra.get("criteria")) or normalize_text_list(raw.get("criteria"))
        data[task_id] = {
            "task_id": task_id,
            "question": item.get("question", "") if isinstance(item.get("question"), str) else "",
            "web": item.get("web", "") if isinstance(item.get("web"), str) else "",
            "criteria": criteria,
            "steps": normalize_text_list(extra.get("steps")) or normalize_text_list(raw.get("steps")),
        }
    return data


def result_label(row: dict[str, Any] | None) -> int | None:
    if not isinstance(row, dict):
        return None
    if row.get("evaluation_status") != "completed":
        return None
    value = row.get("predicted_label")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    return None


def repeat_index(row: dict[str, Any]) -> int:
    try:
        return int(row.get("repeat_index") or 1)
    except Exception:
        return 1


def majority_label(rows: list[dict[str, Any]]) -> tuple[int | None, list[int | None]]:
    ordered = sorted(rows, key=repeat_index)
    labels = [result_label(row) for row in ordered]
    if sum(label == 1 for label in labels) >= 2:
        return 1, labels
    if sum(label == 0 for label in labels) >= 2:
        return 0, labels
    return None, labels


def load_prediction(run_dir: Path, task_id: str, mode: str) -> tuple[str, str]:
    result_path = run_dir / mode / task_id / "result.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return str(data.get("question") or "").strip(), str(data.get("prediction") or "").strip()


def rewrite_prompt(task: dict[str, Any]) -> str:
    criteria_lines = "\n".join(f"- {criterion}" for criterion in task.get("criteria", []))
    step_lines = "\n".join(f"- {step}" for step in task.get("steps", []))
    return "\n\n".join(
        [
            "## Task",
            task.get("question", ""),
            "",
            "## Website",
            task.get("web", ""),
            "",
            "## Native Insta Criteria",
            criteria_lines or "(none)",
            "",
            "## Native Insta Steps (context only; do not preserve route bias)",
            step_lines or "(none)",
        ]
    )


def build_stage1_prompt(question: str, prediction: str, key_points: list[str]) -> str:
    key_point_lines = "\n".join(f"- {point}" for point in key_points) or "(none)"
    return "\n\n".join(
        [
            "## Task",
            question,
            "",
            "## Answer Required Key Points",
            key_point_lines,
            "",
            "## Stage-1 Completeness Instruction",
            (
                "Use the Answer Required Key Points as mandatory final-answer completeness requirements. "
                "This is still defer_framework stage 1: decide whether the final answer appears to solve the task; "
                "do not judge whether the facts are grounded in trajectory evidence yet."
            ),
            "",
            "## Agent Final Answer",
            prediction,
        ]
    )


def find_target_pairs(annotated_rows: list[dict[str, Any]], defer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    defer_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in defer_rows:
        task_id = row.get("task_id")
        mode = row.get("mode")
        if isinstance(task_id, str) and isinstance(mode, str):
            defer_groups[(task_id, mode)].append(row)

    targets: list[dict[str, Any]] = []
    for row in annotated_rows:
        task_id = row.get("task_id")
        mode = row.get("mode")
        if not isinstance(task_id, str) or not isinstance(mode, str):
            continue
        annotated_label = result_label(row)
        defer_label, defer_labels = majority_label(defer_groups.get((task_id, mode), []))
        if annotated_label == 0 and defer_label == 1:
            targets.append(
                {
                    "task_id": task_id,
                    "mode": mode,
                    "annotated_evaluation_type": row.get("evaluation_type", ""),
                    "annotated_reason": row.get("reason", ""),
                    "defer_majority_label": defer_label,
                    "defer_repeat_labels": defer_labels,
                }
            )
    targets.sort(key=lambda item: (item["task_id"], item["mode"]))
    return targets


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing --api-key or OPENAI_API_KEY")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "stage1_with_criteria_results.jsonl"
    rewrites_path = output_dir / "criteria_rewrites.jsonl"
    targets_path = output_dir / "selected_annotated_wrong_defer_correct.jsonl"

    client_kwargs: dict[str, Any] = {"api_key": args.api_key, "timeout": args.request_timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    rewrite_model = args.rewrite_model.strip() or args.model

    dataset = load_dataset(args.dataset)
    annotated_rows = jsonl_load(args.annotated_eval)
    defer_rows = jsonl_load(args.defer_eval)
    targets = find_target_pairs(annotated_rows, defer_rows)
    if args.limit:
        targets = targets[: args.limit]
    write_jsonl(targets_path, targets)

    rewrite_cache: dict[str, dict[str, Any]] = {}
    rewrite_rows: list[dict[str, Any]] = []
    if args.reuse_existing_rewrites and rewrites_path.exists():
        for row in jsonl_load(rewrites_path):
            task_id = row.get("task_id")
            if isinstance(task_id, str) and task_id:
                rewrite_cache[task_id] = row
                rewrite_rows.append(row)

    for task_id in sorted({target["task_id"] for target in targets}):
        if task_id in rewrite_cache:
            continue
        task = dataset.get(task_id, {"task_id": task_id, "question": "", "criteria": [], "steps": [], "web": ""})
        parsed, raw, attempt = call_llm_json(
            client,
            rewrite_model,
            CRITERIA_REWRITE_SYSTEM_PROMPT,
            rewrite_prompt(task),
            args.max_tokens,
            args.request_max_retries,
            args.request_retry_sleep,
        )
        key_points = normalize_text_list(parsed.get("answer_key_points"))
        row = {
            "task_id": task_id,
            "question": task.get("question", ""),
            "web": task.get("web", ""),
            "native_criteria": task.get("criteria", []),
            "native_steps": task.get("steps", []),
            "answer_key_points": key_points,
            "dropped_criteria": normalize_text_list(parsed.get("dropped_criteria")),
            "rewrite_reason": str(parsed.get("reason") or ""),
            "rewrite_model": rewrite_model,
            "rewrite_request_attempt": attempt,
            "rewrite_raw_response": raw,
        }
        rewrite_cache[task_id] = row
        rewrite_rows.append(row)
        write_jsonl(rewrites_path, rewrite_rows)

    stage1_system_prompt = args.stage1_system_prompt.read_text(encoding="utf-8").strip()
    result_rows: list[dict[str, Any]] = []
    for target in targets:
        task_id = target["task_id"]
        mode = target["mode"]
        dataset_task = dataset.get(task_id, {})
        question, prediction = load_prediction(args.run_dir, task_id, mode)
        question = question or str(dataset_task.get("question") or "")
        rewrite = rewrite_cache.get(task_id, {})
        answer_key_points = normalize_text_list(rewrite.get("answer_key_points"))
        stage1_prompt = build_stage1_prompt(question, prediction, answer_key_points)
        parsed, raw, attempt = call_llm_json(
            client,
            args.model,
            stage1_system_prompt,
            stage1_prompt,
            args.max_tokens,
            args.request_max_retries,
            args.request_retry_sleep,
        )
        solved = bool(parsed.get("solved"))
        result_rows.append(
            {
                "task_id": task_id,
                "mode": mode,
                "question": question,
                "annotated_evaluation_type": target.get("annotated_evaluation_type"),
                "annotated_reason": target.get("annotated_reason"),
                "defer_repeat_labels": target.get("defer_repeat_labels"),
                "native_criteria": normalize_text_list(dataset_task.get("criteria")),
                "answer_key_points": answer_key_points,
                "stage1_with_criteria_solved": solved,
                "predicted_label": 1 if solved else 0,
                "stage1_reason": str(parsed.get("reason") or ""),
                "extracted_final_answer": str(parsed.get("extracted_final_answer") or ""),
                "stage1_prompt": stage1_prompt,
                "stage1_raw_response": raw,
                "stage1_request_attempt": attempt,
                "model": args.model,
                "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        write_jsonl(results_path, result_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(args.run_dir),
        "dataset": str(args.dataset),
        "annotated_eval": str(args.annotated_eval),
        "defer_eval": str(args.defer_eval),
        "model": args.model,
        "rewrite_model": rewrite_model,
        "target_pair_count": len(targets),
        "unique_task_count": len({target["task_id"] for target in targets}),
        "stage1_with_criteria_counts": dict(Counter(row["predicted_label"] for row in result_rows)),
        "by_mode": {
            mode: {
                "total": sum(row["mode"] == mode for row in result_rows),
                "solved": sum(row["mode"] == mode and row["stage1_with_criteria_solved"] for row in result_rows),
            }
            for mode in ["search", "operation", "hybrid"]
        },
        "outputs": {
            "targets": str(targets_path),
            "criteria_rewrites": str(rewrites_path),
            "stage1_results": str(results_path),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
