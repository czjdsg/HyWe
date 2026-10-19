#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
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
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "judge_reference_answer.txt"


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
    parser.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL") or None, required=not os.environ.get("JUDGE_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "kimi-k3"))
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--request-max-retries", type=int, default=3)
    parser.add_argument("--request-retry-sleep", type=float, default=2.0)
    # ── 判分口径(2026-09-14 重构)──
    # 1) 预测与参考答案**严格相等**(只去首尾空白)直接判对,不调模型;
    # 2) 否则独立调 `--votes` 次判官,**多数票**决定。票数必须过半(5 票要 >=3 票同向)才出结论,
    #    某几票调用失败导致两边都不过半 -> evaluation_failed,可用 --retry-failed-from 重判。
    # 投票要有意义,各票必须是独立采样:temperature=0 时 5 票几乎逐字相同,多数票退化成单票。
    # kimi-k3 只接受 {0, 0.6, 1.0},默认取 0.6。`--votes 1 --no-exact-match --vote-temperature 0`
    # 可复现重构前的单票口径。
    parser.add_argument("--votes", type=int, default=5)
    parser.add_argument("--vote-temperature", type=float, default=0.6)
    parser.add_argument("--no-exact-match", action="store_true", help="关掉严格匹配直接判对")
    parser.add_argument("--max-concurrent-requests", type=int, default=16,
                        help="所有题、所有票共用的判官并发上限(题级并发仍是 --max-workers)")
    parser.add_argument("--resume", action="store_true",
                        help="output-dir 里已判完(completed / 直判)的题跳过,只判剩下的")
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


_REQUEST_SEMAPHORE: threading.Semaphore | None = None


def judge_once(prompt: str, system_prompt: str, args: argparse.Namespace, vote_index: int) -> dict[str, Any]:
    """调一次判官(含请求级重试)。返回一票:label 为 0/1,失败时为 None。"""
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    last_error = ""
    for attempt in range(1, max(1, args.request_max_retries) + 1):
        try:
            with _REQUEST_SEMAPHORE:
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=args.vote_temperature,
                    max_tokens=args.max_tokens,
                )
            response_text = completion.choices[0].message.content or ""
            parsed = parse_judgment(response_text)
            return {
                "vote": vote_index,
                "label": parsed["predicted_label"],
                "extracted_final_answer": parsed.get("extracted_final_answer"),
                "reasoning": parsed.get("reasoning", ""),
                "raw_judge_response": response_text,
                "request_attempt": attempt,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max(1, args.request_max_retries):
                time.sleep(max(0.0, args.request_retry_sleep) * attempt)
    return {"vote": vote_index, "label": None, "error": last_error[:500],
            "request_attempt": max(1, args.request_max_retries)}


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
                "match_type": "missing_reference_answer",
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
                "match_type": "no_prediction",
                "predicted_label": 0,
                "reasoning": "Agent did not provide a final answer.",
                "extracted_final_answer": None,
            }
        )
        return row

    # 严格匹配:只去首尾空白,大小写/标点/格式有任何差异都交给判官 —— 宁可多调模型,
    # 也不让宽松归一化把"看起来像"的错答直接判对。
    if not args.no_exact_match and prediction == reference_answer:
        row.update(
            {
                "evaluation_status": "exact_match",
                "match_type": "exact_match",
                "predicted_label": 1,
                "reasoning": "Prediction is exactly identical to the reference answer.",
                "extracted_final_answer": prediction,
                "votes": [],
                "vote_counts": {},
            }
        )
        return row

    prompt = build_task_prompt(task)
    row["judge_prompt"] = prompt
    n_votes = max(1, int(args.votes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_votes) as pool:
        votes = list(pool.map(lambda i: judge_once(prompt, system_prompt, args, i), range(1, n_votes + 1)))

    yes = sum(1 for v in votes if v.get("label") == 1)
    no = sum(1 for v in votes if v.get("label") == 0)
    failed = n_votes - yes - no
    need = n_votes // 2 + 1                     # 过半:5 票要 >=3 票
    row["votes"] = votes
    row["vote_counts"] = {"1": yes, "0": no, "failed": failed}
    row["match_type"] = "vote"
    if yes >= need or no >= need:
        label = 1 if yes >= need else 0
        rep = next(v for v in votes if v.get("label") == label)   # 取一张多数方的票填兼容字段
        row.update(
            {
                "evaluation_status": "completed",
                "predicted_label": label,
                "vote_unanimous": (yes == n_votes or no == n_votes),
                "extracted_final_answer": rep.get("extracted_final_answer"),
                "reasoning": rep.get("reasoning", ""),
                "raw_judge_response": rep.get("raw_judge_response", ""),
            }
        )
        return row

    row.update(
        {
            "evaluation_status": "evaluation_failed",
            "predicted_label": None,
            "reasoning": "",
            "extracted_final_answer": None,
            "error": f"no majority: yes={yes} no={no} failed={failed} (need {need}/{n_votes})"
                     + (f"; last error: {next((v.get('error') for v in votes if v.get('error')), '')}"
                        if failed else ""),
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
        "exact_match": sum(1 for row in rows if row.get("evaluation_status") == "exact_match"),
        "vote_unanimous": sum(1 for row in rows if row.get("vote_unanimous") is True),
        "vote_split": sum(1 for row in rows if row.get("vote_unanimous") is False),
        "success_rate_completed_only": (success / len(completed)) if completed else 0.0,
        "success_rate_all": (success / len(rows)) if rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global _REQUEST_SEMAPHORE
    _REQUEST_SEMAPHORE = threading.Semaphore(max(1, int(args.max_concurrent_requests)))

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
        "votes": args.votes,
        "vote_temperature": args.vote_temperature,
        "exact_match_shortcut": not args.no_exact_match,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows_map: dict[tuple[str, str, str], dict[str, Any]] = dict(existing_rows_map)
    done_keys: set[tuple[str, str, str]] = set()
    out_path = args.output_dir / "eval_results.jsonl"
    if args.resume and out_path.exists():
        for row in load_jsonl(out_path):
            if not row.get("task_id"):
                continue
            rows_map[row_key(row)] = row
            # 只有判出结论的才算完成;prediction 变了(补跑过)的要重判
            if row.get("evaluation_status") in {"completed", "exact_match", "no_prediction", "missing_reference_answer"}:
                done_keys.add((row_key(row), str(row.get("prediction", "")).strip(), str(row.get("termination", ""))))
    submit_items = []
    for task in work_items:
        key = row_key(task)
        if retry_keys is not None and key not in retry_keys:
            continue
        if (key, str(task.get("prediction", "")).strip(), str(task.get("termination", ""))) in done_keys:
            continue
        submit_items.append(task)
    print(f"  待判 {len(submit_items)} / 共 {len(work_items)}(resume 跳过 {len(work_items) - len(submit_items) if args.resume else 0})", flush=True)

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
