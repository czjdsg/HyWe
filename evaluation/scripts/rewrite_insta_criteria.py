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
DEFAULT_INPUT_PATH = Path(
    "/data/czj/HybridWebAgentBenchmarks/data_samples_v2/web_insta_v3_test50_nohuman_seed42.jsonl"
)
DEFAULT_PROMPT_PATH = FRAMEWORK_ROOT / "prompts" / "rewrite_insta_criteria_answer_only.txt"
DEFAULT_OUTPUT_DIR = FRAMEWORK_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite Insta native criteria into answer-only key points with an OpenAI-compatible model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "kimi-k2.5"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


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


def build_client(api_key: str, base_url: str) -> OpenAI:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def derive_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_stem = args.input.stem
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.model).strip("-") or "model"
    if args.output is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{dataset_stem}.answer_only_keypoints.{model_slug}.jsonl"
    else:
        output_path = args.output
    if args.summary_output is None:
        summary_path = output_path.with_suffix(".summary.json")
    else:
        summary_path = args.summary_output
    return output_path, summary_path


def build_user_prompt(item: dict[str, Any]) -> str:
    extra = item.get("extra") or {}
    question = str(item.get("question") or "").strip()
    website = str(item.get("website") or item.get("web") or item.get("domain") or "").strip()
    criteria = normalize_text_list(extra.get("criteria"))
    lines = [
        f"Task ID: {item.get('task_id', '')}",
        f"Question: {question}",
    ]
    if website:
        lines.append(f"Target website/domain: {website}")
    lines.append("Native criteria:")
    if criteria:
        lines.extend(f"- {criterion}" for criterion in criteria)
    else:
        lines.append("- [none]")
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


def normalize_result_object(obj: dict[str, Any]) -> dict[str, Any]:
    answer_key_points = normalize_text_list(obj.get("answer_key_points"))

    rewrites: list[dict[str, str]] = []
    for item in obj.get("rewrites") or []:
        if not isinstance(item, dict):
            continue
        original = str(item.get("original") or "").strip()
        rewritten = str(item.get("rewritten") or "").strip()
        if original and rewritten:
            rewrites.append({"original": original, "rewritten": rewritten})

    dropped: list[dict[str, str]] = []
    for item in obj.get("dropped_criteria") or []:
        if not isinstance(item, dict):
            continue
        original = str(item.get("original") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if original and reason:
            dropped.append({"original": original, "reason": reason})

    return {
        "answer_key_points": answer_key_points,
        "rewrites": rewrites,
        "dropped_criteria": dropped,
    }


def call_rewrite_model(
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


def load_existing_records(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            task_id = obj.get("task_id")
            if isinstance(task_id, str) and task_id:
                existing[task_id] = obj
    return existing


def build_record(
    item: dict[str, Any],
    model: str,
    prompt_name: str,
    prompt_path: Path,
    user_prompt: str,
    raw_response: str,
    parsed: dict[str, Any],
    attempt: int,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    extra = item.get("extra") or {}
    return {
        "task_id": item.get("task_id"),
        "question": item.get("question"),
        "website": item.get("website") or item.get("web") or item.get("domain") or "",
        "steps": normalize_text_list(extra.get("steps")),
        "criteria": normalize_text_list(extra.get("criteria")),
        "model": model,
        "prompt_name": prompt_name,
        "prompt_path": str(prompt_path),
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "answer_key_points": parsed.get("answer_key_points", []),
        "rewrites": parsed.get("rewrites", []),
        "dropped_criteria": parsed.get("dropped_criteria", []),
        "attempt": attempt,
        "status": status,
        "error": error,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def generate_one(
    client: OpenAI,
    item: dict[str, Any],
    model: str,
    system_prompt: str,
    prompt_path: Path,
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    prompt_name = "rewrite_insta_criteria_answer_only_v1"
    user_prompt = build_user_prompt(item)
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_rewrite_model(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            parsed = normalize_result_object(extract_json_object(raw_response))
            return build_record(
                item=item,
                model=model,
                prompt_name=prompt_name,
                prompt_path=prompt_path,
                user_prompt=user_prompt,
                raw_response=raw_response,
                parsed=parsed,
                attempt=attempt,
                status="ok",
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(min(2 * attempt, 8))
    return build_record(
        item=item,
        model=model,
        prompt_name=prompt_name,
        prompt_path=prompt_path,
        user_prompt=user_prompt,
        raw_response="",
        parsed={"answer_key_points": [], "rewrites": [], "dropped_criteria": []},
        attempt=max_retries,
        status="error",
        error=last_error,
    )


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    system_prompt = load_prompt(args.prompt_path)
    output_path, summary_path = derive_output_paths(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = jsonl_load(args.input)
    items = []
    for item in all_items:
        extra = item.get("extra") or {}
        criteria = normalize_text_list(extra.get("criteria"))
        if not isinstance(item.get("task_id"), str):
            continue
        if not isinstance(item.get("question"), str):
            continue
        if not criteria:
            continue
        items.append(item)

    if args.limit > 0:
        items = items[: args.limit]

    existing = {} if args.overwrite else load_existing_records(output_path)
    pending = [item for item in items if item["task_id"] not in existing]

    client = build_client(api_key=args.api_key, base_url=args.base_url)
    lock = threading.Lock()
    written = 0
    ok = 0
    error = 0

    mode = "w" if args.overwrite else "a"
    with output_path.open(mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    generate_one,
                    client,
                    item,
                    args.model,
                    system_prompt,
                    args.prompt_path,
                    args.max_tokens,
                    args.max_retries,
                ): item["task_id"]
                for item in pending
            }
            for future in as_completed(futures):
                record = future.result()
                with lock:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    written += 1
                    if record["status"] == "ok":
                        ok += 1
                    else:
                        error += 1

    summary = {
        "input_path": str(args.input),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "prompt_path": str(args.prompt_path),
        "model": args.model,
        "base_url": args.base_url,
        "total_items_with_criteria": len(items),
        "existing_records_kept": 0 if args.overwrite else len(existing),
        "pending_items": len(pending),
        "written_this_run": written,
        "ok_this_run": ok,
        "error_this_run": error,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
