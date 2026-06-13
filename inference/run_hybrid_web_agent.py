import argparse
import asyncio
import copy
import json
import os
import shutil
from datetime import date
from pathlib import Path

from hybrid_agent import HybridBrowserAgent
from hybrid_runtime import HybridPlaywrightComputer
from hybrid_utils import (
    append_jsonl,
    coord_scale_for_model,
    load_jsonl,
    safe_task_id,
    save_json,
)


DEFAULT_OPERATION_START_URL = "https://www.google.com/"
# Optional path to a local GAIA attachment dump; only used when a task record
# carries a relative `attachment` field. Override with HYBRID_GAIA_ATTACHMENT_ROOT.
GAIA_ATTACHMENT_ROOT = Path(os.environ.get("HYBRID_GAIA_ATTACHMENT_ROOT", ""))


def safe_sandbox_path(value: str, fallback: str = "attachment") -> str:
    import re

    name = Path(value or fallback).name.strip() or fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return (name.strip("._-") or fallback)[:180]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--web", default="")
    parser.add_argument("--input-path", default="")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-task-attempts", type=int, default=1)
    parser.add_argument(
        "--agent-mode",
        choices=["hybrid", "hybrid_routed", "hybrid_guided_routed", "operation", "search"],
        default=os.environ.get("AGENT_MODE", "hybrid"),
        help=(
            "Agent ablation mode: hybrid (一体化) uses all tools in one prompt; "
            "hybrid_routed (路由版本) routes per turn between a search-side prompt "
            "and an operation-side prompt, with an explicit `switch` tool that "
            "flips sides; hybrid_guided_routed uses an external router plan to "
            "hard-route S/O while hiding the plan from decision-model prompts; "
            "operation removes search/visit; search removes browser-operation tools."
        ),
    )
    parser.add_argument(
        "--coord-scale",
        type=int,
        default=int(os.environ.get("COORD_SCALE", "-1") or -1),
        help=(
            "Coordinate convention used by the operator model. If left at -1 "
            "(default), it is auto-inferred from --model via "
            "coord_scale_for_model: kimi* -> 1 (0..1 relative), qwen* -> 1000 "
            "(0..1000 logical grid), claude/gpt-4v* -> 0 (raw pixels). Set "
            "explicitly to override the auto-inference; the COORD_SCALE env "
            "var also overrides the default. real_x = x_input * viewport / N "
            "when N > 0, else x_input is passed through as a pixel."
        ),
    )

    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--summary-api-key", default="")
    parser.add_argument("--summary-api-key-env", default="SUMMARY_API_KEY")
    parser.add_argument("--jina-api-key", default="")
    parser.add_argument("--jina-api-key-env", default="JINA_API_KEY")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--summary-base-url", default="")
    parser.add_argument("--summary-model", default="")
    # Routed hybrid: per-side model overrides. When set, S-side uses
    # routed_search_model (+ routed_search_base_url if different) and O-side
    # uses routed_operation_model. Falls back to --model/--base-url if unset.
    parser.add_argument("--routed-search-model", default="")
    parser.add_argument("--routed-search-base-url", default="")
    parser.add_argument("--routed-operation-model", default="")
    parser.add_argument("--routed-operation-base-url", default="")
    parser.add_argument(
        "--guided-route-plan-path",
        default="",
        help=(
            "JSON/JSONL route-plan file for hybrid_guided_routed. Accepts either "
            "a task_id-keyed JSON object or rows with task_id + route_plan/guided_route_plan."
        ),
    )
    parser.add_argument("--router-api-key", default="")
    parser.add_argument("--router-api-key-env", default="ROUTER_API_KEY")
    parser.add_argument("--router-base-url", default="")
    parser.add_argument("--router-model", default="")
    parser.add_argument("--router-timeout", type=float, default=900)
    parser.add_argument("--router-max-tokens", type=int, default=4096)
    parser.add_argument("--api-timeout", type=float, default=900)
    parser.add_argument("--summary-timeout", type=float, default=900)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--summary-tokenizer-path", default="")
    parser.add_argument(
        "--proxy-url",
        default="",
        help="Browser proxy only. Runtime HTTP clients keep the existing shell proxy environment.",
    )
    parser.add_argument("--no-proxy-hosts", default="localhost,127.0.0.1")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Per-turn LLM output cap. Bumped from 2048 to "
                             "16384 so the JSON <report> + <tool_call>/<answer> "
                             "always fit even on long browser observations.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--max-format-retries", type=int, default=3)
    parser.add_argument("--max-observation-chars", type=int, default=0,
                        help="0 = no truncation (full last_observation into agent prompt).")
    parser.add_argument("--max-action-history-chars", type=int, default=0,
                        help="0 = no truncation of the recent-actions block in the agent prompt.")
    parser.add_argument("--visit-max-webpage-tokens", type=int, default=32000)
    parser.add_argument("--visit-max-webpage-chars", type=int, default=0)
    parser.add_argument("--visit-max-images", type=int, default=4)
    parser.add_argument("--visit-max-image-url-chars", type=int, default=2000000)
    parser.add_argument("--visit-summary-max-tokens", type=int, default=4000)
    parser.add_argument(
        "--code-sandbox-base-url",
        default=os.environ.get("CODE_SANDBOX_BASE_URL", "http://127.0.0.1:8080"),
    )
    parser.add_argument("--code-sandbox-request-timeout", type=float, default=900)
    parser.add_argument("--code-sandbox-max-output-chars", type=int, default=0,
                        help="0 = no truncation of code_sandbox stdout in the saved record / agent context.")
    parser.add_argument("--code-sandbox-max-file-bytes", type=int, default=25_000_000)

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--use-agent-images", dest="use_agent_images", action="store_true", default=True)
    parser.add_argument("--no-agent-images", dest="use_agent_images", action="store_false")
    parser.add_argument("--use-input-images", dest="use_input_images", action="store_true", default=True)
    parser.add_argument("--no-input-images", dest="use_input_images", action="store_false")
    parser.add_argument("--max-input-images", type=int, default=8)
    parser.add_argument("--use-visit-images", dest="use_visit_images", action="store_true", default=False)
    parser.add_argument("--no-visit-images", dest="use_visit_images", action="store_false")
    parser.add_argument("--use-css-som", dest="use_css_som", action="store_true", default=True)
    parser.add_argument("--no-css-som", dest="use_css_som", action="store_false")
    parser.add_argument("--use-omni-som", action="store_true")
    parser.add_argument("--omni-url", default="")
    parser.add_argument("--keep-user-info", action="store_true")
    parser.add_argument("--no-sandbox", action="store_true", default=True)
    parser.add_argument("--sandboxed", dest="no_sandbox", action="store_false")

    args = parser.parse_args()
    if args.agent_mode not in {"hybrid", "hybrid_routed", "hybrid_guided_routed", "operation", "search"}:
        args.agent_mode = "hybrid"

    # Resolve --coord-scale=-1 (the default) by auto-inferring from --model.
    # The COORD_SCALE env var (read by argparse) wins over auto-inference;
    # an explicit non-negative --coord-scale on the CLI wins over both.
    if args.coord_scale is None or args.coord_scale < 0:
        args.coord_scale = coord_scale_for_model(args.model)

    if not args.task and not args.input_path:
        parser.error("Either --task or --input-path must be provided.")

    args.api_key = args.api_key or os.environ.get(args.api_key_env, "")
    if not args.api_key:
        parser.error(f"API key missing. Pass --api-key or set {args.api_key_env}.")
    args.summary_api_key = args.summary_api_key or os.environ.get(
        args.summary_api_key_env,
        "",
    )
    args.jina_api_key = args.jina_api_key or os.environ.get(
        args.jina_api_key_env,
        "",
    )
    args.tokenizer_path = args.tokenizer_path or os.environ.get(
        "TOKENIZER_PATH",
        "Qwen/Qwen2.5-7B-Instruct",
    )
    args.summary_base_url = args.summary_base_url or os.environ.get(
        "SUMMARY_BASE_URL",
        args.base_url,
    )
    args.summary_model = args.summary_model or os.environ.get(
        "SUMMARY_MODEL",
        args.model,
    )
    args.summary_api_key = args.summary_api_key or os.environ.get(
        "SUMMARY_API_KEY",
        args.api_key,
    )
    args.router_api_key = (
        args.router_api_key
        or os.environ.get(args.router_api_key_env, "")
        or args.api_key
    )
    args.router_base_url = args.router_base_url or os.environ.get(
        "ROUTER_BASE_URL",
        args.base_url,
    )
    args.router_model = args.router_model or os.environ.get(
        "ROUTER_MODEL",
        args.model,
    )
    args.summary_tokenizer_path = args.summary_tokenizer_path or os.environ.get(
        "SUMMARY_TOKENIZER_PATH",
        args.tokenizer_path,
    )
    # Use --proxy-url if explicitly set; else fall back to TASK_PROXY_URL env;
    # else go direct (empty string). Pre-empty here removes the historical
    # hard-coded "127.0.0.1:8990" default so machines without a local proxy
    # don't try to route through a port that nothing is listening on.
    args.proxy_url = args.proxy_url or os.environ.get("TASK_PROXY_URL", "")

    args.current_date = date.today().isoformat()
    return args


def apply_runtime_proxy(args):
    required_hosts = [item.strip() for item in (args.no_proxy_hosts or "").split(",") if item.strip()]
    if not required_hosts:
        return

    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    merged = [item.strip() for item in existing.split(",") if item.strip()]
    for host in required_hosts:
        if host not in merged:
            merged.append(host)
    no_proxy = ",".join(merged)
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy


def normalize_attachment_files(item: dict) -> list[dict]:
    raw_files = item.get("attachment_files") or item.get("attachments") or []
    if isinstance(raw_files, dict):
        raw_files = [raw_files]
    if not isinstance(raw_files, list):
        raw_files = []

    normalized = []
    seen = set()
    for entry in raw_files:
        if not isinstance(entry, dict):
            continue
        path_value = str(entry.get("path") or entry.get("local_path") or "").strip()
        if not path_value:
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = path.resolve()
        sandbox_path = safe_sandbox_path(
            str(entry.get("sandbox_path") or entry.get("file_name") or path.name)
        )
        key = (str(path), sandbox_path)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "file_name": str(entry.get("file_name") or path.name),
                "path": str(path),
                "sandbox_path": sandbox_path,
                "status": entry.get("status", ""),
                "size_bytes": entry.get("size_bytes", 0),
                "overview": entry.get("overview", ""),
            }
        )

    if normalized:
        return normalized

    extra = item.get("extra") if isinstance(item.get("extra"), dict) else {}
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    file_path = str(extra.get("file_path") or raw.get("file_path") or "").strip()
    if not file_path:
        return []

    dataset = str(item.get("dataset") or "").strip()
    source_path = Path(file_path)
    if not source_path.is_absolute() and dataset == "GAIA":
        source_path = GAIA_ATTACHMENT_ROOT / file_path
    elif not source_path.is_absolute():
        source_path = source_path.resolve()

    file_name = str(extra.get("file_name") or raw.get("file_name") or source_path.name)
    return [
        {
            "file_name": file_name,
            "path": str(source_path),
            "sandbox_path": safe_sandbox_path(file_name or source_path.name),
            "status": str(extra.get("attachment_parse_status") or ""),
            "size_bytes": source_path.stat().st_size if source_path.exists() else 0,
            "overview": "",
        }
    ]


GUIDED_ROUTE_FIELD_NAMES = (
    "guided_route_plan",
    "route_plan",
    "standard_route",
    "gold_route",
    "switch_plan",
    "annotated_route",
)


def extract_guided_route_plan(item: dict):
    for name in GUIDED_ROUTE_FIELD_NAMES:
        value = item.get(name)
        if value:
            return value
    extra = item.get("extra") if isinstance(item.get("extra"), dict) else {}
    for name in GUIDED_ROUTE_FIELD_NAMES:
        value = extra.get(name)
        if value:
            return value
    return None


def scrub_private_guided_fields(item: dict) -> dict:
    def scrub(value):
        if isinstance(value, dict):
            return {
                key: scrub(inner)
                for key, inner in value.items()
                if key not in GUIDED_ROUTE_FIELD_NAMES
            }
        if isinstance(value, list):
            return [scrub(inner) for inner in value]
        return value

    return scrub(copy.deepcopy(item))


def save_guided_router_trace_sidecar(task_dir: Path, task_id: str, agent) -> None:
    trace = getattr(agent, "guided_router_trace", None) if agent else None
    if not trace:
        return
    save_json(
        task_dir / "guided_router_trace.json",
        {
            "task_id": task_id,
            "warning": (
                "Private debug sidecar. Contains hidden guided-route/controller "
                "inputs and must not be used as training trajectory data."
            ),
            "events": copy.deepcopy(trace),
        },
    )


def load_guided_route_plan_map(path_value: str) -> dict[str, object]:
    path = Path(path_value or "")
    if not path_value:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"guided route plan file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = load_jsonl(str(path))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
            rows = data["tasks"]
        elif isinstance(data, dict):
            mapped = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    mapped[str(key)] = extract_guided_route_plan(value) or value
                else:
                    mapped[str(key)] = value
            return mapped
        else:
            raise ValueError(
                "guided route plan file must be JSONL, JSON list, or JSON object"
            )

    mapped = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or row.get("id") or "").strip()
        plan = extract_guided_route_plan(row)
        if task_id and plan:
            mapped[task_id] = plan
    return mapped


def build_task_specs(args):
    if args.input_path:
        raw_items = load_jsonl(args.input_path)
    else:
        raw_items = [{"question": args.task, "task_id": args.task_id, "web": args.web}]

    guided_route_plans = load_guided_route_plan_map(
        getattr(args, "guided_route_plan_path", "")
    )
    specs = []
    for index, item in enumerate(raw_items):
        question = item.get("question") or item.get("task") or item.get("instruction") or ""
        if not question:
            continue

        task_web = (item.get("web") or item.get("url") or args.web or "").strip()
        start_url = task_web
        if args.agent_mode == "search":
            start_url = ""
        elif args.agent_mode == "operation" and not start_url:
            start_url = DEFAULT_OPERATION_START_URL
        input_images = item.get("input_images") or item.get("images") or []
        if isinstance(input_images, str):
            input_images = [input_images]
        elif not isinstance(input_images, list):
            input_images = []
        attachment_files = normalize_attachment_files(item)
        task_id = safe_task_id(
            item.get("task_id", ""),
            fallback=f"task_{index:04d}",
        )
        raw_task_id = str(item.get("task_id") or item.get("id") or "").strip()
        guided_route_plan = (
            extract_guided_route_plan(item)
            or guided_route_plans.get(task_id)
            or guided_route_plans.get(raw_task_id)
        )

        specs.append(
            {
                "task_id": task_id,
                "question": question.strip(),
                "web": start_url,
                "task_web": task_web,
                "input_images": input_images,
                "attachment_files": attachment_files,
                "guided_route_plan": guided_route_plan,
                "raw": item,
            }
        )

    return specs


async def run_single_task(args, spec: dict, output_root: Path) -> dict:
    task_dir = output_root / spec["task_id"]
    max_attempts = max(1, int(args.max_task_attempts or 1))
    attempt_summaries = []
    final_result = None

    for attempt in range(1, max_attempts + 1):
        if task_dir.exists():
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)

        run_args = copy.copy(args)
        run_args.task_id = spec["task_id"]
        run_args.task = spec["question"]
        run_args.initial_url = spec["web"]
        run_args.web = "" if args.agent_mode == "search" else spec["web"]
        run_args.task_web_context = spec.get("task_web", "")
        run_args.input_images = spec.get("input_images", [])
        run_args.attachment_files = spec.get("attachment_files", [])
        run_args.guided_route_plan = spec.get("guided_route_plan")
        run_args.task_dir = str(task_dir)

        save_json(
            task_dir / "meta.json",
            {
                "task_id": spec["task_id"],
                "question": spec["question"],
                "web": spec["web"],
                "task_web": spec.get("task_web", ""),
                "input_images": spec.get("input_images", []),
                "attachment_files": spec.get("attachment_files", []),
                "model": args.model,
                "base_url": args.base_url,
                "api_timeout": args.api_timeout,
                "summary_model": args.summary_model,
                "summary_base_url": args.summary_base_url,
                "summary_timeout": args.summary_timeout,
                "code_sandbox_base_url": args.code_sandbox_base_url,
                "code_sandbox_request_timeout": args.code_sandbox_request_timeout,
                "code_sandbox_max_output_chars": args.code_sandbox_max_output_chars,
                "code_sandbox_max_file_bytes": args.code_sandbox_max_file_bytes,
                "agent_mode": args.agent_mode,
                "guided_route_plan_present": bool(spec.get("guided_route_plan")),
                "current_date": args.current_date,
                "attempt": attempt,
                "max_task_attempts": max_attempts,
            },
        )

        agent = None
        try:
            if args.agent_mode == "search":
                agent = HybridBrowserAgent(
                    web=None,
                    args=run_args,
                    task=spec["question"],
                )
                result = await agent.run()
            else:
                async with HybridPlaywrightComputer(run_args) as web:
                    agent = HybridBrowserAgent(
                        web=web,
                        args=run_args,
                        task=spec["question"],
                    )
                    result = await agent.run()
        except Exception as exc:
            result = {
                "question": spec["question"],
                "prediction": "",
                "termination": "task_error",
                "records": copy.deepcopy(agent.records) if agent else [],
                "usage": copy.deepcopy(agent.usage) if agent else {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if agent is not None:
                agent.close()

        if args.agent_mode == "hybrid_guided_routed":
            save_guided_router_trace_sidecar(task_dir, spec["task_id"], agent)

        result["task_id"] = spec["task_id"]
        result["agent_mode"] = args.agent_mode
        result["web"] = spec["web"]
        result["raw"] = (
            scrub_private_guided_fields(spec["raw"])
            if args.agent_mode == "hybrid_guided_routed"
            else spec["raw"]
        )
        if args.agent_mode == "hybrid_guided_routed":
            result["guided_route_plan_present"] = bool(spec.get("guided_route_plan"))
        result["attempt"] = attempt
        attempt_summaries.append(
            {
                "attempt": attempt,
                "termination": result.get("termination", ""),
                "error": (result.get("error") or "")[:500],
            }
        )
        final_result = result

        if result.get("termination") == "answer":
            break

    if final_result is None:
        final_result = {
            "question": spec["question"],
            "prediction": "",
            "termination": "task_error",
            "records": [],
            "usage": {},
            "error": "No attempt was executed.",
            "task_id": spec["task_id"],
            "agent_mode": args.agent_mode,
            "web": spec["web"],
            "raw": spec["raw"],
            "attempt": 0,
        }

    final_result["attempt_count"] = len(attempt_summaries)
    final_result["attempts"] = attempt_summaries
    save_json(task_dir / "result.json", final_result)
    return final_result


async def main():
    args = parse_args()
    apply_runtime_proxy(args)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.output_path:
        output_path = Path(args.output_path).resolve()
    else:
        output_path = output_root / "results.jsonl"

    specs = build_task_specs(args)
    write_lock = asyncio.Lock()
    worker_limit = max(1, int(args.max_workers or 1))
    semaphore = asyncio.Semaphore(worker_limit)

    async def run_and_persist(spec: dict):
        async with semaphore:
            try:
                result = await run_single_task(args, spec, output_root=output_root)
            except Exception as exc:
                task_dir = output_root / spec["task_id"]
                task_dir.mkdir(parents=True, exist_ok=True)
                result = {
                    "question": spec["question"],
                    "prediction": "",
                    "termination": "task_error",
                    "records": [],
                    "usage": {},
                    "task_id": spec["task_id"],
                    "web": spec["web"],
                    "raw": (
                        scrub_private_guided_fields(spec["raw"])
                        if args.agent_mode == "hybrid_guided_routed"
                        else spec["raw"]
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if args.agent_mode == "hybrid_guided_routed":
                    result["guided_route_plan_present"] = bool(
                        spec.get("guided_route_plan")
                    )
                save_json(task_dir / "result.json", result)

            async with write_lock:
                append_jsonl(str(output_path), result)
                prediction_preview = result.get("prediction", "")
                if isinstance(prediction_preview, (list, dict)):
                    prediction_preview = str(prediction_preview)
                else:
                    prediction_preview = str(prediction_preview or "")
                print(
                    f"[done] {spec['task_id']} termination={result['termination']} attempts={result.get('attempt_count', 1)} prediction={prediction_preview[:120]}"
                )

    await asyncio.gather(*(run_and_persist(spec) for spec in specs))


if __name__ == "__main__":
    asyncio.run(main())
