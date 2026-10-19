import argparse
import asyncio
import copy
import json
import os
import shutil
import traceback
from datetime import date
from pathlib import Path

from hybrid_agent import HybridBrowserAgent
import failure_summary as _fsum
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


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        "--max-iter-restarts",
        type=int,
        default=1,
        help="一道题最多允许撞几次步数上限(--max-iter)。前 N-1 次撞顶不强制收口,整题从头重跑;"
             "第 N 次撞顶才在最后一步强制作答。默认 1 = 原行为(第一次撞顶就强制作答)。"
             "与 --max-task-attempts(报错/格式错重试)分开计数。",
    )
    parser.add_argument(
        "--attempt-offset",
        type=int,
        default=0,
        help="尝试编号从 offset+1 开始。补跑旧 run 里撞顶的题时用:原来那次算第 1 次,补跑传 1 ——"
             "seed 按 (task_id, attempt) 派生,编号不错开的话补跑的第一次会拿到和原来完全相同的 seed 序列,"
             "逐字重放那条撞顶轨迹。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑:输出目录里已有完整 result.json 的任务直接跳过,不重跑。",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["hybrid", "hybrid_routed", "hybrid_guided_routed", "hybrid_react_v2", "operation", "search"],
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
    parser.add_argument("--router-timeout", type=float, default=600)
    parser.add_argument("--router-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--teacher-phase-control",
        action=argparse.BooleanOptionalAction,
        default=env_flag("HYBRID_TEACHER_PHASE_CONTROL", False),
        help=(
            "For hybrid_react_v2 teacher collection, expose only the current "
            "compiled phase and gate switch/answer with a private verifier."
        ),
    )
    parser.add_argument("--teacher-phase-verifier-model", default="")
    parser.add_argument("--teacher-phase-verifier-base-url", default="")
    parser.add_argument("--teacher-phase-verifier-api-key", default="")
    parser.add_argument("--teacher-phase-verifier-api-key-env", default="PHASE_VERIFIER_API_KEY")
    parser.add_argument("--teacher-phase-verifier-timeout", type=float, default=600)
    parser.add_argument("--teacher-phase-verifier-max-tokens", type=int, default=4096)
    parser.add_argument("--teacher-phase-verifier-format-retries", type=int, default=2)
    parser.add_argument(
        "--teacher-phase-verifier-max-images",
        type=int,
        default=0,
        help="0 sends every O-side screenshot in the current phase to the verifier.",
    )
    parser.add_argument(
        "--teacher-phase-check-interval",
        type=int,
        default=8,
        help="Run a proactive phase check after this many new business actions; 0 disables it.",
    )
    parser.add_argument(
        "--teacher-phase-max-actions",
        type=int,
        default=50,
        help=(
            "Stop an unverified O or direct/shallow S phase after this many "
            "business actions; 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--teacher-deep-search-phase-max-actions",
        type=int,
        default=100,
        help=(
            "Stop an unverified S phase whose component depth is 'deep' after "
            "this many business actions; 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--teacher-phase-no-progress-actions",
        type=int,
        default=3,
        help="Run a phase check after this many explicit failures or exact repeated results.",
    )
    parser.add_argument("--api-timeout", type=float, default=600)
    parser.add_argument("--summary-timeout", type=float, default=600)
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
    parser.add_argument(
        "--image-transport",
        choices=["base64", "url"],
        default=os.environ.get("HYBRID_IMAGE_TRANSPORT", "base64"),
        help=(
            "How local screenshots are sent to multimodal models. 'base64' "
            "inlines them as data URLs (default; works with any local server). "
            "'url' copies each image into --image-publish-dir and sends "
            "<--image-url-base>/<relative path>; serve that directory over "
            "HTTPS yourself (useful for remote endpoints with request-size limits)."
        ),
    )
    parser.add_argument(
        "--image-publish-dir",
        default=os.environ.get("HYBRID_IMAGE_PUBLISH_DIR", ""),
        help="Local immutable image pool used by --image-transport=url.",
    )
    parser.add_argument(
        "--image-url-base",
        default=os.environ.get("HYBRID_IMAGE_URL_BASE", ""),
        help="Public HTTPS URL prefix mapped to --image-publish-dir.",
    )
    parser.add_argument("--use-css-som", dest="use_css_som", action="store_true", default=True)
    parser.add_argument("--no-css-som", dest="use_css_som", action="store_false")
    parser.add_argument("--use-omni-som", action="store_true")
    parser.add_argument("--omni-url", default="")
    parser.add_argument("--keep-user-info", action="store_true")
    parser.add_argument("--no-sandbox", action="store_true", default=True)
    parser.add_argument("--sandboxed", dest="no_sandbox", action="store_false")

    args = parser.parse_args()
    if args.agent_mode not in {"hybrid", "hybrid_routed", "hybrid_guided_routed", "hybrid_react_v2", "operation", "search"}:
        args.agent_mode = "hybrid"
    if args.teacher_phase_control and args.agent_mode != "hybrid_react_v2":
        parser.error("--teacher-phase-control requires --agent-mode=hybrid_react_v2")
    # HYBRID_V2_FLAT=1(S+O 摊平消融,见 hybrid_prompts.v2_flat)只在 v2 scaffold 下有意义,
    # 且与锁侧(HYBRID_V2_LOCK_SIDE)互斥。配错时直接报错,免得跑完 100 题才发现消融没生效。
    if (os.environ.get("HYBRID_V2_FLAT") or "").strip() == "1":
        if args.agent_mode != "hybrid_react_v2":
            parser.error("HYBRID_V2_FLAT=1 requires --agent-mode=hybrid_react_v2")
        if (os.environ.get("HYBRID_V2_LOCK_SIDE") or "").strip():
            parser.error("HYBRID_V2_FLAT=1 cannot be combined with HYBRID_V2_LOCK_SIDE")
        if args.teacher_phase_control:
            parser.error("HYBRID_V2_FLAT=1 cannot be combined with --teacher-phase-control")
        print("[flat] HYBRID_V2_FLAT=1: S+O 全量工具、无 switch、统一提示词", flush=True)

    # Resolve --coord-scale=-1 (the default) by auto-inferring from --model.
    # The COORD_SCALE env var (read by argparse) wins over auto-inference;
    # an explicit non-negative --coord-scale on the CLI wins over both.
    if args.coord_scale is None or args.coord_scale < 0:
        args.coord_scale = coord_scale_for_model(args.model)

    if not args.task and not args.input_path:
        parser.error("Either --task or --input-path must be provided.")

    if args.image_transport == "url":
        if not args.image_publish_dir or not args.image_url_base:
            parser.error(
                "--image-transport=url requires --image-publish-dir and "
                "--image-url-base"
            )
        if not args.image_url_base.lower().startswith("https://"):
            parser.error("--image-url-base must use HTTPS")
    os.environ["HYBRID_IMAGE_TRANSPORT"] = args.image_transport
    if args.image_publish_dir:
        os.environ["HYBRID_IMAGE_PUBLISH_DIR"] = str(
            Path(args.image_publish_dir).expanduser().resolve()
        )
    if args.image_url_base:
        os.environ["HYBRID_IMAGE_URL_BASE"] = args.image_url_base.rstrip("/")
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
    args.tokenizer_path = (args.tokenizer_path
                           or os.environ.get("HYBRID_TOKENIZER_PATH")
                           or os.environ.get("TOKENIZER_PATH")
                           or "yzxjb/HyWe-35B")
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
    args.teacher_phase_verifier_api_key = (
        args.teacher_phase_verifier_api_key
        or os.environ.get(args.teacher_phase_verifier_api_key_env, "")
        or args.router_api_key
    )
    args.teacher_phase_verifier_base_url = (
        args.teacher_phase_verifier_base_url
        or os.environ.get("PHASE_VERIFIER_BASE_URL", "")
        or args.router_base_url
    )
    args.teacher_phase_verifier_model = (
        args.teacher_phase_verifier_model
        or os.environ.get("PHASE_VERIFIER_MODEL", "")
        or args.router_model
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

PRIVATE_TEACHER_FIELD_NAMES = (
    "private_teacher_steps",
    "teacher_steps",
    "standard_solution_steps",
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


def extract_private_teacher_steps(item: dict):
    for name in PRIVATE_TEACHER_FIELD_NAMES:
        value = item.get(name)
        if value:
            return value
    extra = item.get("extra") if isinstance(item.get("extra"), dict) else {}
    for name in PRIVATE_TEACHER_FIELD_NAMES:
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
                if key not in GUIDED_ROUTE_FIELD_NAMES + PRIVATE_TEACHER_FIELD_NAMES
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

        task_web = (
            item.get("start_url")
            or item.get("web")
            or item.get("url")
            or args.web
            or ""
        ).strip()
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
        private_teacher_steps = extract_private_teacher_steps(item)

        specs.append(
            {
                "task_id": task_id,
                "question": question.strip(),
                "web": start_url,
                "task_web": task_web,
                "input_images": input_images,
                "attachment_files": attachment_files,
                "guided_route_plan": guided_route_plan,
                "private_teacher_steps": private_teacher_steps,
                "raw": item,
            }
        )

    return specs


async def run_single_task(args, spec: dict, output_root: Path) -> dict:
    task_dir = output_root / spec["task_id"]
    # --resume:已有可解析的 result.json 就跳过,保护长跑不被重启清空
    if getattr(args, "resume", False):
        done_path = task_dir / "result.json"
        if done_path.exists():
            try:
                cached = json.loads(done_path.read_text(encoding="utf-8"))
                # `records` 为 [] 也不是 None —— 一条 0 步就死的
                # task_error(上游 400 "Failed to download multimodal content"
                # 这类取图失败,重试耗尽)会被当成"已完成"永久跳过。
                # 那是纯基础设施损失,不是任务本身的属性,重启时应该重跑。
                _recs = cached.get("records")
                _infra_dead = (
                    cached.get("termination") == "task_error"
                    and not _recs
                )
                if _recs is not None and not _infra_dead:
                    print(f"[skip] {spec['task_id']} 已完成 (resume)", flush=True)
                    return cached
                if _infra_dead:
                    print(f"[retry] {spec['task_id']} 上次 0 步基础设施失败,重跑", flush=True)
            except Exception:
                pass
    max_attempts = max(1, int(args.max_task_attempts or 1))
    # 两个预算分开计数:
    #   step_cap_budget —— 撞步数上限(max_steps_exceeded)或上下文上限(context_exceeded,
    #     需 HYBRID_CONTEXT_RESTART=1)允许几次。前面几次撞顶的
    #     尝试直接丢掉、整题从头重跑(新浏览器、新上下文、新 seed);只有用到最后一次时才让 agent
    #     在第 max_iter 步强制作答。撞顶后硬逼出来的答案几乎全错(BC-EN 撞 300 步的 53 题 0/53),
    #     重开一次的期望收益远高于收一个编出来的答案。
    #   max_attempts —— 报错 / 格式错等其它非 answer 终态的重试次数,语义与原来一致。
    step_cap_budget = max(1, int(getattr(args, "max_iter_restarts", 1) or 1))
    step_cap_hits = 0
    error_attempts = 0
    attempt_summaries = []
    final_result = None

    attempt = max(0, int(getattr(args, "attempt_offset", 0) or 0))
    failure_experiences: list[str] = []      # retry 保活:历次撞顶的失败后验
    while True:
        attempt += 1
        force_final_at_cap = step_cap_hits >= step_cap_budget - 1
        if task_dir.exists():
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)

        run_args = copy.copy(args)
        run_args.task_id = spec["task_id"]
        run_args.task = spec["question"]
        # Per-task step-budget override (input row field `max_iter`), used by
        # training-set resampling to cap each rollout at the gold trajectory
        # length — longer-than-gold rollouts have no selection value.
        try:
            _mi = int((spec.get("raw") or {}).get("max_iter") or 0)
        except Exception:
            _mi = 0
        if _mi > 0:
            run_args.max_iter = _mi
        run_args.initial_url = spec["web"]
        run_args.web = "" if args.agent_mode == "search" else spec["web"]
        run_args.task_web_context = spec.get("task_web", "")
        run_args.input_images = spec.get("input_images", [])
        run_args.attachment_files = spec.get("attachment_files", [])
        run_args.guided_route_plan = spec.get("guided_route_plan")
        run_args.private_teacher_steps = spec.get("private_teacher_steps")
        run_args.task_dir = str(task_dir)
        # 第几次尝试 —— 参与决策模型的 seed 派生(见 hybrid_agent._next_seed)。
        # 不带这个的话第二次尝试会用同一串 seed 逐字重放第一次,重试白跑。
        run_args.attempt = attempt
        run_args.force_final_at_max_iter = force_final_at_cap
        # retry 保活:历次后验**单独传**,由 agent 拼进渲染用的任务文本。
        # 不能改 run_args.task —— agent 是用 task=spec["question"] 构造的,那个参数
        # 压根不看 run_args.task;而且 result["question"] = self.task 是判分字段,
        # 一旦把增强后的题面写进去,判官看到的就不是原题了。
        run_args.failure_experiences = list(failure_experiences)

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
                "private_teacher_steps_present": bool(spec.get("private_teacher_steps")),
                "teacher_phase_control": bool(args.teacher_phase_control),
                "teacher_phase_verifier_model": args.teacher_phase_verifier_model,
                "teacher_phase_check_interval": args.teacher_phase_check_interval,
                "teacher_phase_max_actions": args.teacher_phase_max_actions,
                "teacher_deep_search_phase_max_actions": (
                    args.teacher_deep_search_phase_max_actions
                ),
                "teacher_phase_no_progress_actions": args.teacher_phase_no_progress_actions,
                "current_date": args.current_date,
                "attempt": attempt,
                "max_task_attempts": max_attempts,
                "max_iter_restarts": step_cap_budget,
                "step_cap_hits_before": step_cap_hits,
                "force_final_at_max_iter": force_final_at_cap,
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
                "usage": (
                    agent._usage_snapshot()
                    if agent is not None and hasattr(agent, "_usage_snapshot")
                    else copy.deepcopy(agent.usage) if agent else {}
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
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
            or spec.get("private_teacher_steps")
            else spec["raw"]
        )
        if spec.get("private_teacher_steps"):
            result["private_teacher_steps_present"] = True
        if args.agent_mode == "hybrid_guided_routed":
            result["guided_route_plan_present"] = bool(spec.get("guided_route_plan"))
        result["attempt"] = attempt
        termination = result.get("termination", "")
        attempt_summaries.append(
            {
                "attempt": attempt,
                "termination": termination,
                "n_steps": len(result.get("records") or []),
                "force_final_at_max_iter": force_final_at_cap,
                "error": (result.get("error") or "")[:500],
                "usage": copy.deepcopy(result.get("usage") or {}),
            }
        )
        final_result = result

        if termination == "answer":
            break
        if termination in ("max_steps_exceeded", "context_exceeded") and not force_final_at_cap:
            # 撞顶且还有重开额度:丢掉这次,整题从头来。
            # 先把这次的失败后验收起来,下一次尝试带上(retry 保活)。
            _fs = (result.get("failure_summary") or "").strip()
            if _fs:
                failure_experiences.append(_fs)
            step_cap_hits += 1
            print(
                f"[step-cap] {spec['task_id']} 第 {step_cap_hits}/{step_cap_budget} 次撞"
                + (f" {run_args.max_iter} 步上限" if termination == "max_steps_exceeded" else "上下文上限")
                + ",整题重跑",
                flush=True,
            )
            continue
        error_attempts += 1
        if error_attempts >= max_attempts:
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
            "raw": (
                scrub_private_guided_fields(spec["raw"])
                if args.agent_mode == "hybrid_guided_routed"
                or spec.get("private_teacher_steps")
                else spec["raw"]
            ),
            "attempt": 0,
        }
        if spec.get("private_teacher_steps"):
            final_result["private_teacher_steps_present"] = True
        if args.agent_mode == "hybrid_guided_routed":
            final_result["guided_route_plan_present"] = bool(
                spec.get("guided_route_plan")
            )

    final_result["attempt_count"] = len(attempt_summaries)
    final_result["attempts"] = attempt_summaries
    final_result["step_cap_restarts"] = step_cap_hits
    final_result["failure_experiences"] = failure_experiences
    final_result["total_steps_all_attempts"] = sum(a.get("n_steps", 0) for a in attempt_summaries)
    all_attempts_usage = {
        "rollout_prompt_tokens": sum(
            int((attempt.get("usage") or {}).get("rollout_prompt_tokens", 0) or 0)
            for attempt in attempt_summaries
        ),
        "rollout_completion_tokens": sum(
            int((attempt.get("usage") or {}).get("rollout_completion_tokens", 0) or 0)
            for attempt in attempt_summaries
        ),
    }
    all_attempts_usage["rollout_total_tokens"] = (
        all_attempts_usage["rollout_prompt_tokens"]
        + all_attempts_usage["rollout_completion_tokens"]
    )
    final_result["all_attempts_token_usage"] = all_attempts_usage
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
    # asyncio's default executor caps at min(32, cpu+4) threads; every tool
    # call (search/visit/sandbox) runs through it via asyncio.to_thread, and a
    # visit can hold a thread for 30-90s (per-page summaries). With >24 workers
    # the 32-thread cap serializes tools and caps end-to-end throughput —
    # size it to the worker count instead.
    from concurrent.futures import ThreadPoolExecutor as _TPE
    asyncio.get_running_loop().set_default_executor(
        _TPE(max_workers=max(64, worker_limit * 3), thread_name_prefix="tools")
    )

    async def run_and_persist(spec: dict):
        async with semaphore:
            try:
                try:
                    result = await run_single_task(
                        args,
                        spec,
                        output_root=output_root,
                    )
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
                            or spec.get("private_teacher_steps")
                            else spec["raw"]
                        ),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if args.agent_mode == "hybrid_guided_routed":
                        result["guided_route_plan_present"] = bool(
                            spec.get("guided_route_plan")
                        )
                    if spec.get("private_teacher_steps"):
                        result["private_teacher_steps_present"] = True
                    save_json(task_dir / "result.json", result)

                async with write_lock:
                    append_jsonl(str(output_path), result)
                    prediction_preview = result.get("prediction", "")
                    if isinstance(prediction_preview, (list, dict)):
                        prediction_preview = str(prediction_preview)
                    else:
                        prediction_preview = str(prediction_preview or "")
                    print(
                        f"[done] {spec['task_id']} termination={result['termination']} "
                        f"attempts={result.get('attempt_count', 1)} "
                        f"prediction={prediction_preview[:120]}"
                    )
            finally:
                pass

    await asyncio.gather(*(run_and_persist(spec) for spec in specs))


if __name__ == "__main__":
    asyncio.run(main())
