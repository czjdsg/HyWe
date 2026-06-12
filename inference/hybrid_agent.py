import asyncio
import base64
import copy
import functools
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from hybrid_prompts import (
    get_allowed_tool_names,
    get_system_prompt,
    get_user_prompt_template,
    normalize_agent_mode,
    normalize_routed_side,
)
from hybrid_code_sandbox import CodeSandboxTool, SUPPORTED_LANGUAGES
from hybrid_runtime import image_path_to_data_url, SelectFallbackRequired
from hybrid_search import SearchTool
from hybrid_scholar import ScholarTool
from hybrid_utils import (
    compact_observation_text,
    extract_tag,
    repair_missing_closing_tags,
    sanitize_surrogates,
    strip_reasoning,
)
from hybrid_visit import VisitTool


_REMOTE_REQUEST_LOCK = threading.Lock()
_REMOTE_REQUEST_LAST_TS: dict[str, float] = {}


# Vendored copy of the Mobile-Agent-v3.5 browser_use SoM / clean_html helpers.
# Override with HYBRID_BROWSER_USE_ROOT if you want to point at an external
# checkout instead of the bundled vendor/ tree.
_DEFAULT_BROWSER_USE_ROOT = (
    Path(__file__).resolve().parent.parent / "vendor" / "mobile_agent_browser_use"
)
MOBILE_AGENT_BROWSER_USE_ROOT = Path(
    os.environ.get("HYBRID_BROWSER_USE_ROOT", str(_DEFAULT_BROWSER_USE_ROOT))
)
if str(MOBILE_AGENT_BROWSER_USE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILE_AGENT_BROWSER_USE_ROOT))

try:
    from browser.playwright.clean_html import SALIENT_ATTRIBUTES, process_element_tag  # noqa: E402
except Exception:
    SALIENT_ATTRIBUTES = (
        "alt",
        "aria-describedby",
        "aria-label",
        "aria-role",
        "aria-controls",
        "label",
        "name",
        "placeholder",
        "readonly",
        "title",
        "value",
        "href",
        "role",
    )

    def process_element_tag(element: str, salient_attributes) -> str:
        opening_tag = re.search(r"<([a-zA-Z0-9:_-]+)([^>]*)>", element or "")
        if not opening_tag:
            return element or ""

        tag_name = opening_tag.group(1)
        attrs_text = opening_tag.group(2)
        attrs = {}
        for key, value_double, value_single, value_plain in re.findall(
            r'([^\s=]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))',
            attrs_text,
        ):
            value = value_double or value_single or value_plain or ""
            if key in salient_attributes:
                attrs[key] = value

        filtered = " ".join(f'{key}="{value}"' for key, value in attrs.items())
        return f"<{tag_name}{(' ' + filtered) if filtered else ''}>"


class HybridBrowserAgent:
    def __init__(self, web, args, task: str):
        self.web = web
        self.args = args
        self.task = task
        # Dedicated thread pool so blocking OpenAI-client calls run off the
        # asyncio event-loop thread (see _acreate_completion_with_retries). One
        # agent issues LLM requests serially within a step; a couple of threads
        # cover the occasional overlap (e.g. summary callback during a step).
        self._llm_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix=f"llm-{id(self)}"
        )
        self.agent_mode = normalize_agent_mode(getattr(args, "agent_mode", "hybrid"))
        # For routed modes: runtime tracks the active side (S or O).
        # hybrid_routed lets the decision model call `switch`; guided routed
        # uses a private router/controller to hard-route without exposing the
        # annotated route plan to the decision-model prompt.
        if self.agent_mode in {"hybrid_routed", "hybrid_guided_routed"}:
            has_initial_url = bool((getattr(args, "web", "") or "").strip())
            self.active_mode = "O" if has_initial_url else "S"
        else:
            self.active_mode = None
        self.switch_count = 0
        self.allowed_tool_names = get_allowed_tool_names(self.agent_mode, self.active_mode)
        request_timeout = float(getattr(args, "api_timeout", 180) or 180)
        summary_timeout = float(getattr(args, "summary_timeout", request_timeout) or request_timeout)
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=request_timeout,
        )
        # Routed hybrid: per-side model/client overrides.
        self._routed_search_model = getattr(args, "routed_search_model", "") or ""
        self._routed_operation_model = getattr(args, "routed_operation_model", "") or ""
        s_base = getattr(args, "routed_search_base_url", "") or ""
        o_base = getattr(args, "routed_operation_base_url", "") or ""
        self._routed_search_client = (
            OpenAI(api_key=args.api_key, base_url=s_base, timeout=request_timeout)
            if s_base and s_base != args.base_url else None
        )
        self._routed_operation_client = (
            OpenAI(api_key=args.api_key, base_url=o_base, timeout=request_timeout)
            if o_base and o_base != args.base_url else None
        )
        self._guided_route_plan = getattr(args, "guided_route_plan", None)
        self._guided_router_state: dict | None = None
        self._guided_current_subgoal = ""
        self._guided_force_answer = False
        # Pre-step routing: the previous decision step's action context, stashed
        # at the END of a step and consumed at the START of the next step so the
        # router refreshes the subgoal AFTER fully digesting the latest
        # observation but BEFORE the decision model acts. None until the first
        # real action has been taken.
        self._guided_pending_action_ctx: dict | None = None
        # Result of the most recent pre-step routing call, copied onto the step
        # record the main loop is about to build (router metadata + any error).
        self._guided_last_route_info: dict | None = None
        # When the pre-step router decides to switch sides, it does NOT flip
        # active_mode itself. Instead it stays on the current (from) side and
        # stashes the pending switch here, so the DECISION MODEL emits a real
        # `switch` tool_call this turn (continuous context) and the runtime
        # intercept flips the side -- same as hybrid_routed. {to, reason} or None.
        self._guided_pending_switch: dict | None = None
        self._router_model = getattr(args, "router_model", "") or args.model
        self._router_client = OpenAI(
            api_key=getattr(args, "router_api_key", "") or args.api_key,
            base_url=getattr(args, "router_base_url", "") or args.base_url,
            timeout=float(getattr(args, "router_timeout", request_timeout) or request_timeout),
        )
        self.summary_client = OpenAI(
            api_key=args.summary_api_key,
            base_url=args.summary_base_url,
            timeout=summary_timeout,
        )
        self.search_tool = SearchTool()
        self.scholar_tool = ScholarTool()
        self.code_sandbox_tool = CodeSandboxTool(
            base_url=getattr(args, "code_sandbox_base_url", ""),
            request_timeout=getattr(args, "code_sandbox_request_timeout", 90),
            # 0 = no truncation; pass through as-is (do NOT `or` to a default
            # since that would resurrect the legacy 6000 cap when 0 is set).
            max_output_chars=getattr(args, "code_sandbox_max_output_chars", 0),
        )
        self.attachment_files = self._normalize_attachment_files(
            getattr(args, "attachment_files", [])
        )
        self._sandbox_attachment_files_cache = None
        self._sandbox_attachment_note_cache = ""
        self.visit_tool = VisitTool(
            summarize_fn=self._summarize_visit_content,
            trace_callback=self._record_summary_trace,
            jina_api_key=getattr(args, "jina_api_key", ""),
            tokenizer_path=getattr(args, "summary_tokenizer_path", ""),
            max_webpage_tokens=getattr(args, "visit_max_webpage_tokens", 32000),
            max_webpage_chars=getattr(args, "visit_max_webpage_chars", 0),
            max_images=getattr(args, "visit_max_images", 4),
            max_image_url_chars=getattr(args, "visit_max_image_url_chars", 2_000_000),
        )

        self.last_report = "No previous report."
        self.last_observation = "No previous observation."
        self.last_visit_screenshot_path = ""
        self.last_visit_screenshot_url = ""
        self.records = []
        self.guided_router_trace = []
        self.current_summary_traces = []
        self.current_visit_screenshot_path = ""
        self.current_visit_screenshot_url = ""
        self.usage = {
            "agent_prompt_tokens": 0,
            "agent_completion_tokens": 0,
            "summary_prompt_tokens": 0,
            "summary_completion_tokens": 0,
            "router_prompt_tokens": 0,
            "router_completion_tokens": 0,
        }

    def _client_base_url(self, client) -> str:
        value = getattr(client, "base_url", "") or ""
        return str(value).rstrip("/")

    def _is_local_base_url(self, base_url: str) -> bool:
        return (
            "localhost" in base_url
            or "127.0.0.1" in base_url
            or base_url.startswith("http://[::1]")
            or base_url.startswith("https://[::1]")
        )

    def _respect_remote_request_interval(self, client, min_interval: float = 1.5) -> None:
        base_url = self._client_base_url(client)
        if not base_url or self._is_local_base_url(base_url):
            return

        with _REMOTE_REQUEST_LOCK:
            now = time.time()
            last_ts = _REMOTE_REQUEST_LAST_TS.get(base_url, 0.0)
            wait_time = max(0.0, min_interval - (now - last_ts))
            if wait_time > 0:
                time.sleep(wait_time)
            _REMOTE_REQUEST_LAST_TS[base_url] = time.time()

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        if isinstance(exc, RateLimitError):
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        text = str(exc).lower()
        return (
            "limit_burst_rate" in text
            or "rate limit" in text
            or "ratelimiterror" in text
            or "error code: 429" in text
            or '"code": 429' in text
        )

    def _is_transient_request_error(self, exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError)):
            return True

        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if status_code in {408, 409}:
                return True
            if isinstance(status_code, int) and status_code >= 500:
                return True

        text = str(exc).lower()
        transient_markers = (
            "connection error",
            "connection reset",
            "connection aborted",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "server disconnected",
            "remote protocol error",
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
        )
        return any(marker in text for marker in transient_markers)

    def _create_completion_with_retries(
        self,
        client,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_body: dict,
        min_interval: float = 1.5,
        max_attempts: int = 6,
    ):
        active_messages = sanitize_surrogates(copy.deepcopy(messages))
        image_retry_used = False
        last_exc = None

        # Vertex's claude-* endpoints reject having both `temperature` and `top_p`
        # set simultaneously. When the active model is vertex claude, drop top_p
        # and keep only temperature.
        is_vertex_claude = isinstance(model, str) and model.startswith("vertex_ai.claude")
        for attempt in range(max_attempts):
            self._respect_remote_request_interval(client, min_interval=min_interval)
            try:
                req_kwargs = dict(
                    model=model,
                    messages=active_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=extra_body,
                )
                if not is_vertex_claude:
                    req_kwargs["top_p"] = top_p
                return client.chat.completions.create(**req_kwargs)
            except Exception as exc:
                last_exc = exc
                if (not image_retry_used) and self._should_retry_without_images(exc, active_messages):
                    active_messages = self._strip_images_from_messages(active_messages)
                    image_retry_used = True
                    continue
                if self._is_rate_limit_error(exc) and attempt < max_attempts - 1:
                    time.sleep(min(20.0, 1.5 * (2**attempt)))
                    continue
                if self._is_transient_request_error(exc) and attempt < max_attempts - 1:
                    time.sleep(min(12.0, 1.0 * (2**attempt)))
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM request failed without an exception.")

    def close(self) -> None:
        """Release the LLM thread pool. Safe to call multiple times."""
        executor = getattr(self, "_llm_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
            self._llm_executor = None

    async def _acreate_completion_with_retries(self, *args, **kwargs):
        """Async wrapper around the SYNC LLM call.

        The OpenAI client uses a blocking httpx transport. Calling it directly
        from the async run() loop freezes the whole event loop, so a single
        request that hangs on response headers (e.g. :8002 vision inference
        choking on a high-entropy screenshot) starves every other concurrent
        task. Running the sync helper in a thread-pool executor keeps the event
        loop free: while one task's request blocks in its own thread, the loop
        can still drive the other tasks' playwright awaits and LLM dispatches.
        The retry loop + sleeps live inside the sync helper, so they run in the
        worker thread too and never block the loop.
        """
        loop = asyncio.get_running_loop()
        func = functools.partial(
            self._create_completion_with_retries, *args, **kwargs
        )
        return await loop.run_in_executor(self._llm_executor, func)

    def _has_explicit_initial_page(self) -> bool:
        return bool((self.args.web or "").strip())

    def _has_usable_browser_page(self, state: dict | None) -> bool:
        if not state:
            return False
        current_url = (state.get("current_url") or "").strip()
        if not current_url or current_url == "about:blank":
            return False
        return True

    def _format_recent_actions(self) -> str:
        action_records = [record for record in self.records if record.get("tool_call")][-10:]
        if not action_records:
            return "No previous tool actions."

        # 0 / None / negative => no truncation. Don't use `or` — it would
        # turn an explicit 0 into the legacy 8000 fallback.
        raw_history = getattr(self.args, "max_action_history_chars", 0)
        max_history_chars = int(raw_history) if raw_history is not None else 0
        entries = []
        for record in action_records:
            step_id = record.get("step", "?")
            # Routed mode: render switch steps distinctly so the model can see
            # its own path-switch history in Recent Actions.
            if record.get("is_switch_step"):
                sf = record.get("switch_from", "?")
                st = record.get("switch_to", "?")
                reason = (record.get("switch_reason") or "").strip()
                entries.append(f"Step {step_id}: switch {sf}->{st} — reason: {reason}")
                continue
            action = json.dumps(record.get("tool_call", {}), ensure_ascii=False)
            description = (record.get("action_description") or "").strip()
            # active_mode_before is only set in routed mode; include the side
            # tag so the model can tell which path the action ran under.
            mode_before = record.get("active_mode_before")
            tag = f", {mode_before}" if mode_before in ("S", "O") else ""
            if description:
                entries.append(f"Step {step_id}{tag}: {action} — {description}")
            else:
                entries.append(f"Step {step_id}{tag}: {action}")

        return compact_observation_text(
            "\n\n".join(entries),
            max_chars=max_history_chars,
        )

    def _hides_browser_context(self) -> bool:
        """Modes that should suppress browser screenshot / tabs / state."""
        if self.agent_mode == "search":
            return True
        if self.agent_mode in {"hybrid_routed", "hybrid_guided_routed"} and self.active_mode == "S":
            return True
        return False

    def _format_state_tabs_text(self, state: dict | None) -> str:
        if self._hides_browser_context():
            return ""
        tabs = (state or {}).get("tabs") or []
        if not tabs and self.web is not None:
            tabs = self.web.tabs_snapshot()
        if not tabs:
            return ""
        if len(tabs) == 1:
            return "- Tabs: 1 open"
        lines = [f"- Tabs: {len(tabs)} open"]
        for tab in tabs:
            marker = " (active)" if tab.get("active") else ""
            url = (tab.get("url", "") or "")[:120]
            title = (tab.get("title", "") or "").strip()
            suffix = f"  — {title[:80]}" if title else ""
            lines.append(f"    [{tab.get('index')}]{marker} {url}{suffix}")
        return "\n".join(lines)

    def _build_prompt_text(self, state: dict | None, is_final: bool) -> str:
        task_web = str(getattr(self.args, "task_web_context", getattr(self.args, "web", "")) or "").strip()
        task_with_web = self.task if not task_web else f"{self.task}\nWeb: {task_web}"
        # In modes that hide browser context (search; routed S-side), do not
        # leak current_url/title into the prompt payload so the templates that
        # reference them cannot accidentally surface stale state.
        hides_browser = self._hides_browser_context()
        payload = {
            "current_date": self.args.current_date,
            "task": self.task,
            "task_with_web": task_with_web,
            "initial_url": (
                ""
                if hides_browser
                else getattr(self.args, "initial_url", getattr(self.args, "web", ""))
            ),
            "current_url": state.get("current_url", "") if state else "",
            "current_title": state.get("current_title", "") if state else "",
            "viewport_width": int(getattr(self.args, "window_width", 1080)),
            "viewport_height": int(getattr(self.args, "window_height", 1440)),
            "last_report": self.last_report,
            "last_observation": compact_observation_text(
                self.last_observation,
                max_chars=self.args.max_observation_chars,
            ),
            "recent_actions": self._format_recent_actions(),
            "tabs_text": self._format_state_tabs_text(state),
        }
        has_browser_page = self._has_usable_browser_page(state)

        if self.agent_mode in {"hybrid_routed", "hybrid_guided_routed"}:
            # Routed modes: pick the per-side user prompt template. S-side
            # reuses SEARCH_* (no browser context); O-side reuses OPERATION_*.
            if not self.records:
                prompt_kind = "initial"
            elif is_final:
                prompt_kind = "final"
            else:
                prompt_kind = "turn"
            text = get_user_prompt_template(
                self.agent_mode, prompt_kind, self.active_mode
            ).format(**payload)
            if self.agent_mode == "hybrid_guided_routed":
                text = self._inject_guided_subgoal_block(text, is_final=is_final)
            return sanitize_surrogates(text)

        if self.agent_mode == "operation":
            if not self.records:
                prompt_kind = "initial"
            elif is_final:
                prompt_kind = "final"
            else:
                prompt_kind = "turn"
        elif self.agent_mode == "search":
            if not self.records:
                prompt_kind = "initial_no_browser"
            elif is_final:
                prompt_kind = "final_no_browser"
            else:
                prompt_kind = "turn_no_browser"
        else:
            if not self.records:
                prompt_kind = "initial" if has_browser_page else "initial_no_browser"
            elif is_final:
                prompt_kind = "final" if has_browser_page else "final_no_browser"
            else:
                prompt_kind = "turn" if has_browser_page else "turn_no_browser"

        text = get_user_prompt_template(self.agent_mode, prompt_kind).format(**payload)
        return sanitize_surrogates(text)

    def _inject_guided_subgoal_block(self, prompt_text: str, *, is_final: bool) -> str:
        if self._guided_force_answer or is_final:
            block = (
                "## Guided Routing State\n"
                "The private router has marked the guided subgoals complete, "
                "or the step budget is exhausted. Produce the final answer from "
                "the observed cumulative_progress. Do not include any field for "
                "remaining or future subgoals."
            )
        else:
            current = (self._guided_current_subgoal or "").strip()
            if not current:
                current = "Continue the current routed subgoal."
            # The Current Guided Subgoal is the model's view of what to do
            # next, but the report fields (task_analysis / this_turn /
            # next_action_rationale / action_description) must read as the
            # MODEL's own reasoning grounded in the task and what was just
            # observed — never as "I was told to do X" / "the subgoal says".
            # The persisted trajectory is training data; if the model parrots
            # routing vocabulary into its reasoning, the next turn's context
            # and downstream training both learn to lean on that crutch.
            block = (
                "## Current Guided Subgoal\n"
                f"{current}\n\n"
                "## Guided Routing Rules\n"
                "- Work only on the current guided subgoal above.\n"
                "- Do not create, reveal, or speculate about future subgoals.\n"
                "- Do not include any field for remaining or future subgoals; "
                "the runtime will record the active current subgoal in the saved report.\n"
                "- INTERNALIZE the subgoal — never cite it as the source of your "
                "decision. In task_analysis, this_turn, next_action_rationale, "
                "and action_description, justify what you do by referring to "
                "the user task and the observed page/search state, in your own "
                "words. Avoid phrases like 'the current subgoal', 'the subgoal "
                "instructs/requires/tells me to', 'guided routing has directed', "
                "'hand off to the search/operation-based path', or any mention "
                "of subgoal/router/routing/plan machinery. If a hand-off is in "
                "order, say it as a personal choice grounded in evidence "
                "('I'll search externally because the live page does not show "
                "this'), never as a directive received."
            )
        marker = "\n\n## Required Output Format"
        if marker in prompt_text:
            return prompt_text.replace(marker, "\n\n" + block + marker, 1)
        return prompt_text + "\n\n" + block

    def _guided_expose_switch(self) -> bool:
        """True only on the turn the guided router has decided to hand off, so
        the system prompt + tool schema expose `switch` for the decision model.
        """
        return (
            self.agent_mode == "hybrid_guided_routed"
            and self._guided_pending_switch is not None
        )

    def _build_messages(self, state: dict | None, prompt_text: str) -> list[dict]:
        sys_text = get_system_prompt(
            self.agent_mode, self.active_mode,
            expose_switch=self._guided_expose_switch(),
        )
        # MODEL-AWARE coordinate convention. Append ONLY when the active
        # model is kimi-k2.x — its training mixes 0-1 floats and 0-1000 ints,
        # so we explicitly anchor it to relative 0-1. Other VLMs (Claude
        # computer-use = raw px, Qwen-VL = 0-1000) use their native convention
        # via the runtime's coord_scale and don't need a prompt hint.
        model_lower = (str(getattr(self.args, "model", "")) or "").lower()
        if "kimi" in model_lower:
            sys_text = sys_text + (
                "\n\nCOORDINATE CONVENTION — kimi-k2 family only:\n"
                "Output (x, y) as relative floats in [0.0, 1.0]: (0,0) is the "
                "top-left corner of the viewport, (1,1) is the bottom-right. "
                "Example: x=0.336, y=0.601 means 33.6% across, 60.1% down. "
                "Use 2-3 decimal places. Do NOT output integer pixel values."
            )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_text}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            },
        ]

        hides_browser = self._hides_browser_context()
        use_agent_images = (
            bool(getattr(self.args, "use_agent_images", True))
            and not hides_browser
        )
        use_visit_images = (
            bool(getattr(self.args, "use_visit_images", True))
            and not hides_browser
        )
        use_input_images = bool(getattr(self.args, "use_input_images", True))

        if use_input_images:
            max_input_images = max(0, int(getattr(self.args, "max_input_images", 8) or 0))
            for index, image_item in enumerate(
                self._format_input_images()[:max_input_images],
                start=1,
            ):
                messages[1]["content"].append(
                    {
                        "type": "text",
                        "text": f"Task input image {index}:",
                    }
                )
                messages[1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_item},
                    }
                )

        if (
            use_agent_images
            and self._has_usable_browser_page(state)
            and state
            and state.get("img_path")
        ):
            messages[1]["content"].append(
                {
                    "type": "text",
                    "text": "Current browser screenshot:",
                }
            )
            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(state["img_path"])},
                }
            )

        if (
            use_visit_images
            and self.last_visit_screenshot_path
            and Path(self.last_visit_screenshot_path).exists()
        ):
            messages[1]["content"].append(
                {
                    "type": "text",
                    "text": (
                        "Attached screenshot for the previous visit observation "
                        f"({self.last_visit_screenshot_url or 'visited URL'}):"
                    ),
                }
            )
            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(self.last_visit_screenshot_path)},
                }
            )

        return messages

    def _format_input_images(self) -> list[str]:
        image_urls = []
        for item in getattr(self.args, "input_images", []) or []:
            if isinstance(item, str):
                value = item.strip()
                if not value:
                    continue
                if value.startswith("data:image/"):
                    image_urls.append(value)
                    continue
                path = Path(value)
                if path.exists():
                    image_urls.append(image_path_to_data_url(str(path)))
                continue

            if not isinstance(item, dict):
                continue
            data_url = str(item.get("data_url") or item.get("url") or "").strip()
            if data_url.startswith("data:image/"):
                image_urls.append(data_url)
                continue
            path_value = str(item.get("path") or "").strip()
            if path_value and Path(path_value).exists():
                image_urls.append(image_path_to_data_url(path_value))

        return image_urls

    def _normalize_attachment_files(self, raw_files) -> list[dict]:
        if isinstance(raw_files, dict):
            raw_files = [raw_files]
        if not isinstance(raw_files, list):
            return []

        normalized = []
        seen = set()
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            path_value = str(item.get("path") or item.get("local_path") or "").strip()
            if not path_value:
                continue
            path = Path(path_value)
            sandbox_path = str(
                item.get("sandbox_path") or item.get("file_name") or path.name
            ).strip()
            sandbox_path = Path(sandbox_path).name
            if not sandbox_path:
                sandbox_path = path.name or "attachment"
            key = (str(path), sandbox_path)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "path": str(path),
                    "sandbox_path": sandbox_path,
                    "file_name": str(item.get("file_name") or path.name),
                }
            )
        return normalized

    def _sandbox_attachment_files(self) -> tuple[dict[str, str], str]:
        if self._sandbox_attachment_files_cache is not None:
            return self._sandbox_attachment_files_cache, self._sandbox_attachment_note_cache

        max_file_bytes = int(
            getattr(self.args, "code_sandbox_max_file_bytes", 25_000_000) or 0
        )
        files = {}
        injected = []
        skipped = []
        for item in self.attachment_files:
            path = Path(item.get("path", ""))
            sandbox_path = str(item.get("sandbox_path") or path.name).strip()
            if not sandbox_path:
                continue
            if not path.exists() or not path.is_file():
                skipped.append(f"{sandbox_path} (missing local file: {path})")
                continue
            size = path.stat().st_size
            if max_file_bytes > 0 and size > max_file_bytes:
                skipped.append(
                    f"{sandbox_path} ({size} bytes exceeds max {max_file_bytes})"
                )
                continue
            try:
                files[sandbox_path] = base64.b64encode(path.read_bytes()).decode("utf-8")
                injected.append(sandbox_path)
            except Exception as exc:
                skipped.append(f"{sandbox_path} ({type(exc).__name__}: {exc})")

        notes = []
        if injected:
            notes.append(
                "Sandbox attachment files injected: " + ", ".join(sorted(injected))
            )
        if skipped:
            notes.append("Sandbox attachment files skipped: " + "; ".join(skipped))

        self._sandbox_attachment_files_cache = files
        self._sandbox_attachment_note_cache = "\n".join(notes)
        return files, self._sandbox_attachment_note_cache

    def _active_client_and_model(self) -> tuple:
        """Return (client, model) for the current turn based on active_mode."""
        if self.agent_mode in {"hybrid_routed", "hybrid_guided_routed"} and self.active_mode == "S" and self._routed_search_model:
            client = self._routed_search_client or self.client
            return client, self._routed_search_model
        if self.agent_mode in {"hybrid_routed", "hybrid_guided_routed"} and self.active_mode == "O" and self._routed_operation_model:
            client = self._routed_operation_client or self.client
            return client, self._routed_operation_model
        return self.client, self.args.model

    async def _achat(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> str:
        extra_body = {
            "repetition_penalty": self.args.repetition_penalty,
            "top_k": self.args.top_k,
            "seed": self.args.seed,
        }
        client, model = self._active_client_and_model()
        response = await self._acreate_completion_with_retries(
            client,
            model,
            messages,
            max_tokens=max_tokens or self.args.max_tokens,
            temperature=self.args.temperature if temperature is None else temperature,
            top_p=self.args.top_p,
            extra_body=extra_body,
        )

        if getattr(response, "usage", None):
            self.usage["agent_prompt_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
            self.usage["agent_completion_tokens"] += (
                getattr(response.usage, "completion_tokens", 0) or 0
            )

        message = response.choices[0].message
        content = message.content

        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )

        return content or ""

    def _summary_chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        extra_body = {
            "repetition_penalty": self.args.repetition_penalty,
            "top_k": self.args.top_k,
            "seed": self.args.seed,
        }
        response = self._create_completion_with_retries(
            self.summary_client,
            self.args.summary_model,
            messages,
            max_tokens=max_tokens or self.args.visit_summary_max_tokens,
            temperature=temperature,
            top_p=self.args.top_p,
            extra_body=extra_body,
            min_interval=0.0,
        )

        if getattr(response, "usage", None):
            self.usage["summary_prompt_tokens"] += (
                getattr(response.usage, "prompt_tokens", 0) or 0
            )
            self.usage["summary_completion_tokens"] += (
                getattr(response.usage, "completion_tokens", 0) or 0
            )

        message = response.choices[0].message
        content = message.content
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return content or ""

    def _guided_plan_text(self) -> str:
        plan = self._guided_route_plan
        if isinstance(plan, str):
            return plan.strip()
        try:
            return json.dumps(plan, ensure_ascii=False, indent=2)
        except Exception:
            return str(plan)

    async def _guided_router_chat(self, messages: list[dict]) -> str:
        response = await self._acreate_completion_with_retries(
            self._router_client,
            self._router_model,
            messages,
            max_tokens=int(getattr(self.args, "router_max_tokens", 4096) or 4096),
            temperature=0.0,
            top_p=1.0,
            extra_body={
                "repetition_penalty": self.args.repetition_penalty,
                "top_k": self.args.top_k,
                "seed": self.args.seed,
            },
            min_interval=0.0,
        )
        if getattr(response, "usage", None):
            self.usage["router_prompt_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
            self.usage["router_completion_tokens"] += (
                getattr(response.usage, "completion_tokens", 0) or 0
            )
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return content or ""

    def _record_guided_router_trace(
        self,
        *,
        kind: str,
        messages: list[dict],
        raw_response: str,
        parsed_state: dict | None = None,
        error: str = "",
        related_step: int | None = None,
    ) -> int | None:
        if self.agent_mode != "hybrid_guided_routed":
            return None
        event_id = len(self.guided_router_trace) + 1
        event = {
            "event_id": event_id,
            "kind": kind,
            "related_step": related_step,
            "model": self._router_model,
            "base_url": self._client_base_url(self._router_client),
            "messages": sanitize_surrogates(copy.deepcopy(messages)),
            "raw_response": sanitize_surrogates(raw_response or ""),
            "parsed_state": sanitize_surrogates(copy.deepcopy(parsed_state or {})),
            "error": error,
            "created_at": int(time.time()),
        }
        self.guided_router_trace.append(event)
        self._write_guided_router_trace_checkpoint()
        return event_id

    def _atomic_write_json(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(sanitize_surrogates(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    def _write_guided_router_trace_checkpoint(self) -> None:
        if self.agent_mode != "hybrid_guided_routed" or not self.guided_router_trace:
            return
        task_dir = getattr(self.args, "task_dir", "")
        if not task_dir:
            return
        try:
            self._atomic_write_json(
                Path(task_dir) / "guided_router_trace.json",
                {
                    "task_id": getattr(self.args, "task_id", Path(task_dir).name),
                    "warning": (
                        "Private debug sidecar. Contains hidden guided-route/controller "
                        "inputs and must not be used as training trajectory data."
                    ),
                    "events": copy.deepcopy(self.guided_router_trace),
                },
            )
        except Exception:
            return

    def _write_live_result_checkpoint(
        self,
        *,
        termination: str = "running",
        prediction: str = "",
    ) -> None:
        if self.agent_mode != "hybrid_guided_routed":
            return
        task_dir = getattr(self.args, "task_dir", "")
        if not task_dir:
            return
        try:
            self._atomic_write_json(
                Path(task_dir) / "result.json",
                {
                    "question": self.task,
                    "prediction": prediction,
                    "termination": termination,
                    "records": copy.deepcopy(self.records),
                    "usage": copy.deepcopy(self.usage),
                    "task_id": getattr(self.args, "task_id", Path(task_dir).name),
                    "agent_mode": self.agent_mode,
                    "web": getattr(self.args, "web", ""),
                    "guided_route_plan_present": bool(self._guided_route_plan),
                    "running_checkpoint": termination == "running",
                },
            )
        except Exception:
            return

    @staticmethod
    def _coerce_routed_side(value, default: str | None = None) -> str | None:
        s = str(value or "").strip().upper()
        if s in {"SEARCH", "SEARCH-BASED", "SEARCH_BASED", "S"}:
            return "S"
        if s in {"OPERATION", "OPERATION-BASED", "OPERATION_BASED", "O"}:
            return "O"
        return default

    def _normalize_subgoal_item(self, item, prev_mode: str | None) -> dict | None:
        """Normalize one remaining_subgoals entry into a structured dict.

        Accepts the new structured form:
          {"type":"task","mode":"S","subgoal":"..."}
          {"type":"switch","to":"O","reason":"..."}
        and (back-compat) a bare string -> treated as a task on prev_mode.
        Returns None for empty/invalid entries (dropped by caller).
        """
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return {"type": "task", "mode": prev_mode, "subgoal": text}
        if not isinstance(item, dict):
            return None
        itype = str(item.get("type") or "").strip().lower()
        if itype == "switch" or (item.get("to") and not item.get("subgoal")):
            to = self._coerce_routed_side(item.get("to"))
            if to is None:
                return None
            return {
                "type": "switch",
                "to": to,
                "reason": str(item.get("reason") or item.get("switch_reason") or "").strip(),
            }
        subgoal = str(item.get("subgoal") or item.get("goal") or "").strip()
        if not subgoal:
            return None
        return {
            "type": "task",
            "mode": self._coerce_routed_side(item.get("mode"), prev_mode),
            "subgoal": subgoal,
        }

    def _parse_guided_router_response(self, raw_text: str, *, prev_mode: str | None = None) -> dict:
        text = self._strip_outer_json_fence(strip_reasoning(raw_text))
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("router response must be a JSON object")

        done = bool(obj.get("done", False))

        raw_remaining = obj.get("remaining_subgoals")
        if not isinstance(raw_remaining, list):
            raise ValueError("router remaining_subgoals must be a list")

        # Normalize each entry into a structured dict, threading the running mode
        # so a bare-string or mode-less task entry inherits the side implied by
        # the most recent preceding task / switch target.
        remaining: list[dict] = []
        running_mode = self._coerce_routed_side(obj.get("active_mode"), prev_mode) or prev_mode
        for raw_item in raw_remaining:
            norm = self._normalize_subgoal_item(raw_item, running_mode)
            if norm is None:
                continue
            if norm["type"] == "task":
                if norm.get("mode") in ("S", "O"):
                    running_mode = norm["mode"]
                else:
                    norm["mode"] = running_mode
                remaining.append(norm)
            else:  # switch
                remaining.append(norm)
                running_mode = norm["to"]  # entries after a switch are on the new side

        if done:
            remaining = []
        if not remaining and not done:
            raise ValueError("router remaining_subgoals cannot be empty unless done=true")

        # Derive live fields (active_mode / current_subgoal / switch info) from
        # the head of the structured remaining list. Shared rule with switch
        # consumption via _derive_state_from_remaining.
        derived = self._derive_state_from_remaining(
            remaining, self._coerce_routed_side(obj.get("active_mode"), prev_mode)
        )
        # Optional per-step quality verdict (present on UPDATE calls only; the
        # init call has no prior action to judge). Normalize to a small dict or
        # None — never raise on a malformed verdict, it is advisory metadata.
        quality = obj.get("last_step_quality")
        if isinstance(quality, dict):
            verdict = str(quality.get("verdict") or "").strip().lower()
            quality = {
                "verdict": verdict if verdict in ("good", "acceptable", "poor") else "unknown",
                "reason": str(quality.get("reason") or "").strip()[:300],
            }
        else:
            quality = None

        return {
            "active_mode": derived["active_mode"],
            "remaining_subgoals": remaining,
            "current_subgoal": derived["current_subgoal"],
            "current_is_switch": derived["current_is_switch"],
            "current_switch_to": derived["current_switch_to"],
            "current_switch_reason": derived["current_switch_reason"],
            "completed_subgoals": [
                item.strip()
                for item in (obj.get("completed_subgoals") or [])
                if isinstance(item, str) and item.strip()
            ],
            "done": done,
            "last_step_quality": quality,
        }

    # Shared contract shown to the router on BOTH the first and later calls:
    # role, no-leak rule, and the structured remaining_subgoals schema in which
    # `switch` is itself a list entry (not a separate flag).
    _GUIDED_ROUTER_COMMON = (
        "You are a private routing controller for a guided web-agent rollout. "
        "You receive a human-annotated standard route plan, but that plan is "
        "private and must never be leaked into the public decision trajectory. "
        "You do not generate browser/search actions; you only maintain the "
        "hidden plan as an ordered `remaining_subgoals` list and let the decision "
        "model act on its head entry. Output exactly one JSON object.\n"
        "remaining_subgoals is a list of STRUCTURED entries of two kinds:\n"
        "  - a task entry: {\"type\":\"task\",\"mode\":\"S\"|\"O\",\"subgoal\":\"...\"}\n"
        "    (mode = S for search-based tools, O for browser-operation tools)\n"
        "  - a switch entry: {\"type\":\"switch\",\"to\":\"S\"|\"O\",\"reason\":\"...\"}\n"
        "A `switch` entry is a first-class step that hands control to the other "
        "path; place one wherever the next task entry runs on a different mode "
        "than the previous task entry. The list therefore alternates like "
        "task(S)..., switch->O, task(O)..., and the head entry's mode/switch is "
        "what the decision model will do next. There is NO separate should_switch "
        "flag -- a hand-off is expressed solely by a switch entry at the head.\n"
        "Never invent or pre-bind concrete event names, venues, products, dates, "
        "URLs, or other entities for a later phase from the standard route alone; "
        "keep later subgoals as placeholders (e.g. 'the venues found in search') "
        "until those values actually appear in an observation. Never mention gold "
        "routes, annotations, labels, phases, or any hidden guidance in `reason` "
        "strings -- keep them local and observation-grounded."
    )

    def _guided_router_system_message_init(self) -> str:
        """System prompt for the FIRST router call (cold-start decomposition)."""
        return (
            self._GUIDED_ROUTER_COMMON + "\n\n"
            "This is the FIRST call. There is no observation yet. The "
            "standard_route_plan is a typed list of `task` and `switch` "
            "entries — preserve ALL switch entries exactly where they appear "
            "(including a LEADING switch at position 0 if present: it is "
            "intentional and means the agent must hand off on its very first "
            "action). You may refine each `task` entry into multiple concrete, "
            "ordered, observation-checkable task entries on the SAME mode, but "
            "you may NOT add, drop, move, or invent switch entries — the S<->O "
            "boundaries are fixed by the standard route. Do not create a "
            "separate final-answer subgoal; after all subgoals are done the "
            "decision model emits the answer from whichever side is active."
        )

    def _guided_router_system_message_update(self) -> str:
        """System prompt for LATER router calls (observation-driven update)."""
        return (
            self._GUIDED_ROUTER_COMMON + "\n\n"
            "This is an UPDATE call: you just observed the result of the last "
            "decision action. Update remaining_subgoals from that observation. If "
            "the head task entry is now complete, remove it (record it in "
            "completed_subgoals) and expose the next entry. browser_state is "
            "provided only after O-side actions (the post-action page state); if "
            "absent, do not infer browser state. Bind any placeholders in later "
            "subgoals only from concrete values actually observed.\n"
            "Completion judgement: judge a subgoal by whether enough VALID "
            "evidence has been gathered to satisfy it, NOT by whether every "
            "attempted source succeeded. A single action may probe multiple "
            "candidates/URLs where some fail (e.g. a 403/blocked page) while "
            "others yield valid results. If a subgoal asks for N items and N valid "
            "items now have the required fields, mark it complete and move on (or "
            "expose the switch entry) -- do NOT keep the agent on the same path "
            "just to replace a candidate that failed when an equally valid "
            "alternative is already confirmed. A failed candidate keeps the "
            "subgoal open only if, without it, fewer than N valid items remain. "
            "Cross-check last_decision_report/cumulative_progress: if it already "
            "lists enough confirmed items, prefer advancing over re-searching."
        )

    async def _guided_initialize_router(self) -> None:
        if self.agent_mode != "hybrid_guided_routed":
            return
        if self._guided_router_state is not None:
            return
        if not self._guided_route_plan:
            raise ValueError(
                "hybrid_guided_routed requires a guided route plan for each task "
                "(input field route_plan/guided_route_plan or --guided-route-plan-path)."
            )
        payload = {
            "task": self.task,
            "initial_url": getattr(self.args, "initial_url", getattr(self.args, "web", "")),
            "standard_route_plan": self._guided_route_plan,
            "instructions": (
                "Initialize the structured remaining_subgoals from the standard "
                "route. The standard route is already a typed list of `task` and "
                "`switch` entries.\n"
                "STRICT — SWITCH ENTRIES ARE GROUND TRUTH: preserve EVERY switch "
                "entry exactly where it appears, including a LEADING switch at "
                "position 0 if the standard route starts with one. Do NOT drop, "
                "move, merge, or invent switch entries, and do NOT 'fix' a "
                "leading switch — if the route begins with a switch, the agent's "
                "very first action MUST be that switch (the runtime starts on "
                "the side implied by initial_url, and the leading switch hands "
                "off to the other side before any work). The number and order of "
                "S<->O boundaries must match the standard route's switch entries "
                "one-for-one.\n"
                "TASK ENTRIES — you may refine: decompose any single `task` "
                "entry into several concrete, ordered, observation-checkable "
                "task entries on the SAME mode (do not let refinement create or "
                "remove a mode boundary — that is the switch entries' job). "
                "Write later subgoals that depend on not-yet-discovered values "
                "with placeholders (e.g. 'the venues found in search'); do not "
                "pre-bind concrete entities. Do not create a separate "
                "final-answer subgoal."
            ),
            "required_output": {
                "remaining_subgoals": [
                    {"type": "task", "mode": "S or O", "subgoal": "concrete subgoal text"},
                    {"type": "switch", "to": "S or O", "reason": "why hand off to the other path"},
                ],
                "completed_subgoals": [],
                "done": False,
            },
        }
        messages = [
            {"role": "system", "content": self._guided_router_system_message_init()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        raw = ""
        try:
            raw = await self._guided_router_chat(messages)
            state = self._parse_guided_router_response(raw)
        except Exception as exc:
            self._record_guided_router_trace(
                kind="initialize",
                messages=messages,
                raw_response=raw,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        # SHAPE-1 (mismatched-start) tasks ship a LEADING switch on purpose:
        # the runtime starts on the side implied by initial_url, and the very
        # first action must be that switch to hand off to the side where the
        # work lives. So we KEEP a leading switch — _guided_route_before_step
        # surfaces it as the agent's first action. (Previously this code dropped
        # leading switches, which silently erased the SHAPE-1 training signal.)
        remaining = list(state.get("remaining_subgoals") or [])
        state["remaining_subgoals"] = remaining
        state.update(self._derive_state_from_remaining(remaining, state.get("active_mode")))
        self._record_guided_router_trace(
            kind="initialize",
            messages=messages,
            raw_response=raw,
            parsed_state=state,
        )
        self._guided_apply_router_state(state)

    @staticmethod
    def _derive_state_from_remaining(remaining: list[dict], prev_mode: str | None) -> dict:
        """Derive {active_mode, current_subgoal, current_is_switch,
        current_switch_to, current_switch_reason} from the head of a structured
        remaining_subgoals list. Shared by the parser and switch consumption so
        the derivation rule lives in one place.
        """
        current_item = remaining[0] if remaining else None
        if current_item and current_item.get("type") == "switch":
            switch_to = current_item["to"]
            switch_from = "O" if switch_to == "S" else "S"
            reason = current_item.get("reason") or ""
            to_name = "search-based" if switch_to == "S" else "operation-based"
            return {
                "active_mode": switch_from,
                "current_subgoal": f"Hand off to the {to_name} path now: call `switch`. {reason}".strip(),
                "current_is_switch": True,
                "current_switch_to": switch_to,
                "current_switch_reason": reason,
            }
        if current_item is not None:
            return {
                "active_mode": current_item.get("mode") or prev_mode or "S",
                "current_subgoal": current_item.get("subgoal") or "",
                "current_is_switch": False,
                "current_switch_to": None,
                "current_switch_reason": None,
            }
        return {
            "active_mode": prev_mode or "S",
            "current_subgoal": "",
            "current_is_switch": False,
            "current_switch_to": None,
            "current_switch_reason": None,
        }

    def _guided_apply_router_state(self, state: dict) -> None:
        self._guided_router_state = copy.deepcopy(state)
        self.active_mode = state["active_mode"]
        self._guided_current_subgoal = state.get("current_subgoal") or ""
        self._guided_force_answer = bool(state.get("done")) or not bool(
            state.get("remaining_subgoals")
        )
        # If the head entry is a switch (e.g. a SHAPE-1 LEADING switch right
        # after init), force the decision model to emit `switch` as its first
        # action: expose ONLY the switch tool and stage the pending hand-off so
        # the runtime intercept flips sides and pops the entry afterwards.
        if state.get("current_is_switch"):
            self.allowed_tool_names = {"switch"}
            self._guided_pending_switch = {
                "from": self.active_mode,
                "to": state.get("current_switch_to"),
                "reason": state.get("current_switch_reason") or "",
            }
        else:
            self.allowed_tool_names = get_allowed_tool_names(
                self.agent_mode, self.active_mode)
            self._guided_pending_switch = None

    def _guided_consume_switch_entry(self) -> None:
        """Called right after the decision model emits a real `switch`. Pop the
        head switch entry from remaining_subgoals and re-derive live state so the
        new side activates with the next (post-switch) task subgoal exposed.
        """
        state = self._guided_router_state or {}
        remaining = list(state.get("remaining_subgoals") or [])
        if remaining and remaining[0].get("type") == "switch":
            remaining = remaining[1:]
        new_state = dict(state)
        new_state["remaining_subgoals"] = remaining
        new_state.update(self._derive_state_from_remaining(remaining, state.get("current_switch_to")))
        new_state["done"] = not remaining
        self._guided_apply_router_state(new_state)

    async def _guided_post_action_browser_state(self, mode: str | None) -> dict | None:
        if self.agent_mode != "hybrid_guided_routed" or mode != "O" or self.web is None:
            return None
        try:
            title = await self.web.page.title()
        except Exception:
            title = ""
        return {
            "current_url": getattr(self.web.page, "url", "") or "",
            "current_title": title,
            "tabs": self.web.tabs_snapshot() if hasattr(self.web, "tabs_snapshot") else [],
        }

    async def _guided_route_before_step(self) -> dict | None:
        """Pre-step routing: refresh the hidden subgoal/active_mode BEFORE the
        decision model acts, having fully digested the PREVIOUS step's action +
        observation (stashed in self._guided_pending_action_ctx at the end of
        that step). This eliminates the one-step lag of the old post-action
        scheme, where step N acted on a subgoal computed from step N-2's state.

        If the router decides to hand off, this does NOT flip active_mode or
        insert a record. Instead it keeps the decision model on the current side,
        exposes the `switch` tool, and sets a switch subgoal (via
        self._guided_pending_switch) so the decision model emits a REAL switch
        tool_call this turn -- giving a continuous report. The runtime intercept
        then flips the side. The routing result is stashed in
        self._guided_last_route_info for the main loop to copy onto the step
        record it is about to build.
        """
        self._guided_last_route_info = None
        if self.agent_mode != "hybrid_guided_routed" or self._guided_router_state is None:
            return None
        prev_ctx = self._guided_pending_action_ctx
        # Consume exactly once: clear immediately so a format-error step (which
        # does not re-stash) can't make the next iteration re-digest stale ctx
        # and fire a duplicate router call.
        self._guided_pending_action_ctx = None
        if not prev_ctx:
            # No prior action to digest yet (e.g. right after initialize) or the
            # previous step was the switch hand-off itself (which does not stash
            # ctx). The current router state already drives this step.
            return None
        # Defensive: if a hand-off was pending but the decision model did NOT
        # actually emit `switch` last turn (it disobeyed and ran some other
        # tool), the switch was never consumed. Clear the stale pending state so
        # the router re-evaluates from scratch on this fresh observation rather
        # than carrying a half-applied hand-off.
        if self._guided_pending_switch is not None:
            self._guided_pending_switch = None
            self.allowed_tool_names = get_allowed_tool_names(
                self.agent_mode, self.active_mode
            )
        # The step number this routing decision is FOR (the step about to run).
        upcoming_step = len(self.records) + 1
        observation = prev_ctx.get("observation") or ""
        browser_state = prev_ctx.get("browser_state")
        payload = {
            "task": self.task,
            "standard_route_plan": self._guided_route_plan,
            "previous_router_state": self._guided_router_state,
            "last_decision_action": {
                "step": prev_ctx.get("step"),
                "mode": prev_ctx.get("active_mode"),
                "tool_call": prev_ctx.get("tool_call"),
                "action_description": prev_ctx.get("action_description") or "",
            },
            "last_decision_report": prev_ctx.get("report_parsed") or {},
            "observation_after_last_action": observation,
            "recent_actions": self._format_recent_actions(),
            "instructions": (
                "You are routing the NEXT decision turn, having just observed the "
                "result of the last action. Return the UPDATED structured "
                "remaining_subgoals. If the head task entry is now complete, drop "
                "it (add to completed_subgoals) and expose the next entry; keep the "
                "switch entries in place so a hand-off surfaces at the head exactly "
                "when the next task runs on the other mode. Use browser_state only "
                "when present. Bind placeholders only from concrete values actually "
                "observed. When all subgoals are complete, return an empty "
                "remaining_subgoals with done=true; do not add a switch just for "
                "the final answer. Keep the list a coherent superset-minus-"
                "completed of the previous one -- do not reorder or rewrite "
                "unaffected future entries (it is replayed as part of the decision "
                "model's training trajectory and must evolve monotonically).\n"
                "ALSO assess the LAST decision action's quality in "
                "last_step_quality: did the chosen tool/arguments make sense for "
                "the subgoal it was serving, and did the observation actually "
                "advance it? verdict must be one of good (right action, clear "
                "progress), acceptable (reasonable but suboptimal — e.g. "
                "redundant query, slightly wrong target, partial progress), poor "
                "(wrong tool/side, no progress, harmful or looping). Judge only "
                "the action that just executed, not the whole run."
            ),
            "required_output": {
                "remaining_subgoals": [
                    {"type": "task", "mode": "S or O", "subgoal": "..."},
                    {"type": "switch", "to": "S or O", "reason": "..."},
                ],
                "completed_subgoals": ["subgoals completed by the last action, if any"],
                "done": "true iff the task is ready for the final answer",
                "last_step_quality": {
                    "verdict": "good | acceptable | poor",
                    "reason": "one short sentence grounded in the observation",
                },
            },
        }
        if browser_state is not None:
            payload["browser_state_after_last_action"] = browser_state

        messages = [
            {"role": "system", "content": self._guided_router_system_message_update()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        raw = ""
        from_mode = prev_ctx.get("active_mode")
        try:
            raw = await self._guided_router_chat(messages)
            next_state = self._parse_guided_router_response(raw, prev_mode=from_mode)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            trace_event_id = self._record_guided_router_trace(
                kind="update",
                related_step=upcoming_step,
                messages=messages,
                raw_response=raw,
                error=error,
            )
            # Routing failed: keep the existing router state (subgoal/mode carry
            # over). Record the error so the main loop can stamp it on the step.
            self._guided_last_route_info = {
                "guided_router_error": error,
                "guided_router_trace_event_id": trace_event_id,
            }
            return None

        trace_event_id = self._record_guided_router_trace(
            kind="update",
            related_step=upcoming_step,
            messages=messages,
            raw_response=raw,
            parsed_state=next_state,
        )

        # Stamp the router's quality verdict onto the step it actually judged
        # (the PREVIOUS step, whose action+observation this call digested) so
        # the per-step record carries its own assessment.
        quality = next_state.get("last_step_quality")
        if quality and self.records:
            prev_step_no = prev_ctx.get("step")
            for rec in reversed(self.records):
                if rec.get("step") == prev_step_no:
                    rec["guided_step_quality"] = quality
                    break

        # The router state always carries the structured remaining_subgoals list.
        # Whether we switch this turn is derived from the head entry's type
        # (current_is_switch) -- there is no separate should_switch flag.
        if next_state.get("current_is_switch"):
            to_mode = next_state["current_switch_to"]
            # Keep the decision model on the FROM side this turn and force it to
            # emit `switch`: expose ONLY the switch tool (business tools removed)
            # so it cannot keep acting on the from side or answer early. The
            # injected hand-off subgoal explains why; the runtime intercept flips
            # the side and pops the switch entry afterwards.
            self._guided_apply_router_state(next_state)  # active_mode already = from side
            self._guided_pending_switch = {
                "from": next_state["active_mode"],
                "to": to_mode,
                "reason": next_state.get("current_switch_reason") or "",
            }
            self.allowed_tool_names = {"switch"}
            self._guided_force_answer = False
            self._guided_last_route_info = {
                "guided_router": {
                    "active_mode": self.active_mode,
                    "current_subgoal": self._guided_current_subgoal,
                    "remaining_count": len(next_state.get("remaining_subgoals") or []),
                    "done": False,
                    "current_is_switch": True,
                    "pending_switch_to": to_mode,
                },
                "guided_router_trace_event_id": trace_event_id,
            }
            return next_state

        # Head entry is a task (or done): apply normally.
        self._guided_apply_router_state(next_state)
        self._guided_last_route_info = {
            "guided_router": {
                "active_mode": self.active_mode,
                "current_subgoal": self._guided_current_subgoal,
                "remaining_count": len(next_state.get("remaining_subgoals") or []),
                "done": bool(next_state.get("done")),
                "current_is_switch": False,
            },
            "guided_router_trace_event_id": trace_event_id,
        }
        return next_state

    async def _guided_adjudicate_early_answer(self, answer: str) -> tuple[bool, str]:
        """The decision model wants to answer while the router still has open
        subgoals. Instead of flat-rejecting (which format-errors runs where the
        model is RIGHT and the router state is merely stale), ask the router to
        adjudicate: accept the answer (mark done) or reject with the concrete
        open subgoal. Returns (accept, reason). Never raises."""
        payload = {
            "task": self.task,
            "proposed_final_answer": answer,
            "router_state": self._guided_router_state,
            "last_decision_report": self.last_report or "",
            "recent_actions": self._format_recent_actions(),
            "instructions": (
                "The decision model proposes to FINISH with this answer although "
                "your remaining_subgoals list is not empty. Decide: is the task "
                "genuinely complete (every ask answered and grounded in what was "
                "actually observed), making the remaining entries stale "
                "bookkeeping? Then accept=true. Or is something real still "
                "missing? Then accept=false and name the concrete missing piece "
                "in reason (the decision model will see it)."
            ),
            "required_output": {"accept": "true | false",
                                "reason": "one short sentence"},
        }
        messages = [
            {
                "role": "system",
                "content": self._GUIDED_ROUTER_COMMON
                + "\n\nThis is an EARLY-ANSWER adjudication call. Output exactly "
                "one JSON object with accept / reason.",
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        raw = ""
        try:
            raw = await self._guided_router_chat(messages)
            obj = json.loads(self._strip_outer_json_fence(strip_reasoning(raw)))
            acc = obj.get("accept")
            if isinstance(acc, str):
                acc = acc.strip().lower() == "true"
            else:
                acc = bool(acc)
            reason = str(obj.get("reason") or "").strip()[:300]
            self._record_guided_router_trace(
                kind="early_answer",
                related_step=len(self.records) + 1,
                messages=messages,
                raw_response=raw,
                parsed_state={"accept": acc, "reason": reason},
            )
            return acc, reason
        except Exception as exc:
            self._record_guided_router_trace(
                kind="early_answer",
                related_step=len(self.records) + 1,
                messages=messages,
                raw_response=raw,
                error=f"{type(exc).__name__}: {exc}",
            )
            # Adjudication itself failed — keep the old conservative behavior.
            return False, "router adjudication unavailable"

    async def _guided_final_verdict(self, prediction: str) -> dict | None:
        """ONE extra router call AFTER the decision model emitted its final
        answer: holistic judgement of whether the task was actually solved.
        Recorded in the router trace (kind="final_verdict") and surfaced in
        the run result. Advisory metadata — never raises."""
        if self.agent_mode != "hybrid_guided_routed" or self._guided_router_state is None:
            return None
        payload = {
            "task": self.task,
            "standard_route_plan": self._guided_route_plan,
            "final_answer": prediction,
            "router_state_at_answer": self._guided_router_state,
            "last_decision_report": self.last_report or "",
            "recent_actions": self._format_recent_actions(),
            "instructions": (
                "The decision model has just emitted its FINAL answer. Make ONE "
                "holistic judgement of the whole run: was the task actually "
                "solved? Check the answer against every ask in the task (all "
                "requested fields present, each grounded in something actually "
                "observed during the run — not invented), and whether the "
                "standard route plan's intent was fulfilled. partially_solved "
                "means some asks are answered and grounded but others are "
                "missing or ungrounded."
            ),
            "required_output": {
                "task_solved": "true | false",
                "verdict": "solved | partially_solved | not_solved",
                "reason": "one or two short sentences",
            },
        }
        messages = [
            {
                "role": "system",
                "content": self._GUIDED_ROUTER_COMMON
                + "\n\nThis is the FINAL verdict call: the run has ended and you "
                "are judging the outcome. Output exactly one JSON object with "
                "task_solved / verdict / reason.",
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        raw = ""
        try:
            raw = await self._guided_router_chat(messages)
            text = self._strip_outer_json_fence(strip_reasoning(raw))
            obj = json.loads(text)
            ts = obj.get("task_solved")
            if isinstance(ts, str):
                ts = ts.strip().lower() == "true"
            else:
                ts = bool(ts)
            verdict = {
                "task_solved": ts,
                "verdict": (str(obj.get("verdict") or "").strip().lower() or "unknown"),
                "reason": str(obj.get("reason") or "").strip()[:500],
            }
            self._record_guided_router_trace(
                kind="final_verdict",
                related_step=len(self.records),
                messages=messages,
                raw_response=raw,
                parsed_state=verdict,
            )
            return verdict
        except Exception as exc:
            self._record_guided_router_trace(
                kind="final_verdict",
                related_step=len(self.records),
                messages=messages,
                raw_response=raw,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None

    # DEPRECATED / UNUSED. Previously the guided router synthesized a switch
    # record itself (decision model not involved), which broke decision-model
    # context continuity. Switches are now emitted by the decision model as a
    # real `switch` tool_call (router sets a switch subgoal + exposes the tool
    # in _guided_route_before_step; the runtime intercept in run() handles it).
    # Kept only as documentation of the old behavior; do not call.

    def _guided_patch_parsed_report(self, parsed: dict, *, require_answer: bool) -> dict:
        """Merge the router-maintained state into the decision model's report so
        the persisted trajectory carries a COHERENT remaining_subgoals (this is
        part of the training data). At runtime the decision model only sees
        `current_subgoal` in its prompt (the full plan is not leaked); here, on
        the way to the record, we overwrite `current_subgoal` and inject the
        router's structured `remaining_subgoals` (which evolves monotonically
        across steps and includes `switch` entries).
        """
        if self.agent_mode != "hybrid_guided_routed" or parsed.get("report_parsed") is None:
            return parsed
        patched = dict(parsed)
        report_obj = copy.deepcopy(parsed["report_parsed"])
        is_answer = bool(require_answer or parsed.get("answer"))
        if is_answer:
            report_obj["current_subgoal"] = "Final answer"
            report_obj["remaining_subgoals"] = []
        else:
            current = (self._guided_current_subgoal or "").strip()
            report_obj["current_subgoal"] = current or "Continue the current routed subgoal."
            # Router's coherent structured list as of this step (the decision
            # model acted on its head entry). Falls back to the model's own list
            # only if the router state is somehow missing.
            router_remaining = (self._guided_router_state or {}).get("remaining_subgoals")
            if isinstance(router_remaining, list):
                report_obj["remaining_subgoals"] = copy.deepcopy(router_remaining)
        try:
            patched["report"] = json.dumps(report_obj, ensure_ascii=False, indent=2)
        except Exception:
            patched["report"] = parsed["report"]
        patched["report_parsed"] = report_obj
        return patched

    def _summarize_visit_content(self, prompt_text: str, image_data_urls: list[str] | None = None) -> str:
        content = [{"type": "text", "text": prompt_text}]
        for item in image_data_urls or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": item},
                }
            )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You extract useful page information for a web agent."}],
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        return strip_reasoning(
            self._summary_chat(
                messages,
                max_tokens=self.args.visit_summary_max_tokens,
                temperature=0.0,
            )
        )

    def _record_summary_trace(self, trace: dict) -> None:
        self.current_summary_traces.append(copy.deepcopy(trace))

    def _messages_have_images(self, messages: list[dict]) -> bool:
        for message in messages:
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
        return False

    def _strip_images_from_messages(self, messages: list[dict]) -> list[dict]:
        sanitized = copy.deepcopy(messages)
        for message in sanitized:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            message["content"] = [
                item
                for item in content
                if not (isinstance(item, dict) and item.get("type") == "image_url")
            ]
        return sanitized

    def _should_retry_without_images(self, exc: Exception, messages: list[dict]) -> bool:
        if not self._messages_have_images(messages):
            return False
        text = str(exc).lower()
        return (
            "cannot identify image file" in text
            or "invalid image" in text
            or "unsupported image" in text
        )

    def _parse_tool_call(self, tool_call_obj) -> tuple[dict | None, str]:
        """Validate an already-parsed tool_call dict (post unified-JSON refactor).

        We accept either a dict (the new path) or a JSON string (legacy
        callers) for safety. The argument shape / tool-name allow-list /
        per-tool argument validation below is unchanged.
        """
        if isinstance(tool_call_obj, str):
            try:
                tool_call = json.loads(tool_call_obj)
            except Exception as exc:
                return None, f"Invalid tool_call JSON: {exc}"
        else:
            tool_call = tool_call_obj

        if not isinstance(tool_call, dict):
            return None, "tool_call must be a JSON object."

        name = tool_call.get("name")
        if name not in {
            "search",
            "scholar",
            "visit",
            "click",
            "double_click",
            "hover",
            "fill",
            "select_option",
            "scroll",
            "drag",
            "wait",
            "go_back",
            "goto",
            "new_tab",
            "press_key",
            "switch_tab",
            "close_tab",
            "code_sandbox",
            "switch",  # routed hybrid: hand-off tool
        }:
            return None, f"Unsupported tool name: {name}"

        # Guided hand-off turn: the pre-step router decided to switch and set
        # allowed_tool_names={"switch"} to FORCE a hand-off. Honor that cache
        # verbatim here -- do NOT recompute from active_mode, or path-neutral
        # tools like code_sandbox (and the whole from-side toolset) would leak
        # back in and let the model dodge the switch.
        if (
            getattr(self, "agent_mode", "") == "hybrid_guided_routed"
            and getattr(self, "_guided_pending_switch", None) is not None
        ):
            allowed_tool_names = getattr(self, "allowed_tool_names", {"switch"})
        # Other routed turns: refresh allowed_tool_names from the live
        # active_mode so the per-side restriction (S-side cannot call O tools,
        # etc.) is enforced even if the agent's cache is stale.
        elif getattr(self, "agent_mode", "") in {"hybrid_routed", "hybrid_guided_routed"}:
            allowed_tool_names = get_allowed_tool_names(
                self.agent_mode, getattr(self, "active_mode", None)
            )
        else:
            allowed_tool_names = getattr(
                self,
                "allowed_tool_names",
                get_allowed_tool_names(getattr(self, "agent_mode", "hybrid")),
            )
        if name not in allowed_tool_names:
            available = ", ".join(sorted(allowed_tool_names))
            mode = getattr(self, "agent_mode", "hybrid")
            return None, f"Tool {name!r} is not available in {mode} mode. Available tools: {available}."

        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            return None, "tool_call.arguments must be an object."

        if name in {"search", "scholar"}:
            query = arguments.get("query")
            if isinstance(query, str):
                arguments["query"] = [query]
            if not isinstance(arguments.get("query"), list):
                return None, f"{name} requires a non-empty query list."
            cleaned_query = [
                item.strip()
                for item in arguments["query"]
                if isinstance(item, str) and item.strip()
            ]
            if not cleaned_query:
                return None, f"{name} requires at least one non-empty string query."
            arguments["query"] = cleaned_query
        elif name == "visit":
            goal = arguments.get("goal")
            if not isinstance(goal, str) or not goal.strip():
                return None, "visit requires a non-empty string goal."
            arguments["goal"] = goal.strip()
            url = arguments.get("url")
            if isinstance(url, str):
                url = url.strip()
                if url:
                    arguments["url"] = url
                else:
                    arguments.pop("url", None)
            elif isinstance(url, list):
                cleaned = [
                    item.strip()
                    for item in url
                    if isinstance(item, str) and item.strip()
                ]
                if cleaned:
                    arguments["url"] = cleaned
                else:
                    arguments.pop("url", None)
            elif url is not None:
                return None, "visit url must be a string or list of strings."
            parse_type = arguments.get("parse_type", "auto")
            if parse_type is None:
                parse_type = "auto"
            if not isinstance(parse_type, str):
                return None, "visit parse_type must be one of auto/html/pdf when provided."
            parse_type = parse_type.strip().lower() or "auto"
            if parse_type not in {"auto", "html", "pdf"}:
                return None, "visit parse_type must be one of auto/html/pdf."
            arguments["parse_type"] = parse_type
            if self.agent_mode == "search" and not arguments.get("url"):
                return None, "visit requires url in search mode because no browser state is used."
        elif name == "code_sandbox":
            code = arguments.get("code")
            if not isinstance(code, str) or not code.strip():
                return None, "code_sandbox requires non-empty string code."
            arguments["code"] = code

            language = arguments.get("language", "python")
            if language is None:
                language = "python"
            if not isinstance(language, str):
                return None, "code_sandbox language must be a string."
            language = language.strip() or "python"
            if language not in SUPPORTED_LANGUAGES:
                available = ", ".join(sorted(SUPPORTED_LANGUAGES))
                return None, f"code_sandbox language must be one of: {available}."
            arguments["language"] = language

            stdin = arguments.get("stdin")
            if stdin is not None and not isinstance(stdin, str):
                return None, "code_sandbox stdin must be a string when provided."

            for key in ("run_timeout", "compile_timeout"):
                value = arguments.get(key, 10)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return None, f"code_sandbox {key} must be a number when provided."
                if value <= 0:
                    return None, f"code_sandbox {key} must be positive."
                arguments[key] = min(float(value), 60.0)

            files = arguments.get("files", {})
            if files is None:
                files = {}
            if not isinstance(files, dict):
                return None, "code_sandbox files must be an object mapping file paths to base64 strings."
            cleaned_files = {}
            for path, content in files.items():
                if not isinstance(path, str) or not path.strip():
                    return None, "code_sandbox files keys must be non-empty file paths."
                if content is not None and not isinstance(content, str):
                    return None, "code_sandbox files values must be base64 strings or null."
                cleaned_files[path.strip()] = content
            arguments["files"] = cleaned_files

            fetch_files = arguments.get("fetch_files", [])
            if fetch_files is None:
                fetch_files = []
            if not isinstance(fetch_files, list):
                return None, "code_sandbox fetch_files must be a list of file paths."
            arguments["fetch_files"] = [
                item.strip()
                for item in fetch_files
                if isinstance(item, str) and item.strip()
            ]
        elif name == "switch":
            # Routed hybrid: only `reason` (string) is allowed. No-op on the
            # browser; runtime intercepts to flip active_mode. Allowed in both
            # hybrid_routed (model chooses freely) and hybrid_guided_routed
            # (only exposed on the router-decided hand-off turn).
            if getattr(self, "agent_mode", "") not in {"hybrid_routed", "hybrid_guided_routed"}:
                return None, "switch tool is only available in routed modes."
            reason = arguments.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                return None, "switch requires a non-empty string reason."
            arguments["reason"] = reason.strip()
            # Strip any extras the model may have invented.
            for key in list(arguments.keys()):
                if key != "reason":
                    arguments.pop(key, None)
        elif name in {"click", "hover", "double_click"}:
            for coord in ("x", "y"):
                if coord not in arguments:
                    return None, f"{name} requires {coord}."
                val = arguments[coord]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    return None, f"{name} {coord} must be a number."
                arguments[coord] = float(val)
        elif name == "select_option":
            for coord in ("x", "y"):
                if coord not in arguments:
                    return None, f"select_option requires {coord}."
                val = arguments[coord]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    return None, f"select_option {coord} must be a number."
                arguments[coord] = float(val)
            value = arguments.get("value")
            if not isinstance(value, str) or not value.strip():
                return None, "select_option requires a non-empty string value."
            arguments["value"] = value
        elif name == "fill":
            for coord in ("x", "y"):
                if coord not in arguments:
                    return None, f"fill requires {coord}."
                val = arguments[coord]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    return None, f"fill {coord} must be a number."
                arguments[coord] = float(val)
            if "text" not in arguments:
                return None, "fill requires text."
            if not isinstance(arguments.get("text"), str):
                return None, "fill text must be a string."
            if "submit" in arguments and not isinstance(arguments.get("submit"), bool):
                return None, "fill submit must be a boolean when provided."
        elif name == "scroll":
            if arguments.get("direction") not in {"up", "down"}:
                return None, "scroll requires direction up/down."
            has_x = "x" in arguments
            has_y = "y" in arguments
            if has_x != has_y:
                return None, "scroll requires both x and y (region scroll) or neither (window scroll)."
            if has_x and has_y:
                for coord in ("x", "y"):
                    val = arguments[coord]
                    if isinstance(val, bool) or not isinstance(val, (int, float)):
                        return None, f"scroll {coord} must be a number."
                    arguments[coord] = float(val)
        elif name == "drag":
            for coord in ("from_x", "from_y", "to_x", "to_y"):
                if coord not in arguments:
                    return None, f"drag requires {coord}."
                val = arguments[coord]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    return None, f"drag {coord} must be a number."
                arguments[coord] = float(val)
            if "hold_ms" in arguments:
                hv = arguments.get("hold_ms")
                if isinstance(hv, bool) or not isinstance(hv, (int, float)):
                    return None, "drag hold_ms must be a number when provided."
                arguments["hold_ms"] = max(0, int(hv))
        elif name in {"goto", "new_tab"}:
            url = arguments.get("url")
            if not isinstance(url, str):
                return None, f"{name} url must be a string."
            url = url.strip()
            if not url:
                return None, f"{name} requires url."
            arguments["url"] = url
        elif name == "wait":
            wait_time = arguments.get("time", 5)
            if not isinstance(wait_time, (int, float)) or isinstance(wait_time, bool):
                return None, "wait time must be a number when provided."
            if wait_time < 0:
                return None, "wait time must be non-negative."
            arguments["time"] = min(float(wait_time), 30.0)
        elif name == "press_key":
            keys = arguments.get("keys")
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list) or not keys:
                return None, "press_key requires a non-empty list of key names."
            cleaned_keys = [
                item.strip()
                for item in keys
                if isinstance(item, str) and item.strip()
            ]
            if not cleaned_keys:
                return None, "press_key requires at least one non-empty string key."
            arguments["keys"] = cleaned_keys
        elif name in {"switch_tab", "close_tab"}:
            index = arguments.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                return None, f"{name} index must be an integer."
            if index < 0:
                return None, f"{name} index must be non-negative."
            arguments["index"] = index

        tool_call["arguments"] = arguments
        return tool_call, "SUCCESS"

    # Required key set for the unified single-JSON response, indexed by
    # (agent_mode, is_final). All listed keys must be present; no extras.
    # `cumulative_progress` is the running cross-turn KB; `this_turn` is
    # what THIS turn newly contributed (merged into next turn's
    # cumulative_progress by the agent). Always-null fields are dropped
    # from the schema (turn => no `answer`; final => no `tool_call`).
    _UNIFIED_REQUIRED_KEYS = {
        # Turn variants: both `tool_call` AND `answer` are required keys;
        # exactly one must be non-null (validated in _parse_response).
        # `action_description` is required when tool_call is non-null;
        # MUST be null when answer is set.
        ("hybrid", False): {
            "cumulative_progress", "this_turn", "remaining_subgoals",
            "path_switch", "next_action_rationale",
            "tool_call", "action_description", "answer",
        },
        ("hybrid", True): {
            "cumulative_progress", "this_turn", "remaining_subgoals",
            "answer_plan", "answer",
        },
        ("operation", False): {
            "cumulative_progress", "this_turn", "remaining_subgoals",
            "next_action_rationale",
            "tool_call", "action_description", "answer",
        },
        ("operation", True): {
            "cumulative_progress", "this_turn", "remaining_subgoals",
            "answer_plan", "answer",
        },
        ("search", False): {
            "cumulative_progress", "this_turn", "remaining_subgoals",
            "next_action_rationale",
            "tool_call", "action_description", "answer",
        },
        ("search", True): {
            "cumulative_progress", "this_turn", "remaining_subgoals",
            "answer_plan", "answer",
        },
        # Routed hybrid: schema mirrors search/operation (no path_switch field;
        # switching is now an explicit tool call). Same keys for both S- and
        # O-side because USER_PROMPTS_BY_ROUTED_SIDE picks the per-side user
        # template, not the schema.
        ("hybrid_routed", False): {
            "cumulative_progress", "this_turn", "remaining_subgoals",
            "next_action_rationale",
            "tool_call", "action_description", "answer",
        },
        ("hybrid_routed", True): {
            "cumulative_progress", "this_turn", "remaining_subgoals",
            "answer_plan", "answer",
        },
        ("hybrid_guided_routed", False): {
            "cumulative_progress", "this_turn", "next_action_rationale",
            "tool_call", "action_description", "answer",
        },
        ("hybrid_guided_routed", True): {
            "cumulative_progress", "this_turn", "answer_plan", "answer",
        },
    }

    @staticmethod
    def _strip_outer_json_fence(text: str) -> str:
        """If the model wraps the entire response in ```json ... ```, peel it."""
        s = text.strip()
        if s.startswith("```"):
            m = re.search(r"^```(?:json)?\s*(\{.*\})\s*```\s*$", s, flags=re.DOTALL)
            if m:
                return m.group(1).strip()
        return s

    def _parse_response(self, raw_text: str, require_answer: bool) -> tuple[dict | None, str]:
        """Parse the model's response as a single top-level JSON object and
        validate it against the unified schema for (agent_mode, require_answer).

        Returns (parsed_payload, "SUCCESS") on success, where parsed_payload is:
            {
              "report":        <pretty-printed JSON string of the full obj>,
              "report_parsed": <full obj dict>,
              "tool_call":     <validated tool_call dict, or None>,
              "answer":        <answer string, or "">,
            }
        On failure returns (None, error_message) suitable for the format-retry
        loop.
        """
        text = self._strip_outer_json_fence(strip_reasoning(raw_text))
        if not text:
            return None, "Empty response. Emit a single JSON object as the entire response."
        try:
            obj = json.loads(text)
        except Exception as exc:
            return None, (
                f"Response is not a single valid JSON object "
                f"({type(exc).__name__}: {exc}). Emit ONLY one top-level JSON "
                f"object as your entire response — no XML tags, no prose."
            )
        if not isinstance(obj, dict):
            return None, "Response must be a JSON object, not an array or scalar."

        required = self._UNIFIED_REQUIRED_KEYS.get(
            (self.agent_mode, bool(require_answer)),
            None,
        )
        if required is None:
            return None, f"Unknown (agent_mode, require_answer) combo: ({self.agent_mode}, {require_answer})."

        # All modes: initial turn uses task_analysis schema (fewer fields,
        # focused on task decomposition). Accept task_analysis + remaining_subgoals
        # + tool_call + action_description; normalize to standard shape after.
        if not self.records and "task_analysis" in obj:
            if self.agent_mode == "hybrid":
                required = {"task_analysis", "remaining_subgoals", "path_switch", "next_action_rationale", "tool_call", "action_description"}
            elif self.agent_mode == "hybrid_guided_routed":
                required = {"task_analysis", "next_action_rationale", "tool_call", "action_description"}
            else:
                required = {"task_analysis", "remaining_subgoals", "next_action_rationale", "tool_call", "action_description"}

        if self.agent_mode == "hybrid_guided_routed":
            obj.pop("remaining_subgoals", None)
            obj.pop("current_subgoal", None)

        missing = sorted(required - set(obj.keys()))
        if missing:
            return None, (
                f"Response JSON is missing required key(s): {missing}. "
                f"Required keys for this turn: {sorted(required)}."
            )
        extras = sorted(set(obj.keys()) - required)
        if extras:
            return None, (
                f"Response JSON contains unexpected key(s): {extras}. "
                f"Allowed keys for this turn: {sorted(required)}."
            )

        # Initial turn: task_analysis replaces cumulative_progress/this_turn.
        # Keep as-is in report_parsed — do NOT rename to cumulative_progress.
        # Step 2 will build cumulative_progress from scratch based on step 1's
        # observation. Inject synthetic fields only for _build_parsed downstream.
        is_initial = not self.records and "task_analysis" in obj
        if is_initial:
            v = obj.get("task_analysis")
            if not isinstance(v, str) or not v.strip():
                return None, "`task_analysis` must be a non-empty string."
            obj.setdefault("answer", None)
            obj["cumulative_progress"] = "No progress yet."
        else:
            # Type checks: cumulative_progress + this_turn are non-empty strings.
            for str_key in ("cumulative_progress", "this_turn"):
                v = obj.get(str_key)
                if not isinstance(v, str) or not v.strip():
                    return None, f"`{str_key}` must be a non-empty string."
        if self.agent_mode != "hybrid_guided_routed":
            sg = obj.get("remaining_subgoals")
            if not isinstance(sg, list):
                return None, f"`remaining_subgoals` must be a JSON array (got {type(sg).__name__})."
            for item in sg:
                if not isinstance(item, str):
                    return None, "`remaining_subgoals` items must be strings."

        # Hybrid-only path_switch validation.
        if "path_switch" in required:
            ps = obj.get("path_switch")
            if ps is not None:
                if not isinstance(ps, dict):
                    return None, "`path_switch` must be null or an object."
                if ps.get("to") not in {"search-based", "operation-based"}:
                    return None, "`path_switch.to` must be 'search-based' or 'operation-based'."
                if not isinstance(ps.get("reason"), str) or not ps.get("reason").strip():
                    return None, "`path_switch.reason` must be a non-empty string."

        # Final turn: only `answer` (non-empty string) is required; tool_call
        # is not in the schema at all so the extras-check above already
        # rejected it if accidentally included.
        if require_answer:
            answer_text = obj.get("answer")
            if not isinstance(answer_text, str) or not answer_text.strip():
                return None, "Final turn: `answer` MUST be a non-empty string."
            return self._build_parsed(obj, tool_call=None, answer=answer_text, action_description=None), "SUCCESS"

        # Non-final turn: exactly one of `tool_call` (object) / `answer`
        # (non-empty string) must be set; the other MUST be null.
        tool_call_obj = obj.get("tool_call")
        answer_text = obj.get("answer")
        has_tool = isinstance(tool_call_obj, dict)
        has_ans = isinstance(answer_text, str) and bool(answer_text.strip())
        if has_tool == has_ans:
            return None, (
                "Exactly one of `tool_call` (object) and `answer` "
                "(non-empty string) must be non-null; the other MUST be null."
            )
        # action_description validation (only present in non-final turns).
        action_description = obj.get("action_description")
        if has_ans:
            # Agent chose to terminate early with a final answer.
            if action_description is not None and action_description != "":
                return None, (
                    "When `answer` is set, `action_description` MUST be null (or omitted)."
                )
            return self._build_parsed(obj, tool_call=None, answer=answer_text, action_description=None), "SUCCESS"
        tool_call, status = self._parse_tool_call(tool_call_obj)
        if tool_call is None:
            return None, status
        if not isinstance(action_description, str) or not action_description.strip():
            return None, (
                "When `tool_call` is non-null, `action_description` MUST be a non-empty string "
                "(5-15 words naming the visible target element and expected outcome)."
            )
        return self._build_parsed(obj, tool_call=tool_call, answer="", action_description=action_description.strip()), "SUCCESS"

    def _build_parsed(self, obj: dict, *, tool_call, answer: str, action_description: str | None = None) -> dict:
        # Pretty-printed JSON of the full obj — used for `last_report` injection
        # and the saved step_record["report"] field. Validated tool_call (with
        # arguments coerced/cleaned) is substituted in so downstream consumers
        # see the canonical version, not the model's raw input. We don't add
        # back any "always-null" fields here — keeping the saved record shape
        # aligned with the prompt schema.
        full = dict(obj)
        if tool_call is not None:
            full["tool_call"] = tool_call
        if answer:
            full["answer"] = answer
        if "action_description" in full:
            full["action_description"] = action_description
        try:
            pretty = json.dumps(full, ensure_ascii=False, indent=2)
        except Exception:
            pretty = json.dumps({k: str(v) for k, v in full.items()}, ensure_ascii=False, indent=2)
        return {
            "report": pretty,
            "report_parsed": full,
            "tool_call": tool_call,
            "answer": answer,
            "action_description": action_description,
        }

    def _validate_tool_call_against_state(self, state: dict | None, tool_call: dict) -> str | None:
        name = tool_call["name"]
        arguments = tool_call["arguments"]

        browser_action_tools = {
            "hover",
            "click",
            "double_click",
            "fill",
            "select_option",
            "scroll",
            "drag",
            "wait",
            "go_back",
            "press_key",
        }
        if name in browser_action_tools and not self._has_usable_browser_page(state):
            if self.agent_mode == "operation":
                next_step = "use goto or new_tab to open a page before browser operations."
            else:
                next_step = (
                    "use goto or new_tab to open a page before browser operations. "
                    "search, scholar, and visit may still be used for information gathering."
                )
            return (
                f"{name} is not valid because the browser has no usable loaded page. "
                f"When the browser is blank, {next_step}"
            )

        # Pixel-coordinate bounds check for tools that take (x, y).
        if name not in {"hover", "click", "double_click", "fill", "select_option", "scroll", "drag"}:
            return None

        if name == "scroll" and ("x" not in arguments or "y" not in arguments):
            # Page-level scroll: nothing to check.
            return None

        scale = int(getattr(self.args, "coord_scale", 0) or 0)
        if scale > 0:
            w = h = float(scale)
        else:
            w = float(getattr(self.args, "window_width", 1080))
            h = float(getattr(self.args, "window_height", 1440))

        if name == "drag":
            try:
                pts = [(float(arguments["from_x"]), float(arguments["from_y"])),
                       (float(arguments["to_x"]), float(arguments["to_y"]))]
            except (KeyError, TypeError, ValueError):
                return "drag requires numeric from_x/from_y/to_x/to_y."
            for x, y in pts:
                if not (0 <= x < w and 0 <= y < h):
                    return f"drag coordinate ({x:.0f}, {y:.0f}) out of range [0, {int(w)}) x [0, {int(h)})."
            return None

        try:
            x = float(arguments["x"])
            y = float(arguments["y"])
        except (KeyError, TypeError, ValueError):
            return f"{name} requires numeric x and y."
        if not (0 <= x < w and 0 <= y < h):
            return f"{name} coordinates ({x:.0f}, {y:.0f}) out of range [0, {int(w)}) x [0, {int(h)})."
        return None

    def _scale_xy(self, x: float, y: float) -> tuple[float, float]:
        """Translate model-emitted (x, y) into real viewport pixels when
        --coord-scale is set (e.g. 1000 for Qwen-VL or 1 for kimi-k2.6
        relative-coord prompt). When scale=0 the values pass through."""
        scale = int(getattr(self.args, "coord_scale", 0) or 0)
        if scale <= 0:
            return (x, y)
        vw = float(getattr(self.args, "window_width", 1080))
        vh = float(getattr(self.args, "window_height", 1440))
        return (x * vw / scale, y * vh / scale)

    async def _capture_action_html(self, state: dict, arguments: dict) -> str:
        if state is None:
            return ""
        # For drag, capture the HTML at the start point (the handle/element being grabbed).
        if "from_x" in arguments and "from_y" in arguments:
            try:
                x = float(arguments["from_x"])
                y = float(arguments["from_y"])
            except (TypeError, ValueError):
                return ""
        elif "x" in arguments and "y" in arguments:
            try:
                x = float(arguments["x"])
                y = float(arguments["y"])
            except (TypeError, ValueError):
                return ""
        else:
            return ""
        x, y = self._scale_xy(x, y)
        html = await self.web.element_html_at(x, y)
        if not html:
            return ""
        return process_element_tag(html, SALIENT_ATTRIBUTES)

    def _format_action_observation(
        self,
        *,
        name: str,
        status: str,
        started_url: str,
        ended_url: str,
        action_html: str,
        diagnostics: dict,
        error: str = "",
        hint: str = "",
    ) -> str:
        recent = (
            self.web.consume_recent_messages(max_items=5) if self.web is not None else {}
        )
        console_errors = recent.get("console_errors") or []
        page_errors = recent.get("page_errors") or []
        downloads = recent.get("downloads") or []

        lines = [
            "Action execution result:",
            f"- status: {status}",
            f"- tool: {name}",
            f"- start_url: {started_url}",
        ]
        if status == "success":
            lines.append(f"- end_url: {ended_url}")
        lines.append(f"- target: {action_html or '[no target html]'}")

        target_state = diagnostics.get("target_state") if diagnostics else None
        if target_state:
            lines.append(
                f"- target_state: {json.dumps(target_state, ensure_ascii=False)}"
            )

        if name == "scroll" and diagnostics:
            scroll_target = diagnostics.get("scroll_target")
            if scroll_target == "WINDOW":
                delta = diagnostics.get("delta_y")
                lines.append(
                    f"- scroll_window: before_y={diagnostics.get('before_y')} "
                    f"after_y={diagnostics.get('after_y')} delta_y={delta}"
                )
                if delta == 0:
                    lines.append(
                        "- note: window scroll did not move. Page may be at the top/bottom, "
                        "or a scroll handler intercepted the wheel event."
                    )
            elif scroll_target == "ELEMENT":
                before = diagnostics.get("before") or {}
                after = diagnostics.get("after") or {}
                before_y = before.get("y")
                after_y = after.get("y")
                lines.append(
                    f"- scroll_element: scope={after.get('scope') or before.get('scope')} "
                    f"before_y={before_y} after_y={after_y} max={after.get('max')}"
                )
                if (
                    before_y is not None
                    and after_y is not None
                    and after_y == before_y
                ):
                    lines.append(
                        "- note: target container did not scroll. It may not be scrollable "
                        "in this direction; try scroll with WINDOW instead."
                    )

        if name == "go_back" and diagnostics and diagnostics.get("navigated") is False:
            lines.append(
                "- note: no prior history entry. go_back had no effect."
            )

        if name == "press_key" and diagnostics.get("keys"):
            lines.append(f"- keys: {diagnostics.get('keys')}")

        if name == "switch_tab" and diagnostics:
            lines.append(
                f"- tab_switch: from {diagnostics.get('switched_from')} "
                f"to {diagnostics.get('switched_to')} "
                f"(url: {(diagnostics.get('url','') or '')[:120]})"
            )
        if name == "close_tab" and diagnostics:
            lines.append(
                f"- tab_closed: index={diagnostics.get('closed')} "
                f"was_active={diagnostics.get('was_active')} "
                f"remaining={diagnostics.get('remaining')}"
            )

        if error:
            lines.append(f"- error: {error}")

        if console_errors:
            lines.append(f"- console_errors: {console_errors}")
        if page_errors:
            lines.append(f"- page_errors: {page_errors}")
        if downloads:
            lines.append("- downloaded_files (saved to disk; you can read or "
                         "process them via code_sandbox):")
            for dl in downloads:
                lines.append(
                    f"    path={dl.get('path')}  filename={dl.get('filename')!r}  "
                    f"from_url={(dl.get('url') or '')[:120]}"
                )

        if hint:
            lines.append(f"- hint: {hint}")

        tabs = self.web.tabs_snapshot() if self.web is not None else []
        if len(tabs) > 1:
            lines.append("- open_tabs:")
            for tab in tabs:
                marker = " (active)" if tab.get("active") else ""
                lines.append(
                    f"    [{tab.get('index')}]{marker} {(tab.get('url','') or '')[:120]}"
                )

        if status == "success":
            note = "Inspect the new screenshot and page state before the next step."
            if name in {
                "click",
                "double_click",
                "hover",
                "fill",
                "select_option",
                "scroll",
                "drag",
                "wait",
                "press_key",
                "go_back",
                "goto",
                "new_tab",
                "switch_tab",
                "close_tab",
            }:
                note += (
                    " Re-read coordinates from the new screenshot; do not reuse "
                    "(x, y) values from prior turns since layout may have shifted."
                )
            lines.append(f"- note: {note}")

        return "\n".join(lines)

    async def _execute_tool(self, state: dict, tool_call: dict, step: int) -> str:
        name = tool_call["name"]
        arguments = copy.deepcopy(tool_call["arguments"])
        started_url = ""
        if self.agent_mode != "search" and self.web is not None:
            started_url = self.web.page.url
        self.current_visit_screenshot_path = ""
        self.current_visit_screenshot_url = ""

        if name == "search":
            return self.search_tool.call(arguments.get("query", []))

        if name == "scholar":
            return self.scholar_tool.call(arguments.get("query", []))

        if name == "visit":
            url = arguments.get("url")
            goal = arguments["goal"]
            parse_type = arguments.get("parse_type", "auto")
            if not url:
                current_url = ""
                if (
                    state
                    and state.get("current_url")
                    and state["current_url"] != "about:blank"
                ):
                    current_url = state["current_url"]
                elif started_url and started_url != "about:blank":
                    current_url = started_url
                url = current_url

            if not url:
                if self.agent_mode == "search":
                    return "[visit] No URL was provided. In search mode, pass url explicitly to visit."
                return (
                    "[visit] No readable URL is available yet. "
                    "Provide a url in visit, or use search/goto first."
                )
            observation = await asyncio.to_thread(
                self.visit_tool.call,
                url,
                goal,
                parse_type,
            )

            screenshot_url = ""
            if isinstance(url, list):
                screenshot_url = next(
                    (item.strip() for item in url if isinstance(item, str) and item.strip()),
                    "",
                )
            elif isinstance(url, str):
                screenshot_url = url.strip()

            # Only capture a visit screenshot when the active path actually
            # consumes images. Search mode and the S-side of routed modes hide
            # the browser context (_hides_browser_context), so a captured shot
            # would never be attached to the decision model anyway -- skipping
            # it avoids a wasted (and hang-prone, see :8002 vision) screenshot
            # and keeps the observation text honest (no "will be attached next
            # turn" promise on the S side, where nothing is attached).
            if screenshot_url and not self._hides_browser_context():
                screenshot_result = await self.web.capture_url_screenshot(screenshot_url, step)
                self.current_visit_screenshot_path = screenshot_result.get("img_path", "")
                self.current_visit_screenshot_url = (
                    screenshot_result.get("url", "") or screenshot_url
                )

                if self.current_visit_screenshot_path:
                    observation += (
                        "\n\nVisit screenshot result:\n"
                        "- status: success\n"
                        f"- url: {self.current_visit_screenshot_url}\n"
                        "- note: The screenshot of this visited page will be attached in the next model turn as additional evidence. It does not change the current browser state."
                    )
                else:
                    observation += (
                        "\n\nVisit screenshot result:\n"
                        "- status: failure\n"
                        f"- url: {screenshot_result.get('url', screenshot_url)}\n"
                        f"- error: {screenshot_result.get('error', 'Unknown screenshot failure')}"
                    )
            return observation

        if name == "code_sandbox":
            attachment_files, attachment_note = self._sandbox_attachment_files()
            sandbox_files = {
                **attachment_files,
                **(arguments.get("files", {}) or {}),
            }
            observation = await asyncio.to_thread(
                self.code_sandbox_tool.call,
                arguments["code"],
                arguments.get("language", "python"),
                arguments.get("stdin"),
                arguments.get("run_timeout", 10),
                arguments.get("compile_timeout", 10),
                sandbox_files,
                arguments.get("fetch_files", []),
            )
            if attachment_note:
                return f"{attachment_note}\n\n{observation}"
            return observation

        if name in {"switch_tab", "close_tab"}:
            if self.web is None:
                return f"[{name}] No browser available in current agent mode."
            try:
                if name == "switch_tab":
                    diagnostics = await self.web.switch_tab(arguments["index"]) or {}
                else:
                    diagnostics = await self.web.close_tab(arguments["index"]) or {}
            except Exception as exc:
                return self._format_action_observation(
                    name=name,
                    status="failure",
                    started_url=started_url,
                    ended_url=getattr(self.web.page, "url", ""),
                    action_html="",
                    diagnostics={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            return self._format_action_observation(
                name=name,
                status="success",
                started_url=started_url,
                ended_url=self.web.page.url,
                action_html="",
                diagnostics=diagnostics,
            )

        if state is None and name in {"hover", "click", "double_click", "fill", "select_option", "scroll", "drag", "press_key"}:
            if self.agent_mode == "operation":
                next_step = "Use goto or new_tab first to navigate to a page before interacting."
            else:
                next_step = "Use search, scholar, visit, code_sandbox, goto, or new_tab first to obtain or navigate to a page before interacting."
            return (
                f"[{name}] No current browser page state is available yet. "
                + next_step
            )

        action_html = ""
        diagnostics: dict = {}
        try:
            action_html = await self._capture_action_html(state, arguments)
            if name == "click":
                x, y = self._scale_xy(float(arguments["x"]), float(arguments["y"]))
                diagnostics = await self.web.click_at(x, y) or {}
            elif name == "double_click":
                x, y = self._scale_xy(float(arguments["x"]), float(arguments["y"]))
                diagnostics = await self.web.double_click_at(x, y) or {}
            elif name == "select_option":
                x, y = self._scale_xy(float(arguments["x"]), float(arguments["y"]))
                try:
                    diagnostics = await self.web.select_option_at(x, y, arguments["value"]) or {}
                except SelectFallbackRequired as exc:
                    # Element isn't a native <select>; runtime already issued the click that
                    # opened the popup. Surface this to the agent so it can click the option
                    # from the new screenshot.
                    diagnostics = {"select_fallback": str(exc)}
            elif name == "hover":
                x, y = self._scale_xy(float(arguments["x"]), float(arguments["y"]))
                diagnostics = await self.web.hover_at(x, y) or {}
            elif name == "fill":
                x, y = self._scale_xy(float(arguments["x"]), float(arguments["y"]))
                diagnostics = await self.web.type_text_at(
                    x,
                    y,
                    arguments.get("text", ""),
                    press_enter=bool(arguments.get("submit", False)),
                ) or {}
            elif name == "scroll":
                direction = arguments["direction"]
                if "x" in arguments and "y" in arguments:
                    x, y = self._scale_xy(float(arguments["x"]), float(arguments["y"]))
                    diagnostics = await self.web.scroll_at(x, y, direction) or {}
                else:
                    diagnostics = await self.web.scroll_document(direction) or {}
            elif name == "drag":
                fx, fy = self._scale_xy(float(arguments["from_x"]), float(arguments["from_y"]))
                tx, ty = self._scale_xy(float(arguments["to_x"]), float(arguments["to_y"]))
                hold_ms = int(arguments.get("hold_ms") or 0)
                diagnostics = await self.web.drag_from_to(fx, fy, tx, ty, hold_ms=hold_ms) or {}
            elif name == "wait":
                await asyncio.sleep(arguments.get("time", 5))
            elif name == "go_back":
                diagnostics = await self.web.go_back() or {}
            elif name == "goto":
                await self.web.navigate(arguments["url"])
            elif name == "new_tab":
                diagnostics = await self.web.open_tab(arguments["url"]) or {}
            elif name == "press_key":
                diagnostics = await self.web.press_keys(arguments["keys"]) or {}
            else:
                raise ValueError(f"Unsupported action: {name}")
        except Exception as exc:
            return self._format_action_observation(
                name=name,
                status="failure",
                started_url=started_url,
                ended_url=getattr(self.web.page, "url", ""),
                action_html=action_html,
                diagnostics=diagnostics,
                error=f"{type(exc).__name__}: {exc}",
            )

        ended_url = self.web.page.url
        return self._format_action_observation(
            name=name,
            status="success",
            started_url=started_url,
            ended_url=ended_url,
            action_html=action_html,
            diagnostics=diagnostics,
        )

    async def _current_search_state(self) -> dict | None:
        current_url = (getattr(self.web.page, "url", "") or "").strip()
        if not current_url or current_url == "about:blank":
            return None
        try:
            current_title = await self.web.page.title()
        except Exception:
            current_title = ""
        return {
            "img_path": "",
            "current_url": current_url,
            "current_title": current_title,
        }

    async def run(self) -> dict:
        termination = "max_steps_exceeded"
        prediction = ""

        if self.agent_mode == "hybrid_guided_routed":
            await self._guided_initialize_router()

        for step in range(1, self.args.max_iter + 1):
            # Pre-step routing: refresh the hidden subgoal / active_mode using
            # the PREVIOUS step's action+observation BEFORE the decision model
            # acts this step. May flip active_mode, toggle _guided_force_answer,
            # and append a synthetic switch record -- so it must run before we
            # compute public_step / state / is_final (all depend on the live
            # active_mode and records length).
            if self.agent_mode == "hybrid_guided_routed":
                await self._guided_route_before_step()
            public_step = len(self.records) + 1 if self.agent_mode == "hybrid_guided_routed" else step
            if self.agent_mode == "search":
                state = None
            elif self.agent_mode in {"hybrid_routed", "hybrid_guided_routed"} and self.active_mode == "S":
                # Routed S-side: no browser state shown to the model regardless
                # of whether the browser has a loaded page (search prompts do
                # not reference current_url/title/screenshot).
                state = None
            elif (
                self.agent_mode in ("hybrid", "hybrid_routed", "hybrid_guided_routed")
                and step == 1
                and (
                    not self._has_explicit_initial_page()
                    or getattr(self.web, "initial_navigation_failed", False)
                )
            ):
                state = None
            else:
                state = await self.web.current_state(public_step)
            is_final = step == self.args.max_iter or (
                self.agent_mode == "hybrid_guided_routed" and self._guided_force_answer
            )
            prompt_text = self._build_prompt_text(state, is_final=is_final)
            messages = self._build_messages(state, prompt_text)

            raw_response = ""
            parsed = None
            parse_status = "No model output."
            for _ in range(self.args.max_format_retries):
                raw_response = await self._achat(messages)
                parsed, parse_status = self._parse_response(
                    raw_response,
                    require_answer=is_final,
                )
                if (
                    parsed is not None
                    and self.agent_mode == "hybrid_guided_routed"
                    and parsed.get("answer")
                    and not self._guided_force_answer
                    and not is_final
                ):
                    # Don't flat-reject a confident early answer — the router
                    # state may simply be stale (it lags one observation). Let
                    # the router adjudicate; acceptance avoids format_error
                    # deaths on runs the model actually finished.
                    accept, why = await self._guided_adjudicate_early_answer(
                        parsed["answer"])
                    if accept:
                        self._guided_force_answer = True
                        if self._guided_router_state is not None:
                            self._guided_router_state["done"] = True
                            self._guided_router_state["remaining_subgoals"] = []
                    else:
                        parsed = None
                        parse_status = (
                            "Guided router rejected the early answer: "
                            f"{why or 'an active subgoal remains'}. Do not "
                            "answer yet; choose one allowed tool for the "
                            "current guided subgoal."
                        )
                if parsed is not None and self.agent_mode == "hybrid_guided_routed":
                    parsed = self._guided_patch_parsed_report(
                        parsed,
                        require_answer=is_final,
                    )
                if parsed is not None and parsed["tool_call"] is not None:
                    label_status = self._validate_tool_call_against_state(state, parsed["tool_call"])
                    if label_status is not None:
                        parsed = None
                        parse_status = label_status
                if parsed is not None:
                    break
                messages = copy.deepcopy(messages)
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": raw_response}],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "FORMAT ERROR: "
                                    f"{parse_status} "
                                    "Regenerate the response using exactly the required tags."
                                ),
                            }
                        ],
                    }
                )

            step_record = {
                "step": public_step,
                "current_url": state.get("current_url", "") if state else "",
                "current_title": state.get("current_title", "") if state else "",
                "img_path": state.get("img_path", "") if state else "",
                "prompt_text": prompt_text,
                # Persist the exact system prompt that was sent for THIS step.
                # _build_messages() rendered it from the same self.active_mode we
                # still hold here. For hybrid_routed the model's own `switch`
                # flips active_mode later (at the switch-intercept block); for
                # hybrid_guided_routed the pre-step router already set the side
                # at the top of this iteration. Either way self.active_mode here
                # is the side this step actually used. Storing it lets the viewer
                # show the real prompt directly instead of re-deriving it — and
                # it is correct for every path, including the answer/terminal step.
                "system_prompt": get_system_prompt(
                    self.agent_mode, self.active_mode,
                    expose_switch=self._guided_expose_switch(),
                ),
                "raw_response": raw_response,
            }

            # Stamp this step with the result of the pre-step routing call that
            # selected its subgoal / active_mode (guided mode only).
            if self.agent_mode == "hybrid_guided_routed" and self._guided_last_route_info:
                for k, v in self._guided_last_route_info.items():
                    step_record[k] = v
                self._guided_last_route_info = None

            if parsed is None:
                step_record["observation"] = parse_status
                self.records.append(step_record)
                termination = "format_error"
                self._write_live_result_checkpoint(termination=termination, prediction=prediction)
                break

            # Normalize the persisted report to pretty-printed JSON when we
            # have a parsed dict, so the viewer / Stage 1.2 / next-turn
            # `last_report` injection all see consistent, readable JSON.
            report_pretty = parsed["report"]
            if parsed.get("report_parsed") is not None:
                try:
                    report_pretty = json.dumps(
                        parsed["report_parsed"], ensure_ascii=False, indent=2,
                    )
                except Exception:
                    report_pretty = parsed["report"]
            self.last_report = report_pretty
            step_record["report"] = report_pretty
            if parsed.get("report_parsed") is not None:
                step_record["report_parsed"] = parsed["report_parsed"]

            if parsed["answer"]:
                prediction = parsed["answer"]
                step_record["answer"] = parsed["answer"]
                step_record["observation"] = ""
                # The answer step IS a real model call — stamp the active mode /
                # model it used before breaking out, mirroring the non-answer
                # path below. Without this the terminal record has
                # active_mode_before=None / step_model=None, which makes the
                # viewer fall back to the generic system prompt that was never
                # actually used.
                step_record["active_mode_before"] = self.active_mode
                _, answer_step_model = self._active_client_and_model()
                step_record["step_model"] = answer_step_model
                step_record["active_mode_after"] = self.active_mode
                self.records.append(step_record)
                termination = "answer"
                # Guided mode: one extra router call to judge the whole run
                # now that the final answer exists. Stamped on the terminal
                # record + surfaced in the result dict.
                if self.agent_mode == "hybrid_guided_routed":
                    final_verdict = await self._guided_final_verdict(prediction)
                    if final_verdict is not None:
                        self._guided_final_verdict_result = final_verdict
                        step_record["guided_final_verdict"] = final_verdict
                self._write_live_result_checkpoint(termination=termination, prediction=prediction)
                break

            tool_call = parsed["tool_call"]
            step_record["tool_call"] = tool_call
            step_record["action_description"] = parsed.get("action_description") or ""
            # Always stamp the active mode the agent saw when planning this
            # step (None outside routed mode). Lets the viewer / Stage1 split
            # trajectories cleanly into per-side segments.
            step_record["active_mode_before"] = self.active_mode
            _, step_model = self._active_client_and_model()
            step_record["step_model"] = step_model

            # Routed modes: intercept the switch tool BEFORE _execute_tool.
            # No browser action; flip active_mode, record metadata, advance turn.
            # hybrid_routed: the decision model freely chose switch.
            # hybrid_guided_routed: the pre-step router decided to switch and
            #   exposed the switch tool + a switch subgoal this turn, so the
            #   decision model emitted this switch with a continuous report.
            if (
                self.agent_mode in {"hybrid_routed", "hybrid_guided_routed"}
                and tool_call.get("name") == "switch"
            ):
                from_mode = self.active_mode
                pending = self._guided_pending_switch if self.agent_mode == "hybrid_guided_routed" else None
                to_mode = (pending or {}).get("to") or ("O" if from_mode == "S" else "S")
                reason = (tool_call.get("arguments", {}) or {}).get("reason", "")
                step_record["is_switch_step"] = True
                step_record["switch_from"] = from_mode
                step_record["switch_to"] = to_mode
                step_record["switch_reason"] = reason
                if self.agent_mode == "hybrid_guided_routed":
                    step_record["guided_router_switch"] = True
                from_name = "search-based" if from_mode == "S" else "operation-based"
                to_name = "search-based" if to_mode == "S" else "operation-based"
                observation = (
                    f"[switch] Control handed off from {from_name} path to {to_name} path. "
                    "The switch itself does not change the browser — the current page remains exactly as it was before this switch. "
                    "`cumulative_progress` and `remaining_subgoals` are preserved verbatim."
                )
                self.last_observation = observation
                step_record["observation"] = observation
                # Drop any stale visit screenshots — switching paths.
                self.last_visit_screenshot_path = ""
                self.last_visit_screenshot_url = ""
                if pending is not None:
                    # Guided: pop the head `switch` entry from remaining_subgoals
                    # and re-derive state so the new side activates with the next
                    # (post-switch) task subgoal exposed.
                    self._guided_consume_switch_entry()
                    self._guided_pending_switch = None
                else:
                    # hybrid_routed: plain flip.
                    self.active_mode = to_mode
                    self.allowed_tool_names = get_allowed_tool_names(
                        self.agent_mode, self.active_mode
                    )
                self.switch_count += 1
                step_record["active_mode_after"] = self.active_mode
                self.records.append(step_record)
                # Do NOT stash an action ctx here. The router already decided the
                # post-switch state (applied above), so the NEXT iteration's
                # pre-step router must NOT run (prev_ctx stays None -> it no-ops)
                # and the decision model acts directly on the post-switch subgoal
                # on the new side. The ctx will be stashed again after that first
                # real post-switch action, resuming normal pre-step routing.
                self._guided_pending_action_ctx = None
                self._write_live_result_checkpoint(termination="running", prediction=prediction)
                continue

            self.current_summary_traces = []
            observation = await self._execute_tool(state, tool_call, public_step)
            self.last_observation = observation
            step_record["observation"] = observation
            # Active mode is unchanged by non-switch actions; record it so
            # downstream tooling never has to infer it.
            step_record["active_mode_after"] = self.active_mode
            if tool_call["name"] == "visit":
                step_record["summary_trace"] = copy.deepcopy(self.current_summary_traces)
                step_record["visit_screenshot_path"] = self.current_visit_screenshot_path
                step_record["visit_screenshot_url"] = self.current_visit_screenshot_url
                self.last_visit_screenshot_path = self.current_visit_screenshot_path
                self.last_visit_screenshot_url = self.current_visit_screenshot_url
            else:
                self.last_visit_screenshot_path = ""
                self.last_visit_screenshot_url = ""
            self.records.append(step_record)
            if self.agent_mode == "hybrid_guided_routed":
                # Stash this step's action context for the NEXT step's pre-step
                # routing call to digest. The router is no longer invoked here
                # (post-action); it runs at the top of the next iteration so the
                # refreshed subgoal reflects this observation before the decision
                # model acts again. browser_state is captured only for O actions
                # (S-side passes None -> router sees observation text only).
                browser_state = await self._guided_post_action_browser_state(
                    step_record.get("active_mode_before")
                )
                self._guided_pending_action_ctx = {
                    "step": step_record.get("step"),
                    "active_mode": step_record.get("active_mode_before"),
                    "tool_call": step_record.get("tool_call"),
                    "action_description": step_record.get("action_description") or "",
                    "report_parsed": step_record.get("report_parsed"),
                    "observation": observation,
                    "browser_state": browser_state,
                }
            self._write_live_result_checkpoint(termination="running", prediction=prediction)

        result = {
            "question": self.task,
            "prediction": prediction,
            "termination": termination,
            "records": self.records,
            "usage": self.usage,
        }
        final_verdict = getattr(self, "_guided_final_verdict_result", None)
        if final_verdict is not None:
            result["guided_final_verdict"] = final_verdict
        return result
