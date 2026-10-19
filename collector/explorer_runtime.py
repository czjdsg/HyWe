"""ExplorerRuntime: a slim wrapper around HybridPlaywrightComputer + SearchTool + VisitTool.

It exposes 4 async methods (start / state / act / close) so an external explorer
loop can drive the same low-level tooling that hybrid_routed uses in production.
This is the synth3-B path: we reuse the production browser + S-side clients but
write our own ReAct loop on top so the agent can EXPLORE freely rather than be
locked into solving a specific task.
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

# Make sure we can import the HybridWebAgent inference package.
_INFERENCE_DIR = Path(__file__).resolve().parent.parent / "inference"
sys.path.insert(0, str(_INFERENCE_DIR))

from hybrid_runtime import HybridPlaywrightComputer  # noqa: E402
from hybrid_search import SearchTool  # noqa: E402
from hybrid_visit import VisitTool  # noqa: E402
from hybrid_scholar import ScholarTool  # noqa: E402
from hybrid_code_sandbox import CodeSandboxTool  # noqa: E402

try:
    # select_option_at raises this when the element isn't a native <select>.
    from hybrid_runtime import SelectFallbackRequired  # noqa: E402
except Exception:  # pragma: no cover - older runtime without the sentinel
    class SelectFallbackRequired(Exception):
        pass


def coord_scale_for_model(model: str) -> int:
    """Return the coordinate scale to use given a vision model name.

    Different VLM families train on different coordinate spaces:
      - kimi-k2.x   : RELATIVE [0.0, 1.0]                → scale=1
      - qwen-vl-2.5+: 0..1000 INTEGER logical pixels     → scale=1000
      - Claude (sonnet/opus, computer-use): RAW PIXELS   → scale=0 pass-through
      - GPT-4V / 4o : RAW PIXELS                         → scale=0 pass-through

    scale rule in HybridPlaywrightComputer / explorer_runtime._coord_in_viewport:
      if scale > 0:  x_pixel = x_input * window_width / scale
      else:          x_pixel = x_input          (assumed already pixels)

    Default for unknown models is 1 (relative 0-1), the most common modern
    VLM convention and the safest fallback because out-of-range pixel
    coordinates would error out loudly instead of silently misclicking.
    """
    m = (model or "").lower()
    if "kimi" in m:
        return 1                    # 0-1 relative floats
    if "qwen" in m or "hywe" in m:
        return 1000                 # 0-1000 integer logical pixels
                                    #   (qwen-VL, qwen3.7-plus, qwen3.7-max
                                    #    all emit on a 1000x1000 grid)
    if any(x in m for x in ("claude", "sonnet", "opus", "haiku")):
        return 0                    # raw pixels (Anthropic computer-use)
    if any(x in m for x in ("gpt-4v", "gpt-4o", "gpt4v", "gpt4o")):
        return 0                    # raw pixels (OpenAI vision)
    return 1                        # default: relative 0-1


def _mk_args(*, web: str, task_dir: str, window=(1280, 720),
             coord_scale: int | None = None, model: str = ""):
    """Build a SimpleNamespace that satisfies HybridPlaywrightComputer's expectations.

    If coord_scale is None, derive it from the explorer model name (see
    coord_scale_for_model). Explicit override always wins.
    """
    if coord_scale is None:
        coord_scale = coord_scale_for_model(
            model or os.environ.get("EXPLORER_MODEL", "kimi-k3"))
    return SimpleNamespace(
        agent_mode="operation",        # so blank web is allowed via google home
        web=web,
        window_width=window[0],
        window_height=window[1],
        coord_scale=coord_scale,
        task_dir=task_dir,
        close_new_tabs=False,
        safe_load_timeout_ms=4000,
        headless=True,
        proxy_url=os.environ.get("PROXY_URL", ""),
    )


class ExplorerRuntime:
    """Thin layer that runs a single chromium + S tools, persistent across calls."""

    def __init__(self, *, entry_url: str, task_dir: str, jina_summarize_fn=None,
                 summarize_max_tokens: int = 1500):
        self.entry_url = entry_url or "about:blank"
        self.task_dir = task_dir
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(os.path.join(self.task_dir, "trajectory"), exist_ok=True)

        self._args = _mk_args(web=self.entry_url, task_dir=self.task_dir)
        self.web = HybridPlaywrightComputer(self._args)

        # S-side tools
        self.search = SearchTool()
        # VisitTool needs a summarize callable. By default we hook up an LLM
        # summary (same shape as production hybrid_agent uses) — see
        # _make_llm_summarize. The user can override with their own fn.
        if jina_summarize_fn is None:
            jina_summarize_fn = self._make_llm_summarize(
                summarize_max_tokens=summarize_max_tokens
            )
        self.visit = VisitTool(
            summarize_fn=jina_summarize_fn,
            tokenizer_path=os.environ.get("TOKENIZER_PATH", ""),
            max_webpage_tokens=8000,
        )
        # scholar (academic search) and code_sandbox round out the S-side and
        # O-side tool catalogues that hybrid_routed exposes. Without these the
        # runtime rejected legitimate LLM tool calls as "unknown tool".
        self.scholar = ScholarTool()
        self.code_sandbox = CodeSandboxTool(
            base_url=os.environ.get("CODE_SANDBOX_BASE_URL",
                                    "http://127.0.0.1:8080"),
        )

        # Dual-track step counters: operator captures use op_step,
        # extractor captures use ext_step. Each role increments only its
        # own counter — so screenshots align 1:1 with the agent that took
        # them (op_<N>_full_screenshot.png vs ext_<N>_full_screenshot.png).
        self.op_step = 0
        self.ext_step = 0
        # Legacy field kept for any downstream code that read self.step.
        # Always == max(op_step, ext_step).
        self.step = 0
        self.history: list[dict] = []

    # ---------------------- visit summarizer ----------------------
    @staticmethod
    def _make_llm_summarize(*, summarize_max_tokens: int = 1500):
        """Return a `summarize_fn(prompt_text, image_data_urls)` that calls an
        OpenAI-compatible chat endpoint. Configured via env vars — same defaults
        as production hybrid_agent (deepseek-v4-flash on :8002).

        Falls back to an identity-truncate if the LLM call fails so the visit
        tool never wedges the explorer.
        """
        base_url = os.environ.get(
            "SUMMARY_BASE_URL", "http://localhost:8000/v1"
        ).rstrip("/")
        api_key = os.environ.get(
            "SUMMARY_API_KEY", os.environ.get("EXPLORER_API_KEY", "EMPTY")
        )
        model = os.environ.get("SUMMARY_MODEL", "kimi-k3")
        max_tokens = int(os.environ.get(
            "VISIT_SUMMARY_MAX_TOKENS", str(max(4000, summarize_max_tokens * 4))
        ))
        timeout = int(os.environ.get("SUMMARY_TIMEOUT", "900"))
        system_msg = "You extract useful page information for a web agent."

        def _summarize(prompt_text: str, image_data_urls=None) -> str:
            # Build OpenAI Chat Completions payload — matches production format.
            user_content: list[dict] = [{"type": "text", "text": prompt_text}]
            for url in image_data_urls or []:
                user_content.append({"type": "image_url", "image_url": {"url": url}})
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=_json.dumps(payload).encode("utf-8"),
                method="POST",
            )
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    d = _json.loads(body)
                    return d["choices"][0]["message"]["content"]
            except Exception as exc:
                # Identity-truncate fallback so the explorer keeps moving.
                # Surface the failure inline so it shows up in observations.
                fallback = prompt_text[: summarize_max_tokens * 4]
                return f"[summary-fallback: {type(exc).__name__}: {exc}]\n\n{fallback}"

        return _summarize

    # ---------------------- lifecycle ----------------------
    async def start(self):
        await self.web.__aenter__()
        # Navigate to entry_url if it's something specific (web=="about:blank" was the default).
        if self.entry_url and self.entry_url != "about:blank":
            try:
                await self.web.navigate(self.entry_url)
            except Exception:
                pass

    async def close(self):
        try:
            await self.web.__aexit__(None, None, None)
        except Exception:
            pass

    # ---------------------- state -------------------------
    async def state(self, role: str = "op") -> dict:
        """Return current page state including screenshot path + dom snippet.

        `role` controls which counter is bumped AND which filename prefix is
        used for the screenshot, so operator and fact-extractor get their own
        independent series:
          role="op"  → op_<N>_full_screenshot.png  (N == op_step after bump)
          role="ext" → ext_<N>_full_screenshot.png (N == ext_step after bump)
        Any other role falls back to the legacy 'step' counter without a
        prefix, preserving v7 behavior where there's only one screenshot
        stream.
        """
        if role == "op":
            self.op_step += 1
            self.step = max(self.step, self.op_step)
            st = await self.web.current_state(self.op_step, prefix="op_")
            return st
        if role == "ext":
            self.ext_step += 1
            self.step = max(self.step, self.ext_step)
            st = await self.web.current_state(self.ext_step, prefix="ext_")
            return st
        # legacy
        self.step += 1
        st = await self.web.current_state(self.step)
        return st

    # ---------------------- action ------------------------
    @staticmethod
    def _num(v):
        """Coerce a coordinate-ish value to float, tolerating the malformed
        shapes LLMs emit: [x] / [x, y] lists, numeric strings, etc.
        Raises ValueError with a clean message instead of KeyError/TypeError."""
        if isinstance(v, (list, tuple)):
            if not v:
                raise ValueError("empty coordinate list")
            v = v[0]
        if isinstance(v, bool):  # bool is an int subclass; reject explicitly
            raise ValueError("coordinate must be a number, got bool")
        return float(v)

    def _require_xy(self, args):
        """Pull x/y from args with friendly errors (KeyError -> ValueError)."""
        if "x" not in args or "y" not in args:
            missing = [k for k in ("x", "y") if k not in args]
            raise ValueError(f"missing coordinate(s): {', '.join(missing)}")
        return self._num(args["x"]), self._num(args["y"])

    def _coord_in_viewport(self, x, y):
        scale = self._args.coord_scale
        # kimi-k2.6 emits coords in EITHER 0-1 relative OR 0-1000 logical
        # pixels. When configured for the relative convention (scale==1),
        # auto-detect per call: any coord >1 must be the 0-1000 grid. Other
        # conventions (1000=qwen, 0=raw passthrough) are left untouched.
        if scale == 1 and max(abs(x), abs(y)) > 1.0:
            scale = 1000
        if scale > 0:
            return (x * self._args.window_width / scale,
                    y * self._args.window_height / scale)
        return (x, y)

    async def act(self, tool_call: dict) -> dict:
        """Execute a tool_call dict {name, arguments}. Returns observation dict."""
        name = tool_call.get("name")
        args = tool_call.get("arguments") or {}
        started_url = ""
        try:
            started_url = self.web.page.url
        except Exception:
            pass
        rec = {"step": self.step, "tool": name, "arguments": args,
               "started_url": started_url, "status": "success",
               "error": "", "diagnostics": {}, "ended_url": ""}
        try:
            if name == "click":
                x, y = self._coord_in_viewport(*self._require_xy(args))
                rec["diagnostics"] = await self.web.click_at(x, y) or {}
            elif name == "double_click":
                x, y = self._coord_in_viewport(*self._require_xy(args))
                rec["diagnostics"] = await self.web.double_click_at(x, y) or {}
            elif name == "hover":
                x, y = self._coord_in_viewport(*self._require_xy(args))
                rec["diagnostics"] = await self.web.hover_at(x, y) or {}
            elif name == "fill":
                x, y = self._coord_in_viewport(*self._require_xy(args))
                rec["diagnostics"] = await self.web.type_text_at(
                    x, y, args.get("text", ""), press_enter=bool(args.get("submit", False))
                ) or {}
            elif name == "select_option":
                x, y = self._coord_in_viewport(*self._require_xy(args))
                value = args.get("value", args.get("option", ""))
                try:
                    rec["diagnostics"] = await self.web.select_option_at(x, y, value) or {}
                except SelectFallbackRequired as exc:
                    # Not a native <select>; the runtime already clicked to open
                    # the popup. Tell the agent to pick the option from the new
                    # screenshot instead of failing the step.
                    rec["diagnostics"] = {"select_fallback": str(exc)}
            elif name == "scroll":
                direction = args.get("direction", "down")
                if "x" in args and "y" in args:
                    x, y = self._coord_in_viewport(*self._require_xy(args))
                    rec["diagnostics"] = await self.web.scroll_at(x, y, direction) or {}
                else:
                    rec["diagnostics"] = await self.web.scroll_document(direction) or {}
            elif name == "drag":
                miss = [k for k in ("from_x","from_y","to_x","to_y") if k not in args]
                if miss:
                    raise ValueError(f"drag missing: {', '.join(miss)}")
                fx, fy = self._coord_in_viewport(
                    self._num(args["from_x"]), self._num(args["from_y"]))
                tx, ty = self._coord_in_viewport(
                    self._num(args["to_x"]), self._num(args["to_y"]))
                hold_ms = int(args.get("hold_ms") or 0)
                rec["diagnostics"] = await self.web.drag_from_to(
                    fx, fy, tx, ty, hold_ms=hold_ms) or {}
            elif name == "goto":
                await self.web.navigate(args["url"])
            elif name == "new_tab":
                rec["diagnostics"] = await self.web.open_tab(args["url"]) or {}
            elif name == "switch_tab":
                rec["diagnostics"] = await self.web.switch_tab(int(args["index"])) or {}
            elif name == "close_tab":
                rec["diagnostics"] = await self.web.close_tab(int(args["index"])) or {}
            elif name == "wait":
                await asyncio.sleep(float(args.get("time", 3)))
            elif name == "go_back":
                rec["diagnostics"] = await self.web.go_back() or {}
            elif name == "press_key":
                rec["diagnostics"] = await self.web.press_keys(args["keys"]) or {}
            elif name in ("search", "scholar"):
                q = args.get("query")
                if isinstance(q, str):
                    q = [q]
                if not isinstance(q, list) or not any(
                        isinstance(s, str) and s.strip() for s in q):
                    raise ValueError(f"{name} requires a non-empty query list")
                q = [s.strip() for s in q if isinstance(s, str) and s.strip()]
                client = self.scholar if name == "scholar" else self.search
                rec["diagnostics"] = {"result_text": client.call(q)}
            elif name == "visit":
                url = args.get("url")
                if isinstance(url, str):
                    url = url.strip()
                if not url:
                    # default to the current page, mirroring production behavior
                    try:
                        url = self.web.page.url
                    except Exception:
                        url = ""
                if not url or url == "about:blank":
                    raise ValueError("visit requires a url (or a loaded page)")
                rec["diagnostics"] = {"result_text": self.visit.call(
                    url, args.get("goal", "general info"),
                )}
            elif name == "code_sandbox":
                code = args.get("code")
                if not isinstance(code, str) or not code.strip():
                    raise ValueError("code_sandbox requires non-empty string code")
                out = await asyncio.to_thread(
                    self.code_sandbox.call,
                    code,
                    args.get("language", "python") or "python",
                    args.get("stdin"),
                    args.get("run_timeout", 10),
                    args.get("compile_timeout", 10),
                    args.get("files", {}) or {},
                    args.get("fetch_files", []),
                )
                rec["diagnostics"] = {"result_text": out}
            else:
                rec["status"] = "failure"
                rec["error"] = f"unknown tool: {name}"
        except Exception as exc:
            rec["status"] = "failure"
            rec["error"] = f"{type(exc).__name__}: {exc}"
        try:
            rec["ended_url"] = self.web.page.url
        except Exception:
            pass
        self.history.append(rec)
        return rec
