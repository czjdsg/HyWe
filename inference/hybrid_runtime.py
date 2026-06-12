import asyncio
import base64
import collections
import mimetypes
import os
import sys
import textwrap
from pathlib import Path
from typing import Literal

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from hybrid_utils import sanitize_surrogates


class SelectFallbackRequired(Exception):
    """Raised when select_option_at is invoked on a non-native <select> element."""


PLAYWRIGHT_KEY_MAP = {
    "backspace": "Backspace",
    "tab": "Tab",
    "return": "Enter",
    "enter": "Enter",
    "shift": "Shift",
    "control": "ControlOrMeta",
    "ctrl": "ControlOrMeta",
    "alt": "Alt",
    "option": "Alt",
    "escape": "Escape",
    "esc": "Escape",
    "space": "Space",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "end": "End",
    "home": "Home",
    "left": "ArrowLeft",
    "arrowleft": "ArrowLeft",
    "up": "ArrowUp",
    "arrowup": "ArrowUp",
    "right": "ArrowRight",
    "arrowright": "ArrowRight",
    "down": "ArrowDown",
    "arrowdown": "ArrowDown",
    "insert": "Insert",
    "delete": "Delete",
    "command": "Meta",
    "cmd": "Meta",
    "meta": "Meta",
    "win": "Meta",
    "windows": "Meta",
    "super": "Meta",
}

DEFAULT_BROWSER_FALLBACK_URL = "https://www.google.com/"


def image_path_to_data_url(image_path: str) -> str:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


class HybridPlaywrightComputer:
    def __init__(self, args):
        self.args = args
        self._agent_mode = getattr(args, "agent_mode", "hybrid")
        initial_url = args.web or ""
        if self._agent_mode == "operation" and not initial_url:
            initial_url = "https://www.google.com/"
        elif self._agent_mode == "search":
            initial_url = ""
        self._initial_url = initial_url or "about:blank"
        self.initial_navigation_failed = False
        self._screen_size = (args.window_width, args.window_height)
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._storage_state_path = str(Path(args.task_dir) / "storage_state.json")
        self._download_dir: Path = Path(args.task_dir) / "downloads"
        self._console_messages: collections.deque = collections.deque(maxlen=50)
        self._page_errors: collections.deque = collections.deque(maxlen=20)
        # Each entry: {"path": "<abs path on disk>", "filename": "<original name>",
        #              "url": "<source url>"}; surfaced to the agent in observations.
        self._downloaded_files: collections.deque = collections.deque(maxlen=20)
        self._close_new_tabs = bool(getattr(args, "close_new_tabs", False))
        # When > 0, the next N "page" events from the BrowserContext are
        # treated as agent-initiated (via `open_tab`) and the auto-handler
        # skips them. Avoids a race where the handler closes / re-switches the
        # tab the agent just opened.
        self._suppress_next_page_events: int = 0
        self._safe_load_timeout_ms = int(
            getattr(args, "safe_load_timeout_ms", 3000) or 3000
        )

    def _context_kwargs(self, storage_state=None) -> dict:
        return {
            "viewport": {"width": self._screen_size[0], "height": self._screen_size[1]},
            "device_scale_factor": 1,
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "storage_state": storage_state,
        }

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser page is not initialized.")
        return self._page

    async def _safe_wait_load(self, timeout_ms: int | None = None) -> None:
        timeout = timeout_ms if timeout_ms is not None else self._safe_load_timeout_ms
        try:
            await self.page.wait_for_load_state("load", timeout=timeout)
        except Exception:
            pass

    def _on_dialog(self, dialog) -> None:
        asyncio.create_task(self._dismiss_dialog(dialog))

    async def _dismiss_dialog(self, dialog) -> None:
        try:
            await dialog.dismiss()
        except Exception:
            pass

    def _on_pageerror(self, error) -> None:
        try:
            text = sanitize_surrogates(str(error))[:500]
        except Exception:
            return
        self._page_errors.append(text)

    def _on_console(self, message) -> None:
        try:
            if (message.type or "").lower() != "error":
                return
            text = sanitize_surrogates(message.text)[:500]
        except Exception:
            return
        self._console_messages.append(text)

    def _on_download(self, download) -> None:
        # Fires when chromium accepts a download (e.g., after the agent clicked a
        # "Download PDF" link). Save into <task_dir>/downloads/<suggested_filename>
        # and record the absolute path so the agent sees it in its next observation.
        asyncio.create_task(self._save_download(download))

    async def _save_download(self, download) -> None:
        try:
            self._download_dir.mkdir(parents=True, exist_ok=True)
            suggested = (download.suggested_filename or "download.bin").strip() or "download.bin"
            # avoid clobbering an earlier same-name download
            target = self._download_dir / suggested
            if target.exists():
                stem = target.stem
                suf  = target.suffix
                idx  = 1
                while True:
                    candidate = self._download_dir / f"{stem}__{idx}{suf}"
                    if not candidate.exists():
                        target = candidate
                        break
                    idx += 1
            await download.save_as(str(target))
            self._downloaded_files.append({
                "path": str(target.resolve()),
                "filename": suggested,
                "url": download.url or "",
            })
        except Exception as exc:
            # surface the failure to the agent rather than swallowing
            try:
                self._page_errors.append(
                    f"download failed for {download.suggested_filename!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
            except Exception:
                pass

    def _attach_page_listeners(self, page: Page) -> None:
        page.on("dialog", self._on_dialog)
        page.on("pageerror", self._on_pageerror)
        page.on("console", self._on_console)
        page.on("download", self._on_download)

    def consume_recent_messages(self, max_items: int = 5) -> dict:
        console = list(self._console_messages)[-max_items:]
        page_errors = list(self._page_errors)[-max_items:]
        downloads = list(self._downloaded_files)[-max_items:]
        self._console_messages.clear()
        self._page_errors.clear()
        self._downloaded_files.clear()
        return {"console_errors": console, "page_errors": page_errors, "downloads": downloads}

    async def _handle_new_page(self, new_page: Page):
        if self._suppress_next_page_events > 0:
            # Agent-initiated tab via `open_tab` — skip auto-handling.
            self._suppress_next_page_events -= 1
            return
        if self._close_new_tabs:
            try:
                new_url = (new_page.url or "").strip()
                await new_page.close()
                if (
                    not new_url
                    or new_url == "about:blank"
                    or new_url.startswith("chrome-error://")
                ):
                    return
                await self.page.goto(new_url, wait_until="domcontentloaded")
            except Exception:
                return
            return

        try:
            await new_page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        try:
            new_url = (new_page.url or "").strip()
        except Exception:
            new_url = ""
        if (
            not new_url
            or new_url == "about:blank"
            or new_url.startswith("chrome-error://")
        ):
            try:
                await new_page.close()
            except Exception:
                pass
            return
        self._page = new_page
        self._attach_page_listeners(new_page)

    async def _replace_page(self):
        if self._context is None:
            raise RuntimeError("Browser context is not initialized.")
        if self._page is not None:
            try:
                if not self._page.is_closed():
                    await self._page.close()
            except Exception:
                pass
        self._page = await self._context.new_page()
        self._attach_page_listeners(self._page)

    async def _goto_start_url(self, url: str, wait_until: str = "domcontentloaded") -> bool:
        try:
            await self.page.goto(
                url,
                timeout=60000,
                wait_until=wait_until,
            )
            return True
        except Exception:
            return False

    async def __aenter__(self):
        self._playwright = await async_playwright().start()

        launch_args = ["--disable-blink-features=AutomationControlled"]
        if getattr(self.args, "no_sandbox", True):
            launch_args.append("--no-sandbox")
        if getattr(self.args, "proxy_url", ""):
            launch_args.append(f"--proxy-server={self.args.proxy_url}")

        self._browser = await self._playwright.chromium.launch(
            args=launch_args,
            headless=self.args.headless,
        )

        storage_state = None
        if getattr(self.args, "keep_user_info", False) and os.path.exists(
            self._storage_state_path
        ):
            storage_state = self._storage_state_path

        self._context = await self._browser.new_context(
            **self._context_kwargs(storage_state=storage_state)
        )
        self._page = await self._context.new_page()
        self._attach_page_listeners(self._page)
        if self._initial_url and self._initial_url != "about:blank":
            loaded = await self._goto_start_url(self._initial_url)
            if not loaded:
                self.initial_navigation_failed = True
                await self._replace_page()
                if (
                    self._agent_mode == "operation"
                    and self._initial_url != DEFAULT_BROWSER_FALLBACK_URL
                ):
                    loaded = await self._goto_start_url(
                        DEFAULT_BROWSER_FALLBACK_URL,
                        wait_until="domcontentloaded",
                    )
                if not loaded:
                    await self._replace_page()
                    await self._goto_start_url("about:blank", wait_until="load")
        else:
            await self._goto_start_url("about:blank", wait_until="load")
        self._context.on("page", lambda p: asyncio.create_task(self._handle_new_page(p)))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context and getattr(self.args, "keep_user_info", False):
            await self._context.storage_state(path=self._storage_state_path)
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def current_state(self, step: int, prefix: str = "") -> dict:
        try:
            await self.page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            try:
                await self.page.wait_for_load_state("load", timeout=5000)
            except Exception:
                pass

        await asyncio.sleep(1)

        task_dir = Path(self.args.task_dir)
        screenshot_dir = task_dir / "trajectory"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        img_path = str(screenshot_dir / f"{prefix}{step}_full_screenshot.png")
        try:
            await self.page.screenshot(path=img_path)
        except Exception:
            img_path = ""

        try:
            current_title = await self.page.title()
        except Exception:
            current_title = ""

        current_title = sanitize_surrogates(current_title)
        tabs = sanitize_surrogates(await self.list_tabs())

        return {
            "img_path": img_path,
            "current_url": sanitize_surrogates(self.page.url),
            "current_title": current_title,
            "tabs": tabs,
        }

    async def capture_url_screenshot(self, url: str, step: int) -> dict:
        if not url:
            return {
                "img_path": "",
                "url": "",
                "title": "",
                "error": "Missing URL for visit screenshot.",
            }

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        if self._browser is None or self._context is None:
            return {
                "img_path": "",
                "url": url,
                "title": "",
                "error": "Browser is not initialized.",
            }

        task_dir = Path(self.args.task_dir)
        visit_dir = task_dir / "visit_screenshots"
        visit_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = str(visit_dir / f"visit_step{step}.png")

        storage_state = None
        try:
            storage_state = await self._context.storage_state()
        except Exception:
            storage_state = None

        temp_context = None
        try:
            temp_context = await self._browser.new_context(
                **self._context_kwargs(storage_state=storage_state)
            )
            temp_page = await temp_context.new_page()
            await temp_page.goto(url, timeout=60000, wait_until="domcontentloaded")
            try:
                await temp_page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                try:
                    await temp_page.wait_for_load_state("load", timeout=5000)
                except Exception:
                    pass
            await asyncio.sleep(1)

            try:
                await temp_page.screenshot(path=screenshot_path)
            except Exception:
                return {
                    "img_path": "",
                    "url": url,
                    "title": "",
                    "error": "Failed to capture visit screenshot in viewport mode.",
                }

            try:
                title = await temp_page.title()
            except Exception:
                title = ""

            return {
                "img_path": screenshot_path,
                "url": sanitize_surrogates(temp_page.url),
                "title": sanitize_surrogates(title),
                "error": "",
            }
        except Exception as exc:
            return {
                "img_path": "",
                "url": url,
                "title": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if temp_context is not None:
                await temp_context.close()

    async def extract_page_snapshot(
        self,
        max_text_chars: int = 12000,
        max_links: int = 30,
        max_controls: int = 30,
    ) -> dict:
        payload = await self.page.evaluate(
            """([maxLinks, maxControls]) => {
                const normalize = (s) => String(s || "").replace(/\\s+/g, " ").trim();
                const isVisible = (el) => {
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return (
                        style &&
                        style.visibility !== "hidden" &&
                        style.display !== "none" &&
                        rect.width > 0 &&
                        rect.height > 0
                    );
                };

                const visibleText = normalize(document.body ? document.body.innerText : "");
                const links = [];
                for (const el of Array.from(document.querySelectorAll("a[href]"))) {
                    if (!isVisible(el)) continue;
                    const text = normalize(el.innerText || el.getAttribute("aria-label") || "");
                    const href = el.href || el.getAttribute("href") || "";
                    if (!text && !href) continue;
                    links.push({text, href});
                    if (links.length >= maxLinks) break;
                }

                const controls = [];
                for (const el of Array.from(document.querySelectorAll("input, textarea, select, button"))) {
                    if (!isVisible(el)) continue;
                    const tag = (el.tagName || "").toLowerCase();
                    const type = (el.getAttribute("type") || "").toLowerCase();
                    const label =
                        normalize(el.innerText) ||
                        normalize(el.getAttribute("aria-label")) ||
                        normalize(el.getAttribute("placeholder")) ||
                        normalize(el.getAttribute("name")) ||
                        normalize(el.getAttribute("value"));
                    controls.push({tag, type, label});
                    if (controls.length >= maxControls) break;
                }

                return {
                    url: window.location.href,
                    title: document.title || "",
                    text: visibleText,
                    links,
                    controls,
                };
            }""",
            [max_links, max_controls],
        )

        payload["text"] = (payload.get("text") or "")[:max_text_chars]
        return payload

    async def format_page_snapshot(
        self,
        max_text_chars: int = 12000,
        max_links: int = 30,
        max_controls: int = 30,
    ) -> str:
        snapshot = await self.extract_page_snapshot(
            max_text_chars=max_text_chars,
            max_links=max_links,
            max_controls=max_controls,
        )

        lines = [
            f"URL: {snapshot.get('url', '')}",
            f"Title: {snapshot.get('title', '')}",
            "",
            "Visible text:",
            snapshot.get("text", "") or "[no visible text]",
            "",
            "Visible links:",
        ]

        links = snapshot.get("links") or []
        if links:
            for item in links:
                lines.append(f"- {item.get('text', '')} -> {item.get('href', '')}")
        else:
            lines.append("- [no visible links]")

        lines.append("")
        lines.append("Visible controls:")
        controls = snapshot.get("controls") or []
        if controls:
            for item in controls:
                lines.append(
                    f"- <{item.get('tag', '')}> type={item.get('type', '')} label={item.get('label', '')}"
                )
        else:
            lines.append("- [no visible controls]")

        return textwrap.dedent("\n".join(lines)).strip()

    async def element_html_at(self, x: float, y: float) -> str:
        html = await self.page.evaluate(
            """([x, y]) => {
                const el = document.elementFromPoint(x, y);
                return el ? el.outerHTML : "";
            }""",
            [x, y],
        )
        return html or ""

    async def _element_state_at(self, x: float, y: float) -> dict | None:
        try:
            return await self.page.evaluate(
                """([x, y]) => {
                    const el = document.elementFromPoint(x, y);
                    if (!el) return null;
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    // Walk ancestors to detect sticky/fixed overlay (nav, header,
                    // cookie banner, modal) that intercepted the click. If the
                    // nearest positioned ancestor is sticky/fixed and spans most
                    // of the viewport width, we warn the agent that the click
                    // probably hit an OVERLAY, not the underlying target.
                    let sticky = null;
                    let n = el;
                    while (n && n !== document.body) {
                        const cs = getComputedStyle(n);
                        if (cs.position === 'fixed' || cs.position === 'sticky') {
                            const r = n.getBoundingClientRect();
                            sticky = {
                                tag: (n.tagName || '').toLowerCase(),
                                position: cs.position,
                                width: Math.round(r.width),
                                height: Math.round(r.height),
                                role: n.getAttribute('role') || '',
                            };
                            break;
                        }
                        n = n.parentElement;
                    }
                    // Short, non-PII text label so the agent can recognise what
                    // it actually hit ("Accept all" vs "Loan amount").
                    const text = (el.innerText || el.textContent || '')
                                  .trim().slice(0, 80);
                    const aria = el.getAttribute('aria-label') || '';
                    return {
                        tag: (el.tagName || '').toLowerCase(),
                        role: el.getAttribute('role') || '',
                        text: text,
                        aria_label: aria.slice(0, 80),
                        disabled: !!el.disabled,
                        aria_disabled: el.getAttribute('aria-disabled'),
                        pointer_events: style.pointerEvents,
                        visibility: style.visibility,
                        display: style.display,
                        width: Math.round(rect.width),
                        height: Math.round(rect.height),
                        sticky_overlay: sticky,
                    };
                }""",
                [x, y],
            )
        except Exception:
            return None

    async def _window_scroll_y(self) -> int | None:
        try:
            return int(await self.page.evaluate("() => Math.round(window.scrollY)"))
        except Exception:
            return None

    async def _scroll_position_at(self, x: float, y: float) -> dict | None:
        try:
            return await self.page.evaluate(
                """([x, y]) => {
                    const el = document.elementFromPoint(x, y);
                    if (!el) return null;
                    let node = el;
                    while (node) {
                        const style = getComputedStyle(node);
                        const overflowY = style.overflowY;
                        if ((overflowY === 'auto' || overflowY === 'scroll')
                                && node.scrollHeight > node.clientHeight) {
                            return {
                                scope: 'element',
                                y: Math.round(node.scrollTop),
                                max: Math.round(node.scrollHeight - node.clientHeight),
                            };
                        }
                        node = node.parentElement;
                    }
                    return {
                        scope: 'window',
                        y: Math.round(window.scrollY),
                        max: Math.round(document.documentElement.scrollHeight - window.innerHeight),
                    };
                }""",
                [x, y],
            )
        except Exception:
            return None

    async def click_at(self, x: float, y: float) -> dict:
        target_state = await self._element_state_at(x, y)
        await self.page.mouse.click(x, y)
        await self._safe_wait_load()
        return {"target_state": target_state}

    async def double_click_at(self, x: float, y: float) -> dict:
        target_state = await self._element_state_at(x, y)
        await self.page.mouse.dblclick(x, y)
        await self._safe_wait_load()
        return {"target_state": target_state}

    async def hover_at(self, x: float, y: float) -> dict:
        target_state = await self._element_state_at(x, y)
        await self.page.mouse.move(x, y)
        await asyncio.sleep(0.5)
        return {"target_state": target_state}

    async def drag_from_to(
        self,
        from_x: float,
        from_y: float,
        to_x: float,
        to_y: float,
        hold_ms: int = 0,
        steps: int = 20,
    ) -> dict:
        """Press at (from_x, from_y), optionally hold, move smoothly to (to_x, to_y),
        then release. `steps` controls how many intermediate move events are emitted
        (>=20 is usually enough for JS drag handlers to fire on scrollbars / sliders /
        sortable lists / map panes)."""
        target_state = await self._element_state_at(from_x, from_y)
        await self.page.mouse.move(from_x, from_y)
        await self.page.mouse.down()
        if hold_ms and hold_ms > 0:
            await asyncio.sleep(min(hold_ms, 5000) / 1000)
        await self.page.mouse.move(to_x, to_y, steps=max(2, int(steps)))
        await self.page.mouse.up()
        await self._safe_wait_load()
        return {
            "target_state": target_state,
            "from": [from_x, from_y],
            "to": [to_x, to_y],
            "hold_ms": int(hold_ms or 0),
        }

    async def type_text_at(
        self,
        x: float,
        y: float,
        text: str,
        press_enter: bool = False,
        clear_before_typing: bool = True,
    ) -> dict:
        target_state = await self._element_state_at(x, y)
        await self.page.mouse.click(x, y)
        await self._safe_wait_load()

        if clear_before_typing:
            await self.key_combination(["Control", "A"])
            await asyncio.sleep(0.1)
            await self.key_combination(["Delete"])
            await self._safe_wait_load()

        await self.page.mouse.click(x, y)
        await self.page.keyboard.type(text)
        await self._safe_wait_load()

        if press_enter:
            await self.key_combination(["Enter"])
            await self._safe_wait_load()
        return {"target_state": target_state}

    async def scroll_document(self, direction: Literal["up", "down"]) -> dict:
        before = await self._window_scroll_y()
        if direction == "down":
            await self.key_combination(["PageDown"])
        else:
            await self.key_combination(["PageUp"])
        await self._safe_wait_load()
        after = await self._window_scroll_y()
        delta = (after - before) if (before is not None and after is not None) else None
        return {
            "scroll_target": "WINDOW",
            "before_y": before,
            "after_y": after,
            "delta_y": delta,
        }

    async def scroll_at(self, x: float, y: float, direction: Literal["up", "down"]) -> dict:
        await self.page.mouse.move(x, y)
        await asyncio.sleep(0.1)
        delta = 500 if direction == "down" else -500
        before = await self._scroll_position_at(x, y)
        await self.page.mouse.wheel(0, delta)
        await self._safe_wait_load()
        after = await self._scroll_position_at(x, y)
        return {
            "scroll_target": "ELEMENT",
            "before": before,
            "after": after,
        }

    async def select_option_at(self, x: float, y: float, option: str) -> dict:
        target_state = await self._element_state_at(x, y)
        await self.page.mouse.click(x, y)
        handle = await self.page.evaluate_handle(
            "([x, y]) => document.elementFromPoint(x, y)",
            [x, y],
        )
        tag = await handle.evaluate("el => el && el.tagName")

        if tag == "SELECT":
            try:
                option_labels = await handle.evaluate(
                    "sel => Array.from(sel.options).map(o => (o.label || '').trim())"
                )
            except Exception:
                option_labels = []
            try:
                await handle.evaluate(
                    """(sel, label) => {
                        const want = (label || '').trim();
                        const lower = want.toLowerCase();
                        const opts = [...sel.options];
                        let opt = opts.find(o => (o.label || '').trim() === want);
                        if (!opt) opt = opts.find(o => (o.label || '').trim().toLowerCase() === lower);
                        if (!opt) opt = opts.find(o => (o.label || '').trim().toLowerCase().includes(lower));
                        if (!opt) throw new Error('Option not found: ' + label);
                        sel.value = opt.value;
                        sel.dispatchEvent(new Event('input', {bubbles: true}));
                        sel.dispatchEvent(new Event('change', {bubbles: true}));
                    }""",
                    option,
                )
            except Exception as exc:
                preview = option_labels[:20]
                raise RuntimeError(
                    f"Option {option!r} not found in <select>. Available options: {preview}"
                ) from exc
            await self._safe_wait_load()
            return {"target_state": target_state, "select_kind": "native"}

        raise SelectFallbackRequired(
            f"Element at this label is <{(tag or 'unknown').lower()}>, not a native <select>. "
            "Click already opened the menu (if it has one); use a follow-up click on the option "
            "from the new screenshot, or use press_key with ArrowDown/Enter for keyboard navigation."
        )

    async def press_keys(self, keys: list[str]) -> dict:
        if not keys:
            raise ValueError("press_keys requires a non-empty list of key names.")
        await self.key_combination(keys)
        await self._safe_wait_load()
        return {"keys": list(keys)}

    async def go_back(self) -> dict:
        response = await self.page.go_back()
        await self._safe_wait_load()
        return {"navigated": response is not None}

    async def navigate(self, url: str):
        if not url.startswith(
            ("http://", "https://", "file://", "about:", "chrome:", "data:")
        ):
            url = "https://" + url
        await self.page.goto(url, timeout=60000, wait_until="domcontentloaded")
        await self._safe_wait_load()

    def tabs_snapshot(self) -> list[dict]:
        if self._context is None:
            return []
        out = []
        for index, page in enumerate(self._context.pages):
            try:
                url = page.url
            except Exception:
                url = ""
            out.append(
                {
                    "index": index,
                    "url": url,
                    "active": page is self._page,
                }
            )
        return out

    async def list_tabs(self) -> list[dict]:
        if self._context is None:
            return []
        pages = list(self._context.pages)
        out = []
        for index, page in enumerate(pages):
            try:
                url = page.url
            except Exception:
                url = ""
            title = ""
            try:
                title = await page.title()
            except Exception:
                pass
            out.append(
                {
                    "index": index,
                    "url": url,
                    "title": title,
                    "active": page is self._page,
                }
            )
        return out

    async def open_tab(self, url: str) -> dict:
        """Open `url` in a new tab and switch focus to it.

        Bypasses the `_handle_new_page` auto-handler by directly creating the
        page via the context API. `_close_new_tabs` is intentionally ignored:
        this is an explicit agent action, not a popup."""
        if self._context is None:
            raise RuntimeError("Browser context is not initialized.")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("new_tab requires a non-empty url.")
        url = url.strip()
        if not url.startswith(("http://", "https://", "about:", "chrome:", "file:", "data:")):
            url = "https://" + url

        # Tell the "page" event handler to skip its auto-handling for this open.
        self._suppress_next_page_events += 1
        try:
            new_page = await self._context.new_page()
        except Exception:
            # new_page never returned; cancel the suppression slot.
            self._suppress_next_page_events = max(0, self._suppress_next_page_events - 1)
            raise
        self._attach_page_listeners(new_page)
        try:
            await new_page.goto(url, timeout=60000, wait_until="domcontentloaded")
        except Exception as exc:
            try:
                await new_page.close()
            except Exception:
                pass
            raise RuntimeError(f"new_tab navigation failed: {type(exc).__name__}: {exc}") from exc
        self._page = new_page
        try:
            await new_page.bring_to_front()
        except Exception:
            pass
        opened_index = self._context.pages.index(new_page) if new_page in self._context.pages else -1
        return {
            "opened_index": opened_index,
            "url": getattr(new_page, "url", "") or url,
            "total_tabs": len(self._context.pages),
        }

    async def switch_tab(self, index: int) -> dict:
        if self._context is None:
            raise RuntimeError("Browser context is not initialized.")
        pages = list(self._context.pages)
        if index < 0 or index >= len(pages):
            raise ValueError(
                f"tab index {index} out of range; only {len(pages)} tab(s) open."
            )
        target = pages[index]
        prev_index = pages.index(self._page) if self._page in pages else -1
        self._page = target
        try:
            await target.bring_to_front()
        except Exception:
            pass
        return {
            "switched_from": prev_index,
            "switched_to": index,
            "url": getattr(target, "url", "") or "",
        }

    async def close_tab(self, index: int) -> dict:
        if self._context is None:
            raise RuntimeError("Browser context is not initialized.")
        pages = list(self._context.pages)
        if len(pages) <= 1:
            raise RuntimeError(
                "Cannot close the only remaining tab. Use goto to navigate instead."
            )
        if index < 0 or index >= len(pages):
            raise ValueError(
                f"tab index {index} out of range; only {len(pages)} tab(s) open."
            )
        target = pages[index]
        was_active = target is self._page
        try:
            await target.close()
        except Exception:
            pass
        if was_active:
            remaining = list(self._context.pages)
            if remaining:
                self._page = remaining[-1]
                try:
                    await self._page.bring_to_front()
                except Exception:
                    pass
        return {
            "closed": index,
            "was_active": was_active,
            "remaining": len(self._context.pages),
        }

    async def key_combination(self, keys: list[str]):
        if not keys:
            return
        mapped = [PLAYWRIGHT_KEY_MAP.get(k.lower(), k) for k in keys]
        modifiers = mapped[:-1]
        trigger = mapped[-1]
        held: list[str] = []
        try:
            for key in modifiers:
                await self.page.keyboard.down(key)
                held.append(key)
            await self.page.keyboard.press(trigger)
        finally:
            for key in reversed(held):
                try:
                    await self.page.keyboard.up(key)
                except Exception:
                    pass
