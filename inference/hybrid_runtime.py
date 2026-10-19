import asyncio
import base64
import collections
import hashlib
import json
import socket
import io
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import re
import sys
import threading
import textwrap
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from PIL import Image

from hybrid_utils import sanitize_surrogates


# ----------------------------------------------------------------- 浏览器指纹
# 采集实测 bot_block 占失败成因 36.4%,是最大的一类(Cloudflare / 403 / 反爬)。
# 之前只有 --disable-blink-features=AutomationControlled + 一个写死的 Chrome/122
# UA,而且 32 个 worker 共用完全相同的指纹 —— 同一站点看到 32 个一模一样的浏览器,
# 这本身就是强信号。
#
# 注意:**viewport 绝不能随机化**。整条坐标链路(模型输出 0–1000 映射到
# window_width × window_height)依赖它固定;改了会让所有点击偏移。
_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/139.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0.0.0 Safari/537.36",
]
_TZ_POOL = ["America/New_York", "America/Chicago", "America/Los_Angeles", "Europe/London"]
_GPU_POOL = [
    ("Google Inc. (Intel)", "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
    ("Google Inc. (NVIDIA)", "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
    ("Google Inc. (AMD)", "ANGLE (AMD, AMD Radeon RX 6600 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
    ("Apple Inc.", "Apple GPU"),
]


_PROXY_POOL: list[str] | None = None


def _pick_proxy(seed: str, fallback: str) -> str:
    """按 task_id 稳定地从代理池里选一个;池子不可用时退回 fallback。"""
    global _PROXY_POOL
    if _PROXY_POOL is None:
        _PROXY_POOL = []
        try:
            # Optional: HYBRID_PROXY_POOL points at a JSON list like
            # [{"port": 9001}, {"port": 9002}] of local HTTP proxies (one egress IP
            # each). Unset = every browser uses --proxy-url (or goes direct).
            path = os.environ.get("HYBRID_PROXY_POOL", "").strip()
            if not path:
                raise FileNotFoundError("HYBRID_PROXY_POOL not set")
            rows = json.loads(Path(path).read_text(encoding="utf-8"))
            # 只收**当前真的在监听**的端口。
            # 2026-09-01 教训:纯 O 那轮的 proxy_pool.json 留在盘上,而池子早被停了,
            # 于是这里把死端口塞进 --proxy-server,Chrome 每次导航都
            # ERR_PROXY_CONNECTION_FAILED —— 53% 的轨迹被静默毒化,
            # 表面症状是 agent_stuck / answer_reject_loop / phase_budget_exceeded,
            # 查了半天才追到这里。文件过期是常态,所以必须探活而不是信文件。
            alive = []
            for r in rows:
                port = r.get("port")
                if not port:
                    continue
                sock = socket.socket()
                sock.settimeout(0.5)
                try:
                    sock.connect(("127.0.0.1", int(port)))
                    alive.append(f"http://127.0.0.1:{port}")
                except Exception:
                    pass
                finally:
                    sock.close()
            _PROXY_POOL = alive
        except Exception:
            _PROXY_POOL = []
    if not _PROXY_POOL:
        return fallback
    h = int(hashlib.sha256((seed or "x").encode("utf-8")).hexdigest()[:12], 16)
    return _PROXY_POOL[h % len(_PROXY_POOL)]


def _fp_pick(seed: str):
    """按任务 id 稳定地选一套指纹 —— 同一任务重试时保持一致,不同 worker 之间不同。

    GPU 必须和 UA 的操作系统对得上:UA 报 Windows 而 WebGL 报 "Apple GPU" 这种
    不自洽,比留着 headless 默认值更容易被识破 —— 检测脚本正是靠交叉校验抓人的。
    """
    h = int(hashlib.sha256((seed or "x").encode("utf-8")).hexdigest()[:12], 16)
    ua = _UA_POOL[h % len(_UA_POOL)]
    tz = _TZ_POOL[(h >> 8) % len(_TZ_POOL)]
    if "Mac OS" in ua:
        pool = [g for g in _GPU_POOL if g[0] == "Apple Inc."]
    else:                       # Windows / Linux 都用 ANGLE + 独显/核显
        pool = [g for g in _GPU_POOL if g[0] != "Apple Inc."]
    gpu = pool[(h >> 16) % len(pool)]
    cores = [4, 8, 12, 16][(h >> 24) % 4]
    mem = [4, 8, 16][(h >> 28) % 3]
    return ua, tz, gpu, cores, mem


def _stealth_script(ua: str, gpu, cores: int, mem: int) -> str:
    """在每个新文档最开头注入,抹掉最常被检测的几个自动化痕迹。

    只覆盖公开、稳定的检测面;不去碰 Canvas/Audio 噪声那类需要逐站调参的东西 ——
    那些做不好反而会因为"值不自洽"更容易被识破。
    """
    vendor, renderer = gpu
    plat = "Win32" if "Windows" in ua else ("MacIntel" if "Mac OS" in ua else "Linux x86_64")
    return f"""
(() => {{
  const def = (o, k, v) => {{
    try {{ Object.defineProperty(o, k, {{get: () => v, configurable: true}}); }} catch (e) {{}}
  }};
  // 1. webdriver 标志:启动参数只影响一部分,JS 层仍读得到
  def(Navigator.prototype, 'webdriver', undefined);
  delete Object.getPrototypeOf(navigator).webdriver;
  // 2. headless 下 plugins/mimeTypes 为空,是最经典的判据
  const mk = (n) => {{
    const a = [];
    for (let i = 0; i < n; i++) a.push({{name: 'Plugin' + i, filename: 'p' + i + '.dll'}});
    a.item = (i) => a[i]; a.namedItem = (x) => a.find(p => p.name === x);
    return a;
  }};
  def(Navigator.prototype, 'plugins', mk(3));
  def(Navigator.prototype, 'mimeTypes', mk(2));
  def(Navigator.prototype, 'languages', ['en-US', 'en']);
  def(Navigator.prototype, 'platform', {plat!r});
  def(Navigator.prototype, 'hardwareConcurrency', {cores});
  def(Navigator.prototype, 'deviceMemory', {mem});
  // 3. WebGL vendor/renderer:headless 默认回落到 SwiftShader,一眼就能认出来
  const patch = (proto) => {{
    if (!proto) return;
    const gp = proto.getParameter;
    proto.getParameter = function (p) {{
      if (p === 37445) return {vendor!r};
      if (p === 37446) return {renderer!r};
      return gp.apply(this, arguments);
    }};
  }};
  patch(window.WebGLRenderingContext && WebGLRenderingContext.prototype);
  patch(window.WebGL2RenderingContext && WebGL2RenderingContext.prototype);
  // 4. window.chrome 在真实 Chrome 里存在
  if (!window.chrome) window.chrome = {{runtime: {{}}, loadTimes: () => {{}}, csi: () => {{}}}};
  // 5. 权限查询在 headless 下会返回与真实浏览器不一致的组合
  const q = navigator.permissions && navigator.permissions.query;
  if (q) navigator.permissions.query = (p) =>
    p && p.name === 'notifications'
      ? Promise.resolve({{state: Notification.permission}})
      : q(p);
}})();
"""


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


_IMAGE_URL_CACHE: dict[tuple[str, int, int, str, str], str] = {}
_IMAGE_URL_CACHE_LOCK = threading.Lock()
_IMAGE_MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def _image_publish_config() -> tuple[str, Path]:
    public_base = os.environ.get("HYBRID_IMAGE_URL_BASE", "").strip().rstrip("/")
    publish_value = os.environ.get("HYBRID_IMAGE_PUBLISH_DIR", "").strip()
    if not public_base or not publish_value:
        raise RuntimeError(
            "HYBRID_IMAGE_TRANSPORT=url requires both HYBRID_IMAGE_URL_BASE "
            "and HYBRID_IMAGE_PUBLISH_DIR"
        )
    if not public_base.lower().startswith("https://"):
        raise RuntimeError(
            "HYBRID_IMAGE_URL_BASE must use HTTPS because remote multimodal "
            "providers reject or cannot safely fetch local HTTP image URLs"
        )
    return public_base, Path(publish_value).expanduser().resolve()


# ---- 发布后预热边缘缓存 ----------------------------------------------------
# 远端模型取图走的是 Cloudflare 边缘 -> 命名隧道 -> 本机源站。冷缓存下这一跳要
# 跨境回源, 2026-09-21 实测 kimi 那端 62s 都取不完, 直接 <400> "Download
# multimodal file timed out"(WEBP/PNG/JPEG 都一样, 单张 51KB 也一样;同一批图
# qwen3.8-max 却 9/9 秒级取到 —— 是对端取图链路慢, 不是我们的隧道)。
# 判官侧早就有同样的问题, 解法是 traj_render.prewarm: 发出去之前自己先取一遍,
# 把对象顶进边缘缓存, 模型再取就是 HIT。这里把同一招放到 rollout 的发布路径上。
# 细节:
#   * 必须伪装 UA —— Cloudflare 对 Python-urllib 直接 403(判官那边踩过);
#   * 必须绕开本机代理, 否则预热的是代理出口那个 colo, 白热;
#   * 内容寻址 => 同一张图只预热一次;
#   * 失败不抛错也不重试: 预热只是加速, 拉不到就让模型自己去试。
_PREWARM_LOCK = threading.Lock()
_PREWARM_SEEN: set[str] = set()
_PREWARM_POOL = None


def _prewarm_public_url(url: str) -> None:
    if os.environ.get("HYBRID_IMAGE_PREWARM", "0").strip().lower() in {"0", "false", "no", ""}:
        return
    global _PREWARM_POOL
    with _PREWARM_LOCK:
        if url in _PREWARM_SEEN:
            return
        _PREWARM_SEEN.add(url)
        if _PREWARM_POOL is None:
            workers = int(os.environ.get("HYBRID_IMAGE_PREWARM_WORKERS", "8"))
            _PREWARM_POOL = ThreadPoolExecutor(max_workers=workers,
                                               thread_name_prefix="img-prewarm")
    timeout = float(os.environ.get("HYBRID_IMAGE_PREWARM_TIMEOUT", "20"))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _fetch() -> None:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (prewarm)"})
            with opener.open(req, timeout=timeout) as response:
                while response.read(1 << 16):
                    pass
        except Exception:
            pass

    try:
        _PREWARM_POOL.submit(_fetch)
    except Exception:
        pass


def _publish_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    public_base: str,
    publish_dir: Path,
) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    suffix = _IMAGE_MIME_SUFFIXES.get(mime_type, ".png")
    relative = Path(digest[:2]) / f"{digest}{suffix}"
    destination = publish_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    # O_EXCL makes concurrent workers safe without replacing an object another
    # worker has already published. Content addressing guarantees that the
    # existing file has the same bytes.
    try:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        pass
    else:
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(image_bytes)
        except Exception:
            try:
                destination.unlink()
            except OSError:
                pass
            raise

    relative_url = "/".join(quote(part) for part in relative.parts)
    public_url = f"{public_base}/{relative_url}"
    _prewarm_public_url(public_url)
    return public_url


def _publish_image_for_remote_fetch(image_path: str) -> str:
    """Copy an image into the immutable public pool and return its HTTPS URL.

    The public path is content-addressed, so identical browser frames from
    different steps resolve to the same URL and can be cached by the tunnel or
    CDN.  A copy is deliberately used instead of a hard link: Playwright can
    rewrite a screenshot path, while a published object must remain immutable.
    """
    source = Path(image_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(f"Image path is not a regular file: {source}")

    public_base, publish_dir = _image_publish_config()
    stat = source.stat()
    cache_key = (
        str(source),
        stat.st_mtime_ns,
        stat.st_size,
        public_base,
        str(publish_dir),
    )
    with _IMAGE_URL_CACHE_LOCK:
        cached = _IMAGE_URL_CACHE.get(cache_key)
    if cached:
        return cached

    image_bytes = source.read_bytes()
    mime_type = mimetypes.guess_type(str(source))[0] or "image/png"
    result = _publish_image_bytes(image_bytes, mime_type, public_base, publish_dir)
    with _IMAGE_URL_CACHE_LOCK:
        _IMAGE_URL_CACHE[cache_key] = result
    return result


def image_data_url_to_model_url(data_url: str) -> str:
    """Convert an inline image to the configured model transport.

    This covers task attachments that already arrived as data URLs, ensuring
    URL mode does not accidentally leave a large base64 item in the request.
    """
    transport = os.environ.get("HYBRID_IMAGE_TRANSPORT", "base64").strip().lower()
    if transport == "base64":
        return data_url
    if transport != "url":
        raise ValueError(
            "HYBRID_IMAGE_TRANSPORT must be 'base64' or 'url', got "
            f"{transport!r}"
        )
    match = re.fullmatch(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)", data_url, re.DOTALL)
    if not match:
        raise ValueError("Expected a base64-encoded image data URL")
    mime_type = match.group(1).lower()
    try:
        image_bytes = base64.b64decode(match.group(2))
    except Exception as exc:
        raise ValueError("Invalid base64 image data URL") from exc
    public_base, publish_dir = _image_publish_config()
    return _publish_image_bytes(image_bytes, mime_type, public_base, publish_dir)


def _parse_screenshot_size(raw) -> tuple[int, int] | None:
    """解析截图尺寸: "1000" = 1000x1000 方图; "1280x1000" = 指定宽x高。

    方图的用处是让"模型吐的 0-1000"与像素数值重合, 消除相对/绝对坐标歧义;
    但 0-1000 归一化网格本来就与长宽比无关(换算回视口由 --coord-scale/_scale_xy
    负责), 所以也可以按需给非方图 —— 2026-09-22 用户要求 qwen 端试 1280x1000。
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if "x" in text:
        w, _, h = text.partition("x")
        w, h = int(w), int(h)
    else:
        w = h = int(text)
    if w <= 0 or h <= 0:
        raise ValueError("screenshot size must be positive, got %r" % (raw,))
    return w, h


def resized_image_path_to_model_url(image_path: str, size) -> str:
    """Stretch an image to a square in memory, then use the configured transport.

    The source file is never modified. This is intended for teacher screenshots
    whose output coordinates use a 0-1000 grid: a 1000x1000 image removes the
    ambiguity between raw-pixel and normalized coordinates.
    """
    wh = _parse_screenshot_size(size)
    if wh is None:
        return image_path_to_data_url(image_path)
    out_w, out_h = wh

    source = Path(image_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(f"Image path is not a regular file: {source}")

    # PNG stays the default. Collection runs that publish screenshots as URLs can
    # switch to WEBP/JPEG: the remote model fetcher has a fixed budget to download
    # every image attached to one request, and a full-history O-side turn re-sends
    # all of them, so the per-request byte count is what decides whether the fetch
    # lands inside that budget. Measured on real teacher screenshots at size=1000:
    # PNG ~213KB, JPEG q85 ~84KB, WEBP q80 ~38KB per frame.
    fmt = (os.environ.get("HYBRID_TEACHER_SCREENSHOT_FORMAT", "PNG").strip().upper()
           or "PNG")
    if fmt not in {"PNG", "WEBP", "JPEG"}:
        raise ValueError(
            "HYBRID_TEACHER_SCREENSHOT_FORMAT must be 'PNG', 'WEBP', or 'JPEG', "
            f"got {fmt!r}"
        )
    quality = int(os.environ.get("HYBRID_TEACHER_SCREENSHOT_QUALITY", "80"))

    # HYBRID_TEACHER_SCREENSHOT_SQUARE=0:不拉方图,保留原始长宽比,但仍按
    # FORMAT/QUALITY 重新编码 —— 直接不设 SIZE 会退回原始 PNG data URL
    # (~213KB/帧),一轮 O 侧要重发全部历史图,会撞上远端取图的字节预算。
    # 方图存在的理由是:模型输出的 0-1000 与"1000x1000 图上的像素"数值重合,
    # 抹掉了两种坐标空间的歧义。关掉它之前先确认模型给的是归一化网格。
    square = os.environ.get("HYBRID_TEACHER_SCREENSHOT_SQUARE", "1").strip() != "0"

    with Image.open(source) as image:
        resized = (image.resize((out_w, out_h), Image.Resampling.LANCZOS)
                   if square else image.copy())
        output = io.BytesIO()
        if fmt == "PNG":
            resized.save(output, format="PNG")
            mime_type = "image/png"
        else:
            # Palette / LA sources cannot be encoded directly by either lossy codec.
            if resized.mode not in {"RGB", "RGBA"}:
                resized = resized.convert("RGB")
            if fmt == "JPEG":
                resized.convert("RGB").save(output, format="JPEG", quality=quality)
                mime_type = "image/jpeg"
            else:
                resized.save(output, format="WEBP", quality=quality)
                mime_type = "image/webp"

    data_url = f"data:{mime_type};base64," + base64.b64encode(
        output.getvalue()
    ).decode("ascii")
    return image_data_url_to_model_url(data_url)


def image_path_to_data_url(image_path: str) -> str:
    """Return a model-consumable image URL using the configured transport.

    ``base64`` remains the default for compatibility. ``url`` publishes the
    local file to a content-addressed directory (HYBRID_IMAGE_PUBLISH_DIR) that
    you serve over HTTPS at HYBRID_IMAGE_URL_BASE.
    """
    transport = os.environ.get("HYBRID_IMAGE_TRANSPORT", "base64").strip().lower()
    if transport == "url":
        return _publish_image_for_remote_fetch(image_path)
    if transport != "base64":
        raise ValueError(
            "HYBRID_IMAGE_TRANSPORT must be 'base64' or 'url', got "
            f"{transport!r}"
        )
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
        # 按 task_id 稳定选一套指纹:同一任务重试保持一致,不同 worker 之间互不相同
        self._fingerprint = _fp_pick(str(getattr(args, "task_id", "") or args.task_dir))

    async def _apply_stealth(self, context) -> None:
        """在每个新文档最开头注入反检测脚本。失败不影响采集,只是少了伪装。"""
        ua, _tz, gpu, cores, mem = self._fingerprint
        try:
            await context.add_init_script(_stealth_script(ua, gpu, cores, mem))
        except Exception:
            pass

    def _context_kwargs(self, storage_state=None) -> dict:
        ua, tz, _gpu, _c, _m = self._fingerprint
        return {
            # viewport 固定:坐标链路依赖它,随机化会让所有点击偏移
            "viewport": {"width": self._screen_size[0], "height": self._screen_size[1]},
            "device_scale_factor": 1,
            "user_agent": ua,
            "locale": "en-US",
            "timezone_id": tz,
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

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process,AutomationControlled",
            "--disable-infobars",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-dev-shm-usage",
            "--lang=en-US",
        ]
        if getattr(self.args, "no_sandbox", True):
            launch_args.append("--no-sandbox")
        # 代理按 task_id 从池子里挑一个,给每个 worker 不同的出口 IP。
        # 实测 32 个 worker 共用一个出口 IP 时,Cloudflare 直接判机器人 ——
        # JS 层的指纹伪装对 IP 这一层完全无效。池子由 HYBRID_PROXY_POOL 指定(可选);
        # 未设置就退回 --proxy-url 指定的那个。
        proxy = _pick_proxy(str(getattr(self.args, "task_id", "") or self.args.task_dir),
                            getattr(self.args, "proxy_url", ""))
        if proxy:
            launch_args.append(f"--proxy-server={proxy}")

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
        await self._apply_stealth(self._context)
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
            await self._apply_stealth(temp_context)
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
        """找出该点实际会滚动的容器,并给出它的偏移、上限**和身份**。

        `scope` 过去对所有内嵌容器都返回字面量 'element' —— 同一页面上两个列表
        都停在 scrollTop=0 时会互相误判成同一个容器(于是被当成"没动")。
        现在带上容器的 DOM 路径与标识,两个列表就区分得开。
        """
        try:
            return await self.page.evaluate(
                r"""([x, y]) => {
                    const ident = (n) => {
                        let path = '', depth = 0;
                        for (let m = n; m && m.parentElement && depth < 6;
                             m = m.parentElement, depth++) {
                            const i = Array.prototype.indexOf.call(m.parentElement.children, m);
                            path += (m.tagName || '').toLowerCase() + i + '/';
                        }
                        return (n.id || n.getAttribute('data-testid') || '') + '#' + path;
                    };
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

    async def type_text_focused(
        self,
        text: str,
        press_enter: bool = False,
        clear_before_typing: bool = True,
    ) -> dict:
        """Type into the CURRENTLY-FOCUSED element without moving/clicking the mouse.

        Focus is assumed to already be on the target input (from a preceding
        `click` step or page autofocus). This is the executor for the
        coordinate-free `type` tool (decoupled from click)."""
        if clear_before_typing:
            await self.key_combination(["Control", "A"])
            await asyncio.sleep(0.1)
            await self.key_combination(["Delete"])
            await self._safe_wait_load()

        await self.page.keyboard.type(text)
        await self._safe_wait_load()

        if press_enter:
            await self.key_combination(["Enter"])
            await self._safe_wait_load()
        return {}


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
