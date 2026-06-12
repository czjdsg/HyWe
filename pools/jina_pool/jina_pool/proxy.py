from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .config import PoolSettings
from .store import KeyStore


RETRYABLE_STATUSES = {401, 403, 408, 409, 425, 429, 500, 502, 503, 504}
PASSTHROUGH_STATUSES = {400, 404, 405, 413, 415, 422}
QUOTA_TERMS = (
    "quota",
    "credit",
    "credits",
    "usage limit",
    "exceeded",
    "out of credits",
    "insufficient balance",
    "payment required",
    "monthly limit",
    "rate limit",
)
INVALID_KEY_TERMS = (
    "unauthorized",
    "invalid api key",
    "invalid key",
    "invalid token",
)
ACCESS_BLOCK_TERMS = (
    "error code: 1010",
    "access denied",
    "the owner of this website",
    "cloudflare",
)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
FORWARDED_HEADER_BLACKLIST = {
    "authorization",
    "connection",
    "content-length",
    "host",
}


@dataclass(frozen=True)
class FailureDecision:
    action: str
    retry: bool
    reason: str
    cooldown_seconds: int | None = None


@dataclass(frozen=True)
class ProxyResponse:
    status: int
    body: bytes
    content_type: str
    content_encoding: str | None
    attempts: int


def safe_decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def build_upstream_headers(
    incoming_headers: dict[str, str], authorization: str | None
) -> dict[str, str]:
    headers = {
        name: value
        for name, value in incoming_headers.items()
        if name.lower() not in FORWARDED_HEADER_BLACKLIST
    }
    user_agent = headers.get("User-Agent") or headers.get("user-agent")
    if not user_agent or "python-urllib" in user_agent.lower():
        headers["User-Agent"] = DEFAULT_USER_AGENT
    if authorization:
        headers["Authorization"] = authorization
    return headers


def classify_upstream_failure(
    status_code: int,
    response_text: str,
    cooldown_seconds: int,
) -> FailureDecision:
    lowered = response_text.lower()

    if status_code in PASSTHROUGH_STATUSES:
        return FailureDecision(
            action="passthrough",
            retry=False,
            reason=f"passthrough http {status_code}",
        )

    if status_code == 403 and any(term in lowered for term in ACCESS_BLOCK_TERMS):
        return FailureDecision(
            action="passthrough",
            retry=False,
            reason="passthrough cloudflare/access block http 403",
        )

    if status_code == 402 or any(term in lowered for term in QUOTA_TERMS):
        retry = status_code not in PASSTHROUGH_STATUSES
        return FailureDecision(
            action="exhausted",
            retry=retry,
            reason=f"quota exhausted after http {status_code}",
        )

    if status_code == 401 or any(term in lowered for term in INVALID_KEY_TERMS):
        return FailureDecision(
            action="disabled",
            retry=True,
            reason=f"disable key after http {status_code}",
        )

    if status_code == 429:
        return FailureDecision(
            action="cooldown",
            retry=True,
            reason="cooldown after http 429",
            cooldown_seconds=cooldown_seconds,
        )

    # Jina frequently returns 403 for upstream/WAF blocks that are unrelated to
    # the credential itself. Only explicit invalid-key signals should disable a key.
    if status_code == 403:
        return FailureDecision(
            action="passthrough",
            retry=False,
            reason="passthrough generic http 403",
        )

    if status_code in RETRYABLE_STATUSES:
        return FailureDecision(
            action="cooldown",
            retry=True,
            reason=f"cooldown after http {status_code}",
            cooldown_seconds=cooldown_seconds,
        )

    return FailureDecision(
        action="cooldown",
        retry=True,
        reason=f"fallback cooldown after http {status_code}",
        cooldown_seconds=cooldown_seconds,
    )


class JinaPoolApplication:
    def __init__(self, settings: PoolSettings):
        self.settings = settings
        self.store = KeyStore(settings.db_path)
        self.store.initialize()
        if settings.proxy_url:
            self.opener = build_opener(
                ProxyHandler(
                    {
                        "http": settings.proxy_url,
                        "https": settings.proxy_url,
                    }
                )
            )
        else:
            self.opener = build_opener(ProxyHandler({}))

    def proxy_request(
        self,
        method: str,
        raw_path: str,
        incoming_headers: dict[str, str],
        body: bytes,
    ) -> ProxyResponse:
        parsed = urlsplit(raw_path)
        target = parsed.path.lstrip("/")
        if parsed.query:
            target = f"{target}?{parsed.query}" if target else parsed.query
        if not target:
            payload = {"error": "Missing target URL in request path."}
            return ProxyResponse(
                status=400,
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                content_type="application/json; charset=utf-8",
                content_encoding=None,
                attempts=0,
            )

        upstream_url = f"{self.settings.upstream_base_url}/{target}"
        attempted_ids: set[int] = set()
        last_error = "no keys available"
        anonymous_used = False

        while True:
            authorization = None
            key_id = None

            leased = self.store.lease_next_key(
                excluded_ids=attempted_ids,
                lease_seconds=self.settings.lease_seconds,
            )
            if leased is not None:
                key_id = leased["id"]
                attempted_ids.add(key_id)
                authorization = f"Bearer {leased['api_key']}"
            elif self.settings.allow_anonymous and not anonymous_used:
                anonymous_used = True
            else:
                payload = {
                    "error": "No usable Jina API key is currently available.",
                    "detail": last_error,
                    "summary": self.store.summary(),
                    "allow_anonymous": self.settings.allow_anonymous,
                }
                return ProxyResponse(
                    status=503,
                    body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    content_type="application/json; charset=utf-8",
                    content_encoding=None,
                    attempts=len(attempted_ids) + (1 if anonymous_used else 0),
                )

            headers = build_upstream_headers(incoming_headers, authorization)
            request = Request(
                upstream_url,
                data=body if body else None,
                headers=headers,
                method=method,
            )

            try:
                with self.opener.open(
                    request,
                    timeout=self.settings.request_timeout,
                ) as response:
                    response_body = response.read()
                    content_type = response.headers.get_content_type()
                    charset = response.headers.get_content_charset()
                    if charset:
                        content_type = f"{content_type}; charset={charset}"
                    content_encoding = response.headers.get("Content-Encoding")
                    if key_id is not None:
                        self.store.record_success(key_id, response.status)
                    return ProxyResponse(
                        status=response.status,
                        body=response_body,
                        content_type=content_type,
                        content_encoding=content_encoding,
                        attempts=len(attempted_ids) + (1 if anonymous_used else 0),
                    )
            except HTTPError as exc:
                response_body = exc.read()
                content_type = exc.headers.get_content_type()
                charset = exc.headers.get_content_charset()
                if charset:
                    content_type = f"{content_type}; charset={charset}"
                content_encoding = exc.headers.get("Content-Encoding")
                if key_id is None:
                    return ProxyResponse(
                        status=exc.code,
                        body=response_body,
                        content_type=content_type,
                        content_encoding=content_encoding,
                        attempts=len(attempted_ids) + (1 if anonymous_used else 0),
                    )

                response_text = safe_decode(response_body)
                decision = classify_upstream_failure(
                    exc.code,
                    response_text,
                    self.settings.cooldown_seconds,
                )
                if decision.action == "passthrough":
                    self.store.record_neutral_result(key_id, exc.code)
                    return ProxyResponse(
                        status=exc.code,
                        body=response_body,
                        content_type=content_type,
                        content_encoding=content_encoding,
                        attempts=len(attempted_ids) + (1 if anonymous_used else 0),
                    )
                self.store.record_failure(
                    key_id,
                    status=decision.action,
                    error=decision.reason,
                    http_status=exc.code,
                    cooldown_seconds=decision.cooldown_seconds,
                )
                last_error = f"key {leased.get('label') or key_id}: {decision.reason}"
                if not decision.retry:
                    return ProxyResponse(
                        status=exc.code,
                        body=response_body,
                        content_type=content_type,
                        attempts=len(attempted_ids) + (1 if anonymous_used else 0),
                    )
            except URLError as exc:
                reason = getattr(exc, "reason", exc)
                error_text = f"network error: {reason}"
                if key_id is None:
                    payload = {"error": error_text}
                    return ProxyResponse(
                        status=502,
                        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        content_type="application/json; charset=utf-8",
                        content_encoding=None,
                        attempts=len(attempted_ids) + (1 if anonymous_used else 0),
                    )
                self.store.record_failure(
                    key_id,
                    status="cooldown",
                    error=error_text,
                    cooldown_seconds=self.settings.cooldown_seconds,
                )
                last_error = f"key {leased.get('label') or key_id}: {error_text}"
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                error_text = f"unexpected proxy error: {exc}"
                if key_id is None:
                    payload = {"error": error_text}
                    return ProxyResponse(
                        status=502,
                        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        content_type="application/json; charset=utf-8",
                        content_encoding=None,
                        attempts=len(attempted_ids) + (1 if anonymous_used else 0),
                    )
                self.store.record_failure(
                    key_id,
                    status="cooldown",
                    error=error_text,
                    cooldown_seconds=self.settings.cooldown_seconds,
                )
                last_error = f"key {leased.get('label') or key_id}: {error_text}"

    def create_handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,HEAD,OPTIONS")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                if self.path in {"/healthz"}:
                    payload = {
                        "status": "ok",
                        "upstream_base_url": app.settings.upstream_base_url,
                        "proxy_mode": app.settings.proxy_url or "direct",
                        "summary": app.store.summary(),
                        "allow_anonymous": app.settings.allow_anonymous,
                    }
                    return self._send_json(200, payload)

                if self.path == "/admin/keys":
                    payload = {
                        "keys": app.store.list_keys(include_secret=False),
                        "summary": app.store.summary(),
                        "allow_anonymous": app.settings.allow_anonymous,
                    }
                    return self._send_json(200, payload)

                self._proxy_current_request()

            def do_HEAD(self) -> None:
                self._proxy_current_request(send_body=False)

            def do_POST(self) -> None:
                self._proxy_current_request()

            def _proxy_current_request(self, send_body: bool = True) -> None:
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(content_length) if content_length else b""
                result = app.proxy_request(
                    self.command,
                    self.path,
                    dict(self.headers.items()),
                    body,
                )
                self.send_response(result.status)
                self.send_header("Content-Type", result.content_type)
                if result.content_encoding:
                    self.send_header("Content-Encoding", result.content_encoding)
                self.send_header("Content-Length", str(len(result.body)))
                self.send_header("X-Jina-Pool-Attempts", str(result.attempts))
                self.end_headers()
                if send_body:
                    self.wfile.write(result.body)

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def serve_forever(self) -> None:
        server = ThreadingHTTPServer(
            (self.settings.host, self.settings.port),
            self.create_handler(),
        )
        print(
            f"Jina key pool listening on http://{self.settings.host}:{self.settings.port}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
