from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

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
    "monthly limit",
    "rate limit",
)
INVALID_KEY_TERMS = (
    "unauthorized",
    "invalid api key",
    "invalid key",
    "invalid x-api-key",
    "forbidden",
)


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
    attempts: int


def safe_decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


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

    if status_code == 401 or any(term in lowered for term in INVALID_KEY_TERMS):
        return FailureDecision(
            action="disabled",
            retry=True,
            reason=f"disable key after http {status_code}",
        )

    if status_code == 429:
        action = "exhausted" if any(term in lowered for term in QUOTA_TERMS) else "cooldown"
        return FailureDecision(
            action=action,
            retry=True,
            reason=f"{action} after http 429",
            cooldown_seconds=cooldown_seconds if action == "cooldown" else None,
        )

    if status_code == 403:
        if any(term in lowered for term in QUOTA_TERMS):
            return FailureDecision(
                action="exhausted",
                retry=True,
                reason="quota exhausted after http 403",
            )
        return FailureDecision(
            action="disabled",
            retry=True,
            reason="disable key after http 403",
        )

    if status_code in RETRYABLE_STATUSES:
        return FailureDecision(
            action="cooldown",
            retry=True,
            reason=f"cooldown after http {status_code}",
            cooldown_seconds=cooldown_seconds,
        )

    if any(term in lowered for term in QUOTA_TERMS):
        return FailureDecision(
            action="exhausted",
            retry=True,
            reason=f"quota exhausted after http {status_code}",
        )

    return FailureDecision(
        action="cooldown",
        retry=True,
        reason=f"fallback cooldown after http {status_code}",
        cooldown_seconds=cooldown_seconds,
    )


class SerperPoolApplication:
    def __init__(self, settings: PoolSettings):
        self.settings = settings
        self.store = KeyStore(
            settings.db_path,
            max_success_per_key=settings.max_success_per_key,
        )
        self.store.initialize()

    def proxy_request(self, raw_path: str, body: bytes, content_type: str) -> ProxyResponse:
        parsed = urlsplit(raw_path)
        path = parsed.path or "/"
        if path == "/":
            path = "/search"
        upstream_url = self.settings.upstream_base_url + path
        if parsed.query:
            upstream_url = f"{upstream_url}?{parsed.query}"

        attempted_ids: set[int] = set()
        last_error = "no keys available"

        while True:
            leased = self.store.lease_next_key(
                excluded_ids=attempted_ids,
                lease_seconds=self.settings.lease_seconds,
            )
            if leased is None:
                payload = {
                    "error": "No usable Serper API key is currently available.",
                    "detail": last_error,
                    "summary": self.store.summary(),
                }
                return ProxyResponse(
                    status=503,
                    body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    content_type="application/json; charset=utf-8",
                    attempts=len(attempted_ids),
                )

            key_id = leased["id"]
            attempted_ids.add(key_id)
            headers = {
                "Content-Type": content_type or "application/json",
                "X-API-KEY": leased["api_key"],
            }
            request = Request(upstream_url, data=body, headers=headers, method="POST")

            try:
                with urlopen(request, timeout=self.settings.request_timeout) as response:
                    response_body = response.read()
                    content_type = response.headers.get_content_type()
                    charset = response.headers.get_content_charset()
                    if charset:
                        content_type = f"{content_type}; charset={charset}"
                    self.store.record_success(key_id, response.status)
                    return ProxyResponse(
                        status=response.status,
                        body=response_body,
                        content_type=content_type,
                        attempts=len(attempted_ids),
                    )
            except HTTPError as exc:
                response_body = exc.read()
                response_text = safe_decode(response_body)
                decision = classify_upstream_failure(
                    exc.code,
                    response_text,
                    self.settings.cooldown_seconds,
                )
                if decision.action == "passthrough":
                    content_type = exc.headers.get_content_type()
                    charset = exc.headers.get_content_charset()
                    if charset:
                        content_type = f"{content_type}; charset={charset}"
                    self.store.record_neutral_result(key_id, exc.code)
                    return ProxyResponse(
                        status=exc.code,
                        body=response_body,
                        content_type=content_type,
                        attempts=len(attempted_ids),
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
                        content_type=exc.headers.get_content_type(),
                        attempts=len(attempted_ids),
                    )
            except URLError as exc:
                reason = getattr(exc, "reason", exc)
                error_text = f"network error: {reason}"
                self.store.record_failure(
                    key_id,
                    status="cooldown",
                    error=error_text,
                    cooldown_seconds=self.settings.cooldown_seconds,
                )
                last_error = f"key {leased.get('label') or key_id}: {error_text}"
            except Exception as exc:  # pragma: no cover - defensive guard for runtime
                error_text = f"unexpected proxy error: {exc}"
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
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                if self.path in {"/", "/healthz"}:
                    payload = {
                        "status": "ok",
                        "upstream_base_url": app.settings.upstream_base_url,
                        "summary": app.store.summary(),
                    }
                    return self._send_json(200, payload)

                if self.path == "/admin/keys":
                    payload = {
                        "keys": app.store.list_keys(include_secret=False),
                        "summary": app.store.summary(),
                    }
                    return self._send_json(200, payload)

                self._send_json(404, {"error": "Not found"})

            def do_POST(self) -> None:
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(content_length)
                result = app.proxy_request(
                    self.path,
                    body,
                    self.headers.get("Content-Type", "application/json"),
                )
                self.send_response(result.status)
                self.send_header("Content-Type", result.content_type)
                self.send_header("Content-Length", str(len(result.body)))
                self.send_header("X-Serper-Pool-Attempts", str(result.attempts))
                self.end_headers()
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
            f"Serper key pool listening on http://{self.settings.host}:{self.settings.port}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
