from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


DEFAULT_UPSTREAM_BASE_URL = "https://r.jina.ai"
# Default DB lives next to the package; override with JINA_POOL_DB.
DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "jina_pool.db")


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def env_optional_proxy(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() in {"direct", "off", "none", "no"}:
        return None
    return value


@dataclass(frozen=True)
class PoolSettings:
    host: str
    port: int
    db_path: Path
    upstream_base_url: str
    proxy_url: str | None
    request_timeout: float
    cooldown_seconds: int
    lease_seconds: int
    allow_anonymous: bool

    @classmethod
    def from_env(cls) -> "PoolSettings":
        request_timeout = float(os.getenv("JINA_POOL_REQUEST_TIMEOUT", "60"))
        lease_seconds_raw = os.getenv("JINA_POOL_LEASE_SECONDS")
        lease_seconds = (
            int(lease_seconds_raw)
            if lease_seconds_raw is not None
            else max(int(request_timeout) + 10, 40)
        )
        return cls(
            host=os.getenv("JINA_POOL_HOST", "127.0.0.1"),
            port=int(os.getenv("JINA_POOL_PORT", "8766")),
            db_path=Path(os.getenv("JINA_POOL_DB", DEFAULT_DB_PATH)).expanduser(),
            upstream_base_url=os.getenv(
                "JINA_UPSTREAM_BASE_URL", DEFAULT_UPSTREAM_BASE_URL
            ).rstrip("/"),
            proxy_url=env_optional_proxy("JINA_POOL_PROXY_URL"),
            request_timeout=request_timeout,
            cooldown_seconds=int(os.getenv("JINA_POOL_COOLDOWN_SECONDS", "60")),
            lease_seconds=lease_seconds,
            allow_anonymous=env_flag("JINA_POOL_ALLOW_ANONYMOUS", False),
        )
