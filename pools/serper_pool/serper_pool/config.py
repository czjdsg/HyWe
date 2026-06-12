from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


DEFAULT_UPSTREAM_BASE_URL = "https://google.serper.dev"
# Default DB lives next to the package; override with SERPER_POOL_DB.
DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "serper_pool.db")


@dataclass(frozen=True)
class PoolSettings:
    host: str
    port: int
    db_path: Path
    upstream_base_url: str
    request_timeout: float
    cooldown_seconds: int
    lease_seconds: int
    max_success_per_key: int

    @classmethod
    def from_env(cls) -> "PoolSettings":
        request_timeout = float(os.getenv("SERPER_POOL_REQUEST_TIMEOUT", "20"))
        lease_seconds_raw = os.getenv("SERPER_POOL_LEASE_SECONDS")
        lease_seconds = (
            int(lease_seconds_raw)
            if lease_seconds_raw is not None
            else max(int(request_timeout) + 5, 25)
        )
        return cls(
            host=os.getenv("SERPER_POOL_HOST", "127.0.0.1"),
            port=int(os.getenv("SERPER_POOL_PORT", "8765")),
            db_path=Path(os.getenv("SERPER_POOL_DB", DEFAULT_DB_PATH)).expanduser(),
            upstream_base_url=os.getenv(
                "SERPER_UPSTREAM_BASE_URL", DEFAULT_UPSTREAM_BASE_URL
            ).rstrip("/"),
            request_timeout=request_timeout,
            cooldown_seconds=int(os.getenv("SERPER_POOL_COOLDOWN_SECONDS", "60")),
            lease_seconds=lease_seconds,
            max_success_per_key=int(os.getenv("SERPER_POOL_MAX_SUCCESS_PER_KEY", "2500")),
        )
