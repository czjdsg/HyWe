from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(timestamp: datetime | None) -> str | None:
    if timestamp is None:
        return None
    return timestamp.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def mask_key(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


class KeyStore:
    def __init__(self, db_path: str | Path, max_success_per_key: int = 2500):
        self.db_path = Path(db_path)
        self.max_success_per_key = max_success_per_key
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _exhausted_reason(self) -> str:
        return f"local quota exhausted after {self.max_success_per_key} requests"

    def _sync_exhausted_keys(
        self, connection: sqlite3.Connection, now_iso: str | None = None
    ) -> int:
        updated_at = now_iso or to_iso(utc_now())
        return connection.execute(
            """
            UPDATE api_keys
            SET status = 'exhausted',
                cooldown_until = NULL,
                leased_until = NULL,
                updated_at = ?,
                last_error = ?
            WHERE (success_count >= ? OR total_requests >= ?)
              AND status != 'exhausted'
            """,
            (
                updated_at,
                self._exhausted_reason(),
                self.max_success_per_key,
                self.max_success_per_key,
            ),
        ).rowcount

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._lock:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        label TEXT,
                        api_key TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_used_at TEXT,
                        cooldown_until TEXT,
                        leased_until TEXT,
                        success_count INTEGER NOT NULL DEFAULT 0,
                        total_requests INTEGER NOT NULL DEFAULT 0,
                        consecutive_failures INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT,
                        last_http_status INTEGER
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_api_keys_status_cooldown
                    ON api_keys(status, cooldown_until, leased_until, last_used_at)
                    """
                )
                self._sync_exhausted_keys(connection)

    def add_key(self, api_key: str, label: str | None = None) -> dict[str, Any]:
        cleaned = api_key.strip()
        if not cleaned:
            raise ValueError("API key cannot be empty.")

        now = to_iso(utc_now())
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO api_keys (
                        label, api_key, status, created_at, updated_at, cooldown_until,
                        leased_until, consecutive_failures, last_error, last_http_status
                    )
                    VALUES (?, ?, 'active', ?, ?, NULL, NULL, 0, NULL, NULL)
                    ON CONFLICT(api_key) DO UPDATE SET
                        label = COALESCE(excluded.label, api_keys.label),
                        status = 'active',
                        updated_at = excluded.updated_at,
                        cooldown_until = NULL,
                        leased_until = NULL,
                        consecutive_failures = 0,
                        last_error = NULL,
                        last_http_status = NULL
                    """,
                    (label, cleaned, now, now),
                )
                self._sync_exhausted_keys(connection, now)
                row = connection.execute(
                    """
                    SELECT id, label, api_key, status, success_count, total_requests
                    FROM api_keys
                    WHERE api_key = ?
                    """,
                    (cleaned,),
                ).fetchone()
        return self._serialize_row(row, include_secret=False)

    def import_keys(self, entries: list[tuple[str | None, str]]) -> list[dict[str, Any]]:
        added = []
        for label, api_key in entries:
            added.append(self.add_key(api_key, label))
        return added

    def list_keys(self, include_secret: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            with self._connect() as connection:
                self._sync_exhausted_keys(connection)
                rows = connection.execute(
                    """
                    SELECT
                        id, label, api_key, status, created_at, updated_at, last_used_at,
                        cooldown_until, success_count, total_requests,
                        consecutive_failures, last_error, last_http_status
                    FROM api_keys
                    ORDER BY
                        CASE status
                            WHEN 'active' THEN 0
                            WHEN 'cooldown' THEN 1
                            WHEN 'exhausted' THEN 2
                            WHEN 'disabled' THEN 3
                            ELSE 4
                        END,
                        id ASC
                    """
                ).fetchall()
        return [self._serialize_row(row, include_secret=include_secret) for row in rows]

    def summary(self) -> dict[str, int]:
        with self._lock:
            with self._connect() as connection:
                self._sync_exhausted_keys(connection)
                rows = connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM api_keys
                    GROUP BY status
                    """
                ).fetchall()
        summary = {"total": 0, "active": 0, "cooldown": 0, "exhausted": 0, "disabled": 0}
        for row in rows:
            summary[row["status"]] = row["count"]
            summary["total"] += row["count"]
        return summary

    def lease_next_key(
        self, excluded_ids: set[int] | None = None, lease_seconds: int = 25
    ) -> dict[str, Any] | None:
        excluded_ids = excluded_ids or set()
        now = utc_now()
        now_iso = to_iso(now)
        lease_until = to_iso(now + timedelta(seconds=lease_seconds))

        with self._lock:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    UPDATE api_keys
                    SET status = 'active', cooldown_until = NULL, updated_at = ?
                    WHERE status = 'cooldown' AND cooldown_until IS NOT NULL AND cooldown_until <= ?
                    """,
                    (now_iso, now_iso),
                )
                self._sync_exhausted_keys(connection, now_iso)

                params: list[Any] = [now_iso, now_iso]
                excluded_clause = ""
                if excluded_ids:
                    placeholders = ",".join("?" for _ in excluded_ids)
                    excluded_clause = f"AND id NOT IN ({placeholders})"
                    params.extend(sorted(excluded_ids))

                row = connection.execute(
                    f"""
                    SELECT *
                    FROM api_keys
                    WHERE status IN ('active', 'cooldown')
                      AND (cooldown_until IS NULL OR cooldown_until <= ?)
                      AND (leased_until IS NULL OR leased_until <= ?)
                      {excluded_clause}
                    ORDER BY
                        consecutive_failures ASC,
                        CASE WHEN last_used_at IS NULL THEN 0 ELSE 1 END ASC,
                        last_used_at ASC,
                        total_requests ASC,
                        id ASC
                    LIMIT 1
                    """,
                    params,
                ).fetchone()

                if row is None:
                    connection.commit()
                    return None

                connection.execute(
                    """
                    UPDATE api_keys
                    SET leased_until = ?, last_used_at = ?, updated_at = ?,
                        total_requests = total_requests + 1
                    WHERE id = ?
                    """,
                    (lease_until, now_iso, now_iso, row["id"]),
                )
                connection.commit()

        return dict(row)

    def record_success(self, key_id: int, http_status: int | None = None) -> None:
        now_iso = to_iso(utc_now())
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE api_keys
                    SET status = CASE
                            WHEN success_count + 1 >= ? OR total_requests >= ?
                                THEN 'exhausted'
                            ELSE 'active'
                        END,
                        cooldown_until = NULL,
                        leased_until = NULL,
                        updated_at = ?,
                        success_count = success_count + 1,
                        consecutive_failures = 0,
                        last_error = CASE
                            WHEN success_count + 1 >= ? OR total_requests >= ?
                                THEN ?
                            ELSE NULL
                        END,
                        last_http_status = ?
                    WHERE id = ?
                    """,
                    (
                        self.max_success_per_key,
                        self.max_success_per_key,
                        now_iso,
                        self.max_success_per_key,
                        self.max_success_per_key,
                        self._exhausted_reason(),
                        http_status,
                        key_id,
                    ),
                )

    def record_neutral_result(self, key_id: int, http_status: int | None = None) -> None:
        now_iso = to_iso(utc_now())
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE api_keys
                    SET leased_until = NULL,
                        updated_at = ?,
                        last_error = NULL,
                        last_http_status = ?
                    WHERE id = ?
                    """,
                    (now_iso, http_status, key_id),
                )

    def record_failure(
        self,
        key_id: int,
        *,
        status: str,
        error: str,
        http_status: int | None = None,
        cooldown_seconds: int | None = None,
    ) -> None:
        now = utc_now()
        cooldown_until = None
        if status == "cooldown" and cooldown_seconds:
            cooldown_until = to_iso(now + timedelta(seconds=cooldown_seconds))
        now_iso = to_iso(now)
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE api_keys
                    SET status = ?,
                        cooldown_until = ?,
                        leased_until = NULL,
                        updated_at = ?,
                        consecutive_failures = consecutive_failures + 1,
                        last_error = ?,
                        last_http_status = ?
                    WHERE id = ?
                    """,
                    (status, cooldown_until, now_iso, error[:500], http_status, key_id),
                )

    def enable_key(self, key_id: int) -> None:
        now_iso = to_iso(utc_now())
        with self._lock:
            with self._connect() as connection:
                updated = connection.execute(
                    """
                    UPDATE api_keys
                    SET status = 'active',
                        cooldown_until = NULL,
                        leased_until = NULL,
                        updated_at = ?,
                        consecutive_failures = 0,
                        last_error = NULL,
                        last_http_status = NULL
                    WHERE id = ?
                    """,
                    (now_iso, key_id),
                ).rowcount
                self._sync_exhausted_keys(connection, now_iso)
        if updated == 0:
            raise ValueError(f"Key id {key_id} does not exist.")

    def disable_key(self, key_id: int) -> None:
        self._set_manual_status(key_id, "disabled")

    def delete_key(self, key_id: int) -> None:
        with self._lock:
            with self._connect() as connection:
                deleted = connection.execute(
                    "DELETE FROM api_keys WHERE id = ?",
                    (key_id,),
                ).rowcount
        if deleted == 0:
            raise ValueError(f"Key id {key_id} does not exist.")

    def _set_manual_status(self, key_id: int, status: str) -> None:
        now_iso = to_iso(utc_now())
        with self._lock:
            with self._connect() as connection:
                updated = connection.execute(
                    """
                    UPDATE api_keys
                    SET status = ?,
                        cooldown_until = NULL,
                        leased_until = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (status, now_iso, key_id),
                ).rowcount
        if updated == 0:
            raise ValueError(f"Key id {key_id} does not exist.")

    def _serialize_row(self, row: sqlite3.Row | None, include_secret: bool) -> dict[str, Any]:
        if row is None:
            raise ValueError("Key does not exist.")
        data = dict(row)
        api_key = data.pop("api_key")
        data["masked_key"] = mask_key(api_key)
        if include_secret:
            data["api_key"] = api_key
        return data
