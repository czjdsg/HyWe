from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from serper_pool.proxy import classify_upstream_failure
from serper_pool.store import KeyStore


class KeyStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "pool.db"
        self.store = KeyStore(self.db_path)
        self.store.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_leasing_skips_cooldown_key(self) -> None:
        key_one = self.store.add_key("key-one-12345678", "one")
        key_two = self.store.add_key("key-two-12345678", "two")

        leased_one = self.store.lease_next_key()
        self.assertEqual(leased_one["id"], key_one["id"])
        self.store.record_failure(
            leased_one["id"],
            status="cooldown",
            error="transient failure",
            cooldown_seconds=30,
        )

        leased_two = self.store.lease_next_key()
        self.assertEqual(leased_two["id"], key_two["id"])

    def test_record_success_clears_failure_state(self) -> None:
        key_row = self.store.add_key("key-three-12345678", "three")
        leased = self.store.lease_next_key()
        self.assertEqual(leased["id"], key_row["id"])
        self.store.record_failure(
            leased["id"],
            status="cooldown",
            error="temporary error",
            cooldown_seconds=10,
        )
        self.store.enable_key(key_row["id"])
        leased_again = self.store.lease_next_key()
        self.assertEqual(leased_again["id"], key_row["id"])
        self.store.record_success(leased_again["id"], 200)

        rows = self.store.list_keys()
        self.assertEqual(rows[0]["status"], "active")
        self.assertEqual(rows[0]["success_count"], 1)
        self.assertIsNone(rows[0]["last_error"])


class FailureClassifierTestCase(unittest.TestCase):
    def test_quota_error_marks_key_exhausted(self) -> None:
        decision = classify_upstream_failure(429, "Monthly quota exceeded", 60)
        self.assertEqual(decision.action, "exhausted")
        self.assertTrue(decision.retry)

    def test_bad_request_is_passthrough(self) -> None:
        decision = classify_upstream_failure(400, "invalid payload", 60)
        self.assertEqual(decision.action, "passthrough")
        self.assertFalse(decision.retry)


if __name__ == "__main__":
    unittest.main()
