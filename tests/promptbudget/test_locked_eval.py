# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local single-use locked-holdout ledger."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path
import tempfile
import unittest

from promptbudget.locked_eval import (
    AppendOnlyLedger,
    LockedEvaluationError,
    dev_confirmation_digest,
    holdout_digest,
    require_reservation,
)


class LockedEvaluationTest(unittest.TestCase):
    def test_dev_confirmation_digest_binds_artifact_manifest_and_observed_split(self) -> None:
        first = dev_confirmation_digest(b"artifact-a", b"manifest-a", b"dev-input", b"dev-outcome")
        self.assertNotEqual(first, dev_confirmation_digest(b"artifact-b", b"manifest-a", b"dev-input", b"dev-outcome"))
        self.assertNotEqual(first, dev_confirmation_digest(b"artifact-a", b"manifest-b", b"dev-input", b"dev-outcome"))
        with tempfile.TemporaryDirectory() as directory:
            ledger = AppendOnlyLedger(Path(directory) / "ledger.sqlite")
            ledger.reserve(first, {"observed_split": True})
            with self.assertRaisesRegex(LockedEvaluationError, "already reserved"):
                ledger.reserve(first, {"observed_split": True})

    def test_digest_is_domain_separated_and_order_independent_for_manifest(self) -> None:
        first = holdout_digest(
            b"input", b"outcome", 1, (("group-b", 2), ("group-a", 1))
        )
        second = holdout_digest(
            b"input", b"outcome", 1, (("group-a", 1), ("group-b", 2))
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, holdout_digest(b"input", b"other", 1, (("group-a", 1),)))

    def test_only_one_concurrent_reservation_and_any_retry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.sqlite"
            digest = "a" * 64

            def reserve() -> bool:
                try:
                    AppendOnlyLedger(ledger_path).reserve(digest, {"operator": "test"})
                    return True
                except LockedEvaluationError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(1, sum(pool.map(lambda _ignored: reserve(), range(2))))
            ledger = AppendOnlyLedger(ledger_path)
            ledger.append(digest, "failed", {"reason": "simulated crash"})
            with self.assertRaisesRegex(LockedEvaluationError, "already reserved"):
                ledger.reserve(digest, {"operator": "retry-with-new-artifact"})

    def test_rows_are_append_only_and_core_requires_a_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.sqlite"
            digest = "b" * 64
            ledger = AppendOnlyLedger(ledger_path)
            with self.assertRaises(LockedEvaluationError):
                require_reservation(None, digest)
            ledger.reserve(digest, {"operator": "test"})
            reservation = require_reservation(ledger, digest)
            self.assertEqual(digest, reservation.holdout_digest)
            connection = sqlite3.connect(ledger_path)
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute("DELETE FROM evaluation_events")
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute("UPDATE reservations SET holdout_digest = ?", ("c" * 64,))
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
