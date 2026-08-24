# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Single-use, append-only local ledger for locked PromptBudget evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Mapping, Optional, Sequence, Tuple


class LockedEvaluationError(RuntimeError):
    """Raised when an evaluation cannot obtain or prove its reservation."""


@dataclass(frozen=True)
class Reservation:
    holdout_digest: str
    reserved_at_utc: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def holdout_digest(
    input_bytes: bytes,
    outcome_bytes: bytes,
    schema_version: int,
    group_manifest: Sequence[Tuple[str, int]],
) -> str:
    """Hash all holdout identity components using domain-separated serialization."""

    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise LockedEvaluationError("schema version must be an integer")
    manifest = sorted((str(group), int(count)) for group, count in group_manifest)
    payload = {
        "domain": "promptbudget/locked-holdout/v1",
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "outcome_sha256": hashlib.sha256(outcome_bytes).hexdigest(),
        "schema_version": schema_version,
        "group_manifest": manifest,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


class AppendOnlyLedger:
    """SQLite event ledger whose public mutation API only appends events."""

    _schema_lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)

    def _initialize(self) -> None:
        with self._schema_lock:
            connection = self._connect()
            try:
                connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    holdout_digest TEXT PRIMARY KEY,
                    reserved_at_utc TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    holdout_digest TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK (event_type IN ('reserved', 'completed', 'failed')),
                    created_at_utc TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (holdout_digest) REFERENCES reservations(holdout_digest)
                );
                CREATE TRIGGER IF NOT EXISTS reservations_no_update
                BEFORE UPDATE ON reservations BEGIN
                    SELECT RAISE(ABORT, 'append-only ledger forbids reservation updates');
                END;
                CREATE TRIGGER IF NOT EXISTS reservations_no_delete
                BEFORE DELETE ON reservations BEGIN
                    SELECT RAISE(ABORT, 'append-only ledger forbids reservation deletes');
                END;
                CREATE TRIGGER IF NOT EXISTS events_no_update
                BEFORE UPDATE ON evaluation_events BEGIN
                    SELECT RAISE(ABORT, 'append-only ledger forbids event updates');
                END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete
                BEFORE DELETE ON evaluation_events BEGIN
                    SELECT RAISE(ABORT, 'append-only ledger forbids event deletes');
                END;
                """
                )
            finally:
                connection.close()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _validate_digest(digest: str) -> None:
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise LockedEvaluationError("holdout digest must be lowercase SHA-256")

    def reserve(self, digest: str, metadata: Mapping[str, object]) -> Reservation:
        self._validate_digest(digest)
        timestamp = self._timestamp()
        encoded = _canonical_json(dict(metadata)).decode("utf-8")
        try:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO reservations (holdout_digest, reserved_at_utc, metadata_json) VALUES (?, ?, ?)",
                    (digest, timestamp, encoded),
                )
                connection.execute(
                    "INSERT INTO evaluation_events (holdout_digest, event_type, created_at_utc, metadata_json) VALUES (?, 'reserved', ?, ?)",
                    (digest, timestamp, encoded),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()
        except sqlite3.IntegrityError as error:
            raise LockedEvaluationError("holdout digest is already reserved and cannot be re-evaluated") from error
        return Reservation(digest, timestamp)

    def append(self, digest: str, event_type: str, metadata: Mapping[str, object]) -> None:
        self._validate_digest(digest)
        if event_type not in ("completed", "failed"):
            raise LockedEvaluationError("only completed or failed events may be appended")
        encoded = _canonical_json(dict(metadata)).decode("utf-8")
        connection = self._connect()
        try:
            exists = connection.execute(
                "SELECT 1 FROM reservations WHERE holdout_digest = ?", (digest,)
            ).fetchone()
            if exists is None:
                raise LockedEvaluationError("cannot append an event without a reservation")
            connection.execute(
                "INSERT INTO evaluation_events (holdout_digest, event_type, created_at_utc, metadata_json) VALUES (?, ?, ?, ?)",
                (digest, event_type, self._timestamp(), encoded),
            )
        finally:
            connection.close()

    def reservation(self, digest: str) -> Optional[Reservation]:
        self._validate_digest(digest)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT holdout_digest, reserved_at_utc FROM reservations WHERE holdout_digest = ?", (digest,)
            ).fetchone()
        finally:
            connection.close()
        return None if row is None else Reservation(row[0], row[1])


def require_reservation(ledger: Optional[AppendOnlyLedger], digest: str) -> Reservation:
    """Prevent a scoring core from being invoked outside the locked reservation path."""

    if ledger is None:
        raise LockedEvaluationError("locked evaluation requires a ledger reservation")
    reservation = ledger.reservation(digest)
    if reservation is None:
        raise LockedEvaluationError("locked evaluation requires a ledger reservation")
    return reservation
