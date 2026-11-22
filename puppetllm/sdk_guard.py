"""Durable, fail-closed admission for potentially billable SDK executions.

Only HMAC fingerprints, random lease tokens, timestamps and fixed reason codes
are persisted. Successful executions release their marker; every other outcome
keeps it. There is no expiration, eviction, or automatic retry after a restart.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import Any, Iterator, Mapping


FAILURE_REASONS = frozenset({
    "timeout", "error", "cancelled", "abnormal_exit",
    "upstream_error", "invalid_output", "delivery_error",
})


class GuardError(RuntimeError):
    """Admission cannot be established safely; do not start an SDK execution."""


@dataclass(frozen=True)
class Lease:
    key: str
    token: str = field(repr=False)


@dataclass(frozen=True)
class Blocked:
    key: str
    reason: str
    state: str


class SDKGuard:
    """Share one persistent database between every worker for the same relay.

    ``generation_settings`` passed to :meth:`fingerprint` should contain the
    adapter's system prompt and actual SDK generation options (including the
    input format and upstream target), never the complete relay configuration.
    The database contains a randomly generated local HMAC secret, not API keys.
    """

    def __init__(self, path: str | os.PathLike[str], *, max_entries: int = 10000) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise GuardError("SDK guard requires a positive retention limit")
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._leases: dict[str, Lease] = {}
        try:
            if str(path) == ":memory:":
                raise GuardError("SDK guard requires a persistent database")
            expanded = Path(path).expanduser()
            # Canonicalize directory aliases without following a database-file
            # symlink: O_NOFOLLOW below must still protect the final component.
            db_path = expanded.parent.resolve() / expanded.name
            db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Create the database restrictively before sqlite creates journals.
            fd = os.open(db_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)
            self._connection = sqlite3.connect(
                db_path, timeout=5, isolation_level=None, check_same_thread=False)
            self._connection.execute("PRAGMA synchronous=FULL")
            if self._connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise GuardError("SDK guard database integrity check failed")
            with self._transaction() as db:
                db.execute("""CREATE TABLE IF NOT EXISTS sdk_guard_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL, secret BLOB NOT NULL)""")
                db.execute("""CREATE TABLE IF NOT EXISTS sdk_guard_entries (
                    key TEXT PRIMARY KEY CHECK (length(key) = 64 AND key NOT GLOB '*[^0-9a-f]*'),
                    token TEXT NOT NULL CHECK (length(token) = 64 AND token NOT GLOB '*[^0-9a-f]*'),
                    state TEXT NOT NULL CHECK (state IN ('active', 'blocked')),
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    reason TEXT NOT NULL CHECK (
                        (state = 'active' AND reason = 'in_progress') OR
                        (state = 'blocked' AND reason IN (
                            'timeout', 'error', 'cancelled', 'abnormal_exit',
                            'upstream_error', 'invalid_output', 'delivery_error'))))""")
                metadata = db.execute("SELECT version, secret FROM sdk_guard_meta WHERE id = 1").fetchone()
                if metadata is None:
                    # A missing secret must not reset the identity of old rows.
                    if db.execute("SELECT COUNT(*) FROM sdk_guard_entries").fetchone()[0]:
                        raise GuardError("SDK guard database metadata is missing")
                    metadata = (1, secrets.token_bytes(32))
                    db.execute("INSERT INTO sdk_guard_meta VALUES (1, ?, ?)", metadata)
                if metadata[0] != 1 or not isinstance(metadata[1], bytes) or len(metadata[1]) != 32:
                    raise GuardError("SDK guard database metadata is invalid")
                self._secret = metadata[1]
        except (OSError, sqlite3.Error, TypeError, ValueError):
            self._close_connection()
            raise GuardError("SDK guard database is unavailable; SDK execution is disabled") from None
        except BaseException:
            self._close_connection()
            raise

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            db = self._connection
            if db is None:
                raise GuardError("SDK guard is closed")
            try:
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.commit()
            except BaseException as exc:
                try:
                    db.rollback()
                except sqlite3.Error:
                    pass
                if isinstance(exc, sqlite3.Error):
                    raise GuardError("SDK guard database operation failed; SDK execution is disabled") from None
                raise

    @staticmethod
    def _check_key(key: str) -> None:
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise GuardError("SDK guard requires a valid fingerprint")

    def fingerprint(self, body: Mapping[str, Any], *,
                    generation_settings: Mapping[str, Any] | None = None) -> str:
        """HMAC the resolved model, translated input/images and generation settings.

        Request IDs, deadlines, the derived ``prompt`` and other outer request
        metadata do not identify an SDK execution. Do not strip transcript data:
        a tool argument called ``request_id`` is still part of the model input.
        """
        with self._lock:
            if self._connection is None:
                raise GuardError("SDK guard is closed")
            try:
                value = {"version": 1, "model": body["model"],
                         "transcript": body["transcript"], "images": body.get("images", []),
                         "generation_settings": dict(generation_settings or {})}
                encoded = json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False, separators=(",", ":")).encode("ascii")
            except (KeyError, TypeError, ValueError, AttributeError, RecursionError):
                raise GuardError("SDK guard cannot fingerprint this SDK input") from None
            return hmac.new(self._secret, encoded, hashlib.sha256).hexdigest()

    def begin(self, key: str) -> Lease | Blocked:
        """Commit an active marker before allowing SDK startup; never evict rows."""
        self._check_key(key)
        with self._lock:
            with self._transaction() as db:
                row = db.execute("SELECT state, reason FROM sdk_guard_entries WHERE key = ?", (key,)).fetchone()
                if row is not None:
                    return Blocked(key, row[1], row[0])
                if db.execute("SELECT COUNT(*) FROM sdk_guard_entries").fetchone()[0] >= self.max_entries:
                    return Blocked(key, "capacity", "capacity")
                lease = Lease(key, secrets.token_hex(32))
                now = time.time()
                db.execute("INSERT INTO sdk_guard_entries VALUES (?, ?, 'active', ?, ?, 'in_progress')",
                           (key, lease.token, now, now))
            self._leases[lease.token] = lease
            return lease

    def _finish(self, lease: Lease, reason: str | None) -> bool:
        if not isinstance(lease, Lease):
            raise GuardError("SDK guard requires a valid lease")
        self._check_key(lease.key)
        with self._lock:
            with self._transaction() as db:
                row = db.execute("SELECT token, state FROM sdk_guard_entries WHERE key = ?", (lease.key,)).fetchone()
                if row is None or not hmac.compare_digest(row[0], lease.token):
                    raise GuardError("SDK guard lease is no longer active")
                if row[1] != "active":
                    if reason is not None:
                        return False  # preserve the first recorded failure
                    raise GuardError("SDK guard lease is no longer active")
                if reason is None:
                    db.execute("DELETE FROM sdk_guard_entries WHERE key = ?", (lease.key,))
                else:
                    db.execute("UPDATE sdk_guard_entries SET state = 'blocked', updated_at = ?, reason = ? WHERE key = ?",
                               (time.time(), reason, lease.key))
            self._leases.pop(lease.token, None)
            return True

    def succeed(self, lease: Lease) -> None:
        """Release only a still-active successful execution; never cache output."""
        self._finish(lease, None)

    def fail(self, lease: Lease, reason: str = "error") -> bool:
        """Keep a permanent block using a fixed code, never raw exception text."""
        if not isinstance(reason, str) or reason not in FAILURE_REASONS:
            raise GuardError("SDK guard requires a fixed failure reason code")
        return self._finish(lease, reason)

    def status(self) -> list[dict[str, Any]]:
        """Return fingerprints, states, times and fixed reasons, without payloads."""
        with self._transaction() as db:
            rows = db.execute("SELECT key, state, created_at, updated_at, reason FROM sdk_guard_entries ORDER BY created_at, key").fetchall()
        return [dict(zip(("key", "state", "created_at", "updated_at", "reason"), row)) for row in rows]

    def clear(self, key: str | None = None) -> int:
        """Explicitly remove blocked rows only (``None`` means all blocked rows)."""
        if key is not None:
            self._check_key(key)
        with self._transaction() as db:
            if key is None:
                cursor = db.execute("DELETE FROM sdk_guard_entries WHERE state = 'blocked'")
            else:
                cursor = db.execute("DELETE FROM sdk_guard_entries WHERE state = 'blocked' AND key = ?", (key,))
            return cursor.rowcount

    def recover_active(self) -> int:
        """Mark abandoned executions blocked without allowing an automatic retry.

        The caller MUST hold an exclusive worker-lifetime lock proving that no
        SDK worker is running. Recovery and explicit ``clear`` are separate
        maintenance operations; ordinary startup must never call this method.
        """
        with self._lock:
            with self._transaction() as db:
                cursor = db.execute("UPDATE sdk_guard_entries SET state = 'blocked', updated_at = ?, reason = 'abnormal_exit' WHERE state = 'active'",
                                    (time.time(),))
                count = cursor.rowcount
            self._leases.clear()
            return count

    def _close_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def close(self) -> None:
        """Preserve this instance's unfinished executions as abnormal failures."""
        with self._lock:
            if self._connection is None:
                return
            try:
                if self._leases:
                    with self._transaction() as db:
                        now = time.time()
                        for lease in self._leases.values():
                            db.execute("UPDATE sdk_guard_entries SET state = 'blocked', updated_at = ?, reason = 'abnormal_exit' WHERE key = ? AND token = ? AND state = 'active'",
                                       (now, lease.key, lease.token))
            finally:
                self._leases.clear()
                self._close_connection()

    def __enter__(self) -> SDKGuard:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
