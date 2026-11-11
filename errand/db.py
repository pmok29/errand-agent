"""agent.db: connection, schema, and a transaction helper. The mock provider keeps its own mock.db."""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'chat', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS errands (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
    kind TEXT NOT NULL, state TEXT NOT NULL,
    request_json TEXT NOT NULL DEFAULT '{}', asked_json TEXT NOT NULL DEFAULT '[]',
    selected_json TEXT NOT NULL DEFAULT '{}', booking_id TEXT, parent_errand_id TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS holds (
    hold_id TEXT PRIMARY KEY, errand_id TEXT NOT NULL, provider_id TEXT NOT NULL,
    slot TEXT NOT NULL, quoted_price INTEGER NOT NULL, search_price INTEGER,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY, errand_id TEXT NOT NULL, hash TEXT NOT NULL, code TEXT NOT NULL,
    draft_json TEXT NOT NULL, status TEXT NOT NULL, expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY, errand_id TEXT NOT NULL, action TEXT NOT NULL,
    attempt_no INTEGER NOT NULL, idem_key TEXT NOT NULL, payload_hash TEXT NOT NULL,
    approval_hash TEXT NOT NULL, parent_execution_id TEXT, state TEXT NOT NULL,
    payload_json TEXT NOT NULL, retries INTEGER NOT NULL DEFAULT 0, last_error TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE (errand_id, action, attempt_no)
);
CREATE TABLE IF NOT EXISTS seen_refs (
    errand_id TEXT NOT NULL, kind TEXT NOT NULL, ref TEXT NOT NULL, meta_json TEXT NOT NULL,
    PRIMARY KEY (errand_id, kind, ref)
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global', source TEXT NOT NULL, status TEXT NOT NULL,
    evidence_msg_id TEXT, superseded_by TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS redactions (
    user_id TEXT NOT NULL, value TEXT NOT NULL, variants_json TEXT NOT NULL,
    reason TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS watcher_jobs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, errand_id TEXT, next_run_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS booking_snapshots (
    booking_id TEXT PRIMARY KEY, status TEXT NOT NULL, seen_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, payload_json TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_errands_session ON errands (session_id, state);
CREATE INDEX IF NOT EXISTS ix_holds_errand ON holds (errand_id);
CREATE INDEX IF NOT EXISTS ix_approvals_errand ON approvals (errand_id, code);
CREATE INDEX IF NOT EXISTS ix_executions_errand ON executions (errand_id, action, state);
CREATE INDEX IF NOT EXISTS ix_executions_state ON executions (state);
CREATE INDEX IF NOT EXISTS ix_notifications_session ON notifications (session_id, delivered);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class DB:
    """Autocommit sqlite connection with WAL, a schema, and a (re-entrant) transaction helper."""

    def __init__(self, path: Path | str, schema: str = SCHEMA):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(schema)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction; commits on success, rolls back on any exception.

        Nested use joins the outer transaction, so helpers can be atomic on their own and still
        compose inside a larger unit of work.
        """
        if self.conn.in_transaction:
            yield self.conn
            return
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def one(self, sql: str, *args) -> sqlite3.Row | None:
        return self.conn.execute(sql, args).fetchone()

    def all(self, sql: str, *args) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    def close(self) -> None:
        self.conn.close()
