"""SQLite connection, ordered migrations and the unit-of-work boundary (D11).

The operational store is one file per workspace, opened in WAL mode so the serving API,
the scheduler and the dashboard can read while a tick writes. Timestamps are stored as
ISO-8601 text (naive, like every timestamp in the dataset) and JSON payloads as text:
SQLite is an implementation detail of the adapters, never of the domain.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_SCHEMA_V1: tuple[str, ...] = (
    """
    CREATE TABLE predictions (
        prediction_id TEXT PRIMARY KEY,
        entity_id INTEGER NOT NULL,
        event_ts TEXT NOT NULL,
        scored_at TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version INTEGER NOT NULL,
        features_json TEXT NOT NULL,
        prediction REAL NOT NULL,
        source TEXT NOT NULL CHECK (source IN ('online', 'batch')),
        request_id TEXT
    )
    """,
    """
    CREATE TABLE labels (
        entity_id INTEGER PRIMARY KEY,
        label REAL NOT NULL,
        label_ts TEXT NOT NULL,
        ingested_at TEXT NOT NULL
    )
    """,
    """
    CREATE VIEW labeled_predictions AS
    SELECT
        p.prediction_id AS prediction_id,
        p.entity_id AS entity_id,
        p.event_ts AS event_ts,
        p.scored_at AS scored_at,
        p.model_name AS model_name,
        p.model_version AS model_version,
        p.features_json AS features_json,
        p.prediction AS prediction,
        p.source AS source,
        p.request_id AS request_id,
        l.label AS label,
        l.label_ts AS label_ts,
        l.ingested_at AS ingested_at
    FROM predictions AS p
    JOIN labels AS l ON p.entity_id = l.entity_id
    """,
    """
    CREATE TABLE events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        type TEXT NOT NULL,
        tick INTEGER NOT NULL,
        model_version INTEGER,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE snapshots (
        tick INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,
        model_version INTEGER NOT NULL,
        n_rows INTEGER NOT NULL,
        rmse REAL,
        mae REAL,
        wape REAL,
        bias REAL,
        numeric_drift_share REAL NOT NULL,
        prediction_psi REAL,
        psi_json TEXT NOT NULL,
        verdict TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE kv (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        tick INTEGER,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL,
        error TEXT
    )
    """,
    """
    CREATE TABLE run_steps (
        run_id TEXT NOT NULL,
        idx INTEGER NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT NOT NULL,
        error TEXT,
        PRIMARY KEY (run_id, idx),
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX predictions_event_ts_idx ON predictions(event_ts)",
    "CREATE INDEX predictions_entity_id_idx ON predictions(entity_id)",
    "CREATE INDEX events_type_idx ON events(type)",
)


@dataclass(frozen=True, slots=True)
class Migration:
    """One ordered, append-only schema migration."""

    version: int
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (Migration(version=1, statements=_SCHEMA_V1),)
"""Ordered migrations applied by `migrate` and recorded in `schema_migrations`."""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open `path` with the documented pragmas and a row factory.

    WAL keeps readers from blocking the scheduler writer; `busy_timeout` bounds lock
    waits; foreign keys enforce the run-ledger parent. The parent directory is created so
    a fresh workspace needs no manual setup.
    """
    target = str(path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def migrate(connection: sqlite3.Connection) -> None:
    """Apply every pending migration exactly once, in version order."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    applied = {int(row[0]) for row in connection.execute("SELECT version FROM schema_migrations")}
    for migration in MIGRATIONS:
        if migration.version in applied:
            continue
        with connection:
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version) VALUES (?)", (migration.version,)
            )


class SqliteDatabase:
    """One migrated connection shared by the SQLite repositories."""

    def __init__(self, path: Path | str) -> None:
        self._path = path
        self._connection = connect(path)
        migrate(self._connection)

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection; stores issue their own SQL through it."""
        return self._connection

    @property
    def path(self) -> Path | str:
        """The opened file, exposed for diagnostics and tests."""
        return self._path

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit on success, roll back when the block raises (unit of work)."""
        with self._connection:
            yield self._connection

    def close(self) -> None:
        """Close the connection; safe to call once per workspace."""
        self._connection.close()

    def __enter__(self) -> SqliteDatabase:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["MIGRATIONS", "Migration", "SqliteDatabase", "connect", "migrate"]
