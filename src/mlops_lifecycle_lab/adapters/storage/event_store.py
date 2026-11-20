"""SQLite-backed append-only lifecycle event store (S14)."""

from __future__ import annotations

import json
import sqlite3

from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent


class SqliteEventStore:
    """Chronological audit log with the filters the dashboard and services use."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def append(self, event: LifecycleEvent) -> None:
        """Append one event; the payload is serialised as JSON."""
        payload = event.to_dict()
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO events (ts, type, tick, model_version, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload["ts"],
                    payload["type"],
                    payload["tick"],
                    payload["model_version"],
                    json.dumps(payload["payload"], sort_keys=True),
                ),
            )

    def list(
        self,
        *,
        event_type: EventType | None = None,
        model_version: int | None = None,
        limit: int | None = None,
    ) -> list[LifecycleEvent]:
        """Return matching events in chronological order, keeping the last `limit`."""
        query = "SELECT ts, type, tick, model_version, payload_json FROM events"
        conditions: list[str] = []
        parameters: list[object] = []
        if event_type is not None:
            conditions.append("type = ?")
            parameters.append(event_type.value)
        if model_version is not None:
            conditions.append("model_version = ?")
            parameters.append(model_version)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY event_id"
        rows = self._database.connection.execute(query, parameters).fetchall()
        if limit is not None:
            rows = rows[-limit:] if limit > 0 else []
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> LifecycleEvent:
        values = dict(row)
        return LifecycleEvent.from_dict(
            {
                "ts": values["ts"],
                "type": values["type"],
                "tick": values["tick"],
                "model_version": values["model_version"],
                "payload": json.loads(str(values["payload_json"])),
            }
        )


__all__ = ["SqliteEventStore"]
