"""SQLite key/value state store for the replay cursor and probation state (D11)."""

from __future__ import annotations

import json

from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.application.dto import JSONValue


class SqliteKeyValueStore:
    """JSON-valued state keyed by name; `None` clears a key."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def get(self, key: str) -> JSONValue | None:
        """Return the stored JSON value or `None`."""
        row = self._database.connection.execute(
            "SELECT value_json FROM kv WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value: JSONValue = json.loads(str(row[0]))
        return value

    def set(self, key: str, value: JSONValue) -> None:
        """Store or clear (`None`) a JSON value."""
        with self._database.transaction() as connection:
            if value is None:
                connection.execute("DELETE FROM kv WHERE key = ?", (key,))
                return
            connection.execute(
                "INSERT OR REPLACE INTO kv(key, value_json) VALUES (?, ?)",
                (key, json.dumps(value, sort_keys=True)),
            )


__all__ = ["SqliteKeyValueStore"]
