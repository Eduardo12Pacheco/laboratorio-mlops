"""SQLite run ledger written by the pipeline (`S14`)."""

from __future__ import annotations

from datetime import datetime

from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.application.dto import RunStatus, StepStatus


class SqliteRunLedger:
    """Runs and step attempts persisted in `runs`/`run_steps`."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def start(self, run_id: str, name: str, tick: int | None, started_at: datetime) -> None:
        """Record the start of a run; restarted runs keep their original rows."""
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, name, tick, started_at, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, name, tick, started_at.isoformat(), RunStatus.RUNNING.value),
            )

    def finish_step(
        self,
        run_id: str,
        index: int,
        name: str,
        status: StepStatus,
        started_at: datetime,
        finished_at: datetime,
        error: str | None = None,
    ) -> None:
        """Record one completed attempt; retries use increasing indexes."""
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO run_steps (
                    run_id, idx, name, status, started_at, finished_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    index,
                    name,
                    status.value,
                    started_at.isoformat(),
                    finished_at.isoformat(),
                    error,
                ),
            )

    def finish(
        self,
        run_id: str,
        status: RunStatus,
        finished_at: datetime,
        error: str | None = None,
    ) -> None:
        """Record the final status of a run."""
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE run_id = ?",
                (status.value, finished_at.isoformat(), error, run_id),
            )


__all__ = ["SqliteRunLedger"]
