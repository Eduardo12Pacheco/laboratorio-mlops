"""SQLite run ledger written by the pipeline (`S14`)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.application.dto import RunStatus, StepStatus


@dataclass(frozen=True, slots=True)
class LedgerRun:
    """One run with its measured duration and failed-step count (dashboard read model)."""

    run_id: str
    name: str
    tick: int | None
    status: str
    started_at: datetime
    finished_at: datetime | None
    error: str | None
    duration_seconds: float
    failed_steps: int


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

    def list_runs(self, limit: int | None = None) -> list[LedgerRun]:
        """Return the most recent runs first, with durations and failed-step counts."""
        rows = self._database.connection.execute(
            """
            SELECT
                r.run_id,
                r.name,
                r.tick,
                r.status,
                r.started_at,
                r.finished_at,
                r.error,
                (
                    SELECT COUNT(*) FROM run_steps AS s
                    WHERE s.run_id = r.run_id AND s.status = ?
                ) AS failed_steps
            FROM runs AS r
            ORDER BY r.started_at DESC, r.run_id DESC
            """,
            (StepStatus.FAILED.value,),
        ).fetchall()
        if limit is not None:
            rows = rows[:limit] if limit > 0 else []
        return [self._decode_run(row) for row in rows]

    @staticmethod
    def _decode_run(row: sqlite3.Row) -> LedgerRun:
        values = dict(row)
        started_at = datetime.fromisoformat(str(values["started_at"]))
        finished_raw = values["finished_at"]
        finished_at = None if finished_raw is None else datetime.fromisoformat(str(finished_raw))
        duration = 0.0
        if finished_at is not None:
            duration = max(0.0, (finished_at - started_at).total_seconds())
        tick_raw = values["tick"]
        return LedgerRun(
            run_id=str(values["run_id"]),
            name=str(values["name"]),
            tick=None if tick_raw is None else int(str(tick_raw)),
            status=str(values["status"]),
            started_at=started_at,
            finished_at=finished_at,
            error=None if values["error"] is None else str(values["error"]),
            duration_seconds=duration,
            failed_steps=int(str(values["failed_steps"])),
        )


__all__ = ["LedgerRun", "SqliteRunLedger"]
