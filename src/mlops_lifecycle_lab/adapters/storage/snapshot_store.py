"""SQLite snapshot store, one monitoring snapshot per tick (S10)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.application.dto import MonitoringSnapshot


class SqliteSnapshotStore:
    """Persist and read monitoring snapshots ordered by replay tick."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def save(self, snapshot: MonitoringSnapshot) -> None:
        """Insert or replace the snapshot of its tick."""
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO snapshots (
                    tick, ts, model_version, n_rows, rmse, mae, wape, bias,
                    numeric_drift_share, prediction_psi, psi_json, verdict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.tick,
                    snapshot.ts.isoformat(),
                    snapshot.model_version,
                    snapshot.n_rows,
                    snapshot.rmse,
                    snapshot.mae,
                    snapshot.wape,
                    snapshot.bias,
                    snapshot.numeric_drift_share,
                    snapshot.prediction_psi,
                    json.dumps(dict(snapshot.psi), sort_keys=True),
                    snapshot.verdict,
                ),
            )

    def list(self, limit: int | None = None) -> list[MonitoringSnapshot]:
        """Return snapshots in tick order, optionally the last `limit`."""
        rows = self._database.connection.execute("SELECT * FROM snapshots ORDER BY tick").fetchall()
        if limit is not None:
            rows = rows[-limit:] if limit > 0 else []
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> MonitoringSnapshot:
        values = dict(row)
        return MonitoringSnapshot(
            tick=int(str(values["tick"])),
            ts=datetime.fromisoformat(str(values["ts"])),
            model_version=int(str(values["model_version"])),
            n_rows=int(str(values["n_rows"])),
            rmse=None if values["rmse"] is None else float(str(values["rmse"])),
            mae=None if values["mae"] is None else float(str(values["mae"])),
            wape=None if values["wape"] is None else float(str(values["wape"])),
            bias=None if values["bias"] is None else float(str(values["bias"])),
            numeric_drift_share=float(str(values["numeric_drift_share"])),
            prediction_psi=(
                None if values["prediction_psi"] is None else float(str(values["prediction_psi"]))
            ),
            psi=json.loads(str(values["psi_json"])),
            verdict=str(values["verdict"]),
        )


__all__ = ["SqliteSnapshotStore"]
