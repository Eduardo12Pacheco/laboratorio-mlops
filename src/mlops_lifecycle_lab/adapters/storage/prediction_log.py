"""SQLite prediction log with delayed labels and idempotent writes (S9)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.application.dto import LabelRecord, PredictionRecord
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS

_LABELED_COLUMNS: list[str] = [
    "prediction_id",
    "entity_id",
    "event_ts",
    "scored_at",
    "model_name",
    "model_version",
    "prediction",
    "source",
    "request_id",
    *FEATURE_COLUMNS,
    "label",
]
_TRAINING_COLUMNS: list[str] = [*FEATURE_COLUMNS, "timestamp", "entity_id", "cnt"]


def _features_payload(record: PredictionRecord) -> str:
    return json.dumps(dict(record.features), sort_keys=True)


class SqlitePredictionLog:
    """Append-only predictions joined with labels by `entity_id`."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def append(self, records: Sequence[PredictionRecord]) -> None:
        """Insert predictions, ignoring an already stored `prediction_id`."""
        if not records:
            return
        with self._database.transaction() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO predictions (
                    prediction_id, entity_id, event_ts, scored_at, model_name,
                    model_version, features_json, prediction, source, request_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.prediction_id,
                        record.entity_id,
                        record.event_ts.isoformat(),
                        record.scored_at.isoformat(),
                        record.model_name,
                        record.model_version,
                        _features_payload(record),
                        record.prediction,
                        record.source,
                        record.request_id,
                    )
                    for record in records
                ],
            )

    def attach_labels(self, labels: Sequence[LabelRecord]) -> None:
        """Attach labels to known predictions, ignoring repeated or unknown entities."""
        if not labels:
            return
        ingested_at = datetime.now().isoformat()
        with self._database.transaction() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO labels (entity_id, label, label_ts, ingested_at)
                SELECT ?, ?, ?, ?
                WHERE EXISTS (SELECT 1 FROM predictions WHERE entity_id = ?)
                """,
                [
                    (
                        label.entity_id,
                        label.label,
                        label.label_ts.isoformat(),
                        ingested_at,
                        label.entity_id,
                    )
                    for label in labels
                ],
            )

    def labeled_predictions(
        self,
        start: datetime,
        end: datetime,
        model_version: int | None = None,
    ) -> pd.DataFrame:
        """Return labeled predictions in the half-open `[start, end)` window."""
        query = """
            SELECT prediction_id, entity_id, event_ts, scored_at, model_name,
                   model_version, features_json, prediction, source, request_id, label
            FROM labeled_predictions
            WHERE event_ts >= ? AND event_ts < ?
        """
        parameters: list[object] = [start.isoformat(), end.isoformat()]
        if model_version is not None:
            query += " AND model_version = ?"
            parameters.append(model_version)
        query += " ORDER BY event_ts, entity_id"
        rows = self._database.connection.execute(query, parameters).fetchall()
        if not rows:
            return pd.DataFrame(columns=_LABELED_COLUMNS)
        records: list[dict[str, object]] = []
        for row in rows:
            values = dict(row)
            features = json.loads(str(values["features_json"]))
            records.append(
                {
                    "prediction_id": values["prediction_id"],
                    "entity_id": int(str(values["entity_id"])),
                    "event_ts": datetime.fromisoformat(str(values["event_ts"])),
                    "scored_at": datetime.fromisoformat(str(values["scored_at"])),
                    "model_name": values["model_name"],
                    "model_version": int(str(values["model_version"])),
                    "prediction": float(str(values["prediction"])),
                    "source": values["source"],
                    "request_id": values["request_id"],
                    **{name: float(features[name]) for name in FEATURE_COLUMNS},
                    "label": float(str(values["label"])),
                }
            )
        frame = pd.DataFrame(records, columns=_LABELED_COLUMNS)
        frame["entity_id"] = frame["entity_id"].astype("int64")
        frame["model_version"] = frame["model_version"].astype("int64")
        for name in (*FEATURE_COLUMNS, "prediction", "label"):
            frame[name] = frame[name].astype("float64")
        return frame.reset_index(drop=True)

    def all_labeled_features(self, end: datetime) -> pd.DataFrame:
        """Return labeled production rows before `end`, shaped like a training frame."""
        rows = self._database.connection.execute(
            """
            SELECT features_json, event_ts, entity_id, label
            FROM labeled_predictions
            WHERE event_ts < ?
            ORDER BY event_ts, entity_id
            """,
            (end.isoformat(),),
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=_TRAINING_COLUMNS)
        records: list[dict[str, object]] = []
        for row in rows:
            values = dict(row)
            features = json.loads(str(values["features_json"]))
            records.append(
                {
                    **{name: float(features[name]) for name in FEATURE_COLUMNS},
                    "timestamp": datetime.fromisoformat(str(values["event_ts"])),
                    "entity_id": int(str(values["entity_id"])),
                    "cnt": float(str(values["label"])),
                }
            )
        frame = pd.DataFrame(records, columns=_TRAINING_COLUMNS)
        frame["entity_id"] = frame["entity_id"].astype("int64")
        for name in (*FEATURE_COLUMNS, "cnt"):
            frame[name] = frame[name].astype("float64")
        return frame.reset_index(drop=True)


__all__ = ["SqlitePredictionLog"]
