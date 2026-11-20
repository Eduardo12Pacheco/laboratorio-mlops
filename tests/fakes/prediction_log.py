"""In-memory prediction log implementing the idempotent S9 contract."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from mlops_lifecycle_lab.application.dto import LabelRecord, PredictionRecord
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS

_PREDICTION_COLUMNS = [
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
_TRAINING_COLUMNS = [*FEATURE_COLUMNS, "timestamp", "entity_id", "cnt"]


class InMemoryPredictionLog:
    """Prediction log keyed by prediction id with labels keyed by entity id."""

    def __init__(self) -> None:
        self.records: dict[str, PredictionRecord] = {}
        self.labels: dict[int, LabelRecord] = {}

    def append(self, records: Sequence[PredictionRecord]) -> None:
        """Insert records, ignoring a repeated prediction id."""
        for record in records:
            self.records.setdefault(record.prediction_id, record)

    def attach_labels(self, labels: Sequence[LabelRecord]) -> None:
        """Attach labels by entity id, ignoring repeated entities."""
        for label in labels:
            self.labels.setdefault(label.entity_id, label)

    def labeled_predictions(
        self,
        start: datetime,
        end: datetime,
        model_version: int | None = None,
    ) -> pd.DataFrame:
        """Return labeled predictions inside the half-open window."""
        rows: list[dict[str, object]] = []
        for record in self.records.values():
            label = self.labels.get(record.entity_id)
            if label is None:
                continue
            if not start <= record.event_ts < end:
                continue
            if model_version is not None and record.model_version != model_version:
                continue
            rows.append(
                {
                    "prediction_id": record.prediction_id,
                    "entity_id": record.entity_id,
                    "event_ts": record.event_ts,
                    "scored_at": record.scored_at,
                    "model_name": record.model_name,
                    "model_version": record.model_version,
                    "prediction": record.prediction,
                    "source": record.source,
                    "request_id": record.request_id,
                    **dict(record.features),
                    "label": label.label,
                }
            )
        if not rows:
            return pd.DataFrame(columns=_PREDICTION_COLUMNS)
        return (
            pd.DataFrame(rows, columns=_PREDICTION_COLUMNS)
            .sort_values(["event_ts", "entity_id"])
            .reset_index(drop=True)
        )

    def all_labeled_features(self, end: datetime) -> pd.DataFrame:
        """Return labeled production rows before `end` shaped like a training frame."""
        rows: list[dict[str, object]] = []
        for record in self.records.values():
            label = self.labels.get(record.entity_id)
            if label is None or record.event_ts >= end:
                continue
            rows.append(
                {
                    **dict(record.features),
                    "timestamp": record.event_ts,
                    "entity_id": record.entity_id,
                    "cnt": label.label,
                }
            )
        if not rows:
            return pd.DataFrame(columns=_TRAINING_COLUMNS)
        return (
            pd.DataFrame(rows, columns=_TRAINING_COLUMNS)
            .sort_values(["timestamp", "entity_id"])
            .reset_index(drop=True)
        )
