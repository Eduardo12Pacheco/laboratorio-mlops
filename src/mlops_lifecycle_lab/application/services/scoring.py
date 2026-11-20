"""Serving-side application service: validate, score, clip and log (S8, S9).

`PredictionService` is the single code path behind the online endpoint, the batch
endpoint and the in-process gateway used by the lifecycle tick. It never logs a rejected
batch, clips every prediction at zero and derives deterministic prediction ids so a
client retry cannot duplicate rows.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from mlops_lifecycle_lab.application.dto import (
    LabelRecord,
    PredictionRecord,
    PredictionResult,
    Source,
)
from mlops_lifecycle_lab.application.ports import (
    Clock,
    ModelRegistry,
    PredictionLog,
)
from mlops_lifecycle_lab.application.services._data import clip_predictions
from mlops_lifecycle_lab.domain.errors import NoChampionError, ValidationFailedError
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from mlops_lifecycle_lab.domain.validation import Severity, Verdict, validate_frame

_PREDICTION_NAMESPACE = uuid.NAMESPACE_URL


def _as_float(value: object) -> float:
    """Convert a pandas/numpy scalar to a plain float."""
    return float(np.asarray(cast(Any, value)).item())


def _as_datetime(value: object) -> datetime:
    """Convert a pandas timestamp to a naive `datetime`."""
    return pd.Timestamp(cast(Any, value)).to_pydatetime()


class PredictionService:
    """Score raw rows with the current champion and persist every prediction."""

    def __init__(
        self,
        registry: ModelRegistry,
        prediction_log: PredictionLog,
        clock: Clock,
        *,
        model_name: str = "bike-demand",
    ) -> None:
        self._registry = registry
        self._log = prediction_log
        self._clock = clock
        self._model_name = model_name

    def predict(
        self,
        rows: pd.DataFrame,
        *,
        source: Source = "online",
        request_id: str | None = None,
    ) -> list[PredictionResult]:
        """Validate `rows`, score them and append the predictions atomically.

        Raises:
            NoChampionError: when the registry has no champion.
            ValidationFailedError: when the batch fails the data contract; nothing is
                logged in that case.
        """
        champion = self._registry.get_champion()
        if champion is None:
            raise NoChampionError("no champion model is available for serving")
        validation = validate_frame(rows, context="batch")
        if validation.verdict is Verdict.FAIL:
            failed = tuple(
                rule.name
                for rule in validation.rules
                if not rule.passed and rule.severity is Severity.ERROR
            )
            raise ValidationFailedError(failed_rules=failed)

        features = build_features(rows)
        predictor = self._registry.load_predictor(champion.version)
        predictions = clip_predictions(predictor.predict(features.loc[:, list(FEATURE_COLUMNS)]))
        scored_at = self._clock.now()

        entity_ids = pd.to_numeric(rows["entity_id"], errors="raise").to_numpy(dtype=np.int64)
        timestamps = pd.to_datetime(rows["timestamp"]).to_numpy()
        records: list[PredictionRecord] = []
        results: list[PredictionResult] = []
        for position in range(len(features)):
            entity_id = int(entity_ids[position])
            prediction = float(predictions[position])
            feature_row = {
                name: _as_float(features.iloc[position][name]) for name in FEATURE_COLUMNS
            }
            records.append(
                PredictionRecord(
                    prediction_id=str(
                        uuid.uuid5(
                            _PREDICTION_NAMESPACE,
                            f"{self._model_name}:{champion.version}:{entity_id}",
                        )
                    ),
                    entity_id=entity_id,
                    event_ts=_as_datetime(timestamps[position]),
                    scored_at=scored_at,
                    model_name=self._model_name,
                    model_version=champion.version,
                    features=feature_row,
                    prediction=prediction,
                    source=source,
                    request_id=request_id,
                )
            )
            results.append(
                PredictionResult(
                    entity_id=entity_id,
                    prediction=prediction,
                    model_name=self._model_name,
                    model_version=champion.version,
                    scored_at=scored_at,
                )
            )
        self._log.append(records)
        return results


class InProcessGateway:
    """`ScoringGateway` that calls `PredictionService` without HTTP (drills, tests)."""

    def __init__(self, service: PredictionService, prediction_log: PredictionLog) -> None:
        self._service = service
        self._log = prediction_log

    def predict_batch(self, raw_rows: pd.DataFrame) -> list[PredictionResult]:
        """Score a batch through the serving path, tagged as `batch`."""
        return self._service.predict(raw_rows, source="batch")

    def submit_labels(self, labels: Sequence[LabelRecord]) -> None:
        """Attach delayed labels to their predictions."""
        self._log.attach_labels(labels)


__all__ = ["InProcessGateway", "PredictionService"]
