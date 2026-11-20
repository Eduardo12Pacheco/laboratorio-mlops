"""Scoring service: batch validation, champion loading, idempotent logging and gateway."""

from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.application.dto import LabelRecord, TrainRequest
from mlops_lifecycle_lab.application.services.scoring import InProcessGateway, PredictionService
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.errors import (
    NoChampionError,
    ValidationFailedError,
)
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from mlops_lifecycle_lab.domain.windows import TimeWindow
from tests.fakes.clock import FakeClock
from tests.fakes.data import StaticLabeledData, raw_frame
from tests.fakes.prediction_log import InMemoryPredictionLog
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.tracker import InMemoryTracker
from tests.fakes.trainer import FakeTrainer, FunctionPredictor

NOW = datetime(2012, 6, 10, 12)
TRAIN = TimeWindow(datetime(2011, 1, 1), datetime(2011, 1, 3))
HOLDOUT = TimeWindow(datetime(2011, 1, 3), datetime(2011, 1, 5))


class Harness:
    """Wire a trained champion into the scoring service and prediction log."""

    def __init__(self, *, train: bool = True) -> None:
        self.clock = FakeClock(NOW)
        self.frame = raw_frame(periods=96)
        self.data = StaticLabeledData(self.frame)
        self.registry = InMemoryRegistry()
        self.tracker = InMemoryTracker(self.registry)
        self.log = InMemoryPredictionLog()
        self.service = PredictionService(
            self.registry,
            self.log,
            self.clock,
        )
        if train:
            trainer = FakeTrainer()
            train_service = TrainCandidate(
                self.data,
                trainer,
                self.tracker,
                self.registry,
                self.clock,
                seed=42,
            )
            result = train_service.execute(
                TrainRequest(
                    train_window=TRAIN,
                    holdout_window=HOLDOUT,
                    trigger_reason="BOOTSTRAP",
                    tick=0,
                )
            )
            self.version = result.version
            self.registry.promote(self.version)

    def set_predictions(self, value: float | np.ndarray) -> None:
        """Replace the champion predictor with a deterministic function."""
        if isinstance(value, np.ndarray):
            array = value
            predictor = FunctionPredictor(lambda features: np.resize(array, len(features)))
        else:
            predictor = FunctionPredictor(lambda features: np.full(len(features), value))
        self.registry.set_predictor(self.version, predictor)


def scoring_rows(count: int = 5) -> pd.DataFrame:
    """Return the first `count` raw rows to score."""
    return raw_frame(periods=96).iloc[:count].copy().reset_index(drop=True)


def test_predict_returns_one_non_negative_result_per_row() -> None:
    harness = Harness()
    harness.set_predictions(42.0)
    rows = scoring_rows()

    results = harness.service.predict(rows)

    assert len(results) == 5
    assert all(result.prediction == 42.0 for result in results)
    assert all(result.model_version == harness.version for result in results)
    assert all(result.model_name == "bike-demand" for result in results)
    assert all(result.scored_at == NOW for result in results)
    assert [result.entity_id for result in results] == [1, 2, 3, 4, 5]


def test_predictions_are_clipped_at_zero() -> None:
    harness = Harness()
    harness.set_predictions(-5.0)

    results = harness.service.predict(scoring_rows())

    assert [result.prediction for result in results] == [0.0] * 5
    assert {record.prediction for record in harness.log.records.values()} == {0.0}


def test_predictions_are_logged_with_lineage_and_features() -> None:
    harness = Harness()
    harness.set_predictions(12.5)
    rows = scoring_rows()

    harness.service.predict(rows, request_id="req-1")

    records = list(harness.log.records.values())
    assert len(records) == 5
    record = records[0]
    assert record.prediction_id == str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"bike-demand:{harness.version}:1")
    )
    assert record.entity_id == 1
    assert record.event_ts == pd.Timestamp(rows.loc[0, "timestamp"]).to_pydatetime()
    assert record.scored_at == NOW
    assert record.model_version == harness.version
    assert record.source == "online"
    assert record.request_id == "req-1"
    assert list(record.features) == list(FEATURE_COLUMNS)
    assert record.features["hr"] == 0.0
    assert record.prediction == 12.5


def test_resending_the_same_rows_is_idempotent() -> None:
    harness = Harness()
    harness.set_predictions(7.0)
    rows = scoring_rows()

    first = harness.service.predict(rows)
    second = harness.service.predict(rows)

    assert first == second
    assert len(harness.log.records) == 5


def test_an_invalid_batch_is_rejected_without_logging_anything() -> None:
    harness = Harness()
    harness.set_predictions(7.0)
    rows = scoring_rows()
    rows.loc[2, "temp"] = 2.0

    with pytest.raises(ValidationFailedError) as caught:
        harness.service.predict(rows)

    assert "numeric_range" in caught.value.failed_rules
    assert harness.log.records == {}


def test_a_batch_above_the_cap_is_rejected() -> None:
    harness = Harness()
    harness.set_predictions(7.0)
    rows = raw_frame(periods=1001)

    with pytest.raises(ValidationFailedError) as caught:
        harness.service.predict(rows)

    assert "batch_size" in caught.value.failed_rules
    assert harness.log.records == {}


def test_predict_never_mutates_the_input_frame() -> None:
    harness = Harness()
    harness.set_predictions(7.0)
    rows = scoring_rows()
    before = rows.copy(deep=True)

    harness.service.predict(rows)

    assert rows.equals(before)


def test_serving_without_a_champion_raises() -> None:
    harness = Harness(train=False)

    with pytest.raises(NoChampionError):
        harness.service.predict(scoring_rows())


def test_the_in_process_gateway_scores_as_batch_and_submits_labels() -> None:
    harness = Harness()
    harness.set_predictions(9.0)
    gateway = InProcessGateway(harness.service, harness.log)
    rows = scoring_rows(3)

    results = gateway.predict_batch(rows)
    gateway.submit_labels(
        [LabelRecord(entity_id=result.entity_id, label=10.0, label_ts=NOW) for result in results]
    )

    assert len(results) == 3
    assert {record.source for record in harness.log.records.values()} == {"batch"}
    labeled = harness.log.labeled_predictions(datetime(2011, 1, 1), datetime(2011, 1, 2))
    assert len(labeled) == 3
    assert set(labeled["label"]) == {10.0}
