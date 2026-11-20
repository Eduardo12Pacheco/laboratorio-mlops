"""Shared prediction-log contract: in-memory fake today, SQLite from T6.

The suite uses only the `PredictionLog` port. It pins the behaviours monitoring relies
on: idempotent writes, the labeled join, the half-open window, the model filter and the
training-frame shape of `all_labeled_features`.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import pandas as pd
import pytest

from mlops_lifecycle_lab.application.dto import LabelRecord, PredictionRecord
from mlops_lifecycle_lab.application.ports import PredictionLog
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from tests.fakes.prediction_log import InMemoryPredictionLog

START = datetime(2011, 1, 1)
BASE_FEATURES: dict[str, float] = {
    "season": 1.0,
    "mnth": 1.0,
    "hr": 0.0,
    "holiday": 0.0,
    "weekday": 6.0,
    "workingday": 0.0,
    "weathersit": 1.0,
    "temp": 0.3,
    "atemp": 0.35,
    "hum": 0.5,
    "windspeed": 0.2,
    "t_days": 0.0,
}


@pytest.fixture(params=["memory"])
def log(request: pytest.FixtureRequest) -> Iterator[PredictionLog]:
    """Provide each prediction-log backend under contract (T6 adds SQLite)."""
    backend = str(request.param)
    if backend == "memory":
        yield InMemoryPredictionLog()
    else:  # pragma: no cover - guards a typo in the parameter list
        raise AssertionError(f"unknown prediction-log backend {backend!r}")


def record(
    entity_id: int,
    *,
    hour: int = 0,
    model_version: int = 1,
    prediction: float = 10.0,
    prediction_id: str | None = None,
    source: str = "online",
) -> PredictionRecord:
    """Build one prediction record at `START + hour`."""
    features = {**BASE_FEATURES, "hr": float(hour), "t_days": float(hour) / 24.0}
    return PredictionRecord(
        prediction_id=prediction_id or f"pred-{model_version}-{entity_id}",
        entity_id=entity_id,
        event_ts=START + timedelta(hours=hour),
        scored_at=START + timedelta(hours=hour, minutes=5),
        model_name="bike-demand",
        model_version=model_version,
        features=features,
        prediction=prediction,
        source=source,  # type: ignore[arg-type]
    )


def label(entity_id: int, value: float = 12.0, hour: int = 0) -> LabelRecord:
    """Build one delayed label record."""
    return LabelRecord(entity_id=entity_id, label=value, label_ts=START + timedelta(hours=hour))


def test_predictions_are_invisible_until_the_label_arrives(log: PredictionLog) -> None:
    log.append([record(1), record(2)])

    assert log.labeled_predictions(START, START + timedelta(days=1)).empty
    log.attach_labels([label(1), label(2)])
    labeled = log.labeled_predictions(START, START + timedelta(days=1))

    assert len(labeled) == 2
    assert list(labeled["entity_id"]) == [1, 2]
    assert set(labeled["label"]) == {12.0}


def test_the_labeled_join_keeps_features_and_prediction(log: PredictionLog) -> None:
    log.append([record(1, prediction=42.0)])
    log.attach_labels([label(1, value=40.0)])

    labeled = log.labeled_predictions(START, START + timedelta(days=1))

    row = labeled.iloc[0]
    assert row["prediction"] == 42.0
    assert row["label"] == 40.0
    assert row["model_version"] == 1
    assert row["source"] == "online"
    for feature in FEATURE_COLUMNS:
        assert feature in labeled.columns
    assert float(row["hr"]) == 0.0


def test_appending_the_same_prediction_twice_keeps_one_row(log: PredictionLog) -> None:
    log.append([record(1, prediction=10.0)])
    log.append([record(1, prediction=99.0)])
    log.attach_labels([label(1)])

    labeled = log.labeled_predictions(START, START + timedelta(days=1))

    assert len(labeled) == 1
    assert labeled.iloc[0]["prediction"] == 10.0


def test_the_window_is_half_open(log: PredictionLog) -> None:
    log.append([record(1, hour=0), record(2, hour=1), record(3, hour=2)])
    log.attach_labels([label(1, hour=0), label(2, hour=1), label(3, hour=2)])

    labeled = log.labeled_predictions(START, START + timedelta(hours=2))

    assert list(labeled["entity_id"]) == [1, 2]


def test_labeled_predictions_can_be_filtered_by_model_version(log: PredictionLog) -> None:
    log.append([record(1, model_version=1), record(2, model_version=2)])
    log.attach_labels([label(1), label(2)])

    labeled = log.labeled_predictions(START, START + timedelta(days=1), model_version=2)

    assert list(labeled["model_version"]) == [2]
    assert list(labeled["entity_id"]) == [2]


def test_labels_for_unknown_predictions_create_no_rows(log: PredictionLog) -> None:
    log.append([record(1)])
    log.attach_labels([label(99)])

    assert log.labeled_predictions(START, START + timedelta(days=1)).empty


def test_all_labeled_features_returns_the_training_frame_shape(log: PredictionLog) -> None:
    log.append([record(1, hour=0), record(2, hour=1)])
    log.attach_labels([label(1, value=10.0, hour=0), label(2, value=20.0, hour=1)])

    frame = log.all_labeled_features(START + timedelta(days=1))

    assert list(frame.columns) == [*FEATURE_COLUMNS, "timestamp", "entity_id", "cnt"]
    assert list(frame["entity_id"]) == [1, 2]
    assert list(frame["cnt"]) == [10.0, 20.0]
    assert frame["timestamp"].is_monotonic_increasing


def test_all_labeled_features_is_end_exclusive(log: PredictionLog) -> None:
    log.append([record(1, hour=0), record(2, hour=1)])
    log.attach_labels([label(1, hour=0), label(2, hour=1)])

    frame = log.all_labeled_features(START + timedelta(hours=1))

    assert list(frame["entity_id"]) == [1]


def test_an_empty_log_returns_typed_empty_frames(log: PredictionLog) -> None:
    labeled = log.labeled_predictions(START, START + timedelta(days=1))
    features = log.all_labeled_features(START + timedelta(days=1))

    assert labeled.empty
    assert list(labeled.columns)[:3] == ["prediction_id", "entity_id", "event_ts"]
    assert features.empty
    assert list(features.columns) == [*FEATURE_COLUMNS, "timestamp", "entity_id", "cnt"]


def test_batch_predictions_keep_their_source(log: PredictionLog) -> None:
    log.append([record(1, source="batch")])
    log.attach_labels([label(1)])

    labeled = log.labeled_predictions(START, START + timedelta(days=1))

    assert labeled.iloc[0]["source"] == "batch"
    assert isinstance(labeled, pd.DataFrame)
