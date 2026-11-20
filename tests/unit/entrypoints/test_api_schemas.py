"""Request/response schemas of the serving API: ranges, hourly timestamps, batch cap."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.schema import RAW_COLUMNS
from mlops_lifecycle_lab.entrypoints.api.schemas import (
    BatchPredictRequest,
    LabelsRequest,
    PredictRequest,
    raw_frame,
)


def payload(**overrides: object) -> dict[str, object]:
    """A valid raw prediction payload, with optional overrides."""
    base: dict[str, object] = {
        "entity_id": 1,
        "timestamp": "2012-01-01T00:00:00",
        "season": 1,
        "holiday": 0,
        "workingday": 0,
        "weathersit": 1,
        "temp": 0.24,
        "atemp": 0.28,
        "hum": 0.81,
        "windspeed": 0.1,
    }
    base.update(overrides)
    return base


def test_predict_request_accepts_a_valid_row() -> None:
    request = PredictRequest.model_validate(payload())

    assert request.entity_id == 1
    assert request.timestamp == datetime(2012, 1, 1)
    assert request.temp == pytest.approx(0.24)


@pytest.mark.parametrize(
    "overrides",
    [
        {"temp": 1.01},
        {"temp": -0.01},
        {"season": 5},
        {"season": 0},
        {"weathersit": 5},
        {"holiday": 2},
        {"workingday": -1},
        {"hum": 1.5},
        {"windspeed": 1.1},
        {"entity_id": -1},
        {"unexpected": "field"},
    ],
)
def test_predict_request_rejects_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(payload(**overrides))


@pytest.mark.parametrize(
    "timestamp",
    ["2012-01-01T00:30:00", "2012-01-01T00:00:01", "2012-01-01T00:00:00.500000"],
)
def test_predict_request_requires_hourly_aligned_timestamps(timestamp: str) -> None:
    with pytest.raises(ValidationError, match="hourly"):
        PredictRequest.model_validate(payload(timestamp=timestamp))


def test_predict_request_rejects_timezone_aware_timestamps() -> None:
    with pytest.raises(ValidationError, match="naive"):
        PredictRequest.model_validate(payload(timestamp="2012-01-01T00:00:00+00:00"))


def test_batch_request_caps_rows_at_one_thousand() -> None:
    valid = payload()
    assert len(BatchPredictRequest.model_validate({"rows": [valid] * 1000}).rows) == 1000
    with pytest.raises(ValidationError):
        BatchPredictRequest.model_validate({"rows": [valid] * 1001})
    with pytest.raises(ValidationError):
        BatchPredictRequest.model_validate({"rows": []})


def test_labels_request_rejects_negative_labels() -> None:
    with pytest.raises(ValidationError):
        LabelsRequest.model_validate(
            {"labels": [{"entity_id": 1, "label": -1.0, "label_ts": "2012-01-01T01:00:00"}]}
        )


def test_raw_frame_keeps_the_raw_column_order_and_feeds_build_features() -> None:
    rows = [
        PredictRequest.model_validate(payload(entity_id=1)),
        PredictRequest.model_validate(
            payload(entity_id=2, timestamp="2012-01-01T01:00:00", temp=0.5)
        ),
    ]

    frame = raw_frame(rows)
    assert list(frame.columns) == list(RAW_COLUMNS)
    assert frame["entity_id"].tolist() == [1, 2]
    assert isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype) is False

    expected = pd.DataFrame(
        [{name: getattr(row, name) for name in RAW_COLUMNS} for row in rows],
        columns=list(RAW_COLUMNS),
    )
    pd.testing.assert_frame_equal(
        build_features(frame).reset_index(drop=True),
        build_features(expected).reset_index(drop=True),
    )
