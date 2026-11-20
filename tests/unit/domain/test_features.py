"""The shared feature pipeline: exact columns, calendar derivation and parity."""

from __future__ import annotations

import gzip
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset, from_uci_frame
from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.schema import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TREND_EPOCH,
)

INT_FEATURES = ("season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit")
FLOAT_FEATURES = ("temp", "atemp", "hum", "windspeed", "t_days")
REQUIRED_INPUT = (
    "timestamp",
    "season",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
)


def make_raw(rows: int = 4, start: datetime = datetime(2012, 1, 1)) -> pd.DataFrame:
    """Build a small raw frame with every column `build_features` may touch."""
    timestamps = [start + timedelta(hours=i) for i in range(rows)]
    return pd.DataFrame(
        {
            "entity_id": np.arange(1, rows + 1, dtype=np.int64),
            "timestamp": pd.to_datetime(timestamps),
            "season": np.full(rows, 1, dtype=np.int64),
            "holiday": np.zeros(rows, dtype=np.int64),
            "workingday": np.ones(rows, dtype=np.int64),
            "weathersit": np.full(rows, 1, dtype=np.int64),
            "temp": np.linspace(0.1, 0.4, rows),
            "atemp": np.linspace(0.2, 0.5, rows),
            "hum": np.linspace(0.3, 0.6, rows),
            "windspeed": np.linspace(0.0, 0.3, rows),
            "casual": np.full(rows, 1, dtype=np.int64),
            "registered": np.full(rows, 2, dtype=np.int64),
            "cnt": np.full(rows, 3, dtype=np.int64),
        }
    )


def frame_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def test_feature_frame_has_exactly_the_contract_columns_in_order() -> None:
    features = build_features(make_raw())

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert len(features) == 4
    assert set(features.columns).isdisjoint(LEAKAGE_COLUMNS)


def test_calendar_columns_are_derived_from_the_timestamp() -> None:
    raw = make_raw(rows=5, start=datetime(2012, 1, 1, 5))

    features = build_features(raw)

    assert features["mnth"].tolist() == [1, 1, 1, 1, 1]
    assert features["hr"].tolist() == [5, 6, 7, 8, 9]
    # 2012-01-01 was a Sunday, which is weekday 0 in the dataset convention.
    assert features["weekday"].tolist() == [0, 0, 0, 0, 0]


def test_weekday_uses_the_dataset_convention_monday_is_one() -> None:
    raw = make_raw(rows=2, start=datetime(2012, 1, 2, 0))

    features = build_features(raw)

    assert features["weekday"].tolist() == [1, 1]
    assert raw["timestamp"].dt.dayofweek.tolist() == [0, 0]


def test_t_days_is_fractional_days_since_the_trend_epoch() -> None:
    raw = make_raw(rows=3, start=datetime(2011, 1, 1, 0))

    features = build_features(raw)

    assert features["t_days"].tolist() == pytest.approx([0.0, 1 / 24, 2 / 24])
    later = build_features(make_raw(rows=1, start=datetime(2011, 1, 2)))
    assert later["t_days"].iloc[0] == pytest.approx(1.0)


def test_dtypes_are_the_trainer_contract() -> None:
    features = build_features(make_raw())

    for column in INT_FEATURES:
        assert pd.api.types.is_integer_dtype(features[column]), column
        assert str(features[column].dtype) == "int64", column
    for column in FLOAT_FEATURES:
        assert pd.api.types.is_float_dtype(features[column]), column
        assert str(features[column].dtype) == "float64", column


def test_build_features_never_mutates_the_input() -> None:
    raw = make_raw(rows=6)
    before = frame_hash(raw)

    build_features(raw)

    assert frame_hash(raw) == before
    assert list(raw.columns) == [
        "entity_id",
        "timestamp",
        "season",
        "holiday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        "registered",
        "cnt",
    ]


def test_build_features_is_deterministic_and_idempotent() -> None:
    raw = make_raw(rows=6)

    first = build_features(raw)
    second = build_features(raw.copy(deep=True))

    assert first.equals(second)


@pytest.mark.parametrize("column", REQUIRED_INPUT)
def test_missing_input_column_raises_with_its_name(column: str) -> None:
    raw = make_raw().drop(columns=[column])

    with pytest.raises(ValidationFailedError, match=column):
        build_features(raw)


def test_missing_multiple_columns_lists_all_of_them() -> None:
    raw = make_raw().drop(columns=["temp", "timestamp"])

    with pytest.raises(ValidationFailedError) as excinfo:
        build_features(raw)

    assert set(excinfo.value.failed_rules) == {"temp", "timestamp"}


def test_timestamp_accepts_strings_but_still_derives_the_calendar() -> None:
    raw = make_raw(rows=2)
    raw["timestamp"] = raw["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    features = build_features(raw)

    assert features["mnth"].tolist() == [1, 1]
    assert features["hr"].tolist() == [0, 1]


def test_bundled_derived_calendar_matches_the_original_uci_columns() -> None:
    dataset = BundledDataset()
    with gzip.open(dataset.hour_path, mode="rt", encoding="utf-8") as handle:
        uci = pd.read_csv(handle)
    raw = from_uci_frame(uci)

    features = build_features(raw)

    assert len(features) == 17379
    assert features["mnth"].to_numpy().tolist() == uci["mnth"].to_numpy().tolist()
    assert features["hr"].to_numpy().tolist() == uci["hr"].to_numpy().tolist()
    assert features["weekday"].to_numpy().tolist() == uci["weekday"].to_numpy().tolist()
    assert features["season"].to_numpy().tolist() == uci["season"].to_numpy().tolist()


@settings(max_examples=50, deadline=None)
@given(
    timestamps=st.lists(
        st.datetimes(
            min_value=datetime(2011, 1, 1),
            max_value=datetime(2012, 12, 30),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_generated_raw_frames_keep_the_feature_contract(
    timestamps: list[datetime],
) -> None:
    ordered = pd.to_datetime(sorted(timestamps))
    raw = make_raw(rows=len(ordered))
    raw["timestamp"] = ordered
    before = frame_hash(raw)

    features = build_features(raw)

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert len(features) == len(ordered)
    assert set(features.columns).isdisjoint(LEAKAGE_COLUMNS)
    assert features["mnth"].tolist() == [moment.month for moment in ordered]
    assert features["hr"].tolist() == [moment.hour for moment in ordered]
    assert features["weekday"].tolist() == [(moment.weekday() + 1) % 7 for moment in ordered]
    assert features["t_days"].notna().all()
    assert (features["t_days"] >= 0).all()
    expected_t_days = [
        (pd.Timestamp(moment) - pd.Timestamp(TREND_EPOCH)) / pd.Timedelta(days=1)
        for moment in ordered
    ]
    assert features["t_days"].tolist() == pytest.approx(expected_t_days)
    assert frame_hash(raw) == before
