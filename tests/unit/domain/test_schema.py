"""The schema module is the single source of truth for column names and constants."""

from __future__ import annotations

from datetime import datetime

from mlops_lifecycle_lab.domain.schema import (
    ARCHIVE_END,
    FEATURE_COLUMNS,
    FEATURE_SPEC_VERSION,
    LABEL_COLUMNS,
    LEAKAGE_COLUMNS,
    MONITORED_CATEGORICAL,
    MONITORED_NUMERIC,
    RAW_COLUMNS,
    SEGMENT_COLUMN,
    TARGET_COLUMN,
    TREND_EPOCH,
)


def test_raw_columns_are_the_serving_contract() -> None:
    assert RAW_COLUMNS == (
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
    )


def test_label_columns_are_the_target_contract() -> None:
    assert LABEL_COLUMNS == ("casual", "registered", "cnt")
    assert TARGET_COLUMN == "cnt"


def test_feature_columns_are_exactly_twelve_in_order() -> None:
    assert len(FEATURE_COLUMNS) == 12
    assert FEATURE_COLUMNS == (
        "season",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "t_days",
    )


def test_no_leakage_column_is_a_feature() -> None:
    assert set(LEAKAGE_COLUMNS) == {"yr", "instant", "casual", "registered", "cnt"}
    assert set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_COLUMNS)


def test_monitoring_and_segment_columns() -> None:
    assert MONITORED_NUMERIC == ("temp", "atemp", "hum", "windspeed")
    assert MONITORED_CATEGORICAL == ("season", "weathersit")
    assert SEGMENT_COLUMN == "workingday"
    assert set(MONITORED_NUMERIC) | set(MONITORED_CATEGORICAL) <= set(FEATURE_COLUMNS)


def test_trend_and_archive_epochs() -> None:
    assert datetime(2011, 1, 1) == TREND_EPOCH
    assert datetime(2012, 1, 1) == ARCHIVE_END
    assert FEATURE_SPEC_VERSION == "1"
