"""The validation engine is pure, total over its rules, and honest about severities."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_lifecycle_lab.domain.schema import RAW_COLUMNS
from mlops_lifecycle_lab.domain.validation import (
    Severity,
    ValidationReport,
    Verdict,
    validate_frame,
)

START = datetime(2012, 1, 1)


def make_frame(rows: int = 6, start: datetime = START) -> pd.DataFrame:
    """Build a small, fully valid training-context frame."""
    timestamps = [start + timedelta(hours=i) for i in range(rows)]
    values = np.linspace(0.1, 0.5, rows) if rows else np.array([], dtype=float)
    return pd.DataFrame(
        {
            "entity_id": np.arange(1, rows + 1, dtype=np.int64),
            "timestamp": pd.to_datetime(timestamps),
            "season": np.full(rows, 1, dtype=np.int64),
            "holiday": np.zeros(rows, dtype=np.int64),
            "workingday": np.ones(rows, dtype=np.int64),
            "weathersit": np.full(rows, 1, dtype=np.int64),
            "temp": values,
            "atemp": values + 0.05,
            "hum": values + 0.2,
            "windspeed": values / 2,
            "casual": np.full(rows, 1, dtype=np.int64),
            "registered": np.full(rows, 2, dtype=np.int64),
            "cnt": np.full(rows, 3, dtype=np.int64),
        }
    )


def frame_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def failed_names(report: ValidationReport) -> tuple[str, ...]:
    return tuple(rule.name for rule in report.rules if not rule.passed)


def test_valid_training_frame_passes() -> None:
    report = validate_frame(make_frame(), context="training")

    assert report.verdict is Verdict.PASS
    assert report.rows == 6
    assert failed_names(report) == ()
    assert {rule.name for rule in report.rules} == {
        "required_columns",
        "no_nulls",
        "timestamp_hourly",
        "timestamp_unique",
        "timestamp_monotonic",
        "season_domain",
        "binary_domain",
        "weathersit_domain",
        "numeric_range",
        "hum_positive",
        "gap_hours",
        "label_consistency",
    }


def test_batch_context_swaps_label_rule_for_batch_size_rule() -> None:
    report = validate_frame(make_frame(), context="batch")

    names = {rule.name for rule in report.rules}
    assert "label_consistency" not in names
    assert "batch_size" in names
    assert report.verdict is Verdict.PASS


def test_zero_humidity_is_a_warning_not_an_error() -> None:
    frame = make_frame()
    frame.loc[2, "hum"] = 0.0

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.WARN
    assert failed_names(report) == ("hum_positive",)
    (rule,) = [rule for rule in report.rules if not rule.passed]
    assert rule.severity is Severity.WARN
    assert rule.violations == 1


def test_missing_required_column_fails() -> None:
    report = validate_frame(make_frame().drop(columns=["weathersit"]), context="batch")

    assert report.verdict is Verdict.FAIL
    assert "required_columns" in failed_names(report)


def test_training_context_requires_label_columns() -> None:
    report = validate_frame(make_frame().drop(columns=["cnt"]), context="training")

    assert report.verdict is Verdict.FAIL
    assert "required_columns" in failed_names(report)


def test_null_values_fail() -> None:
    frame = make_frame()
    frame.loc[1, "temp"] = np.nan

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.FAIL
    assert "no_nulls" in failed_names(report)


def test_non_hourly_timestamps_fail() -> None:
    frame = make_frame()
    frame.loc[2, "timestamp"] += pd.Timedelta(minutes=30)

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.FAIL
    assert failed_names(report) == ("timestamp_hourly",)


def test_duplicate_timestamps_fail() -> None:
    frame = make_frame()
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.FAIL
    assert "timestamp_unique" in failed_names(report)


def test_non_monotonic_timestamps_fail() -> None:
    frame = make_frame()
    frame = frame.iloc[::-1].reset_index(drop=True)

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.FAIL
    assert "timestamp_monotonic" in failed_names(report)


def test_season_domain_fails_out_of_range() -> None:
    frame = make_frame()
    frame.loc[0, "season"] = 5

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.FAIL
    assert failed_names(report) == ("season_domain",)


def test_binary_domain_fails_for_holiday_and_workingday() -> None:
    frame = make_frame()
    frame.loc[0, "holiday"] = 2
    frame.loc[1, "workingday"] = -1

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.FAIL
    assert failed_names(report) == ("binary_domain",)
    (rule,) = [rule for rule in report.rules if rule.name == "binary_domain"]
    assert rule.violations == 2


def test_weathersit_domain_fails_out_of_range() -> None:
    frame = make_frame()
    frame.loc[0, "weathersit"] = 0

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.FAIL
    assert failed_names(report) == ("weathersit_domain",)


@pytest.mark.parametrize("column", ["temp", "atemp", "hum", "windspeed"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_numeric_range_fails_outside_unit_interval(column: str, value: float) -> None:
    frame = make_frame()
    frame.loc[0, column] = value

    report = validate_frame(frame, context="batch")

    expected = {"numeric_range"}
    if column == "hum" and value <= 0:
        expected.add("hum_positive")
    assert report.verdict is Verdict.FAIL
    assert set(failed_names(report)) == expected


def test_gap_over_forty_eight_hours_is_a_warning() -> None:
    frame = make_frame(rows=3)
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"] + pd.Timedelta(hours=60)

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.WARN
    assert failed_names(report) == ("gap_hours",)


def test_gap_of_exactly_forty_eight_hours_passes() -> None:
    frame = make_frame(rows=3)
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"] + pd.Timedelta(hours=48)

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.PASS
    assert failed_names(report) == ()


def test_label_consistency_fails_in_training_context() -> None:
    frame = make_frame()
    frame.loc[0, "cnt"] = 99

    report = validate_frame(frame, context="training")

    assert report.verdict is Verdict.FAIL
    assert failed_names(report) == ("label_consistency",)


def test_negative_label_fails_in_training_context() -> None:
    frame = make_frame()
    frame.loc[0, "cnt"] = -1
    frame.loc[0, "casual"] = -2
    frame.loc[0, "registered"] = 1

    report = validate_frame(frame, context="training")

    assert report.verdict is Verdict.FAIL
    assert failed_names(report) == ("label_consistency",)


def test_label_mismatch_is_ignored_in_batch_context() -> None:
    frame = make_frame()
    frame.loc[0, "cnt"] = 99

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.PASS


def test_empty_batch_fails_batch_size() -> None:
    report = validate_frame(make_frame(rows=0), context="batch")

    assert report.verdict is Verdict.FAIL
    assert failed_names(report) == ("batch_size",)


def test_oversized_batch_fails_batch_size() -> None:
    report = validate_frame(make_frame(rows=1001), context="batch")

    assert report.verdict is Verdict.FAIL
    assert failed_names(report) == ("batch_size",)


def test_thousand_row_batch_passes_batch_size() -> None:
    report = validate_frame(make_frame(rows=1000), context="batch")

    assert report.verdict is Verdict.PASS


def test_training_context_does_not_apply_batch_size() -> None:
    report = validate_frame(make_frame(rows=1001), context="training")

    assert report.verdict is Verdict.PASS


def test_report_dict_is_serialisable_and_names_failed_rules() -> None:
    frame = make_frame()
    frame.loc[0, "season"] = 9

    payload = validate_frame(frame, context="batch").to_dict()

    assert payload["verdict"] == "FAIL"
    assert payload["rows"] == 6
    assert payload["failed_rules"] == ["season_domain"]
    assert payload["rules"]["season_domain"]["passed"] is False
    assert payload["rules"]["season_domain"]["violations"] == 1
    assert payload["rules"]["timestamp_hourly"]["passed"] is True


def test_validation_never_mutates_the_input_frame() -> None:
    frame = make_frame()
    frame.loc[0, "temp"] = 7.0
    before = frame_hash(frame)

    validate_frame(frame, context="batch")

    assert frame_hash(frame) == before


def test_unknown_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="context"):
        validate_frame(make_frame(), context="serving")  # type: ignore[arg-type]


@given(rows=st.integers(min_value=1, max_value=40))
def test_any_generated_valid_batch_passes(rows: int) -> None:
    report = validate_frame(make_frame(rows=rows), context="batch")

    assert report.verdict is Verdict.PASS
    assert failed_names(report) == ()


MUTATIONS: dict[str, tuple[float, str]] = {
    "season": (0, "season_domain"),
    "holiday": (2, "binary_domain"),
    "workingday": (5, "binary_domain"),
    "weathersit": (9, "weathersit_domain"),
    "temp": (1.5, "numeric_range"),
    "atemp": (-0.5, "numeric_range"),
    "hum": (1.5, "numeric_range"),
    "windspeed": (2.0, "numeric_range"),
}


@settings(max_examples=50)
@given(rows=st.integers(min_value=1, max_value=40), column=st.sampled_from(sorted(MUTATIONS)))
def test_single_out_of_domain_cell_fails_exactly_its_rule(rows: int, column: str) -> None:
    frame = make_frame(rows=rows)
    value, expected_rule = MUTATIONS[column]
    frame.loc[0, column] = value

    report = validate_frame(frame, context="batch")

    assert report.verdict is Verdict.FAIL
    assert failed_names(report) == (expected_rule,)


def test_feature_frame_does_not_satisfy_the_raw_contract() -> None:
    feature_like = make_frame().drop(columns=list(RAW_COLUMNS))

    report = validate_frame(feature_like, context="batch")

    assert report.verdict is Verdict.FAIL
    assert "required_columns" in failed_names(report)
