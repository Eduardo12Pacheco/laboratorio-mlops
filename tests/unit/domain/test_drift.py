"""PSI drift statistics: stability, symmetry of scale and monotonicity under shifts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_lifecycle_lab.domain.drift import (
    PSI_DRIFTED,
    PSI_MODERATE,
    DriftSummary,
    FeatureDrift,
    psi_categorical,
    psi_numeric,
    summarize_drift,
)
from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.schema import MONITORED_CATEGORICAL, MONITORED_NUMERIC

FLOATS = st.lists(
    st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    min_size=5,
    max_size=100,
)


def reference_sample(size: int = 5000) -> np.ndarray:
    return np.random.default_rng(42).normal(loc=0.5, scale=0.15, size=size)


def test_thresholds_are_the_documented_ones() -> None:
    assert PSI_MODERATE == 0.1
    assert PSI_DRIFTED == 0.2


def test_psi_of_an_identical_distribution_is_zero() -> None:
    sample = reference_sample()

    assert psi_numeric(sample, sample) == pytest.approx(0.0, abs=1e-12)
    assert psi_categorical([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(0.0, abs=1e-12)


def test_psi_sanity_for_small_and_large_mean_shifts() -> None:
    reference = reference_sample()
    small = reference + 0.1 * np.std(reference)
    large = reference + 1.0 * np.std(reference)

    assert psi_numeric(reference, small) < PSI_MODERATE
    assert psi_numeric(reference, large) > PSI_DRIFTED


def test_psi_grows_with_the_size_of_the_mean_shift() -> None:
    reference = reference_sample()
    std = float(np.std(reference))

    values = [
        psi_numeric(reference, reference + shift * std) for shift in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]

    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_categorical_psi_flags_a_changed_weather_mix() -> None:
    reference = [1] * 70 + [2] * 20 + [3] * 10
    current = [1] * 20 + [2] * 30 + [3] * 50

    assert psi_categorical(reference, current) > PSI_DRIFTED


def test_feature_drift_is_a_small_immutable_record() -> None:
    drift = FeatureDrift(feature="temp", kind="numeric", psi=0.25, drifted=True)

    assert drift.feature == "temp"
    assert drift.kind == "numeric"
    assert drift.drifted is True
    with pytest.raises(AttributeError):
        drift.psi = 0.0  # type: ignore[misc]


def test_summarize_drift_reports_every_monitored_feature_in_order() -> None:
    reference = reference_sample(500)
    current = reference.copy()
    current += 1.0 * np.std(reference)
    reference_features = pd.DataFrame(
        {
            "temp": reference,
            "atemp": reference,
            "hum": reference,
            "windspeed": reference,
            "season": np.resize([1, 2, 3, 4], reference.size),
            "weathersit": np.resize([1, 2, 3, 4], reference.size),
        }
    )
    current_features = reference_features.copy(deep=True)
    for column in MONITORED_CATEGORICAL:
        current_features[column] = np.resize([1, 1, 1, 2], reference.size)
    current_features["temp"] = reference + 1.0 * np.std(reference)

    summary = summarize_drift(
        reference_features,
        current_features,
        reference_pred=reference,
        current_pred=reference + 1.0 * np.std(reference),
    )

    assert isinstance(summary, DriftSummary)
    assert [feature.feature for feature in summary.features] == [
        *MONITORED_NUMERIC,
        *MONITORED_CATEGORICAL,
    ]
    assert [feature.kind for feature in summary.features] == ["numeric"] * 4 + ["categorical"] * 2
    drifted_numeric = [
        feature for feature in summary.features if feature.feature in MONITORED_NUMERIC
    ]
    assert summary.numeric_drift_share == pytest.approx(
        sum(feature.drifted for feature in drifted_numeric) / len(MONITORED_NUMERIC)
    )
    assert summary.prediction_psi == pytest.approx(
        psi_numeric(reference, reference + 1.0 * np.std(reference))
    )
    assert summary.numeric_drift_share > 0.0


def test_summary_dict_is_json_serialisable() -> None:
    reference = reference_sample(200)
    features = pd.DataFrame(
        {
            "temp": reference,
            "atemp": reference,
            "hum": reference,
            "windspeed": reference,
            "season": np.resize([1, 2], reference.size),
            "weathersit": np.resize([1, 2], reference.size),
        }
    )

    payload: dict[str, Any] = summarize_drift(
        features, features, reference_pred=reference, current_pred=reference
    ).to_dict()

    assert payload["numeric_drift_share"] == 0.0
    assert payload["prediction_psi"] == pytest.approx(0.0, abs=1e-12)
    assert set(payload["features"]) == {*MONITORED_NUMERIC, *MONITORED_CATEGORICAL}
    assert payload["features"]["temp"]["kind"] == "numeric"


def test_summarize_drift_requires_the_monitored_columns() -> None:
    reference = reference_sample(20)
    features = pd.DataFrame(
        {
            "temp": reference,
            "atemp": reference,
            "hum": reference,
            "windspeed": reference,
            "season": np.resize([1, 2], reference.size),
            "weathersit": np.resize([1, 2], reference.size),
        }
    )

    with pytest.raises(ValidationFailedError, match="weathersit"):
        summarize_drift(
            features.drop(columns=["weathersit"]),
            features,
            reference_pred=reference,
            current_pred=reference,
        )


def test_empty_or_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        psi_numeric([], [])
    with pytest.raises(ValueError, match="empty"):
        psi_categorical([], [])
    with pytest.raises(ValueError, match="bins"):
        psi_numeric([1.0, 2.0], [1.0, 2.0], bins=1)


def test_constant_reference_is_handled_without_nan() -> None:
    value = psi_numeric([0.5] * 100, [0.5] * 50 + [0.9] * 50)

    assert np.isfinite(value)
    assert value >= 0.0


@settings(max_examples=50, deadline=None)
@given(sample=FLOATS)
def test_psi_is_non_negative(sample: list[float]) -> None:
    values = np.asarray(sample, dtype=np.float64)
    shifted = values + 0.5

    assert psi_numeric(values, shifted) >= 0.0
    assert psi_categorical(values, shifted) >= 0.0


AFFINE_FLOATS = st.lists(
    st.integers(min_value=-1_000, max_value=1_000).map(float),
    min_size=100,
    max_size=200,
)


@settings(max_examples=25, deadline=None)
@given(
    sample=AFFINE_FLOATS,
    scale=st.sampled_from([0.5, 1.0, 2.0, 4.0]),
    offset=st.integers(min_value=-8, max_value=8),
)
def test_psi_is_invariant_to_positive_affine_transforms(
    sample: list[float], scale: float, offset: int
) -> None:
    reference = np.asarray(sample, dtype=np.float64)
    current = reference + 1.0
    transformed_reference = scale * reference + offset
    transformed_current = scale * current + offset

    assert psi_numeric(transformed_reference, transformed_current) == pytest.approx(
        psi_numeric(reference, current), rel=0, abs=1e-12
    )
