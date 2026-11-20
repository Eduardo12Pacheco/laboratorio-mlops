"""Regression metrics: definitions, scale invariance and input hygiene."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlops_lifecycle_lab.domain.metrics import RegressionMetrics, compute_metrics

_ERRORS = st.lists(
    st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=50,
)


def test_compute_metrics_known_values() -> None:
    metrics = compute_metrics([3.0, 4.0], [2.0, 2.0])

    assert metrics.n == 2
    assert metrics.rmse == pytest.approx(np.sqrt(2.5))
    assert metrics.mae == pytest.approx(1.5)
    assert metrics.wape == pytest.approx(3.0 / 7.0)
    assert metrics.bias == pytest.approx(-1.5)


def test_perfect_predictions_have_zero_error() -> None:
    metrics = compute_metrics([10, 20, 30], [10, 20, 30])

    assert metrics == RegressionMetrics(rmse=0.0, mae=0.0, wape=0.0, bias=0.0, n=3)


def test_bias_is_mean_prediction_minus_actual() -> None:
    over = compute_metrics([1.0, 2.0], [3.0, 4.0])
    under = compute_metrics([3.0, 4.0], [1.0, 2.0])

    assert over.bias == pytest.approx(2.0)
    assert under.bias == pytest.approx(-2.0)


def test_metrics_accept_pandas_and_numpy_inputs() -> None:
    from_numpy = compute_metrics(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    from_pandas = compute_metrics(pd.Series([1.0, 2.0]), pd.Series([2.0, 4.0]))

    assert from_numpy == from_pandas


def test_metrics_are_frozen() -> None:
    metrics = compute_metrics([1.0], [1.0])

    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.rmse = 1.0  # type: ignore[misc]


def test_to_dict_is_json_serialisable() -> None:
    payload = compute_metrics([2.0, 3.0], [1.0, 5.0]).to_dict()

    assert payload["n"] == 2
    assert set(payload) == {"rmse", "mae", "wape", "bias", "n"}
    assert all(isinstance(payload[key], float) for key in ("rmse", "mae", "wape", "bias"))


def test_zero_actual_total_raises_for_wape() -> None:
    with pytest.raises(ValueError, match="WAPE"):
        compute_metrics([0.0, 0.0], [1.0, 2.0])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        compute_metrics([1.0, 2.0], [1.0])


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [])


def test_nan_input_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_metrics([1.0, float("nan")], [1.0, 1.0])


@settings(max_examples=100, deadline=None)
@given(errors=_ERRORS, actuals=_ERRORS)
def test_rmse_is_at_least_mae_and_both_are_non_negative(
    errors: list[float], actuals: list[float]
) -> None:
    size = min(len(errors), len(actuals))
    y_true = np.array(actuals[:size], dtype=np.float64) + 1.0
    y_pred = y_true + np.array(errors[:size], dtype=np.float64)

    metrics = compute_metrics(y_true, y_pred)

    # With equal errors RMSE equals MAE mathematically; sqrt/mean round-off can put
    # RMSE one ulp below MAE, so compare with a relative float tolerance.
    assert metrics.rmse >= metrics.mae * (1.0 - 1e-12)
    assert metrics.mae >= 0.0
    assert metrics.wape >= 0.0
    assert metrics.n == size


@settings(max_examples=100, deadline=None)
@given(errors=_ERRORS)
def test_wape_is_scale_invariant(errors: list[float]) -> None:
    y_true = np.linspace(10.0, 50.0, len(errors))
    y_pred = y_true + np.array(errors, dtype=np.float64)
    for factor in (0.5, 2.0, 10.0):
        assert compute_metrics(y_true * factor, y_pred * factor).wape == pytest.approx(
            compute_metrics(y_true, y_pred).wape
        )
