"""Unit tests for the LightGBM trainer adapter (S5)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.adapters.models.lightgbm_trainer import LightGbmTrainer
from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS, TARGET_COLUMN


def synthetic_frame(rows: int = 48) -> pd.DataFrame:
    """Build a tiny deterministic frame with every feature and the target."""
    rng = np.random.default_rng(42)
    data: dict[str, object] = {}
    for index, column in enumerate(FEATURE_COLUMNS):
        data[column] = rng.integers(0, 5, size=rows).astype("float64") + index * 0.1
    data[TARGET_COLUMN] = rng.integers(1, 50, size=rows).astype("float64")
    return pd.DataFrame(data)


def test_fit_returns_the_s5_parameters_and_seed() -> None:
    trained = LightGbmTrainer().fit(synthetic_frame(), seed=7)

    assert trained.params["algorithm"] == "lightgbm"
    assert trained.params["objective"] == "poisson"
    assert trained.params["n_estimators"] == 400
    assert trained.params["learning_rate"] == 0.05
    assert trained.params["num_leaves"] == 31
    assert trained.params["min_child_samples"] == 20
    assert trained.params["subsample"] == 0.8
    assert trained.params["subsample_freq"] == 1
    assert trained.params["colsample_bytree"] == 0.8
    assert trained.params["random_state"] == 7
    assert trained.params["n_jobs"] == 1
    assert trained.params["deterministic"] is True
    assert trained.params["force_row_wise"] is True
    assert trained.params["verbose"] == -1


def test_fit_is_deterministic() -> None:
    frame = synthetic_frame()
    features = frame.loc[:, list(FEATURE_COLUMNS)]

    first = LightGbmTrainer().fit(frame, seed=42)
    second = LightGbmTrainer().fit(frame, seed=42)

    assert np.array_equal(first.predictor.predict(features), second.predictor.predict(features))


def test_predictions_are_non_negative_floats() -> None:
    frame = synthetic_frame()
    trained = LightGbmTrainer().fit(frame, seed=42)

    predictions = trained.predictor.predict(frame.loc[:, list(FEATURE_COLUMNS)])

    assert predictions.shape == (len(frame),)
    assert predictions.dtype == np.float64
    assert (predictions >= 0).all()


def test_fit_refuses_a_frame_without_the_contract_columns() -> None:
    frame = synthetic_frame().drop(columns=["hum", TARGET_COLUMN])

    with pytest.raises(ValidationFailedError) as excinfo:
        LightGbmTrainer().fit(frame, seed=42)

    assert "hum" in excinfo.value.failed_rules
    assert TARGET_COLUMN in excinfo.value.failed_rules


def test_the_clock_adapters_expose_naive_time() -> None:
    from mlops_lifecycle_lab.adapters.clock import SimulatedClock, SystemClock

    system = SystemClock()
    assert system.now().tzinfo is None
    assert abs(system.now() - datetime.now()) < timedelta(seconds=5)

    simulated = SimulatedClock(datetime(2012, 1, 1))
    assert simulated.now() == datetime(2012, 1, 1)
    simulated.set(datetime(2012, 2, 1))
    assert simulated.now() == datetime(2012, 2, 1)
    simulated.advance(days=7)
    assert simulated.now() == datetime(2012, 2, 8)
