"""Fault injection used by the drill scenarios (S16).

The faults must be explicit, side-effect free on their input and impossible to trigger by
accident: `celsius_scale` rewrites a batch, `PoisonedTrainer` only corrupts labels while it
is armed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.domain.schema import TARGET_COLUMN
from mlops_lifecycle_lab.entrypoints.drill.faults import PoisonedTrainer, celsius_scale
from tests.fakes.data import raw_frame
from tests.fakes.trainer import FakeTrainer


def test_celsius_scale_rewrites_temperature_columns_and_keeps_the_input() -> None:
    frame = raw_frame(periods=4)
    original = frame.copy()

    corrupt = celsius_scale(frame)

    assert frame.equals(original)
    assert corrupt is not frame
    np.testing.assert_allclose(corrupt["temp"], original["temp"] * 47.0 - 8.0)
    np.testing.assert_allclose(corrupt["atemp"], original["atemp"] * 66.0 - 16.0)
    # The corruption is large enough for the numeric-range rule to reject the batch.
    assert corrupt["temp"].max() > 1.0
    assert corrupt["atemp"].max() > 1.0


def test_poisoned_trainer_scales_labels_only_while_armed() -> None:
    frame = raw_frame(periods=6)
    inner = FakeTrainer()
    poisoned = PoisonedTrainer(inner, label_scale=0.3)

    poisoned.fit(frame, seed=42)
    pd.testing.assert_series_equal(inner.frames[-1][TARGET_COLUMN], frame[TARGET_COLUMN])

    poisoned.arm()
    assert poisoned.active is True
    poisoned.fit(frame, seed=42)
    scaled = inner.frames[-1][TARGET_COLUMN]
    np.testing.assert_allclose(scaled, np.rint(frame[TARGET_COLUMN] * 0.3))
    assert scaled.dtype == np.int64

    poisoned.disarm()
    assert poisoned.active is False
    poisoned.fit(frame, seed=42)
    pd.testing.assert_series_equal(inner.frames[-1][TARGET_COLUMN], frame[TARGET_COLUMN])


def test_poisoned_trainer_requires_a_positive_scale() -> None:
    with pytest.raises(ValueError, match="label_scale"):
        PoisonedTrainer(FakeTrainer(), label_scale=0.0)
