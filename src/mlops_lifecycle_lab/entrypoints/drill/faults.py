"""Fault injection for the drill scenarios (S16).

Every fault is explicit, pure and clearly labelled: the drill is a fault-injection harness,
not a simulation of drift. `celsius_scale` corrupts a batch (temperature columns sent on a
Celsius scale instead of the dataset's normalised scale) and `PoisonedTrainer` corrupts the
training labels while armed. Neither fault is ever reachable from production code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_lifecycle_lab.application.ports import ModelTrainer, TrainedModel
from mlops_lifecycle_lab.domain.schema import TARGET_COLUMN

_TEMP_SCALE = 47.0
_TEMP_OFFSET = -8.0
_ATEMP_SCALE = 66.0
_ATEMP_OFFSET = -16.0


def celsius_scale(batch: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `batch` with `temp`/`atemp` in Celsius-scale values.

    The bundled dataset normalises both columns to `[0, 1]`; this fault maps them back to
    plausible Celsius/feels-like values, which the `numeric_range` batch rule rejects. The
    input frame is never mutated.
    """
    corrupted = batch.copy()
    corrupted["temp"] = corrupted["temp"].astype("float64") * _TEMP_SCALE + _TEMP_OFFSET
    corrupted["atemp"] = corrupted["atemp"].astype("float64") * _ATEMP_SCALE + _ATEMP_OFFSET
    return corrupted


class PoisonedTrainer:
    """Decorator that multiplies labels by `label_scale` while armed (fault injection).

    This is deliberately *not* drift: it injects a bad challenger so the `rollback`
    scenario can prove that probation notices a poisoned model and restores the previous
    champion. The wrapped trainer only sees the poisoned frame while `arm()` was called;
    the drill disarms it before the next tick.
    """

    def __init__(self, inner: ModelTrainer, label_scale: float = 0.3) -> None:
        if not 0.0 < label_scale < 1.0:
            raise ValueError(f"label_scale must be in (0, 1), got {label_scale}")
        self._inner = inner
        self._label_scale = label_scale
        self._active = False

    @property
    def active(self) -> bool:
        """Whether the next `fit` call corrupts its labels."""
        return self._active

    def arm(self) -> None:
        """Enable label corruption for the next `fit` call."""
        self._active = True

    def disarm(self) -> None:
        """Disable label corruption (the normal drill state)."""
        self._active = False

    def fit(self, frame: pd.DataFrame, seed: int) -> TrainedModel:
        """Fit the inner trainer on corrupted labels while armed."""
        if not self._active:
            return self._inner.fit(frame, seed)
        poisoned = frame.copy()
        scaled = np.asarray(poisoned[TARGET_COLUMN], dtype=np.float64) * self._label_scale
        poisoned[TARGET_COLUMN] = np.rint(scaled).astype("int64")
        return self._inner.fit(poisoned, seed)


__all__ = ["PoisonedTrainer", "celsius_scale"]
