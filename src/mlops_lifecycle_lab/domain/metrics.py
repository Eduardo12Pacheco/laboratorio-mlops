"""Regression metrics used by training, monitoring and promotion gates.

The definitions are deliberately boring and explicit so every report in the project can
be traced back to one implementation: RMSE and MAE in demand units, WAPE scale-free and
bias signed as `mean(prediction - actual)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Point metrics for one prediction vector."""

    rmse: float
    mae: float
    wape: float
    bias: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation used by reports and MLflow."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "wape": self.wape,
            "bias": self.bias,
            "n": self.n,
        }


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> RegressionMetrics:
    """Compute RMSE, MAE, WAPE and bias for paired prediction vectors.

    Args:
        y_true: actual target values, one-dimensional.
        y_pred: predictions aligned with `y_true`.

    Returns:
        `RegressionMetrics` with `wape = sum(|pred - actual|) / sum(actual)`.

    Raises:
        ValueError: on empty, non-finite or differently sized inputs, or when the
            actual total is zero (WAPE is undefined).
    """
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("metrics require one-dimensional arrays")
    if actual.size != predicted.size:
        raise ValueError(
            f"length mismatch: y_true has {actual.size} values, y_pred has {predicted.size}"
        )
    if actual.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("metrics require finite values")

    total = float(actual.sum())
    if total == 0.0:
        raise ValueError("WAPE is undefined when the actual total is zero")

    errors = predicted - actual
    return RegressionMetrics(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        wape=float(np.abs(errors).sum() / total),
        bias=float(errors.mean()),
        n=int(actual.size),
    )
