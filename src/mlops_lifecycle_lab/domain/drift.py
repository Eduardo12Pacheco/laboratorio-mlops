"""Population Stability Index drift statistics.

The project decides with these native statistics (D12): they are pure, deterministic and
cheap enough to run every tick. Evidently renders the human-readable report separately.
Numeric PSI uses reference-quantile bins with infinite outer edges; proportions are
clipped at 1e-4 so an empty bin cannot produce an infinite score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.schema import MONITORED_CATEGORICAL, MONITORED_NUMERIC

PSI_MODERATE: float = 0.1
"""PSI above this value is a moderate change."""

PSI_DRIFTED: float = 0.2
"""PSI above this value means the feature drifted."""

_EPSILON = 1e-4

DriftKind = Literal["numeric", "categorical"]


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """Drift evidence for a single feature."""

    feature: str
    kind: DriftKind
    psi: float
    drifted: bool


@dataclass(frozen=True, slots=True)
class DriftSummary:
    """Every monitored feature plus the aggregate drift indicators."""

    features: tuple[FeatureDrift, ...]
    numeric_drift_share: float
    prediction_psi: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation persisted in snapshots and reports."""
        return {
            "features": {
                feature.feature: {
                    "kind": feature.kind,
                    "psi": feature.psi,
                    "drifted": feature.drifted,
                }
                for feature in self.features
            },
            "numeric_drift_share": self.numeric_drift_share,
            "prediction_psi": self.prediction_psi,
        }


def _numeric(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _categorical(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = int(counts.sum())
    if total == 0:
        return np.full(counts.shape, _EPSILON, dtype=np.float64)
    clipped: np.ndarray = np.clip(counts / total, _EPSILON, None)
    return clipped


def psi_numeric(reference: ArrayLike, current: ArrayLike, bins: int = 10) -> float:
    """PSI between two numeric samples using reference-quantile bins.

    Args:
        reference: the champion's reference sample (training window).
        current: the current sample (monitoring window).
        bins: number of quantile bins; must be at least 2.

    Returns:
        The PSI, non-negative and exactly zero when both samples have the same
        distribution over the reference bins.
    """
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}")
    ref = _numeric(reference, "reference")
    cur = _numeric(current, "current")
    quantiles = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if quantiles.size == 1:
        # A degenerate reference cannot be split into quantile bins; keep the value as
        # the boundary so the statistic stays finite instead of crashing a tick.
        edges = np.asarray([-np.inf, quantiles[0], np.inf])
    else:
        edges = quantiles.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
    expected = _proportions(ref, edges)
    actual = _proportions(cur, edges)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_categorical(reference: ArrayLike, current: ArrayLike) -> float:
    """PSI between two categorical samples, over the union of observed levels."""
    ref = _categorical(reference, "reference")
    cur = _categorical(current, "current")
    levels = np.unique(np.concatenate([ref, cur]))

    def proportions(values: np.ndarray) -> np.ndarray:
        shares = np.asarray([np.count_nonzero(values == level) for level in levels], dtype=float)
        return np.clip(shares / values.size, _EPSILON, None)

    expected = proportions(ref)
    actual = proportions(cur)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def summarize_drift(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    reference_pred: ArrayLike,
    current_pred: ArrayLike,
) -> DriftSummary:
    """Compute every monitored feature drift plus the two aggregate indicators.

    Args:
        reference_features: raw-frame columns from the reference window.
        current_features: raw-frame columns from the current window.
        reference_pred: predictions on the reference window.
        current_pred: predictions on the current window.

    Raises:
        ValidationFailedError: if a monitored column is missing from either frame.
    """
    required = (*MONITORED_NUMERIC, *MONITORED_CATEGORICAL)
    missing = [
        column
        for column in required
        if column not in reference_features.columns or column not in current_features.columns
    ]
    if missing:
        raise ValidationFailedError(
            failed_rules=missing,
            message=f"missing monitored columns: {', '.join(missing)}",
        )

    features: list[FeatureDrift] = []
    for column in MONITORED_NUMERIC:
        value = psi_numeric(reference_features[column], current_features[column])
        features.append(
            FeatureDrift(feature=column, kind="numeric", psi=value, drifted=value > PSI_DRIFTED)
        )
    for column in MONITORED_CATEGORICAL:
        value = psi_categorical(reference_features[column], current_features[column])
        features.append(
            FeatureDrift(feature=column, kind="categorical", psi=value, drifted=value > PSI_DRIFTED)
        )

    numeric = [feature for feature in features if feature.kind == "numeric"]
    return DriftSummary(
        features=tuple(features),
        numeric_drift_share=sum(feature.drifted for feature in numeric) / len(numeric),
        prediction_psi=psi_numeric(reference_pred, current_pred),
    )
