"""Small shared helpers for the training and evaluation use cases."""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mlops_lifecycle_lab.domain.errors import NotEnoughLabelsError
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.schema import TARGET_COLUMN
from mlops_lifecycle_lab.domain.windows import TimeWindow


def window_rows(frame: pd.DataFrame, window: TimeWindow) -> pd.DataFrame:
    """Return the rows whose `timestamp` falls inside the half-open `window`."""
    timestamps = pd.to_datetime(frame["timestamp"])
    selected = frame.loc[(timestamps >= window.start) & (timestamps < window.end)]
    return selected.reset_index(drop=True)


def require_rows(frame: pd.DataFrame, what: str) -> None:
    """Raise when a window has no labeled rows to work with."""
    if frame.empty:
        raise NotEnoughLabelsError(f"{what} has no labeled rows")


def sort_for_fingerprint(frame: pd.DataFrame) -> pd.DataFrame:
    """Order rows deterministically before hashing."""
    columns = [name for name in ("timestamp", "entity_id") if name in frame.columns]
    return frame.sort_values(columns, kind="stable").reset_index(drop=True)


def data_fingerprint(frame: pd.DataFrame) -> str:
    """SHA-256 of the sorted labeled frame, stable across platforms for equal values."""
    ordered = sort_for_fingerprint(frame)
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the twelve features plus the `cnt` target for a trainer."""
    features = build_features(raw)
    result = features.copy()
    result[TARGET_COLUMN] = raw[TARGET_COLUMN].to_numpy(dtype=np.float64, copy=True)
    return result


def clip_predictions(predictions: NDArray[np.float64] | object) -> NDArray[np.float64]:
    """Clip model outputs at zero to match the serving contract."""
    return np.maximum(np.asarray(predictions, dtype=np.float64), 0.0)


def window_tag(window: TimeWindow | None) -> str:
    """Canonical `start/end` representation used by the S6 run tags."""
    if window is None:
        return "none"
    return f"{_iso(window.start)}/{_iso(window.end)}"


def _iso(moment: datetime) -> str:
    return moment.isoformat()
