"""Deterministic raw frames and labeled-data repositories for application tests."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from mlops_lifecycle_lab.domain.schema import LABEL_COLUMNS, RAW_COLUMNS, TARGET_COLUMN


def empty_raw_frame() -> pd.DataFrame:
    """An empty frame with the raw contract columns (plus labels)."""
    columns = [*RAW_COLUMNS, *LABEL_COLUMNS]
    return pd.DataFrame({name: pd.Series(dtype="float64") for name in columns})


def raw_frame(
    start: datetime = datetime(2011, 1, 1),
    periods: int = 24,
    *,
    entity_id_start: int = 1,
    base: float = 100.0,
    hour_factor: float = 4.0,
) -> pd.DataFrame:
    """Build valid hourly raw rows with labels `cnt = casual + registered`."""
    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    hours = timestamps.hour.to_numpy()
    counts = np.asarray(base + hour_factor * hours, dtype="int64")
    casual = (counts // 4).astype("int64")
    data = {
        "entity_id": np.arange(entity_id_start, entity_id_start + periods, dtype="int64"),
        "timestamp": timestamps,
        "season": np.full(periods, 1, dtype="int64"),
        "holiday": np.zeros(periods, dtype="int64"),
        "workingday": np.asarray(timestamps.dayofweek < 5, dtype="int64"),
        "weathersit": np.full(periods, 1, dtype="int64"),
        "temp": np.linspace(0.2, 0.8, periods),
        "atemp": np.linspace(0.25, 0.85, periods),
        "hum": np.full(periods, 0.5),
        "windspeed": np.full(periods, 0.2),
        "casual": casual,
        "registered": counts - casual,
        "cnt": counts,
    }
    return pd.DataFrame(data, columns=[*RAW_COLUMNS, *LABEL_COLUMNS])


class StaticLabeledData:
    """Labeled-data repository over a fixed frame, filtered by timestamp."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.copy()
        self.requested_ends: list[datetime] = []

    def training_frame(self, end_exclusive: datetime) -> pd.DataFrame:
        """Return sorted rows strictly before `end_exclusive`."""
        self.requested_ends.append(end_exclusive)
        timestamps = pd.to_datetime(self.frame["timestamp"])
        selected = self.frame.loc[timestamps < end_exclusive]
        return selected.sort_values(["timestamp", "entity_id"]).reset_index(drop=True)


def target_of(frame: pd.DataFrame) -> np.ndarray:
    """Return the `cnt` column as a float array."""
    return np.asarray(frame[TARGET_COLUMN], dtype=np.float64)
