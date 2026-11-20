"""The one feature pipeline shared by training, batch scoring and online serving.

`build_features` is pure: it derives calendar columns from the observation timestamp,
keeps `t_days` as the trend signal and returns exactly the twelve `FEATURE_COLUMNS` in
the order the trainer expects. Leakage columns (`casual`, `registered`, `cnt`, `yr`,
`instant`) are never read, so there is a single definition of "the model inputs".
"""

from __future__ import annotations

import pandas as pd

from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS, TREND_EPOCH

_INPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "season",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the twelve model features from a raw frame.

    Args:
        raw: raw rows containing at least `timestamp` and the nine non-derived raw
            columns. Extra columns (labels, ids) are ignored, never modified.

    Returns:
        A new frame whose columns are exactly `FEATURE_COLUMNS`, preserving the input
        index so callers can align it with labels.

    Raises:
        ValidationFailedError: if a required input column is missing or the timestamp
            cannot be parsed.
    """
    missing = [column for column in _INPUT_COLUMNS if column not in raw.columns]
    if missing:
        raise ValidationFailedError(
            failed_rules=missing,
            message=f"build_features missing required columns: {', '.join(missing)}",
        )

    timestamps = pd.to_datetime(raw["timestamp"])
    if timestamps.isna().any():
        raise ValidationFailedError(message="build_features requires non-null timestamps")

    features = pd.DataFrame(
        {
            "season": raw["season"].astype("int64"),
            "mnth": timestamps.dt.month.astype("int64"),
            "hr": timestamps.dt.hour.astype("int64"),
            "holiday": raw["holiday"].astype("int64"),
            "weekday": ((timestamps.dt.dayofweek + 1) % 7).astype("int64"),
            "workingday": raw["workingday"].astype("int64"),
            "weathersit": raw["weathersit"].astype("int64"),
            "temp": raw["temp"].astype("float64"),
            "atemp": raw["atemp"].astype("float64"),
            "hum": raw["hum"].astype("float64"),
            "windspeed": raw["windspeed"].astype("float64"),
            "t_days": (timestamps - pd.Timestamp(TREND_EPOCH)) / pd.Timedelta(days=1),
        },
        index=raw.index,
    )
    return features.loc[:, list(FEATURE_COLUMNS)]
