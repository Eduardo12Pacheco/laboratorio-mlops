"""Naive profile baseline used as the floor of the promotion gates.

The baseline is deliberately weak (S12/D6): the mean `cnt` per (`workingday`, `hr`)
computed on the same rows the challenger trains on. It proves the model learned something
without pretending to be a competitor. Unseen keys fall back to the global training mean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd

from mlops_lifecycle_lab.domain.errors import ValidationFailedError


def _hour_series(frame: pd.DataFrame) -> pd.Series:
    """Return the hour of day from `hr` when present, else from `timestamp`."""
    if "hr" in frame.columns:
        return pd.to_numeric(frame["hr"], errors="coerce")
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"]).dt.hour
    raise ValidationFailedError(
        failed_rules=["hr"],
        message="baseline needs an `hr` column or a `timestamp` column",
    )


def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationFailedError(
            failed_rules=missing,
            message=f"missing required columns: {', '.join(missing)}",
        )


@dataclass(frozen=True, slots=True)
class ProfileBaseline:
    """Mean demand per (workingday, hour) with a global-mean fallback."""

    profile: Mapping[tuple[int, int], float]
    global_mean: float

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> ProfileBaseline:
        """Fit the profile on a raw or feature frame that contains labels."""
        _require(frame, ("workingday", "cnt"))
        hours = _hour_series(frame)
        grouped = frame.assign(_hr=hours).groupby(["workingday", "_hr"], sort=True)["cnt"].mean()
        profile: dict[tuple[int, int], float] = {}
        keys = cast(list[tuple[int, int]], list(grouped.index))
        for (workingday, hour), value in zip(keys, grouped.to_numpy(), strict=True):
            profile[(int(workingday), int(hour))] = float(value)
        return cls(
            profile=MappingProxyType(profile),
            global_mean=float(frame["cnt"].mean()),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict one value per row, using the global mean for unseen keys."""
        _require(frame, ("workingday",))
        hours = _hour_series(frame)
        workingdays = pd.to_numeric(frame["workingday"], errors="coerce")
        return np.asarray(
            [
                self.profile.get((int(workingday), int(hour)), self.global_mean)
                for workingday, hour in zip(workingdays, hours, strict=False)
            ],
            dtype=np.float64,
        )
