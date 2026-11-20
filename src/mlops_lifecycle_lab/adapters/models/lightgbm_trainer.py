"""LightGBM trainer adapter with the exact S5 hyper-parameters.

The fitted `LGBMRegressor` is itself the `Predictor` exposed by the `TrainedModel`, which
also lets the MLflow tracker log the same object the training service evaluated (no
refit, no serialisation gap).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from mlops_lifecycle_lab.application.ports import TrainedModel
from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS, TARGET_COLUMN

LIGHTGBM_PARAMS: Mapping[str, object] = MappingProxyType(
    {
        "objective": "poisson",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
)
"""Fixed hyper-parameters of S5; no tuning happens anywhere in the lab."""


class LightGbmTrainer:
    """Fit a Poisson LightGBM regressor on the shared feature frame."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        self._params: Mapping[str, object] = dict(params or LIGHTGBM_PARAMS)

    @property
    def params(self) -> Mapping[str, object]:
        """The hyper-parameters that will be logged with every run."""
        return MappingProxyType(dict(self._params))

    def fit(self, frame: pd.DataFrame, seed: int) -> TrainedModel:
        """Fit on `frame` and return the fitted model plus its parameters.

        Raises:
            ValidationFailedError: when the feature columns or the target are missing.
        """
        required = (*FEATURE_COLUMNS, TARGET_COLUMN)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValidationFailedError(
                failed_rules=missing,
                message=f"training frame is missing columns: {', '.join(missing)}",
            )
        estimator_params: dict[str, Any] = {**self._params, "random_state": seed}
        model = LGBMRegressor(**estimator_params)
        model.fit(
            frame.loc[:, list(FEATURE_COLUMNS)],
            frame[TARGET_COLUMN].to_numpy(dtype=np.float64),
        )
        params = {
            **dict(self._params),
            "random_state": seed,
            "algorithm": "lightgbm",
        }
        return TrainedModel(predictor=model, params=params)


__all__ = ["LIGHTGBM_PARAMS", "LightGbmTrainer"]
