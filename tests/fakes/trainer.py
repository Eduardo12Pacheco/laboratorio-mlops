"""Deterministic trainers and predictors for application unit tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mlops_lifecycle_lab.application.ports import TrainedModel


@dataclass(frozen=True, slots=True)
class FunctionPredictor:
    """Predictor backed by an arbitrary deterministic function of the feature frame."""

    fn: Callable[[pd.DataFrame], NDArray[np.float64]]
    calls: list[pd.DataFrame] = field(default_factory=list, compare=False)

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Apply the function to the features (the frame is never mutated)."""
        self.calls.append(features.copy())
        return np.asarray(self.fn(features), dtype=np.float64)


class MeanPredictor:
    """Predict the training mean for every row."""

    def __init__(self, mean: float) -> None:
        self.mean = mean

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Return one constant prediction per row."""
        return np.full(len(features), self.mean, dtype=np.float64)


class FakeTrainer:
    """Trainer that learns the target mean unless a predictor factory is supplied."""

    def __init__(
        self,
        *,
        predictor_factory: Callable[[pd.DataFrame], Callable[[pd.DataFrame], NDArray[np.float64]]]
        | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.predictor_factory = predictor_factory
        self.params = params or {"algorithm": "fake", "n_estimators": 10}
        self.frames: list[pd.DataFrame] = []
        self.seeds: list[int] = []

    def fit(self, frame: pd.DataFrame, seed: int) -> TrainedModel:
        """Return a deterministic predictor and the recorded hyper-parameters."""
        self.frames.append(frame.copy())
        self.seeds.append(seed)
        if self.predictor_factory is not None:
            predictor = FunctionPredictor(self.predictor_factory(frame))
        else:
            predictor = MeanPredictor(float(frame["cnt"].mean()))
        return TrainedModel(
            predictor=predictor,
            params={**self.params, "random_state": seed},
        )
