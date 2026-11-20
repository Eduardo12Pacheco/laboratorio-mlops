"""MLflow model registry adapter using tags and aliases, never stages (S7, D8).

The lifecycle stage is the `mll.stage` tag; the three alias pointers are `champion`,
`previous-champion` and `challenger`. Every transition is validated by the domain state
machine before any write, so an illegal request leaves the registry untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion
from mlflow.exceptions import MlflowException
from numpy.typing import NDArray

from mlops_lifecycle_lab.application.dto import (
    TAG_FAILED_GATES,
    TAG_REJECT_REASON,
    TAG_STAGE,
    decode_version_tags,
)
from mlops_lifecycle_lab.application.ports import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    ALIAS_PREVIOUS_CHAMPION,
    Predictor,
)
from mlops_lifecycle_lab.domain.errors import (
    NoPreviousChampionError,
    UnknownModelVersionError,
)
from mlops_lifecycle_lab.domain.lifecycle import ModelStage, ModelVersion, transition


def _is_not_found(exc: MlflowException) -> bool:
    return exc.error_code == "RESOURCE_DOES_NOT_EXIST" or "not found" in str(exc).lower()


class _PyfuncPredictor:
    """Adapt the MLflow `PyFuncModel` to the `Predictor` port."""

    def __init__(self, model: object) -> None:
        self._model = model

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Predict one value per feature row, returning a float64 array."""
        predictions = self._model.predict(features)  # type: ignore[attr-defined]
        return np.asarray(predictions, dtype=np.float64)


class MlflowRegistry:
    """Tags-and-aliases registry on top of the MLflow model registry."""

    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        client: MlflowClient | None = None,
    ) -> None:
        self._name = model_name
        self._client = client or MlflowClient(tracking_uri=tracking_uri)

    def adopt_candidate(self, version: int, tags: Mapping[str, str]) -> ModelVersion:
        """Tag a freshly registered version as CANDIDATE and point `challenger` at it."""
        raw = self._raw(version)
        if raw is None:
            raise UnknownModelVersionError(f"model version {version} is not registered")
        if TAG_STAGE in raw.tags:
            raise ValueError(f"model version {version} is already registered")
        candidate = decode_version_tags(self._name, version, tags)
        if candidate.stage is not ModelStage.CANDIDATE:
            raise ValueError(f"new versions must be CANDIDATE, got {candidate.stage.value}")
        self._write_tags(version, tags)
        self._client.set_registered_model_alias(self._name, ALIAS_CHALLENGER, str(version))
        return self._require(version)

    def get_champion(self) -> ModelVersion | None:
        """Return the version aliased `champion`."""
        return self.get_alias(ALIAS_CHAMPION)

    def get_version(self, version: int) -> ModelVersion | None:
        """Return a lifecycle version, or `None` when absent or not adopted yet."""
        raw = self._raw(version)
        if raw is None or TAG_STAGE not in raw.tags:
            return None
        return self._decode(raw)

    def list_versions(self) -> list[ModelVersion]:
        """Every adopted version, ordered by version number."""
        raw_versions = self._client.search_model_versions(f"name = '{self._escape(self._name)}'")
        adopted = [raw for raw in raw_versions if TAG_STAGE in raw.tags]
        return [self._decode(raw) for raw in sorted(adopted, key=lambda item: int(item.version))]

    def get_alias(self, alias: str) -> ModelVersion | None:
        """Return the version an alias points at, or `None` when absent."""
        try:
            raw = self._client.get_model_version_by_alias(self._name, alias)
        except MlflowException as exc:
            if _is_not_found(exc):
                return None
            raise
        if TAG_STAGE not in raw.tags:
            return None
        return self._decode(raw)

    def promote(self, version: int) -> ModelVersion:
        """Archive the incumbent, move both aliases and mark the candidate champion."""
        candidate = self._require(version)
        current = self.get_champion()
        if current is not None and current.version == version:
            transition(candidate, ModelStage.CHAMPION)  # raises IllegalTransitionError
        promoted = transition(candidate, ModelStage.CHAMPION)
        archived = None if current is None else transition(current, ModelStage.ARCHIVED)
        if archived is not None:
            self._write_stage(archived)
            self._client.set_registered_model_alias(
                self._name, ALIAS_PREVIOUS_CHAMPION, str(archived.version)
            )
        self._client.set_registered_model_alias(self._name, ALIAS_CHAMPION, str(version))
        self._write_stage(promoted)
        self._client.delete_registered_model_alias(self._name, ALIAS_CHALLENGER)
        return self._require(version)

    def reject(self, version: int, reason: str, failed_gates: Sequence[str]) -> ModelVersion:
        """Reject a candidate and record why."""
        model = self._require(version)
        rejected = transition(model, ModelStage.REJECTED)
        self._write_stage(rejected)
        self._write_tags(
            version,
            {
                TAG_REJECT_REASON: reason,
                TAG_FAILED_GATES: ",".join(failed_gates),
            },
        )
        current_challenger = self.get_alias(ALIAS_CHALLENGER)
        if current_challenger is not None and current_challenger.version == version:
            self._client.delete_registered_model_alias(self._name, ALIAS_CHALLENGER)
        return self._require(version)

    def rollback(self, reason: str) -> ModelVersion:
        """Restore `previous-champion` and mark the bad champion `ROLLED_BACK`."""
        previous = self.get_alias(ALIAS_PREVIOUS_CHAMPION)
        if previous is None:
            raise NoPreviousChampionError("no previous champion alias to roll back to")
        current = self.get_champion()
        if current is None:
            raise NoPreviousChampionError("no current champion to roll back")
        rolled_back = transition(current, ModelStage.ROLLED_BACK)
        restored = transition(previous, ModelStage.CHAMPION)
        self._write_stage(rolled_back)
        self._write_tags(current.version, {TAG_REJECT_REASON: reason})
        self._write_stage(restored)
        self._client.set_registered_model_alias(self._name, ALIAS_CHAMPION, str(previous.version))
        self._client.delete_registered_model_alias(self._name, ALIAS_PREVIOUS_CHAMPION)
        current_challenger = self.get_alias(ALIAS_CHALLENGER)
        if current_challenger is not None and current_challenger.version == current.version:
            self._client.delete_registered_model_alias(self._name, ALIAS_CHALLENGER)
        return self._require(previous.version)

    def load_predictor(self, version: int) -> Predictor:
        """Load the artifact behind `version` through the flavor-agnostic pyfunc API."""
        if self.get_version(version) is None:
            raise UnknownModelVersionError(f"model version {version} is not registered")
        model = mlflow.pyfunc.load_model(f"models:/{self._name}/{version}")
        return _PyfuncPredictor(model)

    def _raw(self, version: int) -> MlflowModelVersion | None:
        try:
            return self._client.get_model_version(self._name, str(version))
        except MlflowException as exc:
            if _is_not_found(exc):
                return None
            raise

    def _require(self, version: int) -> ModelVersion:
        model = self.get_version(version)
        if model is None:
            raise UnknownModelVersionError(f"model version {version} is not registered")
        return model

    def _decode(self, raw: MlflowModelVersion) -> ModelVersion:
        return decode_version_tags(self._name, int(raw.version), dict(raw.tags))

    def _write_tags(self, version: int, tags: Mapping[str, str]) -> None:
        for key, value in tags.items():
            self._client.set_model_version_tag(self._name, str(version), str(key), str(value))

    def _write_stage(self, model: ModelVersion) -> None:
        self._client.set_model_version_tag(
            self._name, str(model.version), TAG_STAGE, model.stage.value
        )

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")


__all__ = ["MlflowRegistry"]
