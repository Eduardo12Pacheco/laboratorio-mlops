"""MLflow experiment tracking adapter (S6).

Every training run is tracked in one experiment; the model is logged through the
LightGBM flavor with an inferred signature and registered so the registry can adopt it.
The tracker owns only the MLflow API surface: protocol decisions (windows, gates,
fingerprints) stay in the application layer.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo

from mlops_lifecycle_lab.application.dto import JSONValue
from mlops_lifecycle_lab.application.ports import (
    LoggedModel,
    RunHandle,
    TrainedModel,
)


def _param_value(value: object) -> str | int | float | bool:
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class _MlflowRun:
    """Open MLflow run exposed through the application `RunHandle` port."""

    def __init__(self, active_run: mlflow.ActiveRun, model_name: str) -> None:
        self._active_run = active_run
        self._model_name = model_name
        self.run_id = str(active_run.info.run_id)

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log parameters, converting unsupported values to text."""
        mlflow.log_params({str(key): _param_value(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Log finite numeric metrics as floats."""
        mlflow.log_metrics({str(key): float(value) for key, value in metrics.items()})

    def log_json(self, name: str, obj: JSONValue) -> None:
        """Log a JSON document exactly as given (arrays stay arrays)."""
        mlflow.log_text(json.dumps(obj, indent=2, sort_keys=True), name)

    def log_model(self, trained: TrainedModel, input_example: pd.DataFrame) -> LoggedModel:
        """Log the fitted model, register a version and return its number."""
        info: ModelInfo = mlflow.lightgbm.log_model(
            trained.predictor,
            name="model",
            input_example=input_example,
            registered_model_name=self._model_name,
        )
        version = getattr(info, "registered_model_version", None)
        if version is None:
            versions = mlflow.MlflowClient().search_model_versions(
                f"name = '{self._model_name}' AND run_id = '{self.run_id}'"
            )
            if not versions:
                raise RuntimeError(f"MLflow did not register a model version for run {self.run_id}")
            version = versions[0].version
        uri = str(getattr(info, "model_uri", f"runs:/{self.run_id}/model"))
        return LoggedModel(uri=uri, registered_version=int(version))


class MlflowTracker:
    """Experiment tracker backed by a local MLflow tracking URI."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        artifact_root: Path,
        model_name: str,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._artifact_root = artifact_root
        self._model_name = model_name

    @contextmanager
    def run(self, name: str, tags: Mapping[str, str]) -> Iterator[RunHandle]:
        """Open a run; it is closed as FAILED when the body raises."""
        experiment_id = self._ensure_experiment()
        mlflow.set_tracking_uri(self._tracking_uri)
        active = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=name,
            tags={str(key): str(value) for key, value in tags.items()},
        )
        handle: RunHandle = _MlflowRun(active, self._model_name)
        try:
            yield handle
        except BaseException:
            mlflow.end_run(status="FAILED")
            raise
        else:
            mlflow.end_run(status="FINISHED")

    def load_json(self, name: str, run_id: str) -> JSONValue | None:
        """Read a JSON artifact of `run_id`; `None` when it is absent."""
        mlflow.set_tracking_uri(self._tracking_uri)
        try:
            text = mlflow.artifacts.load_text(f"runs:/{run_id}/{name}")
        except (MlflowException, OSError):
            return None
        value: JSONValue = json.loads(text)
        return value

    def ensure_ready(self) -> str:
        """Create the tracking database/experiment if needed and return the experiment id."""
        return self._ensure_experiment()

    def _ensure_experiment(self) -> str:
        mlflow.set_tracking_uri(self._tracking_uri)
        existing = mlflow.get_experiment_by_name(self._experiment_name)
        if existing is not None:
            return str(existing.experiment_id)
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        return str(
            mlflow.create_experiment(
                self._experiment_name,
                artifact_location=self._artifact_root.as_uri(),
            )
        )


__all__ = ["MlflowTracker"]
