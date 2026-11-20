"""Integration tests for the MLflow tracker adapter (S6)."""

from __future__ import annotations

import json
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from mlflow import MlflowClient

from mlops_lifecycle_lab.adapters.tracking.mlflow_tracker import MlflowTracker
from mlops_lifecycle_lab.application.ports import TrainedModel
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from tests.integration.conftest import MlflowWorkspace


def training_frame(rows: int = 24) -> pd.DataFrame:
    """A tiny feature frame with a positive target for a Poisson model."""
    rng = np.random.default_rng(42)
    data = {name: rng.random(rows) for name in FEATURE_COLUMNS}
    return pd.DataFrame(data)


def trained_model(frame: pd.DataFrame) -> TrainedModel:
    """Fit a cheap deterministic model on the frame."""
    model = LGBMRegressor(n_estimators=3, n_jobs=1, deterministic=True, verbose=-1, random_state=42)
    model.fit(frame, np.linspace(1, 20, len(frame)))
    return TrainedModel(predictor=model, params={"algorithm": "lightgbm", "n_estimators": 3})


@pytest.fixture
def tracker(mlflow_workspace: MlflowWorkspace) -> tuple[MlflowTracker, str]:
    """A tracker with a unique registered model name for this test."""
    model_name = f"tracker-{uuid4().hex}"
    return (
        MlflowTracker(
            tracking_uri=mlflow_workspace.uri,
            experiment_name="bike-demand",
            artifact_root=mlflow_workspace.root / "artifacts",
            model_name=model_name,
        ),
        model_name,
    )


def test_run_logs_params_metrics_artifacts_and_a_model(
    tracker: tuple[MlflowTracker, str],
    mlflow_workspace: MlflowWorkspace,
) -> None:
    adapter, model_name = tracker
    frame = training_frame()

    with adapter.run("train-MANUAL", {"mll.trigger_reason": "MANUAL"}) as run:
        run.log_params({"algorithm": "lightgbm", "n_estimators": 3, "seed": 42})
        run.log_metrics({"val_rmse": 50.5, "val_wape": 0.2})
        run.log_json("validation_report.json", {"verdict": "WARN", "rows": len(frame)})
        run.log_json("feature_columns.json", list(FEATURE_COLUMNS))
        logged = run.log_model(trained_model(frame), input_example=frame)
        run_id = run.run_id

    assert logged.registered_version >= 1
    assert logged.uri.startswith("runs:/")
    client = MlflowClient(tracking_uri=mlflow_workspace.uri)
    stored = client.get_run(run_id)
    assert stored.data.params["algorithm"] == "lightgbm"
    assert stored.data.params["seed"] == "42"
    assert stored.data.metrics["val_rmse"] == pytest.approx(50.5)
    assert stored.data.metrics["val_wape"] == pytest.approx(0.2)
    assert stored.data.tags["mll.trigger_reason"] == "MANUAL"
    assert stored.info.status == "FINISHED"

    artifacts = {artifact.path for artifact in client.list_artifacts(run_id)}
    assert {"validation_report.json", "feature_columns.json", "model"} <= artifacts
    report = mlflow.artifacts.load_dict(f"runs:/{run_id}/validation_report.json")
    assert report == {"verdict": "WARN", "rows": len(frame)}
    columns = json.loads(mlflow.artifacts.load_text(f"runs:/{run_id}/feature_columns.json"))
    assert columns == list(FEATURE_COLUMNS)

    version = client.search_model_versions(f"name = '{model_name}'")[0]
    assert str(version.run_id) == run_id


def test_registered_model_loads_as_a_predictor(
    tracker: tuple[MlflowTracker, str],
    mlflow_workspace: MlflowWorkspace,
) -> None:
    adapter, model_name = tracker
    frame = training_frame()
    with adapter.run("train-MANUAL", {}) as run:
        logged = run.log_model(trained_model(frame), input_example=frame)

    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/{logged.registered_version}")
    predictions = np.asarray(loaded.predict(frame), dtype=np.float64)

    assert predictions.shape == (len(frame),)
    assert np.isfinite(predictions).all()


def test_a_failing_run_is_closed_as_failed(
    tracker: tuple[MlflowTracker, str],
    mlflow_workspace: MlflowWorkspace,
) -> None:
    adapter, _ = tracker
    with pytest.raises(RuntimeError):
        with adapter.run("train-MANUAL", {}) as run:
            run_id = run.run_id
            raise RuntimeError("boom")

    assert MlflowClient(tracking_uri=mlflow_workspace.uri).get_run(run_id).info.status == "FAILED"


def test_log_json_keeps_a_list_as_a_json_array(
    tracker: tuple[MlflowTracker, str],
    mlflow_workspace: MlflowWorkspace,
) -> None:
    adapter, _ = tracker
    with adapter.run("train-MANUAL", {}) as run:
        run.log_json("feature_columns.json", ["season", "mnth"])
        run_id = run.run_id

    text = mlflow.artifacts.load_text(f"runs:/{run_id}/feature_columns.json")
    assert json.loads(text) == ["season", "mnth"]
