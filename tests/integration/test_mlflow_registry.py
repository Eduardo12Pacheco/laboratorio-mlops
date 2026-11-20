"""Integration tests for the MLflow alias/tag registry adapter (S7)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from mlflow import MlflowClient

from mlops_lifecycle_lab.adapters.registry.mlflow_registry import MlflowRegistry
from mlops_lifecycle_lab.adapters.tracking.mlflow_tracker import MlflowTracker
from mlops_lifecycle_lab.application.dto import encode_version_tags
from mlops_lifecycle_lab.application.ports import TrainedModel
from mlops_lifecycle_lab.domain.errors import IllegalTransitionError
from mlops_lifecycle_lab.domain.lifecycle import ModelStage, ModelVersion
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from mlops_lifecycle_lab.domain.windows import TimeWindow
from tests.fakes.mlflow_workspace import MlflowWorkspace


def candidate(version: int) -> ModelVersion:
    """Build the lineage that `TrainCandidate` hands to the registry."""
    return ModelVersion(
        name="bike-demand",
        version=version,
        stage=ModelStage.CANDIDATE,
        run_id=f"run-{version}",
        metrics={"rmse": 50.0 + version},
        train_window=TimeWindow(datetime(2011, 1, 1), datetime(2011, 10, 1)),
        holdout_window=TimeWindow(datetime(2011, 10, 1), datetime(2012, 1, 1)),
        data_fingerprint=f"fingerprint-{version}",
        feature_spec_version="1",
        created_at=datetime(2012, 1, 1),
        trigger_reason="MANUAL",
        tags={"mll.git_sha": "deadbeef"},
    )


@pytest.fixture
def workspace(mlflow_workspace: MlflowWorkspace) -> tuple[MlflowTracker, MlflowRegistry, str]:
    """A tracker/registry pair with a model name unique to this test."""
    name = f"registry-{uuid4().hex}"
    tracker = MlflowTracker(
        tracking_uri=mlflow_workspace.uri,
        experiment_name="bike-demand",
        artifact_root=mlflow_workspace.root / "artifacts",
        model_name=name,
    )
    registry = MlflowRegistry(tracking_uri=mlflow_workspace.uri, model_name=name)
    return tracker, registry, name


def register(tracker: MlflowTracker, frame: pd.DataFrame) -> int:
    """Log one tiny model and return its MLflow version."""
    model = LGBMRegressor(n_estimators=2, n_jobs=1, verbose=-1).fit(
        frame, np.linspace(1, 10, len(frame))
    )
    with tracker.run("train-MANUAL", {}) as run:
        logged = run.log_model(TrainedModel(model, {"algorithm": "lightgbm"}), frame)
        return logged.registered_version


def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({name: rng.random(12) for name in FEATURE_COLUMNS})


def test_stage_lives_in_a_tag_and_aliases_move_atomically(
    workspace: tuple[MlflowTracker, MlflowRegistry, str],
    mlflow_workspace: MlflowWorkspace,
) -> None:
    tracker, registry, name = workspace
    same_frame = frame()
    first = register(tracker, same_frame)
    adopted = registry.adopt_candidate(first, encode_version_tags(candidate(first)))
    registry.promote(first)
    second = register(tracker, same_frame)
    registry.adopt_candidate(second, encode_version_tags(candidate(second)))

    promoted = registry.promote(second)

    client = MlflowClient(tracking_uri=mlflow_workspace.uri)
    first_raw = client.get_model_version(name, str(first))
    second_raw = client.get_model_version(name, str(second))
    assert first_raw.tags["mll.stage"] == ModelStage.ARCHIVED.value
    assert second_raw.tags["mll.stage"] == ModelStage.CHAMPION.value
    assert promoted.stage is ModelStage.CHAMPION
    assert client.get_model_version_by_alias(name, "previous-champion").version == first
    assert client.get_model_version_by_alias(name, "champion").version == second
    assert adopted.stage is ModelStage.CANDIDATE

    restored = registry.rollback("poisoned")

    assert restored.version == first
    assert client.get_model_version(name, str(second)).tags["mll.stage"] == "ROLLED_BACK"
    assert client.get_model_version_by_alias(name, "champion").version == first


def test_an_illegal_transition_never_writes_a_tag(
    workspace: tuple[MlflowTracker, MlflowRegistry, str],
    mlflow_workspace: MlflowWorkspace,
) -> None:
    tracker, registry, name = workspace
    version = register(tracker, frame())
    registry.adopt_candidate(version, encode_version_tags(candidate(version)))
    registry.promote(version)
    client = MlflowClient(tracking_uri=mlflow_workspace.uri)
    before = client.get_model_version(name, str(version)).tags

    with pytest.raises(IllegalTransitionError):
        registry.promote(version)

    assert client.get_model_version(name, str(version)).tags == before
    assert client.get_model_version_by_alias(name, "champion").version == version


def test_the_adapter_never_uses_deprecated_stage_apis() -> None:
    source = Path("src/mlops_lifecycle_lab/adapters/registry/mlflow_registry.py").read_text()
    assert "transition_model_version_stage" not in source
    assert "get_latest_versions" not in source
