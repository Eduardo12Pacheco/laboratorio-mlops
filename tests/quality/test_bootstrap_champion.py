"""Quality gate: bootstrap v1 through the real adapters must satisfy S5.

This is the integration-level proof that the LightGBM trainer, the MLflow tracker and
registry, the SQLite stores and the application use cases compose into a champion that
meets the planning metrics. Numbers are asserted with the S5 tolerances, never loosened.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from mlops_lifecycle_lab.adapters.clock import SystemClock
from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset
from mlops_lifecycle_lab.adapters.models.lightgbm_trainer import LightGbmTrainer
from mlops_lifecycle_lab.adapters.registry.mlflow_registry import MlflowRegistry
from mlops_lifecycle_lab.adapters.storage.event_store import SqliteEventStore
from mlops_lifecycle_lab.adapters.storage.kv_store import SqliteKeyValueStore
from mlops_lifecycle_lab.adapters.storage.labeled_data import CompositeLabeledData
from mlops_lifecycle_lab.adapters.storage.prediction_log import SqlitePredictionLog
from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.adapters.tracking.mlflow_tracker import MlflowTracker
from mlops_lifecycle_lab.application.dto import TrainRequest
from mlops_lifecycle_lab.application.services.bootstrap import (
    BOOTSTRAP_HOLDOUT_WINDOW,
    BOOTSTRAP_TRAIN_WINDOW,
    BootstrapChampion,
)
from mlops_lifecycle_lab.application.services.evaluation import EvaluateChallenger
from mlops_lifecycle_lab.application.services.promotion import (
    PromoteChallenger,
    RejectChallenger,
)
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.events import EventType
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from tests.fakes.mlflow_workspace import MlflowWorkspace


def test_bootstrap_champion_meets_the_s5_quality_gate(
    tmp_path: Path,
    mlflow_workspace: MlflowWorkspace,
) -> None:
    workspace = tmp_path
    database = SqliteDatabase(workspace / "mll.db")
    try:
        model_name = f"quality-{uuid4().hex}"
        dataset = BundledDataset()
        predictions = SqlitePredictionLog(database)
        archive = CompositeLabeledData(dataset, predictions)
        events = SqliteEventStore(database)
        kv = SqliteKeyValueStore(database)
        registry = MlflowRegistry(tracking_uri=mlflow_workspace.uri, model_name=model_name)
        tracker = MlflowTracker(
            tracking_uri=mlflow_workspace.uri,
            experiment_name="bike-demand",
            artifact_root=mlflow_workspace.root / "artifacts",
            model_name=model_name,
        )
        clock = SystemClock()
        trainer = LightGbmTrainer()
        training = TrainCandidate(
            archive,
            trainer,
            tracker,
            registry,
            clock,
            model_name=model_name,
        )
        bootstrap = BootstrapChampion(
            registry,
            events,
            clock,
            training,
            EvaluateChallenger(archive, registry, tracker, events, clock),
            PromoteChallenger(registry, events, kv, clock),
            RejectChallenger(registry, events, clock),
        )

        champion = bootstrap.execute()

        assert champion.version == 1
        assert champion.stage.value == "CHAMPION"
        assert registry.get_champion() == champion
        metrics = champion.metrics
        assert metrics["rmse"] <= 55.0
        assert metrics["wape"] <= 0.26
        completed = events.list(event_type=EventType.BOOTSTRAP_COMPLETED)[-1]
        baseline_rmse = float(str(completed.payload["baseline_rmse"]))
        assert metrics["rmse"] / baseline_rmse <= 0.85
        assert abs(metrics["rmse"] - 50.88) <= 2.0

        assert bootstrap.execute() == champion
        assert len(registry.list_versions()) == 1

        same_windows = TrainRequest(
            train_window=BOOTSTRAP_TRAIN_WINDOW,
            holdout_window=BOOTSTRAP_HOLDOUT_WINDOW,
            trigger_reason="MANUAL",
            tick=1,
        )
        second = training.execute(same_windows)
        assert second.data_fingerprint == completed.payload["data_fingerprint"]

        raw = dataset.raw()
        holdout = raw.loc[
            (raw["timestamp"] >= BOOTSTRAP_HOLDOUT_WINDOW.start)
            & (raw["timestamp"] < BOOTSTRAP_HOLDOUT_WINDOW.end)
        ]
        features = build_features(holdout).loc[:, list(FEATURE_COLUMNS)]
        first_predictions = registry.load_predictor(champion.version).predict(features)
        second_predictions = registry.load_predictor(second.version).predict(features)
        assert np.array_equal(first_predictions, second_predictions)
    finally:
        database.close()
