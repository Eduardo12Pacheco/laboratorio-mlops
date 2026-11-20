"""Composition root: build every adapter, port and use case from typed settings (S21).

Entrypoints never instantiate adapters directly. `build_container` is the single wiring
point, which keeps the dependency direction one-way and lets tests substitute the clock,
trainer, renderer or dataset without patching internals. It also materialises the
operational SQLite file and the MLflow tracking database, so a fresh workspace is ready
before the first CLI command runs.

`serving_mode="http"` is intentionally deferred to the serving task: the HTTP adapter is
part of the serving API milestone, and the drill/tests use the in-process gateway so the
lifecycle can be verified without a server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mlops_lifecycle_lab.adapters.clock import SystemClock
from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset
from mlops_lifecycle_lab.adapters.data.replay_source import ReplayBatchSource
from mlops_lifecycle_lab.adapters.drift.evidently_reports import (
    EvidentlyReportRenderer,
    NullReportRenderer,
)
from mlops_lifecycle_lab.adapters.models.lightgbm_trainer import LightGbmTrainer
from mlops_lifecycle_lab.adapters.registry.mlflow_registry import MlflowRegistry
from mlops_lifecycle_lab.adapters.storage.event_store import SqliteEventStore
from mlops_lifecycle_lab.adapters.storage.kv_store import SqliteKeyValueStore
from mlops_lifecycle_lab.adapters.storage.labeled_data import CompositeLabeledData
from mlops_lifecycle_lab.adapters.storage.prediction_log import SqlitePredictionLog
from mlops_lifecycle_lab.adapters.storage.run_ledger import SqliteRunLedger
from mlops_lifecycle_lab.adapters.storage.snapshot_store import SqliteSnapshotStore
from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.adapters.tracking.mlflow_tracker import MlflowTracker
from mlops_lifecycle_lab.application.ports import Clock, DriftReportRenderer, ModelTrainer
from mlops_lifecycle_lab.application.services.bootstrap import BootstrapChampion
from mlops_lifecycle_lab.application.services.evaluation import EvaluateChallenger
from mlops_lifecycle_lab.application.services.lifecycle import RunLifecycleTick
from mlops_lifecycle_lab.application.services.monitoring import ComputeSnapshot
from mlops_lifecycle_lab.application.services.promotion import (
    PromoteChallenger,
    RejectChallenger,
    RollbackChampion,
)
from mlops_lifecycle_lab.application.services.scoring import InProcessGateway, PredictionService
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.policies import (
    PolicyConfig,
    ProbationPolicy,
    PromotionPolicy,
    RetrainTriggerPolicy,
)


class ContainerSettings(Protocol):
    """Structural view of the typed settings, so the composition root stays in its layer.

    The architecture keeps `bootstrap`, `config` and `logging_setup` independent; the CLI
    passes a real `Settings` object and it satisfies this protocol structurally.
    """

    workspace: Path
    mlflow_tracking_uri: str | None
    mlflow_artifact_root: Path | None
    model_name: str
    experiment_name: str
    seed: int
    window_days: int
    holdout_days: int
    probation_days: int
    serving_mode: str
    fast: bool
    slo_wape: float
    abs_wape_max: float
    baseline_max_ratio: float
    improvement_min: float
    segment_max_regression: float
    bootstrap_resamples: int
    min_labeled_rows: int
    cooldown_days: int
    probation_min_rows: int

    @property
    def effective_tracking_uri(self) -> str:
        """MLflow tracking URI resolved from the workspace or an override."""
        ...

    @property
    def effective_artifact_root(self) -> Path:
        """Directory where MLflow artifacts are written."""
        ...


@dataclass(slots=True)
class Container:
    """Every wired component an entrypoint is allowed to use."""

    settings: ContainerSettings
    clock: Clock
    dataset: BundledDataset
    database: SqliteDatabase
    prediction_log: SqlitePredictionLog
    labeled_data: CompositeLabeledData
    events: SqliteEventStore
    snapshots: SqliteSnapshotStore
    kv: SqliteKeyValueStore
    ledger: SqliteRunLedger
    tracker: MlflowTracker
    registry: MlflowRegistry
    trainer: ModelTrainer
    renderer: DriftReportRenderer
    prediction_service: PredictionService
    gateway: InProcessGateway
    batch_source: ReplayBatchSource
    policies: PolicyConfig
    monitoring: ComputeSnapshot
    bootstrap: BootstrapChampion
    training: TrainCandidate
    evaluation: EvaluateChallenger
    promotion: PromoteChallenger
    rejection: RejectChallenger
    rollback: RollbackChampion
    lifecycle: RunLifecycleTick

    def close(self) -> None:
        """Close the operational database; the composition root owns the connection."""
        self.database.close()


def build_container(
    settings: ContainerSettings,
    *,
    clock: Clock | None = None,
    trainer: ModelTrainer | None = None,
    renderer: DriftReportRenderer | None = None,
    dataset: BundledDataset | None = None,
) -> Container:
    """Wire settings and adapters into a ready-to-use `Container`.

    Raises:
        ValueError: for `serving_mode="http"` until the serving adapter lands; the
            in-process gateway is the only scoring path the lifecycle needs.
    """
    if settings.serving_mode != "inprocess":
        raise ValueError(
            "only the in-process serving mode is wired in this milestone; "
            "set MLL_SERVING_MODE=inprocess"
        )

    settings.workspace.mkdir(parents=True, exist_ok=True)
    database = SqliteDatabase(settings.workspace / "mll.db")
    dataset = dataset or BundledDataset()
    prediction_log = SqlitePredictionLog(database)
    labeled_data = CompositeLabeledData(dataset, prediction_log)
    events = SqliteEventStore(database)
    snapshots = SqliteSnapshotStore(database)
    kv = SqliteKeyValueStore(database)
    ledger = SqliteRunLedger(database)
    tracker = MlflowTracker(
        tracking_uri=settings.effective_tracking_uri,
        experiment_name=settings.experiment_name,
        artifact_root=settings.effective_artifact_root,
        model_name=settings.model_name,
    )
    tracker.ensure_ready()
    registry = MlflowRegistry(
        tracking_uri=settings.effective_tracking_uri,
        model_name=settings.model_name,
    )
    policies = PolicyConfig.from_settings(settings)
    active_clock: Clock = clock or SystemClock()
    active_trainer: ModelTrainer = trainer or LightGbmTrainer()
    active_renderer: DriftReportRenderer = (
        renderer
        if renderer is not None
        else (NullReportRenderer() if settings.fast else EvidentlyReportRenderer())
    )

    prediction_service = PredictionService(
        registry,
        prediction_log,
        active_clock,
        model_name=settings.model_name,
    )
    gateway = InProcessGateway(prediction_service, prediction_log)
    training = TrainCandidate(
        labeled_data,
        active_trainer,
        tracker,
        registry,
        active_clock,
        seed=settings.seed,
        model_name=settings.model_name,
    )
    evaluation = EvaluateChallenger(
        labeled_data,
        registry,
        tracker,
        events,
        active_clock,
        PromotionPolicy(policies),
    )
    promotion = PromoteChallenger(registry, events, kv, active_clock, policies)
    rejection = RejectChallenger(registry, events, active_clock)
    rollback = RollbackChampion(registry, events, kv, active_clock)
    bootstrap = BootstrapChampion(
        registry,
        events,
        active_clock,
        training,
        evaluation,
        promotion,
        rejection,
    )
    monitoring = ComputeSnapshot(
        prediction_log=prediction_log,
        labeled_data=labeled_data,
        registry=registry,
        snapshots=snapshots,
        events=events,
        clock=active_clock,
        tracker=tracker,
        window_days=settings.window_days,
    )
    lifecycle = RunLifecycleTick(
        registry=registry,
        events=events,
        kv=kv,
        ledger=ledger,
        gateway=gateway,
        batch_source=ReplayBatchSource(dataset),
        monitoring=monitoring,
        training=training,
        evaluation=evaluation,
        promotion=promotion,
        rejection=rejection,
        rollback=rollback,
        trigger_policy=RetrainTriggerPolicy(policies),
        probation_policy=ProbationPolicy(policies),
        clock=active_clock,
        config=policies,
    )
    return Container(
        settings=settings,
        clock=active_clock,
        dataset=dataset,
        database=database,
        prediction_log=prediction_log,
        labeled_data=labeled_data,
        events=events,
        snapshots=snapshots,
        kv=kv,
        ledger=ledger,
        tracker=tracker,
        registry=registry,
        trainer=active_trainer,
        renderer=active_renderer,
        prediction_service=prediction_service,
        gateway=gateway,
        batch_source=ReplayBatchSource(dataset),
        policies=policies,
        monitoring=monitoring,
        bootstrap=bootstrap,
        training=training,
        evaluation=evaluation,
        promotion=promotion,
        rejection=rejection,
        rollback=rollback,
        lifecycle=lifecycle,
    )


__all__ = ["Container", "build_container"]
