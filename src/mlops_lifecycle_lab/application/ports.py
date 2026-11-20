"""Ports of the hexagonal architecture (application core).

Every port is a `Protocol` so adapters and test doubles are interchangeable without
inheritance. The signatures are the contract that T6 adapters (SQLite, MLflow, LightGBM,
APScheduler, HTTP) and the in-memory fakes both implement; contract suites in
`tests/contract/` are re-run against both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mlops_lifecycle_lab.application.dto import (
    JSONValue,
    LabelRecord,
    MonitoringSnapshot,
    PredictionRecord,
    PredictionResult,
    RawBatch,
    RunStatus,
    StepStatus,
)
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.lifecycle import ModelVersion
from mlops_lifecycle_lab.domain.windows import TimeWindow

ALIAS_CHAMPION = "champion"
ALIAS_PREVIOUS_CHAMPION = "previous-champion"
ALIAS_CHALLENGER = "challenger"
"""Aliases of `S7`; model stages are never used (deprecated in MLflow)."""


class Clock(Protocol):
    """Source of the current time; replay and drills use a simulated clock."""

    def now(self) -> datetime:
        """Return the current instant (naive, matching the dataset)."""
        ...


class Predictor(Protocol):
    """A loaded model that turns feature rows into demand predictions."""

    def predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        """Predict one non-negative value per feature row."""
        ...


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """A fitted predictor plus the hyper-parameters that produced it."""

    predictor: Predictor
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class LoggedModel:
    """Result of logging a model to the experiment tracker."""

    uri: str
    registered_version: int

    def __post_init__(self) -> None:
        if self.registered_version < 1:
            raise ValueError(f"registered_version must be >= 1, got {self.registered_version}")


class ModelTrainer(Protocol):
    """Strategy that fits a candidate model on a labeled frame."""

    def fit(self, frame: pd.DataFrame, seed: int) -> TrainedModel:
        """Fit on `frame` (feature columns plus `cnt`) and return the trained model."""
        ...


class RunHandle(Protocol):
    """One open experiment run; all logging happens through this handle."""

    run_id: str

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log hyper-parameters and seeds."""
        ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Log numeric metrics."""
        ...

    def log_json(self, name: str, obj: JSONValue) -> None:
        """Log a JSON-serialisable artifact under `name`."""
        ...

    def log_model(self, trained: TrainedModel, input_example: pd.DataFrame) -> LoggedModel:
        """Log the model with an inferred signature and register a new version."""
        ...


class ExperimentTracker(Protocol):
    """Experiment tracking + model registration entry point (`MlflowTracker`)."""

    def run(
        self,
        name: str,
        tags: Mapping[str, str],
    ) -> AbstractContextManager[RunHandle]:
        """Open a run as a context manager; the run closes even when the body raises."""
        ...

    def load_json(self, name: str, run_id: str) -> JSONValue | None:
        """Read a JSON artifact of a closed run, or `None` when it does not exist.

        Monitoring needs the champion's `reference_predictions.json` to compute the
        prediction-distribution PSI; the tracker is the only component that knows where
        artifacts live, so reading them is part of the same port.
        """
        ...


class ModelRegistry(Protocol):
    """Champion/challenger registry driven by tags and aliases (`S7`)."""

    def adopt_candidate(self, version: int, tags: Mapping[str, str]) -> ModelVersion:
        """Tag a freshly registered version as `CANDIDATE` and point `challenger` at it."""
        ...

    def get_champion(self) -> ModelVersion | None:
        """Return the version currently pointed at by the `champion` alias."""
        ...

    def get_version(self, version: int) -> ModelVersion | None:
        """Return one version, or `None` when it does not exist."""
        ...

    def list_versions(self) -> list[ModelVersion]:
        """Return every registered version ordered by version number."""
        ...

    def get_alias(self, alias: str) -> ModelVersion | None:
        """Return the version pointed at by `alias`, or `None`."""
        ...

    def promote(self, version: int) -> ModelVersion:
        """Promote a candidate, archiving the current champion and moving the aliases."""
        ...

    def reject(self, version: int, reason: str, failed_gates: Sequence[str]) -> ModelVersion:
        """Reject a candidate and record the reason and the failing gates."""
        ...

    def rollback(self, reason: str) -> ModelVersion:
        """Restore `previous-champion`, marking the current champion `ROLLED_BACK`."""
        ...

    def load_predictor(self, version: int) -> Predictor:
        """Load the artifact of `version` as a predictor."""
        ...


class PredictionLog(Protocol):
    """Idempotent prediction log with delayed label attachment (`S9`)."""

    def append(self, records: Sequence[PredictionRecord]) -> None:
        """Persist predictions; a repeated `prediction_id` is ignored."""
        ...

    def attach_labels(self, labels: Sequence[LabelRecord]) -> None:
        """Attach delayed labels by `entity_id`; unknown or repeated labels are ignored."""
        ...

    def labeled_predictions(
        self,
        start: datetime,
        end: datetime,
        model_version: int | None = None,
    ) -> pd.DataFrame:
        """Return labeled predictions with `start <= event_ts < end` as a frame."""
        ...

    def all_labeled_features(self, end: datetime) -> pd.DataFrame:
        """Return labeled production rows before `end`, shaped like a training frame."""
        ...


class LabeledDataRepository(Protocol):
    """Archive plus labeled production rows, the single training data source (D6)."""

    def training_frame(self, end_exclusive: datetime) -> pd.DataFrame:
        """Return raw rows with labels whose timestamp is before `end_exclusive`, sorted."""
        ...


class EventStore(Protocol):
    """Append-only lifecycle event audit log."""

    def append(self, event: LifecycleEvent) -> None:
        """Append one event."""
        ...

    def list(
        self,
        *,
        event_type: EventType | None = None,
        model_version: int | None = None,
        limit: int | None = None,
    ) -> list[LifecycleEvent]:
        """Return matching events in chronological order, keeping the last `limit`."""
        ...


class SnapshotStore(Protocol):
    """Persisted monitoring snapshots, one per tick."""

    def save(self, snapshot: MonitoringSnapshot) -> None:
        """Persist or replace the snapshot of its tick."""
        ...

    def list(self, limit: int | None = None) -> list[MonitoringSnapshot]:
        """Return snapshots in tick order, optionally the last `limit`."""
        ...


class KeyValueStore(Protocol):
    """Small JSON-value state store (replay cursor, probation, last retrain)."""

    def get(self, key: str) -> JSONValue | None:
        """Return the stored value or `None`."""
        ...

    def set(self, key: str, value: JSONValue) -> None:
        """Store a JSON-serialisable value; `None` clears the key."""
        ...


class RunLedger(Protocol):
    """Run and step ledger written by `Pipeline` (`S14`)."""

    def start(self, run_id: str, name: str, tick: int | None, started_at: datetime) -> None:
        """Record the start of a run."""
        ...

    def finish_step(
        self,
        run_id: str,
        index: int,
        name: str,
        status: StepStatus,
        started_at: datetime,
        finished_at: datetime,
        error: str | None = None,
    ) -> None:
        """Record one completed (successful or failed) attempt of a step."""
        ...

    def finish(
        self,
        run_id: str,
        status: RunStatus,
        finished_at: datetime,
        error: str | None = None,
    ) -> None:
        """Record the final status of a run."""
        ...


class DriftReportRenderer(Protocol):
    """Human-readable drift report (Evidently); the Null Object returns `None`."""

    def render(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        path: Path,
    ) -> Path | None:
        """Render a report at `path` and return it, or `None` when disabled."""
        ...


class ScoringGateway(Protocol):
    """Scores batches through the API and submits delayed labels (`S14`)."""

    def predict_batch(self, raw_rows: pd.DataFrame) -> list[PredictionResult]:
        """Score every raw row through the serving path."""
        ...

    def submit_labels(self, labels: Sequence[LabelRecord]) -> None:
        """Submit the labels that became available at the tick end."""
        ...


class BatchSource(Protocol):
    """Weekly replay input: raw rows plus the labels hidden until the tick end."""

    def week(self, window: TimeWindow) -> RawBatch:
        """Return the batch for `window` with its labels."""
        ...


__all__ = [
    "ALIAS_CHALLENGER",
    "ALIAS_CHAMPION",
    "ALIAS_PREVIOUS_CHAMPION",
    "BatchSource",
    "Clock",
    "DriftReportRenderer",
    "EventStore",
    "ExperimentTracker",
    "KeyValueStore",
    "LabeledDataRepository",
    "LoggedModel",
    "ModelRegistry",
    "ModelTrainer",
    "PredictionLog",
    "Predictor",
    "RunHandle",
    "RunLedger",
    "ScoringGateway",
    "SnapshotStore",
    "TrainedModel",
]
