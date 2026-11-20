"""Typed values exchanged between the application, its ports and the entrypoints.

The DTOs are frozen and dependency-free beyond pandas: they carry exactly the data the
use cases need, validate the cheap invariants at construction time, and expose a
JSON-serialisable representation for the stores, reports and the dashboard.

`encode_version_tags`/`decode_version_tags` are the contract between the training and
evaluation services and any `ModelRegistry` adapter: MLflow only stores strings on a
model version, so the lineage of a `ModelVersion` travels as `mll.*` tags and is rebuilt
by the adapter. Registry contract tests exercise the same round-trip.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd

from mlops_lifecycle_lab.domain.events import LifecycleEvent
from mlops_lifecycle_lab.domain.gates import GateReport
from mlops_lifecycle_lab.domain.lifecycle import ModelStage, ModelVersion
from mlops_lifecycle_lab.domain.metrics import RegressionMetrics
from mlops_lifecycle_lab.domain.policies import PromotionDecision
from mlops_lifecycle_lab.domain.validation import ValidationReport
from mlops_lifecycle_lab.domain.windows import TimeWindow

type JSONValue = bool | int | float | str | list[JSONValue] | dict[str, JSONValue] | None
"""A JSON-serialisable value; stores and trackers accept nothing richer."""

Source = Literal["online", "batch"]
"""Where a prediction was produced: the online endpoint or a batch scoring pass."""

_SOURCES: frozenset[str] = frozenset({"online", "batch"})

TAG_STAGE = "mll.stage"
TAG_RUN_ID = "mll.run_id"
TAG_TRAIN_WINDOW = "mll.train_window"
TAG_HOLDOUT_WINDOW = "mll.holdout_window"
TAG_DATA_FINGERPRINT = "mll.data_fingerprint"
TAG_FEATURE_SPEC_VERSION = "mll.feature_spec_version"
TAG_CREATED_AT = "mll.created_at"
TAG_TRIGGER_REASON = "mll.trigger_reason"
TAG_GIT_SHA = "mll.git_sha"
TAG_REJECT_REASON = "mll.reject_reason"
TAG_FAILED_GATES = "mll.failed_gates"

METRIC_TAGS: Mapping[str, str] = MappingProxyType(
    {
        "rmse": "mll.rmse",
        "mae": "mll.mae",
        "wape": "mll.wape",
        "bias": "mll.bias",
    }
)
"""Registry metric name -> tag key, so a version can be listed without its run."""

_RESERVED_TAGS: frozenset[str] = frozenset(
    {
        TAG_STAGE,
        TAG_RUN_ID,
        TAG_TRAIN_WINDOW,
        TAG_HOLDOUT_WINDOW,
        TAG_DATA_FINGERPRINT,
        TAG_FEATURE_SPEC_VERSION,
        TAG_CREATED_AT,
        TAG_TRIGGER_REASON,
        *METRIC_TAGS.values(),
    }
)


class RunStatus(StrEnum):
    """Final status of one pipeline run."""

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class StepStatus(StrEnum):
    """Status of one pipeline step attempt."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One persisted prediction, exactly the row the prediction log stores (S9)."""

    prediction_id: str
    entity_id: int
    event_ts: datetime
    scored_at: datetime
    model_name: str
    model_version: int
    features: Mapping[str, float]
    prediction: float
    source: Source
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.prediction_id.strip():
            raise ValueError("prediction_id must not be empty")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.entity_id < 0:
            raise ValueError(f"entity_id must be >= 0, got {self.entity_id}")
        if self.model_version < 1:
            raise ValueError(f"model_version must be >= 1, got {self.model_version}")
        if self.source not in _SOURCES:
            raise ValueError(f"unknown prediction source {self.source!r}")
        if self.prediction < 0.0:
            raise ValueError(f"prediction must be >= 0, got {self.prediction}")
        _finite(self.prediction, "prediction")
        object.__setattr__(
            self,
            "features",
            MappingProxyType({str(name): float(value) for name, value in self.features.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation used by the SQLite prediction log."""
        return {
            "prediction_id": self.prediction_id,
            "entity_id": self.entity_id,
            "event_ts": self.event_ts.isoformat(),
            "scored_at": self.scored_at.isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "features": dict(self.features),
            "prediction": self.prediction,
            "source": self.source,
            "request_id": self.request_id,
        }


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """A delayed ground-truth label attached to a previously scored entity."""

    entity_id: int
    label: float
    label_ts: datetime

    def __post_init__(self) -> None:
        if self.entity_id < 0:
            raise ValueError(f"entity_id must be >= 0, got {self.entity_id}")
        _finite(self.label, "label")


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """One prediction returned to an API caller or a batch client."""

    entity_id: int
    prediction: float
    model_name: str
    model_version: int
    scored_at: datetime

    def __post_init__(self) -> None:
        if self.entity_id < 0:
            raise ValueError(f"entity_id must be >= 0, got {self.entity_id}")
        if self.model_version < 1:
            raise ValueError(f"model_version must be >= 1, got {self.model_version}")
        if self.prediction < 0.0:
            raise ValueError(f"prediction must be >= 0, got {self.prediction}")
        _finite(self.prediction, "prediction")


@dataclass(frozen=True, slots=True)
class RawBatch:
    """One week of raw rows plus the labels that become available at the tick end."""

    raw: pd.DataFrame
    labels: pd.DataFrame

    def __post_init__(self) -> None:
        if self.raw.empty:
            raise ValueError("batch source returned no raw rows")
        if self.labels.empty:
            raise ValueError("batch source returned no labels")


@dataclass(frozen=True, slots=True)
class TrainRequest:
    """Inputs of one challenger training run."""

    train_window: TimeWindow
    holdout_window: TimeWindow
    trigger_reason: str
    tick: int | None = None

    def __post_init__(self) -> None:
        if not self.trigger_reason.strip():
            raise ValueError("trigger_reason must not be empty")
        if self.tick is not None and self.tick < 0:
            raise ValueError(f"tick must be >= 0, got {self.tick}")


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Everything a caller needs after a candidate was trained and registered."""

    version: int
    model: ModelVersion
    metrics: RegressionMetrics
    baseline_rmse: float
    data_fingerprint: str
    run_id: str
    training_rows: int
    holdout_rows: int
    validation: ValidationReport


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Inputs of a gate evaluation for one candidate version."""

    version: int
    train_window: TimeWindow
    holdout_window: TimeWindow
    training_verdict: str
    tick: int = 0
    trigger_reason: str | None = None

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")
        if self.training_verdict not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"unknown training verdict {self.training_verdict!r}")
        if self.tick < 0:
            raise ValueError(f"tick must be >= 0, got {self.tick}")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """The gate decision plus the evidence used to make it."""

    version: int
    decision: PromotionDecision
    challenger_metrics: RegressionMetrics
    champion_metrics: RegressionMetrics | None
    baseline_rmse: float
    gate_report: GateReport


@dataclass(frozen=True, slots=True)
class MonitoringSnapshot:
    """Performance and drift of the current champion over the trailing window (S10)."""

    tick: int
    ts: datetime
    model_version: int
    n_rows: int
    rmse: float | None
    mae: float | None
    wape: float | None
    bias: float | None
    numeric_drift_share: float
    prediction_psi: float | None
    psi: Mapping[str, float] = field(default_factory=dict)
    verdict: str = "PASS"

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError(f"tick must be >= 0, got {self.tick}")
        if self.model_version < 1:
            raise ValueError(f"model_version must be >= 1, got {self.model_version}")
        if self.n_rows < 0:
            raise ValueError(f"n_rows must be >= 0, got {self.n_rows}")
        for name in ("rmse", "mae", "wape", "bias", "prediction_psi"):
            value = getattr(self, name)
            if value is not None:
                _finite(float(value), name)
        if not math.isfinite(self.numeric_drift_share):
            raise ValueError("numeric_drift_share must be finite")
        object.__setattr__(
            self, "psi", MappingProxyType({str(k): float(v) for k, v in self.psi.items()})
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation used by the snapshot store and reports."""
        return {
            "tick": self.tick,
            "ts": self.ts.isoformat(),
            "model_version": self.model_version,
            "n_rows": self.n_rows,
            "rmse": self.rmse,
            "mae": self.mae,
            "wape": self.wape,
            "bias": self.bias,
            "numeric_drift_share": self.numeric_drift_share,
            "prediction_psi": self.prediction_psi,
            "psi": dict(self.psi),
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class TickReport:
    """What one lifecycle tick did, returned by the use case in T7."""

    tick: int
    window: TimeWindow
    champion_version: int | None
    validation_verdict: str
    action: str
    snapshot: MonitoringSnapshot | None
    events: tuple[LifecycleEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation used by the drill and the API."""
        return {
            "tick": self.tick,
            "window": {
                "start": self.window.start.isoformat(),
                "end": self.window.end.isoformat(),
            },
            "champion_version": self.champion_version,
            "validation_verdict": self.validation_verdict,
            "action": self.action,
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }


def _window_to_tag(window: TimeWindow | None) -> str:
    if window is None:
        return "none"
    return f"{window.start.isoformat()}/{window.end.isoformat()}"


def _window_from_tag(value: str, name: str) -> TimeWindow | None:
    if value == "none":
        return None
    try:
        start, end = value.split("/")
        return TimeWindow(datetime.fromisoformat(start), datetime.fromisoformat(end))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid {name} tag {value!r}") from exc


def encode_version_tags(version: ModelVersion) -> dict[str, str]:
    """Encode a `ModelVersion` as MLflow-compatible string tags."""
    tags = {str(key): str(value) for key, value in version.tags.items()}
    tags[TAG_STAGE] = version.stage.value
    tags[TAG_RUN_ID] = version.run_id
    tags[TAG_TRAIN_WINDOW] = _window_to_tag(version.train_window)
    tags[TAG_HOLDOUT_WINDOW] = _window_to_tag(version.holdout_window)
    tags[TAG_DATA_FINGERPRINT] = version.data_fingerprint
    tags[TAG_FEATURE_SPEC_VERSION] = version.feature_spec_version
    tags[TAG_CREATED_AT] = version.created_at.isoformat()
    tags[TAG_TRIGGER_REASON] = version.trigger_reason or ""
    for metric, tag in METRIC_TAGS.items():
        if metric in version.metrics:
            tags[tag] = repr(float(version.metrics[metric]))
    return tags


def decode_version_tags(name: str, version: int, tags: Mapping[str, str]) -> ModelVersion:
    """Rebuild a `ModelVersion` from the tags written by `encode_version_tags`."""
    try:
        stage = ModelStage(tags[TAG_STAGE])
        run_id = tags[TAG_RUN_ID]
        train_window = _window_from_tag(tags[TAG_TRAIN_WINDOW], TAG_TRAIN_WINDOW)
        holdout_window = _window_from_tag(tags[TAG_HOLDOUT_WINDOW], TAG_HOLDOUT_WINDOW)
        data_fingerprint = tags[TAG_DATA_FINGERPRINT]
        feature_spec_version = tags[TAG_FEATURE_SPEC_VERSION]
        created_at = datetime.fromisoformat(tags[TAG_CREATED_AT])
    except KeyError as exc:
        raise ValueError(f"model version {name} v{version} is missing tag {exc.args[0]!r}") from exc
    if train_window is None:
        raise ValueError(f"model version {name} v{version} has no training window")
    metrics: dict[str, float] = {}
    for metric, tag in METRIC_TAGS.items():
        if tag in tags:
            metrics[metric] = float(tags[tag])
    extra_tags = {
        str(key): str(value)
        for key, value in tags.items()
        if key.startswith("mll.") and key not in _RESERVED_TAGS
    }
    return ModelVersion(
        name=name,
        version=version,
        stage=stage,
        run_id=run_id,
        metrics=metrics,
        train_window=train_window,
        holdout_window=holdout_window,
        data_fingerprint=data_fingerprint,
        feature_spec_version=feature_spec_version,
        created_at=created_at,
        trigger_reason=tags.get(TAG_TRIGGER_REASON) or None,
        tags=extra_tags,
    )


__all__ = [
    "METRIC_TAGS",
    "TAG_CREATED_AT",
    "TAG_DATA_FINGERPRINT",
    "TAG_FAILED_GATES",
    "TAG_FEATURE_SPEC_VERSION",
    "TAG_GIT_SHA",
    "TAG_HOLDOUT_WINDOW",
    "TAG_REJECT_REASON",
    "TAG_RUN_ID",
    "TAG_STAGE",
    "TAG_TRAIN_WINDOW",
    "TAG_TRIGGER_REASON",
    "EvaluationRequest",
    "EvaluationResult",
    "JSONValue",
    "LabelRecord",
    "MonitoringSnapshot",
    "PredictionRecord",
    "PredictionResult",
    "RawBatch",
    "RunStatus",
    "Source",
    "StepStatus",
    "TickReport",
    "TrainRequest",
    "TrainResult",
    "decode_version_tags",
    "encode_version_tags",
]
