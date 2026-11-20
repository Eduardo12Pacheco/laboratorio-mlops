"""Model lifecycle state machine (S7).

MLflow stages are deprecated, so the authoritative stage lives in the ``mll.stage`` tag and
this module is the only place that decides which stage changes are legal. Registry adapters
validate a transition here *before* writing a tag or alias, which keeps an illegal
transition from ever reaching MLflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from mlops_lifecycle_lab.domain.errors import IllegalTransitionError
from mlops_lifecycle_lab.domain.windows import TimeWindow


class ModelStage(StrEnum):
    """Lifecycle stage encoded in the ``mll.stage`` model-version tag."""

    CANDIDATE = "CANDIDATE"
    CHAMPION = "CHAMPION"
    ARCHIVED = "ARCHIVED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


ALLOWED: Mapping[ModelStage, frozenset[ModelStage]] = {
    ModelStage.CANDIDATE: frozenset({ModelStage.CHAMPION, ModelStage.REJECTED}),
    ModelStage.CHAMPION: frozenset({ModelStage.ARCHIVED, ModelStage.ROLLED_BACK}),
    ModelStage.ARCHIVED: frozenset({ModelStage.CHAMPION}),
    ModelStage.REJECTED: frozenset(),
    ModelStage.ROLLED_BACK: frozenset(),
}
"""Every legal transition. Terminal stages have no outgoing edge."""


def allowed_targets(stage: ModelStage) -> frozenset[ModelStage]:
    """Return the stages `stage` may move to, or an empty set for terminal stages."""
    return ALLOWED.get(stage, frozenset())


def can_transition(source: ModelStage, target: ModelStage) -> bool:
    """Whether `source -> target` is one of the documented transitions."""
    return target in allowed_targets(source)


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """A registered model version plus the lineage metadata the lab cares about.

    The mapping fields are frozen into read-only proxies on construction so a caller cannot
    mutate the metrics or tags of a version it received from the registry.
    """

    name: str
    version: int
    stage: ModelStage
    run_id: str
    metrics: Mapping[str, float]
    train_window: TimeWindow
    holdout_window: TimeWindow | None
    data_fingerprint: str
    feature_spec_version: str
    created_at: datetime
    trigger_reason: str | None
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model name must not be empty")
        if self.version < 1:
            raise ValueError(f"model version must be >= 1, got {self.version}")
        if not self.feature_spec_version.strip():
            raise ValueError("feature_spec_version must not be empty")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))


def transition(version: ModelVersion, target: ModelStage) -> ModelVersion:
    """Return a copy of `version` in `target`, or raise `IllegalTransitionError`.

    The input is never mutated: a lifecycle change produces a new value, so callers can
    keep the previous version for audit until the registry write succeeds.
    """
    if not can_transition(version.stage, target):
        allowed = ", ".join(sorted(stage.value for stage in allowed_targets(version.stage)))
        raise IllegalTransitionError(
            f"illegal transition for {version.name} v{version.version}: "
            f"{version.stage.value} -> {target.value}; allowed: {allowed or 'none'}"
        )
    return replace(version, stage=target)


__all__ = [
    "ALLOWED",
    "ModelStage",
    "ModelVersion",
    "allowed_targets",
    "can_transition",
    "transition",
]
