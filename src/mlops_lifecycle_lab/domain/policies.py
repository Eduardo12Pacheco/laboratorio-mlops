"""Lifecycle policies: retrain trigger, promotion decisions and probation (S11-S13).

The policies are pure: they receive already-computed snapshots and gate evidence and
return typed decisions. Thresholds live in one immutable `PolicyConfig` built from the
application settings, so the CLI, the drill and the scheduler can never disagree about
what counts as a breach.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol

from mlops_lifecycle_lab.domain.gates import (
    AbsoluteWapeGate,
    BeatsBaselineGate,
    DataValidGate,
    Gate,
    GateContext,
    GateReport,
    GateResult,
    ImprovementGate,
    SegmentRegressionGate,
    SignificanceGate,
)

_TOLERANCE = 1e-12


class _SettingsView(Protocol):
    """Structural view of the numeric settings mirrored by `PolicyConfig`."""

    slo_wape: float
    abs_wape_max: float
    baseline_max_ratio: float
    improvement_min: float
    segment_max_regression: float
    bootstrap_resamples: int
    window_days: int
    min_labeled_rows: int
    cooldown_days: int
    holdout_days: int
    probation_days: int
    probation_min_rows: int
    seed: int


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Every numeric threshold used by the three lifecycle policies."""

    slo_wape: float = 0.30
    abs_wape_max: float = 0.30
    baseline_max_ratio: float = 0.85
    improvement_min: float = 0.05
    segment_max_regression: float = 0.10
    significance_confidence: float = 0.95
    bootstrap_resamples: int = 2000
    window_days: int = 28
    min_labeled_rows: int = 300
    cooldown_days: int = 28
    holdout_days: int = 14
    probation_days: int = 21
    probation_min_rows: int = 150
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.slo_wape <= 1.0:
            raise ValueError(f"slo_wape must be in (0, 1], got {self.slo_wape}")
        if not 0.0 < self.abs_wape_max <= 1.0:
            raise ValueError(f"abs_wape_max must be in (0, 1], got {self.abs_wape_max}")
        if self.baseline_max_ratio <= 0.0:
            raise ValueError(f"baseline_max_ratio must be positive, got {self.baseline_max_ratio}")
        if not 0.0 <= self.improvement_min <= 1.0:
            raise ValueError(f"improvement_min must be in [0, 1], got {self.improvement_min}")
        if self.segment_max_regression < 0.0:
            raise ValueError(
                f"segment_max_regression must be non-negative, got {self.segment_max_regression}"
            )
        if not 0.0 < self.significance_confidence < 1.0:
            raise ValueError(
                f"significance_confidence must be in (0, 1), got {self.significance_confidence}"
            )
        for name in (
            "bootstrap_resamples",
            "window_days",
            "min_labeled_rows",
            "cooldown_days",
            "holdout_days",
            "probation_days",
            "probation_min_rows",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    @classmethod
    def from_settings(cls, settings: _SettingsView) -> PolicyConfig:
        """Build the immutable policy config from the application settings."""
        return cls(
            slo_wape=settings.slo_wape,
            abs_wape_max=settings.abs_wape_max,
            baseline_max_ratio=settings.baseline_max_ratio,
            improvement_min=settings.improvement_min,
            segment_max_regression=settings.segment_max_regression,
            bootstrap_resamples=settings.bootstrap_resamples,
            window_days=settings.window_days,
            min_labeled_rows=settings.min_labeled_rows,
            cooldown_days=settings.cooldown_days,
            holdout_days=settings.holdout_days,
            probation_days=settings.probation_days,
            probation_min_rows=settings.probation_min_rows,
            seed=settings.seed,
        )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Outcome of evaluating the gates for one challenger."""

    approved: bool
    report: GateReport
    reason: str

    @property
    def failed_gates(self) -> tuple[str, ...]:
        """Names of the gates that rejected the challenger."""
        return self.report.failed_gates


class PromotionPolicy:
    """Evaluate G1-G3 for the first champion and G1-G6 for every later challenger."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self._config = config or PolicyConfig()

    def evaluate(self, context: GateContext) -> PromotionDecision:
        """Evaluate every applicable gate, without short-circuiting."""
        data_valid = DataValidGate()
        absolute_wape = AbsoluteWapeGate(self._config.abs_wape_max)
        beats_baseline = BeatsBaselineGate(self._config.baseline_max_ratio)
        gates: list[Gate] = [data_valid, absolute_wape, beats_baseline]
        if not context.is_first_champion:
            gates.extend(
                [
                    ImprovementGate(self._config.improvement_min),
                    SignificanceGate(
                        confidence=self._config.significance_confidence,
                        resamples=self._config.bootstrap_resamples,
                        seed=self._config.seed,
                    ),
                    SegmentRegressionGate(self._config.segment_max_regression),
                ]
            )

        results: list[GateResult] = [gate.evaluate(context) for gate in gates]
        approved = all(result.passed for result in results)
        failed = [result.name for result in results if not result.passed]
        report = GateReport(results=tuple(results), approved=approved)
        reason = "all gates passed" if approved else ", ".join(failed)
        return PromotionDecision(approved=approved, report=report, reason=reason)


class TriggerOutcome(StrEnum):
    """Why the retrain trigger did or did not fire."""

    PERFORMANCE_SLO_BREACH = "PERFORMANCE_SLO_BREACH"
    MANUAL = "MANUAL"
    HEALTHY = "HEALTHY"
    INSUFFICIENT_LABELS = "INSUFFICIENT_LABELS"
    COOLDOWN = "COOLDOWN"
    IN_PROBATION = "IN_PROBATION"


class TriggerReason(StrEnum):
    """The two reasons that can start a retrain."""

    PERFORMANCE_SLO_BREACH = "PERFORMANCE_SLO_BREACH"
    MANUAL = "MANUAL"


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Outcome plus, when retraining, the trigger recorded in the event payload."""

    outcome: TriggerOutcome
    trigger: TriggerReason | None = None
    detail: str = ""

    @property
    def should_retrain(self) -> bool:
        """Whether the caller must start a retrain attempt."""
        return self.trigger is not None


class SnapshotView(Protocol):
    """The fields the trigger and probation policies need from a monitoring snapshot."""

    n_rows: int
    wape: float | None


class RetrainTriggerPolicy:
    """Label-based SLO trigger with a cooldown and a probation veto (S11, D5).

    Covariate drift is deliberately absent from this policy: PSI against a training window
    is informational here and must never trigger a retrain (ADR-0004).
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self._config = config or PolicyConfig()

    def evaluate(
        self,
        snapshot: SnapshotView,
        last_retrain_at: datetime | None,
        now: datetime,
        in_probation: bool,
        manual: bool = False,
    ) -> TriggerDecision:
        """Decide in the documented order: probation, manual, labels, cooldown, SLO."""
        if in_probation:
            return TriggerDecision(
                outcome=TriggerOutcome.IN_PROBATION,
                detail="probation is active; rollback takes priority over retraining",
            )
        if manual:
            return TriggerDecision(
                outcome=TriggerOutcome.MANUAL,
                trigger=TriggerReason.MANUAL,
                detail="manual retrain requested",
            )
        if snapshot.n_rows < self._config.min_labeled_rows or snapshot.wape is None:
            return TriggerDecision(
                outcome=TriggerOutcome.INSUFFICIENT_LABELS,
                detail=(
                    f"{snapshot.n_rows} labeled rows; at least "
                    f"{self._config.min_labeled_rows} are required"
                ),
            )
        if last_retrain_at is not None and now - last_retrain_at < timedelta(
            days=self._config.cooldown_days
        ):
            return TriggerDecision(
                outcome=TriggerOutcome.COOLDOWN,
                detail=(
                    f"last retrain at {last_retrain_at.isoformat()}; cooldown is "
                    f"{self._config.cooldown_days} days"
                ),
            )
        if snapshot.wape + _TOLERANCE >= self._config.slo_wape:
            return TriggerDecision(
                outcome=TriggerOutcome.PERFORMANCE_SLO_BREACH,
                trigger=TriggerReason.PERFORMANCE_SLO_BREACH,
                detail=(f"trailing WAPE {snapshot.wape:.4f} >= SLO {self._config.slo_wape:.4f}"),
            )
        return TriggerDecision(
            outcome=TriggerOutcome.HEALTHY,
            detail=(f"trailing WAPE {snapshot.wape:.4f} < SLO {self._config.slo_wape:.4f}"),
        )


class ProbationVerdict(StrEnum):
    """What the probation watcher should do at this tick."""

    CONTINUE = "CONTINUE"
    PASSED = "PASSED"
    ROLLBACK = "ROLLBACK"


class ProbationPolicy:
    """Watch a freshly promoted champion: roll back on a breach, pass at the deadline."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self._config = config or PolicyConfig()

    def evaluate(
        self,
        snapshot: SnapshotView,
        promoted_at: datetime,
        now: datetime,
    ) -> ProbationVerdict:
        """Rollback wins over passing; a missing WAPE can never force a rollback."""
        if (
            snapshot.n_rows >= self._config.probation_min_rows
            and snapshot.wape is not None
            and snapshot.wape > self._config.slo_wape + _TOLERANCE
        ):
            return ProbationVerdict.ROLLBACK
        if now - promoted_at >= timedelta(days=self._config.probation_days):
            return ProbationVerdict.PASSED
        return ProbationVerdict.CONTINUE


__all__ = [
    "PolicyConfig",
    "ProbationPolicy",
    "ProbationVerdict",
    "PromotionDecision",
    "PromotionPolicy",
    "RetrainTriggerPolicy",
    "SnapshotView",
    "TriggerDecision",
    "TriggerOutcome",
    "TriggerReason",
]
