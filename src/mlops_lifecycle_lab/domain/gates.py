"""Promotion gates G1-G6 for the champion/challenger protocol (S12).

Every gate is a pure computation over the same `GateContext`, so a report can be replayed
without MLflow or any adapter. Gates are evaluated in the documented order and never
short-circuit: a rejection must show *every* failing gate, not just the first one (D6).
Boundary comparisons allow a one-part-in-a-trillion tolerance so a value that is exactly on
a threshold is not rejected by floating-point noise; a value above the threshold by any
meaningful margin still fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from mlops_lifecycle_lab.domain.errors import GateEvaluationError
from mlops_lifecycle_lab.domain.schema import SEGMENT_COLUMN
from mlops_lifecycle_lab.domain.stats import day_block_bootstrap_ci

GATE_DATA_VALID = "DATA_VALID"
GATE_ABS_WAPE = "ABS_WAPE"
GATE_BEATS_BASELINE = "BEATS_BASELINE"
GATE_IMPROVEMENT = "IMPROVEMENT"
GATE_SIGNIFICANCE = "SIGNIFICANCE"
GATE_SEGMENT = "SEGMENT"

GATE_NAMES: tuple[str, ...] = (
    GATE_DATA_VALID,
    GATE_ABS_WAPE,
    GATE_BEATS_BASELINE,
    GATE_IMPROVEMENT,
    GATE_SIGNIFICANCE,
    GATE_SEGMENT,
)
"""Gate names in evaluation order, exactly as reported in MLflow and the drill."""

_TOLERANCE = 1e-12
_VALID_VERDICTS = frozenset({"PASS", "WARN", "FAIL"})


@dataclass(frozen=True, slots=True)
class GateResult:
    """Evidence and verdict of one gate."""

    name: str
    passed: bool
    observed: float | str
    threshold: float | str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation saved in the gate report artifact."""
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class GateReport:
    """All gate results plus the aggregate decision."""

    results: tuple[GateResult, ...]
    approved: bool

    @property
    def failed_gates(self) -> tuple[str, ...]:
        """Names of the gates that did not pass, in evaluation order."""
        return tuple(result.name for result in self.results if not result.passed)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation saved as a training artifact."""
        return {
            "approved": self.approved,
            "results": [result.to_dict() for result in self.results],
        }


def _vector(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise GateEvaluationError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise GateEvaluationError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise GateEvaluationError(f"{name} must contain finite values")
    return array


@dataclass(frozen=True, slots=True)
class GateContext:
    """Everything the six gates are allowed to look at.

    Attributes:
        timestamps: one timestamp per holdout row; used for the day blocks of G5.
        actuals: actual demand for the same rows; used to compute WAPE in G2.
        challenger_errors: signed residuals `prediction - actual` of the challenger.
        champion_errors: signed residuals of the incumbent; absent for the first champion.
        segments: segment label (``workingday``) per row; required by G6.
        baseline_rmse: RMSE of the naive profile baseline on the same holdout; G3.
        training_verdict: `PASS`, `WARN` or `FAIL` of the training-frame validation; G1.
        is_first_champion: when true only G1-G3 are applicable.
    """

    timestamps: ArrayLike
    actuals: ArrayLike
    challenger_errors: ArrayLike
    champion_errors: ArrayLike | None = None
    segments: ArrayLike | None = None
    baseline_rmse: float | None = None
    training_verdict: str = "PASS"
    is_first_champion: bool = False

    def __post_init__(self) -> None:
        actuals = _vector(self.actuals, "actuals")
        challenger = _vector(self.challenger_errors, "challenger_errors")
        if actuals.size != challenger.size:
            raise GateEvaluationError(
                f"length mismatch: actuals has {actuals.size} values, "
                f"challenger_errors has {challenger.size}"
            )

        champion: np.ndarray | None = None
        if self.champion_errors is not None:
            champion = _vector(self.champion_errors, "champion_errors")
            if champion.size != challenger.size:
                raise GateEvaluationError(
                    f"length mismatch: champion_errors has {champion.size} values, "
                    f"challenger_errors has {challenger.size}"
                )

        segments: np.ndarray | None = None
        if self.segments is not None:
            segments = np.asarray(self.segments)
            if segments.ndim != 1 or segments.size != challenger.size:
                raise GateEvaluationError(
                    "segments must be one-dimensional and match the error vectors"
                )

        timestamps = np.asarray(self.timestamps)
        if timestamps.ndim != 1 or timestamps.size != challenger.size:
            raise GateEvaluationError(
                "timestamps must be one-dimensional and match the error vectors"
            )

        verdict = str(self.training_verdict)
        if verdict not in _VALID_VERDICTS:
            raise GateEvaluationError(f"unknown training verdict {verdict!r}")

        if self.baseline_rmse is not None and (
            not math.isfinite(self.baseline_rmse) or self.baseline_rmse < 0.0
        ):
            raise GateEvaluationError("baseline_rmse must be finite and non-negative")

        object.__setattr__(self, "actuals", actuals)
        object.__setattr__(self, "challenger_errors", challenger)
        object.__setattr__(self, "champion_errors", champion)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "training_verdict", verdict)


class Gate(Protocol):
    """One promotion gate."""

    name: str

    def evaluate(self, context: GateContext) -> GateResult:
        """Evaluate the gate and return its evidence, never raising for a simple fail."""
        ...


def _as_array(values: ArrayLike) -> np.ndarray:
    """Narrow an already-validated `ArrayLike` field back to a float array."""
    return np.asarray(values, dtype=np.float64)


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors**2)))


def _wape(actuals: np.ndarray, errors: np.ndarray) -> float:
    total = float(actuals.sum())
    if total <= 0.0:
        raise GateEvaluationError("WAPE is undefined when the actual total is zero")
    return float(np.abs(errors).sum() / total)


def _at_most(value: float, limit: float) -> bool:
    return value <= limit + _TOLERANCE


def _at_least(value: float, limit: float) -> bool:
    return value + _TOLERANCE >= limit


class DataValidGate:
    """G1: the training frame must not have failed the data contract."""

    name = GATE_DATA_VALID

    def evaluate(self, context: GateContext) -> GateResult:
        verdict = context.training_verdict
        return GateResult(
            name=self.name,
            passed=verdict != "FAIL",
            observed=verdict,
            threshold="not FAIL",
            detail=f"training frame validation verdict is {verdict}",
        )


class AbsoluteWapeGate:
    """G2: the challenger must meet the absolute WAPE budget on the holdout."""

    name = GATE_ABS_WAPE

    def __init__(self, max_wape: float = 0.30) -> None:
        if not 0.0 < max_wape <= 1.0:
            raise ValueError(f"max_wape must be in (0, 1], got {max_wape}")
        self._max_wape = max_wape

    def evaluate(self, context: GateContext) -> GateResult:
        wape = _wape(_as_array(context.actuals), _as_array(context.challenger_errors))
        return GateResult(
            name=self.name,
            passed=_at_most(wape, self._max_wape),
            observed=wape,
            threshold=self._max_wape,
            detail=f"challenger holdout WAPE {wape:.4f} vs budget {self._max_wape:.4f}",
        )


class BeatsBaselineGate:
    """G3: the challenger must beat the naive profile baseline by the given ratio."""

    name = GATE_BEATS_BASELINE

    def __init__(self, max_ratio: float = 0.85) -> None:
        if max_ratio <= 0.0:
            raise ValueError(f"max_ratio must be positive, got {max_ratio}")
        self._max_ratio = max_ratio

    def evaluate(self, context: GateContext) -> GateResult:
        if context.baseline_rmse is None:
            raise GateEvaluationError("BEATS_BASELINE requires a baseline RMSE")
        challenger_rmse = _rmse(_as_array(context.challenger_errors))
        threshold = self._max_ratio * context.baseline_rmse
        return GateResult(
            name=self.name,
            passed=_at_most(challenger_rmse, threshold),
            observed=challenger_rmse,
            threshold=float(threshold),
            detail=(
                f"challenger RMSE {challenger_rmse:.3f} must be <= "
                f"{self._max_ratio:.2f} x baseline {context.baseline_rmse:.3f}"
            ),
        )


class ImprovementGate:
    """G4: the challenger must improve on the champion's RMSE by a minimum fraction."""

    name = GATE_IMPROVEMENT

    def __init__(self, min_improvement: float = 0.05) -> None:
        if not 0.0 <= min_improvement <= 1.0:
            raise ValueError(f"min_improvement must be in [0, 1], got {min_improvement}")
        self._min_improvement = min_improvement

    def evaluate(self, context: GateContext) -> GateResult:
        if context.champion_errors is None:
            raise GateEvaluationError("IMPROVEMENT requires champion errors")
        champion_rmse = _rmse(_as_array(context.champion_errors))
        challenger_rmse = _rmse(_as_array(context.challenger_errors))
        if champion_rmse == 0.0:
            improvement = 1.0 if challenger_rmse == 0.0 else -1.0
        else:
            improvement = (champion_rmse - challenger_rmse) / champion_rmse
        return GateResult(
            name=self.name,
            passed=_at_least(improvement, self._min_improvement),
            observed=improvement,
            threshold=self._min_improvement,
            detail=(
                f"champion RMSE {champion_rmse:.3f} -> challenger {challenger_rmse:.3f} "
                f"({improvement:+.2%})"
            ),
        )


class SignificanceGate:
    """G5: the day-block bootstrap must put the paired improvement strictly above zero."""

    name = GATE_SIGNIFICANCE

    def __init__(
        self,
        confidence: float = 0.95,
        resamples: int = 2000,
        seed: int = 42,
    ) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if resamples <= 0:
            raise ValueError(f"resamples must be positive, got {resamples}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        self._confidence = confidence
        self._resamples = resamples
        self._seed = seed

    def evaluate(self, context: GateContext) -> GateResult:
        if context.champion_errors is None:
            raise GateEvaluationError("SIGNIFICANCE requires champion errors")
        champion = _as_array(context.champion_errors)
        challenger = _as_array(context.challenger_errors)
        lower, upper = day_block_bootstrap_ci(
            context.timestamps,
            champion**2,
            challenger**2,
            resamples=self._resamples,
            seed=self._seed,
            confidence=self._confidence,
        )
        return GateResult(
            name=self.name,
            passed=lower > _TOLERANCE,
            observed=float(lower),
            threshold=0.0,
            detail=(
                f"{self._confidence:.0%} CI of mean squared-error improvement: "
                f"[{lower:.3f}, {upper:.3f}]"
            ),
        )


class SegmentRegressionGate:
    """G6: no segment may get worse than the champion by more than the allowed ratio."""

    name = GATE_SEGMENT

    def __init__(self, max_regression: float = 0.10) -> None:
        if max_regression < 0.0:
            raise ValueError(f"max_regression must be non-negative, got {max_regression}")
        self._max_regression = max_regression

    def evaluate(self, context: GateContext) -> GateResult:
        if context.champion_errors is None:
            raise GateEvaluationError("SEGMENT requires champion errors")
        if context.segments is None:
            raise GateEvaluationError(f"SEGMENT requires {SEGMENT_COLUMN} segment values")
        champion_errors = _as_array(context.champion_errors)
        challenger_errors = _as_array(context.challenger_errors)
        segments = np.asarray(context.segments)

        regressions: list[tuple[str, float, float, float]] = []
        for label in np.unique(segments):
            mask = segments == label
            champion_rmse = _rmse(champion_errors[mask])
            challenger_rmse = _rmse(challenger_errors[mask])
            if champion_rmse == 0.0:
                regression = 0.0 if challenger_rmse == 0.0 else math.inf
            else:
                regression = (challenger_rmse - champion_rmse) / champion_rmse
            regressions.append((str(label), regression, champion_rmse, challenger_rmse))

        worst = max(regressions, key=lambda entry: entry[1])
        detail = "; ".join(
            f"{SEGMENT_COLUMN}={label}: {regression:+.1%} "
            f"(champion {champion_rmse:.3f}, challenger {challenger_rmse:.3f})"
            for label, regression, champion_rmse, challenger_rmse in regressions
        )
        return GateResult(
            name=self.name,
            passed=_at_most(worst[1], self._max_regression),
            observed=float(worst[1]),
            threshold=self._max_regression,
            detail=detail,
        )


__all__ = [
    "GATE_ABS_WAPE",
    "GATE_BEATS_BASELINE",
    "GATE_DATA_VALID",
    "GATE_IMPROVEMENT",
    "GATE_NAMES",
    "GATE_SEGMENT",
    "GATE_SIGNIFICANCE",
    "AbsoluteWapeGate",
    "BeatsBaselineGate",
    "DataValidGate",
    "Gate",
    "GateContext",
    "GateReport",
    "GateResult",
    "ImprovementGate",
    "SegmentRegressionGate",
    "SignificanceGate",
]
