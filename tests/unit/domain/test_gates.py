"""Promotion gates G1-G6: boundary behaviour, evidence requirements and reports."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlops_lifecycle_lab.domain.errors import GateEvaluationError
from mlops_lifecycle_lab.domain.gates import (
    GATE_NAMES,
    AbsoluteWapeGate,
    BeatsBaselineGate,
    DataValidGate,
    GateContext,
    GateReport,
    GateResult,
    ImprovementGate,
    SegmentRegressionGate,
    SignificanceGate,
)
from mlops_lifecycle_lab.domain.policies import PromotionPolicy


def hourly(count: int) -> list[datetime]:
    start = datetime(2012, 1, 1)
    return [start + timedelta(hours=offset) for offset in range(count)]


def make_context(
    champion_errors: Sequence[float] | None,
    challenger_errors: Sequence[float],
    *,
    actual: float = 1.0,
    segments: Sequence[int] | None = None,
    baseline_rmse: float | None = 100.0,
    training_verdict: str = "PASS",
    is_first_champion: bool = False,
) -> GateContext:
    count = len(challenger_errors)
    return GateContext(
        timestamps=hourly(count),
        actuals=np.full(count, actual),
        champion_errors=None if champion_errors is None else np.asarray(champion_errors),
        challenger_errors=np.asarray(challenger_errors),
        segments=None if segments is None else np.asarray(segments),
        baseline_rmse=baseline_rmse,
        training_verdict=training_verdict,
        is_first_champion=is_first_champion,
    )


def test_gate_names_and_order_are_exactly_the_contract() -> None:
    assert GATE_NAMES == (
        "DATA_VALID",
        "ABS_WAPE",
        "BEATS_BASELINE",
        "IMPROVEMENT",
        "SIGNIFICANCE",
        "SEGMENT",
    )


@pytest.mark.parametrize(
    ("verdict", "passed"),
    [("PASS", True), ("WARN", True), ("FAIL", False)],
)
def test_data_valid_gate_only_rejects_a_failed_training_frame(verdict: str, passed: bool) -> None:
    result = DataValidGate().evaluate(make_context([1.0], [1.0], training_verdict=verdict))

    assert result.name == "DATA_VALID"
    assert result.passed is passed
    assert result.observed == verdict
    assert result.threshold == "not FAIL"


def test_absolute_wape_boundary_of_exactly_point_three_zero_passes() -> None:
    result = AbsoluteWapeGate(0.30).evaluate(make_context([1.0], [30.0], actual=100.0))

    assert result.passed is True
    assert result.observed == pytest.approx(0.30)
    assert result.threshold == 0.30


def test_absolute_wape_of_point_three_zero_zero_one_fails() -> None:
    result = AbsoluteWapeGate(0.30).evaluate(make_context([1.0], [30.01], actual=100.0))

    assert result.passed is False
    assert result.observed == pytest.approx(0.3001)


def test_baseline_ratio_of_exactly_point_eight_five_passes() -> None:
    result = BeatsBaselineGate(0.85).evaluate(make_context([50.0], [85.0], baseline_rmse=100.0))

    assert result.passed is True
    assert result.observed == pytest.approx(85.0)
    assert result.threshold == pytest.approx(85.0)


def test_baseline_ratio_of_point_eight_five_zero_one_fails() -> None:
    result = BeatsBaselineGate(0.85).evaluate(make_context([50.0], [85.01], baseline_rmse=100.0))

    assert result.passed is False


def test_improvement_of_exactly_five_percent_passes() -> None:
    result = ImprovementGate(0.05).evaluate(make_context([100.0], [95.0]))

    assert result.passed is True
    assert result.observed == pytest.approx(0.05)


def test_improvement_of_four_point_nine_nine_percent_fails() -> None:
    result = ImprovementGate(0.05).evaluate(make_context([100.0], [95.01]))

    assert result.passed is False
    assert result.observed == pytest.approx(0.0499, abs=1e-6)


def test_segment_regression_of_exactly_ten_percent_passes() -> None:
    champion = [100.0] * 24 + [50.0] * 24
    challenger = [110.0] * 24 + [45.0] * 24

    result = SegmentRegressionGate(0.10).evaluate(
        make_context(champion, challenger, segments=[0] * 24 + [1] * 24)
    )

    assert result.passed is True
    assert result.observed == pytest.approx(0.10)
    assert result.threshold == 0.10
    assert "workingday=0" in result.detail or "0" in result.detail


def test_segment_regression_of_ten_point_zero_one_percent_fails() -> None:
    champion = [100.0] * 24 + [50.0] * 24
    challenger = [110.01] * 24 + [45.0] * 24

    result = SegmentRegressionGate(0.10).evaluate(
        make_context(champion, challenger, segments=[0] * 24 + [1] * 24)
    )

    assert result.passed is False
    assert result.observed == pytest.approx(0.1001, abs=1e-6)


def test_significance_fails_when_the_interval_lower_bound_is_zero() -> None:
    errors = np.linspace(0.0, 10.0, 48)

    result = SignificanceGate().evaluate(make_context(errors, errors.copy()))

    assert result.passed is False
    assert "CI" in result.detail


def test_significance_passes_for_a_strictly_better_challenger() -> None:
    result = SignificanceGate(resamples=500).evaluate(
        make_context(np.full(48, 100.0), np.full(48, 50.0))
    )

    assert result.passed is True
    assert result.observed > 0.0


def test_significance_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(4)
    champion = rng.uniform(1.0, 10.0, 240)
    challenger = rng.uniform(0.5, 9.0, 240)
    context = make_context(champion, challenger)

    first = SignificanceGate(resamples=300, seed=42).evaluate(context)
    second = SignificanceGate(resamples=300, seed=42).evaluate(context)

    assert first == second


def test_gates_that_need_a_champion_raise_without_one() -> None:
    context = make_context(None, [1.0] * 48, is_first_champion=True)

    for gate in (
        ImprovementGate(0.05),
        SignificanceGate(),
        SegmentRegressionGate(0.10),
    ):
        with pytest.raises(GateEvaluationError):
            gate.evaluate(context)


def test_baseline_gate_requires_a_baseline_rmse() -> None:
    with pytest.raises(GateEvaluationError, match="baseline"):
        BeatsBaselineGate(0.85).evaluate(make_context([1.0], [1.0], baseline_rmse=None))


def test_segment_gate_requires_segment_values() -> None:
    with pytest.raises(GateEvaluationError, match="segment"):
        SegmentRegressionGate(0.10).evaluate(make_context([1.0] * 24, [1.0] * 24))


def test_wape_gate_rejects_a_zero_actual_total() -> None:
    with pytest.raises(GateEvaluationError, match="WAPE"):
        AbsoluteWapeGate(0.30).evaluate(make_context([1.0], [1.0], actual=0.0))


def test_gate_context_validates_array_lengths() -> None:
    with pytest.raises(GateEvaluationError, match="length"):
        GateContext(
            timestamps=hourly(2),
            actuals=np.ones(2),
            challenger_errors=np.ones(3),
        )


def test_gate_context_rejects_empty_evidence() -> None:
    with pytest.raises(GateEvaluationError, match="empty"):
        GateContext(timestamps=[], actuals=[], challenger_errors=[])


def test_the_s16a_synthetic_context_fails_exactly_three_gates() -> None:
    """Champion RMSE 52.9 vs challenger 58.8 with workingday=0 +20.3% (S16a)."""
    segment_rows = 24
    champion_segment_zero = 40.0
    champion_segment_one = math.sqrt(2 * 52.9**2 - champion_segment_zero**2)
    challenger_segment_zero = champion_segment_zero * 1.203
    challenger_segment_one = math.sqrt(2 * 58.8**2 - challenger_segment_zero**2)
    context = make_context(
        [champion_segment_zero] * segment_rows + [champion_segment_one] * segment_rows,
        [challenger_segment_zero] * segment_rows + [challenger_segment_one] * segment_rows,
        actual=250.0,
        segments=[0] * segment_rows + [1] * segment_rows,
        baseline_rmse=149.0,
    )

    assert math.sqrt(np.mean(np.asarray(context.champion_errors) ** 2)) == pytest.approx(52.9)
    assert math.sqrt(np.mean(np.asarray(context.challenger_errors) ** 2)) == pytest.approx(58.8)

    decision = PromotionPolicy().evaluate(context)

    assert decision.approved is False
    assert [result.name for result in decision.report.results] == list(GATE_NAMES)
    assert decision.report.failed_gates == (
        "IMPROVEMENT",
        "SIGNIFICANCE",
        "SEGMENT",
    )
    assert "IMPROVEMENT" in decision.reason


@given(
    st.floats(min_value=0.1, max_value=50.0),
    st.floats(min_value=0.1, max_value=50.0),
    st.floats(min_value=0.1, max_value=200.0),
)
def test_the_policy_evaluates_every_gate_and_never_short_circuits(
    champion_rmse: float, challenger_rmse: float, baseline_rmse: float
) -> None:
    context = make_context(
        np.full(48, champion_rmse),
        np.full(48, challenger_rmse),
        actual=200.0,
        segments=[0] * 24 + [1] * 24,
        baseline_rmse=baseline_rmse,
    )

    decision = PromotionPolicy().evaluate(context)

    assert [result.name for result in decision.report.results] == list(GATE_NAMES)
    assert decision.approved == all(result.passed for result in decision.report.results)
    assert decision.report.approved == decision.approved


def test_gate_report_serialises_and_lists_failed_gates() -> None:
    results = (
        GateResult("DATA_VALID", True, "PASS", "not FAIL", "ok"),
        GateResult("ABS_WAPE", True, 0.2, 0.3, "within budget"),
        GateResult("IMPROVEMENT", False, 0.01, 0.05, "too small"),
    )

    report = GateReport(results=results, approved=False)

    assert report.failed_gates == ("IMPROVEMENT",)
    assert report.to_dict() == {
        "approved": False,
        "results": [
            {
                "name": "DATA_VALID",
                "passed": True,
                "observed": "PASS",
                "threshold": "not FAIL",
                "detail": "ok",
            },
            {
                "name": "ABS_WAPE",
                "passed": True,
                "observed": 0.2,
                "threshold": 0.3,
                "detail": "within budget",
            },
            {
                "name": "IMPROVEMENT",
                "passed": False,
                "observed": 0.01,
                "threshold": 0.05,
                "detail": "too small",
            },
        ],
    }
