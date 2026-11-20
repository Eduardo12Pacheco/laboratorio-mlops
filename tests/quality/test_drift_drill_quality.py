"""Quality gate for the committed drift drill against S15 numbers."""

from __future__ import annotations

import pytest

from mlops_lifecycle_lab.domain.events import EventType
from mlops_lifecycle_lab.entrypoints.drill.report import DrillResult

pytestmark = pytest.mark.slow


def test_drift_drill_meets_s15_quality_tolerances(drift_drill_result: object) -> None:
    assert isinstance(drift_drill_result, DrillResult)
    result = drift_drill_result
    summary = result.summary

    # As-served quality vs the never-retrained v1 counterfactual (S15).
    assert summary.as_served.rmse == pytest.approx(70.72, rel=0.05)
    assert summary.counterfactual_frozen_v1.rmse == pytest.approx(109.92, rel=0.05)
    assert summary.as_served.rmse <= 0.75 * summary.counterfactual_frozen_v1.rmse
    assert summary.as_served.rmse < summary.counterfactual_frozen_v1.rmse
    assert summary.as_served.wape <= 0.22
    assert summary.as_served.wape < summary.counterfactual_frozen_v1.wape

    # The automation ran exactly once before the final rejection, and v2 passed probation.
    assert summary.promotions == 1
    assert summary.rejections == 1
    assert summary.rollbacks == 0
    assert summary.quarantined_batches == 0
    assert summary.versions == 3
    assert result.first_retrain_tick == 13

    promoted = next(
        event for event in result.events() if event.type is EventType.CHALLENGER_PROMOTED
    )
    assert promoted.tick == 13
    assert promoted.payload["previous_champion"] == 1
    approved = next(
        report
        for report in result.gate_reports
        if report.tick == 13 and report.decision == "APPROVED"
    )
    assert all(gate["passed"] for gate in approved.gates)
    improvement = next(gate for gate in approved.gates if gate["name"] == "IMPROVEMENT")
    assert float(improvement["observed"]) >= 0.50

    passed = [event for event in result.events() if event.type is EventType.PROBATION_PASSED]
    assert [event.tick for event in passed][-1] == 16
    assert result.ticks[-1].events[-1].type is EventType.CHALLENGER_REJECTED
