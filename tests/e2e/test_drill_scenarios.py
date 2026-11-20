"""End-to-end fault scenarios: gate rejection, quarantine and poisoned rollback (S16)."""

from __future__ import annotations

import pytest

from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.entrypoints.drill.report import DrillResult

pytestmark = pytest.mark.slow


def _result(results: dict[str, object], name: str) -> DrillResult:
    result = results[name]
    assert isinstance(result, DrillResult)
    return result


def _events(result: DrillResult, event_type: EventType) -> list[LifecycleEvent]:
    return [event for event in result.events() if event.type is event_type]


def test_gate_reject_ends_with_the_three_failing_gates(
    scenario_drill_results: dict[str, object],
) -> None:
    result = _result(scenario_drill_results, "gate-reject")

    rejected = _events(result, EventType.CHALLENGER_REJECTED)
    assert len(rejected) == 1
    assert rejected[0].tick == 23
    assert rejected[0].payload["failed_gates"] == [
        "IMPROVEMENT",
        "SIGNIFICANCE",
        "SEGMENT",
    ]
    assert result.ticks[-1].champion_version == 2
    assert result.ticks[-1].action == "SCORED"


def test_bad_batch_quarantines_exactly_the_celsius_week(
    scenario_drill_results: dict[str, object],
) -> None:
    result = _result(scenario_drill_results, "bad-batch")

    quarantined = _events(result, EventType.BATCH_QUARANTINED)
    assert len(quarantined) == 1
    assert quarantined[0].tick == 7
    assert "numeric_range" in quarantined[0].payload["failed_rules"]
    assert result.ticks[6].action == "QUARANTINED"
    assert result.ticks[6].tick_end.isoformat() == "2012-02-19T00:00:00"
    # Nothing was scored that week and the champion was untouched before/after.
    assert EventType.BATCH_SCORED not in {event.type for event in result.ticks[6].events}
    assert result.ticks[5].champion_version == result.ticks[7].champion_version == 1
    assert result.summary.quarantined_batches == 1
    assert result.summary.promotions == 0


def test_rollback_forces_a_poisoned_challenger_then_restores_the_previous_champion(
    scenario_drill_results: dict[str, object],
) -> None:
    result = _result(scenario_drill_results, "rollback")

    forced = _events(result, EventType.PROMOTION_FORCED)
    assert len(forced) == 1
    assert forced[0].tick == 23
    assert forced[0].model_version == 3
    assert forced[0].payload["previous_champion"] == 2

    rolled_back = _events(result, EventType.ROLLBACK_EXECUTED)
    assert len(rolled_back) == 1
    assert rolled_back[0].tick == 24
    assert rolled_back[0].payload["restored_version"] == 2
    assert rolled_back[0].payload["rolled_back_version"] == 3
    assert result.ticks[-1].champion_version == 2
    assert result.summary.rollbacks == 1
