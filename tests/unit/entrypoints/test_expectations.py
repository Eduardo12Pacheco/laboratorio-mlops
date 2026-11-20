"""Golden expectations for the drill: make, load and compare with tolerances."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.entrypoints.drill.expectations import (
    GOLDEN_SCHEMA,
    compare,
    load_golden,
    make_golden,
)
from mlops_lifecycle_lab.entrypoints.drill.report import (
    DrillResult,
    DrillSummary,
    MetricsSummary,
    TickRecord,
)


def _event(event_type: EventType, tick: int) -> LifecycleEvent:
    return LifecycleEvent(
        ts=datetime(2012, 1, 1),
        type=event_type,
        tick=tick,
        model_version=1,
        payload={},
    )


def _result(weeks: int = 3, retrain_tick: int | None = 2) -> DrillResult:
    ticks = []
    for number in range(1, weeks + 1):
        types = [EventType.BATCH_VALIDATED, EventType.SNAPSHOT_COMPUTED]
        if number == retrain_tick:
            types.append(EventType.RETRAIN_TRIGGERED)
        ticks.append(
            TickRecord(
                tick=number,
                tick_end=datetime(2012, 1, 1),
                champion_version=1,
                snapshot=None,
                validation_verdict="PASS",
                action="SCORED",
                events=tuple(_event(event_type, number) for event_type in types),
            )
        )
    return DrillResult(
        scenario="drift",
        seed=42,
        weeks=weeks,
        ticks=tuple(ticks),
        gate_reports=(),
        summary=DrillSummary(
            as_served=MetricsSummary(70.0, 45.0, 0.195, 100),
            counterfactual_frozen_v1=MetricsSummary(110.0, 74.0, 0.317, 100),
            promotions=1,
            rejections=1,
            rollbacks=0,
            quarantined_batches=0,
            versions=3,
        ),
    )


def test_make_golden_captures_event_sequences_and_metrics() -> None:
    golden = make_golden(_result())

    assert golden["schema"] == GOLDEN_SCHEMA
    assert golden["scenario"] == "drift"
    assert golden["weeks"] == 3
    assert golden["first_retrain_tick"] == 2
    assert golden["ticks"][1]["events"][-1] == "RETRAIN_TRIGGERED"
    assert golden["metrics"]["as_served"]["rmse"] == 70.0


def test_identical_result_compares_clean() -> None:
    result = _result()

    assert compare(result, make_golden(result)) == []


def test_event_sequence_change_is_reported() -> None:
    result = _result()
    golden = make_golden(result)
    golden["ticks"][0]["events"] = ["BATCH_VALIDATED"]

    mismatches = compare(result, golden)

    assert mismatches
    assert "events" in mismatches[0].path


def test_metric_outside_the_relative_tolerance_is_reported() -> None:
    result = _result()
    golden = make_golden(result)
    golden["metrics"]["as_served"]["rmse"] = 80.0

    mismatches = compare(result, golden)

    assert any("as_served.rmse" in mismatch.path for mismatch in mismatches)


def test_metric_inside_the_relative_tolerance_is_accepted() -> None:
    result = _result()
    golden = make_golden(result)
    golden["metrics"]["as_served"]["rmse"] = 70.0 * 1.04

    assert compare(result, golden) == []


def test_summary_counts_are_compared() -> None:
    result = _result()
    golden = make_golden(result)
    golden["summary"]["promotions"] = 2

    mismatches = compare(result, golden)

    assert any("promotions" in mismatch.path for mismatch in mismatches)


def test_a_retrain_tick_inside_the_slack_shifts_later_comparisons() -> None:
    result = _result(weeks=4, retrain_tick=3)
    golden = make_golden(_result(weeks=4, retrain_tick=2))

    assert compare(result, golden) == []


def test_a_retrain_tick_outside_the_slack_is_reported() -> None:
    result = _result(weeks=4, retrain_tick=4)
    golden = make_golden(_result(weeks=4, retrain_tick=2))

    assert any("retrain" in mismatch.path for mismatch in compare(result, golden))


def test_a_prefix_run_only_compares_the_ticks_it_ran() -> None:
    result = _result(weeks=2, retrain_tick=2)
    golden = make_golden(_result(weeks=4, retrain_tick=2))

    assert compare(result, golden) == []


def test_a_run_longer_than_the_golden_is_reported() -> None:
    result = _result(weeks=5, retrain_tick=2)
    golden = make_golden(_result(weeks=4, retrain_tick=2))

    assert any("weeks" in mismatch.path for mismatch in compare(result, golden))


def test_load_golden_round_trips_and_rejects_a_bad_schema(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(make_golden(_result())), encoding="utf-8")
    assert load_golden(path)["scenario"] == "drift"

    path.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_golden(path)


def test_result_can_be_replaced_for_the_prefix_test() -> None:
    result = _result(weeks=2)
    assert replace(result, weeks=2).ticks == result.ticks
