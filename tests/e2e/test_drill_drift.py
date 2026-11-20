"""End-to-end drift drill against the committed golden (S15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlops_lifecycle_lab.domain.events import EventType
from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine
from mlops_lifecycle_lab.entrypoints.drill.expectations import compare, load_golden
from mlops_lifecycle_lab.entrypoints.drill.report import DrillResult

pytestmark = pytest.mark.slow

GOLDEN = Path("tests/golden/drill_drift.json")


def _result(fixture: object) -> DrillResult:
    assert isinstance(fixture, DrillResult)
    return fixture


def _events(result: DrillResult, event_type: EventType) -> list[object]:
    return [event for event in result.events() if event.type is event_type]


def test_drift_matches_the_golden_event_sequence_and_metrics(
    drift_drill_result: object,
) -> None:
    result = _result(drift_drill_result)

    assert compare(result, load_golden(GOLDEN)) == []


def test_drift_summary_reproduces_s15(drift_drill_result: object) -> None:
    result = _result(drift_drill_result)
    summary = result.summary

    assert summary.as_served.rmse == pytest.approx(70.72, rel=0.05)
    assert summary.counterfactual_frozen_v1.rmse == pytest.approx(109.92, rel=0.05)
    assert summary.as_served.rmse <= 0.75 * summary.counterfactual_frozen_v1.rmse
    assert summary.as_served.wape <= 0.22
    assert summary.promotions == 1
    assert summary.rejections == 1
    assert summary.rollbacks == 0
    assert summary.quarantined_batches == 0
    assert result.first_retrain_tick == 13
    assert result.ticks[12].tick_end.isoformat() == "2012-04-01T00:00:00"

    promoted = _events(result, EventType.CHALLENGER_PROMOTED)
    assert len(promoted) == 1
    assert promoted[0].tick == 13  # type: ignore[attr-defined]
    approved = [
        report
        for report in result.gate_reports
        if report.tick == 13 and report.decision == "APPROVED"
    ]
    assert len(approved) == 1
    assert approved[0].gates
    assert all(gate["passed"] for gate in approved[0].gates)
    improvement = next(gate for gate in approved[0].gates if gate["name"] == "IMPROVEMENT")
    assert float(improvement["observed"]) >= 0.50

    passed = _events(result, EventType.PROBATION_PASSED)
    # v1 finishes its bootstrap probation at tick 3; v2 passes three ticks after promotion.
    assert [event.tick for event in passed] == [3, 16]  # type: ignore[attr-defined]
    last = result.ticks[-1]
    assert last.tick == 52
    assert [event.type for event in last.events][-1] is EventType.CHALLENGER_REJECTED


def test_drift_report_files_are_reproducible(
    drift_drill_result: object,
    tmp_path: Path,
) -> None:
    result = _result(drift_drill_result)
    json_path, markdown_path = result.write(tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema"] == "drill-report/v1"
    assert "promotions=1" in markdown_path.read_text(encoding="utf-8")

    second = DrillEngine().run(
        "drift",
        workspace=tmp_path / "second-workspace",
        report_dir=tmp_path / "second-reports",
        fast=True,
    )
    second_path, _ = second.write(tmp_path / "again")
    assert second_path.read_bytes() == json_path.read_bytes()


def test_drift_run_reports_the_expected_actions(drift_drill_result: object) -> None:
    result = _result(drift_drill_result)

    assert isinstance(result.ticks[0].events, tuple)
    assert [tick.snapshot is not None for tick in result.ticks].count(True) == 52
    assert result.ticks[12].snapshot is not None
    snapshot = result.ticks[12].snapshot
    assert snapshot is not None
    assert snapshot.numeric_drift_share >= 0.5
