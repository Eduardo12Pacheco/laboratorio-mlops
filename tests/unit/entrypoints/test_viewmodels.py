"""T11: dashboard view models over a real 16-week fast drill workspace (S17)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mlops_lifecycle_lab.bootstrap.container import Container, build_container
from mlops_lifecycle_lab.config import Settings
from mlops_lifecycle_lab.domain.events import EventType
from mlops_lifecycle_lab.entrypoints.dashboard.viewmodels import (
    ViewError,
    drift_matrix,
    drill_reports,
    drill_state_at,
    events_table,
    gate_report,
    ledger_table,
    overview,
    performance_series,
    registry_table,
)
from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine

GATE_NAMES = {"DATA_VALID", "ABS_WAPE", "BEATS_BASELINE", "IMPROVEMENT", "SIGNIFICANCE", "SEGMENT"}


@pytest.fixture(scope="module")
def drilled_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dashboard-workspace")
    workspace = root / "ws"
    DrillEngine().run("drift", weeks=16, workspace=workspace, fast=True)
    return workspace


@pytest.fixture(scope="module")
def container(drilled_workspace: Path) -> Iterator[Container]:
    instance = build_container(Settings(workspace=drilled_workspace, fast=True))
    yield instance
    instance.close()


def test_overview_reports_the_promoted_challenger(container: Container) -> None:
    view = overview(container)
    assert view.champion_version == 2
    assert view.status in {"healthy", "degraded"}
    labels = {tile.label for tile in view.kpis}
    assert {"Champion", "WAPE 28d", "RMSE 28d", "Labeled rows"} <= labels
    assert len(view.recent_events) == 5
    assert "SNAPSHOT_COMPUTED" in {event.type for event in view.recent_events}


def test_performance_series_covers_every_tick_and_both_champions(container: Container) -> None:
    view = performance_series(container)
    assert len(view.points) == 16
    assert view.slo == pytest.approx(0.30)
    assert view.final_champion == 2
    assert {band.version for band in view.bands} == {1, 2}
    assert view.actual_vs_predicted
    assert all(row.predicted >= 0 for row in view.actual_vs_predicted)
    assert view.bias is not None


def test_drift_matrix_exposes_every_monitored_feature(container: Container) -> None:
    view = drift_matrix(container)
    assert view.features == ("atemp", "hum", "season", "temp", "weathersit", "windspeed")
    assert len(view.ticks) == 16
    assert view.values["temp"][13] >= 0.0
    assert view.thresholds == (0.1, 0.2)


def test_registry_table_marks_aliases_and_exposes_the_gate_report(container: Container) -> None:
    view = registry_table(container)
    assert [row.version for row in view.rows] == [1, 2]
    by_version = {row.version: row for row in view.rows}
    assert by_version[2].aliases == ("champion",)
    assert by_version[1].aliases == ("previous-champion",)
    assert by_version[2].stage == "CHAMPION"
    assert by_version[2].fingerprint
    assert view.mlflow_url.endswith("14000")

    report = gate_report(container, 2)
    assert report.version == 2
    assert report.decision == "APPROVED"
    assert {gate.name for gate in report.gates} == GATE_NAMES
    assert all(gate.passed for gate in report.gates)


def test_events_table_filters_and_orders_the_audit_log(container: Container) -> None:
    rows = events_table(
        container, types=[EventType.CHALLENGER_PROMOTED, EventType.PROBATION_PASSED]
    )
    assert rows
    assert {row.type for row in rows} == {"CHALLENGER_PROMOTED", "PROBATION_PASSED"}
    assert rows[0].ts >= rows[-1].ts

    all_rows = events_table(container, limit=3)
    assert len(all_rows) == 3


def test_ledger_table_reports_durations_and_failures(container: Container) -> None:
    rows = ledger_table(container)
    assert len(rows) == 16
    assert all(row.status == "SUCCEEDED" for row in rows)
    assert all(row.duration_seconds >= 0 for row in rows)
    assert all(row.failed_steps == 0 for row in rows)
    assert rows[0].started_at >= rows[-1].started_at


def test_drill_reports_can_be_scrubbed_by_tick(drilled_workspace: Path) -> None:
    reports = drill_reports(drilled_workspace / "reports")
    assert len(reports) == 1
    report = reports[0]
    assert report.scenario == "drift"
    assert report.weeks == 16
    assert report.summary["promotions"] == 1

    early = drill_state_at(report, 1)
    assert early.champion_version == 1
    promoted = drill_state_at(report, 13)
    assert promoted.champion_version == 2
    assert "CHALLENGER_PROMOTED" in {event.type for event in promoted.events}


def test_empty_workspace_raises_a_view_error_with_the_quickstart_hint(
    tmp_path: Path,
) -> None:
    empty = build_container(Settings(workspace=tmp_path / "empty", fast=True))
    try:
        with pytest.raises(ViewError, match="no lifecycle data") as error:
            overview(empty)
        assert "mll quickstart" in error.value.hint
        with pytest.raises(ViewError, match="No model versions") as versions_error:
            registry_table(empty)
        assert "mll quickstart" in versions_error.value.hint
    finally:
        empty.close()
