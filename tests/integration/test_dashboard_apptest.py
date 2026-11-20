"""T11: every Streamlit page runs through AppTest, including empty and broken states (S17)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine

APP = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mlops_lifecycle_lab"
    / "entrypoints"
    / "dashboard"
    / "app.py"
)
PAGES = ("Overview", "Monitoring", "Registry", "Lifecycle", "Drill")


@pytest.fixture(scope="module")
def drilled_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dashboard-apptest")
    workspace = root / "ws"
    DrillEngine().run("drift", weeks=16, workspace=workspace, fast=True)
    return workspace


def _app(monkeypatch: pytest.MonkeyPatch, workspace: Path, page: str) -> AppTest:
    monkeypatch.setenv("MLL_WORKSPACE", str(workspace))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    if page != "Overview":
        at.sidebar.radio[0].set_value(page).run()
    return at


def _assert_heading(at: AppTest, page: str) -> None:
    assert [title.value for title in at.title] == [page]
    assert not at.exception


def test_overview_renders_metrics(
    monkeypatch: pytest.MonkeyPatch, drilled_workspace: Path
) -> None:
    at = _app(monkeypatch, drilled_workspace, "Overview")
    _assert_heading(at, "Overview")
    assert len(at.metric) >= 4


def test_monitoring_renders_plots(
    monkeypatch: pytest.MonkeyPatch, drilled_workspace: Path
) -> None:
    at = _app(monkeypatch, drilled_workspace, "Monitoring")
    _assert_heading(at, "Monitoring")
    assert at.get("plotly_chart") or at.dataframe


def test_registry_renders_the_version_table(
    monkeypatch: pytest.MonkeyPatch, drilled_workspace: Path
) -> None:
    at = _app(monkeypatch, drilled_workspace, "Registry")
    _assert_heading(at, "Registry")
    assert at.dataframe


def test_lifecycle_renders_the_event_and_ledger_tables(
    monkeypatch: pytest.MonkeyPatch, drilled_workspace: Path
) -> None:
    at = _app(monkeypatch, drilled_workspace, "Lifecycle")
    _assert_heading(at, "Lifecycle")
    assert len(at.dataframe) >= 2


def test_drill_renders_the_timeline(
    monkeypatch: pytest.MonkeyPatch, drilled_workspace: Path
) -> None:
    at = _app(monkeypatch, drilled_workspace, "Drill")
    _assert_heading(at, "Drill")
    assert at.selectbox
    assert at.get("plotly_chart") or at.dataframe


@pytest.mark.parametrize("page", PAGES)
def test_empty_workspace_shows_the_quickstart_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, page: str
) -> None:
    at = _app(monkeypatch, tmp_path / "empty", page)
    assert not at.exception
    assert any("mll quickstart" in element.value for element in at.info)


def test_corrupted_database_shows_an_error_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "broken"
    workspace.mkdir()
    (workspace / "mll.db").write_bytes(b"this is not a sqlite database")
    at = _app(monkeypatch, workspace, "Overview")
    assert not at.exception
    assert at.error
