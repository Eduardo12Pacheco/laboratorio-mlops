"""Shared pytest configuration for the mlops-lifecycle-lab test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.fakes.mlflow_workspace import MlflowWorkspace


@pytest.fixture(autouse=True)
def clean_mll_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never inherit MLL_* settings from the developer shell."""
    for key in list(os.environ):
        if key.startswith("MLL_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def mlflow_workspace(tmp_path_factory: pytest.TempPathFactory) -> MlflowWorkspace:
    """One SQLite tracking store for the whole session (per the T6 isolation rule)."""
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    root: Path = tmp_path_factory.mktemp("mlflow")
    return MlflowWorkspace(uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}", root=root)


@pytest.fixture(scope="session")
def drift_drill_result(tmp_path_factory: pytest.TempPathFactory) -> object:
    """One verified 52-week `drift` drill shared by the slow e2e and quality tests."""
    from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine

    root = tmp_path_factory.mktemp("drill-drift")
    return DrillEngine().run(
        "drift",
        workspace=root / "workspace",
        report_dir=root / "reports",
        fast=True,
    )


@pytest.fixture(scope="session")
def scenario_drill_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """The three fault scenarios, run once for the slow e2e suite."""
    from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine

    root = tmp_path_factory.mktemp("drill-scenarios")
    results: dict[str, object] = {}
    for name in ("gate-reject", "bad-batch", "rollback"):
        results[name] = DrillEngine().run(
            name,
            workspace=root / name,
            report_dir=root / "reports",
            fast=True,
        )
    return results
