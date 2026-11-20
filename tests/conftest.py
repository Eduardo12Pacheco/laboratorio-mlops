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
