"""Session-scoped MLflow workspace shared by the integration and contract suites."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True, slots=True)
class MlflowWorkspace:
    """A temporary MLflow tracking URI and artifact root."""

    uri: str
    root: Path


@pytest.fixture(scope="session")
def mlflow_workspace(tmp_path_factory: pytest.TempPathFactory) -> MlflowWorkspace:
    """One SQLite tracking store for the whole session (per the T6 isolation rule)."""
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    root = tmp_path_factory.mktemp("mlflow")
    return MlflowWorkspace(uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}", root=root)
