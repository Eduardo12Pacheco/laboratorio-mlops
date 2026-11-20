"""Session-scoped MLflow workspace shared by the integration and contract suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MlflowWorkspace:
    """A temporary MLflow tracking URI and artifact root."""

    uri: str
    root: Path
