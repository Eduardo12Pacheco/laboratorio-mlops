"""Typed settings for the lab.

Every setting is read from an ``MLL_*`` environment variable (case-insensitive), e.g.
``MLL_WORKSPACE`` or ``MLL_SLO_WAPE``. Defaults are the production values from the plan;
tests and drills override them explicitly.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PRIVACY_DEFAULT_ENV_VARS: dict[str, str] = {
    "MLFLOW_DISABLE_TELEMETRY": "true",
    "DO_NOT_TRACK": "1",
    "ITERATIVE_DO_NOT_TRACK": "1",
    "EVIDENTLY_DISABLE_TELEMETRY": "1",
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}
"""Environment switches that keep every bundled tool fully offline (D22)."""


class Settings(BaseSettings):
    """Runtime configuration shared by the CLI, API, dashboard and drills."""

    model_config = SettingsConfigDict(
        env_prefix="MLL_",
        case_sensitive=False,
        extra="ignore",
    )

    workspace: Path = Path(".mll")
    mlflow_tracking_uri: str | None = None
    mlflow_artifact_root: Path | None = None
    model_name: str = "bike-demand"
    experiment_name: str = "bike-demand"
    seed: int = Field(default=42, ge=0)

    slo_wape: float = Field(default=0.30, gt=0, le=1)
    abs_wape_max: float = Field(default=0.30, gt=0, le=1)
    baseline_max_ratio: float = Field(default=0.85, gt=0)
    improvement_min: float = Field(default=0.05, ge=0, le=1)
    segment_max_regression: float = Field(default=0.10, ge=0)
    bootstrap_resamples: int = Field(default=2000, gt=0)
    window_days: int = Field(default=28, gt=0)
    min_labeled_rows: int = Field(default=300, gt=0)
    cooldown_days: int = Field(default=28, gt=0)
    holdout_days: int = Field(default=14, gt=0)
    probation_days: int = Field(default=21, gt=0)
    probation_min_rows: int = Field(default=150, gt=0)

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=14080, gt=0, le=65535)
    api_url: str = "http://127.0.0.1:14080"
    admin_api_key: SecretStr | None = None
    reload_interval_seconds: float = Field(default=5.0, gt=0)

    replay_start: date = date(2012, 1, 1)
    replay_interval_seconds: float = Field(default=20.0, gt=0)
    replay_weeks: int = Field(default=52, gt=0)
    serving_mode: Literal["inprocess", "http"] = "inprocess"
    fast: bool = False

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    @field_validator("admin_api_key", mode="before")
    @classmethod
    def _empty_admin_key_is_unset(cls, value: object) -> object:
        """Treat ``MLL_ADMIN_API_KEY=`` as absent instead of an empty required key."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def effective_tracking_uri(self) -> str:
        """MLflow tracking URI: explicit override, else the workspace SQLite file."""
        if self.mlflow_tracking_uri is not None:
            return self.mlflow_tracking_uri
        return f"sqlite:///{(self.workspace / 'mlflow.db').as_posix()}"

    @property
    def effective_artifact_root(self) -> Path:
        """Directory for MLflow artifacts: explicit override, else under the workspace."""
        if self.mlflow_artifact_root is not None:
            return self.mlflow_artifact_root
        return self.workspace / "mlartifacts"


def apply_privacy_defaults() -> None:
    """Disable telemetry for every bundled tool without overriding explicit user values."""
    for key, value in PRIVACY_DEFAULT_ENV_VARS.items():
        os.environ.setdefault(key, value)
