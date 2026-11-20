from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults

PRIVACY_DEFAULTS = {
    "MLFLOW_DISABLE_TELEMETRY": "true",
    "DO_NOT_TRACK": "1",
    "ITERATIVE_DO_NOT_TRACK": "1",
    "EVIDENTLY_DISABLE_TELEMETRY": "1",
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}


def test_settings_defaults() -> None:
    settings = Settings()

    assert settings.workspace == Path(".mll")
    assert settings.mlflow_tracking_uri is None
    assert settings.mlflow_artifact_root is None
    assert settings.model_name == "bike-demand"
    assert settings.experiment_name == "bike-demand"
    assert settings.seed == 42
    assert settings.slo_wape == pytest.approx(0.30)
    assert settings.abs_wape_max == pytest.approx(0.30)
    assert settings.baseline_max_ratio == pytest.approx(0.85)
    assert settings.improvement_min == pytest.approx(0.05)
    assert settings.segment_max_regression == pytest.approx(0.10)
    assert settings.bootstrap_resamples == 2000
    assert settings.window_days == 28
    assert settings.min_labeled_rows == 300
    assert settings.cooldown_days == 28
    assert settings.holdout_days == 14
    assert settings.probation_days == 21
    assert settings.probation_min_rows == 150
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 14080
    assert settings.api_url == "http://127.0.0.1:14080"
    assert settings.admin_api_key is None
    assert settings.reload_interval_seconds == pytest.approx(5.0)
    assert settings.replay_start == date(2012, 1, 1)
    assert settings.replay_interval_seconds == pytest.approx(20.0)
    assert settings.replay_weeks == 52
    assert settings.log_level == "INFO"
    assert settings.log_format == "console"


def test_env_override_for_slo_wape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLL_SLO_WAPE", "0.25")

    assert Settings().slo_wape == pytest.approx(0.25)


def test_env_override_for_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MLL_WORKSPACE", str(tmp_path))

    assert Settings().workspace == tmp_path


def test_invalid_slo_wape_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLL_SLO_WAPE", "2")

    with pytest.raises(ValidationError):
        Settings()


def test_non_positive_counts_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLL_WINDOW_DAYS", "0")

    with pytest.raises(ValidationError):
        Settings()


def test_effective_tracking_uri_defaults_to_workspace_sqlite(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path)

    assert settings.effective_tracking_uri == f"sqlite:///{tmp_path.as_posix()}/mlflow.db"


def test_explicit_tracking_uri_is_preserved() -> None:
    settings = Settings(mlflow_tracking_uri="http://127.0.0.1:14000")

    assert settings.effective_tracking_uri == "http://127.0.0.1:14000"


def test_effective_artifact_root_defaults_to_workspace() -> None:
    settings = Settings(workspace=Path("runtime"))

    assert settings.effective_artifact_root == Path("runtime/mlartifacts")


def test_explicit_artifact_root_is_preserved(tmp_path: Path) -> None:
    settings = Settings(mlflow_artifact_root=tmp_path)

    assert settings.effective_artifact_root == tmp_path


def test_admin_api_key_is_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLL_ADMIN_API_KEY", "super-secret-value")
    settings = Settings()

    assert isinstance(settings.admin_api_key, SecretStr)
    assert settings.admin_api_key.get_secret_value() == "super-secret-value"
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings)


def test_privacy_defaults_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in PRIVACY_DEFAULTS:
        monkeypatch.delenv(key, raising=False)

    apply_privacy_defaults()

    for key, value in PRIVACY_DEFAULTS.items():
        assert os.environ[key] == value


def test_privacy_defaults_do_not_override_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in PRIVACY_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "false")

    apply_privacy_defaults()

    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "false"
    assert os.environ["DO_NOT_TRACK"] == "1"
