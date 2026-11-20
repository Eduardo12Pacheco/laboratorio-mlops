"""``mll mlflow-ui``: launch the local MLflow UI on the documented host port (T10, L4)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner, Result

from mlops_lifecycle_lab.config import Settings
from mlops_lifecycle_lab.entrypoints.cli.app import app
from mlops_lifecycle_lab.entrypoints.cli.commands import mlflow_ui as ui_module

runner = CliRunner()


def invoke(args: list[str]) -> Result:
    return runner.invoke(app, args, env={"NO_COLOR": "1", "TERM": "dumb"}, terminal_width=200)


def combined(result: Result) -> str:
    return result.output + result.stderr


def test_ui_command_uses_the_workspace_tracking_uri_and_port(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path / "ws")

    command = ui_module.ui_command(settings.effective_tracking_uri, port=14999)

    assert command[:3] == [sys.executable, "-m", "mlflow"]
    assert command[3] == "ui"
    assert "--backend-store-uri" in command
    uri = command[command.index("--backend-store-uri") + 1]
    assert uri == settings.effective_tracking_uri
    assert uri.startswith("sqlite:///")
    assert command[command.index("--port") + 1] == "14999"


def test_mlflow_ui_cli_runs_the_ui_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str]) -> SimpleNamespace:
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ui_module.subprocess, "run", fake_run)

    result = invoke(["mlflow-ui", "--port", "14999", "--workspace", str(tmp_path / "ws")])

    assert result.exit_code == 0, combined(result)
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--port") + 1] == "14999"


def test_mlflow_ui_cli_propagates_a_failed_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ui_module.subprocess,
        "run",
        lambda _command: SimpleNamespace(returncode=1),
    )

    result = invoke(["mlflow-ui", "--workspace", str(tmp_path / "ws")])

    assert result.exit_code == 1, combined(result)
    assert "mlflow ui exited with code 1" in combined(result)


def test_mlflow_ui_cli_handles_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def interrupted(_command: list[str]) -> subprocess.CompletedProcess[object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(ui_module.subprocess, "run", interrupted)

    result = invoke(["mlflow-ui", "--workspace", str(tmp_path / "ws")])

    assert result.exit_code == 0, combined(result)
