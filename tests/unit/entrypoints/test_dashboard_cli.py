"""T11: `mll dashboard` launches the packaged Streamlit app for the chosen workspace."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import typer

from mlops_lifecycle_lab.entrypoints.cli.commands.dashboard import (
    dashboard,
    dashboard_command,
)
from mlops_lifecycle_lab.entrypoints.dashboard import app as dashboard_app


def test_dashboard_command_runs_the_packaged_app() -> None:
    argv = dashboard_command(host="127.0.0.1", port=14511)
    assert argv[:3] == [sys.executable, "-m", "streamlit"]
    assert argv[3] == "run"
    assert argv[4] == str(Path(dashboard_app.__file__).resolve())
    assert "--server.port" in argv
    assert argv[argv.index("--server.port") + 1] == "14511"
    assert "--theme.light.primaryColor" in argv
    assert "--theme.dark.primaryColor" in argv
    assert "--theme.background" not in argv  # let Streamlit's light/dark auto theme win


def test_dashboard_exports_the_workspace_and_propagates_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["env"] = env
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    dashboard(host="127.0.0.1", port=14511, workspace=tmp_path)
    assert captured["env"]["MLL_WORKSPACE"] == str(tmp_path)
    assert captured["argv"] == dashboard_command(host="127.0.0.1", port=14511)


def test_dashboard_propagates_a_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, env: subprocess.CompletedProcess(argv, 3),
    )
    with pytest.raises(typer.Exit) as error:
        dashboard(host="127.0.0.1", port=14511, workspace=tmp_path)
    assert error.value.exit_code == 1
