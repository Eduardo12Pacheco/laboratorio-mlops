"""``mll quickstart`` with the README default workspace and with an explicit absolute one (S5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from mlops_lifecycle_lab.entrypoints.cli.app import app

pytestmark = pytest.mark.slow

runner = CliRunner()


def invoke(args: list[str]) -> Result:
    """Invoke the CLI with a wide, colour-free terminal for stable assertions."""
    return runner.invoke(app, args, env={"NO_COLOR": "1", "TERM": "dumb"}, terminal_width=200)


def combined(result: Result) -> str:
    return result.output + result.stderr


def test_quickstart_with_the_default_relative_workspace_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    result = invoke(["quickstart"])

    assert result.exit_code == 0, combined(result)
    assert "champion: v2" in result.output
    assert (tmp_path / ".mll" / "mll.db").exists()


def test_quickstart_accepts_an_explicit_absolute_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "explicit"

    result = invoke(["quickstart", "--workspace", str(workspace)])

    assert result.exit_code == 0, combined(result)
    assert "champion: v2" in result.output
    assert (workspace / "mll.db").exists()
