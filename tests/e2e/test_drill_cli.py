"""CLI surface of the drill: run, check, registry table and monitoring snapshot."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from mlops_lifecycle_lab.entrypoints.cli.app import app

pytestmark = pytest.mark.slow

runner = CliRunner()


def invoke(args: list[str]) -> Result:
    """Invoke the CLI with a wide, colour-free terminal for stable assertions."""
    return runner.invoke(
        app,
        args,
        env={"NO_COLOR": "1", "TERM": "dumb"},
        terminal_width=200,
    )


def combined(result: Result) -> str:
    return result.output + result.stderr


def test_drill_check_and_registry_list_from_a_drill_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"

    drill = invoke(
        [
            "drill",
            "run",
            "--scenario",
            "drift",
            "--weeks",
            "16",
            "--workspace",
            str(workspace),
            "--fast",
            "--check",
        ]
    )

    assert drill.exit_code == 0, combined(drill)
    assert "promotions=1 rejections=0 rollbacks=0 quarantined=0" in drill.output
    assert "as-served RMSE" in drill.output
    assert "golden OK: drift" in drill.output
    assert (workspace / "reports" / "drill-drift.json").exists()

    listing = invoke(["registry", "list", "--workspace", str(workspace)])

    assert listing.exit_code == 0, combined(listing)
    header = listing.output.splitlines()[0]
    for column in ("version", "stage", "rmse", "wape", "train_window", "aliases"):
        assert column in header
    assert "v1" in listing.output
    assert "v2" in listing.output
    assert "CHAMPION" in listing.output

    shown = invoke(["registry", "show", "2", "--workspace", str(workspace)])
    assert shown.exit_code == 0, combined(shown)
    assert "stage: CHAMPION" in shown.output
    assert "train_window:" in shown.output

    monitored = invoke(["monitor", "snapshot", "--at", "2012-04-29", "--workspace", str(workspace)])
    assert monitored.exit_code == 0, combined(monitored)
    assert "champion: v2" in monitored.output
    assert "wape:" in monitored.output


def test_registry_list_is_empty_before_bootstrap(tmp_path: Path) -> None:
    result = invoke(["registry", "list", "--workspace", str(tmp_path / "empty")])

    assert result.exit_code == 0, combined(result)
    assert "version" in result.output
    assert "stage" in result.output


def test_drill_rejects_unknown_scenarios_and_weeks(tmp_path: Path) -> None:
    unknown = invoke(["drill", "run", "--scenario", "nope", "--workspace", str(tmp_path)])
    assert unknown.exit_code == 2, combined(unknown)
    assert "unknown scenario" in combined(unknown)

    zero = invoke(
        ["drill", "run", "--scenario", "drift", "--weeks", "0", "--workspace", str(tmp_path)]
    )
    assert zero.exit_code != 0, combined(zero)
