"""`mll serve` and `mll score` are registered and expose their documented options."""

from __future__ import annotations

from typer.testing import CliRunner

from mlops_lifecycle_lab.entrypoints.cli.app import app

runner = CliRunner()


def test_serve_help_lists_host_port_and_workspace() -> None:
    result = runner.invoke(app, ["serve", "--help"], terminal_width=200)
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--workspace" in result.output


def test_score_help_lists_input_output_source_and_workspace() -> None:
    result = runner.invoke(app, ["score", "--help"], terminal_width=200)
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--source" in result.output
    assert "--workspace" in result.output
