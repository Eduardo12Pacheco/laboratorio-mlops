from __future__ import annotations

from typer.testing import CliRunner

from mlops_lifecycle_lab.entrypoints.cli.app import app


def test_version_option_prints_distribution_and_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0, result.output
    assert "mlops-lifecycle-lab 0.1.0" in result.output


def test_no_arguments_shows_help() -> None:
    result = CliRunner().invoke(app, [])

    assert "Usage:" in result.output
