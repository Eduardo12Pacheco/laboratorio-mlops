"""`mll serve` and `mll score` are registered and expose their documented options."""

from __future__ import annotations

from typer.testing import CliRunner, Result

from mlops_lifecycle_lab.entrypoints.cli.app import app

runner = CliRunner()


def invoke(args: list[str]) -> Result:
    """Invoke with a wide, colour-free terminal for stable cross-platform assertions."""
    return runner.invoke(
        app,
        args,
        env={"NO_COLOR": "1", "TERM": "dumb"},
        terminal_width=200,
    )


def test_serve_help_lists_host_port_and_workspace() -> None:
    result = invoke(["serve", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--workspace" in result.output


def test_score_help_lists_input_output_source_and_workspace() -> None:
    result = invoke(["score", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--source" in result.output
    assert "--workspace" in result.output
