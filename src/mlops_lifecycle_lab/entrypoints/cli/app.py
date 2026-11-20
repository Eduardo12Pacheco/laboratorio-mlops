"""Typer entry point for the ``mll`` command.

Only the ``--version`` flag exists in the scaffold; feature commands are registered by
later tasks.
"""

from __future__ import annotations

import typer

from mlops_lifecycle_lab import DISTRIBUTION_NAME, __version__
from mlops_lifecycle_lab.config import apply_privacy_defaults
from mlops_lifecycle_lab.entrypoints.cli.commands.data import data_app
from mlops_lifecycle_lab.entrypoints.cli.commands.score import score
from mlops_lifecycle_lab.entrypoints.cli.commands.serve import serve
from mlops_lifecycle_lab.entrypoints.cli.commands.validate import validate

app = typer.Typer(
    name="mll",
    help="Local-first MLOps lifecycle lab.",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
)

app.add_typer(data_app, name="data")
app.command("validate")(validate)
app.command("serve")(serve)
app.command("score")(score)


@app.callback()
def cli(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed version and exit.",
        is_eager=True,
    ),
) -> None:
    """Manage the bike-demand lifecycle lab."""
    if version:
        typer.echo(f"{DISTRIBUTION_NAME} {__version__}")
        raise typer.Exit()


def main() -> None:
    """Console-script entry point: apply offline defaults, then parse arguments."""
    apply_privacy_defaults()
    app()
