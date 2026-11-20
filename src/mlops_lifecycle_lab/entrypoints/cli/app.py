"""Typer entry point for the ``mll`` command.

The app is assembled here from one module per command. Every subcommand applies the
offline/privacy defaults and depends on the composition root, never on concrete adapters
directly.
"""

from __future__ import annotations

import typer

from mlops_lifecycle_lab import DISTRIBUTION_NAME, __version__
from mlops_lifecycle_lab.config import apply_privacy_defaults
from mlops_lifecycle_lab.entrypoints.cli.commands.bootstrap import bootstrap
from mlops_lifecycle_lab.entrypoints.cli.commands.data import data_app
from mlops_lifecycle_lab.entrypoints.cli.commands.doctor import doctor
from mlops_lifecycle_lab.entrypoints.cli.commands.drill import drill_app
from mlops_lifecycle_lab.entrypoints.cli.commands.mlflow_ui import mlflow_ui
from mlops_lifecycle_lab.entrypoints.cli.commands.monitor import monitor_app
from mlops_lifecycle_lab.entrypoints.cli.commands.quickstart import quickstart
from mlops_lifecycle_lab.entrypoints.cli.commands.registry import registry_app
from mlops_lifecycle_lab.entrypoints.cli.commands.replay import replay_app
from mlops_lifecycle_lab.entrypoints.cli.commands.scheduler import scheduler_app
from mlops_lifecycle_lab.entrypoints.cli.commands.score import score
from mlops_lifecycle_lab.entrypoints.cli.commands.serve import serve
from mlops_lifecycle_lab.entrypoints.cli.commands.train import train
from mlops_lifecycle_lab.entrypoints.cli.commands.validate import validate

app = typer.Typer(
    name="mll",
    help="Local-first MLOps lifecycle lab.",
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,
)

app.add_typer(data_app, name="data")
app.add_typer(registry_app, name="registry")
app.add_typer(monitor_app, name="monitor")
app.add_typer(drill_app, name="drill")
app.add_typer(replay_app, name="replay")
app.add_typer(scheduler_app, name="scheduler")
app.command("validate")(validate)
app.command("bootstrap")(bootstrap)
app.command("train")(train)
app.command("serve")(serve)
app.command("score")(score)
app.command("quickstart")(quickstart)
app.command("mlflow-ui")(mlflow_ui)
app.command("doctor")(doctor)


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
