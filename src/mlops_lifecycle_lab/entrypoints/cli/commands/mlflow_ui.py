"""``mll mlflow-ui``: open the MLflow UI for the workspace tracking store (T10, S6, L4).

The UI is an operator convenience, not a product dependency: it runs the installed MLflow
module against the same SQLite tracking URI the lab uses and binds the documented host port
so parallel portfolio stacks never clash.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 14000
"""L4 reserves 14000 for the local MLflow UI."""


def ui_command(
    tracking_uri: str,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
) -> list[str]:
    """Build the argv for ``python -m mlflow ui`` against `tracking_uri`."""
    return [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        tracking_uri,
        "--host",
        host,
        "--port",
        str(port),
    ]


def mlflow_ui(
    port: Annotated[
        int, typer.Option("--port", min=1, max=65535, help="Host port (default: 14000).")
    ] = DEFAULT_PORT,
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Serve the MLflow UI for the workspace tracking store."""
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    command = ui_command(settings.effective_tracking_uri, port=port)
    typer.echo(f"mlflow ui: {settings.effective_tracking_uri} on http://{DEFAULT_HOST}:{port}")
    try:
        completed = subprocess.run(command)  # noqa: S603 - fixed argv, no shell
    except KeyboardInterrupt:
        raise typer.Exit(code=0) from None
    if completed.returncode != 0:
        typer.echo(f"mlflow ui exited with code {completed.returncode}", err=True)
        raise typer.Exit(code=1)


__all__ = ["DEFAULT_PORT", "mlflow_ui", "ui_command"]
