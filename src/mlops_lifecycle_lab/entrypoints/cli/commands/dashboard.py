"""``mll dashboard``: launch the Streamlit Control Room for a workspace (S17)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.entrypoints.dashboard import app as dashboard_app
from mlops_lifecycle_lab.entrypoints.dashboard.theme import DARK, LIGHT

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 14501
"""L4 reserves host port 14501 for this dashboard."""


def dashboard_command(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> list[str]:
    """Build the argv for ``python -m streamlit run`` against the packaged app."""
    app_path = Path(dashboard_app.__file__).resolve()
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--theme.light.primaryColor",
        LIGHT.accent,
        "--theme.dark.primaryColor",
        DARK.accent,
        "--theme.light.font",
        "sans serif",
        "--theme.dark.font",
        "sans serif",
    ]


def dashboard(
    host: Annotated[
        str, typer.Option("--host", help="Address the dashboard binds to (default: 127.0.0.1).")
    ] = DEFAULT_HOST,
    port: Annotated[
        int, typer.Option("--port", min=1, max=65535, help="Host port (default: 14501).")
    ] = DEFAULT_PORT,
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Serve the five-page Control Room."""
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    environment = os.environ.copy()
    environment["MLL_WORKSPACE"] = str(settings.workspace)
    typer.echo(f"control room: http://{host}:{port} (workspace {settings.workspace})")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            dashboard_command(host=host, port=port), env=environment
        )
    except KeyboardInterrupt:
        raise typer.Exit(code=0) from None
    if completed.returncode != 0:
        typer.echo(f"dashboard exited with code {completed.returncode}", err=True)
        raise typer.Exit(code=1)


__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "dashboard", "dashboard_command"]
