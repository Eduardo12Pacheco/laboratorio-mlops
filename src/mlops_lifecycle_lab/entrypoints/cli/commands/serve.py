"""``mll serve``: run the serving API with hot reload (S8)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from fastapi import FastAPI

from mlops_lifecycle_lab.bootstrap.container import build_container
from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.entrypoints.api.app import create_app
from mlops_lifecycle_lab.logging_setup import configure_logging


def serve(
    host: Annotated[
        str | None, typer.Option("--host", help="Bind address (default: MLL_API_HOST).")
    ] = None,
    port: Annotated[
        int | None, typer.Option("--port", help="Bind port (default: MLL_API_PORT).")
    ] = None,
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Start the FastAPI serving app; the champion hot-reloads without a restart."""
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    bind_host = host or settings.api_host
    bind_port = port or settings.api_port
    configure_logging(settings.log_level, settings.log_format)

    def factory() -> FastAPI:
        return create_app(build_container(settings))

    uvicorn.run(factory, factory=True, host=bind_host, port=bind_port)


__all__ = ["serve"]
