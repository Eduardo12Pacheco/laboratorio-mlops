"""``mll bootstrap``: train, gate and promote the first champion (S5, D3)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.bootstrap.container import build_container
from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.domain.errors import DomainError


def bootstrap(
    if_missing: Annotated[
        bool,
        typer.Option("--if-missing", help="Do nothing when a champion already exists."),
    ] = False,
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Bootstrap v1 idempotently; print the champion version."""
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    container = build_container(settings)
    try:
        existing = container.registry.get_champion()
        if if_missing and existing is not None:
            typer.echo(f"champion: v{existing.version} (already bootstrapped)")
            return
        champion = container.bootstrap.execute()
        typer.echo(f"champion: v{champion.version}")
        typer.echo(
            f"rmse={champion.metrics.get('rmse', 0.0):.4f} "
            f"wape={champion.metrics.get('wape', 0.0):.4f}"
        )
    except DomainError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        container.close()


__all__ = ["bootstrap"]
