"""``mll monitor snapshot``: recompute the S10 monitoring snapshot for a date."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.bootstrap.container import build_container
from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.domain.errors import DomainError
from mlops_lifecycle_lab.domain.events import EventType

monitor_app = typer.Typer(
    name="monitor",
    help="Compute the monitoring snapshot for the current champion.",
    no_args_is_help=True,
)


@monitor_app.command("snapshot")
def snapshot(
    at: Annotated[
        datetime,
        typer.Option(
            "--at",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
            help="Exclusive end of the trailing 28-day window (YYYY-MM-DD).",
        ),
    ],
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Print the champion's trailing metrics and drift statistics."""
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    container = build_container(settings)
    try:
        champion = container.registry.get_champion()
        if champion is None:
            typer.echo("NoChampionError: bootstrap a champion first (mll bootstrap)", err=True)
            raise typer.Exit(code=1)
        snapshot = container.monitoring.execute(at, champion.version)
        typer.echo(f"champion: v{snapshot.model_version} at {snapshot.ts.isoformat()}")
        typer.echo(f"rows: {snapshot.n_rows}")
        for name in ("rmse", "mae", "wape", "bias"):
            value = getattr(snapshot, name)
            typer.echo(f"{name}: {'n/a' if value is None else f'{value:.4f}'}")
        typer.echo(f"numeric_drift_share: {snapshot.numeric_drift_share:.2f}")
        typer.echo(
            "prediction_psi: "
            f"{'n/a' if snapshot.prediction_psi is None else f'{snapshot.prediction_psi:.4f}'}"
        )
        for feature, value in sorted(snapshot.psi.items()):
            typer.echo(f"psi {feature}: {value:.4f}")
        kind = EventType.SNAPSHOT_COMPUTED.value
        typer.echo(f"event: {kind} (tick {snapshot.tick})")
    except DomainError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        container.close()


__all__ = ["monitor_app"]
