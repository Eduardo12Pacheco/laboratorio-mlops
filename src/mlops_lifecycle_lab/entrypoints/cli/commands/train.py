"""``mll train``: train a challenger on the S12 windows and register it (S5, S6)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.application.dto import TrainRequest
from mlops_lifecycle_lab.application.services.lifecycle import TRAINING_ORIGIN
from mlops_lifecycle_lab.bootstrap.container import build_container
from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.domain.errors import DomainError
from mlops_lifecycle_lab.domain.windows import TimeWindow


def train(
    tick_end: Annotated[
        datetime,
        typer.Option(
            "--tick-end",
            formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
            help="Exclusive end of the holdout / current tick end (YYYY-MM-DD).",
        ),
    ],
    trigger: Annotated[
        str, typer.Option("--trigger", help="Trigger reason logged with the run.")
    ] = "MANUAL",
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Train, register and report a candidate without promoting it."""
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    if not trigger.strip():
        raise typer.BadParameter("trigger must not be empty", param_hint="--trigger")
    cut = tick_end - timedelta(days=settings.holdout_days)
    request = TrainRequest(
        train_window=TimeWindow(TRAINING_ORIGIN, cut),
        holdout_window=TimeWindow(cut, tick_end),
        trigger_reason=trigger.strip(),
    )
    container = build_container(settings)
    try:
        result = container.training.execute(request)
        typer.echo(f"registered v{result.version} (tick end {tick_end.date().isoformat()})")
        typer.echo(
            f"rmse={result.metrics.rmse:.4f} wape={result.metrics.wape:.4f} "
            f"rows={result.training_rows} holdout={result.holdout_rows}"
        )
    except DomainError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        container.close()


__all__ = ["train"]
