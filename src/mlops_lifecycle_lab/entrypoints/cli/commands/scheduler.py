"""``mll scheduler``: run the lifecycle tick on a replay schedule (S14, T10, D10).

The scheduler drives the same `RunLifecycleTick` use case as the drill. It bootstraps v1
when the workspace has no champion (the Compose `init` service normally does this first),
then advances the persisted `replay_tick` cursor one simulated week per interval. Use
``--mode once`` to execute exactly one tick (useful for cron-style external schedulers).
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.adapters.clock import SimulatedClock
from mlops_lifecycle_lab.adapters.scheduling.apscheduler_runner import (
    ReplayScheduler,
    TickFunction,
)
from mlops_lifecycle_lab.application.services.lifecycle import TickRequest
from mlops_lifecycle_lab.bootstrap.container import Container, build_container
from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.domain.errors import DomainError
from mlops_lifecycle_lab.domain.windows import tick_window
from mlops_lifecycle_lab.logging_setup import configure_logging

scheduler_app = typer.Typer(
    name="scheduler",
    help="Run the lifecycle tick on the persisted weekly replay.",
    no_args_is_help=True,
)

REPLAY_START = datetime(2012, 1, 1)
"""First instant of the replayed 2012 half (D3); shared with the drill timeline."""

MODES = ("replay", "once")


def _tick_function(container: Container, clock: SimulatedClock) -> TickFunction:
    """Bind one lifecycle tick to the simulated replay clock."""

    def tick(number: int) -> None:
        window = tick_window(number, REPLAY_START, 7)
        clock.set(window.end)
        container.lifecycle.execute(TickRequest(tick=number, window=window))

    return tick


def _settings(workspace: Path | None) -> Settings:
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    return settings


@scheduler_app.command("start")
def start(
    mode: Annotated[
        str,
        typer.Option("--mode", help="replay (continuous weekly ticks) or once (single tick)."),
    ] = "replay",
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Start the replay scheduler (or run a single tick with ``--mode once``)."""
    if mode not in MODES:
        raise typer.BadParameter(f"mode must be one of: {', '.join(MODES)}", param_hint="--mode")
    settings = _settings(workspace)
    configure_logging(settings.log_level, settings.log_format)
    clock = SimulatedClock(REPLAY_START)
    container = build_container(settings, clock=clock)
    scheduler: ReplayScheduler | None = None
    try:
        if container.registry.get_champion() is None:
            champion = container.bootstrap.execute()
            typer.echo(f"bootstrap: champion v{champion.version}")
        scheduler = ReplayScheduler(
            _tick_function(container, clock),
            settings.replay_interval_seconds,
            container.kv,
            settings.replay_weeks,
        )
        if mode == "once":
            executed = scheduler.run_once()
            if executed is None:
                typer.echo(
                    f"scheduler: no tick executed (tick {scheduler.cursor} / "
                    f"{settings.replay_weeks})"
                )
            else:
                typer.echo(f"scheduler: tick {executed} / {settings.replay_weeks}")
            return
        scheduler.start()
        with suppress(KeyboardInterrupt):
            scheduler.wait()
        typer.echo(
            f"scheduler: replay complete at tick {scheduler.cursor} / {settings.replay_weeks}"
        )
    except DomainError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=True)
        container.close()


__all__ = ["REPLAY_START", "scheduler_app"]
