"""``mll quickstart``: a working lab in one command (S20, T10).

Doctor-lite (Python, runtime imports, LightGBM) → the drill engine's idempotent bootstrap
of v1 → the 16-week drift replay → the next step. The Evidently reports are kept enabled
because a quickstart should show the real artifacts, not the fast-mode placeholders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.entrypoints.cli.commands.doctor import (
    QUICKSTART_CHECKS,
    collect_checks,
    format_table,
    hard_failures,
)
from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine
from mlops_lifecycle_lab.entrypoints.drill.report import DrillResult

DEFAULT_WEEKS = 16


def _run_drill(workspace: Path, weeks: int) -> DrillResult:
    """Run the drift scenario against the real stack, Evidently reports included."""
    return DrillEngine().run("drift", weeks=weeks, workspace=workspace, fast=False)


def quickstart(
    weeks: Annotated[
        int,
        typer.Option("--weeks", min=1, help="Weekly ticks of the drift drill (default: 16)."),
    ] = DEFAULT_WEEKS,
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Check the environment, bootstrap v1, replay 16 weeks and print the next step."""
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})

    checks = collect_checks(
        settings, checks=QUICKSTART_CHECKS, include_docker=False, include_ports=False
    )
    failures = hard_failures(checks)
    if failures:
        typer.echo(format_table(checks))
        for failure in failures:
            typer.echo(f"{failure.name}: {failure.detail}", err=True)
        typer.echo(f"quickstart: {len(failures)} prerequisite check(s) failed", err=True)
        raise typer.Exit(code=1)
    typer.echo("doctor: OK (python, runtime imports, lightgbm)")

    try:
        result = _run_drill(settings.workspace, weeks)
    except Exception as exc:  # report any drill failure with one clear line
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    champion = result.ticks[-1].champion_version if result.ticks else None
    if champion is None:
        typer.echo("quickstart: the drill produced no ticks", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"champion: v{champion}")
    typer.echo(f"Next: mll dashboard --workspace {settings.workspace}")


__all__ = ["DEFAULT_WEEKS", "quickstart"]
