"""``mll replay``: inspect or reset the persisted replay cursor (S14, T10).

The cursor is the KV key ``replay_tick`` written by the drill engine and by the scheduler.
Reading it does not build the composition root (no MLflow, no trainer) so operators can
inspect a workspace quickly; an absent workspace simply reports tick 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.adapters.storage.kv_store import SqliteKeyValueStore
from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults

replay_app = typer.Typer(
    name="replay",
    help="Inspect or reset the weekly replay cursor.",
    no_args_is_help=True,
)

KEY_REPLAY_TICK = "replay_tick"


def _settings(workspace: Path | None) -> Settings:
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    return settings


def _cursor(settings: Settings) -> int:
    database_path = settings.workspace / "mll.db"
    if not database_path.exists():
        return 0
    database = SqliteDatabase(database_path)
    try:
        value = SqliteKeyValueStore(database).get(KEY_REPLAY_TICK)
    finally:
        database.close()
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


@replay_app.command("status")
def status(
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Print the completed tick and the configured replay length."""
    settings = _settings(workspace)
    typer.echo(f"tick {_cursor(settings)} / {settings.replay_weeks}")


@replay_app.command("reset")
def reset(
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Clear the cursor so the next scheduler run starts at tick 1."""
    settings = _settings(workspace)
    database = SqliteDatabase(settings.workspace / "mll.db")
    try:
        SqliteKeyValueStore(database).set(KEY_REPLAY_TICK, None)
    finally:
        database.close()
    typer.echo(f"tick 0 / {settings.replay_weeks}")


__all__ = ["KEY_REPLAY_TICK", "replay_app"]
