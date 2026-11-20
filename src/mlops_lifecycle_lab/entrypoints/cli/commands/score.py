"""``mll score``: batch-score a raw-schema CSV through the serving service (S8)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from mlops_lifecycle_lab.bootstrap.container import build_container
from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.domain.errors import DomainError
from mlops_lifecycle_lab.domain.schema import RAW_COLUMNS


def score(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, dir_okay=False, help="Raw-schema CSV.")
    ],
    output: Annotated[Path, typer.Option("--output", help="Destination CSV.")],
    source: Annotated[
        str, typer.Option("--source", help="Prediction source tag: batch or online.")
    ] = "batch",
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Score a CSV offline with the current champion and write entity/prediction/version."""
    if source not in ("batch", "online"):
        raise typer.BadParameter("source must be 'batch' or 'online'", param_hint="--source")
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    frame = pd.read_csv(input_path)
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise typer.BadParameter(
            f"input CSV is missing columns: {', '.join(missing)}", param_hint="--input"
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    container = build_container(settings)
    try:
        results = container.prediction_service.predict(frame, source=source)  # type: ignore[arg-type]
    except DomainError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        container.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "entity_id": result.entity_id,
                "prediction": result.prediction,
                "model_version": result.model_version,
            }
            for result in results
        ]
    ).to_csv(output, index=False)
    typer.echo(f"scored {len(results)} rows with model v{results[0].model_version}")


__all__ = ["score"]
