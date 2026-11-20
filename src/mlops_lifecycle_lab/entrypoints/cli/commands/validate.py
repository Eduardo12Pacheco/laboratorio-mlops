"""``mll validate``: run the data contract against the bundle, a CSV or a batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import pandas as pd
import typer

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset, from_uci_frame
from mlops_lifecycle_lab.domain.validation import (
    Context,
    ValidationReport,
    Verdict,
    validate_frame,
)


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "dteday" in frame.columns:
        return from_uci_frame(frame)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def _echo_report(report: ValidationReport, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return
    typer.echo(f"verdict: {report.verdict.value} ({report.rows} rows, {report.context})")
    for rule in report.rules:
        state = "PASS" if rule.passed else "FAIL"
        typer.echo(
            f"{state} {rule.severity.value:<5} {rule.name}: "
            f"{rule.violations}/{rule.total} {rule.message}"
        )


def validate(
    bundled: Annotated[
        bool, typer.Option("--bundled", help="Validate the bundled dataset instead of a CSV.")
    ] = False,
    path: Annotated[Path | None, typer.Argument(help="CSV with UCI or raw-schema columns.")] = None,
    context: Annotated[
        str, typer.Option("--context", help="Validation context: batch or training.")
    ] = "training",
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Validate a frame and exit 0 on PASS/WARN, 3 on FAIL."""
    if context not in ("batch", "training"):
        raise typer.BadParameter("context must be 'batch' or 'training'", param_hint="--context")
    if bundled == (path is not None):
        raise typer.BadParameter("provide exactly one input: --bundled or PATH")
    if bundled:
        frame = BundledDataset().raw()
    else:
        if path is None:
            raise typer.BadParameter("provide exactly one input: --bundled or PATH")
        if not path.is_file():
            raise typer.BadParameter(f"file not found: {path}", param_hint="PATH")
        frame = _load_frame(path)

    report = validate_frame(frame, context=cast(Context, context))
    _echo_report(report, json_output)
    if report.verdict is Verdict.FAIL:
        raise typer.Exit(code=3)
