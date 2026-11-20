"""``mll drill run``: reproducible weekly replays of the four scenarios (S15, S16)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from mlops_lifecycle_lab.config import apply_privacy_defaults
from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine
from mlops_lifecycle_lab.entrypoints.drill.expectations import compare, load_golden
from mlops_lifecycle_lab.entrypoints.drill.report import DrillResult, render_console
from mlops_lifecycle_lab.entrypoints.drill.scenarios import SCENARIOS, get_scenario

drill_app = typer.Typer(
    name="drill",
    help="Run the reproducible drift and fault-injection drills.",
    no_args_is_help=True,
)

_SCENARIO_NAMES: tuple[str, ...] = ("all", *sorted(SCENARIOS))
Golden = tuple[DrillResult, tuple[str, ...]]


def _golden_path(scenario: str) -> Path:
    return Path("tests") / "golden" / f"drill_{scenario.replace('-', '_')}.json"


def _run_one(
    name: str,
    *,
    weeks: int | None,
    workspace: Path,
    report_dir: Path | None,
    seed: int,
    via: Literal["inprocess", "http"],
    api_url: str | None,
    fast: bool,
) -> DrillResult:
    engine = DrillEngine()
    return engine.run(
        name,
        weeks=weeks,
        workspace=workspace,
        seed=seed,
        via=via,
        report_dir=report_dir,
        fast=fast,
        api_url=api_url,
    )


def _print_summary(result: DrillResult, json_path: Path) -> None:
    summary = result.summary
    typer.echo(f"drill: {result.scenario} ({result.weeks} weeks) -> {json_path}")
    typer.echo(
        f"promotions={summary.promotions} rejections={summary.rejections} "
        f"rollbacks={summary.rollbacks} quarantined={summary.quarantined_batches}"
    )
    ratio = summary.as_served.rmse / summary.counterfactual_frozen_v1.rmse
    typer.echo(
        f"as-served RMSE {summary.as_served.rmse:.2f} "
        f"(frozen-v1 {summary.counterfactual_frozen_v1.rmse:.2f}, ratio {ratio:.2f})"
    )


@drill_app.command("run")
def run(
    scenario: Annotated[
        str, typer.Option("--scenario", help="drift, gate-reject, bad-batch, rollback or all.")
    ] = "drift",
    weeks: Annotated[
        int | None, typer.Option("--weeks", min=1, help="Weekly ticks (default: scenario).")
    ] = None,
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
    report_dir: Annotated[
        Path | None, typer.Option("--report-dir", help="Report directory (default: ws/reports).")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", min=0, help="Model seed.")] = 42,
    via: Annotated[str, typer.Option("--via", help="Score via inprocess or http.")] = "inprocess",
    api_url: Annotated[
        str | None, typer.Option("--api-url", help="API base URL when --via http.")
    ] = None,
    fast: Annotated[
        bool, typer.Option("--fast", help="Skip Evidently HTML reports (Null renderer).")
    ] = False,
    check: Annotated[
        bool, typer.Option("--check", help="Compare against tests/golden; exit 4 on mismatch.")
    ] = False,
) -> None:
    """Replay a scenario, write JSON/Markdown reports and optionally check the golden."""
    apply_privacy_defaults()
    name = scenario.strip().lower()
    if name not in _SCENARIO_NAMES:
        raise typer.BadParameter(
            f"unknown scenario {scenario!r}; choose one of: {', '.join(_SCENARIO_NAMES)}",
            param_hint="--scenario",
        )
    if via not in ("inprocess", "http"):
        raise typer.BadParameter("via must be 'inprocess' or 'http'", param_hint="--via")
    base = Path(workspace) if workspace is not None else Path(".mll")
    names = sorted(SCENARIOS) if name == "all" else [name]
    results: list[DrillResult] = []
    for scenario_name in names:
        resolved = get_scenario(scenario_name)
        target_workspace = base / scenario_name if name == "all" else base
        result = _run_one(
            scenario_name,
            weeks=weeks,
            workspace=target_workspace,
            report_dir=report_dir,
            seed=seed,
            via=cast(Literal["inprocess", "http"], via),
            api_url=api_url,
            fast=fast,
        )
        results.append(result)
        json_path = Path(report_dir) if report_dir is not None else target_workspace / "reports"
        _print_summary(result, json_path / f"drill-{resolved.name}.json")
        if sys.stdout.isatty() and not check:
            render_console(result)

    if not check:
        return
    failures = 0
    for result in results:
        path = _golden_path(result.scenario)
        if not path.exists():
            typer.echo(f"golden file not found: {path}", err=True)
            raise typer.Exit(code=1)
        mismatches = compare(result, load_golden(path))
        if mismatches:
            failures += 1
            typer.echo(f"golden mismatch: {result.scenario} ({path})", err=True)
            for mismatch in mismatches:
                typer.echo(f"  {mismatch}", err=True)
        else:
            typer.echo(f"golden OK: {result.scenario}")
    if failures:
        raise typer.Exit(code=4)


__all__ = ["drill_app"]
