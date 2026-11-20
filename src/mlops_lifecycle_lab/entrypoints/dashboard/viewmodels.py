"""Read-only view models for the Control Room pages (S17).

The views are deliberately thin: every number they render comes from one of these pure
functions, which only read through the container ports (registry, event store, snapshot
store, prediction log, key/value state and run ledger). A missing workspace raises
`ViewError` with a user-safe message and the command that fixes it, so the UI never leaks
an adapter traceback and never shows made-up data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pandas as pd

from mlops_lifecycle_lab.application.dto import JSONValue
from mlops_lifecycle_lab.application.ports import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    ALIAS_PREVIOUS_CHAMPION,
)
from mlops_lifecycle_lab.bootstrap.container import Container
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent

MLFLOW_UI_URL = "http://127.0.0.1:14000"
"""L4 reserves host port 14000 for the local MLflow UI."""

DRILL_SCHEMA = "drill-report/v1"
QUICKSTART_HINT = "Run `mll quickstart --workspace <dir>` to create a workspace."
"""Displayed by every empty state; tests assert this literal."""

STATUS_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "healthy": "Healthy",
        "degraded": "Degraded (SLO breach)",
        "probation": "In probation",
        "rejected": "Rejected",
        "rolled_back": "Rolled back",
        "no_champion": "No champion",
    }
)


class ViewError(Exception):
    """A page cannot be rendered with the current workspace, with a safe explanation."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


@dataclass(frozen=True, slots=True)
class Kpi:
    """One dashboard tile."""

    label: str
    value: str
    caption: str | None = None


@dataclass(frozen=True, slots=True)
class EventRow:
    """One audit event as a table/timeline row."""

    ts: datetime
    type: str
    tick: int
    model_version: int | None
    summary: str


@dataclass(frozen=True, slots=True)
class OverviewView:
    """Status banner, KPI tiles and the most recent events."""

    status: str
    status_label: str
    champion_version: int | None
    kpis: tuple[Kpi, ...]
    recent_events: tuple[EventRow, ...]


@dataclass(frozen=True, slots=True)
class SeriesPoint:
    """One monitoring snapshot projected for the WAPE chart."""

    tick: int
    ts: datetime
    wape: float | None
    rmse: float | None
    champion_version: int


@dataclass(frozen=True, slots=True)
class VersionBand:
    """A contiguous run of ticks served by one champion version."""

    version: int
    start_tick: int
    end_tick: int


@dataclass(frozen=True, slots=True)
class ActualPredicted:
    """One labeled production row in the last 14 days."""

    ts: datetime
    actual: float
    predicted: float


@dataclass(frozen=True, slots=True)
class PerformanceView:
    """WAPE over time, champion bands and the recent actual-vs-predicted series."""

    points: tuple[SeriesPoint, ...]
    bands: tuple[VersionBand, ...]
    slo: float
    bias: float | None
    actual_vs_predicted: tuple[ActualPredicted, ...]
    verdicts: tuple[tuple[int, str], ...]

    @property
    def final_champion(self) -> int | None:
        """Champion of the last tick, or `None` with no snapshots."""
        return self.points[-1].champion_version if self.points else None


@dataclass(frozen=True, slots=True)
class DriftView:
    """Numeric/categorical PSI per feature and tick (heatmap input)."""

    features: tuple[str, ...]
    ticks: tuple[int, ...]
    values: Mapping[str, Mapping[int, float]]
    thresholds: tuple[float, float] = (0.1, 0.2)


@dataclass(frozen=True, slots=True)
class RegistryRow:
    """One registered version as the registry page shows it."""

    version: int
    stage: str
    aliases: tuple[str, ...]
    rmse: float | None
    wape: float | None
    train_window: str
    fingerprint: str
    created: datetime
    run_id: str


@dataclass(frozen=True, slots=True)
class RegistryView:
    """The version table plus the MLflow UI link."""

    rows: tuple[RegistryRow, ...]
    mlflow_url: str = MLFLOW_UI_URL


@dataclass(frozen=True, slots=True)
class GateRow:
    """One promotion gate outcome."""

    name: str
    passed: bool
    observed: str
    threshold: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class GateReportView:
    """The complete gate report of one evaluated version."""

    version: int
    tick: int
    decision: str
    gates: tuple[GateRow, ...]


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """One run-ledger row with its duration and failed-step count."""

    run_id: str
    name: str
    tick: int | None
    status: str
    started_at: datetime
    finished_at: datetime | None
    duration_seconds: float
    failed_steps: int
    error: str | None


@dataclass(frozen=True, slots=True)
class ReportEvent:
    """One event embedded in a committed drill report."""

    ts: datetime
    type: str
    tick: int
    model_version: int | None


@dataclass(frozen=True, slots=True)
class DrillTick:
    """One tick of a committed drill report."""

    tick: int
    tick_end: datetime
    champion_version: int | None
    action: str
    validation_verdict: str
    wape: float | None
    rmse: float | None
    n_rows: int
    numeric_drift_share: float
    events: tuple[ReportEvent, ...]


@dataclass(frozen=True, slots=True)
class DrillComparison:
    """As-served vs frozen-v1 metrics of a drill run."""

    as_served_rmse: float
    frozen_rmse: float
    as_served_wape: float
    frozen_wape: float

    @property
    def ratio(self) -> float:
        """As-served RMSE divided by the frozen-v1 RMSE."""
        return self.as_served_rmse / self.frozen_rmse if self.frozen_rmse else math.inf


@dataclass(frozen=True, slots=True)
class DrillReport:
    """A reproducible `drill-report/v1` document read from disk."""

    path: Path
    scenario: str
    seed: int
    weeks: int
    ticks: tuple[DrillTick, ...]
    summary: Mapping[str, JSONValue]

    @property
    def comparison(self) -> DrillComparison:
        """The headline metrics of the summary."""
        as_served = _mapping(self.summary.get("as_served"))
        frozen = _mapping(self.summary.get("counterfactual_frozen_v1"))
        return DrillComparison(
            as_served_rmse=_float(as_served.get("rmse")),
            frozen_rmse=_float(frozen.get("rmse")),
            as_served_wape=_float(as_served.get("wape")),
            frozen_wape=_float(frozen.get("wape")),
        )

    @property
    def first_retrain_tick(self) -> int | None:
        """First tick with a `RETRAIN_TRIGGERED` event."""
        for tick in self.ticks:
            if any(event.type == EventType.RETRAIN_TRIGGERED.value for event in tick.events):
                return tick.tick
        return None


@dataclass(frozen=True, slots=True)
class DrillState:
    """The report slice shown for the selected tick."""

    tick: int
    tick_end: datetime
    champion_version: int | None
    action: str
    validation_verdict: str
    wape: float | None
    rmse: float | None
    n_rows: int
    numeric_drift_share: float
    events: tuple[ReportEvent, ...]


# -- shared helpers -------------------------------------------------------------------
def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _float(value: object) -> float:
    if not isinstance(value, int | float):
        raise ViewError(f"drill report has a non-numeric metric: {value!r}")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def require_workspace(container: Container) -> None:
    if container.registry.list_versions() or container.events.list(limit=1):
        return
    raise ViewError("This workspace has no lifecycle data yet.", hint=QUICKSTART_HINT)


def _reference_now(container: Container) -> datetime:
    snapshots = container.snapshots.list(limit=1)
    if snapshots:
        return snapshots[-1].ts
    events = container.events.list(limit=1)
    if events:
        return events[-1].ts
    return container.clock.now()


def _event_summary(event: LifecycleEvent) -> str:
    payload = event.payload
    for key in ("reason", "failed_rules", "failed_gates", "verdict", "action", "rows"):
        if key in payload:
            value = payload[key]
            if isinstance(value, Sequence) and not isinstance(value, str):
                return f"{key}={', '.join(str(item) for item in value)}"
            return f"{key}={value}"
    return ""


def _event_row(event: LifecycleEvent) -> EventRow:
    return EventRow(
        ts=event.ts,
        type=event.type.value,
        tick=event.tick,
        model_version=event.model_version,
        summary=_event_summary(event),
    )


# -- page view models -----------------------------------------------------------------
def overview(container: Container) -> OverviewView:
    """Status, KPI tiles and the last five events."""
    require_workspace(container)
    champion = container.registry.get_champion()
    snapshots = container.snapshots.list(limit=1)
    latest = snapshots[-1] if snapshots else None
    now = _reference_now(container)
    slo = float(container.settings.slo_wape)

    probation_version = container.kv.get("probation_version")
    if probation_version is not None:
        status = "probation"
    elif latest is not None and latest.wape is not None and latest.wape > slo:
        status = "degraded"
    elif champion is None:
        status = "no_champion"
    else:
        status = "healthy"

    days_since_retrain = "n/a"
    last_retrain = container.kv.get("last_retrain_at")
    if isinstance(last_retrain, str):
        try:
            days_since_retrain = str(max(0, (now - datetime.fromisoformat(last_retrain)).days))
        except ValueError:
            days_since_retrain = "n/a"

    probation_left = "n/a"
    probation_until = container.kv.get("probation_until")
    if isinstance(probation_until, str):
        try:
            remaining = datetime.fromisoformat(probation_until) - now
            probation_left = str(max(0, remaining.days))
        except ValueError:
            probation_left = "n/a"

    kpis = (
        Kpi("Champion", "none" if champion is None else f"v{champion.version}"),
        Kpi(
            "WAPE 28d",
            "n/a" if latest is None or latest.wape is None else f"{latest.wape:.4f}",
            f"SLO {slo:.2f}",
        ),
        Kpi(
            "RMSE 28d",
            "n/a" if latest is None or latest.rmse is None else f"{latest.rmse:.2f}",
        ),
        Kpi("Labeled rows", "0" if latest is None else str(latest.n_rows)),
        Kpi("Days since retrain", days_since_retrain),
        Kpi("Probation days left", probation_left),
    )
    recent = tuple(_event_row(event) for event in container.events.list(limit=5))
    return OverviewView(
        status=status,
        status_label=STATUS_LABELS[status],
        champion_version=None if champion is None else champion.version,
        kpis=kpis,
        recent_events=recent,
    )


def performance_series(container: Container) -> PerformanceView:
    """WAPE per tick with champion bands and the last 14 days of actuals vs predictions."""
    require_workspace(container)
    snapshots = container.snapshots.list()
    points = tuple(
        SeriesPoint(
            tick=snapshot.tick,
            ts=snapshot.ts,
            wape=snapshot.wape,
            rmse=snapshot.rmse,
            champion_version=snapshot.model_version,
        )
        for snapshot in snapshots
    )
    bands: list[VersionBand] = []
    for point in points:
        if bands and bands[-1].version == point.champion_version:
            previous = bands[-1]
            bands[-1] = VersionBand(previous.version, previous.start_tick, point.tick)
        else:
            bands.append(VersionBand(point.champion_version, point.tick, point.tick))

    end = points[-1].ts if points else _reference_now(container)
    start = end - timedelta(days=14)
    labeled = container.prediction_log.labeled_predictions(start, end)
    recent: list[ActualPredicted] = []
    if not labeled.empty:
        timestamps = pd.to_datetime(labeled["event_ts"])
        ordered = labeled.assign(_ts=timestamps).sort_values("_ts")
        recent = [
            ActualPredicted(
                ts=cast(pd.Timestamp, row["_ts"]).to_pydatetime(),
                actual=float(row["label"]),
                predicted=float(row["prediction"]),
            )
            for _, row in ordered.iterrows()
        ]

    latest = snapshots[-1] if snapshots else None
    return PerformanceView(
        points=points,
        bands=tuple(bands),
        slo=float(container.settings.slo_wape),
        bias=None if latest is None else latest.bias,
        actual_vs_predicted=tuple(recent),
        verdicts=tuple((snapshot.tick, snapshot.verdict) for snapshot in snapshots),
    )


def drift_matrix(container: Container) -> DriftView:
    """PSI per monitored feature and tick."""
    require_workspace(container)
    snapshots = container.snapshots.list()
    features = tuple(sorted({feature for snapshot in snapshots for feature in snapshot.psi}))
    values: dict[str, Mapping[int, float]] = {}
    for feature in features:
        values[feature] = MappingProxyType(
            {
                snapshot.tick: float(snapshot.psi[feature])
                for snapshot in snapshots
                if feature in snapshot.psi
            }
        )
    return DriftView(
        features=features,
        ticks=tuple(snapshot.tick for snapshot in snapshots),
        values=MappingProxyType(values),
    )


def registry_table(container: Container) -> RegistryView:
    """Every registered version with its aliases, metrics and lineage."""
    versions = container.registry.list_versions()
    if not versions:
        raise ViewError("No model versions are registered in this workspace.", hint=QUICKSTART_HINT)
    alias_versions = {
        alias: container.registry.get_alias(alias)
        for alias in (
            ALIAS_CHAMPION,
            ALIAS_PREVIOUS_CHAMPION,
            ALIAS_CHALLENGER,
        )
    }
    rows = []
    for version in versions:
        aliases = tuple(
            alias
            for alias, alias_version in alias_versions.items()
            if alias_version is not None and alias_version.version == version.version
        )
        rows.append(
            RegistryRow(
                version=version.version,
                stage=version.stage.value,
                aliases=aliases,
                rmse=version.metrics.get("rmse"),
                wape=version.metrics.get("wape"),
                train_window=f"v{version.version}: {version.train_window.start.date()} → "
                f"{version.train_window.end.date()}",
                fingerprint=version.data_fingerprint[:8],
                created=version.created_at,
                run_id=version.run_id,
            )
        )
    return RegistryView(rows=tuple(rows))


def gate_report(container: Container, version: int) -> GateReportView:
    """The latest gate report of `version`, parsed from its `GATES_EVALUATED` event."""
    events = container.events.list(event_type=EventType.GATES_EVALUATED, model_version=version)
    if not events:
        raise ViewError(f"No gate report was recorded for model v{version}.")
    event = events[-1]
    report = _mapping(event.payload.get("gate_report"))
    raw_results = report.get("results")
    results = raw_results if isinstance(raw_results, Sequence) else []
    gates = []
    for result in results:
        values = _mapping(result)
        if not values:
            continue
        gates.append(
            GateRow(
                name=str(values.get("name", "?")),
                passed=bool(values.get("passed", False)),
                observed=_format_value(values.get("observed")),
                threshold=_format_value(values.get("threshold")),
                detail=str(values.get("detail", "")),
            )
        )
    decision = "APPROVED" if event.payload.get("approved") is True else "REJECTED"
    return GateReportView(
        version=version,
        tick=event.tick,
        decision=decision,
        gates=tuple(gates),
    )


def events_table(
    container: Container,
    types: Sequence[EventType] | None = None,
    limit: int | None = 50,
) -> tuple[EventRow, ...]:
    """Audit events, newest first, optionally filtered by type."""
    events = container.events.list()
    if types is not None:
        wanted = set(types)
        events = [event for event in events if event.type in wanted]
    events.sort(key=lambda event: event.ts, reverse=True)
    if limit is not None:
        events = events[:limit]
    return tuple(_event_row(event) for event in events)


def ledger_table(container: Container, limit: int | None = 50) -> tuple[LedgerRow, ...]:
    """Run ledger rows, newest first."""
    runs = container.ledger.list_runs(limit=limit)
    return tuple(
        LedgerRow(
            run_id=run.run_id,
            name=run.name,
            tick=run.tick,
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            duration_seconds=run.duration_seconds,
            failed_steps=run.failed_steps,
            error=run.error,
        )
        for run in runs
    )


def drill_reports(report_dir: Path) -> tuple[DrillReport, ...]:
    """Parse every valid `drill-report/v1` file in `report_dir`, sorted by scenario."""
    directory = Path(report_dir)
    if not directory.is_dir():
        return ()
    reports: list[DrillReport] = []
    for path in sorted(directory.glob("drill-*.json")):
        report = _load_drill_report(path)
        if report is not None:
            reports.append(report)
    return tuple(sorted(reports, key=lambda report: report.scenario))


def _load_drill_report(path: Path) -> DrillReport | None:
    try:
        document = _mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    if document.get("schema") != DRILL_SCHEMA:
        return None
    raw_ticks = document.get("ticks")
    if not isinstance(raw_ticks, Sequence):
        return None
    ticks = []
    for raw_tick in raw_ticks:
        tick = _mapping(raw_tick)
        snapshot = _mapping(tick.get("snapshot"))
        raw_events = tick.get("events")
        events = []
        if isinstance(raw_events, Sequence):
            for raw_event in raw_events:
                event = _mapping(raw_event)
                try:
                    events.append(
                        ReportEvent(
                            ts=datetime.fromisoformat(str(event["ts"])),
                            type=str(event["type"]),
                            tick=int(str(event["tick"])),
                            model_version=(
                                None
                                if event.get("model_version") is None
                                else int(str(event["model_version"]))
                            ),
                        )
                    )
                except (KeyError, ValueError):
                    continue
        try:
            ticks.append(
                DrillTick(
                    tick=int(str(tick["tick"])),
                    tick_end=datetime.fromisoformat(str(tick["tick_end"])),
                    champion_version=(
                        None
                        if tick.get("champion_version") is None
                        else int(str(tick["champion_version"]))
                    ),
                    action=str(tick.get("action", "")),
                    validation_verdict=str(tick.get("validation_verdict", "")),
                    wape=_optional_float(snapshot.get("wape")),
                    rmse=_optional_float(snapshot.get("rmse")),
                    n_rows=int(str(snapshot.get("n_rows", 0))),
                    numeric_drift_share=_optional_float(snapshot.get("numeric_drift_share")) or 0.0,
                    events=tuple(events),
                )
            )
        except (KeyError, ValueError):
            return None
    return DrillReport(
        path=path,
        scenario=str(document.get("scenario", path.stem)),
        seed=int(str(document.get("seed", 0))),
        weeks=int(str(document.get("weeks", len(ticks)))),
        ticks=tuple(ticks),
        summary=_mapping(document.get("summary")),
    )


def drill_state_at(report: DrillReport, tick: int) -> DrillState:
    """Return the report slice for `tick` (1-based)."""
    for candidate in report.ticks:
        if candidate.tick == tick:
            return DrillState(
                tick=candidate.tick,
                tick_end=candidate.tick_end,
                champion_version=candidate.champion_version,
                action=candidate.action,
                validation_verdict=candidate.validation_verdict,
                wape=candidate.wape,
                rmse=candidate.rmse,
                n_rows=candidate.n_rows,
                numeric_drift_share=candidate.numeric_drift_share,
                events=candidate.events,
            )
    raise ViewError(f"Tick {tick} is not present in the {report.scenario} report.")


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "n/a" if value is None else str(value)


__all__ = [
    "DRILL_SCHEMA",
    "MLFLOW_UI_URL",
    "QUICKSTART_HINT",
    "STATUS_LABELS",
    "ActualPredicted",
    "DriftView",
    "DrillComparison",
    "DrillReport",
    "DrillState",
    "DrillTick",
    "EventRow",
    "GateReportView",
    "GateRow",
    "Kpi",
    "LedgerRow",
    "OverviewView",
    "PerformanceView",
    "RegistryRow",
    "RegistryView",
    "ReportEvent",
    "SeriesPoint",
    "VersionBand",
    "ViewError",
    "drift_matrix",
    "drill_reports",
    "drill_state_at",
    "events_table",
    "gate_report",
    "ledger_table",
    "overview",
    "performance_series",
    "registry_table",
    "require_workspace",
]
