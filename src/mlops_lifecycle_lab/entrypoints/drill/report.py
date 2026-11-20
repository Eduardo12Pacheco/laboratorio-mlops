"""Reproducible `drill-report/v1` documents and summaries (S15).

Reports are pure data: no wall-clock timestamps and no absolute paths, so two runs of the
same scenario are byte-identical and a diff is meaningful. The JSON document is the source
of truth; the Markdown renderer and the Rich console table only display it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from mlops_lifecycle_lab.application.dto import MonitoringSnapshot, TickReport
from mlops_lifecycle_lab.application.ports import ModelRegistry, PredictionLog
from mlops_lifecycle_lab.application.services._data import clip_predictions
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.metrics import compute_metrics
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS

SCHEMA = "drill-report/v1"
GATE_APPROVED = "APPROVED"
GATE_REJECTED = "REJECTED"

_LIBS: tuple[tuple[str, str], ...] = (
    ("lightgbm", "lightgbm"),
    ("mlflow", "mlflow"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "scikit-learn"),
    ("evidently", "evidently"),
)


def library_versions() -> dict[str, str]:
    """Versions of the stack that produced the report (informational, not compared)."""
    versions: dict[str, str] = {}
    for label, package in _LIBS:
        try:
            versions[label] = version(package)
        except PackageNotFoundError:
            versions[label] = "unknown"
    return versions


@dataclass(frozen=True, slots=True)
class TickRecord:
    """One replay week as it appears in the report."""

    tick: int
    tick_end: datetime
    champion_version: int | None
    snapshot: MonitoringSnapshot | None
    validation_verdict: str
    action: str
    events: tuple[LifecycleEvent, ...]

    @classmethod
    def from_report(cls, report: TickReport) -> TickRecord:
        """Project a `TickReport` into its report row."""
        return cls(
            tick=report.tick,
            tick_end=report.window.end,
            champion_version=report.champion_version,
            snapshot=report.snapshot,
            validation_verdict=report.validation_verdict,
            action=report.action,
            events=report.events,
        )

    def event_types(self) -> list[str]:
        """Event type names in append order (the golden sequence)."""
        return [event.type.value for event in self.events]

    def to_dict(self) -> dict[str, Any]:
        """JSON representation of one tick."""
        return {
            "tick": self.tick,
            "tick_end": self.tick_end.isoformat(),
            "champion_version": self.champion_version,
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "validation_verdict": self.validation_verdict,
            "action": self.action,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(frozen=True, slots=True)
class GateReportRecord:
    """The gate evidence of one evaluation, flattened for the report."""

    tick: int
    version: int
    decision: str
    gates: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_event(cls, event: LifecycleEvent) -> GateReportRecord | None:
        """Return a record for a `GATES_EVALUATED` event, else `None`."""
        if event.type is not EventType.GATES_EVALUATED or event.model_version is None:
            return None
        artifact = event.payload.get("gate_report")
        results = artifact.get("results") if isinstance(artifact, Mapping) else None
        if not isinstance(results, Sequence):
            return None
        gates = tuple(
            {
                "name": result["name"],
                "passed": bool(result["passed"]),
                "observed": result["observed"],
                "threshold": result["threshold"],
            }
            for result in results
            if isinstance(result, Mapping)
        )
        decision = GATE_APPROVED if event.payload.get("approved") is True else GATE_REJECTED
        return cls(
            tick=event.tick,
            version=event.model_version,
            decision=decision,
            gates=gates,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON representation of one gate report."""
        return {
            "tick": self.tick,
            "version": self.version,
            "decision": self.decision,
            "gates": [dict(gate) for gate in self.gates],
        }


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    """Performance over the replayed rows: as served vs a frozen-v1 counterfactual."""

    rmse: float
    mae: float
    wape: float
    n: int

    def __post_init__(self) -> None:
        for name in ("rmse", "mae", "wape"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")

    def to_dict(self) -> dict[str, float | int]:
        """JSON representation of the performance summary."""
        return {"rmse": self.rmse, "mae": self.mae, "wape": self.wape, "n": self.n}


@dataclass(frozen=True, slots=True)
class DrillSummary:
    """The headline result of a drill run."""

    as_served: MetricsSummary
    counterfactual_frozen_v1: MetricsSummary
    promotions: int
    rejections: int
    rollbacks: int
    quarantined_batches: int
    versions: int

    def to_dict(self) -> dict[str, Any]:
        """JSON representation of the summary."""
        return {
            "as_served": self.as_served.to_dict(),
            "counterfactual_frozen_v1": self.counterfactual_frozen_v1.to_dict(),
            "promotions": self.promotions,
            "rejections": self.rejections,
            "rollbacks": self.rollbacks,
            "quarantined_batches": self.quarantined_batches,
            "versions": self.versions,
        }


@dataclass(frozen=True, slots=True)
class DrillResult:
    """A complete, reproducible drill run."""

    scenario: str
    seed: int
    weeks: int
    ticks: tuple[TickRecord, ...]
    gate_reports: tuple[GateReportRecord, ...]
    summary: DrillSummary

    @property
    def first_retrain_tick(self) -> int | None:
        """First tick with a `RETRAIN_TRIGGERED` event, or `None`."""
        for tick in self.ticks:
            if EventType.RETRAIN_TRIGGERED in {event.type for event in tick.events}:
                return tick.tick
        return None

    def events(self) -> list[LifecycleEvent]:
        """Every event of the run, in tick order."""
        return [event for tick in self.ticks for event in tick.events]

    def to_dict(self) -> dict[str, Any]:
        """The `drill-report/v1` JSON document."""
        return {
            "schema": SCHEMA,
            "scenario": self.scenario,
            "seed": self.seed,
            "weeks": self.weeks,
            "libs": library_versions(),
            "ticks": [tick.to_dict() for tick in self.ticks],
            "gate_reports": [report.to_dict() for report in self.gate_reports],
            "summary": self.summary.to_dict(),
        }

    def to_json(self) -> str:
        """Serialise deterministically (sorted keys, fixed indent)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def write(self, report_dir: Path) -> tuple[Path, Path]:
        """Write the JSON and Markdown reports; return both paths."""
        report_dir.mkdir(parents=True, exist_ok=True)
        stem = f"drill-{self.scenario}"
        json_path = report_dir / f"{stem}.json"
        markdown_path = report_dir / f"{stem}.md"
        json_path.write_text(self.to_json(), encoding="utf-8")
        markdown_path.write_text(self.markdown(), encoding="utf-8")
        return json_path, markdown_path

    def markdown(self) -> str:
        """Human-readable timeline, gate tables and summary."""
        lines = [
            f"# Drill report: {self.scenario}",
            "",
            f"- seed: {self.seed}",
            f"- weeks: {self.weeks}",
            f"- versions: {self.summary.versions}",
            f"- promotions: {self.summary.promotions}",
            f"- rejections: {self.summary.rejections}",
            f"- rollbacks: {self.summary.rollbacks}",
            f"- quarantined batches: {self.summary.quarantined_batches}",
            "",
            "## Timeline",
            "",
            "| tick | end | champion | WAPE-28d | numeric drift share | action |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for tick in self.ticks:
            snapshot = tick.snapshot
            wape = "n/a" if snapshot is None or snapshot.wape is None else f"{snapshot.wape:.4f}"
            drift = "n/a" if snapshot is None else f"{snapshot.numeric_drift_share:.2f}"
            champion = "n/a" if tick.champion_version is None else f"v{tick.champion_version}"
            lines.append(
                f"| {tick.tick} | {tick.tick_end.date().isoformat()} | {champion} | "
                f"{wape} | {drift} | {tick.action} |"
            )
        lines.extend(["", "## Gate reports", ""])
        if not self.gate_reports:
            lines.append("No gate evaluation in this run.")
        for report in self.gate_reports:
            lines.extend(
                [
                    f"### tick {report.tick} - v{report.version} - {report.decision}",
                    "",
                    "| gate | passed | observed | threshold |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for gate in report.gates:
                lines.append(
                    f"| {gate['name']} | {gate['passed']} | "
                    f"{_format_value(gate['observed'])} | {_format_value(gate['threshold'])} |"
                )
            lines.append("")
        served = self.summary.as_served
        frozen = self.summary.counterfactual_frozen_v1
        ratio = served.rmse / frozen.rmse
        lines.extend(
            [
                "## Summary",
                "",
                "| metric | as-served | frozen v1 |",
                "| --- | --- | --- |",
                f"| RMSE | {served.rmse:.4f} | {frozen.rmse:.4f} |",
                f"| MAE | {served.mae:.4f} | {frozen.mae:.4f} |",
                f"| WAPE | {served.wape:.4f} | {frozen.wape:.4f} |",
                f"| rows | {served.n} | {frozen.n} |",
                "",
                f"as-served RMSE is {ratio:.2f}x the frozen-v1 RMSE.",
                "",
                f"promotions={self.summary.promotions} "
                f"rejections={self.summary.rejections} "
                f"rollbacks={self.summary.rollbacks} "
                f"quarantined={self.summary.quarantined_batches}",
                "",
            ]
        )
        return "\n".join(lines)


def summarize_performance(
    registry: ModelRegistry,
    prediction_log: PredictionLog,
    start: datetime,
    end: datetime,
    *,
    counterfactual_version: int = 1,
) -> tuple[MetricsSummary, MetricsSummary]:
    """Compute as-served and frozen-v1 metrics over the replayed labeled rows."""
    labeled = prediction_log.labeled_predictions(start, end)
    if labeled.empty:
        raise ValueError("the drill produced no labeled predictions to summarize")
    actuals = labeled["label"].to_numpy(dtype=float)
    served = compute_metrics(actuals, labeled["prediction"].to_numpy(dtype=float))

    raw = prediction_log.all_labeled_features(end)
    timestamps = pd.to_datetime(raw["timestamp"])
    raw = raw.loc[(timestamps >= start) & (timestamps < end)].reset_index(drop=True)
    features = build_features(raw).loc[:, list(FEATURE_COLUMNS)]
    predictor = registry.load_predictor(counterfactual_version)
    frozen_predictions = clip_predictions(predictor.predict(features))
    frozen = compute_metrics(raw["cnt"].to_numpy(dtype=float), frozen_predictions)

    return (
        MetricsSummary(rmse=served.rmse, mae=served.mae, wape=served.wape, n=served.n),
        MetricsSummary(rmse=frozen.rmse, mae=frozen.mae, wape=frozen.wape, n=frozen.n),
    )


def render_console(result: DrillResult) -> None:
    """Print the compact Rich table used when stdout is a terminal."""
    table = Table(title=f"Drill {result.scenario} ({result.weeks} weeks)")
    table.add_column("tick", justify="right")
    table.add_column("end")
    table.add_column("champion")
    table.add_column("WAPE-28d", justify="right")
    table.add_column("drift", justify="right")
    table.add_column("action")
    for tick in result.ticks:
        snapshot = tick.snapshot
        wape = "-" if snapshot is None or snapshot.wape is None else f"{snapshot.wape:.3f}"
        drift = "-" if snapshot is None else f"{snapshot.numeric_drift_share:.2f}"
        table.add_row(
            str(tick.tick),
            tick.tick_end.date().isoformat(),
            "-" if tick.champion_version is None else f"v{tick.champion_version}",
            wape,
            drift,
            tick.action,
        )
    Console().print(table)


def _format_value(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, int | float) else str(value)


__all__ = [
    "GATE_APPROVED",
    "GATE_REJECTED",
    "SCHEMA",
    "DrillResult",
    "DrillSummary",
    "GateReportRecord",
    "MetricsSummary",
    "TickRecord",
    "library_versions",
    "render_console",
    "summarize_performance",
]
