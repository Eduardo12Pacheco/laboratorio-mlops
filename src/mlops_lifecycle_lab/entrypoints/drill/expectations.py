"""Golden expectations for a drill run (S15, D19).

A golden file stores the per-tick event-type sequence and the headline metrics. Reports
are compared with an explicit relative tolerance, and the first-retrain tick may move by
one week: when it does, every later tick is compared with the corresponding shift, exactly
as the plan allows. `--check` exits 4 and prints every mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlops_lifecycle_lab.entrypoints.drill.report import DrillResult

GOLDEN_SCHEMA = "drill-golden/v1"
DEFAULT_TOLERANCE = 0.05
DEFAULT_RETRAIN_SLACK = 1


@dataclass(frozen=True, slots=True)
class Mismatch:
    """One difference between a run and its golden file."""

    path: str
    expected: object
    actual: object

    def __str__(self) -> str:
        return f"{self.path}: expected {self.expected!r}, got {self.actual!r}"


def make_golden(
    result: DrillResult,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    retrain_tick_slack: int = DEFAULT_RETRAIN_SLACK,
) -> dict[str, Any]:
    """Build the golden document for a verified run."""
    return {
        "schema": GOLDEN_SCHEMA,
        "scenario": result.scenario,
        "weeks": result.weeks,
        "tolerance": tolerance,
        "retrain_tick_slack": retrain_tick_slack,
        "first_retrain_tick": result.first_retrain_tick,
        "ticks": [{"tick": tick.tick, "events": tick.event_types()} for tick in result.ticks],
        "metrics": {
            "as_served": result.summary.as_served.to_dict(),
            "counterfactual_frozen_v1": result.summary.counterfactual_frozen_v1.to_dict(),
        },
        "summary": {
            "promotions": result.summary.promotions,
            "rejections": result.summary.rejections,
            "rollbacks": result.summary.rollbacks,
            "quarantined_batches": result.summary.quarantined_batches,
            "versions": result.summary.versions,
        },
    }


def load_golden(path: Path) -> dict[str, Any]:
    """Load and validate a golden document.

    Raises:
        ValueError: when the file is not a `drill-golden/v1` document.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != GOLDEN_SCHEMA:
        raise ValueError(f"{path} has an invalid schema; expected {GOLDEN_SCHEMA}")
    return payload


def compare(result: DrillResult, golden: dict[str, Any]) -> list[Mismatch]:
    """Return every mismatch between `result` and `golden` (empty means pass)."""
    mismatches: list[Mismatch] = []
    if golden.get("schema") != GOLDEN_SCHEMA:
        return [Mismatch("schema", GOLDEN_SCHEMA, golden.get("schema"))]
    if result.scenario != golden.get("scenario"):
        mismatches.append(Mismatch("scenario", golden.get("scenario"), result.scenario))
    golden_weeks = int(golden.get("weeks", 0))
    if result.weeks > golden_weeks:
        mismatches.append(Mismatch("weeks", f"<= {golden_weeks}", result.weeks))

    slack = int(golden.get("retrain_tick_slack", DEFAULT_RETRAIN_SLACK))
    golden_retrain = golden.get("first_retrain_tick")
    actual_retrain = result.first_retrain_tick
    shift = 0
    if golden_retrain is not None and actual_retrain is not None:
        shift = actual_retrain - int(golden_retrain)
        if abs(shift) > slack:
            mismatches.append(
                Mismatch("first_retrain_tick", f"{golden_retrain} +/- {slack}", actual_retrain)
            )
            shift = 0
    elif (golden_retrain is None) != (actual_retrain is None):
        mismatches.append(Mismatch("first_retrain_tick", golden_retrain, actual_retrain))

    golden_ticks = {int(entry["tick"]): entry for entry in golden.get("ticks", [])}
    threshold = (
        min(actual_retrain, int(golden_retrain))
        if actual_retrain is not None and golden_retrain is not None
        else None
    )
    for tick in result.ticks:
        shifted = threshold is not None and tick.tick >= threshold
        expected_tick = tick.tick - shift if shifted else tick.tick
        entry = golden_ticks.get(expected_tick)
        if entry is None:
            mismatches.append(Mismatch(f"ticks[{tick.tick}].events", "a golden tick", "missing"))
            continue
        expected_events = list(entry.get("events", []))
        actual_events = tick.event_types()
        if expected_events != actual_events:
            mismatches.append(
                Mismatch(f"ticks[{tick.tick}].events", expected_events, actual_events)
            )

    if result.weeks == golden_weeks:
        tolerance = float(golden.get("tolerance", DEFAULT_TOLERANCE))
        metrics = golden.get("metrics", {})
        _compare_metric(mismatches, "as_served", metrics.get("as_served"), result, tolerance)
        _compare_metric(
            mismatches,
            "counterfactual_frozen_v1",
            metrics.get("counterfactual_frozen_v1"),
            result,
            tolerance,
        )
        summary = golden.get("summary", {})
        actual_summary = {
            "promotions": result.summary.promotions,
            "rejections": result.summary.rejections,
            "rollbacks": result.summary.rollbacks,
            "quarantined_batches": result.summary.quarantined_batches,
            "versions": result.summary.versions,
        }
        for key, actual in actual_summary.items():
            expected = summary.get(key)
            if expected is not None and int(expected) != actual:
                mismatches.append(Mismatch(f"summary.{key}", expected, actual))
    return mismatches


def _compare_metric(
    mismatches: list[Mismatch],
    name: str,
    expected: object,
    result: DrillResult,
    tolerance: float,
) -> None:
    if not isinstance(expected, dict):
        return
    actual = (
        result.summary.as_served if name == "as_served" else result.summary.counterfactual_frozen_v1
    )
    for metric in ("rmse", "mae", "wape"):
        if metric not in expected:
            continue
        wanted = float(expected[metric])
        got = float(getattr(actual, metric))
        allowed = abs(wanted) * tolerance
        if abs(got - wanted) > allowed:
            mismatches.append(Mismatch(f"metrics.{name}.{metric}", wanted, got))
    if "n" in expected and int(expected["n"]) != actual.n:
        mismatches.append(Mismatch(f"metrics.{name}.n", expected["n"], actual.n))


__all__ = [
    "DEFAULT_RETRAIN_SLACK",
    "DEFAULT_TOLERANCE",
    "GOLDEN_SCHEMA",
    "Mismatch",
    "compare",
    "load_golden",
    "make_golden",
]
