"""Pure data-contract validation engine.

Every rule is evaluated (no short-circuit) and returns a `RuleResult` with its own
violation count, severity and tolerance. A failed ERROR rule makes the whole report
`FAIL`; a failed WARN rule makes it `WARN`. The engine never mutates its input, so the
same frame can be validated and then used for scoring or training.

`hum_positive` is the one rule that flags real data (22 rows on 2011-03-10). The
planning note mentioned a 0.5% tolerance, but S3 and the T2 acceptance require the
bundled dataset to be exactly one failed warning with 22 violations, so the tolerance
is zero: any zero humidity is reported, never silently accepted.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import pandas as pd

from mlops_lifecycle_lab.domain.schema import LABEL_COLUMNS, RAW_COLUMNS

Context = Literal["batch", "training"]

_MAX_GAP = pd.Timedelta(hours=48)
_MAX_BATCH_ROWS = 1000


class Severity(StrEnum):
    """How much a rule violation matters for the verdict."""

    ERROR = "ERROR"
    WARN = "WARN"


class Verdict(StrEnum):
    """Outcome of a validation run."""

    PASS = "PASS"  # noqa: S105 - verdict label, not a credential
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class RuleResult:
    """One rule evaluation."""

    name: str
    severity: Severity
    total: int
    violations: int
    tolerance: float
    passed: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation used by the CLI and reports."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "total": self.total,
            "violations": self.violations,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregated verdict plus every rule result, in evaluation order."""

    verdict: Verdict
    rules: tuple[RuleResult, ...]
    rows: int
    context: Context

    @property
    def failed_rules(self) -> tuple[str, ...]:
        """Names of rules that did not pass, in evaluation order."""
        return tuple(rule.name for rule in self.rules if not rule.passed)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation used by the CLI and reports."""
        return {
            "verdict": self.verdict.value,
            "rows": self.rows,
            "context": self.context,
            "failed_rules": list(self.failed_rules),
            "rules": {rule.name: rule.to_dict() for rule in self.rules},
        }


def _rule(
    name: str,
    severity: Severity,
    total: int,
    violations: int,
    tolerance: float = 0.0,
    message: str | None = None,
) -> RuleResult:
    passed = violations <= tolerance * total
    if message is None:
        message = "ok" if passed else f"{violations} of {total} checked values violate the rule"
    return RuleResult(
        name=name,
        severity=severity,
        total=total,
        violations=violations,
        tolerance=tolerance,
        passed=passed,
        message=message,
    )


def _missing(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


def _missing_rule(name: str, severity: Severity, columns: Iterable[str]) -> RuleResult:
    missing = list(columns)
    return _rule(
        name,
        severity,
        total=1,
        violations=1,
        tolerance=0.0,
        message=f"missing required columns: {', '.join(missing)}",
    )


def _required_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> RuleResult:
    missing = _missing(frame, required)
    return _rule(
        "required_columns",
        Severity.ERROR,
        total=len(required),
        violations=len(missing),
        tolerance=0.0,
        message=("ok" if not missing else f"missing required columns: {', '.join(missing)}"),
    )


def _no_nulls(frame: pd.DataFrame, required: tuple[str, ...]) -> RuleResult:
    missing = _missing(frame, required)
    if missing:
        return _missing_rule("no_nulls", Severity.ERROR, missing)
    violations = int(frame[list(required)].isna().any(axis=1).sum())
    return _rule("no_nulls", Severity.ERROR, total=len(frame), violations=violations)


def _timestamps(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["timestamp"], errors="coerce")


def _timestamp_hourly(frame: pd.DataFrame) -> RuleResult:
    if _missing(frame, ["timestamp"]):
        return _missing_rule("timestamp_hourly", Severity.ERROR, ["timestamp"])
    timestamps = _timestamps(frame)
    bad = (
        (timestamps.dt.minute != 0)
        | (timestamps.dt.second != 0)
        | (timestamps.dt.microsecond != 0)
        | timestamps.isna()
    )
    return _rule(
        "timestamp_hourly",
        Severity.ERROR,
        total=len(frame),
        violations=int(bad.sum()),
    )


def _timestamp_unique(frame: pd.DataFrame) -> RuleResult:
    if _missing(frame, ["timestamp"]):
        return _missing_rule("timestamp_unique", Severity.ERROR, ["timestamp"])
    timestamps = _timestamps(frame)
    violations = int(timestamps.duplicated().sum())
    return _rule("timestamp_unique", Severity.ERROR, total=len(frame), violations=violations)


def _timestamp_monotonic(frame: pd.DataFrame) -> RuleResult:
    if _missing(frame, ["timestamp"]):
        return _missing_rule("timestamp_monotonic", Severity.ERROR, ["timestamp"])
    timestamps = _timestamps(frame)
    passed = bool(timestamps.is_monotonic_increasing)
    return _rule(
        "timestamp_monotonic",
        Severity.ERROR,
        total=1,
        violations=0 if passed else 1,
        tolerance=0.0,
        message="ok" if passed else "timestamps are not monotonically increasing",
    )


def _season_domain(frame: pd.DataFrame) -> RuleResult:
    if _missing(frame, ["season"]):
        return _missing_rule("season_domain", Severity.ERROR, ["season"])
    values = pd.to_numeric(frame["season"], errors="coerce")
    bad = ~values.isin([1, 2, 3, 4])
    return _rule("season_domain", Severity.ERROR, total=len(frame), violations=int(bad.sum()))


def _binary_domain(frame: pd.DataFrame) -> RuleResult:
    columns = ["holiday", "workingday"]
    missing = _missing(frame, columns)
    if missing:
        return _missing_rule("binary_domain", Severity.ERROR, missing)
    bad = pd.Series(False, index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = bad | ~values.isin([0, 1])
    return _rule("binary_domain", Severity.ERROR, total=len(frame), violations=int(bad.sum()))


def _weathersit_domain(frame: pd.DataFrame) -> RuleResult:
    if _missing(frame, ["weathersit"]):
        return _missing_rule("weathersit_domain", Severity.ERROR, ["weathersit"])
    values = pd.to_numeric(frame["weathersit"], errors="coerce")
    bad = ~values.isin([1, 2, 3, 4])
    return _rule("weathersit_domain", Severity.ERROR, total=len(frame), violations=int(bad.sum()))


def _numeric_range(frame: pd.DataFrame) -> RuleResult:
    columns = ["temp", "atemp", "hum", "windspeed"]
    missing = _missing(frame, columns)
    if missing:
        return _missing_rule("numeric_range", Severity.ERROR, missing)
    bad = pd.Series(False, index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = bad | ~values.between(0, 1)
    return _rule("numeric_range", Severity.ERROR, total=len(frame), violations=int(bad.sum()))


def _hum_positive(frame: pd.DataFrame) -> RuleResult:
    if _missing(frame, ["hum"]):
        return _missing_rule("hum_positive", Severity.WARN, ["hum"])
    values = pd.to_numeric(frame["hum"], errors="coerce")
    bad = ~(values > 0)
    return _rule(
        "hum_positive",
        Severity.WARN,
        total=len(frame),
        violations=int(bad.sum()),
        tolerance=0.0,
        message="ok"
        if not int(bad.sum())
        else f"{int(bad.sum())} rows have zero or missing humidity",
    )


def _gap_hours(frame: pd.DataFrame) -> RuleResult:
    if _missing(frame, ["timestamp"]):
        return _missing_rule("gap_hours", Severity.WARN, ["timestamp"])
    timestamps = _timestamps(frame).dropna().drop_duplicates().sort_values()
    gaps = timestamps.diff().dropna()
    violations = int((gaps > _MAX_GAP).sum())
    max_gap = gaps.max() if not gaps.empty else pd.Timedelta(0)
    return _rule(
        "gap_hours",
        Severity.WARN,
        total=len(gaps),
        violations=violations,
        tolerance=0.0,
        message="ok" if not violations else f"largest gap is {max_gap} > {_MAX_GAP}",
    )


def _label_consistency(frame: pd.DataFrame) -> RuleResult:
    columns = ["cnt", "casual", "registered"]
    missing = _missing(frame, columns)
    if missing:
        return _missing_rule("label_consistency", Severity.ERROR, missing)
    cnt = pd.to_numeric(frame["cnt"], errors="coerce")
    casual = pd.to_numeric(frame["casual"], errors="coerce")
    registered = pd.to_numeric(frame["registered"], errors="coerce")
    bad = (cnt != casual + registered) | (cnt < 0)
    return _rule("label_consistency", Severity.ERROR, total=len(frame), violations=int(bad.sum()))


def _batch_size(frame: pd.DataFrame) -> RuleResult:
    size = len(frame)
    valid = 1 <= size <= _MAX_BATCH_ROWS
    return _rule(
        "batch_size",
        Severity.ERROR,
        total=1,
        violations=0 if valid else 1,
        tolerance=0.0,
        message="ok" if valid else f"batch size {size} is outside [1, {_MAX_BATCH_ROWS}]",
    )


def validate_frame(frame: pd.DataFrame, context: Context = "batch") -> ValidationReport:
    """Validate `frame` against the raw contract for `context`.

    Args:
        frame: raw rows (`RAW_COLUMNS`, plus labels for the training context).
        context: `"batch"` checks the service limits and ignores labels; `"training"`
        requires labels and checks their internal consistency, without a batch cap.

    Returns:
        A `ValidationReport`; the input frame is never mutated.
    """
    if context not in ("batch", "training"):
        raise ValueError(f"unknown validation context: {context!r}")

    required = (*RAW_COLUMNS, *LABEL_COLUMNS) if context == "training" else RAW_COLUMNS
    rules = [
        _required_columns(frame, required),
        _no_nulls(frame, required),
        _timestamp_hourly(frame),
        _timestamp_unique(frame),
        _timestamp_monotonic(frame),
        _season_domain(frame),
        _binary_domain(frame),
        _weathersit_domain(frame),
        _numeric_range(frame),
        _hum_positive(frame),
        _gap_hours(frame),
    ]
    rules.append(_label_consistency(frame) if context == "training" else _batch_size(frame))

    verdict = Verdict.PASS
    if any(not rule.passed and rule.severity is Severity.ERROR for rule in rules):
        verdict = Verdict.FAIL
    elif any(not rule.passed and rule.severity is Severity.WARN for rule in rules):
        verdict = Verdict.WARN
    return ValidationReport(
        verdict=verdict,
        rules=tuple(rules),
        rows=len(frame),
        context=context,
    )
