"""Evidently drift report adapter (S10, D12).

Decisions use the project's native PSI statistics; this adapter exists for the
human-readable artifacts linked from the drill reports and the dashboard. It renders one
`DataDriftPreset(method="psi")` report over exactly the monitored columns so the HTML and
the native numbers describe the same features. `NullReportRenderer` is the Null Object
used by tests and `--fast` drills, where an HTML report would only slow the run down.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.schema import MONITORED_CATEGORICAL, MONITORED_NUMERIC

if TYPE_CHECKING:  # Evidently takes seconds to import; it loads on the first rendered report.
    from evidently.core.report import Snapshot

_MONITORED_COLUMNS: tuple[str, ...] = (*MONITORED_NUMERIC, *MONITORED_CATEGORICAL)


class EvidentlyReportRenderer:
    """Render a PSI data-drift report with Evidently 0.7."""

    def __init__(self, timestamp: datetime | None = None) -> None:
        self._timestamp = timestamp

    def render(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        path: Path,
    ) -> Path | None:
        """Render the report at `path` and return it.

        Args:
            reference: the champion's training window.
            current: the monitored window.
            path: output HTML file; parent directories are created.

        Raises:
            ValidationFailedError: inherited from the native drift contract when a
                monitored column is missing from either frame.
        """
        snapshot = self._run(reference, current)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(path))
        return path

    def drift_values(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
        """Return Evidently's per-column PSI values (useful for the agreement test)."""
        payload = self._run(reference, current).dict()
        values: dict[str, float] = {}
        for metric in payload.get("metrics", []):
            config = metric.get("config") or {}
            column = config.get("column")
            value = metric.get("value")
            if (
                config.get("type", "").endswith("ValueDrift")
                and isinstance(column, str)
                and isinstance(value, (int, float))
            ):
                values[column] = float(value)
        return values

    def _run(self, reference: pd.DataFrame, current: pd.DataFrame) -> Snapshot:
        """Build the Evidently snapshot over the monitored columns only."""
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        missing = [
            column
            for column in _MONITORED_COLUMNS
            if column not in reference.columns or column not in current.columns
        ]
        if missing:
            raise ValidationFailedError(
                failed_rules=missing,
                message=f"missing monitored columns: {', '.join(missing)}",
            )
        definition = DataDefinition(
            numerical_columns=list(MONITORED_NUMERIC),
            categorical_columns=list(MONITORED_CATEGORICAL),
        )
        reference_dataset = Dataset.from_pandas(
            reference.loc[:, list(_MONITORED_COLUMNS)],
            data_definition=definition,
        )
        current_dataset = Dataset.from_pandas(
            current.loc[:, list(_MONITORED_COLUMNS)],
            data_definition=definition,
        )
        report = Report(
            [DataDriftPreset(method="psi")],
            metadata={"mll.timestamp": self._timestamp.isoformat() if self._timestamp else ""},
        )
        return report.run(current_data=current_dataset, reference_data=reference_dataset)


class NullReportRenderer:
    """Null Object: report rendering disabled (tests, `--fast` drills)."""

    def render(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        path: Path,
    ) -> Path | None:
        """Return `None` without writing anything."""
        return None


__all__ = ["EvidentlyReportRenderer", "NullReportRenderer"]
