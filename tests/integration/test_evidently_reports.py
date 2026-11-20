"""Integration test: Evidently HTML report adapter (S10, D12).

The human-readable report is generated from the same monitored columns as the native
PSI statistics. This test pins both the artifact (a real HTML report over 100 KB) and the
agreement with the native statistic that actually drives decisions: for the March-2012
window against the 2011 training reference, `temp` must be flagged by both.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset
from mlops_lifecycle_lab.adapters.drift.evidently_reports import (
    EvidentlyReportRenderer,
    NullReportRenderer,
)
from mlops_lifecycle_lab.domain.drift import psi_numeric
from mlops_lifecycle_lab.domain.schema import MONITORED_CATEGORICAL, MONITORED_NUMERIC


def monitored_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference (2011 training) and current (March 2012) monitored feature frames."""
    raw = BundledDataset().raw()
    reference = raw.loc[
        (raw["timestamp"] >= datetime(2011, 1, 1)) & (raw["timestamp"] < datetime(2011, 10, 1))
    ]
    current = raw.loc[
        (raw["timestamp"] >= datetime(2012, 3, 1)) & (raw["timestamp"] < datetime(2012, 3, 29))
    ]
    return reference, current


def test_renderer_writes_a_real_html_report(tmp_path: Path) -> None:
    reference, current = monitored_windows()
    renderer = EvidentlyReportRenderer()
    target = tmp_path / "evidently" / "drift-2012-03.html"

    rendered = renderer.render(reference, current, target)

    assert rendered == target
    assert target.is_file()
    assert target.stat().st_size > 100_000
    text = target.read_text(encoding="utf-8", errors="ignore")
    assert "html" in text.lower()
    assert "temp" in text


def test_evidently_and_native_statistics_agree_that_temp_drifted() -> None:
    reference, current = monitored_windows()

    values = EvidentlyReportRenderer().drift_values(reference, current)

    assert set(values) >= set(MONITORED_NUMERIC) | set(MONITORED_CATEGORICAL)
    assert values["temp"] > 0.2
    assert psi_numeric(reference["temp"], current["temp"]) > 0.2


def test_null_renderer_disables_reports_without_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    reference, current = monitored_windows()
    target = tmp_path / "never.html"

    result = NullReportRenderer().render(reference, current, target)

    assert result is None
    assert not target.exists()
