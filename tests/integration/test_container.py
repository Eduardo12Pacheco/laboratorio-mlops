"""Composition-root and replay-source tests (T7).

`build_container` is the only place that wires adapters to use cases. It must create the
operational store and the MLflow database in the requested workspace, honour `MLL_FAST`
for report rendering, and expose the same ports every entrypoint consumes. The replay
source must keep labels out of the scoring frame: they only travel in the separate label
frame, so the serving path can never see the future.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset
from mlops_lifecycle_lab.adapters.data.replay_source import ReplayBatchSource
from mlops_lifecycle_lab.adapters.drift.evidently_reports import (
    EvidentlyReportRenderer,
    NullReportRenderer,
)
from mlops_lifecycle_lab.bootstrap.container import Container, build_container
from mlops_lifecycle_lab.config import Settings
from mlops_lifecycle_lab.domain.schema import RAW_COLUMNS
from mlops_lifecycle_lab.domain.windows import TimeWindow


def settings_for(tmp_path: Path, *, fast: bool = True) -> Settings:
    """Settings pointing every artifact at one temporary workspace."""
    return Settings(workspace=tmp_path, fast=fast)


def test_container_builds_a_complete_in_process_stack(tmp_path: Path) -> None:
    container = build_container(settings_for(tmp_path))

    try:
        assert isinstance(container, Container)
        assert container.settings.workspace == tmp_path
        assert (tmp_path / "mll.db").is_file()
        assert (tmp_path / "mlflow.db").is_file()
        assert isinstance(container.renderer, NullReportRenderer)
        assert isinstance(container.batch_source, ReplayBatchSource)
        assert container.lifecycle is not None
        assert container.bootstrap is not None
        assert container.gateway is not None
    finally:
        container.close()


def test_container_uses_evidently_when_fast_mode_is_off(tmp_path: Path) -> None:
    container = build_container(settings_for(tmp_path, fast=False))

    try:
        assert isinstance(container.renderer, EvidentlyReportRenderer)
    finally:
        container.close()


def test_container_rejects_http_serving_until_the_serving_task(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path, fast=True, serving_mode="http")

    with pytest.raises(ValueError, match="in-process"):
        build_container(settings)


def test_replay_source_hides_labels_from_the_scoring_frame() -> None:
    source = ReplayBatchSource(BundledDataset())
    window = TimeWindow(datetime(2012, 1, 1), datetime(2012, 1, 8))

    batch = source.week(window)
    labels = source.labels_for(window)

    assert list(batch.raw.columns) == list(RAW_COLUMNS)
    assert "cnt" not in batch.raw.columns
    assert 0 < len(batch.raw) <= 168
    assert list(labels.columns) == ["entity_id", "label", "label_ts"]
    assert len(labels) == len(batch.raw)
    assert labels["label"].equals(batch.labels["label"])
    assert labels["entity_id"].tolist() == batch.labels["entity_id"].tolist()
