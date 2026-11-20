"""Monitoring snapshot use case: trailing metrics, drift and prediction PSI (S10).

The snapshot is the bridge between the prediction log, the champion's training reference
and the automated lifecycle: it must filter by champion version and by the trailing
window, report NaN metrics as `None` when there is no evidence, and persist both the
snapshot and the audit event that the dashboard reads.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.application.dto import (
    LabelRecord,
    PredictionRecord,
    TrainRequest,
)
from mlops_lifecycle_lab.application.services.monitoring import ComputeSnapshot
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.errors import UnknownModelVersionError
from mlops_lifecycle_lab.domain.events import EventType
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS, RAW_COLUMNS, TARGET_COLUMN
from mlops_lifecycle_lab.domain.windows import TimeWindow
from tests.fakes.clock import FakeClock
from tests.fakes.data import StaticLabeledData, raw_frame
from tests.fakes.prediction_log import InMemoryPredictionLog
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.stores import InMemoryEventStore, InMemorySnapshotStore
from tests.fakes.tracker import InMemoryTracker
from tests.fakes.trainer import FakeTrainer

NOW = datetime(2012, 4, 1)
WINDOW_DAYS = 28
TRAIN = TimeWindow(datetime(2011, 1, 1), datetime(2011, 10, 1))
HOLDOUT = TimeWindow(datetime(2011, 10, 1), datetime(2012, 1, 1))


def historical_frame() -> pd.DataFrame:
    """A year of hourly rows before the monitored replay window."""
    return raw_frame(start=datetime(2011, 1, 1), periods=24 * 365)


def current_rows(start: datetime = NOW - timedelta(days=WINDOW_DAYS)) -> pd.DataFrame:
    """Rows of the trailing window (winter 2012), used as production batch."""
    return raw_frame(start=start, periods=24 * WINDOW_DAYS, entity_id_start=10_000)


def record_for(
    row: pd.Series,
    *,
    model_version: int = 1,
    prediction: float | None = None,
    scored_at: datetime = NOW,
) -> PredictionRecord:
    """Build one persisted prediction from a raw row."""
    features = build_features(row.to_frame().T).iloc[0]
    return PredictionRecord(
        prediction_id=f"p-{int(row['entity_id'])}",
        entity_id=int(row["entity_id"]),
        event_ts=pd.Timestamp(row["timestamp"]).to_pydatetime(),
        scored_at=scored_at,
        model_name="bike-demand",
        model_version=model_version,
        features={name: float(features[name]) for name in FEATURE_COLUMNS},
        prediction=float(row[TARGET_COLUMN] if prediction is None else prediction),
        source="batch",
    )


class Harness:
    """Wire the monitoring service to in-memory adapters and a trained champion."""

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.clock = FakeClock(NOW)
        self.data = StaticLabeledData(frame if frame is not None else historical_frame())
        self.prediction_log = InMemoryPredictionLog()
        self.registry = InMemoryRegistry()
        self.tracker = InMemoryTracker(self.registry)
        self.events = InMemoryEventStore()
        self.snapshots = InMemorySnapshotStore()
        training = TrainCandidate(
            self.data,
            FakeTrainer(),
            self.tracker,
            self.registry,
            self.clock,
            seed=42,
        )
        trained = training.execute(
            TrainRequest(train_window=TRAIN, holdout_window=HOLDOUT, trigger_reason="BOOTSTRAP")
        )
        self.champion = self.registry.promote(trained.version)
        self.service = ComputeSnapshot(
            prediction_log=self.prediction_log,
            labeled_data=self.data,
            registry=self.registry,
            snapshots=self.snapshots,
            events=self.events,
            clock=self.clock,
            tracker=self.tracker,
            window_days=WINDOW_DAYS,
        )

    def add_predictions(
        self,
        rows: pd.DataFrame,
        *,
        model_version: int | None = None,
        prediction: float | None = None,
        attach_labels: bool = True,
    ) -> list[PredictionRecord]:
        """Persist predictions (and labels) for `rows` under the champion version."""
        version = self.champion.version if model_version is None else model_version
        records = [
            record_for(row, model_version=version, prediction=prediction)
            for _, row in rows.iterrows()
        ]
        self.prediction_log.append(records)
        if attach_labels:
            self.prediction_log.attach_labels(
                [
                    LabelRecord(
                        entity_id=record.entity_id,
                        label=float(rows.iloc[position][TARGET_COLUMN]),
                        label_ts=record.event_ts,
                    )
                    for position, record in enumerate(records)
                ]
            )
        return records


def test_snapshot_computes_trailing_metrics_for_the_champion_only() -> None:
    harness = Harness()
    rows = current_rows()
    harness.add_predictions(rows)

    outside = raw_frame(start=NOW - timedelta(days=40), periods=24, entity_id_start=90_000)
    harness.add_predictions(outside)
    other_version = raw_frame(start=NOW - timedelta(days=3), periods=24, entity_id_start=91_000)
    harness.add_predictions(other_version, model_version=99)

    snapshot = harness.service.execute(NOW, harness.champion.version, tick=13)

    assert snapshot.tick == 13
    assert snapshot.model_version == harness.champion.version
    assert snapshot.n_rows == len(rows)
    expected = harness.prediction_log.labeled_predictions(
        NOW - timedelta(days=WINDOW_DAYS), NOW, model_version=harness.champion.version
    )
    assert snapshot.rmse == pytest.approx(0.0, abs=1e-9)
    assert snapshot.mae == pytest.approx(0.0, abs=1e-9)
    assert snapshot.wape == pytest.approx(0.0, abs=1e-9)
    assert snapshot.bias == pytest.approx(0.0, abs=1e-9)
    assert len(expected) == len(rows)
    assert set(snapshot.psi) == {"temp", "atemp", "hum", "windspeed", "season", "weathersit"}
    assert snapshot.prediction_psi is not None
    assert snapshot.prediction_psi >= 0.0


def test_snapshot_persists_itself_and_the_audit_event() -> None:
    harness = Harness()
    harness.add_predictions(current_rows())

    snapshot = harness.service.execute(NOW, harness.champion.version, tick=7)

    assert harness.snapshots.list() == [snapshot]
    events = harness.events.list(event_type=EventType.SNAPSHOT_COMPUTED)
    assert len(events) == 1
    assert events[0].tick == 7
    assert events[0].model_version == harness.champion.version
    assert events[0].payload["n_rows"] == snapshot.n_rows


def test_snapshot_without_labeled_rows_encodes_metrics_as_none() -> None:
    harness = Harness()

    snapshot = harness.service.execute(NOW, harness.champion.version, tick=1)

    assert snapshot.n_rows == 0
    assert snapshot.rmse is None
    assert snapshot.mae is None
    assert snapshot.wape is None
    assert snapshot.bias is None
    assert snapshot.numeric_drift_share == 0.0
    assert snapshot.prediction_psi is None
    assert dict(snapshot.psi) == {}
    assert harness.snapshots.list() == [snapshot]
    assert len(harness.events.list(event_type=EventType.SNAPSHOT_COMPUTED)) == 1


def test_snapshot_detects_numeric_drift_from_the_training_reference() -> None:
    harness = Harness()
    rows = current_rows()
    rows = rows.copy()
    rows["temp"] = np.linspace(0.85, 1.0, len(rows))
    rows["atemp"] = np.linspace(0.88, 1.0, len(rows))
    harness.add_predictions(rows)

    snapshot = harness.service.execute(NOW, harness.champion.version, tick=4)

    assert snapshot.psi["temp"] > 0.2
    assert snapshot.psi["atemp"] > 0.2
    assert snapshot.numeric_drift_share >= 0.5


def test_snapshot_uses_the_reference_predictions_artifact() -> None:
    harness = Harness()
    harness.add_predictions(current_rows())

    snapshot = harness.service.execute(NOW, harness.champion.version, tick=5)

    artifact = harness.tracker.load_json("reference_predictions.json", harness.champion.run_id)
    assert artifact is not None
    assert artifact["count"] == len(artifact["values"])
    assert snapshot.prediction_psi is not None
    assert np.isfinite(snapshot.prediction_psi)


def test_snapshot_rejects_an_unknown_champion_version() -> None:
    harness = Harness()

    with pytest.raises(UnknownModelVersionError, match="42"):
        harness.service.execute(NOW, 42, tick=1)


def test_current_rows_shape_matches_the_raw_contract() -> None:
    rows = current_rows()

    assert list(rows.columns) == [*RAW_COLUMNS, "casual", "registered", "cnt"]
