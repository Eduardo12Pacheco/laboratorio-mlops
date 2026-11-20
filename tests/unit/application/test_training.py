"""Training use case: data assembly, fingerprint, S6 lineage and registry adoption."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.application.dto import TrainRequest
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.errors import NotEnoughLabelsError, ValidationFailedError
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.lifecycle import ModelStage
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from mlops_lifecycle_lab.domain.windows import TimeWindow
from tests.fakes.clock import FakeClock
from tests.fakes.data import StaticLabeledData, raw_frame, target_of
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.tracker import InMemoryTracker
from tests.fakes.trainer import FakeTrainer

NOW = datetime(2012, 1, 1)
TRAIN = TimeWindow(datetime(2011, 1, 1), datetime(2011, 1, 3))
HOLDOUT = TimeWindow(datetime(2011, 1, 3), datetime(2011, 1, 5))


def shifted_frame(shift: int = 20) -> pd.DataFrame:
    """Build 96 hourly rows whose holdout half is shifted by `shift` rentals."""
    frame = raw_frame(periods=96)
    mask = frame["timestamp"] >= HOLDOUT.start
    frame.loc[mask, "cnt"] = frame.loc[mask, "cnt"] + shift
    frame.loc[mask, "registered"] = frame.loc[mask, "registered"] + shift
    return frame


class Harness:
    """Wires the training service to the in-memory doubles."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.clock = FakeClock(NOW)
        self.data = StaticLabeledData(frame)
        self.registry = InMemoryRegistry()
        self.tracker = InMemoryTracker(self.registry)
        self.trainer = FakeTrainer()
        self.service = TrainCandidate(
            self.data,
            self.trainer,
            self.tracker,
            self.registry,
            self.clock,
            seed=42,
        )


def execute(harness: Harness, *, tick: int | None = 5) -> object:
    """Run one training request with the standard windows."""
    return harness.service.execute(
        TrainRequest(
            train_window=TRAIN,
            holdout_window=HOLDOUT,
            trigger_reason="MANUAL",
            tick=tick,
        )
    )


def test_training_registers_a_candidate_with_metrics_and_lineage() -> None:
    harness = Harness(shifted_frame())

    result = execute(harness)

    assert result.version == 1
    assert result.model.stage is ModelStage.CANDIDATE
    assert result.model.run_id == "run-1"
    assert result.model.train_window == TRAIN
    assert result.model.holdout_window == HOLDOUT
    assert result.model.trigger_reason == "MANUAL"
    assert result.model.metrics["wape"] == pytest.approx(result.metrics.wape)
    assert result.training_rows == 48
    assert result.holdout_rows == 48
    assert result.metrics.n == 48
    assert result.baseline_rmse > 0.0
    assert len(result.data_fingerprint) == 64
    assert result.validation.verdict.value == "PASS"
    assert harness.registry.list_versions() == [result.model]


def test_training_is_deterministic_for_the_same_inputs() -> None:
    frame = shifted_frame()
    first = execute(Harness(frame))
    second = execute(Harness(frame))

    assert first.data_fingerprint == second.data_fingerprint
    assert first.metrics == second.metrics
    assert first.version == second.version == 1

    first_harness = Harness(frame)
    second_harness = Harness(frame)
    execute(first_harness)
    execute(second_harness)
    holdout = frame[frame["timestamp"] >= HOLDOUT.start].reset_index(drop=True)
    features = build_features(holdout)
    first_predictions = first_harness.registry.load_predictor(1).predict(features)
    second_predictions = second_harness.registry.load_predictor(1).predict(features)

    assert np.array_equal(first_predictions, second_predictions)


def test_training_logs_every_s6_param_metric_tag_and_artifact() -> None:
    harness = Harness(shifted_frame())

    execute(harness, tick=7)

    run = harness.tracker.latest_run()
    assert run.name.startswith("train-MANUAL")
    assert run.closed is True
    assert run.params["algorithm"] == "fake"
    assert run.params["seed"] == 42
    assert run.tags["mll.trigger_reason"] == "MANUAL"
    assert run.tags["mll.feature_spec_version"] == "1"
    assert run.tags["mll.git_sha"] == "unknown"
    assert run.tags["mll.train_window"] == "2011-01-01T00:00:00/2011-01-03T00:00:00"
    assert run.tags["mll.holdout_window"] == "2011-01-03T00:00:00/2011-01-05T00:00:00"
    assert set(run.metrics) >= {
        "val_rmse",
        "val_mae",
        "val_wape",
        "val_bias",
        "baseline_rmse",
        "rows",
    }
    assert run.metrics["rows"] == 48.0
    assert set(run.jsons) == {"validation_report.json", "feature_columns.json"}
    assert run.jsons["feature_columns.json"] == list(FEATURE_COLUMNS)
    assert len(run.models) == 1
    trained, input_example = run.models[0]
    assert list(input_example.columns) == list(FEATURE_COLUMNS)
    assert np.isfinite(trained.predictor.predict(input_example)).all()


def test_a_failed_training_contract_prevents_training_and_registration() -> None:
    frame = raw_frame(periods=96)
    frame.loc[0, "season"] = 9
    harness = Harness(frame)

    with pytest.raises(ValidationFailedError) as caught:
        execute(harness)

    assert "season_domain" in caught.value.failed_rules
    assert harness.trainer.frames == []
    assert harness.tracker.runs == []
    assert harness.registry.list_versions() == []


def test_an_empty_training_window_is_rejected() -> None:
    harness = Harness(raw_frame(start=datetime(2011, 1, 3), periods=48))

    with pytest.raises(NotEnoughLabelsError, match="training window"):
        execute(harness)


def test_an_empty_holdout_window_is_rejected() -> None:
    harness = Harness(raw_frame(start=datetime(2011, 1, 1), periods=48))

    with pytest.raises(NotEnoughLabelsError, match="holdout"):
        execute(harness)


def test_the_data_fingerprint_changes_when_a_training_label_changes() -> None:
    base = shifted_frame()
    changed = base.copy()
    changed.loc[0, "cnt"] = changed.loc[0, "cnt"] + 1
    changed.loc[0, "registered"] = changed.loc[0, "registered"] + 1

    first = execute(Harness(base))
    second = execute(Harness(changed))

    assert first.data_fingerprint != second.data_fingerprint


def test_training_never_mutates_the_repository_frame() -> None:
    frame = shifted_frame()
    harness = Harness(frame)
    before = harness.data.frame.copy(deep=True)

    execute(harness)

    assert harness.data.frame.equals(before)
    assert target_of(harness.trainer.frames[0]).shape == (48,)
