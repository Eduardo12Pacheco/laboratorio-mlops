"""Evaluation use case: champion re-scoring, gates, artifact and audit event."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.application.dto import EvaluationRequest, TrainRequest
from mlops_lifecycle_lab.application.services.evaluation import EvaluateChallenger
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.errors import UnknownModelVersionError
from mlops_lifecycle_lab.domain.events import EventType
from mlops_lifecycle_lab.domain.policies import PolicyConfig, PromotionPolicy
from mlops_lifecycle_lab.domain.windows import TimeWindow
from tests.fakes.clock import FakeClock
from tests.fakes.data import StaticLabeledData, raw_frame, target_of
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.stores import InMemoryEventStore
from tests.fakes.tracker import InMemoryTracker
from tests.fakes.trainer import FakeTrainer, FunctionPredictor

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


def holdout_rows(data: StaticLabeledData) -> pd.DataFrame:
    """Return the holdout rows in the order the evaluation service sees them."""
    frame = data.training_frame(HOLDOUT.end)
    return frame[frame["timestamp"] >= HOLDOUT.start].reset_index(drop=True)


class Harness:
    """Wires training and evaluation against shared in-memory adapters."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.clock = FakeClock(NOW)
        self.data = StaticLabeledData(frame)
        self.registry = InMemoryRegistry()
        self.tracker = InMemoryTracker(self.registry)
        self.events = InMemoryEventStore()
        self.trainer = FakeTrainer()
        self.train_service = TrainCandidate(
            self.data,
            self.trainer,
            self.tracker,
            self.registry,
            self.clock,
            seed=42,
        )
        self.evaluate_service = EvaluateChallenger(
            self.data,
            self.registry,
            self.tracker,
            self.events,
            self.clock,
            PromotionPolicy(PolicyConfig()),
        )

    def train(self) -> object:
        """Train and register one candidate with the standard windows."""
        return self.train_service.execute(
            TrainRequest(
                train_window=TRAIN,
                holdout_window=HOLDOUT,
                trigger_reason="MANUAL",
                tick=1,
            )
        )

    def evaluate(self, version: int, *, verdict: str = "PASS", tick: int = 1) -> object:
        """Evaluate one candidate."""
        return self.evaluate_service.execute(
            EvaluationRequest(
                version=version,
                train_window=TRAIN,
                holdout_window=HOLDOUT,
                training_verdict=verdict,
                tick=tick,
            )
        )

    def perfect_champion(self) -> int:
        """Register and promote v1 with a perfect predictor; return its version."""
        trained = self.train()
        champion = self.registry.promote(trained.version)
        actuals = target_of(holdout_rows(self.data))
        self.registry.set_predictor(
            champion.version,
            FunctionPredictor(lambda features: np.resize(actuals, len(features))),
        )
        return champion.version

    def gate_report_runs(self) -> list[object]:
        """Every tracker run that logged a gate report artifact."""
        return [run for run in self.tracker.runs if "gate_report.json" in run.jsons]


def test_the_first_champion_is_evaluated_with_g1_to_g3_only() -> None:
    harness = Harness(shifted_frame())
    trained = harness.train()
    actuals = target_of(holdout_rows(harness.data))
    harness.registry.set_predictor(
        trained.version,
        FunctionPredictor(lambda features: np.resize(actuals, len(features))),
    )

    result = harness.evaluate(trained.version, tick=1)

    assert result.decision.approved is True
    assert [gate.name for gate in result.gate_report.results] == [
        "DATA_VALID",
        "ABS_WAPE",
        "BEATS_BASELINE",
    ]
    assert result.champion_metrics is None
    assert result.challenger_metrics.rmse == pytest.approx(0.0)
    assert result.baseline_rmse > 0.0
    assert [event.type for event in harness.events.events] == [EventType.GATES_EVALUATED]
    assert harness.events.events[0].tick == 1
    assert harness.events.events[0].model_version == trained.version
    assert harness.events.events[0].payload["approved"] is True


def test_the_gate_report_is_saved_exactly_once_with_its_decision() -> None:
    harness = Harness(shifted_frame())
    trained = harness.train()
    actuals = target_of(holdout_rows(harness.data))
    harness.registry.set_predictor(
        trained.version,
        FunctionPredictor(lambda features: np.resize(actuals, len(features))),
    )

    harness.evaluate(trained.version, tick=4)

    reports = harness.gate_report_runs()
    assert len(reports) == 1
    artifact = reports[0].jsons["gate_report.json"]
    assert artifact["version"] == trained.version
    assert artifact["approved"] is True
    assert [entry["name"] for entry in artifact["results"]] == [
        "DATA_VALID",
        "ABS_WAPE",
        "BEATS_BASELINE",
    ]
    assert reports[0].tags["mll.model_version"] == str(trained.version)


def test_a_later_challenger_is_scored_against_the_champion_on_the_same_holdout() -> None:
    harness = Harness(shifted_frame())
    champion_version = harness.perfect_champion()
    challenger = harness.train()
    actuals = target_of(holdout_rows(harness.data))
    harness.registry.set_predictor(
        challenger.version,
        FunctionPredictor(lambda features: np.resize(actuals + 10.0, len(features))),
    )

    result = harness.evaluate(challenger.version, tick=2)

    assert [gate.name for gate in result.gate_report.results] == [
        "DATA_VALID",
        "ABS_WAPE",
        "BEATS_BASELINE",
        "IMPROVEMENT",
        "SIGNIFICANCE",
        "SEGMENT",
    ]
    assert result.decision.approved is False
    assert set(result.decision.failed_gates) == {"IMPROVEMENT", "SIGNIFICANCE", "SEGMENT"}
    assert result.champion_metrics is not None
    assert result.champion_metrics.rmse == pytest.approx(0.0)
    assert result.challenger_metrics.rmse == pytest.approx(10.0)
    assert harness.events.events[-1].payload["approved"] is False
    assert harness.events.events[-1].payload["failed_gates"] == list(
        result.decision.failed_gates
    )
    assert champion_version == 1


def test_a_failed_training_contract_fails_the_data_valid_gate() -> None:
    harness = Harness(shifted_frame())
    trained = harness.train()
    actuals = target_of(holdout_rows(harness.data))
    harness.registry.set_predictor(
        trained.version,
        FunctionPredictor(lambda features: np.resize(actuals, len(features))),
    )

    result = harness.evaluate(trained.version, verdict="FAIL")

    assert result.decision.approved is False
    assert result.decision.failed_gates == ("DATA_VALID",)
    assert result.gate_report.results[0].observed == "FAIL"


def test_evaluating_an_unknown_version_raises() -> None:
    harness = Harness(shifted_frame())

    with pytest.raises(UnknownModelVersionError, match="99"):
        harness.evaluate(99)


def test_evaluation_uses_the_training_rows_for_the_naive_baseline() -> None:
    harness = Harness(shifted_frame())
    trained = harness.train()
    actuals = target_of(holdout_rows(harness.data))
    harness.registry.set_predictor(
        trained.version,
        FunctionPredictor(lambda features: np.resize(actuals, len(features))),
    )

    result = harness.evaluate(trained.version)

    assert result.baseline_rmse > 0.0
    assert result.decision.approved is True
