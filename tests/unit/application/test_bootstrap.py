"""Bootstrap use case: v1 on the S5 windows, first-champion gates and idempotence."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.application.services.bootstrap import BootstrapChampion
from mlops_lifecycle_lab.application.services.evaluation import EvaluateChallenger
from mlops_lifecycle_lab.application.services.promotion import (
    PromoteChallenger,
    RejectChallenger,
)
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.errors import GateEvaluationError
from mlops_lifecycle_lab.domain.events import EventType
from mlops_lifecycle_lab.domain.lifecycle import ModelStage
from mlops_lifecycle_lab.domain.policies import PolicyConfig, PromotionPolicy
from tests.fakes.clock import FakeClock
from tests.fakes.data import StaticLabeledData, raw_frame
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.stores import InMemoryEventStore, InMemoryKeyValueStore
from tests.fakes.tracker import InMemoryTracker
from tests.fakes.trainer import FakeTrainer

NOW = datetime(2012, 1, 1)


def hourly_mean_predictor(frame: pd.DataFrame) -> object:
    """A deterministic learner that memorises the mean demand per hour."""
    profile = frame.groupby("hr")["cnt"].mean().to_dict()
    return lambda features: np.asarray(
        [profile[int(hour)] for hour in features["hr"]], dtype=np.float64
    )


class Harness:
    """Wire the bootstrap service and its dependencies to in-memory doubles."""

    def __init__(self, reject: bool = False) -> None:
        self.clock = FakeClock(NOW)
        self.data = StaticLabeledData(raw_frame(periods=24 * 365))
        self.registry = InMemoryRegistry()
        self.tracker = InMemoryTracker(self.registry)
        self.events = InMemoryEventStore()
        self.kv = InMemoryKeyValueStore()
        self.trainer = FakeTrainer(
            predictor_factory=(
                (lambda frame: lambda features: np.zeros(len(features)))
                if reject
                else hourly_mean_predictor
            )
        )
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
        self.promote_service = PromoteChallenger(
            self.registry, self.events, self.kv, self.clock, PolicyConfig()
        )
        self.reject_service = RejectChallenger(self.registry, self.events, self.clock)
        self.bootstrap = BootstrapChampion(
            self.registry,
            self.events,
            self.clock,
            self.train_service,
            self.evaluate_service,
            self.promote_service,
            self.reject_service,
        )


def test_bootstrap_trains_v1_on_the_s5_windows_and_promotes_it() -> None:
    harness = Harness()

    champion = harness.bootstrap.execute()

    assert champion.version == 1
    assert champion.stage is ModelStage.CHAMPION
    assert harness.registry.get_champion() == champion
    assert [event.type for event in harness.events.events] == [
        EventType.GATES_EVALUATED,
        EventType.CHALLENGER_PROMOTED,
        EventType.PROBATION_STARTED,
        EventType.BOOTSTRAP_COMPLETED,
    ]
    assert {event.tick for event in harness.events.events} == {0}
    assert harness.events.events[-1].model_version == champion.version
    assert harness.data.requested_ends[0] == datetime(2012, 1, 1)

    training_run = harness.tracker.runs[0]
    assert training_run.tags["mll.train_window"] == ("2011-01-01T00:00:00/2011-10-01T00:00:00")
    assert training_run.tags["mll.holdout_window"] == ("2011-10-01T00:00:00/2012-01-01T00:00:00")
    assert training_run.tags["mll.trigger_reason"] == "BOOTSTRAP"


def test_bootstrap_uses_only_the_first_champion_gates() -> None:
    harness = Harness()

    harness.bootstrap.execute()

    evaluation_run = next(run for run in harness.tracker.runs if "gate_report.json" in run.jsons)
    artifact = evaluation_run.jsons["gate_report.json"]
    assert [entry["name"] for entry in artifact["results"]] == [
        "DATA_VALID",
        "ABS_WAPE",
        "BEATS_BASELINE",
    ]
    assert artifact["approved"] is True


def test_bootstrap_is_idempotent_when_a_champion_already_exists() -> None:
    harness = Harness()
    first = harness.bootstrap.execute()
    runs_after_first = len(harness.tracker.runs)
    versions_after_first = harness.registry.list_versions()

    second = harness.bootstrap.execute()

    assert second == first
    assert len(harness.tracker.runs) == runs_after_first
    assert harness.registry.list_versions() == versions_after_first
    assert harness.events.types().count(EventType.BOOTSTRAP_COMPLETED) == 1


def test_bootstrap_rejects_a_candidate_that_fails_the_first_champion_gates() -> None:
    harness = Harness(reject=True)

    with pytest.raises(GateEvaluationError, match="bootstrap"):
        harness.bootstrap.execute()

    assert harness.registry.get_champion() is None
    candidate = harness.registry.get_version(1)
    assert candidate is not None
    assert candidate.stage is ModelStage.REJECTED
    assert [event.type for event in harness.events.events] == [
        EventType.GATES_EVALUATED,
        EventType.CHALLENGER_REJECTED,
    ]
