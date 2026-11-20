"""Promotion, rejection and rollback use cases over an approved gate decision."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.application.dto import EvaluationRequest, TrainRequest
from mlops_lifecycle_lab.application.services.evaluation import EvaluateChallenger
from mlops_lifecycle_lab.application.services.promotion import (
    KEY_PROBATION_UNTIL,
    KEY_PROBATION_VERSION,
    PromoteChallenger,
    RejectChallenger,
    RollbackChampion,
)
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.errors import (
    GateEvaluationError,
    IllegalTransitionError,
    NoPreviousChampionError,
    UnknownModelVersionError,
)
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.lifecycle import ModelStage
from mlops_lifecycle_lab.domain.policies import PolicyConfig, PromotionPolicy
from mlops_lifecycle_lab.domain.windows import TimeWindow
from tests.fakes.clock import FakeClock
from tests.fakes.data import StaticLabeledData, raw_frame, target_of
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.stores import InMemoryEventStore, InMemoryKeyValueStore
from tests.fakes.tracker import InMemoryTracker
from tests.fakes.trainer import FakeTrainer, FunctionPredictor

NOW = datetime(2012, 6, 10)
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
    """Wire training, evaluation and the three promotion services to fakes."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.clock = FakeClock(NOW)
        self.data = StaticLabeledData(frame)
        self.registry = InMemoryRegistry()
        self.tracker = InMemoryTracker(self.registry)
        self.events = InMemoryEventStore()
        self.kv = InMemoryKeyValueStore()
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
        self.promote_service = PromoteChallenger(
            self.registry,
            self.events,
            self.kv,
            self.clock,
            PolicyConfig(),
        )
        self.reject_service = RejectChallenger(self.registry, self.events, self.clock)
        self.rollback_service = RollbackChampion(
            self.registry, self.events, self.kv, self.clock
        )

    def train(self, tick: int = 1) -> int:
        """Train one candidate version and return its number."""
        result = self.train_service.execute(
            TrainRequest(
                train_window=TRAIN,
                holdout_window=HOLDOUT,
                trigger_reason="MANUAL",
                tick=tick,
            )
        )
        return result.version

    def evaluate(self, version: int, tick: int = 1) -> object:
        """Evaluate a candidate (approval depends on the fake trainer)."""
        return self.evaluate_service.execute(
            EvaluationRequest(
                version=version,
                train_window=TRAIN,
                holdout_window=HOLDOUT,
                training_verdict="PASS",
                tick=tick,
            )
        )

    def approve(self, version: int) -> None:
        """Append an approved GATES_EVALUATED event without running the gates."""
        self.events.append(
            LifecycleEvent(
                ts=self.clock.now(),
                type=EventType.GATES_EVALUATED,
                tick=1,
                model_version=version,
                payload={"approved": True, "failed_gates": [], "reason": "all gates passed"},
            )
        )

    def prepare_approved_candidate(self) -> int:
        """Train a candidate and record an approval for it."""
        version = self.train()
        self.approve(version)
        return version


def test_an_unapproved_candidate_cannot_be_promoted() -> None:
    harness = Harness(shifted_frame())
    version = harness.train()

    with pytest.raises(GateEvaluationError, match="approved"):
        harness.promote_service.execute(version)

    assert harness.registry.get_champion() is None
    assert harness.events.types() == []


def test_an_approved_candidate_is_promoted_and_probation_starts() -> None:
    harness = Harness(shifted_frame())
    version = harness.prepare_approved_candidate()

    promoted = harness.promote_service.execute(version)

    assert promoted.stage is ModelStage.CHAMPION
    assert harness.registry.get_champion() == promoted
    assert harness.registry.get_alias("champion") == promoted
    assert harness.kv.get(KEY_PROBATION_VERSION) == version
    expected_until = (NOW + timedelta(days=21)).isoformat()
    assert harness.kv.get(KEY_PROBATION_UNTIL) == expected_until
    assert harness.events.types()[-2:] == [
        EventType.CHALLENGER_PROMOTED,
        EventType.PROBATION_STARTED,
    ]
    assert harness.events.events[-2].model_version == version
    assert harness.events.events[-1].payload["until"] == expected_until


def test_promotion_archives_the_previous_champion_and_moves_the_aliases() -> None:
    harness = Harness(shifted_frame())
    first = harness.prepare_approved_candidate()
    harness.promote_service.execute(first)
    second = harness.prepare_approved_candidate()

    promoted = harness.promote_service.execute(second)

    assert promoted.version == second
    archived = harness.registry.get_version(first)
    assert archived is not None
    assert archived.stage is ModelStage.ARCHIVED
    previous = harness.registry.get_alias("previous-champion")
    assert previous is not None
    assert previous.version == first
    assert harness.registry.get_alias("challenger") == promoted
    assert harness.events.events[-2].payload["previous_champion"] == first


def test_forced_promotion_requires_a_reason() -> None:
    harness = Harness(shifted_frame())
    version = harness.train()

    with pytest.raises(ValueError, match="reason"):
        harness.promote_service.execute(version, forced=True)
    with pytest.raises(ValueError, match="reason"):
        harness.promote_service.execute(version, forced=True, reason="   ")


def test_forced_promotion_bypasses_the_gates_and_emits_the_event() -> None:
    harness = Harness(shifted_frame())
    version = harness.train()

    promoted = harness.promote_service.execute(
        version, forced=True, reason="drill: poison the champion"
    )

    assert promoted.stage is ModelStage.CHAMPION
    assert harness.events.types()[-3:] == [
        EventType.PROMOTION_FORCED,
        EventType.CHALLENGER_PROMOTED,
        EventType.PROBATION_STARTED,
    ]
    forced = harness.events.events[-3]
    assert forced.payload["reason"] == "drill: poison the champion"
    assert forced.model_version == version


def test_promoting_an_unknown_version_raises() -> None:
    harness = Harness(shifted_frame())

    with pytest.raises(UnknownModelVersionError, match="77"):
        harness.promote_service.execute(77)


def test_promoting_an_already_promoted_version_raises() -> None:
    harness = Harness(shifted_frame())
    version = harness.prepare_approved_candidate()
    harness.promote_service.execute(version)

    with pytest.raises(IllegalTransitionError):
        harness.promote_service.execute(version)


def test_rejection_records_the_failed_gates_and_emits_an_event() -> None:
    harness = Harness(shifted_frame())
    version = harness.train()

    rejected = harness.reject_service.execute(
        version,
        failed_gates=["IMPROVEMENT", "SEGMENT"],
        reason="worse than champion",
    )

    assert rejected.stage is ModelStage.REJECTED
    assert rejected.tags["mll.failed_gates"] == "IMPROVEMENT,SEGMENT"
    assert rejected.tags["mll.reject_reason"] == "worse than champion"
    assert harness.registry.get_alias("challenger") is None
    assert harness.events.events[-1].type is EventType.CHALLENGER_REJECTED
    assert harness.events.events[-1].payload["failed_gates"] == ["IMPROVEMENT", "SEGMENT"]


def test_rollback_restores_the_previous_champion_and_clears_probation() -> None:
    harness = Harness(shifted_frame())
    first = harness.prepare_approved_candidate()
    harness.promote_service.execute(first)
    second = harness.prepare_approved_candidate()
    harness.promote_service.execute(second)

    restored = harness.rollback_service.execute("poisoned challenger")

    assert restored.version == first
    assert restored.stage is ModelStage.CHAMPION
    bad = harness.registry.get_version(second)
    assert bad is not None
    assert bad.stage is ModelStage.ROLLED_BACK
    assert harness.registry.get_champion() == restored
    assert harness.registry.get_alias("previous-champion") is None
    assert harness.kv.get(KEY_PROBATION_VERSION) is None
    assert harness.kv.get(KEY_PROBATION_UNTIL) is None
    event = harness.events.events[-1]
    assert event.type is EventType.ROLLBACK_EXECUTED
    assert event.payload["reason"] == "poisoned challenger"
    assert event.payload["restored_version"] == first
    assert event.payload["rolled_back_version"] == second


def test_a_second_rollback_raises_no_previous_champion() -> None:
    harness = Harness(shifted_frame())
    first = harness.prepare_approved_candidate()
    harness.promote_service.execute(first)

    with pytest.raises(NoPreviousChampionError):
        harness.rollback_service.execute("first rollback")
    with pytest.raises(NoPreviousChampionError):
        harness.rollback_service.execute("second rollback")


def test_rollback_requires_a_reason() -> None:
    harness = Harness(shifted_frame())

    with pytest.raises(ValueError, match="reason"):
        harness.rollback_service.execute("  ")


def test_the_promoted_predictor_is_the_evaluated_artifact() -> None:
    harness = Harness(shifted_frame())
    candidate = harness.train()
    holdout = harness.data.training_frame(HOLDOUT.end)
    actuals = target_of(holdout[holdout["timestamp"] >= HOLDOUT.start].reset_index(drop=True))
    harness.registry.set_predictor(
        candidate,
        FunctionPredictor(lambda features: np.resize(actuals, len(features))),
    )
    harness.evaluate(candidate)

    promoted = harness.promote_service.execute(candidate)

    predictor = harness.registry.load_predictor(promoted.version)
    holdout = harness.data.training_frame(HOLDOUT.end)
    holdout = holdout[holdout["timestamp"] >= HOLDOUT.start].reset_index(drop=True)
    predictions = predictor.predict(build_features(holdout))
    assert np.array_equal(predictions, actuals)
