"""Lifecycle tick use case: one replay week, end to end, with a run ledger (S14).

The tick is the orchestration seam of the lab: validate the batch, score it through the
gateway, ingest the delayed labels, compute the monitoring snapshot, then either watch
probation or run the retrain/gate/promote-or-reject protocol. These tests drive it with
in-memory doubles and assert the event order, the quarantine path, the probation rollback,
the drift-only veto and the failure/idempotency guarantees.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.application.dto import (
    RawBatch,
    encode_version_tags,
)
from mlops_lifecycle_lab.application.pipeline import PipelineError
from mlops_lifecycle_lab.application.ports import Predictor
from mlops_lifecycle_lab.application.services.evaluation import EvaluateChallenger
from mlops_lifecycle_lab.application.services.lifecycle import (
    RunLifecycleTick,
    TickInterventions,
    TickRequest,
)
from mlops_lifecycle_lab.application.services.monitoring import ComputeSnapshot
from mlops_lifecycle_lab.application.services.promotion import (
    KEY_PROBATION_UNTIL,
    KEY_PROBATION_VERSION,
    PromoteChallenger,
    RejectChallenger,
    RollbackChampion,
)
from mlops_lifecycle_lab.application.services.scoring import InProcessGateway, PredictionService
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.events import EventType
from mlops_lifecycle_lab.domain.lifecycle import ModelStage, ModelVersion
from mlops_lifecycle_lab.domain.policies import (
    PolicyConfig,
    ProbationPolicy,
    PromotionPolicy,
    RetrainTriggerPolicy,
)
from mlops_lifecycle_lab.domain.schema import RAW_COLUMNS, TARGET_COLUMN
from mlops_lifecycle_lab.domain.windows import TimeWindow, tick_window
from tests.fakes.clock import FakeClock
from tests.fakes.data import StaticLabeledData, raw_frame
from tests.fakes.prediction_log import InMemoryPredictionLog
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.stores import (
    InMemoryEventStore,
    InMemoryKeyValueStore,
    InMemorySnapshotStore,
    RecordingLedger,
)
from tests.fakes.tracker import InMemoryTracker
from tests.fakes.trainer import FakeTrainer, FunctionPredictor, MeanPredictor

REPLAY_START = datetime(2012, 1, 1)
CONFIG = PolicyConfig()


def demand(rows: pd.DataFrame) -> pd.DataFrame:
    """Give the frame a deterministic demand target and consistent label columns."""
    frame = rows.copy()
    hours = pd.to_datetime(frame["timestamp"]).dt.hour.to_numpy()
    counts = np.rint(100.0 + 20.0 * hours + 200.0 * frame["temp"]).astype("int64")
    frame[TARGET_COLUMN] = counts
    frame["casual"] = counts // 4
    frame["registered"] = counts - frame["casual"]
    return frame


def history(periods: int = 24 * 365) -> pd.DataFrame:
    """Historical rows (2011 plus a little 2012) with deterministic demand."""
    return demand(raw_frame(start=datetime(2011, 1, 1), periods=periods))


def production(periods: int = 24 * 70) -> pd.DataFrame:
    """Replayed production rows (2012) with deterministic demand."""
    return demand(raw_frame(start=REPLAY_START, periods=periods, entity_id_start=1_000_000))


def formula(features: pd.DataFrame) -> np.ndarray:
    """The exact demand function used to build the synthetic frames."""
    return np.asarray(100.0 + 20.0 * features["hr"] + 200.0 * features["temp"], dtype=np.float64)


class WeeklySource:
    """`BatchSource` over a fixed frame, with optional per-window corruption."""

    def __init__(
        self,
        frame: pd.DataFrame,
        overrides: dict[datetime, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
    ) -> None:
        self._frame = frame
        self._overrides = overrides or {}
        self.requested: list[TimeWindow] = []

    def week(self, window: TimeWindow) -> RawBatch:
        """Return raw rows without labels plus the labels available at the tick end."""
        self.requested.append(window)
        timestamps = pd.to_datetime(self._frame["timestamp"])
        rows = self._frame.loc[
            (timestamps >= window.start) & (timestamps < window.end)
        ].reset_index(drop=True)
        if window.start in self._overrides:
            rows = self._overrides[window.start](rows).reset_index(drop=True)
        labels = pd.DataFrame(
            {
                "entity_id": rows["entity_id"].to_numpy(dtype=np.int64),
                "label": rows[TARGET_COLUMN].to_numpy(dtype=np.float64),
                "label_ts": rows["timestamp"].to_numpy(),
            }
        )
        return RawBatch(raw=rows.loc[:, list(RAW_COLUMNS)], labels=labels)


class SpyGateway(InProcessGateway):
    """In-process gateway that counts calls (to prove a quarantine never scores)."""

    def __init__(self, service: PredictionService, prediction_log: InMemoryPredictionLog) -> None:
        super().__init__(service, prediction_log)
        self.batch_calls = 0
        self.label_calls = 0

    def predict_batch(self, raw_rows: pd.DataFrame) -> list[object]:
        """Count and delegate."""
        self.batch_calls += 1
        return super().predict_batch(raw_rows)

    def submit_labels(self, labels: object) -> None:
        """Count and delegate."""
        self.label_calls += 1
        super().submit_labels(labels)  # type: ignore[arg-type]


class FailingGateway:
    """Gateway that always fails, to exercise retry + ledger failure paths."""

    def __init__(self) -> None:
        self.calls = 0

    def predict_batch(self, raw_rows: pd.DataFrame) -> list[object]:
        """Fail on every attempt."""
        self.calls += 1
        raise RuntimeError("gateway unavailable")

    def submit_labels(self, labels: object) -> None:
        """Never reached in the failure test."""


def register(
    registry: InMemoryRegistry,
    version: int,
    predictor: Predictor,
    *,
    train_window: TimeWindow,
    holdout_window: TimeWindow,
) -> ModelVersion:
    """Register and promote a synthetic model version with a fixed predictor."""
    model = ModelVersion(
        name="bike-demand",
        version=version,
        stage=ModelStage.CANDIDATE,
        run_id=f"run-{version}",
        metrics={"rmse": 10.0, "wape": 0.1},
        train_window=train_window,
        holdout_window=holdout_window,
        data_fingerprint=f"fingerprint-{version}",
        feature_spec_version="1",
        created_at=datetime(2012, 1, 1),
        trigger_reason="MANUAL",
    )
    registry.adopt_candidate(version, encode_version_tags(model))
    registry.set_predictor(version, predictor)
    return registry.promote(version)


class Harness:
    """Full in-memory lifecycle wiring, with knobs for the scenario under test."""

    def __init__(
        self,
        *,
        champion_predictor: Predictor | None = None,
        production_frame: pd.DataFrame | None = None,
        trainer: FakeTrainer | None = None,
        gateway: object | None = None,
        overrides: dict[datetime, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
    ) -> None:
        self.clock = FakeClock(REPLAY_START)
        self.data = StaticLabeledData(pd.concat([history(), production()], ignore_index=True))
        self.prediction_log = InMemoryPredictionLog()
        self.registry = InMemoryRegistry()
        self.tracker = InMemoryTracker(self.registry, first_version=2)
        self.events = InMemoryEventStore()
        self.snapshots = InMemorySnapshotStore()
        self.kv = InMemoryKeyValueStore()
        self.ledger = RecordingLedger()
        frame = production() if production_frame is None else production_frame
        self.batch_source = WeeklySource(frame, overrides)
        self.champion = register(
            self.registry,
            1,
            champion_predictor or FunctionPredictor(formula),
            train_window=TimeWindow(datetime(2011, 1, 1), datetime(2011, 10, 1)),
            holdout_window=TimeWindow(datetime(2011, 10, 1), datetime(2012, 1, 1)),
        )
        self.trainer = trainer or FakeTrainer(predictor_factory=lambda _: formula)
        scoring = PredictionService(self.registry, self.prediction_log, self.clock)
        self.gateway = gateway or SpyGateway(scoring, self.prediction_log)
        self.monitoring = ComputeSnapshot(
            prediction_log=self.prediction_log,
            labeled_data=self.data,
            registry=self.registry,
            snapshots=self.snapshots,
            events=self.events,
            clock=self.clock,
            tracker=self.tracker,
            window_days=28,
        )
        training = TrainCandidate(
            self.data, self.trainer, self.tracker, self.registry, self.clock, seed=42
        )
        evaluation = EvaluateChallenger(
            self.data,
            self.registry,
            self.tracker,
            self.events,
            self.clock,
            PromotionPolicy(CONFIG),
        )
        promotion = PromoteChallenger(self.registry, self.events, self.kv, self.clock, CONFIG)
        rejection = RejectChallenger(self.registry, self.events, self.clock)
        rollback = RollbackChampion(self.registry, self.events, self.kv, self.clock)
        self.tick_service = RunLifecycleTick(
            registry=self.registry,
            events=self.events,
            kv=self.kv,
            ledger=self.ledger,
            gateway=self.gateway,  # type: ignore[arg-type]
            batch_source=self.batch_source,
            monitoring=self.monitoring,
            training=training,
            evaluation=evaluation,
            promotion=promotion,
            rejection=rejection,
            rollback=rollback,
            trigger_policy=RetrainTriggerPolicy(CONFIG),
            probation_policy=ProbationPolicy(CONFIG),
            clock=self.clock,
            config=CONFIG,
        )

    def run(
        self,
        tick: int,
        *,
        manual: bool = False,
        interventions: TickInterventions | None = None,
    ) -> object:
        """Execute one tick with the simulated clock at its end."""
        window = tick_window(tick, REPLAY_START, days=7)
        self.clock.set(window.end)
        return self.tick_service.execute(
            TickRequest(
                tick=tick,
                window=window,
                manual_trigger=manual,
                interventions=interventions or TickInterventions(),
            )
        )

    def types(self) -> list[EventType]:
        """Every event type in append order."""
        return self.events.types()

    def events_of(self, event_type: EventType) -> list[object]:
        """All events of one type."""
        return self.events.list(event_type=event_type)


def test_healthy_stream_never_retrains() -> None:
    harness = Harness()

    for tick in range(1, 6):
        report = harness.run(tick)
        assert report.snapshot is not None
        assert report.snapshot.n_rows > 0

    assert EventType.RETRAIN_TRIGGERED not in harness.types()
    assert harness.kv.get("last_completed_tick") == 5
    assert harness.types().count(EventType.SNAPSHOT_COMPUTED) == 5


def test_slo_breach_triggers_trains_gates_and_promotes_in_order() -> None:
    harness = Harness(champion_predictor=MeanPredictor(50.0))

    for tick in range(1, 9):
        harness.run(tick)
        if harness.events_of(EventType.CHALLENGER_PROMOTED):
            break

    triggered = harness.events_of(EventType.RETRAIN_TRIGGERED)
    assert len(triggered) == 1
    promoted = harness.events_of(EventType.CHALLENGER_PROMOTED)
    assert len(promoted) == 1
    tick = promoted[0].tick
    same_tick = [event.type for event in harness.events.events if event.tick == tick]
    start = same_tick.index(EventType.RETRAIN_TRIGGERED)
    assert same_tick[start : start + 5] == [
        EventType.RETRAIN_TRIGGERED,
        EventType.CHALLENGER_REGISTERED,
        EventType.GATES_EVALUATED,
        EventType.CHALLENGER_PROMOTED,
        EventType.PROBATION_STARTED,
    ]
    assert harness.registry.get_champion() is not None
    assert harness.registry.get_champion().version == 2
    assert harness.kv.get("last_retrain_at") is not None

    for follow_up in range(tick + 1, tick + 5):
        harness.run(follow_up)
        if harness.events_of(EventType.PROBATION_PASSED):
            break

    passed = harness.events_of(EventType.PROBATION_PASSED)
    assert len(passed) == 1
    assert passed[0].tick == tick + 3


def test_failed_batch_is_quarantined_and_never_scored() -> None:
    corrupt = {REPLAY_START: lambda rows: rows.assign(temp=5.0)}
    harness = Harness(overrides=corrupt)

    harness.run(1)

    quarantined = harness.events_of(EventType.BATCH_QUARANTINED)
    assert len(quarantined) == 1
    assert "numeric_range" in quarantined[0].payload["failed_rules"]
    assert harness.events_of(EventType.BATCH_SCORED) == []
    assert harness.gateway.batch_calls == 0
    assert harness.gateway.label_calls == 0
    window = tick_window(1, REPLAY_START, days=7)
    assert harness.prediction_log.labeled_predictions(window.start, window.end).empty
    assert harness.registry.get_champion().version == 1


def test_probation_breach_rolls_back_and_never_retrains() -> None:
    harness = Harness()
    registered = register(
        harness.registry,
        2,
        MeanPredictor(5.0),
        train_window=TimeWindow(datetime(2011, 1, 1), datetime(2011, 10, 1)),
        holdout_window=TimeWindow(datetime(2011, 10, 1), datetime(2012, 1, 1)),
    )
    harness.kv.set(KEY_PROBATION_VERSION, registered.version)
    harness.kv.set(KEY_PROBATION_UNTIL, (REPLAY_START + timedelta(days=7)).isoformat())

    harness.run(1)

    rollbacks = harness.events_of(EventType.ROLLBACK_EXECUTED)
    assert len(rollbacks) == 1
    assert harness.events_of(EventType.RETRAIN_TRIGGERED) == []
    assert harness.registry.get_champion().version == 1
    assert harness.kv.get(KEY_PROBATION_VERSION) is None


def test_drift_alone_never_retrains() -> None:
    frame = demand(raw_frame(start=REPLAY_START, periods=24 * 30, entity_id_start=2_000_000))
    frame["temp"] = 0.99
    frame["atemp"] = 0.99
    hours = pd.to_datetime(frame["timestamp"]).dt.hour.to_numpy()
    frame[TARGET_COLUMN] = np.rint(100.0 + 20.0 * hours + 200.0 * frame["temp"]).astype("int64")
    harness = Harness(production_frame=frame)

    harness.run(1)
    harness.run(2)
    harness.run(3)

    assert EventType.RETRAIN_TRIGGERED not in harness.types()
    snapshots = harness.snapshots.list()
    assert snapshots
    assert snapshots[-1].numeric_drift_share >= 0.5


def test_failing_step_is_retried_and_ledger_records_failure_without_state() -> None:
    failing = FailingGateway()
    harness = Harness(gateway=failing)

    with pytest.raises(PipelineError):
        harness.run(1)

    attempts = [entry for entry in harness.ledger.steps if entry.name == "score_batch"]
    assert len(attempts) == 2
    assert all(entry.status.value == "FAILED" for entry in attempts)
    assert harness.ledger.finishes[-1].status.value == "FAILED"
    assert harness.kv.get("last_completed_tick") is None
    assert harness.kv.get("last_retrain_at") is None
    assert failing.calls == 2


def test_running_the_same_tick_twice_is_a_noop() -> None:
    harness = Harness()

    first = harness.run(3)
    event_count = len(harness.events.events)
    prediction_count = len(harness.prediction_log.records)
    second = harness.run(3)

    assert first.action != "ALREADY_COMPLETED"
    assert second.action == "ALREADY_COMPLETED"
    assert len(harness.events.events) == event_count
    assert len(harness.prediction_log.records) == prediction_count


def test_manual_trigger_bypasses_the_slo_without_bypassing_the_gates() -> None:
    harness = Harness(champion_predictor=MeanPredictor(50.0))

    harness.run(1, manual=True)

    assert len(harness.events_of(EventType.RETRAIN_TRIGGERED)) == 1
    # The synthetic challenger is perfect, so the gates approve and promote it.
    assert len(harness.events_of(EventType.CHALLENGER_PROMOTED)) == 1
