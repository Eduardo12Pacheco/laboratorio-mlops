"""The drill engine loop: bootstrap, weekly ticks, replay cursor and report output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest

from mlops_lifecycle_lab.application.dto import (
    LabelRecord,
    MonitoringSnapshot,
    PredictionRecord,
    TickReport,
    encode_version_tags,
)
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from mlops_lifecycle_lab.domain.windows import TimeWindow
from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine
from mlops_lifecycle_lab.entrypoints.drill.faults import PoisonedTrainer
from mlops_lifecycle_lab.entrypoints.drill.scenarios import Scenario, get_scenario
from tests.fakes.data import raw_frame
from tests.fakes.prediction_log import InMemoryPredictionLog
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.stores import InMemoryKeyValueStore
from tests.fakes.trainer import MeanPredictor
from tests.unit.entrypoints.conftest import model_version

REPLAY_START = datetime(2012, 1, 1)


class FakeBootstrap:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self) -> object:
        self.calls += 1
        return model_version(1)


class FakeLifecycle:
    """Produces deterministic ticks and predictions without training."""

    def __init__(self, clock: object, log: InMemoryPredictionLog) -> None:
        self.clock = clock
        self.log = log
        self.requests: list[object] = []
        self.ticks_seen: list[int] = []

    def execute(self, request: object) -> TickReport:
        self.requests.append(request)
        window: TimeWindow = request.window  # type: ignore[attr-defined]
        tick: int = request.tick  # type: ignore[attr-defined]
        assert self.clock.now() == window.end  # type: ignore[attr-defined]
        frame = raw_frame(start=window.start, periods=3, entity_id_start=tick * 100)
        features = build_features(frame)
        records = []
        for position, row in frame.iterrows():
            records.append(
                PredictionRecord(
                    prediction_id=f"p-{tick}-{position}",
                    entity_id=int(row["entity_id"]),
                    event_ts=row["timestamp"].to_pydatetime(),
                    scored_at=window.end,
                    model_name="bike-demand",
                    model_version=1,
                    features={
                        name: float(features.loc[position, name]) for name in FEATURE_COLUMNS
                    },
                    prediction=float(row["cnt"]),
                    source="batch",
                )
            )
        self.log.append(records)
        self.log.attach_labels(
            [
                LabelRecord(
                    entity_id=int(row["entity_id"]),
                    label=float(row["cnt"]),
                    label_ts=row["timestamp"].to_pydatetime(),
                )
                for _, row in frame.iterrows()
            ]
        )
        events = [
            self._event(EventType.BATCH_VALIDATED, tick, 1, {"verdict": "PASS"}),
            self._event(EventType.BATCH_SCORED, tick, 1, {"rows": 3}),
            self._event(EventType.SNAPSHOT_COMPUTED, tick, 1, {"n_rows": 3}),
        ]
        if tick == 2:
            events.extend(
                [
                    self._event(EventType.RETRAIN_TRIGGERED, tick, 1, {}),
                    self._event(
                        EventType.GATES_EVALUATED,
                        tick,
                        2,
                        {
                            "approved": True,
                            "gate_report": {
                                "approved": True,
                                "results": [
                                    {
                                        "name": "IMPROVEMENT",
                                        "passed": True,
                                        "observed": 0.56,
                                        "threshold": 0.05,
                                        "detail": "ok",
                                    }
                                ],
                            },
                        },
                    ),
                    self._event(EventType.CHALLENGER_PROMOTED, tick, 2, {}),
                ]
            )
        snapshot = MonitoringSnapshot(
            tick=tick,
            ts=window.end,
            model_version=1,
            n_rows=3,
            rmse=1.0,
            mae=1.0,
            wape=0.1,
            bias=0.0,
            numeric_drift_share=0.25,
            prediction_psi=0.1,
            psi={"temp": 0.2},
        )
        return TickReport(
            tick=tick,
            window=window,
            champion_version=1,
            validation_verdict="PASS",
            action="SCORED",
            snapshot=snapshot,
            events=tuple(events),
        )

    def _event(
        self,
        event_type: EventType,
        tick: int,
        version: int | None,
        payload: dict[str, object],
    ) -> LifecycleEvent:
        return LifecycleEvent(
            ts=self.clock.now(),  # type: ignore[attr-defined]
            type=event_type,
            tick=tick,
            model_version=version,
            payload=payload,
        )


@dataclass
class FakeContainer:
    bootstrap: FakeBootstrap
    lifecycle: FakeLifecycle
    registry: InMemoryRegistry
    prediction_log: InMemoryPredictionLog
    kv: InMemoryKeyValueStore
    closed: bool = field(default=False)

    def close(self) -> None:
        self.closed = True


def _factory(container: FakeContainer, captured: dict[str, object]) -> object:
    def factory(settings: object, clock: object, trainer: object) -> FakeContainer:
        captured["settings"] = settings
        captured["clock"] = clock
        captured["trainer"] = trainer
        container.lifecycle.clock = clock
        return container

    return factory


def _container() -> FakeContainer:
    registry = InMemoryRegistry()
    registry.adopt_candidate(1, encode_version_tags(model_version(1)))
    registry.set_predictor(1, MeanPredictor(2.0))
    log = InMemoryPredictionLog()
    clock = _CapturingClock()
    return FakeContainer(
        bootstrap=FakeBootstrap(),
        lifecycle=FakeLifecycle(clock, log),
        registry=registry,
        prediction_log=log,
        kv=InMemoryKeyValueStore(),
    )


class _CapturingClock:
    """Simulated clock that the fake lifecycle compares against each tick end."""

    def __init__(self) -> None:
        self._now = REPLAY_START

    def now(self) -> datetime:
        return self._now

    def set(self, moment: datetime) -> None:
        self._now = moment


def test_engine_runs_bootstrap_and_every_week_and_writes_the_report(tmp_path: Path) -> None:
    container = _container()
    captured: dict[str, object] = {}
    engine = DrillEngine(_factory(container, captured))  # type: ignore[arg-type]

    result = engine.run(
        "drift",
        weeks=2,
        workspace=tmp_path / "ws",
        report_dir=tmp_path / "reports",
        seed=42,
        fast=True,
    )

    assert container.bootstrap.calls == 1
    assert len(result.ticks) == 2
    assert result.ticks[0].tick_end == datetime(2012, 1, 8)
    assert result.summary.promotions == 1
    assert result.summary.versions == 1
    assert len(result.gate_reports) == 1
    assert result.gate_reports[0].decision == "APPROVED"
    assert container.kv.get("replay_tick") == 2
    assert container.closed is True
    assert isinstance(captured["trainer"], PoisonedTrainer)
    assert captured["trainer"].active is False  # type: ignore[attr-defined]
    assert (tmp_path / "reports" / "drill-drift.json").exists()
    assert (tmp_path / "reports" / "drill-drift.md").exists()
    payload = json.loads((tmp_path / "reports" / "drill-drift.json").read_text())
    assert payload["summary"]["as_served"]["rmse"] == pytest.approx(0.0)
    assert payload["summary"]["counterfactual_frozen_v1"]["rmse"] > 0.0


def test_engine_defaults_the_report_directory_under_the_workspace(tmp_path: Path) -> None:
    container = _container()
    engine = DrillEngine(_factory(container, {}))  # type: ignore[arg-type]

    engine.run("drift", weeks=1, workspace=tmp_path / "ws", fast=True)

    assert (tmp_path / "ws" / "reports" / "drill-drift.json").exists()


def test_engine_uses_the_scenario_default_weeks_when_weeks_is_none(tmp_path: Path) -> None:
    container = _container()
    engine = DrillEngine(_factory(container, {}))  # type: ignore[arg-type]

    result = engine.run("bad-batch", weeks=None, workspace=tmp_path / "ws", fast=True)

    assert result.weeks == get_scenario("bad-batch").default_weeks


def test_engine_rejects_unknown_scenarios_and_invalid_week_counts(tmp_path: Path) -> None:
    engine = DrillEngine(_factory(_container(), {}))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unknown drill scenario"):
        engine.run("nope", weeks=1, workspace=tmp_path / "ws")
    with pytest.raises(ValueError, match="weeks"):
        engine.run("drift", weeks=0, workspace=tmp_path / "ws")


def test_engine_accepts_a_scenario_object(tmp_path: Path) -> None:
    engine = DrillEngine(_factory(_container(), {}))  # type: ignore[arg-type]
    scenario = Scenario(name="drift", default_weeks=1, script=get_scenario("drift").script)

    result = engine.run(scenario, weeks=1, workspace=tmp_path / "ws", fast=True)

    assert result.scenario == "drift"
    assert len(result.ticks) == 1
