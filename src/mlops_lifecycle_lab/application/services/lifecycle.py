"""The lifecycle tick: one replay week from batch to decision (S14).

This use case is the orchestration seam. It runs as a `Pipeline` so every tick and step
lands in the run ledger, and it appends domain events instead of returning ad-hoc dicts:
validate the weekly batch (quarantine on FAIL), score through the `ScoringGateway`, ingest
the delayed labels, compute the monitoring snapshot, then either watch probation (rollback
has priority) or run the retrain/gate/promote-or-reject protocol.

The tick is idempotent per tick number: the KV key `last_completed_tick` is written only
after every step succeeded, so a crash or a failed pipeline step leaves the workspace in a
state that a restart can safely retry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd

from mlops_lifecycle_lab.application.dto import (
    EvaluationRequest,
    LabelRecord,
    MonitoringSnapshot,
    RawBatch,
    TickReport,
    TrainRequest,
)
from mlops_lifecycle_lab.application.pipeline import Pipeline, Step
from mlops_lifecycle_lab.application.ports import (
    BatchSource,
    Clock,
    EventStore,
    KeyValueStore,
    ModelRegistry,
    RunLedger,
    ScoringGateway,
)
from mlops_lifecycle_lab.application.services.evaluation import EvaluateChallenger
from mlops_lifecycle_lab.application.services.monitoring import ComputeSnapshot
from mlops_lifecycle_lab.application.services.promotion import (
    KEY_PROBATION_UNTIL,
    KEY_PROBATION_VERSION,
    PromoteChallenger,
    RejectChallenger,
    RollbackChampion,
)
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.errors import NoChampionError
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.policies import (
    PolicyConfig,
    ProbationPolicy,
    ProbationVerdict,
    RetrainTriggerPolicy,
    SnapshotView,
    TriggerDecision,
)
from mlops_lifecycle_lab.domain.validation import (
    Severity,
    ValidationReport,
    Verdict,
    validate_frame,
)
from mlops_lifecycle_lab.domain.windows import TimeWindow

KEY_LAST_COMPLETED_TICK = "last_completed_tick"
KEY_LAST_RETRAIN_AT = "last_retrain_at"
"""KV keys owned by the tick; `last_retrain_at` drives the retrain cooldown."""

TRAINING_ORIGIN = datetime(2011, 1, 1)
"""First date of the bundled archive; challengers train on everything before the cut."""

ACTION_SCORED = "SCORED"
ACTION_QUARANTINED = "QUARANTINED"
ACTION_PROMOTED = "PROMOTED"
ACTION_FORCED = "PROMOTION_FORCED"
ACTION_REJECTED = "REJECTED"
ACTION_ROLLBACK = "ROLLBACK"
ACTION_PROBATION_PASSED = "PROBATION_PASSED"
ACTION_PROBATION_CONTINUE = "PROBATION_CONTINUE"
ACTION_ALREADY_COMPLETED = "ALREADY_COMPLETED"
"""Values of `TickReport.action`, consumed by the drill report and the dashboard."""


@dataclass(frozen=True, slots=True)
class TickRequest:
    """Inputs of one lifecycle tick."""

    tick: int
    window: TimeWindow
    manual_trigger: bool = False
    interventions: TickInterventions = field(default_factory=lambda: TickInterventions())

    def __post_init__(self) -> None:
        if self.tick < 1:
            raise ValueError(f"tick must be >= 1, got {self.tick}")


@dataclass(slots=True)
class TickDecisionView:
    """Mutable decision inputs exposed to a drill intervention hook."""

    tick: int
    window: TimeWindow
    snapshot: MonitoringSnapshot | None
    manual_trigger: bool = False
    force_promotion: bool = False
    promotion_reason: str = ""


@dataclass(frozen=True, slots=True)
class TickInterventions:
    """Typed fault-injection hooks used only by drill scenarios (no-ops by default).

    `before_validate` may return a modified raw frame (e.g. the celsius-scale batch of
    the `bad-batch` scenario). `before_decide` may request a manual retrain or a forced
    promotion at a given tick (the `gate-reject`/`rollback` scenarios). Neither hook can
    change the validation rules or the gate policy.
    """

    before_validate: Callable[[int, TimeWindow, pd.DataFrame], pd.DataFrame] | None = None
    before_decide: Callable[[TickDecisionView], None] | None = None


class RunLifecycleTick:
    """Execute one replay week with retries, ledger rows and audit events."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        events: EventStore,
        kv: KeyValueStore,
        ledger: RunLedger,
        gateway: ScoringGateway,
        batch_source: BatchSource,
        monitoring: ComputeSnapshot,
        training: TrainCandidate,
        evaluation: EvaluateChallenger,
        promotion: PromoteChallenger,
        rejection: RejectChallenger,
        rollback: RollbackChampion,
        trigger_policy: RetrainTriggerPolicy,
        probation_policy: ProbationPolicy,
        clock: Clock,
        config: PolicyConfig | None = None,
        training_origin: datetime = TRAINING_ORIGIN,
    ) -> None:
        self._registry = registry
        self._events = events
        self._kv = kv
        self._ledger = ledger
        self._gateway = gateway
        self._batch_source = batch_source
        self._monitoring = monitoring
        self._training = training
        self._evaluation = evaluation
        self._promotion = promotion
        self._rejection = rejection
        self._rollback = rollback
        self._trigger_policy = trigger_policy
        self._probation_policy = probation_policy
        self._clock = clock
        self._config = config or PolicyConfig()
        self._training_origin = training_origin

    def execute(self, request: TickRequest) -> TickReport:
        """Run one tick, or return the previous result when it already completed."""
        if self._already_completed(request.tick):
            return self._completed_report(request)

        state: dict[str, Any] = {
            "tick": request.tick,
            "window": request.window,
            "request": request,
            "action": ACTION_SCORED,
            "quarantined": False,
        }
        steps = [
            Step("load_batch", lambda _context: self._load_batch(state)),
            Step("validate_batch", lambda _context: self._validate_batch(state)),
            Step("score_batch", lambda _context: self._score_batch(state), retries=1),
            Step("ingest_labels", lambda _context: self._ingest_labels(state)),
            Step("monitoring_snapshot", lambda _context: self._snapshot(state)),
            Step("decide", lambda _context: self._decide(state)),
            Step("persist_state", lambda _context: self._persist(state)),
        ]
        pipeline = Pipeline(
            steps,
            self._ledger,
            name=f"tick-{request.tick}",
            clock=self._clock,
        )
        pipeline.run(state)

        champion = self._registry.get_champion()
        return TickReport(
            tick=request.tick,
            window=request.window,
            champion_version=None if champion is None else champion.version,
            validation_verdict=self._verdict_of(state),
            action=str(state["action"]),
            snapshot=cast(MonitoringSnapshot | None, state.get("snapshot")),
            events=self._events_for(request.tick),
        )

    # -- pipeline steps ---------------------------------------------------------------
    def _load_batch(self, state: dict[str, Any]) -> None:
        window = cast(TimeWindow, state["window"])
        state["batch"] = self._batch_source.week(window)

    def _validate_batch(self, state: dict[str, Any]) -> None:
        request = cast(TickRequest, state["request"])
        batch = cast(RawBatch, state["batch"])
        raw = batch.raw
        if request.interventions.before_validate is not None:
            raw = request.interventions.before_validate(request.tick, request.window, raw.copy())
        validation = validate_frame(raw, context="batch")
        state["raw"] = raw
        state["validation"] = validation
        champion = self._registry.get_champion()
        version = None if champion is None else champion.version

        if validation.verdict is Verdict.FAIL:
            state["quarantined"] = True
            state["action"] = ACTION_QUARANTINED
            failed = [
                rule.name
                for rule in validation.rules
                if not rule.passed and rule.severity is Severity.ERROR
            ]
            self._emit(
                EventType.BATCH_QUARANTINED,
                request.tick,
                version,
                {
                    "verdict": Verdict.FAIL.value,
                    "rows": len(raw),
                    "failed_rules": failed,
                },
            )
            return
        warnings = [
            rule.name
            for rule in validation.rules
            if not rule.passed and rule.severity is Severity.WARN
        ]
        self._emit(
            EventType.BATCH_VALIDATED,
            request.tick,
            version,
            {
                "verdict": validation.verdict.value,
                "rows": len(raw),
                "warnings": warnings,
            },
        )

    def _score_batch(self, state: dict[str, Any]) -> None:
        if state["quarantined"]:
            return
        request = cast(TickRequest, state["request"])
        raw = cast(pd.DataFrame, state["raw"])
        predictions = self._gateway.predict_batch(raw)
        state["predictions"] = predictions
        champion = self._registry.get_champion()
        version = (
            predictions[0].model_version if predictions else getattr(champion, "version", None)
        )
        self._emit(
            EventType.BATCH_SCORED,
            request.tick,
            version,
            {"rows": len(predictions), "model_version": version},
        )

    def _ingest_labels(self, state: dict[str, Any]) -> None:
        if state["quarantined"]:
            return
        request = cast(TickRequest, state["request"])
        batch = cast(RawBatch, state["batch"])
        entity_ids = pd.to_numeric(batch.labels["entity_id"]).to_numpy(dtype=np.int64)
        values = pd.to_numeric(batch.labels["label"]).to_numpy(dtype=np.float64)
        timestamps = pd.to_datetime(batch.labels["label_ts"])
        labels = [
            LabelRecord(
                entity_id=int(entity_ids[position]),
                label=float(values[position]),
                label_ts=pd.Timestamp(timestamps.iloc[position]).to_pydatetime(),
            )
            for position in range(len(entity_ids))
        ]
        self._gateway.submit_labels(labels)
        self._emit(
            EventType.LABELS_INGESTED,
            request.tick,
            self._champion_version(),
            {"rows": len(labels)},
        )

    def _snapshot(self, state: dict[str, Any]) -> None:
        request = cast(TickRequest, state["request"])
        champion = self._registry.get_champion()
        if champion is None:
            raise NoChampionError("the lifecycle tick requires a champion model")
        state["snapshot"] = self._monitoring.execute(
            request.window.end,
            champion.version,
            tick=request.tick,
        )

    def _decide(self, state: dict[str, Any]) -> None:
        request = cast(TickRequest, state["request"])
        snapshot = cast(MonitoringSnapshot, state["snapshot"])
        view = TickDecisionView(
            tick=request.tick,
            window=request.window,
            snapshot=snapshot,
            manual_trigger=request.manual_trigger,
        )
        if request.interventions.before_decide is not None:
            request.interventions.before_decide(view)

        probation_version = self._probation_version()
        in_probation = probation_version is not None
        if probation_version is not None:
            verdict = self._probation_verdict(probation_version, snapshot, request)
            if verdict is ProbationVerdict.ROLLBACK:
                self._rollback.execute(
                    f"probation breach: WAPE {_wape_text(snapshot)}",
                    tick=request.tick,
                )
                state["action"] = ACTION_ROLLBACK
                return
            if verdict is ProbationVerdict.PASSED:
                self._emit(
                    EventType.PROBATION_PASSED,
                    request.tick,
                    probation_version,
                    {"passed_at": request.window.end.isoformat()},
                )
                self._kv.set(KEY_PROBATION_VERSION, None)
                self._kv.set(KEY_PROBATION_UNTIL, None)
                in_probation = False
                state["action"] = ACTION_PROBATION_PASSED
            else:
                state["action"] = ACTION_PROBATION_CONTINUE
                return

        decision = self._trigger_policy.evaluate(
            cast(SnapshotView, snapshot),
            self._last_retrain_at(),
            request.window.end,
            in_probation,
            manual=view.manual_trigger,
        )
        if not decision.should_retrain:
            return
        self._retrain(state, decision, view)

    def _retrain(
        self,
        state: dict[str, Any],
        decision: TriggerDecision,
        view: TickDecisionView,
    ) -> None:
        request = cast(TickRequest, state["request"])
        trigger = decision.trigger
        if trigger is None:
            raise ValueError("a retrain decision must carry a trigger reason")
        cut = request.window.end - timedelta(days=self._config.holdout_days)
        train_window = TimeWindow(self._training_origin, cut)
        holdout_window = TimeWindow(cut, request.window.end)
        self._emit(
            EventType.RETRAIN_TRIGGERED,
            request.tick,
            self._champion_version(),
            {
                "trigger": trigger.value,
                "outcome": decision.outcome.value,
                "cut": cut.isoformat(),
                "holdout_window": (
                    f"{holdout_window.start.isoformat()}/{holdout_window.end.isoformat()}"
                ),
            },
        )
        trained = self._training.execute(
            TrainRequest(
                train_window=train_window,
                holdout_window=holdout_window,
                trigger_reason=trigger.value,
                tick=request.tick,
            )
        )
        self._emit(
            EventType.CHALLENGER_REGISTERED,
            request.tick,
            trained.version,
            {
                "metrics": trained.metrics.to_dict(),
                "training_rows": trained.training_rows,
                "holdout_rows": trained.holdout_rows,
                "data_fingerprint": trained.data_fingerprint,
            },
        )
        evaluation = self._evaluation.execute(
            EvaluationRequest(
                version=trained.version,
                train_window=train_window,
                holdout_window=holdout_window,
                training_verdict=trained.validation.verdict.value,
                tick=request.tick,
                trigger_reason=trigger.value,
            )
        )
        if view.force_promotion:
            self._promotion.execute(
                trained.version,
                forced=True,
                reason=view.promotion_reason or "forced by drill intervention",
                tick=request.tick,
            )
            state["action"] = ACTION_FORCED
        elif evaluation.decision.approved:
            self._promotion.execute(trained.version, tick=request.tick)
            state["action"] = ACTION_PROMOTED
        else:
            self._rejection.execute(
                trained.version,
                failed_gates=evaluation.decision.failed_gates,
                reason=evaluation.decision.reason,
                tick=request.tick,
            )
            state["action"] = ACTION_REJECTED
        self._kv.set(KEY_LAST_RETRAIN_AT, request.window.end.isoformat())

    def _persist(self, state: dict[str, Any]) -> None:
        request = cast(TickRequest, state["request"])
        self._kv.set(KEY_LAST_COMPLETED_TICK, request.tick)

    # -- helpers ----------------------------------------------------------------------
    def _already_completed(self, tick: int) -> bool:
        value = self._kv.get(KEY_LAST_COMPLETED_TICK)
        try:
            completed = 0 if value is None else int(str(value))
        except (TypeError, ValueError):
            completed = 0
        return tick <= completed

    def _completed_report(self, request: TickRequest) -> TickReport:
        champion = self._registry.get_champion()
        events = self._events_for(request.tick)
        verdict = "PASS"
        for event in events:
            if event.type in (EventType.BATCH_VALIDATED, EventType.BATCH_QUARANTINED):
                verdict = str(event.payload.get("verdict", verdict))
        return TickReport(
            tick=request.tick,
            window=request.window,
            champion_version=None if champion is None else champion.version,
            validation_verdict=verdict,
            action=ACTION_ALREADY_COMPLETED,
            snapshot=None,
            events=events,
        )

    def _probation_version(self) -> int | None:
        value = self._kv.get(KEY_PROBATION_VERSION)
        if value is None or self._kv.get(KEY_PROBATION_UNTIL) is None:
            return None
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def _probation_verdict(
        self,
        version: int,
        snapshot: MonitoringSnapshot,
        request: TickRequest,
    ) -> ProbationVerdict:
        until = datetime.fromisoformat(str(self._kv.get(KEY_PROBATION_UNTIL)))
        promoted_at = until - timedelta(days=self._config.probation_days)
        verdict = self._probation_policy.evaluate(
            cast(SnapshotView, snapshot), promoted_at, request.window.end
        )
        if verdict is ProbationVerdict.PASSED and version != snapshot.model_version:
            # A stale probation record must never pass; let the trigger policy decide.
            return ProbationVerdict.CONTINUE
        return verdict

    def _last_retrain_at(self) -> datetime | None:
        value = self._kv.get(KEY_LAST_RETRAIN_AT)
        if value is None:
            return None
        return datetime.fromisoformat(str(value))

    def _champion_version(self) -> int | None:
        champion = self._registry.get_champion()
        return None if champion is None else champion.version

    def _events_for(self, tick: int) -> tuple[LifecycleEvent, ...]:
        return tuple(event for event in self._events.list() if event.tick == tick)

    def _emit(
        self,
        event_type: EventType,
        tick: int,
        model_version: int | None,
        payload: Mapping[str, Any],
    ) -> None:
        self._events.append(
            LifecycleEvent(
                ts=self._clock.now(),
                type=event_type,
                tick=tick,
                model_version=model_version,
                payload=dict(payload),
            )
        )

    @staticmethod
    def _verdict_of(state: Mapping[str, Any]) -> str:
        validation = cast(ValidationReport | None, state.get("validation"))
        return "PASS" if validation is None else validation.verdict.value


def _wape_text(snapshot: MonitoringSnapshot) -> str:
    return "n/a" if snapshot.wape is None else f"{snapshot.wape:.4f}"


__all__ = [
    "ACTION_ALREADY_COMPLETED",
    "ACTION_FORCED",
    "ACTION_PROBATION_CONTINUE",
    "ACTION_PROBATION_PASSED",
    "ACTION_PROMOTED",
    "ACTION_QUARANTINED",
    "ACTION_REJECTED",
    "ACTION_ROLLBACK",
    "ACTION_SCORED",
    "KEY_LAST_COMPLETED_TICK",
    "KEY_LAST_RETRAIN_AT",
    "TRAINING_ORIGIN",
    "RunLifecycleTick",
    "TickDecisionView",
    "TickInterventions",
    "TickRequest",
]
