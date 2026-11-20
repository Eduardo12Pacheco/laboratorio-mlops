"""Evaluate a challenger against the champion with the six promotion gates (S12).

The use case re-scores the incumbent on the *same* holdout as the challenger, fits the
naive baseline on the same training rows, builds the pure `GateContext` and delegates the
decision to `PromotionPolicy`. It then persists the complete gate report as a tracker
artifact and appends `GATES_EVALUATED` so promotion can refuse an unapproved candidate
and the dashboard can show the evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mlops_lifecycle_lab.application.dto import EvaluationRequest, EvaluationResult
from mlops_lifecycle_lab.application.ports import (
    Clock,
    EventStore,
    ExperimentTracker,
    LabeledDataRepository,
    ModelRegistry,
)
from mlops_lifecycle_lab.application.services._data import (
    clip_predictions,
    require_rows,
    window_rows,
)
from mlops_lifecycle_lab.domain.baseline import ProfileBaseline
from mlops_lifecycle_lab.domain.errors import UnknownModelVersionError
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.gates import GateContext
from mlops_lifecycle_lab.domain.metrics import compute_metrics
from mlops_lifecycle_lab.domain.policies import PromotionPolicy
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS, SEGMENT_COLUMN, TARGET_COLUMN


class EvaluateChallenger:
    """Score champion and challenger on one holdout and record the gate decision."""

    def __init__(
        self,
        labeled_data: LabeledDataRepository,
        registry: ModelRegistry,
        tracker: ExperimentTracker,
        event_store: EventStore,
        clock: Clock,
        policy: PromotionPolicy | None = None,
    ) -> None:
        self._labeled_data = labeled_data
        self._registry = registry
        self._tracker = tracker
        self._events = event_store
        self._clock = clock
        self._policy = policy or PromotionPolicy()

    def execute(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate `request.version` and persist the full gate report."""
        candidate = self._registry.get_version(request.version)
        if candidate is None:
            raise UnknownModelVersionError(f"model version {request.version} is not registered")

        rows = self._labeled_data.training_frame(request.holdout_window.end)
        holdout = window_rows(rows, request.holdout_window)
        train_rows = window_rows(rows, request.train_window)
        require_rows(holdout, "holdout window")
        require_rows(train_rows, "training window")

        holdout_features = build_features(holdout).loc[:, list(FEATURE_COLUMNS)]
        actuals = np.asarray(holdout[TARGET_COLUMN], dtype=np.float64)
        challenger_predictions = clip_predictions(
            self._registry.load_predictor(request.version).predict(holdout_features)
        )
        challenger_metrics = compute_metrics(actuals, challenger_predictions)

        champion = self._registry.get_champion()
        champion_metrics = None
        champion_errors: np.ndarray | None = None
        if champion is not None:
            champion_predictions = clip_predictions(
                self._registry.load_predictor(champion.version).predict(holdout_features)
            )
            champion_metrics = compute_metrics(actuals, champion_predictions)
            champion_errors = champion_predictions - actuals

        baseline = ProfileBaseline.fit(train_rows)
        baseline_rmse = compute_metrics(actuals, baseline.predict(holdout)).rmse
        context = GateContext(
            timestamps=holdout["timestamp"].to_numpy(),
            actuals=actuals,
            challenger_errors=challenger_predictions - actuals,
            champion_errors=champion_errors,
            segments=holdout[SEGMENT_COLUMN].to_numpy(),
            baseline_rmse=baseline_rmse,
            training_verdict=request.training_verdict,
            is_first_champion=champion is None,
        )
        decision = self._policy.evaluate(context)

        artifact: Mapping[str, Any] = {
            "schema": "gate-report/v1",
            "version": request.version,
            "tick": request.tick,
            "champion_version": None if champion is None else champion.version,
            "decision": "APPROVED" if decision.approved else "REJECTED",
            "approved": decision.approved,
            "reason": decision.reason,
            "failed_gates": list(decision.failed_gates),
            "results": [gate.to_dict() for gate in decision.report.results],
        }
        tags = {
            "mll.model_version": str(request.version),
            "mll.tick": str(request.tick),
            "mll.decision": str(artifact["decision"]),
        }
        with self._tracker.run(f"evaluate-v{request.version}", tags) as run:
            run.log_json("gate_report.json", dict(artifact))

        event = LifecycleEvent(
            ts=self._clock.now(),
            type=EventType.GATES_EVALUATED,
            tick=request.tick,
            model_version=request.version,
            payload={
                "approved": decision.approved,
                "failed_gates": list(decision.failed_gates),
                "reason": decision.reason,
                "champion_version": None if champion is None else champion.version,
                "gate_report": decision.report.to_dict(),
            },
        )
        self._events.append(event)

        return EvaluationResult(
            version=request.version,
            decision=decision,
            challenger_metrics=challenger_metrics,
            champion_metrics=champion_metrics,
            baseline_rmse=baseline_rmse,
            gate_report=decision.report,
        )
