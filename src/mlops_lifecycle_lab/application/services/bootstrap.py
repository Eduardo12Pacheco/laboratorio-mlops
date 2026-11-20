"""Bootstrap the first champion on the fixed S5 windows (`D3`).

`BootstrapChampion` is idempotent: once a champion exists it returns it untouched, so
`mll bootstrap --if-missing`, the drill and the Compose `init` service can all call it
safely. The first promotion is evaluated with gates G1-G3 only, exactly like
`PromotionPolicy` requires for a challenger with no incumbent.
"""

from __future__ import annotations

from datetime import datetime

from mlops_lifecycle_lab.application.dto import (
    EvaluationRequest,
    TrainRequest,
)
from mlops_lifecycle_lab.application.ports import Clock, EventStore, ModelRegistry
from mlops_lifecycle_lab.application.services.evaluation import EvaluateChallenger
from mlops_lifecycle_lab.application.services.promotion import (
    PromoteChallenger,
    RejectChallenger,
)
from mlops_lifecycle_lab.application.services.training import TrainCandidate
from mlops_lifecycle_lab.domain.errors import GateEvaluationError
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.lifecycle import ModelVersion
from mlops_lifecycle_lab.domain.windows import TimeWindow

BOOTSTRAP_TRIGGER_REASON = "BOOTSTRAP"
BOOTSTRAP_TRAIN_WINDOW = TimeWindow(datetime(2011, 1, 1), datetime(2011, 10, 1))
BOOTSTRAP_HOLDOUT_WINDOW = TimeWindow(datetime(2011, 10, 1), datetime(2012, 1, 1))
"""Fixed windows of S5: train before 2011-10-01, validate on 2011 Q4."""


class BootstrapChampion:
    """Train, gate and promote v1 if (and only if) there is no champion yet."""

    def __init__(
        self,
        registry: ModelRegistry,
        event_store: EventStore,
        clock: Clock,
        train_candidate: TrainCandidate,
        evaluate_challenger: EvaluateChallenger,
        promote_challenger: PromoteChallenger,
        reject_challenger: RejectChallenger,
    ) -> None:
        self._registry = registry
        self._events = event_store
        self._clock = clock
        self._train = train_candidate
        self._evaluate = evaluate_challenger
        self._promote = promote_challenger
        self._reject = reject_challenger

    def execute(self) -> ModelVersion:
        """Return the current champion, or bootstrap one.

        Raises:
            GateEvaluationError: when the freshly trained v1 fails the first-champion
                gates; the candidate is rejected before the error is raised.
        """
        existing = self._registry.get_champion()
        if existing is not None:
            return existing

        trained = self._train.execute(
            TrainRequest(
                train_window=BOOTSTRAP_TRAIN_WINDOW,
                holdout_window=BOOTSTRAP_HOLDOUT_WINDOW,
                trigger_reason=BOOTSTRAP_TRIGGER_REASON,
                tick=0,
            )
        )
        evaluation = self._evaluate.execute(
            EvaluationRequest(
                version=trained.version,
                train_window=BOOTSTRAP_TRAIN_WINDOW,
                holdout_window=BOOTSTRAP_HOLDOUT_WINDOW,
                training_verdict=trained.validation.verdict.value,
                tick=0,
                trigger_reason=BOOTSTRAP_TRIGGER_REASON,
            )
        )
        if not evaluation.decision.approved:
            self._reject.execute(
                trained.version,
                failed_gates=evaluation.decision.failed_gates,
                reason=evaluation.decision.reason,
            )
            raise GateEvaluationError(
                f"bootstrap champion rejected by gates: {evaluation.decision.reason}"
            )

        champion = self._promote.execute(trained.version)
        self._events.append(
            LifecycleEvent(
                ts=self._clock.now(),
                type=EventType.BOOTSTRAP_COMPLETED,
                tick=0,
                model_version=champion.version,
                payload={
                    "metrics": trained.metrics.to_dict(),
                    "baseline_rmse": trained.baseline_rmse,
                    "data_fingerprint": trained.data_fingerprint,
                    "train_window": BOOTSTRAP_TRAIN_WINDOW.start.isoformat(),
                    "holdout_window": BOOTSTRAP_HOLDOUT_WINDOW.end.isoformat(),
                },
            )
        )
        return champion


__all__ = [
    "BOOTSTRAP_HOLDOUT_WINDOW",
    "BOOTSTRAP_TRAIN_WINDOW",
    "BOOTSTRAP_TRIGGER_REASON",
    "BootstrapChampion",
]
