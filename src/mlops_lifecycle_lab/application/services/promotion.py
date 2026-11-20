"""Promotion, rejection and rollback use cases (`S13`).

Promotion is guarded by the audit log instead of a mutable in-memory flag: the last
`GATES_EVALUATED` event for a version must say `approved`, which survives restarts and
matches what the dashboard shows. A forced promotion is the only bypass, requires a
reason and is itself recorded as `PROMOTION_FORCED`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from mlops_lifecycle_lab.application.ports import Clock, EventStore, KeyValueStore, ModelRegistry
from mlops_lifecycle_lab.domain.errors import (
    GateEvaluationError,
    UnknownModelVersionError,
)
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.lifecycle import ModelVersion
from mlops_lifecycle_lab.domain.policies import PolicyConfig

KEY_PROBATION_VERSION = "probation_version"
KEY_PROBATION_UNTIL = "probation_until"
"""KV keys holding the active probation window (cleared on rollback)."""


class PromoteChallenger:
    """Promote an approved (or explicitly forced) candidate and start probation."""

    def __init__(
        self,
        registry: ModelRegistry,
        event_store: EventStore,
        kv: KeyValueStore,
        clock: Clock,
        policy: PolicyConfig | None = None,
    ) -> None:
        self._registry = registry
        self._events = event_store
        self._kv = kv
        self._clock = clock
        self._policy = policy or PolicyConfig()

    def execute(
        self,
        version: int,
        forced: bool = False,
        reason: str | None = None,
        *,
        tick: int = 0,
    ) -> ModelVersion:
        """Promote `version`, refusing an unapproved candidate unless forced.

        Raises:
            ValueError: when `forced` is set without a non-empty reason.
            UnknownModelVersionError: when the version does not exist.
            GateEvaluationError: when the version has no approved gate evaluation.
            IllegalTransitionError: when the state machine forbids the promotion.
        """
        reason_text = (reason or "").strip()
        if forced and not reason_text:
            raise ValueError("forced promotion requires a non-empty reason")
        candidate = self._registry.get_version(version)
        if candidate is None:
            raise UnknownModelVersionError(f"model version {version} is not registered")
        if not forced and not self._is_approved(version):
            raise GateEvaluationError(
                f"challenger v{version} was not approved by the promotion gates"
            )

        previous = self._registry.get_champion()
        promoted = self._registry.promote(version)
        if forced:
            self._append(
                EventType.PROMOTION_FORCED,
                version,
                tick,
                {"reason": reason_text, "previous_champion": self._version_of(previous)},
            )
        self._append(
            EventType.CHALLENGER_PROMOTED,
            version,
            tick,
            {
                "previous_champion": self._version_of(previous),
                "forced": forced,
                "reason": reason_text or None,
            },
        )

        until = self._clock.now() + timedelta(days=self._policy.probation_days)
        self._kv.set(KEY_PROBATION_VERSION, version)
        self._kv.set(KEY_PROBATION_UNTIL, until.isoformat())
        self._append(
            EventType.PROBATION_STARTED,
            version,
            tick,
            {"until": until.isoformat(), "days": self._policy.probation_days},
        )
        return promoted

    def _is_approved(self, version: int) -> bool:
        events = self._events.list(event_type=EventType.GATES_EVALUATED, model_version=version)
        if not events:
            return False
        return events[-1].payload.get("approved") is True

    def _append(
        self,
        event_type: EventType,
        version: int,
        tick: int,
        payload: dict[str, object],
    ) -> None:
        self._events.append(
            LifecycleEvent(
                ts=self._clock.now(),
                type=event_type,
                tick=tick,
                model_version=version,
                payload=payload,
            )
        )

    @staticmethod
    def _version_of(model: ModelVersion | None) -> int | None:
        return None if model is None else model.version


class RejectChallenger:
    """Mark a candidate rejected and record the failing gates."""

    def __init__(self, registry: ModelRegistry, event_store: EventStore, clock: Clock) -> None:
        self._registry = registry
        self._events = event_store
        self._clock = clock

    def execute(
        self,
        version: int,
        failed_gates: Sequence[str],
        reason: str = "",
        *,
        tick: int = 0,
    ) -> ModelVersion:
        """Reject `version`, tagging the failed gates and appending the event."""
        rejected = self._registry.reject(version, reason, failed_gates)
        self._events.append(
            LifecycleEvent(
                ts=self._clock.now(),
                type=EventType.CHALLENGER_REJECTED,
                tick=tick,
                model_version=version,
                payload={
                    "failed_gates": list(failed_gates),
                    "reason": reason,
                },
            )
        )
        return rejected


class RollbackChampion:
    """Restore the previous champion and clear the probation state."""

    def __init__(
        self,
        registry: ModelRegistry,
        event_store: EventStore,
        kv: KeyValueStore,
        clock: Clock,
    ) -> None:
        self._registry = registry
        self._events = event_store
        self._kv = kv
        self._clock = clock

    def execute(self, reason: str, *, tick: int = 0) -> ModelVersion:
        """Roll back to `previous-champion`, raising when there is none.

        Raises:
            ValueError: when the reason is empty.
            NoPreviousChampionError: when there is no previous champion (including a
                second consecutive rollback).
        """
        reason_text = reason.strip()
        if not reason_text:
            raise ValueError("rollback requires a non-empty reason")
        previous = self._registry.get_champion()
        restored = self._registry.rollback(reason_text)
        self._kv.set(KEY_PROBATION_VERSION, None)
        self._kv.set(KEY_PROBATION_UNTIL, None)
        self._events.append(
            LifecycleEvent(
                ts=self._clock.now(),
                type=EventType.ROLLBACK_EXECUTED,
                tick=tick,
                model_version=restored.version,
                payload={
                    "reason": reason_text,
                    "restored_version": restored.version,
                    "rolled_back_version": None if previous is None else previous.version,
                },
            )
        )
        return restored


__all__ = [
    "KEY_PROBATION_UNTIL",
    "KEY_PROBATION_VERSION",
    "PromoteChallenger",
    "RejectChallenger",
    "RollbackChampion",
]
