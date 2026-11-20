"""Domain events appended to the lifecycle audit log (S14).

Events are plain, frozen values so they can be persisted in SQLite, rendered by the
dashboard and replayed from a report without importing any adapter. Payloads must be
JSON-serialisable; ``to_dict``/``from_dict`` are the canonical conversion.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class EventType(StrEnum):
    """Every event the lifecycle tick can append to the audit log."""

    BOOTSTRAP_COMPLETED = "BOOTSTRAP_COMPLETED"
    BATCH_VALIDATED = "BATCH_VALIDATED"
    BATCH_QUARANTINED = "BATCH_QUARANTINED"
    BATCH_SCORED = "BATCH_SCORED"
    LABELS_INGESTED = "LABELS_INGESTED"
    SNAPSHOT_COMPUTED = "SNAPSHOT_COMPUTED"
    RETRAIN_TRIGGERED = "RETRAIN_TRIGGERED"
    CHALLENGER_REGISTERED = "CHALLENGER_REGISTERED"
    GATES_EVALUATED = "GATES_EVALUATED"
    CHALLENGER_PROMOTED = "CHALLENGER_PROMOTED"
    CHALLENGER_REJECTED = "CHALLENGER_REJECTED"
    PROMOTION_FORCED = "PROMOTION_FORCED"
    PROBATION_STARTED = "PROBATION_STARTED"
    PROBATION_PASSED = "PROBATION_PASSED"
    ROLLBACK_EXECUTED = "ROLLBACK_EXECUTED"


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """One immutable audit-log entry.

    Attributes:
        ts: event time, naive like every timestamp in the dataset.
        type: one of `EventType`.
        tick: replay tick the event belongs to; 0 is the bootstrap.
        model_version: affected model version, when the event is about one.
        payload: JSON-serialisable event detail (gate report, metrics, reasons).
    """

    ts: datetime
    type: EventType
    tick: int
    model_version: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError(f"tick must be >= 0, got {self.tick}")
        if self.model_version is not None and self.model_version <= 0:
            raise ValueError(f"model_version must be positive, got {self.model_version}")
        object.__setattr__(self, "type", EventType(self.type))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation persisted by the event store."""
        return {
            "ts": self.ts.isoformat(),
            "type": self.type.value,
            "tick": self.tick,
            "model_version": self.model_version,
            "payload": copy.deepcopy(dict(self.payload)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LifecycleEvent:
        """Rebuild an event from `to_dict` output (or the SQLite JSON column)."""
        try:
            event_type = EventType(str(data["type"]))
            ts = datetime.fromisoformat(str(data["ts"]))
            tick = int(data["tick"])
        except KeyError as exc:
            raise ValueError(f"event is missing field {exc.args[0]!r}") from exc
        raw_version = data.get("model_version")
        return cls(
            ts=ts,
            type=event_type,
            tick=tick,
            model_version=None if raw_version is None else int(raw_version),
            payload=dict(data.get("payload") or {}),
        )
