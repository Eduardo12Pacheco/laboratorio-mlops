"""In-memory stores for events, snapshots, key/value state and the run ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from mlops_lifecycle_lab.application.dto import MonitoringSnapshot, RunStatus, StepStatus
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent


class InMemoryEventStore:
    """Append-only event log with the filtering the dashboard and services need."""

    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []

    def append(self, event: LifecycleEvent) -> None:
        """Append one event."""
        self.events.append(event)

    def list(
        self,
        *,
        event_type: EventType | None = None,
        model_version: int | None = None,
        limit: int | None = None,
    ) -> list[LifecycleEvent]:
        """Return matching events in chronological order, keeping the last `limit`."""
        selected = [
            event
            for event in self.events
            if (event_type is None or event.type is event_type)
            and (model_version is None or event.model_version == model_version)
        ]
        if limit is not None:
            selected = selected[-limit:]
        return selected

    def types(self) -> list[EventType]:
        """Event types in append order, convenient for order assertions."""
        return [event.type for event in self.events]


class InMemorySnapshotStore:
    """Snapshot store keyed by tick, ordered chronologically."""

    def __init__(self) -> None:
        self.snapshots: list[MonitoringSnapshot] = []

    def save(self, snapshot: MonitoringSnapshot) -> None:
        """Persist a snapshot, replacing the entry of the same tick."""
        self.snapshots = [entry for entry in self.snapshots if entry.tick != snapshot.tick]
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda entry: entry.tick)

    def list(self, limit: int | None = None) -> list[MonitoringSnapshot]:
        """Return snapshots in tick order, optionally the last `limit`."""
        return self.snapshots[-limit:] if limit is not None else list(self.snapshots)


class InMemoryKeyValueStore:
    """Dictionary-backed key/value state with JSON-like values."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def get(self, key: str) -> object | None:
        """Return the stored value or `None`."""
        return self.values.get(key)

    def set(self, key: str, value: object) -> None:
        """Store or clear (`None`) a value."""
        if value is None:
            self.values.pop(key, None)
        else:
            self.values[key] = value


@dataclass(frozen=True, slots=True)
class StepLedgerEntry:
    """One attempt recorded by the fake ledger."""

    run_id: str
    index: int
    name: str
    status: StepStatus
    started_at: datetime
    finished_at: datetime
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RunLedgerEntry:
    """One final run status recorded by the fake ledger."""

    run_id: str
    status: RunStatus
    finished_at: datetime
    error: str | None = None


@dataclass
class RecordingLedger:
    """Run ledger that keeps every call so tests can assert each attempt."""

    starts: list[tuple[str, str, int | None, datetime]] = field(default_factory=list)
    steps: list[StepLedgerEntry] = field(default_factory=list)
    finishes: list[RunLedgerEntry] = field(default_factory=list)

    def start(self, run_id: str, name: str, tick: int | None, started_at: datetime) -> None:
        """Record the start of a run."""
        self.starts.append((run_id, name, tick, started_at))

    def finish_step(
        self,
        run_id: str,
        index: int,
        name: str,
        status: StepStatus,
        started_at: datetime,
        finished_at: datetime,
        error: str | None = None,
    ) -> None:
        """Record one attempt of one step."""
        self.steps.append(
            StepLedgerEntry(
                run_id=run_id,
                index=index,
                name=name,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                error=error,
            )
        )

    def finish(
        self,
        run_id: str,
        status: RunStatus,
        finished_at: datetime,
        error: str | None = None,
    ) -> None:
        """Record the final run status."""
        self.finishes.append(
            RunLedgerEntry(
                run_id=run_id,
                status=status,
                finished_at=finished_at,
                error=error,
            )
        )
