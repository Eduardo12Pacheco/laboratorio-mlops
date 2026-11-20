"""Clocks used by the application: wall clock and deterministic replay clock."""

from __future__ import annotations

from datetime import datetime, timedelta


class SystemClock:
    """Wall clock returning naive local time, matching the dataset timestamps."""

    def now(self) -> datetime:
        """Return the current instant without timezone information."""
        return datetime.now()


class SimulatedClock:
    """Settable clock for the drill and the replay scheduler."""

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        """Return the simulated instant."""
        return self._now

    def set(self, moment: datetime) -> None:
        """Move the clock to `moment`."""
        self._now = moment

    def advance(self, **delta: float) -> None:
        """Advance the clock by a `timedelta` keyword delta (days=, hours=, seconds=...)."""
        self._now += timedelta(**delta)


__all__ = ["SimulatedClock", "SystemClock"]
