"""Deterministic clock for unit tests and drills."""

from __future__ import annotations

from datetime import datetime, timedelta


class FakeClock:
    """A settable clock implementing the application `Clock` port."""

    def __init__(self, now: datetime = datetime(2012, 1, 1)) -> None:
        self._now = now

    def now(self) -> datetime:
        """Return the frozen instant."""
        return self._now

    def set(self, moment: datetime) -> None:
        """Move the clock to `moment`."""
        self._now = moment

    def advance(self, **delta: float) -> None:
        """Advance the clock by a `timedelta` keyword delta (days=, hours=, seconds=...)."""
        self._now += timedelta(**delta)
