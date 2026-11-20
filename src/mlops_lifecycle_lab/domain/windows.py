"""Half-open time windows shared by training, replay and drills.

Windows are always `[start, end)`. A "tick" is one replay step of `days` days, numbered
from 1; tick `k` ends at `start + k * days`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A half-open interval `[start, end)` of naive datetimes."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(f"start must be before end, got {self.start} >= {self.end}")

    @property
    def duration(self) -> timedelta:
        """Length of the window."""
        return self.end - self.start

    def contains(self, moment: datetime) -> bool:
        """Whether `moment` falls inside the half-open interval."""
        return self.start <= moment < self.end


def tick_window(tick: int, start: datetime, days: int) -> TimeWindow:
    """Return the window of replay tick `tick` (1-based) starting from `start`."""
    if tick < 1:
        raise ValueError(f"tick must be >= 1, got {tick}")
    if days <= 0:
        raise ValueError(f"days must be > 0, got {days}")
    step = timedelta(days=days)
    return TimeWindow(start + step * (tick - 1), start + step * tick)
