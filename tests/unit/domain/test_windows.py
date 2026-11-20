"""Time windows are the shared calendar abstraction for training, replay and drills."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mlops_lifecycle_lab.domain.windows import TimeWindow, tick_window


def test_time_window_exposes_duration_and_containment() -> None:
    window = TimeWindow(datetime(2012, 1, 1), datetime(2012, 1, 8))

    assert window.duration == timedelta(days=7)
    assert window.contains(datetime(2012, 1, 1))
    assert window.contains(datetime(2012, 1, 7, 23))
    assert not window.contains(datetime(2012, 1, 8))


def test_time_window_rejects_empty_or_inverted_ranges() -> None:
    with pytest.raises(ValueError, match="start"):
        TimeWindow(datetime(2012, 1, 8), datetime(2012, 1, 1))
    with pytest.raises(ValueError, match="start"):
        TimeWindow(datetime(2012, 1, 1), datetime(2012, 1, 1))


def test_tick_window_is_one_based_and_end_exclusive() -> None:
    first = tick_window(1, datetime(2012, 1, 1), days=7)
    second = tick_window(2, datetime(2012, 1, 1), days=7)

    assert first == TimeWindow(datetime(2012, 1, 1), datetime(2012, 1, 8))
    assert second == TimeWindow(datetime(2012, 1, 8), datetime(2012, 1, 15))
    assert first.end == second.start


def test_tick_window_fifty_two_ends_on_december_thirtieth() -> None:
    window = tick_window(52, datetime(2012, 1, 1), days=7)

    assert window.start == datetime(2012, 12, 23)
    assert window.end == datetime(2012, 12, 30)


def test_tick_window_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="tick"):
        tick_window(0, datetime(2012, 1, 1), days=7)
    with pytest.raises(ValueError, match="days"):
        tick_window(1, datetime(2012, 1, 1), days=0)
