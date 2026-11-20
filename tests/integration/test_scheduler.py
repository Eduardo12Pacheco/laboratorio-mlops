"""Replay scheduler: overlap guard, persisted cursor, restart and clean stop (S14, T10).

The scheduler owns only scheduling and the replay cursor. The tick function owns the run
ledger (the lifecycle pipeline records every attempt), so these tests drive it with a
recorder that behaves like the real tick: a failure is recorded as FAILED and re-raised.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from datetime import datetime

import pytest
from structlog.testing import capture_logs

from mlops_lifecycle_lab.adapters.scheduling.apscheduler_runner import (
    KEY_REPLAY_TICK,
    ReplayScheduler,
)
from mlops_lifecycle_lab.application.dto import RunStatus
from tests.fakes.stores import InMemoryKeyValueStore, RecordingLedger

INTERVAL = 0.02


class TickRecorder:
    """Tick callable that tracks overlap, latency and optional failure."""

    def __init__(
        self,
        *,
        delay: float = 0.0,
        fail_on: int | None = None,
        ledger: RecordingLedger | None = None,
    ) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.ledger = ledger
        self.ticks: list[int] = []
        self.active = 0
        self.max_active = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, tick: int) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.ticks.append(tick)
        self.started.set()
        now = datetime(2012, 1, 1)
        run_id = f"tick-{tick}"
        if self.ledger is not None:
            self.ledger.start(run_id, run_id, tick, now)
        try:
            if self.delay > 0:
                time.sleep(self.delay)
            if self.fail_on is not None and tick == self.fail_on:
                raise RuntimeError("gateway unavailable")
        except RuntimeError as exc:
            if self.ledger is not None:
                self.ledger.finish(run_id, RunStatus.FAILED, now, str(exc))
            raise
        finally:
            with self._lock:
                self.active -= 1
        if self.ledger is not None:
            self.ledger.finish(run_id, RunStatus.SUCCEEDED, now)


def wait_for(predicate: object, timeout: float = 10.0) -> None:
    """Poll a zero-argument predicate until it is true, or fail after `timeout`."""
    check = predicate
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():  # type: ignore[operator]
            return
        time.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout}s")


def test_slow_tick_never_runs_concurrently_with_itself() -> None:
    recorder = TickRecorder(delay=0.15)
    scheduler = ReplayScheduler(recorder, INTERVAL, InMemoryKeyValueStore(), max_ticks=3)
    scheduler.start()
    try:
        assert scheduler.wait(15) is True
    finally:
        scheduler.shutdown(wait=True)

    assert recorder.ticks == [1, 2, 3]
    assert recorder.max_active == 1


def test_cursor_survives_a_restart_and_never_repeats_a_tick() -> None:
    kv = InMemoryKeyValueStore()
    first = TickRecorder()
    first_scheduler = ReplayScheduler(first, INTERVAL, kv, max_ticks=5)
    first_scheduler.start()
    try:
        wait_for(lambda: first_scheduler.cursor >= 2)
    finally:
        first_scheduler.shutdown(wait=True)

    prefix = list(first.ticks)
    assert prefix
    assert kv.get(KEY_REPLAY_TICK) == prefix[-1]

    second = TickRecorder()
    second_scheduler = ReplayScheduler(second, INTERVAL, kv, max_ticks=5)
    second_scheduler.start()
    try:
        assert second_scheduler.wait(15) is True
    finally:
        second_scheduler.shutdown(wait=True)

    assert prefix + second.ticks == [1, 2, 3, 4, 5]
    assert kv.get(KEY_REPLAY_TICK) == 5


def test_stops_at_max_ticks_and_logs_replay_complete() -> None:
    kv = InMemoryKeyValueStore()
    recorder = TickRecorder()
    scheduler = ReplayScheduler(recorder, INTERVAL, kv, max_ticks=2)
    with capture_logs() as logs:
        scheduler.start()
        assert scheduler.wait(10) is True

    assert scheduler.cursor == 2
    assert scheduler.is_running is False
    assert recorder.ticks == [1, 2]
    assert any(str(entry.get("event")) == "replay_complete" for entry in logs)
    scheduler.shutdown(wait=True)


def test_shutdown_returns_within_two_seconds_while_a_tick_is_running() -> None:
    recorder = TickRecorder(delay=0.4)
    scheduler = ReplayScheduler(recorder, INTERVAL, InMemoryKeyValueStore(), max_ticks=50)
    scheduler.start()
    try:
        assert recorder.started.wait(5) is True
        started = time.monotonic()
        scheduler.shutdown(wait=True)
        elapsed = time.monotonic() - started
    finally:
        scheduler.shutdown(wait=True)

    assert elapsed < 2.0
    assert scheduler.is_running is False


def test_failed_tick_is_logged_ledger_failed_and_cursor_stays() -> None:
    kv = InMemoryKeyValueStore()
    ledger = RecordingLedger()
    recorder = TickRecorder(fail_on=2, ledger=ledger)
    scheduler = ReplayScheduler(recorder, INTERVAL, kv, max_ticks=5)

    assert scheduler.run_once() == 1
    assert scheduler.cursor == 1

    with capture_logs() as logs:
        assert scheduler.run_once() is None

    assert scheduler.cursor == 1
    assert recorder.ticks == [1, 2]
    assert any(str(entry.get("event")) == "replay_tick_failed" for entry in logs)
    assert any(entry.status is RunStatus.FAILED for entry in ledger.finishes)
    assert kv.get(KEY_REPLAY_TICK) == 1
    scheduler.shutdown(wait=True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics only")
def test_sigterm_stops_the_replay_scheduler() -> None:
    recorder = TickRecorder()
    scheduler = ReplayScheduler(recorder, INTERVAL, InMemoryKeyValueStore(), max_ticks=100)
    scheduler.start()
    try:
        assert recorder.started.wait(5) is True
        os.kill(os.getpid(), signal.SIGTERM)
        assert scheduler.wait(2.0) is True
        assert scheduler.is_running is False
    finally:
        scheduler.shutdown(wait=True)
