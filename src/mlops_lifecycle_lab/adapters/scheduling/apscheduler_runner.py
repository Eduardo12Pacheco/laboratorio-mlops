"""APScheduler replay runner with a persisted cursor (S14, D10, ADR-0007).

The scheduler is deliberately thin: it owns *when* the lifecycle tick runs and *where* the
replay is, never *what* a tick does. The tick function passed by the CLI is the same
`RunLifecycleTick` use case the drill calls, so scheduling and drills share one code path.

Guarantees:
    * one interval job with ``max_instances=1`` so a slow tick never overlaps itself;
    * the cursor lives in the operational KV store (``replay_tick``), so a restart
      continues at ``cursor + 1`` and never repeats a completed tick;
    * the cursor advances only after the tick returned; a failing tick leaves it intact
      (the tick itself records the failed run in the ledger);
    * the scheduler stops itself when the last tick completed and logs ``replay_complete``;
    * SIGINT/SIGTERM (POSIX) and SIGINT/SIGBREAK (Windows) shut it down cleanly.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from contextlib import suppress
from types import FrameType
from typing import Any

import structlog
from apscheduler.schedulers.background import BackgroundScheduler

from mlops_lifecycle_lab.application.ports import KeyValueStore

KEY_REPLAY_TICK = "replay_tick"
"""KV key shared with the drill engine (`entrypoints.drill.engine`)."""

JOB_ID = "mll-replay"
LOGGER = structlog.get_logger(__name__)

TickFunction = Callable[[int], None]
"""Runs one lifecycle tick; it records its own ledger rows."""

_TERMINATION_SIGNALS: tuple[str, ...] = (
    "SIGTERM",
    "SIGBREAK",
    "SIGINT",
)
"""Every signal name the scheduler tries to handle; absent names are skipped."""


class ReplayScheduler:
    """Background weekly replay driven by an interval job and a durable cursor."""

    def __init__(
        self,
        tick_fn: TickFunction,
        interval_seconds: float,
        kv: KeyValueStore,
        max_ticks: int,
        *,
        scheduler: BackgroundScheduler | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be > 0, got {interval_seconds}")
        if max_ticks < 0:
            raise ValueError(f"max_ticks must be >= 0, got {max_ticks}")
        self._tick_fn = tick_fn
        self._interval_seconds = interval_seconds
        self._kv = kv
        self._max_ticks = max_ticks
        self._scheduler = scheduler or BackgroundScheduler(timezone="UTC")
        self._lock = threading.RLock()
        self._stopped = threading.Event()
        self._previous_handlers: dict[signal.Signals, Any] = {}
        self._job_id = JOB_ID

    # -- state ------------------------------------------------------------------
    @property
    def cursor(self) -> int:
        """The last successfully completed tick, read from the KV store."""
        value = self._kv.get(KEY_REPLAY_TICK)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return 0

    @property
    def max_ticks(self) -> int:
        """The replay length in weekly ticks."""
        return self._max_ticks

    @property
    def is_running(self) -> bool:
        """Whether the underlying APScheduler is running."""
        return bool(self._scheduler.running)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the scheduler stopped (or `timeout` expires); `True` if stopped."""
        return self._stopped.wait(timeout)

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        """Start the interval job; the first tick runs after one interval."""
        if self._scheduler.running:
            return
        self._install_signal_handlers()
        self._scheduler.add_job(
            self._run_next_tick,
            trigger="interval",
            seconds=self._interval_seconds,
            id=self._job_id,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=max(1, round(self._interval_seconds)),
        )
        self._scheduler.start()

    def run_once(self) -> int | None:
        """Run the next tick synchronously; return it, or `None` when it failed/finished."""
        return self._run_next_tick()

    def shutdown(self, wait: bool = True) -> None:
        """Stop the scheduler and restore the previous signal handlers."""
        self._stopped.set()
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
        self._restore_signal_handlers()

    # -- job --------------------------------------------------------------------
    def _run_next_tick(self) -> int | None:
        """Execute `cursor + 1`, advancing the cursor only after success."""
        with self._lock:
            cursor = self.cursor
            if cursor >= self._max_ticks:
                self._complete(cursor)
                return None
            number = cursor + 1
            try:
                self._tick_fn(number)
            except Exception as exc:  # boundary: never kill the scheduler
                LOGGER.exception(
                    "replay_tick_failed",
                    tick=number,
                    error=type(exc).__name__,
                    detail=str(exc),
                )
                return None
            self._kv.set(KEY_REPLAY_TICK, number)
            LOGGER.info("replay_tick_completed", tick=number, max_ticks=self._max_ticks)
            if number >= self._max_ticks:
                self._complete(number)
                return number
            return number

    def _complete(self, cursor: int) -> None:
        """Stop the interval job and log the terminal state exactly once."""
        if self._stopped.is_set():
            return
        LOGGER.info("replay_complete", ticks=cursor, max_ticks=self._max_ticks)
        self._stopped.set()
        with suppress(Exception):
            self._scheduler.remove_job(self._job_id)
        if self._scheduler.running:
            with suppress(Exception):  # shutdown races are harmless here
                self._scheduler.shutdown(wait=False)

    # -- signals ----------------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        """Install termination handlers when running in the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return
        for name in _TERMINATION_SIGNALS:
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            except (OSError, ValueError):
                continue

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Stop the replay loop and let the CLI close the container."""
        LOGGER.info("replay_stopping", signal=signum)
        self._stopped.set()
        if self._scheduler.running:
            with suppress(Exception):  # already stopping
                self._scheduler.shutdown(wait=False)

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, previous in self._previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError):
                continue
        self._previous_handlers.clear()


__all__ = ["KEY_REPLAY_TICK", "ReplayScheduler", "TickFunction"]
