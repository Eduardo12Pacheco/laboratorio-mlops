"""A tiny `Pipeline`/`Step` runner with retries and a run ledger (D10, S14).

The lifecycle tick and every CLI command that performs several fallible operations run
through this module, so retries and ledger rows exist in exactly one place. A pipeline
never hides a failure: after the retry budget is spent it raises `PipelineError` naming
the step, while the ledger keeps one row per attempt for the dashboard.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from mlops_lifecycle_lab.application.dto import RunStatus, StepStatus
from mlops_lifecycle_lab.application.ports import Clock, RunLedger
from mlops_lifecycle_lab.domain.errors import DomainError


class _SystemClock:
    """Default wall clock; adapters replace it with `SystemClock`."""

    def now(self) -> datetime:
        return datetime.now()


class PipelineError(DomainError):
    """A pipeline step failed after exhausting its retries."""

    def __init__(self, step: str, attempts: int, original: Exception) -> None:
        self.step = step
        self.attempts = attempts
        self.original = original
        super().__init__(
            f"pipeline step {step!r} failed after {attempts} attempt(s): "
            f"{type(original).__name__}: {original}"
        )


@dataclass(frozen=True, slots=True)
class Step:
    """One named pipeline operation with its retry policy."""

    name: str
    fn: Callable[[object], object]
    retries: int = 0
    backoff_seconds: float = 0.0
    retry_on: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("step name must not be empty")
        if self.retries < 0:
            raise ValueError(f"retries must be >= 0, got {self.retries}")
        if self.backoff_seconds < 0.0:
            raise ValueError(f"backoff_seconds must be >= 0, got {self.backoff_seconds}")
        if not self.retry_on:
            raise ValueError("retry_on must contain at least one exception type")


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Final outcome of one step (with the number of attempts it took)."""

    name: str
    status: StepStatus
    attempts: int
    started_at: datetime
    finished_at: datetime
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """Result of a successful pipeline execution."""

    run_id: str
    name: str
    context: object
    steps: tuple[StepRecord, ...] = field(default_factory=tuple)
    status: RunStatus = RunStatus.SUCCEEDED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


def _tick_of(context: object) -> int | None:
    if isinstance(context, Mapping):
        candidate = context.get("tick")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


class Pipeline:
    """Sequential steps over one context, recording starts, attempts and the outcome."""

    def __init__(
        self,
        steps: Sequence[Step],
        ledger: RunLedger,
        *,
        name: str = "pipeline",
        sleeper: Callable[[float], None] = time.sleep,
        clock: Clock | None = None,
    ) -> None:
        names = [step.name for step in steps]
        if len(names) != len(set(names)):
            raise ValueError("pipeline step names must be unique")
        self._steps = tuple(steps)
        self._ledger = ledger
        self._name = name
        self._sleeper = sleeper
        self._clock = clock or _SystemClock()

    def run(self, context: object) -> PipelineRun:
        """Execute every step in order, retrying as configured.

        A step may return a replacement context (or `None` to keep the current one).

        Raises:
            PipelineError: when a step still fails after its retry budget.
        """
        run_id = uuid.uuid4().hex
        started_at = self._clock.now()
        self._ledger.start(run_id, self._name, _tick_of(context), started_at)

        records: list[StepRecord] = []
        ledger_index = 0
        current = context
        for step in self._steps:
            attempts = 0
            while True:
                attempts += 1
                attempt_started = self._clock.now()
                try:
                    result = step.fn(current)
                except Exception as exc:
                    finished_at = self._clock.now()
                    error = f"{type(exc).__name__}: {exc}"
                    self._ledger.finish_step(
                        run_id,
                        ledger_index,
                        step.name,
                        StepStatus.FAILED,
                        attempt_started,
                        finished_at,
                        error,
                    )
                    ledger_index += 1
                    if isinstance(exc, step.retry_on) and attempts <= step.retries:
                        if step.backoff_seconds > 0.0:
                            self._sleeper(step.backoff_seconds)
                        continue
                    failed_at = self._clock.now()
                    self._ledger.finish(run_id, RunStatus.FAILED, failed_at, error)
                    raise PipelineError(step.name, attempts, exc) from exc
                else:
                    finished_at = self._clock.now()
                    if result is not None:
                        current = result
                    self._ledger.finish_step(
                        run_id,
                        ledger_index,
                        step.name,
                        StepStatus.SUCCEEDED,
                        attempt_started,
                        finished_at,
                    )
                    ledger_index += 1
                    records.append(
                        StepRecord(
                            name=step.name,
                            status=StepStatus.SUCCEEDED,
                            attempts=attempts,
                            started_at=attempt_started,
                            finished_at=finished_at,
                        )
                    )
                    break

        finished_at = self._clock.now()
        self._ledger.finish(run_id, RunStatus.SUCCEEDED, finished_at)
        return PipelineRun(
            run_id=run_id,
            name=self._name,
            context=current,
            steps=tuple(records),
            status=RunStatus.SUCCEEDED,
            started_at=started_at,
            finished_at=finished_at,
        )


__all__ = ["Pipeline", "PipelineError", "PipelineRun", "Step", "StepRecord"]
