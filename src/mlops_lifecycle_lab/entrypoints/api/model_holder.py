"""Hot-reloading model holder backed by the registry aliases (S8, D13).

The holder loads a newly aliased champion *before* swapping the serving reference, so a
slow or failing artifact load never interrupts predictions: the old model keeps serving
and `mll_model_reload_failures_total` is incremented. Reloads happen every
`MLL_RELOAD_INTERVAL_SECONDS` in a lifespan task, without restarting the process.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime

import structlog

from mlops_lifecycle_lab.application.ports import (
    Clock,
    ModelRegistry,
    Predictor,
    ServingModel,
)
from mlops_lifecycle_lab.entrypoints.api.metrics import ApiMetrics

LOGGER = structlog.get_logger(__name__)


class ModelHolder:
    """Owns the loaded champion predictor and polls the registry for changes."""

    def __init__(
        self,
        registry: ModelRegistry,
        metrics: ApiMetrics,
        clock: Clock,
        *,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self._registry = registry
        self._metrics = metrics
        self._clock = clock
        self._interval = interval_seconds
        self._lock = threading.Lock()
        self._current: ServingModel | None = None
        self._loaded_at: datetime | None = None
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def current(self) -> ServingModel | None:
        """The atomically-loaded serving model, or `None` before the first success."""
        with self._lock:
            return self._current

    @property
    def loaded_at(self) -> datetime | None:
        """When the currently served model was loaded."""
        with self._lock:
            return self._loaded_at

    @property
    def is_ready(self) -> bool:
        """True only when a champion artifact is loaded and can be predicted with."""
        return self.current() is not None

    def refresh(self, *, force: bool = False) -> bool:
        """Poll the registry and swap the predictor when the champion changed.

        Returns True when a new model became active. Any failure is logged and counted,
        and the previous model remains active.
        """
        try:
            champion = self._registry.get_champion()
            if champion is None:
                return False
            current = self.current()
            if (
                not force
                and current is not None
                and current.model.version == champion.version
                and current.model.stage is champion.stage
            ):
                return False
            predictor = self._registry.load_predictor(champion.version)
            self._swap(ServingModel(model=champion, predictor=predictor), self._clock.now())
        except Exception as exc:
            LOGGER.warning("model_reload_failed", error=str(exc))
            self._metrics.record_reload_failure()
            return False
        self._metrics.set_champion(champion.version, champion.stage.value)
        LOGGER.info(
            "model_reload_succeeded",
            model_version=champion.version,
            stage=champion.stage.value,
        )
        return True

    async def start(self) -> None:
        """Load the current champion and start the periodic refresh loop."""
        self._stop = asyncio.Event()
        await asyncio.to_thread(self.refresh, force=True)
        self._task = asyncio.create_task(self._poll())

    async def stop(self) -> None:
        """Stop the periodic refresh loop and wait for it to finish."""
        self._stop.set()
        task = self._task
        self._task = None
        if task is not None:
            await task

    async def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except TimeoutError:
                await asyncio.to_thread(self.refresh)

    def _swap(self, serving: ServingModel, loaded_at: datetime) -> None:
        with self._lock:
            self._current = serving
            self._loaded_at = loaded_at


def predictor_of(holder: ModelHolder) -> Predictor | None:
    """Return the loaded predictor, if any (small helper for diagnostics/tests)."""
    current = holder.current()
    return None if current is None else current.predictor


__all__ = ["ModelHolder", "predictor_of"]
