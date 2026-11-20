"""Model holder: atomic hot reload, failure isolation and reload metrics."""

from __future__ import annotations

from datetime import datetime

from mlops_lifecycle_lab.application.ports import Predictor
from mlops_lifecycle_lab.domain.lifecycle import ModelVersion
from mlops_lifecycle_lab.entrypoints.api.metrics import ApiMetrics
from mlops_lifecycle_lab.entrypoints.api.model_holder import ModelHolder
from tests.fakes.clock import FakeClock
from tests.fakes.registry import InMemoryRegistry
from tests.unit.entrypoints.conftest import register_champion


class ToggleRegistry:
    """Registry proxy that can fail reads or loads on demand."""

    def __init__(self, inner: InMemoryRegistry) -> None:
        self.inner = inner
        self.fail_champion = False
        self.fail_load_for: int | None = None

    def get_champion(self) -> ModelVersion | None:
        if self.fail_champion:
            raise RuntimeError("registry unavailable")
        return self.inner.get_champion()

    def load_predictor(self, version: int) -> Predictor:
        if self.fail_load_for == version:
            raise RuntimeError("artifact store unavailable")
        return self.inner.load_predictor(version)


def make_holder(
    registry: InMemoryRegistry,
    metrics: ApiMetrics | None = None,
) -> ModelHolder:
    return ModelHolder(
        registry,
        metrics or ApiMetrics(),
        FakeClock(datetime(2012, 1, 1)),
        interval_seconds=3600.0,
    )


def test_refresh_loads_the_champion_and_stamps_the_clock() -> None:
    registry = InMemoryRegistry()
    register_champion(registry, version=1, prediction=123.0)
    metrics = ApiMetrics()
    holder = make_holder(registry, metrics)

    assert holder.current() is None
    assert holder.is_ready is False
    assert holder.refresh() is True

    serving = holder.current()
    assert serving is not None
    assert serving.model.version == 1
    assert serving.predictor.predict.__self__ is not None  # type: ignore[attr-defined]
    assert holder.loaded_at == datetime(2012, 1, 1)
    assert holder.is_ready is True
    assert "mll_champion_version 1.0" in metrics.render().decode()


def test_refresh_swaps_to_a_new_champion() -> None:
    registry = InMemoryRegistry()
    register_champion(registry, version=1, prediction=100.0)
    holder = make_holder(registry)
    holder.refresh()

    register_champion(registry, version=2, prediction=200.0)
    assert holder.refresh() is True
    serving = holder.current()
    assert serving is not None
    assert serving.model.version == 2


def test_failed_registry_read_keeps_the_previous_model_and_counts_the_failure() -> None:
    inner = InMemoryRegistry()
    register_champion(inner, version=1, prediction=100.0)
    registry = ToggleRegistry(inner)
    metrics = ApiMetrics()
    holder = make_holder(registry, metrics)  # type: ignore[arg-type]
    holder.refresh()

    registry.fail_champion = True
    assert holder.refresh() is False
    serving = holder.current()
    assert serving is not None
    assert serving.model.version == 1
    assert "mll_model_reload_failures_total 1.0" in metrics.render().decode()


def test_failed_artifact_load_keeps_the_previous_model() -> None:
    inner = InMemoryRegistry()
    register_champion(inner, version=1, prediction=100.0)
    registry = ToggleRegistry(inner)
    metrics = ApiMetrics()
    holder = make_holder(registry, metrics)  # type: ignore[arg-type]
    holder.refresh()

    register_champion(inner, version=2, prediction=200.0)
    registry.fail_load_for = 2
    assert holder.refresh() is False
    serving = holder.current()
    assert serving is not None
    assert serving.model.version == 1
    assert "mll_model_reload_failures_total 1.0" in metrics.render().decode()
