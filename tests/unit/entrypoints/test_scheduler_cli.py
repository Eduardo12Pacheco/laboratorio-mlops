"""``mll scheduler``: replay ticking through the composition root (S14, T10)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from typer.testing import CliRunner, Result

from mlops_lifecycle_lab.adapters.clock import SimulatedClock
from mlops_lifecycle_lab.application.services.lifecycle import TickRequest
from mlops_lifecycle_lab.domain.windows import tick_window
from mlops_lifecycle_lab.entrypoints.cli.app import app
from mlops_lifecycle_lab.entrypoints.cli.commands import scheduler as scheduler_module
from tests.fakes.stores import InMemoryKeyValueStore

runner = CliRunner()


def invoke(args: list[str]) -> Result:
    return runner.invoke(app, args, env={"NO_COLOR": "1", "TERM": "dumb"}, terminal_width=200)


def combined(result: Result) -> str:
    return result.output + result.stderr


class RecordingLifecycle:
    def __init__(self) -> None:
        self.requests: list[TickRequest] = []

    def execute(self, request: TickRequest) -> object:
        self.requests.append(request)
        return object()


class RecordingBootstrap:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(version=1)


class FakeRegistry:
    def __init__(self, champion: object | None = None) -> None:
        self._champion = champion

    def get_champion(self) -> object | None:
        return self._champion


class FakeContainer:
    def __init__(self, champion: object | None = None) -> None:
        self.clock = SimulatedClock(datetime(2012, 1, 1))
        self.kv = InMemoryKeyValueStore()
        self.lifecycle = RecordingLifecycle()
        self.bootstrap = RecordingBootstrap()
        self.registry = FakeRegistry(champion)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeReplayScheduler:
    instances: ClassVar[list[FakeReplayScheduler]] = []

    def __init__(
        self,
        tick_fn: Callable[[int], None],
        interval_seconds: float,
        kv: object,
        max_ticks: int,
    ) -> None:
        self.tick_fn = tick_fn
        self.interval_seconds = interval_seconds
        self.kv = kv
        self.max_ticks = max_ticks
        self.started = False
        self.waited = False
        self.shut_down = False
        self.cursor = 0
        self.run_once_result: int | None = 1
        FakeReplayScheduler.instances.append(self)

    def start(self) -> None:
        self.started = True

    def wait(self, timeout: float | None = None) -> bool:
        self.waited = True
        return True

    def shutdown(self, wait: bool = True) -> None:
        self.shut_down = True

    def run_once(self) -> int | None:
        self.cursor = self.run_once_result or 0
        return self.run_once_result


def test_tick_function_advances_the_simulated_clock_and_window() -> None:
    container = FakeContainer()
    tick_fn = scheduler_module._tick_function(container, container.clock)  # type: ignore[arg-type]

    tick_fn(5)

    request = container.lifecycle.requests[0]
    assert request.tick == 5
    assert request.window == tick_window(5, scheduler_module.REPLAY_START, 7)
    assert container.clock.now() == request.window.end


def test_scheduler_once_bootstraps_and_runs_one_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeReplayScheduler.instances.clear()
    container = FakeContainer()
    monkeypatch.setattr(scheduler_module, "build_container", lambda *_args, **_kwargs: container)
    monkeypatch.setattr(scheduler_module, "ReplayScheduler", FakeReplayScheduler)

    result = invoke(["scheduler", "start", "--mode", "once", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, combined(result)
    assert container.bootstrap.calls == 1
    assert container.closed is True
    replay = FakeReplayScheduler.instances[0]
    assert replay.max_ticks == 52
    assert replay.run_once_result == 1
    assert "tick 1 / 52" in result.output


def test_scheduler_replay_starts_waits_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeReplayScheduler.instances.clear()
    container = FakeContainer(champion=object())
    monkeypatch.setattr(scheduler_module, "build_container", lambda *_args, **_kwargs: container)
    monkeypatch.setattr(scheduler_module, "ReplayScheduler", FakeReplayScheduler)

    result = invoke(["scheduler", "start", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, combined(result)
    assert container.bootstrap.calls == 0
    replay = FakeReplayScheduler.instances[0]
    assert replay.started is True
    assert replay.waited is True
    assert replay.shut_down is True
    assert "replay complete" in result.output


def test_scheduler_rejects_an_unknown_mode(tmp_path: Path) -> None:
    result = invoke(["scheduler", "start", "--mode", "turbo", "--workspace", str(tmp_path)])

    assert result.exit_code == 2, combined(result)
    assert "mode" in combined(result)
