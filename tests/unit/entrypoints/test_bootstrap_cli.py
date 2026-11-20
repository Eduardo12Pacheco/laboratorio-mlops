"""``mll bootstrap --at``: deterministic bootstrap for replay workspaces (T12).

The Compose ``init`` service bootstraps before the scheduler starts replaying 2012. Without
a simulated time the persisted probation window is a 2026 wall-clock instant, so the replay
thinks v1 is still on probation and tries to roll it back instead of retraining.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from mlops_lifecycle_lab.adapters.clock import SimulatedClock
from mlops_lifecycle_lab.entrypoints.cli.app import app
from mlops_lifecycle_lab.entrypoints.cli.commands import bootstrap as bootstrap_module

runner = CliRunner()
REPLAY_START = datetime(2012, 1, 1)


def invoke(args: list[str]) -> Result:
    return runner.invoke(app, args, env={"NO_COLOR": "1", "TERM": "dumb"}, terminal_width=200)


def combined(result: Result) -> str:
    return result.output + result.stderr


class FakeBootstrap:
    def execute(self) -> SimpleNamespace:
        return SimpleNamespace(version=1, metrics={"rmse": 50.88, "wape": 0.2273})


class FakeContainer:
    def __init__(self) -> None:
        self.bootstrap = FakeBootstrap()
        self.registry = SimpleNamespace(get_champion=lambda: None)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_bootstrap_at_uses_a_simulated_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    container = FakeContainer()
    captured: dict[str, Any] = {}

    def fake_build_container(_settings: object, **kwargs: object) -> FakeContainer:
        captured.update(kwargs)
        return container

    monkeypatch.setattr(bootstrap_module, "build_container", fake_build_container)

    result = invoke(
        [
            "bootstrap",
            "--if-missing",
            "--at",
            "2012-01-01T00:00:00",
            "--workspace",
            str(tmp_path),
        ]
    )

    assert result.exit_code == 0, combined(result)
    assert "champion: v1" in result.output
    clock = captured["clock"]
    assert isinstance(clock, SimulatedClock)
    assert clock.now() == REPLAY_START
    assert container.closed is True


def test_bootstrap_without_at_uses_the_system_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    container = FakeContainer()
    captured: dict[str, Any] = {}

    def fake_build_container(_settings: object, **kwargs: object) -> FakeContainer:
        captured.update(kwargs)
        return container

    monkeypatch.setattr(bootstrap_module, "build_container", fake_build_container)

    result = invoke(["bootstrap", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, combined(result)
    assert "clock" not in captured
