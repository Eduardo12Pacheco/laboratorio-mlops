"""``mll quickstart``: doctor-lite, bootstrap + 16-week drill, next step (S20, T10)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner, Result

from mlops_lifecycle_lab.adapters.storage.kv_store import SqliteKeyValueStore
from mlops_lifecycle_lab.adapters.storage.sqlite import SqliteDatabase
from mlops_lifecycle_lab.entrypoints.cli.app import app
from mlops_lifecycle_lab.entrypoints.cli.commands import quickstart as quickstart_module
from mlops_lifecycle_lab.entrypoints.cli.commands.doctor import CheckResult, CheckStatus

runner = CliRunner()


def invoke(args: list[str]) -> Result:
    return runner.invoke(app, args, env={"NO_COLOR": "1", "TERM": "dumb"}, terminal_width=200)


def combined(result: Result) -> str:
    return result.output + result.stderr


def ok_checks(*_args: object, **_kwargs: object) -> list[CheckResult]:
    return [CheckResult("lightgbm", CheckStatus.OK, "fit 10 rows", hard=True)]


def set_replay_tick(workspace: Path, tick: int) -> None:
    database = SqliteDatabase(workspace / "mll.db")
    try:
        SqliteKeyValueStore(database).set("replay_tick", tick)
    finally:
        database.close()


def test_quickstart_runs_the_default_sixteen_week_drill_and_prints_next_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "ws"
    calls: dict[str, object] = {}

    def fake_drill(target: Path, weeks: int) -> object:
        calls["workspace"] = target
        calls["weeks"] = weeks
        set_replay_tick(target, weeks)
        return SimpleNamespace(ticks=[SimpleNamespace(champion_version=2)])

    monkeypatch.setattr(quickstart_module, "collect_checks", ok_checks)
    monkeypatch.setattr(quickstart_module, "_run_drill", fake_drill)

    result = invoke(["quickstart", "--workspace", str(workspace)])

    assert result.exit_code == 0, combined(result)
    assert "champion: v2" in result.output
    assert "Next: mll dashboard" in result.output
    assert calls == {"workspace": workspace, "weeks": 16}

    status = invoke(["replay", "status", "--workspace", str(workspace)])
    assert status.exit_code == 0, combined(status)
    assert "tick 16 / 52" in status.output

    reset = invoke(["replay", "reset", "--workspace", str(workspace)])
    assert reset.exit_code == 0, combined(reset)
    assert "tick 0 / 52" in reset.output

    after = invoke(["replay", "status", "--workspace", str(workspace)])
    assert after.exit_code == 0, combined(after)
    assert "tick 0 / 52" in after.output


def test_quickstart_accepts_a_custom_week_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_drill(target: Path, weeks: int) -> object:
        calls["weeks"] = weeks
        return SimpleNamespace(ticks=[SimpleNamespace(champion_version=2)])

    monkeypatch.setattr(quickstart_module, "collect_checks", ok_checks)
    monkeypatch.setattr(quickstart_module, "_run_drill", fake_drill)

    result = invoke(["quickstart", "--weeks", "4", "--workspace", str(tmp_path / "ws")])

    assert result.exit_code == 0, combined(result)
    assert calls["weeks"] == 4


def test_quickstart_stops_on_a_hard_doctor_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def failing(*_args: object, **_kwargs: object) -> list[CheckResult]:
        return [CheckResult("lightgbm", CheckStatus.FAIL, "libgomp missing", hard=True)]

    monkeypatch.setattr(quickstart_module, "collect_checks", failing)
    monkeypatch.setattr(
        quickstart_module,
        "_run_drill",
        lambda *_args, **_kwargs: pytest.fail("the drill must not run"),
    )

    result = invoke(["quickstart", "--workspace", str(tmp_path / "ws")])

    assert result.exit_code == 1, combined(result)
    assert "lightgbm  FAIL" in result.output


def test_quickstart_reports_a_failed_drill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(quickstart_module, "collect_checks", ok_checks)

    def broken(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("drill blew up")

    monkeypatch.setattr(quickstart_module, "_run_drill", broken)

    result = invoke(["quickstart", "--workspace", str(tmp_path / "ws")])

    assert result.exit_code == 1, combined(result)
    assert "drill blew up" in combined(result)


def test_run_drill_uses_the_reproducible_drift_scenario_without_fast_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeEngine:
        def run(self, scenario: str, **kwargs: object) -> object:
            captured["scenario"] = scenario
            captured.update(kwargs)
            return SimpleNamespace(ticks=[])

    monkeypatch.setattr(quickstart_module, "DrillEngine", FakeEngine)

    quickstart_module._run_drill(tmp_path, 3)

    assert captured["scenario"] == "drift"
    assert captured["weeks"] == 3
    assert captured["workspace"] == tmp_path
    assert captured["fast"] is False
