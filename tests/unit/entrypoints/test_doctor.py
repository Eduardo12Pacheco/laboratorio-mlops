"""``mll doctor``: hard checks fail the command, informational ones never do (S25, T10)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner, Result

from mlops_lifecycle_lab.config import Settings
from mlops_lifecycle_lab.entrypoints.cli.app import app
from mlops_lifecycle_lab.entrypoints.cli.commands import doctor as doctor_module
from mlops_lifecycle_lab.entrypoints.cli.commands.doctor import (
    CheckResult,
    CheckStatus,
    collect_checks,
    format_table,
)

runner = CliRunner()


def invoke(args: list[str]) -> Result:
    return runner.invoke(app, args, env={"NO_COLOR": "1", "TERM": "dumb"}, terminal_width=200)


def combined(result: Result) -> str:
    return result.output + result.stderr


def test_collect_checks_verifies_lightgbm_and_the_workspace(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path / "ws")

    checks = collect_checks(settings, include_docker=False, include_ports=False)
    by_name = {check.name: check for check in checks}

    assert by_name["python"].status is CheckStatus.OK
    assert by_name["runtime_imports"].status is CheckStatus.OK
    assert by_name["lightgbm"].status is CheckStatus.OK
    assert "fit 10 rows" in by_name["lightgbm"].detail
    assert by_name["sqlite_wal"].status is CheckStatus.OK
    assert by_name["workspace_writable"].status is CheckStatus.OK
    assert by_name["mlflow_sqlite"].status is CheckStatus.OK
    assert (tmp_path / "ws").is_dir()
    assert "lightgbm  OK" in format_table(checks)


def test_informational_checks_warn_but_never_fail(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setattr(doctor_module.shutil, "which", lambda _name: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(doctor_module, "_port_free", lambda _port: False)  # type: ignore[attr-defined]

    checks = collect_checks(Settings(workspace=tmp_path / "ws"))

    by_name = {check.name: check for check in checks}
    assert by_name["docker"].status is CheckStatus.WARN
    assert by_name["ports"].status is CheckStatus.WARN
    assert "14000" in by_name["ports"].detail
    assert doctor_module.hard_failures(checks) == []


def test_doctor_cli_exits_zero_and_prints_lightgbm_ok(tmp_path: Path) -> None:
    result = invoke(["doctor", "--workspace", str(tmp_path / "ws")])

    assert result.exit_code == 0, combined(result)
    assert "lightgbm  OK" in result.output
    assert "doctor: OK" in result.output


def test_doctor_exits_one_on_a_hard_failure(monkeypatch: object) -> None:
    failures = [CheckResult("lightgbm", CheckStatus.FAIL, "libgomp missing", hard=True)]
    monkeypatch.setattr(doctor_module, "collect_checks", lambda *_args, **_kwargs: failures)  # type: ignore[attr-defined]

    result = invoke(["doctor"])

    assert result.exit_code == 1, combined(result)
    assert "lightgbm  FAIL" in result.output
    assert "doctor: 1 check failed" in result.output


def test_warn_only_checks_keep_the_exit_code_zero(monkeypatch: object) -> None:
    warnings = [CheckResult("docker", CheckStatus.WARN, "docker not found", hard=False)]
    monkeypatch.setattr(doctor_module, "collect_checks", lambda *_args, **_kwargs: warnings)  # type: ignore[attr-defined]

    result = invoke(["doctor"])

    assert result.exit_code == 0, combined(result)
    assert "docker  WARN" in result.output
