"""Static contract for the PowerShell installer (`install.ps1`, S20).

The executor machine has no ``pwsh``; when one is available the script is parsed with the
real PowerShell language parser, otherwise the test is skipped with an explicit reason
(D27). The static assertions still pin the documented parameters, strict mode, exit
codes and host ports so a regression cannot slip through on Linux CI.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PS1 = ROOT / "install.ps1"
PWSH = shutil.which("pwsh") or shutil.which("powershell")


def _text() -> str:
    return PS1.read_text(encoding="utf-8")


def test_declares_the_documented_parameters() -> None:
    text = _text()

    for parameter in (
        "$Check",
        "$L1",
        "$Uninstall",
        "$Yes",
        "$NoStart",
        "$ProjectName",
    ):
        assert parameter in text, parameter
    assert "Set-StrictMode -Version Latest" in text
    assert "$ErrorActionPreference = 'Stop'" in text


def test_uses_the_documented_exit_codes() -> None:
    text = _text()

    for code in ("0", "1", "2", "3", "4"):
        assert f"exit {code}" in text, code
    assert "prerequisite" in text.lower()
    assert "uninstall" in text.lower()
    assert "aborted" in text.lower()


def test_uses_the_portfolio_host_ports() -> None:
    text = _text()

    for port in ("14000", "14080", "14501"):
        assert port in text, port


def test_checks_the_docker_engine_os() -> None:
    text = _text().lower()

    assert "ostype" in text
    assert "windows" in text
    assert "linux" in text


def test_warns_instead_of_failing_on_windows_containers() -> None:
    text = _text().lower()

    # The plan (T13) requires a warning on Windows containers, not a hard failure.
    assert "warn" in text
    assert "windows containers" in text or "engine os" in text


@pytest.mark.skipif(PWSH is None, reason="pwsh/powershell not installed on this machine")
def test_script_parses_with_the_power_shell_language_parser() -> None:
    command = (
        "$errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{PS1.as_posix()}', [ref]$null, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
