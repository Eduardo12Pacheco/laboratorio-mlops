"""``mll doctor``: verify the machine can run the lab (S25, D26, T10).

Hard checks (Python 3.12, runtime imports, LightGBM fit, SQLite WAL, a writable workspace
and MLflow tracking database) make the command exit 1 when they fail. Docker and host-port
availability are informational: the L1 path works without Docker and a stale dev port is
not a reason to declare the installation broken.
"""

from __future__ import annotations

import importlib
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults


class CheckStatus(StrEnum):
    """Outcome of one diagnostic check."""

    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One row of the doctor report."""

    name: str
    status: CheckStatus
    detail: str
    hard: bool = True


RUNTIME_MODULES: tuple[str, ...] = (
    "mlflow",
    "lightgbm",
    "sklearn",
    "pandas",
    "numpy",
    "scipy",
    "fastapi",
    "uvicorn",
    "httpx",
    "pydantic",
    "pydantic_settings",
    "evidently",
    "apscheduler",
    "streamlit",
    "plotly",
    "typer",
    "structlog",
    "prometheus_client",
)
"""Every runtime dependency of the packaged application (D9)."""

MLFLOW_PORT = 14000
DASHBOARD_PORT = 14501
"""Host ports reserved by L4 for the Compose stack."""

QUICKSTART_CHECKS: tuple[str, ...] = ("python", "runtime_imports", "lightgbm")
"""Doctor-lite subset used by `mll quickstart` (the drill needs no Docker)."""


def format_table(checks: Sequence[CheckResult]) -> str:
    """Render the checks as a compact aligned table."""
    lines = [f"{'check'}  {'status'}  {'detail'}"]
    lines.append(f"{'-' * 5}  {'-' * 6}  {'-' * 6}")
    for check in checks:
        details = check.detail or ""
        lines.append(f"{check.name}  {check.status.value:<4} {details}".rstrip())
    return "\n".join(lines)


def hard_failures(checks: Sequence[CheckResult]) -> list[CheckResult]:
    """Checks that must fail the command."""
    return [check for check in checks if check.hard and check.status is CheckStatus.FAIL]


def collect_checks(
    settings: Settings,
    *,
    checks: Sequence[str] | None = None,
    include_docker: bool = True,
    include_ports: bool = True,
) -> list[CheckResult]:
    """Run the requested checks (all by default) and never raise on a failed probe."""
    registry: dict[str, Callable[[Settings], CheckResult]] = {
        "python": _check_python,
        "runtime_imports": _check_runtime_imports,
        "lightgbm": _check_lightgbm,
        "sqlite_wal": _check_sqlite_wal,
        "workspace_writable": _check_workspace_writable,
        "mlflow_sqlite": _check_mlflow_sqlite,
        "docker": _check_docker,
        "ports": _check_ports,
    }
    selected = (
        list(checks)
        if checks is not None
        else [
            "python",
            "runtime_imports",
            "lightgbm",
            "sqlite_wal",
            "workspace_writable",
            "mlflow_sqlite",
        ]
    )
    if include_docker and "docker" not in selected:
        selected.append("docker")
    if include_ports and "ports" not in selected:
        selected.append("ports")

    results: list[CheckResult] = []
    for name in selected:
        checker = registry.get(name)
        if checker is None:
            results.append(CheckResult(name, CheckStatus.FAIL, "unknown check", hard=True))
            continue
        try:
            results.append(checker(settings))
        except Exception as exc:  # a probe must never crash the report
            results.append(
                CheckResult(name, CheckStatus.FAIL, f"{type(exc).__name__}: {exc}", hard=True)
            )
    return results


def doctor(
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Run the environment checks and exit 1 when a hard check fails."""
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    checks = collect_checks(settings)
    typer.echo(format_table(checks))
    failures = hard_failures(checks)
    if failures:
        count = len(failures)
        noun = "check" if count == 1 else "checks"
        typer.echo(f"doctor: {count} {noun} failed", err=True)
        raise typer.Exit(code=1)
    typer.echo("doctor: OK")


# -- individual checks ------------------------------------------------------------
def _check_python(settings: Settings) -> CheckResult:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info[:2] != (3, 12):
        return CheckResult("python", CheckStatus.FAIL, f"Python 3.12 required, found {version}")
    return CheckResult("python", CheckStatus.OK, version)


def _check_runtime_imports(settings: Settings) -> CheckResult:
    failures: list[str] = []
    for module in RUNTIME_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # report every broken import
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    if failures:
        return CheckResult("runtime_imports", CheckStatus.FAIL, "; ".join(failures))
    return CheckResult("runtime_imports", CheckStatus.OK, f"{len(RUNTIME_MODULES)} packages import")


def _check_lightgbm(settings: Settings) -> CheckResult:
    import numpy as np
    from lightgbm import LGBMRegressor

    features = np.arange(20, dtype=float).reshape(10, 2)
    target = np.arange(10, dtype=float)
    started = time.perf_counter()
    LGBMRegressor(n_estimators=3, n_jobs=1, verbose=-1).fit(features, target)
    elapsed = time.perf_counter() - started
    return CheckResult("lightgbm", CheckStatus.OK, f"fit 10 rows in {elapsed:.2f}s")


def _check_sqlite_wal(settings: Settings) -> CheckResult:
    with tempfile.TemporaryDirectory(prefix="mll-doctor-") as directory:
        connection = sqlite3.connect(Path(directory) / "probe.db")
        try:
            mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0])
        finally:
            connection.close()
    if mode.lower() != "wal":
        return CheckResult("sqlite_wal", CheckStatus.FAIL, f"journal_mode={mode}")
    return CheckResult("sqlite_wal", CheckStatus.OK, "journal_mode=wal")


def _check_workspace_writable(settings: Settings) -> CheckResult:
    workspace = settings.workspace
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=workspace, prefix=".doctor-", delete=False) as handle:
            handle.write(b"ok")
            probe = Path(handle.name)
        probe.unlink()
    except OSError as exc:
        return CheckResult("workspace_writable", CheckStatus.FAIL, f"{workspace}: {exc}")
    return CheckResult("workspace_writable", CheckStatus.OK, str(workspace))


def _check_mlflow_sqlite(settings: Settings) -> CheckResult:
    uri = settings.effective_tracking_uri
    if not uri.startswith("sqlite:///"):
        return CheckResult(
            "mlflow_sqlite",
            CheckStatus.WARN,
            f"remote tracking URI {uri}; not probed",
            hard=False,
        )
    path = Path(uri.removeprefix("sqlite:///"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS _mll_doctor_probe (x INTEGER)")
            connection.execute("DROP TABLE _mll_doctor_probe")
            connection.commit()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return CheckResult("mlflow_sqlite", CheckStatus.FAIL, f"{path}: {exc}")
    return CheckResult("mlflow_sqlite", CheckStatus.OK, f"writable: {path}")


def _check_docker(settings: Settings) -> CheckResult:
    executable = shutil.which("docker")
    if executable is None:
        return CheckResult(
            "docker", CheckStatus.WARN, "docker not found (optional for the L1 path)", hard=False
        )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [executable, "info", "--format", "{{.OSType}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("docker", CheckStatus.WARN, f"unavailable: {exc}", hard=False)
    if completed.returncode != 0:
        detail = "daemon not reachable (optional for the L1 path)"
        return CheckResult("docker", CheckStatus.WARN, detail, hard=False)
    engine = completed.stdout.strip() or "unknown"
    return CheckResult("docker", CheckStatus.WARN, f"present, engine={engine}", hard=False)


def _check_ports(settings: Settings) -> CheckResult:
    ports = (
        ("mlflow", MLFLOW_PORT),
        ("api", settings.api_port),
        ("dashboard", DASHBOARD_PORT),
    )
    busy = [f"{port} ({name})" for name, port in ports if not _port_free(port)]
    if busy:
        return CheckResult("ports", CheckStatus.WARN, f"busy: {', '.join(busy)}", hard=False)
    rendered = ", ".join(str(port) for _name, port in ports)
    return CheckResult("ports", CheckStatus.OK, f"free: {rendered}", hard=False)


def _port_free(port: int) -> bool:
    """Whether a loopback TCP port can be bound right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


__all__ = [
    "QUICKSTART_CHECKS",
    "CheckResult",
    "CheckStatus",
    "collect_checks",
    "doctor",
    "format_table",
    "hard_failures",
]
