"""Exit-code contract for the L2/L1 installer (`install.sh`, S20).

The script is driven through a controlled ``PATH`` containing shell shims for ``docker``
and ``uv`` plus symlinks to the host utilities it needs. That keeps the matrix hermetic:
the real Docker daemon, the real ports and the real ``uv`` are never consulted.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "install.sh"

# Utilities the installer may call. Everything else must be a shell builtin.
BASE_COMMANDS = (
    "awk",
    "bash",
    "cat",
    "chmod",
    "cp",
    "cut",
    "df",
    "dirname",
    "env",
    "grep",
    "head",
    "ls",
    "mkdir",
    "mv",
    "printf",
    "rm",
    "sed",
    "sleep",
    "sort",
    "touch",
    "tr",
    "uname",
)

PORTS = (14000, 14080, 14501)


def _tool_dir(root: Path) -> Path:
    """A PATH directory with symlinks to the host utilities the installer needs."""
    tool_dir = root / "bin"
    tool_dir.mkdir()
    for name in BASE_COMMANDS:
        resolved = shutil.which(name)
        if resolved is not None:
            (tool_dir / name).symlink_to(resolved)
    bash = shutil.which("bash")
    assert bash is not None, "bash is required to run the installer tests"
    (tool_dir / "bash").unlink(missing_ok=True)
    (tool_dir / "bash").symlink_to(bash)
    return tool_dir


def _write_shim(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _docker_shim(
    *,
    docker_version: str = "29.0.0",
    compose_version: str = "v2.40.3",
    engine_os: str = "linux",
    project_running: bool = True,
) -> str:
    ps_body = "echo 0123456789ab" if project_running else "exit 0"
    return f"""#!/usr/bin/env bash
case "${{1:-}}" in
  --version)
    echo "Docker version {docker_version}, build test"
    ;;
  compose)
    shift
    if [ "${{1:-}}" = "-p" ]; then
      shift 2
    fi
    case "${{1:-}}" in
      version)
        echo "Docker Compose version {compose_version}"
        ;;
      ps)
        {ps_body}
        ;;
      *)
        exit 0
        ;;
    esac
    ;;
  info)
    if [ "${{2:-}}" = "--format" ]; then
      echo "{engine_os}"
    else
      echo "Server reachable"
    fi
    ;;
  *)
    exit 0
    ;;
esac
"""


def _uv_shim() -> str:
    return """#!/usr/bin/env bash
case "$1 $2" in
  "python find")
    echo "/usr/local/bin/python3.12"
    ;;
  "run "*)
    echo "lightgbm  OK"
    ;;
  *)
    exit 0
    ;;
esac
"""


def _run(
    tmp_path: Path,
    args: list[str],
    *,
    docker: str | None = None,
    uv: bool = False,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    tool_dir = _tool_dir(tmp_path)
    shims = tmp_path / "shims"
    shims.mkdir()
    if docker is not None:
        _write_shim(shims, "docker", docker)
    if uv:
        _write_shim(shims, "uv", _uv_shim())

    env = {key: value for key, value in os.environ.items() if not key.startswith("MLL_")}
    env.pop("COMPOSE_PROJECT_NAME", None)
    env["PATH"] = f"{shims}:{tool_dir}"
    env["HOME"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir(exist_ok=True)

    bash = shutil.which("bash")
    assert bash is not None
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [bash, str(INSTALL), *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=60,
        check=False,
    )


def _combined(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout + result.stderr).lower()


def test_missing_docker_fails_with_exit_code_1(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--check"])

    assert result.returncode == 1, _combined(result)
    assert "docker not found" in _combined(result)


def test_compose_v1_fails_with_exit_code_1(tmp_path: Path) -> None:
    docker = _docker_shim(compose_version="1.29.2")

    result = _run(tmp_path, ["--check"], docker=docker)

    assert result.returncode == 1, _combined(result)
    assert "compose v2" in _combined(result)


def test_windows_engine_fails_with_exit_code_1(tmp_path: Path) -> None:
    docker = _docker_shim(engine_os="windows")

    result = _run(tmp_path, ["--check"], docker=docker)

    assert result.returncode == 1, _combined(result)
    assert "linux containers" in _combined(result)


def test_busy_port_fails_and_names_the_port(tmp_path: Path) -> None:
    docker = _docker_shim(project_running=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", PORTS[0]))
        listener.listen(1)

        result = _run(tmp_path, ["--check"], docker=docker)

    assert result.returncode == 1, _combined(result)
    assert str(PORTS[0]) in _combined(result)


def test_check_passes_when_prerequisites_are_met(tmp_path: Path) -> None:
    docker = _docker_shim(project_running=False)

    result = _run(tmp_path, ["--check"], docker=docker)

    assert result.returncode == 0, _combined(result)
    assert "[ ok ]" in _combined(result)


def test_project_owned_ports_are_not_reported_as_busy(tmp_path: Path) -> None:
    docker = _docker_shim(project_running=True)

    result = _run(tmp_path, ["--check"], docker=docker)

    assert result.returncode == 0, _combined(result)
    assert "owned by compose project" in _combined(result)


def test_l1_check_skips_docker_and_runs_doctor(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--check", "--l1"], uv=True)

    assert result.returncode == 0, _combined(result)
    assert "uv" in _combined(result)
    assert "docker not found" not in _combined(result)


def test_l1_check_without_uv_fails(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--check", "--l1"])

    assert result.returncode == 1, _combined(result)
    assert "uv not found" in _combined(result)


def test_unknown_flag_is_a_usage_error(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--not-a-real-flag"])

    assert result.returncode == 2, _combined(result)


def test_uninstall_without_confirmation_exits_4(tmp_path: Path) -> None:
    docker = _docker_shim()

    result = _run(tmp_path, ["--uninstall"], docker=docker, stdin="n\n")

    assert result.returncode == 4, _combined(result)
    assert "aborted" in _combined(result)


def test_uninstall_yes_tears_the_project_down(tmp_path: Path) -> None:
    docker = _docker_shim()

    result = _run(tmp_path, ["--uninstall", "--yes"], docker=docker)

    assert result.returncode == 0, _combined(result)
    assert "uninstalled" in _combined(result)


def test_help_documents_every_flag(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--help"])

    assert result.returncode == 0, _combined(result)
    for flag in ("--check", "--l1", "--uninstall", "--yes", "--no-start", "--project-name"):
        assert flag in _combined(result)
