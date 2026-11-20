"""Loading the CLI must be quiet, side-effect free and telemetry-safe (S18, S20).

Each probe runs in a fresh interpreter with the environment of a plain user shell (no CI or
pytest markers, which MLflow treats as "telemetry off"), so it observes what a human
running ``mll`` would get on first import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_KEPT_ENVIRONMENT = (
    "PATH",
    "SYSTEMROOT",
    "HOME",
    "USERPROFILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
)

_PROBE = """
import json
import sys

import mlops_lifecycle_lab.entrypoints.cli.app  # noqa: F401
import mlflow.telemetry as telemetry

print(json.dumps({
    "streamlit_loaded": "streamlit" in sys.modules,
    "telemetry_client": telemetry.get_telemetry_client() is not None,
}))
"""


def _plain_user_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key in _KEPT_ENVIRONMENT}


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    cwd = tmp_path_factory.mktemp("import-probe")
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-I", "-c", _PROBE],
        cwd=cwd,
        env=_plain_user_environment(),
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    result: dict[str, object] = json.loads(completed.stdout.strip().splitlines()[-1])
    result["workspace_created"] = (cwd / ".mll").exists()
    return result


def test_importing_the_cli_does_not_execute_the_dashboard(probe: dict[str, object]) -> None:
    assert probe["streamlit_loaded"] is False
    assert probe["workspace_created"] is False


def test_importing_the_cli_disables_mlflow_telemetry_before_mlflow_loads(
    probe: dict[str, object],
) -> None:
    assert probe["telemetry_client"] is False
