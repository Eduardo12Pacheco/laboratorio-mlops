"""Docker-marked wrapper around the real Compose smoke (T12, S19).

The default pytest run excludes the ``docker`` marker; CI and the executor run it on a
machine with a Linux Docker engine. The script itself owns project isolation (unique
project name), port checks, the champion assertion and unconditional teardown.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "smoke_compose.sh"


@pytest.mark.docker
def test_compose_stack_smoke() -> None:
    assert SMOKE.is_file(), "scripts/smoke_compose.sh is missing"
    bash = shutil.which("bash")
    if bash is None:
        pytest.fail("bash is not on PATH")

    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [bash, str(SMOKE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "SMOKE OK champion=2" in completed.stdout, output
