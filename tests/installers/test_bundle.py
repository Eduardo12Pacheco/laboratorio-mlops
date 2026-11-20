"""Compose bundle build contract (release asset, S23/T13).

The bundle is the source subset a user can download and run without cloning the
repository: installer scripts, Compose stack, locked Python project and package source.
Tests and docs are deliberately excluded, and the archive is reproducible.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_compose_bundle.py"

REQUIRED = {
    "Dockerfile",
    "compose.yaml",
    "install.sh",
    "install.ps1",
    "pyproject.toml",
    "uv.lock",
    ".env.example",
    "LICENSE",
    "README.md",
}
EXCLUDED_PREFIXES = ("tests/", "docs/", ".github/", ".venv/", ".git/")


def _build(out: Path) -> Path:
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(SCRIPT), "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    archives = sorted(out.glob("*-compose.zip"))
    assert len(archives) == 1, result.stdout + result.stderr
    return archives[0]


def test_bundle_contains_the_runnable_source_subset(tmp_path: Path) -> None:
    archive = _build(tmp_path)

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()

    assert set(names) >= REQUIRED
    assert ".streamlit/config.toml" in names
    assert "src/mlops_lifecycle_lab/entrypoints/cli/app.py" in names
    assert "src/mlops_lifecycle_lab/_data/hour.csv.gz" in names
    for prefix in EXCLUDED_PREFIXES:
        assert not any(name.startswith(prefix) for name in names), prefix
    assert not any("__pycache__" in name for name in names)
    assert not any(name.endswith(".pyc") for name in names)


def test_installers_are_executable_inside_the_archive(tmp_path: Path) -> None:
    archive = _build(tmp_path)

    with zipfile.ZipFile(archive) as bundle:
        for name in ("install.sh", "install.ps1"):
            info = bundle.getinfo(name)
            mode = info.external_attr >> 16
            assert mode & 0o111, f"{name} is not executable (mode {oct(mode)})"


def test_bundle_is_reproducible(tmp_path: Path) -> None:
    first = _build(tmp_path / "one")
    second = _build(tmp_path / "two")

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    assert digest(first) == digest(second)
    assert first.name == second.name
