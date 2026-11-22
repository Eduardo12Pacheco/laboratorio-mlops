#!/usr/bin/env python3
"""Build the source-only Compose bundle shipped as a release asset (S23, D17).

The archive is the smallest tree that can run `./install.sh` from a fresh download:
the Compose stack, the installer scripts, the locked Python project and the package
source. Tests, docs and CI configuration are excluded. Entries are sorted and stamped
with a fixed timestamp, so two builds of the same version are byte-identical.
"""

from __future__ import annotations

import argparse
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_DATE_TIME = (2026, 1, 1, 0, 0, 0)
EXECUTABLE = {"install.sh", "install.ps1"}

TOP_LEVEL_FILES = (
    ".dockerignore",
    ".env.example",
    "Dockerfile",
    "LICENSE",
    "README.en.md",
    "README.md",
    "compose.yaml",
    "install.ps1",
    "install.sh",
    "pyproject.toml",
    "uv.lock",
)
EXTRA_DIRECTORIES = (".streamlit", "src")
EXCLUDED_PARTS = {"__pycache__", ".venv", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def project_version(root: Path = ROOT) -> str:
    """Return the version declared in `pyproject.toml`."""
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = document["project"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject.toml does not declare a valid project version")
    return version


def project_name(root: Path = ROOT) -> str:
    """Return the distribution name declared in `pyproject.toml`."""
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    name = document["project"]["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("pyproject.toml does not declare a valid project name")
    return name


def iter_bundle_files(root: Path = ROOT) -> list[Path]:
    """Resolve the bundle file list relative to `root`, sorted for reproducibility."""
    selected: list[Path] = []
    for relative in TOP_LEVEL_FILES:
        candidate = root / relative
        if candidate.is_file():
            selected.append(candidate)
    for directory in EXTRA_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            continue
        for candidate in base.rglob("*"):
            if not candidate.is_file():
                continue
            if any(part in EXCLUDED_PARTS for part in candidate.relative_to(root).parts):
                continue
            if candidate.suffix in EXCLUDED_SUFFIXES:
                continue
            selected.append(candidate)
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def _zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_DATE_TIME)
    info.create_system = 3  # Unix, so the executable bits survive extraction.
    info.external_attr = mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build_bundle(out_dir: Path, root: Path = ROOT) -> Path:
    """Write `<name>-<version>-compose.zip` into `out_dir` and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{project_name(root)}-{project_version(root)}-compose.zip"
    files = iter_bundle_files(root)
    if not files:
        raise ValueError(f"no bundle files found under {root}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            relative = path.relative_to(root).as_posix()
            mode = 0o755 if relative in EXECUTABLE else 0o644
            bundle.writestr(_zip_info(relative, mode), path.read_bytes())
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="directory for the bundle zip")
    args = parser.parse_args()
    archive = build_bundle(args.out)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
