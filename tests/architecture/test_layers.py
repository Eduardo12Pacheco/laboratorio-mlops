"""Architecture contracts: the import-linter rules are real and they bite."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from importlinter.cli import lint_imports

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_import_linter_contracts_are_kept(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = lint_imports(
        config_filename=str(PYPROJECT),
        no_cache=True,
        no_logo=True,
    )

    output = capsys.readouterr().out
    assert exit_code == 0, output
    assert "3 kept, 0 broken" in output


def test_planted_layer_violation_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package = tmp_path / "planted_violation"
    (package / "domain").mkdir(parents=True)
    (package / "adapters").mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "adapters" / "__init__.py").write_text("", encoding="utf-8")
    (package / "domain" / "__init__.py").write_text(
        "from planted_violation import adapters\n\n__all__ = ['adapters']\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [tool.importlinter]
            root_package = "planted_violation"

            [[tool.importlinter.contracts]]
            name = "Planted layers"
            type = "layers"
            layers = [
              "planted_violation.adapters",
              "planted_violation.domain",
            ]
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    exit_code = lint_imports(
        config_filename=str(tmp_path / "pyproject.toml"),
        no_cache=True,
        no_logo=True,
    )

    output = capsys.readouterr().out
    assert exit_code == 1, output
    assert "broken" in output.lower()
