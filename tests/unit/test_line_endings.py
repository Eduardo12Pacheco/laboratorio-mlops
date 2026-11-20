"""Shell scripts must stay LF on every checkout: bash rejects CRLF (S20, S25).

Git for Windows checks text files out with CRLF unless `.gitattributes` says otherwise, which
turns `install.sh` and `scripts/*.sh` into scripts that fail with `$'\\r': command not found`.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _shell_scripts() -> list[Path]:
    return [ROOT / "install.sh", *sorted((ROOT / "scripts").glob("*.sh"))]


def test_gitattributes_pins_shell_scripts_and_the_dockerfile_to_lf() -> None:
    rules = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "*.sh text eol=lf" in rules
    assert "Dockerfile text eol=lf" in rules


def test_shell_scripts_are_checked_out_with_lf_endings() -> None:
    scripts = _shell_scripts()
    assert len(scripts) >= 2
    for script in scripts:
        assert b"\r\n" not in script.read_bytes(), f"{script.name} has CRLF line endings"
