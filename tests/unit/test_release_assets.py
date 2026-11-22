"""Release assets (S24, T15): the README pair, screenshots and hero GIF are product.

The checks are dependency-free so the default suite keeps running without the optional
``capture`` group: PNG and GIF dimensions are read from their headers and the size
budgets are plain file sizes. Pixel-level animation checks live in the ``ui`` suite.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = ROOT / "docs" / "assets" / "screenshots"
HERO = ROOT / "docs" / "assets" / "hero.gif"
PAGES = ("overview", "monitoring", "registry", "lifecycle", "drill")
SCHEMES = ("light", "dark")
README_LINE_BUDGET = 150
PNG_SIZE_BUDGET = 400_000
GIF_SIZE_BUDGET = 3_000_000
GIF_MAX_SIZE = (1280, 800)

RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"


def _png_dimensions(data: bytes) -> tuple[int, int]:
    assert data.startswith(b"\x89PNG\r\n\x1a\n"), "not a PNG file"
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _gif_dimensions(data: bytes) -> tuple[int, int]:
    assert data.startswith(b"GIF8"), "not a GIF file"
    return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")


def _readmes() -> tuple[Path, Path]:
    return ROOT / "README.md", ROOT / "README.en.md"


def test_readmes_exist_and_stay_within_the_line_budget() -> None:
    for path in _readmes():
        assert path.is_file(), f"missing {path.name}"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= README_LINE_BUDGET, f"{path.name} has {len(lines)} lines"


def test_readmes_hold_the_generated_results_block_and_quickstart() -> None:
    for path in _readmes():
        text = path.read_text(encoding="utf-8")
        assert RESULTS_START in text, f"{path.name}: missing results:start"
        assert RESULTS_END in text, f"{path.name}: missing results:end"
        block = text.split(RESULTS_START, 1)[1].split(RESULTS_END, 1)[0]
        assert "As-served RMSE" in block
        assert "install.sh" in text or "install.ps1" in text
        assert "docs/assets/hero.gif" in text
        assert "MIT" in text
        assert "CC BY 4.0" in text


def test_english_and_spanish_readmes_are_mirrors() -> None:
    spanish, english = _readmes()
    assert "**Español** · [English](README.en.md)" in spanish.read_text(encoding="utf-8")
    assert "[Español](README.md) · **English**" in english.read_text(encoding="utf-8")


def test_every_page_screenshot_is_captured_in_both_themes() -> None:
    for scheme in SCHEMES:
        for page in PAGES:
            path = SCREENSHOTS / f"{page}-{scheme}.png"
            assert path.is_file(), f"missing screenshot {path.relative_to(ROOT)}"
            data = path.read_bytes()
            assert _png_dimensions(data) == (1440, 900), path.name
            assert len(data) <= PNG_SIZE_BUDGET, f"{path.name}: {len(data)} bytes"


def test_hero_gif_is_small_enough_and_uses_the_documented_canvas() -> None:
    assert HERO.is_file(), "missing docs/assets/hero.gif"
    data = HERO.read_bytes()
    width, height = _gif_dimensions(data)
    assert width <= GIF_MAX_SIZE[0], (width, height)
    assert height <= GIF_MAX_SIZE[1], (width, height)
    assert len(data) <= GIF_SIZE_BUDGET, f"hero.gif: {len(data)} bytes"


def test_package_version_is_one_point_zero() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["version"] == "1.0.0"
