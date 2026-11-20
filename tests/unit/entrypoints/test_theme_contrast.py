"""T11: the dashboard palette must stay readable and never rely on colour alone."""

from __future__ import annotations

import pytest

from mlops_lifecycle_lab.entrypoints.dashboard.theme import (
    DARK,
    LIGHT,
    STATUS_MARKS,
    contrast_ratio,
    status_marks,
)

EXPECTED_MARKS = {
    "healthy": ("✔", "Healthy"),
    "degraded": ("▲", "Degraded"),
    "probation": ("◔", "Probation"),
    "rejected": ("✖", "Rejected"),
    "rolled_back": ("↩", "Rolled back"),
}


def test_contrast_ratio_matches_the_wcag_reference_values() -> None:
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast_ratio("#FFFFFF", "#FFFFFF") == pytest.approx(1.0)


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_every_text_pair_meets_wcag_aa(theme: object) -> None:
    text_pairs = theme.text_pairs()  # type: ignore[attr-defined]
    assert text_pairs
    for foreground, background in text_pairs:
        assert contrast_ratio(foreground, background) >= 4.5, (foreground, background)


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_every_status_mark_meets_the_non_text_contrast_floor(theme: object) -> None:
    marks = status_marks(theme)  # type: ignore[arg-type]
    assert set(marks) == set(EXPECTED_MARKS)
    for key, (icon, label) in EXPECTED_MARKS.items():
        mark = marks[key]
        assert (mark.icon, mark.label) == (icon, label)
        assert contrast_ratio(mark.color, theme.background) >= 3.0, key  # type: ignore[attr-defined]


def test_status_marks_are_shared_between_themes_except_for_colour() -> None:
    assert set(STATUS_MARKS) == set(EXPECTED_MARKS)
    for key, mark in STATUS_MARKS.items():
        assert (mark.icon, mark.label) == EXPECTED_MARKS[key]


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=["light", "dark"])
def test_categorical_palette_has_enough_distinct_colours(theme: object) -> None:
    palette = theme.categorical  # type: ignore[attr-defined]
    assert len(palette) >= 6
    assert len(set(palette)) == len(palette)
