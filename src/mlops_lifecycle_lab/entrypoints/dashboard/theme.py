"""Palette tokens, contrast maths and status marks for the Control Room (S17).

The dashboard never conveys status by colour alone: every status is an `(icon, label,
colour)` triple, and the tests hold each colour to the WCAG contrast floors (4.5:1 for
text, 3:1 for non-text marks). Light and dark palettes are explicit so the Plotly helpers
can use the same categorical colours regardless of the Streamlit theme.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

MIN_TEXT_CONTRAST = 4.5
MIN_MARK_CONTRAST = 3.0


def _channel(value: float) -> float:
    """Linearise one sRGB channel exactly as WCAG 2.x relative luminance requires."""
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    """Return the WCAG relative luminance of a ``#RRGGBB`` colour."""
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError(f"expected a #RRGGBB colour, got {color!r}")
    try:
        red, green, blue = (int(color[offset : offset + 2], 16) / 255.0 for offset in (1, 3, 5))
    except ValueError as exc:
        raise ValueError(f"expected a #RRGGBB colour, got {color!r}") from exc
    return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two colours (1.0 … 21.0)."""
    lighter = max(relative_luminance(foreground), relative_luminance(background))
    darker = min(relative_luminance(foreground), relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


@dataclass(frozen=True, slots=True)
class StatusMark:
    """A status value that stays readable without colour: icon + label + colour."""

    icon: str
    label: str
    color: str


STATUS_MARKS: Mapping[str, StatusMark] = MappingProxyType(
    {
        "healthy": StatusMark("✔", "Healthy", "#047857"),
        "degraded": StatusMark("▲", "Degraded", "#B91C1C"),
        "probation": StatusMark("◔", "Probation", "#B45309"),
        "rejected": StatusMark("✖", "Rejected", "#7F1D1D"),
        "rolled_back": StatusMark("↩", "Rolled back", "#6D28D9"),
    }
)
"""The canonical status vocabulary; colours are overridden per theme by `status_marks`."""

_OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")
"""Colour-blind-safe categorical palette (versions, series)."""


@dataclass(frozen=True, slots=True)
class Theme:
    """One dashboard palette plus the text pairs the contrast test checks."""

    name: str
    background: str
    surface: str
    text: str
    muted: str
    accent: str
    border: str
    grid: str
    categorical: tuple[str, ...] = _OKABE_ITO
    status: Mapping[str, str] | None = None

    def text_pairs(self) -> tuple[tuple[str, str], ...]:
        """Every foreground/background pair used for readable text."""
        return (
            (self.text, self.background),
            (self.text, self.surface),
            (self.muted, self.background),
            (self.muted, self.surface),
            (self.accent, self.background),
            (self.accent, self.surface),
        )


LIGHT = Theme(
    name="light",
    background="#FFFFFF",
    surface="#F2F4F7",
    text="#111827",
    muted="#4B5563",
    accent="#0B5394",
    border="#D0D5DD",
    grid="#E4E7EC",
    status=MappingProxyType(
        {
            "healthy": "#047857",
            "degraded": "#B91C1C",
            "probation": "#B45309",
            "rejected": "#7F1D1D",
            "rolled_back": "#6D28D9",
        }
    ),
)
"""Light palette; `st.context.theme.type == "light"`."""

DARK = Theme(
    name="dark",
    background="#0E1117",
    surface="#161B22",
    text="#F0F6FC",
    muted="#C9D1D9",
    accent="#79C0FF",
    border="#30363D",
    grid="#21262D",
    status=MappingProxyType(
        {
            "healthy": "#3FB950",
            "degraded": "#FF7B72",
            "probation": "#E3B341",
            "rejected": "#FFA198",
            "rolled_back": "#D2A8FF",
        }
    ),
)
"""Dark palette (Streamlit's dark background)."""

THEMES: Mapping[str, Theme] = MappingProxyType({"light": LIGHT, "dark": DARK})


def theme_for(name: str | None) -> Theme:
    """Resolve Streamlit's ``st.context.theme.type`` value to a palette."""
    return THEMES.get((name or "light").lower(), LIGHT)


def status_marks(theme: Theme) -> Mapping[str, StatusMark]:
    """Status marks coloured for `theme`, keeping the canonical icons and labels."""
    colors = theme.status or {}
    return MappingProxyType(
        {
            key: StatusMark(mark.icon, mark.label, colors.get(key, mark.color))
            for key, mark in STATUS_MARKS.items()
        }
    )


def plotly_layout(theme: Theme, *, height: int = 320) -> dict[str, object]:
    """A transparent Plotly layout that inherits Streamlit's theme typography."""
    return {
        "height": height,
        "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": theme.text},
        "xaxis": {"gridcolor": theme.grid, "zeroline": False},
        "yaxis": {"gridcolor": theme.grid, "zeroline": False},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    }


__all__ = [
    "DARK",
    "LIGHT",
    "MIN_MARK_CONTRAST",
    "MIN_TEXT_CONTRAST",
    "STATUS_MARKS",
    "THEMES",
    "StatusMark",
    "Theme",
    "contrast_ratio",
    "plotly_layout",
    "relative_luminance",
    "status_marks",
    "theme_for",
]
