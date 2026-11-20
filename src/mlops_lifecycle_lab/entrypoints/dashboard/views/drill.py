"""Drill: scrub a committed replay report tick by tick (S17, S15)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from mlops_lifecycle_lab.bootstrap.container import Container
from mlops_lifecycle_lab.entrypoints.dashboard.theme import Theme, plotly_layout
from mlops_lifecycle_lab.entrypoints.dashboard.viewmodels import (
    DrillReport,
    ViewError,
    drill_reports,
    drill_state_at,
)
from mlops_lifecycle_lab.entrypoints.dashboard.views._common import (
    page_header,
    show_frame,
)

REPOSITORY_REPORTS = Path(__file__).resolve().parents[5] / "docs" / "results"
"""Curated reports committed by T14, when running from a clone."""


def _available_reports(container: Container) -> tuple[DrillReport, ...]:
    reports: list[DrillReport] = []
    for directory in (container.settings.workspace / "reports", REPOSITORY_REPORTS):
        for report in drill_reports(directory):
            if all(existing.path != report.path for existing in reports):
                reports.append(report)
    return tuple(reports)


def _comparison_chart(report: DrillReport, theme: Theme) -> go.Figure:
    """RMSE and WAPE on separate panels: their scales differ by orders of magnitude."""
    comparison = report.comparison
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("RMSE (lower is better)", "WAPE (lower is better)")
    )
    series = (
        ("as served", comparison.as_served_rmse, comparison.as_served_wape, theme.categorical[0]),
        ("frozen v1", comparison.frozen_rmse, comparison.frozen_wape, theme.categorical[3]),
    )
    for name, rmse, wape, color in series:
        figure.add_trace(
            go.Bar(name=name, x=[name], y=[rmse], marker_color=color, legendgroup=name),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                name=name,
                x=[name],
                y=[wape],
                marker_color=color,
                legendgroup=name,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    figure.update_layout(
        # Extra top margin: room for the title above the two subplot titles.
        **plotly_layout(theme, height=320, top=96),
        title=f"As served vs frozen v1 · ratio {comparison.ratio:.2f}x",
    )
    figure.update_xaxes(gridcolor=theme.grid, zeroline=False)
    figure.update_yaxes(gridcolor=theme.grid, zeroline=False)
    figure.update_annotations(font_color=theme.text)
    return figure


def render(container: Container, theme: Theme) -> None:
    """Render the Drill page."""
    page_header(
        "Drill",
        "Scrub a reproducible replay report to see what the automation decided, tick by tick.",
    )
    reports = _available_reports(container)
    if not reports:
        st.info(
            "No drill reports were found. Run "
            "`mll drill run --scenario drift --weeks 16 --workspace <dir>` first."
        )
        return

    selected = st.selectbox(
        "Report",
        options=list(range(len(reports))),
        format_func=lambda index: (
            f"{reports[index].scenario} · {reports[index].weeks} weeks · {reports[index].path.name}"
        ),
    )
    report = reports[selected]
    ticks = [tick.tick for tick in report.ticks]
    tick = st.slider("Tick", min_value=min(ticks), max_value=max(ticks), value=min(ticks))
    try:
        state = drill_state_at(report, tick)
    except ViewError as exc:
        st.info(f"{exc.message}\n\n{exc.hint}")
        return

    columns = st.columns(4)
    columns[0].metric(
        "Champion", "none" if state.champion_version is None else f"v{state.champion_version}"
    )
    columns[1].metric("WAPE 28d", "n/a" if state.wape is None else f"{state.wape:.4f}")
    columns[2].metric("RMSE 28d", "n/a" if state.rmse is None else f"{state.rmse:.2f}")
    columns[3].metric("Numeric drift share", f"{state.numeric_drift_share:.2f}")
    st.caption(
        f"tick {state.tick} ends {state.tick_end.date().isoformat()} · "
        f"validation {state.validation_verdict} · action {state.action}"
    )

    st.plotly_chart(_comparison_chart(report, theme), theme="streamlit", use_container_width=True)
    st.subheader("Events at this tick")
    if state.events:
        show_frame(
            pd.DataFrame(
                [
                    {
                        "when": event.ts.isoformat(sep=" ", timespec="minutes"),
                        "type": event.type,
                        "tick": event.tick,
                        "version": event.model_version,
                    }
                    for event in state.events
                ]
            )
        )
    else:
        st.info("No lifecycle events at this tick.")

    st.caption("Reproduce this report:")
    st.code(f"mll drill run --scenario {report.scenario} --weeks {report.weeks} --workspace <dir>")
