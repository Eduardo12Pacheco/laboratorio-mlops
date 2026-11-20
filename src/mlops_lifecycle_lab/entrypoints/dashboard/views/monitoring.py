"""Monitoring: label-based performance and covariate drift in one place (S17, S10)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mlops_lifecycle_lab.bootstrap.container import Container
from mlops_lifecycle_lab.entrypoints.dashboard.theme import Theme, plotly_layout
from mlops_lifecycle_lab.entrypoints.dashboard.viewmodels import (
    PerformanceView,
    drift_matrix,
    performance_series,
)
from mlops_lifecycle_lab.entrypoints.dashboard.views._common import page_header, show_frame


def _wape_chart(view: PerformanceView, theme: Theme) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[point.tick for point in view.points],
            y=[point.wape for point in view.points],
            mode="lines+markers",
            name="WAPE 28d",
            line={"color": theme.accent, "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point.tick for point in view.points],
            y=[view.slo for _ in view.points],
            mode="lines",
            name=f"SLO {view.slo:.2f}",
            line={"color": "#B91C1C", "width": 1, "dash": "dash"},
        )
    )
    for index, band in enumerate(view.bands):
        figure.add_vrect(
            x0=band.start_tick - 0.5,
            x1=band.end_tick + 0.5,
            fillcolor=theme.categorical[index % len(theme.categorical)],
            opacity=0.10,
            line_width=0,
            annotation_text=f"v{band.version}",
            annotation_position="top left",
        )
    figure.update_layout(**plotly_layout(theme, height=340), title="WAPE over the replay")
    return figure


def _actual_predicted_chart(view: PerformanceView, theme: Theme) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[row.ts for row in view.actual_vs_predicted],
            y=[row.actual for row in view.actual_vs_predicted],
            mode="lines",
            name="actual",
            line={"color": theme.text, "width": 1.5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[row.ts for row in view.actual_vs_predicted],
            y=[row.predicted for row in view.actual_vs_predicted],
            mode="lines",
            name="predicted",
            line={"color": theme.accent, "width": 1.5},
        )
    )
    figure.update_layout(**plotly_layout(theme, height=320), title="Last 14 days")
    return figure


def _drift_chart(container: Container, theme: Theme) -> go.Figure:
    view = drift_matrix(container)
    z = [[view.values[feature].get(tick) for tick in view.ticks] for feature in view.features]
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=list(view.ticks),
            y=list(view.features),
            colorscale=["#F0F9E8", "#BAE4BC", "#7BCCC4", "#43A2CA", "#0868AC"],
            zmin=0.0,
            zmax=0.5,
            colorbar={"title": "PSI"},
            hovertemplate="feature=%{y}<br>tick=%{x}<br>PSI=%{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        **plotly_layout(theme, height=320),
        title="Numeric and categorical PSI (0.1 moderate, 0.2 drifted)",
    )
    return figure


@st.fragment(run_every="5s")
def _body(container: Container, theme: Theme) -> None:
    view = performance_series(container)
    if not view.points:
        st.info("No monitoring snapshots yet. Run `mll quickstart` to replay a workspace.")
        return

    latest = view.points[-1]
    columns = st.columns(3)
    columns[0].metric(
        "WAPE 28d",
        "n/a" if latest.wape is None else f"{latest.wape:.4f}",
        help=f"SLO {view.slo:.2f}",
    )
    columns[1].metric("RMSE 28d", "n/a" if latest.rmse is None else f"{latest.rmse:.2f}")
    columns[2].metric("Bias", "n/a" if view.bias is None else f"{view.bias:+.2f}")

    st.plotly_chart(_wape_chart(view, theme), theme="streamlit", use_container_width=True)
    if view.actual_vs_predicted:
        st.plotly_chart(
            _actual_predicted_chart(view, theme), theme="streamlit", use_container_width=True
        )
    else:
        st.info("No labeled predictions in the trailing 14 days.")
    st.plotly_chart(_drift_chart(container, theme), theme="streamlit", use_container_width=True)

    st.subheader("Validation verdicts")
    show_frame(
        pd.DataFrame([{"tick": tick, "verdict": verdict} for tick, verdict in view.verdicts]),
        empty_message="No validation verdicts yet.",
    )


def render(container: Container, theme: Theme) -> None:
    """Render the Monitoring page."""
    page_header(
        "Monitoring",
        "Label-based performance against the SLO, separate from covariate drift.",
    )
    _body(container, theme)
