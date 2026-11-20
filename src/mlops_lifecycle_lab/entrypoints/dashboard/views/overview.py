"""Overview: the one-screen health of the lifecycle lab (S17)."""

from __future__ import annotations

import streamlit as st

from mlops_lifecycle_lab.bootstrap.container import Container
from mlops_lifecycle_lab.entrypoints.dashboard.theme import Theme, status_marks
from mlops_lifecycle_lab.entrypoints.dashboard.viewmodels import (
    STATUS_LABELS,
    overview,
)
from mlops_lifecycle_lab.entrypoints.dashboard.views._common import (
    events_frame,
    page_header,
    show_frame,
    status_banner,
)


@st.fragment(run_every="5s")
def _body(container: Container, theme: Theme) -> None:
    view = overview(container)
    marks = status_marks(theme)
    if view.status == "no_champion":
        st.info("No champion is loaded yet. Run `mll bootstrap` to train the first one.")
    else:
        status_banner(
            marks[view.status],
            f"Champion v{view.champion_version} · SLO WAPE {container.settings.slo_wape:.2f}",
        )
        st.caption(f"Status: {STATUS_LABELS[view.status]}")

    columns = st.columns(3)
    for index, tile in enumerate(view.kpis):
        with columns[index % 3]:
            st.metric(tile.label, tile.value, help=tile.caption)

    st.subheader("Last five events")
    show_frame(events_frame(view.recent_events), empty_message="No lifecycle events yet.")


def render(container: Container, theme: Theme) -> None:
    """Render the Overview page."""
    page_header(
        "Overview",
        "Champion health, the SLO, and what the lifecycle did most recently.",
    )
    _body(container, theme)
