"""Lifecycle: the audit-event timeline and the run ledger (S17, S14)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mlops_lifecycle_lab.bootstrap.container import Container
from mlops_lifecycle_lab.domain.events import EventType
from mlops_lifecycle_lab.entrypoints.dashboard.theme import Theme
from mlops_lifecycle_lab.entrypoints.dashboard.viewmodels import events_table, ledger_table
from mlops_lifecycle_lab.entrypoints.dashboard.views._common import (
    events_frame,
    page_header,
    show_frame,
)


def _ledger_frame(container: Container) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run": row.name,
                "tick": row.tick,
                "status": row.status,
                "started": row.started_at.isoformat(sep=" ", timespec="seconds"),
                "duration (s)": round(row.duration_seconds, 2),
                "failed steps": row.failed_steps,
                "error": row.error or "",
            }
            for row in ledger_table(container)
        ]
    )


def render(container: Container, theme: Theme) -> None:
    """Render the Lifecycle page."""
    del theme
    page_header(
        "Lifecycle",
        "The append-only audit trail and the run ledger behind every tick.",
    )
    selected = st.multiselect(
        "Event types",
        options=[event_type.value for event_type in EventType],
        default=[],
        help="Leave empty to show every event type.",
    )
    types = [EventType(value) for value in selected] if selected else None
    st.subheader("Audit timeline")
    show_frame(events_frame(events_table(container, types=types, limit=100)))

    st.subheader("Run ledger")
    show_frame(_ledger_frame(container), empty_message="No pipeline runs have been recorded.")
