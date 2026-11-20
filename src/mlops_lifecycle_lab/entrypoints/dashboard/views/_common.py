"""Small presentation helpers shared by the five Control Room pages (S17)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from mlops_lifecycle_lab.entrypoints.dashboard.theme import StatusMark


def page_header(title: str, purpose: str) -> None:
    """Render the page title and its one-sentence purpose."""
    st.title(title)
    st.caption(purpose)


def status_banner(mark: StatusMark, detail: str = "") -> None:
    """Render a status without relying on colour: the icon and label carry the meaning."""
    icon, label = mark.icon, mark.label
    st.markdown(f"### {icon} {label}")
    if detail:
        st.caption(detail)


def events_frame(rows: Sequence[object]) -> pd.DataFrame:
    """Turn event rows into the table shown by Overview and Lifecycle."""
    return pd.DataFrame(
        [
            {
                "when": row.ts.isoformat(sep=" ", timespec="minutes"),  # type: ignore[attr-defined]
                "type": row.type,  # type: ignore[attr-defined]
                "tick": row.tick,  # type: ignore[attr-defined]
                "version": row.model_version,  # type: ignore[attr-defined]
                "summary": row.summary,  # type: ignore[attr-defined]
            }
            for row in rows
        ]
    )


def show_frame(frame: pd.DataFrame, *, empty_message: str = "No rows.") -> None:
    """Render a dataframe with an explicit empty state."""
    if frame.empty:
        st.info(empty_message)
        return
    st.dataframe(frame, hide_index=True)


__all__ = ["events_frame", "page_header", "show_frame", "status_banner"]
