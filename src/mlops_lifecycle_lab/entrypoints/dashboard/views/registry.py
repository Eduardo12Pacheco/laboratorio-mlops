"""Registry: versions, aliases and the evidence behind every promotion (S17, S7)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mlops_lifecycle_lab.bootstrap.container import Container
from mlops_lifecycle_lab.entrypoints.dashboard.theme import Theme
from mlops_lifecycle_lab.entrypoints.dashboard.viewmodels import (
    ViewError,
    gate_report,
    registry_table,
)
from mlops_lifecycle_lab.entrypoints.dashboard.views._common import page_header, show_frame


def _versions_frame(container: Container) -> tuple[pd.DataFrame, list[int], int]:
    view = registry_table(container)
    frame = pd.DataFrame(
        [
            {
                "version": row.version,
                "stage": row.stage,
                "aliases": ", ".join(row.aliases) or "-",
                "rmse": None if row.rmse is None else round(row.rmse, 2),
                "wape": None if row.wape is None else round(row.wape, 4),
                "train window": row.train_window,
                "fingerprint": row.fingerprint,
                "created": row.created.isoformat(sep=" ", timespec="minutes"),
            }
            for row in view.rows
        ]
    )
    versions = [row.version for row in view.rows]
    champion = next((row.version for row in view.rows if "champion" in row.aliases), versions[-1])
    return frame, versions, champion


def _gate_frame(container: Container, version: int) -> tuple[pd.DataFrame, str]:
    report = gate_report(container, version)
    frame = pd.DataFrame(
        [
            {
                "gate": gate.name,
                "result": "✔ pass" if gate.passed else "✖ fail",
                "observed": gate.observed,
                "threshold": gate.threshold,
                "detail": gate.detail,
            }
            for gate in report.gates
        ]
    )
    return frame, report.decision


def _lineage(container: Container, version: int) -> None:
    view = registry_table(container)
    row = next((candidate for candidate in view.rows if candidate.version == version), None)
    if row is None:
        return
    with st.expander("Lineage and tags"):
        st.markdown(
            "\n".join(
                [
                    f"- run: `{row.run_id}`",
                    f"- data fingerprint: `{row.fingerprint}` (first 8 characters)",
                    f"- training window: {row.train_window}",
                    "- stage source: the `mll.stage` tag",
                    "- aliases: champion / previous-champion / challenger",
                ]
            )
        )


def render(container: Container, theme: Theme) -> None:
    """Render the Registry page."""
    del theme  # the registry page uses Streamlit's native table colours
    page_header(
        "Registry",
        "Every version, its aliases, and the gates that decided its fate.",
    )
    frame, versions, default_version = _versions_frame(container)
    show_frame(frame, empty_message="No registered versions yet. Run `mll bootstrap`.")
    st.link_button("Open the MLflow UI", "http://127.0.0.1:14000")

    selected = st.selectbox(
        "Version",
        options=versions,
        index=versions.index(default_version),
        format_func=lambda version: f"v{version}",
    )
    try:
        gates, decision = _gate_frame(container, selected)
    except ViewError as exc:
        st.info(f"{exc.message}\n\n{exc.hint}")
    else:
        st.subheader(f"Gate report · v{selected} · {decision}")
        show_frame(gates)
        _lineage(container, selected)
