"""Streamlit Control Room entry point (S17).

The script is intentionally thin: it resolves the workspace from `MLL_WORKSPACE`, builds
the same composition root the CLI and API use, and dispatches to one view module per page.
All data shaping lives in `viewmodels.py`, so the pages stay renderable and testable.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path

import streamlit as st

from mlops_lifecycle_lab.bootstrap.container import Container, build_container
from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.entrypoints.dashboard.theme import Theme, theme_for
from mlops_lifecycle_lab.entrypoints.dashboard.viewmodels import ViewError, require_workspace
from mlops_lifecycle_lab.entrypoints.dashboard.views import (
    drill,
    lifecycle,
    monitoring,
    overview,
    registry,
)

Page = Callable[[Container, Theme], None]

PAGES: dict[str, Page] = {
    "Overview": overview.render,
    "Monitoring": monitoring.render,
    "Registry": registry.render,
    "Lifecycle": lifecycle.render,
    "Drill": drill.render,
}


@st.cache_resource(show_spinner=False)
def container_for(workspace: str, fast: bool) -> Container:
    """Build (and cache across reruns) the composition root for one workspace."""
    return build_container(Settings(workspace=Path(workspace), fast=fast))


def current_theme() -> Theme:
    """Resolve the active Streamlit theme, defaulting to light in bare test runs."""
    try:
        return theme_for(st.context.theme.type)
    except (AttributeError, KeyError):
        return theme_for("light")


def main() -> None:
    """Render the selected page and map failures to the three documented UI states."""
    apply_privacy_defaults()
    st.set_page_config(
        page_title="MLOps Control Room",
        page_icon="🚲",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    settings = Settings()
    theme = current_theme()
    st.sidebar.header("MLOps Control Room")
    st.sidebar.caption(f"workspace: `{settings.workspace}`")
    page = st.sidebar.radio("Page", list(PAGES), key="control-room-page")

    try:
        container = container_for(str(settings.workspace), settings.fast)
        require_workspace(container)
        PAGES[page](container, theme)
    except ViewError as exc:
        message = exc.message if not exc.hint else f"{exc.message}\n\n{exc.hint}"
        st.info(message)
    except Exception as exc:  # a broken workspace must never crash the server
        st.error(f"Could not load the {page} page: {exc}")
        with st.expander("Technical detail"):
            st.code(traceback.format_exc())


main()
