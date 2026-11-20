"""Shared pytest configuration for the mlops-lifecycle-lab test suite."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def clean_mll_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never inherit MLL_* settings from the developer shell."""
    for key in list(os.environ):
        if key.startswith("MLL_"):
            monkeypatch.delenv(key, raising=False)
