"""T11: the real Streamlit app renders all five pages in light and dark (marker ui)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Protocol

import pytest

from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine

pytestmark = pytest.mark.ui

PORT = 14511
BASE_URL = f"http://127.0.0.1:{PORT}"
PAGES = ("Overview", "Monitoring", "Registry", "Lifecycle", "Drill")


class _ConsoleMessage(Protocol):
    type: str
    text: str


def _record_console(message: _ConsoleMessage, target: list[str]) -> None:
    # A browser can report transient resource-load failures (for example when the host's
    # network interface changes). They are not application exceptions; JavaScript errors
    # and any other console error still fail the test.
    if message.type == "error" and not message.text.startswith("Failed to load resource"):
        target.append(message.text)


def _record_pageerror(error: object, target: list[str]) -> None:
    target.append(str(error))


def _wait_until_healthy(timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed loopback http URL
                f"{BASE_URL}/_stcore/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    raise AssertionError(f"dashboard did not become healthy at {BASE_URL}")


@pytest.fixture(scope="module")
def drilled_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dashboard-ui")
    workspace = root / "ws"
    DrillEngine().run("drift", weeks=16, workspace=workspace, fast=True)
    return workspace


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory: pytest.TempPathFactory, drilled_workspace: Path) -> Iterator[str]:
    executable = shutil.which("mll")
    if executable is None:
        pytest.fail("mll console script is not on PATH")
    environment = os.environ.copy()
    environment["MLL_WORKSPACE"] = str(drilled_workspace)
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [executable, "dashboard", "--port", str(PORT), "--workspace", str(drilled_workspace)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_until_healthy()
        yield BASE_URL
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def test_every_page_renders_in_light_and_dark(dashboard: str, tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    backgrounds: dict[str, str] = {}
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for scheme in ("light", "dark"):
                context = browser.new_context(
                    color_scheme=scheme, viewport={"width": 1440, "height": 900}
                )
                page = context.new_page()
                errors: list[str] = []
                page.on("console", partial(_record_console, target=errors))
                page.on("pageerror", partial(_record_pageerror, target=errors))
                page.goto(dashboard, wait_until="load")
                page.get_by_role("heading", name="Overview", exact=True).wait_for(timeout=60000)
                backgrounds[scheme] = page.evaluate(
                    "() => getComputedStyle(document.body).backgroundColor"
                )
                for name in PAGES:
                    if name != "Overview":
                        page.locator('[data-testid="stSidebar"] label').filter(
                            has_text=name
                        ).first.click()
                    page.get_by_role("heading", name=name, exact=True).wait_for(timeout=60000)
                    page.wait_for_timeout(750)
                    page.screenshot(
                        path=str(tmp_path / f"{name.lower()}-{scheme}.png"), full_page=True
                    )
                    assert not errors, (scheme, name, errors)
                context.close()
        finally:
            browser.close()
    assert backgrounds["light"] != backgrounds["dark"]
    assert len(list(tmp_path.glob("*.png"))) == len(PAGES) * 2
