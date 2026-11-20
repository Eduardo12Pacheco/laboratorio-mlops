#!/usr/bin/env python3
"""Capture the real Control Room screenshots and hero-frame sources (S24, T15).

The script is deliberately reproducible: it seeds a stable, gitignored workspace with
``mll drill run --scenario drift --weeks 52``, launches ``mll dashboard`` on a free
loopback port, and drives the real app with Playwright. It writes one 1440x900 PNG per
page and theme under ``docs/assets/screenshots`` and, optionally, the eight frames
``scripts/make_gif.py`` needs under ``--hero-output``.

Run it with the optional capture group (Pillow + Playwright)::

    uv run --group capture python scripts/capture_screenshots.py --hero-output /tmp/hero

The workspace lives at a stable path under the system temporary directory so the sidebar
caption, and therefore every committed pixel, is stable across runs without leaking an
owner-home path into the assets.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - imported only by type checkers
    from playwright.sync_api import Page

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "screenshots"
DEFAULT_WORKSPACE = Path(tempfile.gettempdir()) / "mlops-lifecycle-lab-screenshot-workspace"
PAGES = ("Overview", "Monitoring", "Registry", "Lifecycle", "Drill")
SCHEMES = ("light", "dark")
VIEWPORT = {"width": 1440, "height": 900}
PNG_BUDGET_BYTES = 400_000
HEALTH_TIMEOUT_SECONDS = 120.0
SETTLE_TIMEOUT_MS = 60_000
SPINNER = '[data-testid="stSpinner"]'


class _ConsoleMessage(Protocol):
    type: str
    text: str


def _record_pageerror(error: object, target: list[str]) -> None:
    target.append(str(error))


def _record_console(message: _ConsoleMessage, target: list[str]) -> None:
    # Transient host network changes can make the browser report a failed resource load;
    # real JavaScript or application errors still abort the capture.
    if message.type == "error" and not message.text.startswith("Failed to load resource"):
        target.append(message.text)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _command(*args: str) -> list[str]:
    executable = shutil.which("mll")
    if executable is not None:
        return [executable, *args]
    return [sys.executable, "-m", "mlops_lifecycle_lab", *args]


def _reset_workspace(workspace: Path) -> None:
    """Remove a previous screenshot workspace, refusing unsafe absolute targets."""
    resolved = (ROOT / workspace).resolve() if not workspace.is_absolute() else workspace.resolve()
    managed = (ROOT / ".mll").resolve()
    scratch = Path(tempfile.gettempdir()).resolve()
    if not (resolved == managed or managed in resolved.parents or scratch in resolved.parents):
        raise SystemExit(f"refusing to reset {resolved}: keep it under {managed} or {scratch}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _seed(workspace: Path, weeks: int, *, fast: bool) -> None:
    command = _command(
        "drill",
        "run",
        "--scenario",
        "drift",
        "--weeks",
        str(weeks),
        "--workspace",
        str(workspace),
    )
    if fast:
        command.append("--fast")
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603 - fixed argv, no shell


@contextmanager
def _dashboard(workspace: Path) -> Iterator[str]:
    port = _free_port()
    environment = os.environ.copy()
    environment["MLL_WORKSPACE"] = str(workspace)
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        _command("dashboard", "--port", str(port), "--workspace", str(workspace)),
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_healthy(base_url)
        yield base_url
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def _wait_until_healthy(base_url: str) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed loopback URL
                f"{base_url}/_stcore/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError(f"dashboard did not become healthy at {base_url}")


def _settle(page: Page) -> None:
    page.wait_for_load_state("networkidle", timeout=SETTLE_TIMEOUT_MS)
    page.locator(SPINNER).first.wait_for(state="hidden", timeout=SETTLE_TIMEOUT_MS)
    page.wait_for_timeout(800)


def _open_page(page: Page, name: str) -> None:
    if name != "Overview":
        page.locator('[data-testid="stSidebar"] label').filter(has_text=name).first.click()
    page.get_by_role("heading", name=name, exact=True).wait_for(timeout=SETTLE_TIMEOUT_MS)
    _settle(page)


def _screenshot(page: Page, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), full_page=False)
    _optimize_png(path)


def _optimize_png(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.save(path, format="PNG", optimize=True)
    if path.stat().st_size <= PNG_BUDGET_BYTES:
        return
    with Image.open(path) as image:
        image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256).save(
            path, format="PNG", optimize=True
        )
    if path.stat().st_size > PNG_BUDGET_BYTES:
        raise RuntimeError(f"{path.name} exceeds the {PNG_BUDGET_BYTES}-byte PNG budget")


def _scroll_to(page: Page, text: str) -> None:
    locator = page.get_by_text(text, exact=False).first
    locator.scroll_into_view_if_needed(timeout=SETTLE_TIMEOUT_MS)
    _settle(page)


def _capture(base_url: str, output: Path, hero_output: Path | None) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for scheme in SCHEMES:
                context = browser.new_context(
                    color_scheme=scheme,
                    viewport=VIEWPORT,
                    device_scale_factor=1,
                )
                page = context.new_page()
                errors: list[str] = []
                page.on("console", partial(_record_console, target=errors))
                page.on("pageerror", partial(_record_pageerror, target=errors))
                page.goto(base_url, wait_until="load")
                page.get_by_role("heading", name="Overview", exact=True).wait_for(
                    timeout=SETTLE_TIMEOUT_MS
                )
                _settle(page)
                for name in PAGES:
                    _open_page(page, name)
                    canonical = output / f"{name.lower()}-{scheme}.png"
                    _screenshot(page, canonical)
                    if hero_output is not None:
                        _capture_hero_frame(page, name, scheme, canonical, hero_output)
                    if errors:
                        raise RuntimeError(f"{scheme}/{name}: browser errors: {errors}")
                context.close()
        finally:
            browser.close()


def _capture_hero_frame(
    page: Page,
    name: str,
    scheme: str,
    canonical: Path,
    hero_output: Path,
) -> None:
    """Write the eight frames of the hero GIF, in display order, into `hero_output`."""
    if scheme == "light":
        if name == "Overview":
            _copy(canonical, hero_output / "01-overview.png")
        elif name == "Monitoring":
            _copy(canonical, hero_output / "02-monitoring.png")
            _scroll_to(page, "Validation verdicts")
            _screenshot(page, hero_output / "03-monitoring-scrolled.png")
        elif name == "Registry":
            _scroll_to(page, "Gate report")
            _screenshot(page, hero_output / "04-registry-gates.png")
        elif name == "Lifecycle":
            _copy(canonical, hero_output / "05-lifecycle.png")
        elif name == "Drill":
            _copy(canonical, hero_output / "06-drill.png")
    elif name == "Registry":
        _copy(canonical, hero_output / "07-registry-dark.png")
    elif name == "Overview":
        _copy(canonical, hero_output / "08-overview-dark.png")


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--hero-output",
        type=Path,
        default=None,
        help="Also write the eight hero-GIF frames into this directory.",
    )
    parser.add_argument("--weeks", type=int, default=52)
    parser.add_argument("--fast", action="store_true", help="Skip Evidently HTML reports.")
    parser.add_argument(
        "--reuse-workspace",
        action="store_true",
        help="Keep an existing workspace instead of replaying the drill from scratch.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.reuse_workspace:
        _reset_workspace(args.workspace)
    _seed(args.workspace, args.weeks, fast=args.fast)
    with _dashboard(args.workspace) as base_url:
        _capture(base_url, args.output, args.hero_output)
    written = sorted(path.name for path in args.output.glob("*.png"))
    print(f"captured {len(written)} screenshots into {args.output}")
    if args.hero_output is not None:
        frames = sorted(path.name for path in args.hero_output.glob("*.png"))
        print(f"captured {len(frames)} hero frames into {args.hero_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
