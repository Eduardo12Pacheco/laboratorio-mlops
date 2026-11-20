#!/usr/bin/env python3
"""Render the two curated Evidently HTML reports committed under docs/assets (T14).

These are the human-readable companions to the native PSI numbers in
`docs/results/drill-drift.json`:

- `drift-2012-04-01.html`: the tick that triggered the first retrain, with v1's training
  window as the reference and the trailing 28 days as the current window;
- `drift-2012-12-30.html`: the final tick, with v2's training window as the reference.

Evidently inlines its whole JavaScript dashboard (about 3.5 MB) in every file. The script
extracts the identical bundle into a single `evidently-dashboard.js` shared by both
reports, which keeps the two committed reports inside the plan's 5 MB documentation
budget without dropping any content or depending on a CDN.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset
from mlops_lifecycle_lab.adapters.drift.evidently_reports import EvidentlyReportRenderer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "reports"
SHARED_SCRIPT = "evidently-dashboard.js"
MIN_BUNDLE_BYTES = 1_000_000

# (filename, reference start, reference end, current start, current end)
REPORTS: tuple[tuple[str, datetime, datetime, datetime, datetime], ...] = (
    (
        "drift-2012-04-01.html",
        datetime(2011, 1, 1),
        datetime(2011, 10, 1),
        datetime(2012, 3, 4),
        datetime(2012, 4, 1),
    ),
    (
        "drift-2012-12-30.html",
        datetime(2011, 1, 1),
        datetime(2012, 3, 18),
        datetime(2012, 12, 2),
        datetime(2012, 12, 30),
    ),
)


def _window(raw: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    mask = (raw["timestamp"] >= start) & (raw["timestamp"] < end)
    return raw.loc[mask].copy()


def _extract_shared_script(html: str, output_dir: Path) -> str:
    """Move the largest inline script to one shared file and return the new HTML."""
    matches = list(re.finditer(r"<script[^>]*>(.*?)</script>", html, re.DOTALL))
    if not matches:
        return html
    largest = max(matches, key=lambda match: len(match.group(1)))
    bundle = largest.group(1)
    if len(bundle) < MIN_BUNDLE_BYTES:
        return html
    name = SHARED_SCRIPT
    target = output_dir / name
    if not target.exists() or target.read_text(encoding="utf-8") != bundle:
        target.write_text(bundle, encoding="utf-8")
    replacement = f'<script src="{name}"></script>'
    return html[: largest.start()] + replacement + html[largest.end() :]


def render(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Render both curated reports and return the written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = BundledDataset().raw()
    written: list[Path] = []
    for name, ref_start, ref_end, current_start, current_end in REPORTS:
        reference = _window(raw, ref_start, ref_end)
        current = _window(raw, current_start, current_end)
        if reference.empty or current.empty:
            raise ValueError(f"{name}: empty reference or current window")
        renderer = EvidentlyReportRenderer(timestamp=current_end)
        temporary = output_dir / f".{name}.tmp"
        renderer.render(reference, current, temporary)
        html = temporary.read_text(encoding="utf-8")
        temporary.unlink()
        html = _extract_shared_script(html, output_dir)
        destination = output_dir / name
        destination.write_text(html, encoding="utf-8")
        written.append(destination)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (default: docs/assets/reports).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in render(args.output_dir):
        print(f"wrote {path}")
    total = sum(item.stat().st_size for item in args.output_dir.glob("*") if item.is_file())
    print(f"bundle size: {total} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
