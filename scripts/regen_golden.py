#!/usr/bin/env python
"""Regenerate the committed drill goldens from a verified engine run (D19).

Run this only after investigating a genuine behaviour change, and mention the generating
command in the commit body. The script uses the Null drift renderer (`fast=True`) because
the golden compares lifecycle behaviour, not Evidently HTML.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from mlops_lifecycle_lab.entrypoints.drill.engine import DrillEngine
from mlops_lifecycle_lab.entrypoints.drill.expectations import make_golden
from mlops_lifecycle_lab.entrypoints.drill.scenarios import SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", *sorted(SCENARIOS)],
        help="Scenario to regenerate (default: all).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/golden"),
        help="Golden destination directory (default: tests/golden).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        with tempfile.TemporaryDirectory(prefix=f"drill-golden-{name}-") as root:
            workspace = Path(root) / "workspace"
            result = DrillEngine().run(
                name,
                workspace=workspace,
                report_dir=Path(root) / "reports",
                fast=True,
            )
            golden = make_golden(result)
            destination = args.out_dir / f"drill_{name.replace('-', '_')}.json"
            destination.write_text(
                json.dumps(golden, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary = result.summary
            print(
                f"{name}: weeks={result.weeks} retrain_tick={result.first_retrain_tick} "
                f"promotions={summary.promotions} rejections={summary.rejections} "
                f"rollbacks={summary.rollbacks} "
                f"quarantined={summary.quarantined_batches} -> {destination}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
