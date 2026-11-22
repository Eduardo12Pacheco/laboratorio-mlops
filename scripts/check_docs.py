#!/usr/bin/env python3
"""Check the documentation set and committed artifacts (S24, T14).

The script is intentionally dependency-free so anyone can run it from a fresh clone:

- required documents and the ADR set exist, and every ADR keeps the four-part
  Context / Decision / Consequences / Evidence shape with at least 20 non-empty lines;
- every relative Markdown link and image path resolves;
- every `docs/results/drill-*.{json,md}` exists, parses and uses `drill-report/v1`;
- `results-table.md` matches a fresh render from those JSON reports;
- the README / model-card results blocks, when present, match the table verbatim;
- the curated Evidently report bundle exists, references only local assets and stays
  within the 5 MB documentation budget;
- no absolute owner-home paths leaked into the documentation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DOCS = (
    "ARCHITECTURE.md",
    "DATA.md",
    "MODEL_CARD.md",
    "OPERATIONS.md",
)
ADR_HEADINGS = ("## Context", "## Decision", "## Consequences", "## Evidence")
ADR_MIN_NON_EMPTY_LINES = 20
ADR_MIN_COUNT = 9
RESULT_SCHEMA = "drill-report/v1"
RESULT_SCENARIOS = ("drift", "gate-reject", "bad-batch", "rollback")
REPORT_FILES = ("drift-2012-04-01.html", "drift-2012-12-30.html")
REPORT_BUNDLE_MAX_BYTES = 5_000_000
RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"

_LINK_RE = re.compile(r"\]\(([^)]+)\)")
_HTML_RESOURCE_RE = re.compile(r"""(?:src|href)="([^"]+)\"""")
_ABSOLUTE_HOME_RE = re.compile(r"(/home/[^/\s]+/|C:\\\\Users\\\\)")


def _markdown_files(root: Path) -> list[Path]:
    files = sorted((root / "docs").rglob("*.md"))
    for name in ("README.md", "README.en.md", "CONTRIBUTING.md", "SECURITY.md"):
        candidate = root / name
        if candidate.is_file():
            files.append(candidate)
    return files


def _load_table_module() -> ModuleType:
    """Load `render_results_table.py` without requiring a package or working directory."""
    path = Path(__file__).with_name("render_results_table.py")
    spec = importlib.util.spec_from_file_location("mll_render_results_table", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_required_docs(root: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_DOCS:
        path = root / "docs" / name
        if not path.is_file():
            errors.append(f"missing required document: docs/{name}")
        elif not path.read_text(encoding="utf-8").strip():
            errors.append(f"empty required document: docs/{name}")
    return errors


def _check_links(root: Path) -> list[str]:
    errors: list[str] = []
    for path in _markdown_files(root):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for raw_target in _LINK_RE.findall(text):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1].strip()
            if target.startswith(("#", "http://", "https://", "mailto:", "data:")):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            if target.startswith("/"):
                errors.append(f"{relative}: absolute link target {target!r}")
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{relative}: broken relative link {target!r}")
    return errors


def _check_adrs(root: Path) -> list[str]:
    errors: list[str] = []
    adrs = sorted((root / "docs" / "adr").glob("*.md"))
    if len(adrs) < ADR_MIN_COUNT:
        errors.append(f"docs/adr: expected at least {ADR_MIN_COUNT} ADRs, found {len(adrs)}")
    for path in adrs:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        non_empty = [line for line in text.splitlines() if line.strip()]
        if len(non_empty) < ADR_MIN_NON_EMPTY_LINES:
            errors.append(
                f"{relative}: expected at least {ADR_MIN_NON_EMPTY_LINES} non-empty lines, "
                f"found {len(non_empty)}"
            )
        for heading in ADR_HEADINGS:
            if not any(line.strip() == heading for line in text.splitlines()):
                errors.append(f"{relative}: missing {heading!r} section")
    return errors


def _check_results(root: Path, table_module: ModuleType) -> list[str]:
    errors: list[str] = []
    results_dir = root / "docs" / "results"
    for scenario in RESULT_SCENARIOS:
        json_path = results_dir / f"drill-{scenario}.json"
        markdown_path = results_dir / f"drill-{scenario}.md"
        if not json_path.is_file():
            errors.append(f"missing committed drill result: docs/results/drill-{scenario}.json")
            continue
        if not markdown_path.is_file():
            errors.append(f"missing committed drill result: docs/results/drill-{scenario}.md")
        try:
            payload: Any = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append(f"docs/results/drill-{scenario}.json: invalid JSON ({error})")
            continue
        if not isinstance(payload, dict) or payload.get("schema") != RESULT_SCHEMA:
            errors.append(f"docs/results/drill-{scenario}.json: expected schema {RESULT_SCHEMA!r}")
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            errors.append(f"docs/results/drill-{scenario}.json: missing summary")
            continue
        for section in ("as_served", "counterfactual_frozen_v1"):
            if not isinstance(summary.get(section), dict):
                errors.append(f"docs/results/drill-{scenario}.json: missing summary.{section}")
        if not isinstance(payload.get("weeks"), int):
            errors.append(f"docs/results/drill-{scenario}.json: missing integer weeks")

    table_path = results_dir / "results-table.md"
    if not table_path.is_file():
        errors.append("missing generated table: docs/results/results-table.md")
    else:
        try:
            expected = table_module.render_table(results_dir)
        except (FileNotFoundError, ValueError) as error:
            errors.append(f"docs/results/results-table.md: cannot render ({error})")
        else:
            if table_path.read_text(encoding="utf-8") != expected:
                errors.append(
                    "docs/results/results-table.md is stale; run "
                    "`uv run python scripts/render_results_table.py`"
                )
    return errors


def _check_results_blocks(root: Path, table_module: ModuleType) -> list[str]:
    errors: list[str] = []
    results_dir = root / "docs" / "results"
    try:
        expected = table_module.render_table(results_dir).strip()
    except (FileNotFoundError, ValueError):
        return errors
    for name in ("README.md", "README.en.md", "docs/MODEL_CARD.md"):
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if RESULTS_START not in text or RESULTS_END not in text:
            continue
        try:
            block = text.split(RESULTS_START, 1)[1].split(RESULTS_END, 1)[0].strip()
        except IndexError:
            errors.append(f"{name}: malformed results block markers")
            continue
        if block != expected:
            errors.append(f"{name} results block does not match docs/results/results-table.md")
    return errors


def _check_report_bundle(root: Path) -> list[str]:
    errors: list[str] = []
    reports = root / "docs" / "assets" / "reports"
    if not reports.is_dir():
        return ["missing curated report directory: docs/assets/reports"]
    total = 0
    for path in reports.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    if total > REPORT_BUNDLE_MAX_BYTES:
        errors.append(f"docs/assets/reports: {total} bytes exceeds the 5 MB documentation budget")
    for name in REPORT_FILES:
        html = reports / name
        if not html.is_file():
            errors.append(f"missing curated report: docs/assets/reports/{name}")
            continue
        text = html.read_text(encoding="utf-8", errors="replace")
        for target in _HTML_RESOURCE_RE.findall(text):
            if target.startswith(("http://", "https://", "data:", "#", "mailto:")):
                continue
            resolved = (html.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{name}: missing local report asset {target!r}")
    return errors


def _check_no_home_paths(root: Path) -> list[str]:
    errors: list[str] = []
    for path in _markdown_files(root):
        text = path.read_text(encoding="utf-8")
        if _ABSOLUTE_HOME_RE.search(text):
            errors.append(f"{path.relative_to(root).as_posix()}: absolute owner-home path")
    return errors


def check_tree(root: Path) -> list[str]:
    """Return every documentation problem found under `root` (empty means valid)."""
    table_module = _load_table_module()
    errors: list[str] = []
    errors.extend(_check_required_docs(root))
    errors.extend(_check_links(root))
    errors.extend(_check_adrs(root))
    errors.extend(_check_results(root, table_module))
    errors.extend(_check_results_blocks(root, table_module))
    errors.extend(_check_report_bundle(root))
    errors.extend(_check_no_home_paths(root))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root to check (default: the repository containing this script).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root.resolve()
    errors = check_tree(root)
    if errors:
        for error in errors:
            print(f"- {error}")
        print(f"docs FAILED ({len(errors)} problem(s))")
        return 1
    print("docs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
