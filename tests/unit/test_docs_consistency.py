"""Documentation consistency gates (S24, T14).

The docs are part of the product: links must resolve, ADRs must keep their four-part
shape, committed drill results must stay parseable and the results table must be
reproducible from the JSON reports. The tests drive the same script a reviewer runs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run_script(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _check_docs(root: Path) -> subprocess.CompletedProcess[str]:
    return _run_script(str(ROOT / "scripts" / "check_docs.py"), "--root", str(root))


def _scaffold(root: Path) -> None:
    """Minimal tree that satisfies every check except the one under test."""
    docs = root / "docs"
    (docs / "adr").mkdir(parents=True)
    (docs / "results").mkdir(parents=True)
    (docs / "ARCHITECTURE.md").write_text("# Architecture\n\nSee [data](DATA.md).\n")
    (docs / "DATA.md").write_text("# Data\n")
    (docs / "MODEL_CARD.md").write_text("# Model card\n")
    (docs / "OPERATIONS.md").write_text("# Operations\n")
    adr = docs / "adr" / "0001-example.md"
    context = "\n".join(f"Context line {index}." for index in range(18))
    adr.write_text(
        "# ADR-0001: Example\n\n"
        f"## Contexto\n\n{context}\n\n"
        "## Decisión\n\nDecision line.\n\n"
        "## Consecuencias\n\nConsequences line.\n\n"
        "## Evidencia\n\nEvidence line.\n"
    )
    report = {
        "schema": "drill-report/v1",
        "scenario": "drift",
        "weeks": 52,
        "summary": {
            "as_served": {"rmse": 70.0, "mae": 45.0, "wape": 0.19, "n": 1},
            "counterfactual_frozen_v1": {
                "rmse": 109.0,
                "mae": 74.0,
                "wape": 0.31,
                "n": 1,
            },
            "promotions": 1,
            "rejections": 1,
            "rollbacks": 0,
            "quarantined_batches": 0,
            "versions": 3,
        },
    }
    for scenario in ("drift", "gate-reject", "bad-batch", "rollback"):
        report["scenario"] = scenario
        (docs / "results" / f"drill-{scenario}.json").write_text(json.dumps(report))
        (docs / "results" / f"drill-{scenario}.md").write_text(f"# Drill report: {scenario}\n")
    rendered = _run_script(
        str(ROOT / "scripts" / "render_results_table.py"),
        "--results-dir",
        str(docs / "results"),
        cwd=root,
    )
    assert rendered.returncode == 0, rendered.stderr


def test_repository_docs_pass_the_consistency_check() -> None:
    completed = _check_docs(ROOT)
    assert completed.returncode == 0, completed.stderr
    assert "docs OK" in completed.stdout


def test_committed_results_table_is_reproducible() -> None:
    completed = _run_script(str(ROOT / "scripts" / "render_results_table.py"), "--check")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "results table OK" in completed.stdout


def test_missing_relative_link_is_reported(tmp_path: Path) -> None:
    _scaffold(tmp_path)
    (tmp_path / "docs" / "ARCHITECTURE.md").write_text("# Architecture\n\nSee [gone](gone.md).\n")
    completed = _check_docs(tmp_path)
    assert completed.returncode == 1
    assert "gone.md" in completed.stdout


def test_adr_must_keep_its_four_sections(tmp_path: Path) -> None:
    _scaffold(tmp_path)
    adr = tmp_path / "docs" / "adr" / "0001-example.md"
    adr.write_text("# ADR-0001\n\n## Contexto\n\nonly\n")
    completed = _check_docs(tmp_path)
    assert completed.returncode == 1
    assert "0001-example.md" in completed.stdout
    assert "Evidencia" in completed.stdout


def test_drill_json_must_carry_the_report_schema(tmp_path: Path) -> None:
    _scaffold(tmp_path)
    payload = json.loads((tmp_path / "docs" / "results" / "drill-drift.json").read_text())
    payload["schema"] = "drill-golden/v1"
    (tmp_path / "docs" / "results" / "drill-drift.json").write_text(json.dumps(payload))
    completed = _check_docs(tmp_path)
    assert completed.returncode == 1
    assert "drill-report/v1" in completed.stdout


@pytest.mark.parametrize("readme", ["README.md", "README.en.md"])
def test_readme_results_block_must_match_the_generated_table(tmp_path: Path, readme: str) -> None:
    _scaffold(tmp_path)
    (tmp_path / readme).write_text(
        "# Demo\n\n<!-- results:start -->\nstale numbers\n<!-- results:end -->\n"
    )
    completed = _check_docs(tmp_path)
    assert completed.returncode == 1
    assert f"{readme} results block" in completed.stdout


def test_report_bundle_stays_under_the_size_limit(tmp_path: Path) -> None:
    _scaffold(tmp_path)
    reports = tmp_path / "docs" / "assets" / "reports"
    reports.mkdir(parents=True)
    (reports / "drift-2012-04-01.html").write_text("<html></html>")
    (reports / "drill").write_bytes(b"x" * (5 * 1024 * 1024))
    completed = _check_docs(tmp_path)
    assert completed.returncode == 1
    assert "5 MB" in completed.stdout


def test_results_table_renderer_is_deterministic_and_ordered(tmp_path: Path) -> None:
    results = tmp_path / "results"
    shutil.copytree(ROOT / "docs" / "results", results)
    first = _run_script(
        str(ROOT / "scripts" / "render_results_table.py"),
        "--results-dir",
        str(results),
        cwd=tmp_path,
    )
    assert first.returncode == 0, first.stderr
    table = (results / "results-table.md").read_text()
    second = _run_script(
        str(ROOT / "scripts" / "render_results_table.py"),
        "--results-dir",
        str(results),
        cwd=tmp_path,
    )
    assert second.returncode == 0, second.stderr
    assert (results / "results-table.md").read_text() == table
    assert table.index("drift") < table.index("gate-reject") < table.index("bad-batch")
    assert table.index("bad-batch") < table.index("rollback")
