"""The `drill-report/v1` document, its markdown renderer and the performance summary."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mlops_lifecycle_lab.application.dto import (
    LabelRecord,
    MonitoringSnapshot,
    PredictionRecord,
    encode_version_tags,
)
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from mlops_lifecycle_lab.entrypoints.drill.report import (
    GATE_APPROVED,
    SCHEMA,
    DrillResult,
    DrillSummary,
    GateReportRecord,
    MetricsSummary,
    TickRecord,
    library_versions,
    summarize_performance,
)
from tests.fakes.data import raw_frame
from tests.fakes.prediction_log import InMemoryPredictionLog
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.trainer import MeanPredictor
from tests.unit.entrypoints.conftest import model_version


def _event(event_type: EventType, tick: int = 1) -> LifecycleEvent:
    return LifecycleEvent(
        ts=datetime(2012, 1, 8),
        type=event_type,
        tick=tick,
        model_version=1,
        payload={"rows": 3},
    )


def _tick(tick: int = 1) -> TickRecord:
    snapshot = MonitoringSnapshot(
        tick=tick,
        ts=datetime(2012, 1, 8),
        model_version=1,
        n_rows=3,
        rmse=10.0,
        mae=8.0,
        wape=0.2,
        bias=0.0,
        numeric_drift_share=0.5,
        prediction_psi=0.1,
        psi={"temp": 0.3},
    )
    return TickRecord(
        tick=tick,
        tick_end=datetime(2012, 1, 8),
        champion_version=1,
        snapshot=snapshot,
        validation_verdict="WARN",
        action="SCORED",
        events=(
            _event(EventType.BATCH_VALIDATED, tick),
            _event(EventType.SNAPSHOT_COMPUTED, tick),
        ),
    )


def _summary() -> DrillSummary:
    return DrillSummary(
        as_served=MetricsSummary(rmse=70.72, mae=45.89, wape=0.195, n=8686),
        counterfactual_frozen_v1=MetricsSummary(rmse=109.92, mae=74.52, wape=0.317, n=8686),
        promotions=1,
        rejections=1,
        rollbacks=0,
        quarantined_batches=0,
        versions=3,
    )


def _result() -> DrillResult:
    return DrillResult(
        scenario="drift",
        seed=42,
        weeks=52,
        ticks=(_tick(1),),
        gate_reports=(
            GateReportRecord(
                tick=13,
                version=2,
                decision=GATE_APPROVED,
                gates=(
                    {
                        "name": "IMPROVEMENT",
                        "passed": True,
                        "observed": 0.56,
                        "threshold": 0.05,
                    },
                ),
            ),
        ),
        summary=_summary(),
    )


def test_library_versions_cover_the_reported_stack() -> None:
    versions = library_versions()

    assert set(versions) == {
        "lightgbm",
        "mlflow",
        "pandas",
        "numpy",
        "scikit-learn",
        "evidently",
    }
    assert all(isinstance(value, str) and value for value in versions.values())


def test_tick_record_serialises_snapshot_events_and_verdict() -> None:
    payload = _tick().to_dict()

    assert payload["tick"] == 1
    assert payload["tick_end"] == "2012-01-08T00:00:00"
    assert payload["champion_version"] == 1
    assert payload["validation_verdict"] == "WARN"
    assert payload["action"] == "SCORED"
    assert payload["snapshot"]["wape"] == 0.2
    assert [event["type"] for event in payload["events"]] == [
        "BATCH_VALIDATED",
        "SNAPSHOT_COMPUTED",
    ]


def test_report_document_has_the_pinned_schema_and_no_absolute_paths() -> None:
    payload = _result().to_dict()

    assert payload["schema"] == SCHEMA
    assert set(payload) == {
        "schema",
        "scenario",
        "seed",
        "weeks",
        "libs",
        "ticks",
        "gate_reports",
        "summary",
    }
    assert set(payload["summary"]) == {
        "as_served",
        "counterfactual_frozen_v1",
        "promotions",
        "rejections",
        "rollbacks",
        "quarantined_batches",
        "versions",
    }
    text = _result().to_json()
    assert ("/" + "tmp") not in text
    assert "/home/" not in text
    assert json.loads(text)["gate_reports"][0]["decision"] == "APPROVED"


def test_report_json_is_byte_identical_for_identical_results() -> None:
    assert _result().to_json() == _result().to_json()


def test_gate_report_parses_the_gates_evaluated_event() -> None:
    event = LifecycleEvent(
        ts=datetime(2012, 4, 1),
        type=EventType.GATES_EVALUATED,
        tick=13,
        model_version=2,
        payload={
            "approved": True,
            "gate_report": {
                "approved": True,
                "results": [
                    {
                        "name": "IMPROVEMENT",
                        "passed": True,
                        "observed": 0.56,
                        "threshold": 0.05,
                        "detail": "ok",
                    }
                ],
            },
        },
    )

    record = GateReportRecord.from_event(event)

    assert record is not None
    assert record.tick == 13
    assert record.version == 2
    assert record.decision == GATE_APPROVED
    assert record.gates[0]["name"] == "IMPROVEMENT"
    assert GateReportRecord.from_event(_event(EventType.BATCH_VALIDATED)) is None


def test_markdown_contains_the_timeline_gates_and_summary() -> None:
    text = _result().markdown()

    assert "# Drill report: drift" in text
    assert "| 1 | 2012-01-08" in text
    assert "IMPROVEMENT" in text
    assert "as-served" in text
    assert "frozen v1" in text
    assert "promotions=1" in text


def test_result_writes_json_and_markdown(tmp_path: Path) -> None:
    paths = _result().write(tmp_path)

    assert {path.name for path in paths} == {"drill-drift.json", "drill-drift.md"}
    assert all(path.exists() for path in paths)
    assert json.loads(paths[0].read_text(encoding="utf-8"))["schema"] == SCHEMA


def test_summarize_performance_computes_as_served_and_frozen_v1() -> None:
    log = InMemoryPredictionLog()
    registry = InMemoryRegistry()
    registry.adopt_candidate(1, encode_version_tags(model_version(1)))
    registry.set_predictor(1, MeanPredictor(2.0))
    frame = raw_frame(periods=4)
    features = build_features(frame)
    for position, row in frame.iterrows():
        log.append(
            [
                PredictionRecord(
                    prediction_id=f"p-{position}",
                    entity_id=int(row["entity_id"]),
                    event_ts=row["timestamp"].to_pydatetime(),
                    scored_at=row["timestamp"].to_pydatetime(),
                    model_name="bike-demand",
                    model_version=1,
                    features={
                        name: float(features.loc[position, name]) for name in FEATURE_COLUMNS
                    },
                    prediction=float(row["cnt"]),
                    source="batch",
                )
            ]
        )
        log.attach_labels(
            [
                LabelRecord(
                    entity_id=int(row["entity_id"]),
                    label=float(row["cnt"]),
                    label_ts=row["timestamp"].to_pydatetime(),
                )
            ]
        )

    served, frozen = summarize_performance(
        registry,
        log,
        datetime(2011, 1, 1),
        datetime(2011, 1, 2),
    )

    assert served.n == 4
    assert served.rmse == pytest.approx(0.0)
    assert frozen.n == 4
    assert frozen.rmse == pytest.approx(float(np.sqrt(np.mean((frame["cnt"] - 2.0) ** 2))))
    assert frozen.wape > 0
