"""Pipeline, run ledger and DTO contracts of the application layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from mlops_lifecycle_lab.application.dto import (
    EvaluationRequest,
    MonitoringSnapshot,
    PredictionRecord,
    RunStatus,
    StepStatus,
    TickReport,
    TrainRequest,
    decode_version_tags,
    encode_version_tags,
)
from mlops_lifecycle_lab.application.pipeline import Pipeline, PipelineError, Step
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.lifecycle import ModelStage, ModelVersion
from mlops_lifecycle_lab.domain.windows import TimeWindow
from tests.fakes.clock import FakeClock
from tests.fakes.stores import RecordingLedger

NOW = datetime(2012, 4, 1, 12)


def test_pipeline_runs_steps_in_order_and_records_the_ledger() -> None:
    ledger = RecordingLedger()
    seen: list[str] = []

    def first(context: dict[str, Any]) -> dict[str, Any]:
        seen.append("first")
        return context

    def second(context: dict[str, Any]) -> dict[str, Any]:
        seen.append("second")
        return context

    run = Pipeline([Step("first", first), Step("second", second)], ledger, name="tick").run(
        {"tick": 3}
    )

    assert seen == ["first", "second"]
    assert run.status is RunStatus.SUCCEEDED
    assert run.run_id
    assert ledger.starts[0][1] == "tick"
    assert ledger.starts[0][2] == 3
    assert [(entry.name, entry.status) for entry in ledger.steps] == [
        ("first", StepStatus.SUCCEEDED),
        ("second", StepStatus.SUCCEEDED),
    ]
    assert [entry.index for entry in ledger.steps] == [0, 1]
    assert ledger.finishes[-1].status is RunStatus.SUCCEEDED


def test_pipeline_retries_a_failing_step_and_records_every_attempt() -> None:
    ledger = RecordingLedger()
    sleeps: list[float] = []
    attempts: list[int] = []

    def flaky(context: dict[str, Any]) -> dict[str, Any]:
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return context

    run = Pipeline(
        [Step("flaky", flaky, retries=2, backoff_seconds=0.5)],
        ledger,
        sleeper=sleeps.append,
    ).run({})

    assert run.status is RunStatus.SUCCEEDED
    assert attempts == [1, 2, 3]
    assert sleeps == [0.5, 0.5]
    assert [entry.status for entry in ledger.steps] == [
        StepStatus.FAILED,
        StepStatus.FAILED,
        StepStatus.SUCCEEDED,
    ]
    assert ledger.steps[0].error is not None
    assert "ConnectionError" in ledger.steps[0].error
    assert run.steps[0].attempts == 3


def test_pipeline_raises_after_the_retry_budget_and_marks_the_run_failed() -> None:
    ledger = RecordingLedger()

    def broken(context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    with pytest.raises(PipelineError, match="broken") as caught:
        Pipeline([Step("broken", broken, retries=2)], ledger).run({})

    assert caught.value.step == "broken"
    assert caught.value.attempts == 3
    assert len(ledger.steps) == 3
    assert {entry.status for entry in ledger.steps} == {StepStatus.FAILED}
    assert ledger.finishes[-1].status is RunStatus.FAILED
    assert "boom" in (ledger.finishes[-1].error or "")


def test_pipeline_does_not_retry_an_exception_outside_retry_on() -> None:
    ledger = RecordingLedger()
    attempts: list[int] = []

    def broken(context: dict[str, Any]) -> dict[str, Any]:
        attempts.append(1)
        raise KeyError("not retryable")

    with pytest.raises(PipelineError):
        Pipeline(
            [Step("broken", broken, retries=5, retry_on=(ValueError,))],
            ledger,
        ).run({})

    assert len(attempts) == 1
    assert len(ledger.steps) == 1


def test_pipeline_replaces_the_context_with_the_step_result() -> None:
    def enrich(context: dict[str, Any]) -> dict[str, Any]:
        return {**context, "enriched": True}

    run = Pipeline([Step("enrich", enrich)], RecordingLedger()).run({"base": 1})

    assert run.context == {"base": 1, "enriched": True}


def test_prediction_record_freezes_features_and_rejects_an_unknown_source() -> None:
    features = {"temp": 0.5, "hr": 12}
    record = PredictionRecord(
        prediction_id="p1",
        entity_id=7,
        event_ts=NOW,
        scored_at=NOW,
        model_name="bike-demand",
        model_version=1,
        features=features,
        prediction=42.0,
        source="online",
    )
    features["temp"] = 0.9

    assert record.features["temp"] == 0.5
    with pytest.raises(ValueError, match="unknown prediction source"):
        PredictionRecord(
            prediction_id="p2",
            entity_id=7,
            event_ts=NOW,
            scored_at=NOW,
            model_name="bike-demand",
            model_version=1,
            features={"temp": 0.5},
            prediction=42.0,
            source="stream",  # type: ignore[arg-type]
        )


def test_monitoring_snapshot_serialises_undefined_metrics_as_none() -> None:
    snapshot = MonitoringSnapshot(
        tick=1,
        ts=NOW,
        model_version=1,
        n_rows=0,
        rmse=None,
        mae=None,
        wape=None,
        bias=None,
        numeric_drift_share=0.0,
        prediction_psi=None,
        psi={"temp": 0.1},
        verdict="PASS",
    )

    payload = snapshot.to_dict()

    assert payload["n_rows"] == 0
    assert payload["rmse"] is None
    assert payload["psi"] == {"temp": 0.1}
    assert payload["verdict"] == "PASS"


def test_tick_report_serialises_events_and_snapshot() -> None:
    snapshot = MonitoringSnapshot(
        tick=2,
        ts=NOW,
        model_version=1,
        n_rows=10,
        rmse=1.0,
        mae=0.5,
        wape=0.1,
        bias=0.0,
        numeric_drift_share=0.25,
        prediction_psi=0.2,
    )
    event = LifecycleEvent(ts=NOW, type=EventType.BATCH_SCORED, tick=2, model_version=1)
    report = TickReport(
        tick=2,
        window=TimeWindow(datetime(2012, 3, 25), NOW),
        champion_version=1,
        validation_verdict="PASS",
        action="SCORED",
        snapshot=snapshot,
        events=(event,),
    )

    payload = report.to_dict()

    assert payload["tick"] == 2
    assert payload["window"]["start"] == "2012-03-25T00:00:00"
    assert payload["snapshot"]["n_rows"] == 10
    assert payload["events"][0]["type"] == "BATCH_SCORED"


def test_train_and_evaluation_requests_require_a_reason_and_positive_version() -> None:
    window = TimeWindow(datetime(2012, 1, 1), datetime(2012, 2, 1))
    request = TrainRequest(
        train_window=window,
        holdout_window=window,
        trigger_reason="MANUAL",
        tick=3,
    )
    assert request.tick == 3
    with pytest.raises(ValueError, match="trigger_reason"):
        TrainRequest(train_window=window, holdout_window=window, trigger_reason=" ")
    with pytest.raises(ValueError, match="version"):
        EvaluationRequest(
            version=0,
            train_window=window,
            holdout_window=window,
            training_verdict="PASS",
        )


def test_model_version_tags_round_trip_losslessly() -> None:
    version = ModelVersion(
        name="bike-demand",
        version=2,
        stage=ModelStage.CHAMPION,
        run_id="run-2",
        metrics={"rmse": 53.3, "mae": 30.0, "wape": 0.147, "bias": 1.5},
        train_window=TimeWindow(datetime(2011, 1, 1), datetime(2012, 3, 18)),
        holdout_window=TimeWindow(datetime(2012, 3, 18), datetime(2012, 4, 1)),
        data_fingerprint="abc123",
        feature_spec_version="1",
        created_at=NOW,
        trigger_reason="PERFORMANCE_SLO_BREACH",
        tags={"mll.git_sha": "deadbeef"},
    )

    restored = decode_version_tags("bike-demand", 2, encode_version_tags(version))

    assert restored == version


def test_pipeline_uses_the_clock_for_ledger_timestamps() -> None:
    clock = FakeClock(NOW)
    ledger = RecordingLedger()
    Pipeline([Step("noop", lambda context: context)], ledger, clock=clock).run({})

    assert ledger.starts[0][3] == NOW
    assert ledger.steps[0].finished_at == NOW
