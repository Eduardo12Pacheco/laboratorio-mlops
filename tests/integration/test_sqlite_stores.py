"""Integration tests for the SQLite operational store (S9, S14, D11).

The stores share one database file, exactly like the composition root does. The tests
pin the pragmas, the ordered migrations, the idempotent transaction boundary and the
repository behaviour that monitoring and the dashboard rely on.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset
from mlops_lifecycle_lab.adapters.storage.event_store import SqliteEventStore
from mlops_lifecycle_lab.adapters.storage.kv_store import SqliteKeyValueStore
from mlops_lifecycle_lab.adapters.storage.labeled_data import CompositeLabeledData
from mlops_lifecycle_lab.adapters.storage.prediction_log import SqlitePredictionLog
from mlops_lifecycle_lab.adapters.storage.run_ledger import SqliteRunLedger
from mlops_lifecycle_lab.adapters.storage.snapshot_store import SqliteSnapshotStore
from mlops_lifecycle_lab.adapters.storage.sqlite import MIGRATIONS, SqliteDatabase, connect
from mlops_lifecycle_lab.application.dto import (
    LabelRecord,
    MonitoringSnapshot,
    PredictionRecord,
    RunStatus,
    StepStatus,
)
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS, LABEL_COLUMNS, RAW_COLUMNS
from mlops_lifecycle_lab.domain.validation import Verdict, validate_frame

START = datetime(2011, 1, 1)
PRODUCTION_START = datetime(2012, 1, 1)
BASE_FEATURES: dict[str, float] = {
    "season": 1.0,
    "mnth": 1.0,
    "hr": 0.0,
    "holiday": 0.0,
    "weekday": 6.0,
    "workingday": 0.0,
    "weathersit": 1.0,
    "temp": 0.3,
    "atemp": 0.35,
    "hum": 0.5,
    "windspeed": 0.2,
    "t_days": 365.0,
}


@pytest.fixture
def database(tmp_path: Path) -> Iterator[SqliteDatabase]:
    """A migrated database file in a temporary workspace."""
    db = SqliteDatabase(tmp_path / "mll.db")
    yield db
    db.close()


def test_connect_sets_the_documented_pragmas(tmp_path: Path) -> None:
    connection = connect(tmp_path / "mll.db")
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_migrate_is_ordered_and_idempotent(database: SqliteDatabase) -> None:
    applied = database.connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()

    assert [row[0] for row in applied] == [migration.version for migration in MIGRATIONS]
    database.connection.execute("SELECT 1")
    from mlops_lifecycle_lab.adapters.storage.sqlite import migrate

    migrate(database.connection)
    assert (
        database.connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        == len(MIGRATIONS)
    )


def test_transaction_rolls_back_on_error(database: SqliteDatabase) -> None:
    with pytest.raises(RuntimeError):
        with database.transaction():
            database.connection.execute(
                "INSERT INTO kv(key, value_json) VALUES (?, ?)", ("boom", "1")
            )
            raise RuntimeError("abort")

    assert database.connection.execute("SELECT COUNT(*) FROM kv").fetchone()[0] == 0


def make_event(tick: int, event_type: EventType = EventType.BATCH_VALIDATED) -> LifecycleEvent:
    """Build one audit event with a nested JSON payload."""
    return LifecycleEvent(
        ts=START + timedelta(days=tick),
        type=event_type,
        tick=tick,
        model_version=2 if tick > 1 else None,
        payload={"verdict": "PASS", "details": {"rows": 10 + tick}},
    )


def test_event_store_round_trips_and_filters(database: SqliteDatabase) -> None:
    store = SqliteEventStore(database)
    first = make_event(1)
    second = make_event(2, EventType.GATES_EVALUATED)
    third = make_event(3, EventType.GATES_EVALUATED)
    for event in (first, second, third):
        store.append(event)

    assert store.list() == [first, second, third]
    assert store.list(event_type=EventType.GATES_EVALUATED) == [second, third]
    assert store.list(model_version=2) == [second, third]
    assert store.list(event_type=EventType.GATES_EVALUATED, limit=1) == [third]
    assert store.list(limit=2) == [second, third]


def snapshot(tick: int, verdict: str = "PASS") -> MonitoringSnapshot:
    """Build a monitoring snapshot for one tick."""
    return MonitoringSnapshot(
        tick=tick,
        ts=PRODUCTION_START + timedelta(days=7 * tick),
        model_version=1,
        n_rows=100 + tick,
        rmse=50.0 + tick,
        mae=30.0,
        wape=0.2,
        bias=1.0,
        numeric_drift_share=0.5,
        prediction_psi=0.21,
        psi={"temp": 0.3, "hum": 0.1},
        verdict=verdict,
    )


def test_snapshot_store_replaces_a_tick_and_keeps_order(database: SqliteDatabase) -> None:
    store = SqliteSnapshotStore(database)
    store.save(snapshot(2))
    store.save(snapshot(1))
    store.save(snapshot(2, verdict="WARN"))

    snapshots = store.list()

    assert [entry.tick for entry in snapshots] == [1, 2]
    assert snapshots[1].verdict == "WARN"
    assert snapshots[1].psi == {"temp": 0.3, "hum": 0.1}
    assert store.list(limit=1)[0].tick == 2


def test_key_value_store_round_trips_json_and_clears(database: SqliteDatabase) -> None:
    store = SqliteKeyValueStore(database)

    assert store.get("missing") is None
    store.set("probation_until", "2012-04-22T00:00:00")
    store.set("counter", {"value": 3})
    assert store.get("probation_until") == "2012-04-22T00:00:00"
    assert store.get("counter") == {"value": 3}

    store.set("counter", None)
    assert store.get("counter") is None


def test_run_ledger_records_runs_and_every_attempt(database: SqliteDatabase) -> None:
    ledger = SqliteRunLedger(database)
    ledger.start("run-1", "tick", 3, START)
    ledger.finish_step(
        "run-1",
        0,
        "validate",
        StepStatus.FAILED,
        START,
        START + timedelta(seconds=1),
        "transient",
    )
    ledger.finish_step(
        "run-1",
        1,
        "validate",
        StepStatus.SUCCEEDED,
        START + timedelta(seconds=1),
        START + timedelta(seconds=2),
    )
    ledger.finish("run-1", RunStatus.SUCCEEDED, START + timedelta(seconds=3))

    run = database.connection.execute("SELECT * FROM runs WHERE run_id = 'run-1'").fetchone()
    assert run["name"] == "tick"
    assert run["tick"] == 3
    assert run["status"] == "SUCCEEDED"
    steps = database.connection.execute(
        "SELECT idx, status, error FROM run_steps WHERE run_id = 'run-1' ORDER BY idx"
    ).fetchall()
    assert [(row["idx"], row["status"], row["error"]) for row in steps] == [
        (0, "FAILED", "transient"),
        (1, "SUCCEEDED", None),
    ]


def prediction(entity_id: int, version: int = 1) -> PredictionRecord:
    """Build one production prediction in the 2012 replay half."""
    return PredictionRecord(
        prediction_id=f"pred-{version}-{entity_id}",
        entity_id=entity_id,
        event_ts=PRODUCTION_START + timedelta(hours=entity_id),
        scored_at=PRODUCTION_START + timedelta(hours=entity_id),
        model_name="bike-demand",
        model_version=version,
        features=BASE_FEATURES,
        prediction=10.0,
        source="batch",
    )


def test_composite_labeled_data_merges_archive_and_production(
    database: SqliteDatabase,
) -> None:
    log = SqlitePredictionLog(database)
    log.append([prediction(1), prediction(2)])
    log.attach_labels(
        [
            LabelRecord(entity_id=1, label=11.0, label_ts=PRODUCTION_START),
            LabelRecord(entity_id=2, label=22.0, label_ts=PRODUCTION_START + timedelta(hours=1)),
        ]
    )
    repository = CompositeLabeledData(BundledDataset(), log)

    frame = repository.training_frame(PRODUCTION_START + timedelta(days=1))

    assert list(frame.columns) == [*RAW_COLUMNS, *LABEL_COLUMNS]
    assert len(frame) == 8760 + 2
    production = frame.loc[frame["timestamp"] >= PRODUCTION_START]
    assert list(production["entity_id"]) == [1, 2]
    assert production["cnt"].tolist() == [11.0, 22.0]
    assert frame["timestamp"].is_monotonic_increasing
    assert validate_frame(frame, context="training").verdict is Verdict.WARN


def test_composite_labeled_data_lets_the_archive_win_duplicates(
    database: SqliteDatabase,
) -> None:
    log = SqlitePredictionLog(database)
    log.append([prediction(1)])
    log.attach_labels([LabelRecord(entity_id=1, label=999.0, label_ts=PRODUCTION_START)])
    repository = CompositeLabeledData(BundledDataset(), log)

    frame = repository.training_frame(PRODUCTION_START + timedelta(days=1))

    duplicate = frame.loc[frame["entity_id"] == 1]
    assert len(duplicate) == 1
    assert duplicate.iloc[0]["cnt"] == 16
    assert duplicate.iloc[0]["timestamp"] < PRODUCTION_START


def test_composite_labeled_data_is_end_exclusive_and_typed_when_empty(
    database: SqliteDatabase,
) -> None:
    log = SqlitePredictionLog(database)
    repository = CompositeLabeledData(BundledDataset(), log)

    frame = repository.training_frame(START)

    assert frame.empty
    assert list(frame.columns) == [*RAW_COLUMNS, *LABEL_COLUMNS]
    assert list(FEATURE_COLUMNS) == [
        column for column in frame.columns if column in FEATURE_COLUMNS
    ]
    assert isinstance(frame, pd.DataFrame)


def test_prediction_log_keeps_the_documented_feature_round_trip(
    database: SqliteDatabase,
) -> None:
    log = SqlitePredictionLog(database)
    record = prediction(5)
    log.append([record])
    log.attach_labels([LabelRecord(entity_id=5, label=12.0, label_ts=record.event_ts)])

    labeled = log.labeled_predictions(PRODUCTION_START, PRODUCTION_START + timedelta(days=1))

    assert labeled.iloc[0]["prediction_id"] == record.prediction_id
    for feature in FEATURE_COLUMNS:
        assert float(labeled.iloc[0][feature]) == float(BASE_FEATURES[feature])
    assert set(labeled["event_ts"]) == {record.event_ts}
