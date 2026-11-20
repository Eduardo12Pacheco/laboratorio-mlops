"""Real-stack serving: bootstrap v1 through LightGBM/MLflow/SQLite, then serve it (T8)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from mlops_lifecycle_lab.bootstrap.container import Container, build_container
from mlops_lifecycle_lab.config import Settings
from mlops_lifecycle_lab.domain.windows import TimeWindow
from mlops_lifecycle_lab.entrypoints.api.app import create_app

FIRST_WEEK = TimeWindow(datetime(2012, 1, 1), datetime(2012, 1, 8))
BATCH_END = datetime(2012, 1, 10)


def raw_payload(row: pd.Series) -> dict[str, object]:
    """Convert one raw-schema row into the API JSON payload."""
    return {
        "entity_id": int(row["entity_id"]),
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        "season": int(row["season"]),
        "holiday": int(row["holiday"]),
        "workingday": int(row["workingday"]),
        "weathersit": int(row["weathersit"]),
        "temp": float(row["temp"]),
        "atemp": float(row["atemp"]),
        "hum": float(row["hum"]),
        "windspeed": float(row["windspeed"]),
    }


def count_predictions(container: Container) -> int:
    with container.database.transaction() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])


def test_real_stack_serves_the_bootstrapped_champion(tmp_path: Path) -> None:
    container = build_container(Settings(workspace=tmp_path, fast=True))
    try:
        champion = container.bootstrap.execute()
        assert champion.version == 1

        with TestClient(create_app(container)) as client:
            ready = client.get("/readyz")
            assert ready.status_code == 200

            raw = container.dataset.raw()
            first = raw.loc[raw["timestamp"] == datetime(2012, 1, 1, 0, 0)].iloc[0]
            response = client.post("/v1/predict", json=raw_payload(first))
            assert response.status_code == 200
            body = response.json()
            assert body["model_version"] == 1
            assert body["prediction"] >= 0

            # The dataset is genuinely missing one hour in this week, so take the 168
            # contiguous rows starting at 2012-01-01 00:00 (the API batch cap is 1000).
            start = int(raw.index[raw["timestamp"] == datetime(2012, 1, 1)][0])
            batch = raw.iloc[start + 1 : start + 169]
            payload = {"rows": [raw_payload(row) for _, row in batch.iterrows()]}
            assert len(payload["rows"]) == 168

            scored = client.post("/v1/predict/batch", json=payload)
            assert scored.status_code == 200
            assert len(scored.json()) == 168
            # The online prediction above covers the first row; the batch is disjoint.
            assert count_predictions(container) == 1 + 168
            with container.database.transaction() as connection:
                batch_rows_count = connection.execute(
                    "SELECT COUNT(*) FROM predictions WHERE source = 'batch'"
                ).fetchone()[0]
            assert int(batch_rows_count) == 168

            assert client.post("/v1/predict/batch", json=payload).status_code == 200
            assert count_predictions(container) == 1 + 168

            labels = {
                "labels": [
                    {
                        "entity_id": int(row["entity_id"]),
                        "label": float(row["cnt"]),
                        "label_ts": (
                            pd.Timestamp(row["timestamp"]) + pd.Timedelta(hours=1)
                        ).isoformat(),
                    }
                    for _, row in batch.iterrows()
                ]
            }
            assert client.post("/v1/labels", json=labels).status_code == 200
            labeled = container.prediction_log.labeled_predictions(FIRST_WEEK.start, BATCH_END)
            assert len(labeled) == 168
    finally:
        container.close()


def test_model_endpoint_reports_lineage_of_the_loaded_champion(tmp_path: Path) -> None:
    container = build_container(Settings(workspace=tmp_path, fast=True))
    try:
        container.bootstrap.execute()
        with TestClient(create_app(container)) as client:
            body = client.get("/v1/model").json()
        assert body["name"] == "bike-demand"
        assert body["version"] == 1
        assert body["stage"] == "CHAMPION"
        assert body["data_fingerprint"]
        assert body["train_window"]["start"].startswith("2011-01-01")
        assert "rmse" in body["metrics"]
    finally:
        container.close()
