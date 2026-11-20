"""Serving API routes: readiness, validation, request ids, reload auth and metrics."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.entrypoints.api.app import create_app
from mlops_lifecycle_lab.entrypoints.api.schemas import PredictRequest, raw_frame
from tests.unit.entrypoints.conftest import (
    ApiSettings,
    FakeApiContainer,
    predict_payload,
    register_champion,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    container = FakeApiContainer()
    register_champion(container.registry, version=1, prediction=123.0)
    with TestClient(create_app(container)) as test_client:  # type: ignore[arg-type]
        yield test_client


@pytest.fixture
def empty_client() -> Iterator[TestClient]:
    with TestClient(create_app(FakeApiContainer())) as test_client:  # type: ignore[arg-type]
        yield test_client


def test_healthz_is_always_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_is_503_without_a_champion(empty_client: TestClient) -> None:
    assert empty_client.get("/readyz").status_code == 503


def test_readyz_is_200_with_a_loaded_champion(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict_returns_a_non_negative_prediction_and_a_request_id(client: TestClient) -> None:
    response = client.post("/v1/predict", json=predict_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] >= 0
    assert body["model_version"] == 1
    assert body["model_name"] == "bike-demand"
    assert response.headers["X-Request-ID"]


def test_predict_echoes_a_supplied_request_id(client: TestClient) -> None:
    response = client.post(
        "/v1/predict", json=predict_payload(), headers={"X-Request-ID": "req-42"}
    )
    assert response.headers["X-Request-ID"] == "req-42"


def test_predict_returns_503_without_a_champion(empty_client: TestClient) -> None:
    response = empty_client.post("/v1/predict", json=predict_payload())
    assert response.status_code == 503
    assert response.json()["detail"] == "no champion loaded"


@pytest.mark.parametrize(
    "payload",
    [
        predict_payload(temp=1.01),
        predict_payload(season=5),
        {**predict_payload(), "extra": 1},
        predict_payload(timestamp="2012-01-01T00:30:00"),
    ],
)
def test_predict_rejects_invalid_payloads_with_422(
    client: TestClient, payload: dict[str, Any]
) -> None:
    assert client.post("/v1/predict", json=payload).status_code == 422


def test_batch_predict_rejects_more_than_one_thousand_rows(client: TestClient) -> None:
    response = client.post("/v1/predict/batch", json={"rows": [predict_payload()] * 1001})
    assert response.status_code == 422


def test_batch_predict_scores_and_labels_rows(client: TestClient) -> None:
    rows = [
        predict_payload(entity_id=1, timestamp="2012-01-01T00:00:00"),
        predict_payload(entity_id=2, timestamp="2012-01-01T01:00:00"),
    ]
    response = client.post("/v1/predict/batch", json={"rows": rows})
    assert response.status_code == 200
    assert len(response.json()) == 2

    labels = {
        "labels": [
            {"entity_id": 1, "label": 10.0, "label_ts": "2012-01-01T01:00:00"},
            {"entity_id": 2, "label": 20.0, "label_ts": "2012-01-01T02:00:00"},
        ]
    }
    assert client.post("/v1/labels", json=labels).status_code == 200


def test_api_feature_path_matches_build_features(client: TestClient) -> None:
    payload = predict_payload()
    assert client.post("/v1/predict", json=payload).status_code == 200

    log = client.app.state.container.prediction_log
    (record,) = log.records.values()
    expected = build_features(raw_frame([PredictRequest.model_validate(payload)])).iloc[0]
    for column in expected.index:
        assert record.features[column] == pytest.approx(float(expected[column]))


def test_metrics_endpoint_exposes_the_predictions_counter(client: TestClient) -> None:
    client.post("/v1/predict", json=predict_payload())
    text = client.get("/metrics").text
    assert 'mll_predictions_total{model_version="1"} 1.0' in text


def test_admin_reload_requires_the_configured_key() -> None:
    container = FakeApiContainer(settings=ApiSettings(admin_api_key=SecretStr("top-secret")))
    register_champion(container.registry, version=1)
    with TestClient(create_app(container)) as client:  # type: ignore[arg-type]
        assert client.post("/v1/admin/reload").status_code == 401
        assert client.post("/v1/admin/reload", headers={"X-API-Key": "wrong"}).status_code == 401
        response = client.post("/v1/admin/reload", headers={"X-API-Key": "top-secret"})
        assert response.status_code == 200
        assert response.json()["version"] == 1
