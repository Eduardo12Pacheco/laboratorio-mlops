"""HTTP scoring gateway: chunking, retries, error mapping and label submission."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime

import httpx
import pandas as pd
import pytest

from mlops_lifecycle_lab.adapters.serving_client.http_gateway import (
    GatewayUnavailableError,
    HttpScoringGateway,
)
from mlops_lifecycle_lab.application.dto import LabelRecord
from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from tests.fakes.data import raw_frame


def prediction_payload(entity_id: int) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "prediction": 42.0,
        "model_name": "bike-demand",
        "model_version": 1,
        "scored_at": "2012-01-01T00:00:00",
    }


def make_gateway(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[HttpScoringGateway, httpx.Client]:
    client = httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(handler))
    gateway = HttpScoringGateway("http://api.test", client, sleep=lambda _seconds: None)
    return gateway, client


def test_retries_transient_failures_then_succeeds() -> None:
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        if len(attempts) < 3:
            return httpx.Response(503, json={"detail": "unavailable"})
        body = json.loads(request.content)
        payload = [prediction_payload(row["entity_id"]) for row in body["rows"]]
        return httpx.Response(200, json=payload)

    gateway, client = make_gateway(handler)
    try:
        results = gateway.predict_batch(raw_frame(periods=3))
    finally:
        client.close()

    assert len(attempts) == 3
    assert [result.entity_id for result in results] == [1, 2, 3]
    assert all(result.model_version == 1 for result in results)


def test_raises_after_three_failed_attempts() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"detail": "unavailable"})

    gateway, client = make_gateway(handler)
    try:
        with pytest.raises(GatewayUnavailableError):
            gateway.predict_batch(raw_frame(periods=1))
    finally:
        client.close()
    assert len(calls) == 3


def test_maps_422_to_validation_failed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"detail": "batch failed", "failed_rules": ["numeric_range"]},
        )

    gateway, client = make_gateway(handler)
    try:
        with pytest.raises(ValidationFailedError) as caught:
            gateway.predict_batch(raw_frame(periods=1))
    finally:
        client.close()
    assert caught.value.failed_rules == ("numeric_range",)


def test_splits_batches_larger_than_one_thousand_rows() -> None:
    requested_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_sizes.append(len(body["rows"]))
        return httpx.Response(
            200, json=[prediction_payload(row["entity_id"]) for row in body["rows"]]
        )

    gateway, client = make_gateway(handler)
    frame = raw_frame(periods=1000)
    extra = frame.iloc[[0]].copy()
    extra["entity_id"] = 5000
    bigger = pd.concat([frame, extra], ignore_index=True)
    try:
        results = gateway.predict_batch(bigger)
    finally:
        client.close()

    assert requested_sizes == [1000, 1]
    assert len(results) == 1001


def test_submits_labels_by_entity_id() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"attached": 1})

    gateway, client = make_gateway(handler)
    try:
        gateway.submit_labels(
            [LabelRecord(entity_id=1, label=12.0, label_ts=datetime(2012, 1, 1, 1))]
        )
    finally:
        client.close()

    assert captured == [
        {"labels": [{"entity_id": 1, "label": 12.0, "label_ts": "2012-01-01T01:00:00"}]}
    ]
