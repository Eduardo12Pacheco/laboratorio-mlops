"""Hot reload: a promoted champion is served within the reload interval (T8, S8)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from mlops_lifecycle_lab.application.ports import Predictor
from mlops_lifecycle_lab.domain.lifecycle import ModelVersion
from mlops_lifecycle_lab.entrypoints.api.app import create_app
from tests.fakes.registry import InMemoryRegistry
from tests.unit.entrypoints.conftest import (
    ApiSettings,
    FakeApiContainer,
    register_champion,
)


class FailingAfterFirstRead:
    """Registry proxy that works once, then fails every champion read."""

    def __init__(self, inner: InMemoryRegistry) -> None:
        self.inner = inner
        self.reads = 0

    def get_champion(self) -> ModelVersion | None:
        self.reads += 1
        if self.reads > 1:
            raise RuntimeError("registry unavailable")
        return self.inner.get_champion()

    def load_predictor(self, version: int) -> Predictor:
        return self.inner.load_predictor(version)


def make_container(interval: float = 0.2) -> FakeApiContainer:
    container = FakeApiContainer(settings=ApiSettings(reload_interval_seconds=interval))
    register_champion(container.registry, version=1, prediction=100.0)
    return container


def wait_for_version(client: TestClient, version: int, timeout: float = 3.0) -> int:
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        response = client.get("/v1/model")
        if response.status_code == 200:
            last = response.json()["version"]
            if last == version:
                return last
        time.sleep(0.05)
    return last


def test_promoted_champion_is_served_without_restart() -> None:
    container = make_container()
    with TestClient(create_app(container)) as client:  # type: ignore[arg-type]
        assert client.get("/v1/model").json()["version"] == 1
        register_champion(container.registry, version=2, prediction=200.0)
        assert wait_for_version(client, 2) == 2
        body = client.post(
            "/v1/predict",
            json={
                "entity_id": 1,
                "timestamp": "2012-01-01T00:00:00",
                "season": 1,
                "holiday": 0,
                "workingday": 0,
                "weathersit": 1,
                "temp": 0.2,
                "atemp": 0.2,
                "hum": 0.5,
                "windspeed": 0.1,
            },
        ).json()
        assert body["model_version"] == 2


def test_failing_reload_keeps_serving_and_counts_the_failure() -> None:
    inner = FakeApiContainer(settings=ApiSettings(reload_interval_seconds=0.05))
    register_champion(inner.registry, version=1, prediction=100.0)
    failing = FailingAfterFirstRead(inner.registry)
    container = FakeApiContainer(
        settings=inner.settings,
        registry=failing,  # type: ignore[arg-type]
        prediction_log=inner.prediction_log,
        clock=inner.clock,
    )
    with TestClient(create_app(container)) as client:  # type: ignore[arg-type]
        assert client.get("/readyz").status_code == 200
        deadline = time.monotonic() + 3.0
        text = ""
        while time.monotonic() < deadline:
            text = client.get("/metrics").text
            if "mll_model_reload_failures_total" in text:
                break
            time.sleep(0.05)
        assert "mll_model_reload_failures_total" in text
        response = client.post(
            "/v1/predict",
            json={
                "entity_id": 1,
                "timestamp": "2012-01-01T00:00:00",
                "season": 1,
                "holiday": 0,
                "workingday": 0,
                "weathersit": 1,
                "temp": 0.2,
                "atemp": 0.2,
                "hum": 0.5,
                "windspeed": 0.1,
            },
        )
        assert response.status_code == 200
        assert response.json()["model_version"] == 1
