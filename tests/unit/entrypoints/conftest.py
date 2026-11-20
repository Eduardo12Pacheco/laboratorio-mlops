"""Shared fixtures and lightweight doubles for the entrypoint (API) unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import SecretStr

from mlops_lifecycle_lab.application.dto import encode_version_tags
from mlops_lifecycle_lab.domain.lifecycle import ModelStage, ModelVersion
from mlops_lifecycle_lab.domain.windows import TimeWindow
from tests.fakes.clock import FakeClock
from tests.fakes.prediction_log import InMemoryPredictionLog
from tests.fakes.registry import InMemoryRegistry
from tests.fakes.trainer import MeanPredictor

TRAIN_WINDOW = TimeWindow(datetime(2011, 1, 1), datetime(2011, 10, 1))


@dataclass
class ApiSettings:
    """The subset of settings the API factory reads."""

    model_name: str = "bike-demand"
    admin_api_key: SecretStr | None = None
    reload_interval_seconds: float = 3600.0


@dataclass
class FakeApiContainer:
    """Structural stand-in for the composition root accepted by `create_app`."""

    settings: ApiSettings = field(default_factory=ApiSettings)
    clock: FakeClock = field(default_factory=lambda: FakeClock(datetime(2012, 1, 1)))
    registry: InMemoryRegistry = field(default_factory=InMemoryRegistry)
    prediction_log: InMemoryPredictionLog = field(default_factory=InMemoryPredictionLog)


def model_version(version: int = 1, stage: ModelStage = ModelStage.CANDIDATE) -> ModelVersion:
    """A complete, valid model version ready to be tagged into a registry."""
    return ModelVersion(
        name="bike-demand",
        version=version,
        stage=stage,
        run_id=f"run-{version}",
        metrics={"rmse": 50.0, "wape": 0.22},
        train_window=TRAIN_WINDOW,
        holdout_window=None,
        data_fingerprint=f"fingerprint-{version}",
        feature_spec_version="1",
        created_at=datetime(2012, 1, 1),
        trigger_reason="MANUAL",
    )


def register_champion(
    registry: InMemoryRegistry,
    version: int = 1,
    prediction: float = 123.0,
) -> ModelVersion:
    """Adopt, attach a predictor and promote one version to champion."""
    registry.adopt_candidate(version, encode_version_tags(model_version(version)))
    registry.set_predictor(version, MeanPredictor(prediction))
    return registry.promote(version)


def predict_payload(**overrides: object) -> dict[str, object]:
    """A valid raw prediction payload for route tests, with optional overrides."""
    base: dict[str, object] = {
        "entity_id": 1,
        "timestamp": "2012-01-01T00:00:00",
        "season": 1,
        "holiday": 0,
        "workingday": 0,
        "weathersit": 1,
        "temp": 0.24,
        "atemp": 0.28,
        "hum": 0.81,
        "windspeed": 0.1,
    }
    base.update(overrides)
    return base
