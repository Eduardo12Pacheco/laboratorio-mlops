"""FastAPI dependencies: typed access to the app state built by `create_app`."""

from __future__ import annotations

import secrets
from typing import Annotated, Protocol

from fastapi import Header, HTTPException, Request, status
from pydantic import SecretStr

from mlops_lifecycle_lab.application.ports import Clock, ModelRegistry, PredictionLog
from mlops_lifecycle_lab.application.services.scoring import PredictionService
from mlops_lifecycle_lab.bootstrap.container import ContainerSettings
from mlops_lifecycle_lab.entrypoints.api.metrics import ApiMetrics
from mlops_lifecycle_lab.entrypoints.api.model_holder import ModelHolder


class ApiContainer(Protocol):
    """Composition-root subset accepted by `create_app`; `Container` satisfies it."""

    settings: ContainerSettings
    clock: Clock
    registry: ModelRegistry
    prediction_log: PredictionLog


def get_container(request: Request) -> ApiContainer:
    """Return the composition root stored on the app state."""
    return request.app.state.container  # type: ignore[no-any-return]


def get_holder(request: Request) -> ModelHolder:
    """Return the process-wide model holder."""
    return request.app.state.model_holder  # type: ignore[no-any-return]


def get_metrics(request: Request) -> ApiMetrics:
    """Return the per-app Prometheus registry."""
    return request.app.state.metrics  # type: ignore[no-any-return]


def get_prediction_service(request: Request) -> PredictionService:
    """Return the serving service bound to the model holder."""
    return request.app.state.prediction_service  # type: ignore[no-any-return]


def require_admin_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Enforce `MLL_ADMIN_API_KEY` when configured; open on loopback otherwise."""
    configured: SecretStr | None = request.app.state.container.settings.admin_api_key
    if configured is None:
        return
    expected = configured.get_secret_value()
    supplied = x_api_key or ""
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid admin api key",
        )


__all__ = [
    "ApiContainer",
    "get_container",
    "get_holder",
    "get_metrics",
    "get_prediction_service",
    "require_admin_api_key",
]
