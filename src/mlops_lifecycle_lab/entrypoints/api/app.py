"""FastAPI application factory for the serving API (S8, D13).

The factory takes the composition root, builds one `PredictionService` bound to the
hot-reloading `ModelHolder`, installs request-id middleware and typed exception handlers,
and keeps the app free of global state so tests can build isolated instances.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from mlops_lifecycle_lab import __version__
from mlops_lifecycle_lab.application.services.scoring import PredictionService
from mlops_lifecycle_lab.config import apply_privacy_defaults
from mlops_lifecycle_lab.domain.errors import NoChampionError, ValidationFailedError
from mlops_lifecycle_lab.entrypoints.api import deps
from mlops_lifecycle_lab.entrypoints.api.metrics import ApiMetrics
from mlops_lifecycle_lab.entrypoints.api.middleware import RequestIDMiddleware
from mlops_lifecycle_lab.entrypoints.api.model_holder import ModelHolder
from mlops_lifecycle_lab.entrypoints.api.routes import router


async def _on_no_champion(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "no champion loaded"},
    )


async def _on_validation_failed(request: Request, exc: Exception) -> JSONResponse:
    failed = list(exc.failed_rules) if isinstance(exc, ValidationFailedError) else []
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": "batch failed the data contract",
            "failed_rules": failed,
        },
    )


def create_app(container: deps.ApiContainer) -> FastAPI:
    """Build a fully wired serving application around an existing container."""
    apply_privacy_defaults()
    metrics = ApiMetrics()
    holder = ModelHolder(
        container.registry,
        metrics,
        container.clock,
        interval_seconds=container.settings.reload_interval_seconds,
    )
    service = PredictionService(
        container.registry,
        container.prediction_log,
        container.clock,
        model_name=container.settings.model_name,
        serving_model=holder,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await holder.start()
        try:
            yield
        finally:
            await holder.stop()

    app = FastAPI(
        title="mlops-lifecycle-lab serving API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.container = container
    app.state.metrics = metrics
    app.state.model_holder = holder
    app.state.prediction_service = service
    app.add_middleware(RequestIDMiddleware)
    app.add_exception_handler(NoChampionError, _on_no_champion)
    app.add_exception_handler(ValidationFailedError, _on_validation_failed)
    app.include_router(router)
    return app


__all__ = ["create_app"]
