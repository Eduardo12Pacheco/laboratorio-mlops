"""HTTP routes of the serving API (S8): lifecycle probes, predictions, labels, metrics."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from mlops_lifecycle_lab.application.services.scoring import PredictionService
from mlops_lifecycle_lab.entrypoints.api.deps import (
    ApiContainer,
    get_container,
    get_holder,
    get_metrics,
    get_prediction_service,
    require_admin_api_key,
)
from mlops_lifecycle_lab.entrypoints.api.metrics import ApiMetrics
from mlops_lifecycle_lab.entrypoints.api.model_holder import ModelHolder
from mlops_lifecycle_lab.entrypoints.api.schemas import (
    BatchPredictRequest,
    LabelsRequest,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    label_records,
    raw_frame,
)

router = APIRouter()


def _model_info(holder: ModelHolder) -> ModelInfoResponse:
    serving = holder.current()
    if serving is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no champion loaded",
        )
    model = serving.model
    return ModelInfoResponse(
        name=model.name,
        version=model.version,
        stage=model.stage.value,
        loaded_at=holder.loaded_at or model.created_at,
        metrics=dict(model.metrics),
        train_window={
            "start": model.train_window.start,
            "end": model.train_window.end,
        },
        data_fingerprint=model.data_fingerprint,
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is up, regardless of whether a model is loaded."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(holder: Annotated[ModelHolder, Depends(get_holder)]) -> Response:
    """Readiness: 200 only when a champion artifact is loaded."""
    if not holder.is_ready:
        return Response(
            content='{"detail":"no champion loaded"}',
            media_type="application/json",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(
        content='{"status":"ready"}',
        media_type="application/json",
        status_code=status.HTTP_200_OK,
    )


@router.get("/v1/model", response_model=ModelInfoResponse)
def model_info(holder: Annotated[ModelHolder, Depends(get_holder)]) -> ModelInfoResponse:
    """Return the model version currently served."""
    return _model_info(holder)


@router.post("/v1/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    request: Request,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
    metrics: Annotated[ApiMetrics, Depends(get_metrics)],
) -> PredictResponse:
    """Score one raw row online and persist the prediction."""
    started = time.perf_counter()
    (result,) = service.predict(
        raw_frame([payload]),
        source="online",
        request_id=request.state.request_id,
    )
    metrics.observe_latency(time.perf_counter() - started)
    metrics.record_predictions(result.model_version)
    return PredictResponse.from_result(result)


@router.post("/v1/predict/batch", response_model=list[PredictResponse])
def predict_batch(
    payload: BatchPredictRequest,
    request: Request,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
    metrics: Annotated[ApiMetrics, Depends(get_metrics)],
) -> list[PredictResponse]:
    """Score a bounded batch of raw rows and persist every prediction."""
    started = time.perf_counter()
    results = service.predict(
        raw_frame(payload.rows),
        source="batch",
        request_id=request.state.request_id,
    )
    metrics.observe_latency(time.perf_counter() - started)
    metrics.record_predictions(results[0].model_version, count=len(results))
    return [PredictResponse.from_result(result) for result in results]


@router.post("/v1/labels")
def submit_labels(
    payload: LabelsRequest,
    container: Annotated[ApiContainer, Depends(get_container)],
) -> dict[str, int]:
    """Attach delayed ground-truth labels to previously scored entities."""
    labels = label_records(payload.labels)
    container.prediction_log.attach_labels(labels)
    return {"attached": len(labels)}


@router.post(
    "/v1/admin/reload",
    response_model=ModelInfoResponse,
    dependencies=[Depends(require_admin_api_key)],
)
async def admin_reload(
    holder: Annotated[ModelHolder, Depends(get_holder)],
) -> ModelInfoResponse:
    """Force a registry poll now instead of waiting for the next interval."""
    await asyncio.to_thread(holder.refresh, force=True)
    return _model_info(holder)


@router.get("/metrics")
def metrics_endpoint(metrics: Annotated[ApiMetrics, Depends(get_metrics)]) -> Response:
    """Prometheus text exposition (no separate Prometheus server is shipped)."""
    return Response(
        content=metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


__all__ = ["router"]
