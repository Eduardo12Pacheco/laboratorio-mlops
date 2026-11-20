"""HTTP `ScoringGateway` used by the scheduler and drills in container mode (S14, D16).

The adapter speaks the same JSON API a human would call, splits batches at the API cap,
retries transient connection/5xx failures with exponential backoff and translates a 422
into the domain `ValidationFailedError` so a quarantined batch behaves identically in
in-process and HTTP mode.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from datetime import datetime
from typing import Any, cast

import httpx
import pandas as pd

from mlops_lifecycle_lab.application.dto import LabelRecord, PredictionResult
from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.schema import RAW_COLUMNS

MAX_BATCH_ROWS = 1000


class GatewayUnavailableError(RuntimeError):
    """The serving API stayed unavailable after every retry."""


def _row_payload(row: Mapping[Hashable, Any]) -> dict[str, object]:
    return {
        "entity_id": int(row["entity_id"]),
        "timestamp": pd.Timestamp(cast(Any, row["timestamp"])).isoformat(),
        "season": int(row["season"]),
        "holiday": int(row["holiday"]),
        "workingday": int(row["workingday"]),
        "weathersit": int(row["weathersit"]),
        "temp": float(row["temp"]),
        "atemp": float(row["atemp"]),
        "hum": float(row["hum"]),
        "windspeed": float(row["windspeed"]),
    }


def _label_payload(label: LabelRecord) -> dict[str, object]:
    return {
        "entity_id": label.entity_id,
        "label": label.label,
        "label_ts": label.label_ts.isoformat(),
    }


class HttpScoringGateway:
    """`ScoringGateway` over the FastAPI serving endpoints."""

    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        *,
        max_attempts: int = 3,
        backoff_seconds: float = 0.1,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url, timeout=30.0)
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def close(self) -> None:
        """Close the underlying HTTP client when this gateway created it."""
        if self._owns_client:
            self._client.close()

    def predict_batch(self, raw_rows: pd.DataFrame) -> list[PredictionResult]:
        """Score every row, splitting the frame into API-sized chunks."""
        rows = raw_rows.loc[:, list(RAW_COLUMNS)]
        results: list[PredictionResult] = []
        for start in range(0, len(rows), MAX_BATCH_ROWS):
            chunk = rows.iloc[start : start + MAX_BATCH_ROWS]
            payload = {"rows": [_row_payload(record) for record in chunk.to_dict("records")]}
            response = self._post("/v1/predict/batch", payload)
            results.extend(self._parse_predictions(response.json()))
        return results

    def submit_labels(self, labels: Sequence[LabelRecord]) -> None:
        """Submit delayed labels; an empty sequence is a no-op."""
        if not labels:
            return
        self._post("/v1/labels", {"labels": [_label_payload(label) for label in labels]})

    def _post(self, path: str, payload: Mapping[str, object]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.post(path, json=payload)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code == 422:
                    raise ValidationFailedError(self._failed_rules(response))
                if response.status_code < 500:
                    response.raise_for_status()
                    return response
                last_error = GatewayUnavailableError(
                    f"{path} failed with HTTP {response.status_code}"
                )
            if attempt < self._max_attempts:
                self._sleep(self._backoff_seconds * (2 ** (attempt - 1)))
        raise GatewayUnavailableError(
            f"{path} failed after {self._max_attempts} attempts"
        ) from last_error

    @staticmethod
    def _failed_rules(response: httpx.Response) -> tuple[str, ...]:
        try:
            body = response.json()
        except ValueError:
            return ()
        if not isinstance(body, dict):
            return ()
        rules = body.get("failed_rules") or []
        return tuple(str(rule) for rule in rules)

    @staticmethod
    def _parse_predictions(body: object) -> list[PredictionResult]:
        if not isinstance(body, list):
            raise GatewayUnavailableError("serving API returned a non-list payload")
        parsed: list[PredictionResult] = []
        for item in body:
            if not isinstance(item, dict):
                raise GatewayUnavailableError("serving API returned a malformed prediction")
            parsed.append(
                PredictionResult(
                    entity_id=int(item["entity_id"]),
                    prediction=float(item["prediction"]),
                    model_name=str(item["model_name"]),
                    model_version=int(item["model_version"]),
                    scored_at=datetime.fromisoformat(str(item["scored_at"])),
                )
            )
        return parsed


__all__ = ["MAX_BATCH_ROWS", "GatewayUnavailableError", "HttpScoringGateway"]
