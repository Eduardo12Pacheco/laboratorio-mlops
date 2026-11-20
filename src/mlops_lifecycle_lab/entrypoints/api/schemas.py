"""Pydantic v2 request/response schemas of the serving API (S8).

The request schema mirrors the raw dataset columns with the same numeric domains as the
domain validation engine. `raw_frame` is the single conversion from typed requests to
the raw DataFrame that `PredictionService` feeds into `build_features`, which keeps the
online path and the training/batch path on the same feature implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mlops_lifecycle_lab.application.dto import LabelRecord, PredictionResult
from mlops_lifecycle_lab.domain.schema import RAW_COLUMNS


class PredictRequest(BaseModel):
    """One raw row to score; rejects unknown fields and out-of-domain values."""

    model_config = ConfigDict(extra="forbid")

    entity_id: int = Field(ge=0)
    timestamp: datetime
    season: int = Field(ge=1, le=4)
    holiday: Literal[0, 1]
    workingday: Literal[0, 1]
    weathersit: int = Field(ge=1, le=4)
    temp: float = Field(ge=0.0, le=1.0)
    atemp: float = Field(ge=0.0, le=1.0)
    hum: float = Field(ge=0.0, le=1.0)
    windspeed: float = Field(ge=0.0, le=1.0)

    @field_validator("timestamp")
    @classmethod
    def _hourly_naive(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError("timestamp must be naive (dataset timezone-less UTC)")
        if value.minute or value.second or value.microsecond:
            raise ValueError("timestamp must be hourly aligned")
        return value


class PredictResponse(BaseModel):
    """One prediction returned to an API caller."""

    entity_id: int
    prediction: float = Field(ge=0.0)
    model_name: str
    model_version: int = Field(ge=1)
    scored_at: datetime

    @classmethod
    def from_result(cls, result: PredictionResult) -> PredictResponse:
        """Build a response from the application DTO."""
        return cls(
            entity_id=result.entity_id,
            prediction=result.prediction,
            model_name=result.model_name,
            model_version=result.model_version,
            scored_at=result.scored_at,
        )


class BatchPredictRequest(BaseModel):
    """A bounded batch of raw rows (at most 1000, per S8)."""

    model_config = ConfigDict(extra="forbid")

    rows: list[PredictRequest] = Field(min_length=1, max_length=1000)


class LabelItem(BaseModel):
    """A delayed ground-truth label for one previously scored entity."""

    model_config = ConfigDict(extra="forbid")

    entity_id: int = Field(ge=0)
    label: float = Field(ge=0.0)
    label_ts: datetime


class LabelsRequest(BaseModel):
    """The labels that became available at the end of a replay week."""

    model_config = ConfigDict(extra="forbid")

    labels: list[LabelItem] = Field(min_length=1)


class ModelInfoResponse(BaseModel):
    """The model currently served, with the lineage the dashboard/ops need."""

    name: str
    version: int
    stage: str
    loaded_at: datetime
    metrics: dict[str, float]
    train_window: dict[str, datetime]
    data_fingerprint: str


def raw_frame(rows: Sequence[PredictRequest]) -> pd.DataFrame:
    """Convert typed requests into the raw DataFrame contract, in column order."""
    records = [{name: getattr(row, name) for name in RAW_COLUMNS} for row in rows]
    frame = pd.DataFrame(records, columns=list(RAW_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    for column in ("entity_id", "season", "holiday", "workingday", "weathersit"):
        frame[column] = frame[column].astype("int64")
    for column in ("temp", "atemp", "hum", "windspeed"):
        frame[column] = frame[column].astype("float64")
    return frame


def label_records(items: Sequence[LabelItem]) -> list[LabelRecord]:
    """Convert request items into application `LabelRecord`s."""
    return [
        LabelRecord(entity_id=item.entity_id, label=item.label, label_ts=item.label_ts)
        for item in items
    ]


__all__ = [
    "BatchPredictRequest",
    "LabelItem",
    "LabelsRequest",
    "ModelInfoResponse",
    "PredictRequest",
    "PredictResponse",
    "label_records",
    "raw_frame",
]
