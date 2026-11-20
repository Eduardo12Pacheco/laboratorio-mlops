"""Column contracts and calendar constants.

One module owns every column name so training, serving, validation and monitoring can
never drift apart. The tuples are ordered: `build_features` and the bundled dataset
rely on that order, and tests assert it.
"""

from __future__ import annotations

from datetime import datetime

RAW_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "timestamp",
    "season",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
)
"""Columns accepted by the serving API and produced by the batch source."""

LABEL_COLUMNS: tuple[str, ...] = ("casual", "registered", "cnt")
"""Label columns; `cnt` is the target and the other two only exist in the archive."""

FEATURE_COLUMNS: tuple[str, ...] = (
    "season",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "t_days",
)
"""Exactly the twelve model inputs, in the order the trainer and predictor expect."""

TARGET_COLUMN = "cnt"
"""The regression target: hourly total rentals."""

LEAKAGE_COLUMNS: tuple[str, ...] = ("yr", "instant", "casual", "registered", "cnt")
"""Columns that must never reach the feature frame (time proxies or label leakage)."""

TREND_EPOCH = datetime(2011, 1, 1)
"""Origin of the fractional-days trend feature `t_days`."""

FEATURE_SPEC_VERSION = "1"
"""Version of the feature definition; logged with every run for lineage."""

SEGMENT_COLUMN = "workingday"
"""Segment used by the fairness-of-quality promotion gate."""

MONITORED_NUMERIC: tuple[str, ...] = ("temp", "atemp", "hum", "windspeed")
"""Numeric features with a PSI drift statistic."""

MONITORED_CATEGORICAL: tuple[str, ...] = ("season", "weathersit")
"""Categorical features with a PSI drift statistic."""

ARCHIVE_END = datetime(2012, 1, 1)
"""The bundled archive contains 2011 only; the 2012 half is replayed as production."""
