"""Bundled UCI dataset adapter.

The wheel ships `_data/hour.csv.gz` plus `_data/manifest.json`; nothing is downloaded at
runtime. Every read path that matters for provenance (`verify`) hashes the decompressed
bytes, so a tampered bundle is refused instead of silently trained on.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_lifecycle_lab.domain.errors import ChecksumMismatchError
from mlops_lifecycle_lab.domain.schema import (
    ARCHIVE_END,
    LABEL_COLUMNS,
    RAW_COLUMNS,
)
from mlops_lifecycle_lab.domain.windows import TimeWindow

_UCI_COLUMNS: tuple[str, ...] = (
    "instant",
    "dteday",
    "season",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
)


def _package_data_dir() -> Path:
    return Path(str(files("mlops_lifecycle_lab"))) / "_data"


@dataclass(frozen=True, slots=True)
class VerifiedDataset:
    """Provenance of a bundle that passed checksum verification."""

    sha256: str
    size: int
    rows: int
    start: datetime
    end: datetime


def from_uci_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert original UCI columns into the project's raw schema.

    The conversion derives the observation timestamp from `dteday` + `hr` and reuses
    `instant` as the entity id. The input frame is never mutated.
    """
    missing = [column for column in _UCI_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"missing UCI columns: {', '.join(missing)}")

    timestamps = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(
        frame["hr"].astype("int64"), unit="h"
    )
    raw = pd.DataFrame(
        {
            "entity_id": frame["instant"].astype("int64"),
            "timestamp": timestamps,
            "season": frame["season"].astype("int64"),
            "holiday": frame["holiday"].astype("int64"),
            "workingday": frame["workingday"].astype("int64"),
            "weathersit": frame["weathersit"].astype("int64"),
            "temp": frame["temp"].astype("float64"),
            "atemp": frame["atemp"].astype("float64"),
            "hum": frame["hum"].astype("float64"),
            "windspeed": frame["windspeed"].astype("float64"),
            "casual": frame["casual"].astype("int64"),
            "registered": frame["registered"].astype("int64"),
            "cnt": frame["cnt"].astype("int64"),
        }
    )
    return raw.loc[:, [*RAW_COLUMNS, *LABEL_COLUMNS]].reset_index(drop=True)


class BundledDataset:
    """Read-only access to the bundled dataset and its manifest.

    `source` may be a directory containing `hour.csv.gz` (and optionally a
    `manifest.json`) or a single `.csv.gz` file. When no sibling manifest exists the
    provenance recorded in the installed package is used, which is what
    `mll data verify --path` needs for a tampered copy.
    """

    def __init__(self, source: Path | str | None = None) -> None:
        self._package_data = _package_data_dir()
        if source is None:
            self._hour_path = self._package_data / "hour.csv.gz"
            self._manifest_path = self._package_data / "manifest.json"
        else:
            candidate = Path(source)
            if candidate.is_dir():
                self._hour_path = candidate / "hour.csv.gz"
                self._manifest_path = self._first_existing(
                    candidate / "manifest.json", self._package_data / "manifest.json"
                )
            else:
                self._hour_path = candidate
                self._manifest_path = self._first_existing(
                    candidate.parent / "manifest.json", self._package_data / "manifest.json"
                )

    @staticmethod
    def _first_existing(*candidates: Path) -> Path:
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[-1]

    @property
    def hour_path(self) -> Path:
        """Path to the compressed CSV, exposed for diagnostics."""
        return self._hour_path

    def manifest(self) -> dict[str, Any]:
        """Return the parsed provenance manifest."""
        payload: dict[str, Any] = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return payload

    def verify(self) -> VerifiedDataset:
        """Hash the decompressed CSV and compare it with the manifest.

        Raises:
            ChecksumMismatchError: when the decompressed bytes do not match the pinned
                sha256; the bundle must not be used after this failure.
        """
        manifest = self.manifest()
        entry = manifest["files"]["hour.csv"]
        payload = gzip.decompress(self._hour_path.read_bytes())
        actual = hashlib.sha256(payload).hexdigest()
        expected = str(entry["sha256"])
        if actual != expected:
            raise ChecksumMismatchError("hour.csv", expected=expected, actual=actual)
        return VerifiedDataset(
            sha256=actual,
            size=len(payload),
            rows=int(entry["rows"]),
            start=datetime.fromisoformat(str(entry["start"])),
            end=datetime.fromisoformat(str(entry["end"])),
        )

    def raw(self) -> pd.DataFrame:
        """Return the full raw frame (`RAW_COLUMNS + LABEL_COLUMNS`)."""
        with gzip.open(self._hour_path, mode="rt", encoding="utf-8") as handle:
            uci = pd.read_csv(handle)
        return from_uci_frame(uci)

    def archive(self, end: datetime) -> pd.DataFrame:
        """Return archive rows strictly before `end`, capped at `ARCHIVE_END`."""
        raw = self.raw()
        boundary = min(pd.Timestamp(end), pd.Timestamp(ARCHIVE_END))
        selected = raw.loc[raw["timestamp"] < boundary]
        return selected.reset_index(drop=True)

    def window(self, window: TimeWindow) -> pd.DataFrame:
        """Return rows in the half-open `[start, end)` replay window."""
        raw = self.raw()
        selected = raw.loc[
            (raw["timestamp"] >= pd.Timestamp(window.start))
            & (raw["timestamp"] < pd.Timestamp(window.end))
        ]
        return selected.reset_index(drop=True)
