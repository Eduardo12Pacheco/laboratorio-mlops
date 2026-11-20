"""Training data repository: bundled archive plus labeled production rows (D6).

The archive wins when an entity id appears twice. Production labels only carry `cnt`, so
the composite frame reconstructs the two audit-only label columns (`casual` written as 0
and `registered` as `cnt`): their sum keeps the domain `label_consistency` rule honest and
neither column ever reaches the feature frame (`LEAKAGE_COLUMNS`).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from mlops_lifecycle_lab.application.ports import PredictionLog
from mlops_lifecycle_lab.domain.schema import LABEL_COLUMNS, RAW_COLUMNS


class ArchiveSource(Protocol):
    """Read-only archive of labeled rows before a boundary."""

    def archive(self, end: datetime) -> pd.DataFrame:
        """Return archived raw rows strictly before `end`."""
        ...


class CompositeLabeledData:
    """Union of the archive and the labeled prediction log, de-duplicated by entity."""

    def __init__(self, archive: ArchiveSource, prediction_log: PredictionLog) -> None:
        self._archive = archive
        self._predictions = prediction_log

    def training_frame(self, end_exclusive: datetime) -> pd.DataFrame:
        """Return raw rows (features plus labels) before `end_exclusive`, sorted."""
        archived = self._archive.archive(end_exclusive)
        production = self._predictions.all_labeled_features(end_exclusive)
        frames: list[pd.DataFrame] = []
        if not archived.empty:
            frames.append(archived.loc[:, [*RAW_COLUMNS, *LABEL_COLUMNS]])
        if not production.empty:
            frames.append(self._production_raw(production))
        if not frames:
            return pd.DataFrame(columns=[*RAW_COLUMNS, *LABEL_COLUMNS])
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["entity_id"], keep="first")
        combined = combined.sort_values(["timestamp", "entity_id"], kind="stable")
        return combined.reset_index(drop=True)

    @staticmethod
    def _production_raw(production: pd.DataFrame) -> pd.DataFrame:
        raw = production.copy()
        raw["casual"] = 0
        raw["registered"] = raw["cnt"]
        return raw.loc[:, [*RAW_COLUMNS, *LABEL_COLUMNS]]


__all__ = ["ArchiveSource", "CompositeLabeledData"]
