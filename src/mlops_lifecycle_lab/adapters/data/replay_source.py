"""Replay batch source over the bundled dataset (S14, D3).

The replay path is the production input: tick `k` gets exactly the rows of its half-open
week. Labels are kept out of the raw frame and returned in a separate label frame, so a
scoring path can never accidentally see the future; a caller that wants them must ask for
`labels_for(window)` explicitly (the lifecycle tick does so only after scoring).
"""

from __future__ import annotations

import pandas as pd

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset
from mlops_lifecycle_lab.application.dto import RawBatch
from mlops_lifecycle_lab.domain.schema import RAW_COLUMNS, TARGET_COLUMN
from mlops_lifecycle_lab.domain.windows import TimeWindow

_LABEL_FRAME_COLUMNS: tuple[str, ...] = ("entity_id", "label", "label_ts")


class ReplayBatchSource:
    """`BatchSource` view over `BundledDataset.window` with delayed labels."""

    def __init__(self, dataset: BundledDataset) -> None:
        self._dataset = dataset

    def week(self, window: TimeWindow) -> RawBatch:
        """Return the raw rows of `window` plus the labels that arrive at its end."""
        rows = self._dataset.window(window)
        if rows.empty:
            raise ValueError(
                f"the bundled dataset has no rows in "
                f"[{window.start.isoformat()}, {window.end.isoformat()})"
            )
        raw = rows.loc[:, list(RAW_COLUMNS)].reset_index(drop=True)
        return RawBatch(raw=raw, labels=self._labels(rows))

    def labels_for(self, window: TimeWindow) -> pd.DataFrame:
        """Return only the hidden labels of `window` (entity id, value and timestamp)."""
        return self._labels(self._dataset.window(window))

    @staticmethod
    def _labels(rows: pd.DataFrame) -> pd.DataFrame:
        labels = pd.DataFrame(
            {
                "entity_id": rows["entity_id"].to_numpy(dtype="int64"),
                "label": rows[TARGET_COLUMN].to_numpy(dtype="float64"),
                "label_ts": pd.to_datetime(rows["timestamp"]).to_numpy(),
            },
            columns=list(_LABEL_FRAME_COLUMNS),
        )
        return labels


__all__ = ["ReplayBatchSource"]
