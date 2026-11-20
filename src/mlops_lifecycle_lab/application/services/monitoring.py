"""Monitoring snapshot use case: trailing metrics plus drift evidence (S10).

One snapshot answers two different questions and keeps them separable, because the
lifecycle couples them differently: *how is the current champion doing on labelled
traffic* (WAPE/RMSE/MAE/bias over the trailing window) and *how far did the inputs move
away from the champion's training reference* (native PSI, never a retrain trigger on its
own — see ADR-0004). The result is persisted for the dashboard and appended to the audit
log so a replay can be reconstructed from SQLite alone.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from mlops_lifecycle_lab.application.dto import MonitoringSnapshot
from mlops_lifecycle_lab.application.ports import (
    Clock,
    EventStore,
    ExperimentTracker,
    LabeledDataRepository,
    ModelRegistry,
    PredictionLog,
    SnapshotStore,
)
from mlops_lifecycle_lab.application.services._data import window_rows
from mlops_lifecycle_lab.domain.drift import summarize_drift
from mlops_lifecycle_lab.domain.errors import UnknownModelVersionError
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.lifecycle import ModelVersion
from mlops_lifecycle_lab.domain.metrics import compute_metrics
from mlops_lifecycle_lab.domain.schema import (
    FEATURE_COLUMNS,
    MONITORED_CATEGORICAL,
    MONITORED_NUMERIC,
)

_DRIFT_COLUMNS: tuple[str, ...] = (*MONITORED_NUMERIC, *MONITORED_CATEGORICAL)
_REFERENCE_PREDICTIONS_ARTIFACT = "reference_predictions.json"


class ComputeSnapshot:
    """Compute, persist and audit one monitoring snapshot for the current champion."""

    def __init__(
        self,
        *,
        prediction_log: PredictionLog,
        labeled_data: LabeledDataRepository,
        registry: ModelRegistry,
        snapshots: SnapshotStore,
        events: EventStore,
        clock: Clock,
        tracker: ExperimentTracker,
        window_days: int = 28,
    ) -> None:
        if window_days <= 0:
            raise ValueError(f"window_days must be positive, got {window_days}")
        self._prediction_log = prediction_log
        self._labeled_data = labeled_data
        self._registry = registry
        self._snapshots = snapshots
        self._events = events
        self._clock = clock
        self._tracker = tracker
        self._window_days = window_days

    def execute(
        self,
        now: datetime,
        champion_version: int,
        *,
        tick: int = 0,
    ) -> MonitoringSnapshot:
        """Build the snapshot of the trailing window ending at `now`.

        Args:
            now: exclusive end of the monitoring window (the tick end in replay).
            champion_version: version whose predictions are evaluated; older predictions
                of a superseded champion are ignored.
            tick: replay tick stamped on the snapshot and its event.

        Raises:
            UnknownModelVersionError: when the version is not in the registry.
        """
        champion = self._registry.get_version(champion_version)
        if champion is None:
            raise UnknownModelVersionError(f"model version {champion_version} is not registered")
        start = now - timedelta(days=self._window_days)
        labeled = self._prediction_log.labeled_predictions(
            start, now, model_version=champion_version
        )

        if labeled.empty:
            snapshot = MonitoringSnapshot(
                tick=tick,
                ts=now,
                model_version=champion_version,
                n_rows=0,
                rmse=None,
                mae=None,
                wape=None,
                bias=None,
                numeric_drift_share=0.0,
                prediction_psi=None,
                psi={},
            )
        else:
            actuals = np.asarray(labeled["label"], dtype=np.float64)
            predictions = np.asarray(labeled["prediction"], dtype=np.float64)
            metrics = compute_metrics(actuals, predictions)
            psi, drift_share, prediction_psi = self._drift(champion, labeled)
            snapshot = MonitoringSnapshot(
                tick=tick,
                ts=now,
                model_version=champion_version,
                n_rows=metrics.n,
                rmse=metrics.rmse,
                mae=metrics.mae,
                wape=metrics.wape,
                bias=metrics.bias,
                numeric_drift_share=drift_share,
                prediction_psi=prediction_psi,
                psi=psi,
            )

        self._snapshots.save(snapshot)
        self._events.append(
            LifecycleEvent(
                ts=self._clock.now(),
                type=EventType.SNAPSHOT_COMPUTED,
                tick=tick,
                model_version=champion_version,
                payload=snapshot.to_dict(),
            )
        )
        return snapshot

    def _drift(
        self,
        champion: ModelVersion,
        labeled: pd.DataFrame,
    ) -> tuple[dict[str, float], float, float | None]:
        """Feature drift and prediction PSI against the champion's references."""
        training = self._labeled_data.training_frame(champion.train_window.end)
        reference_rows = window_rows(training, champion.train_window)
        current_features = labeled.loc[:, list(_DRIFT_COLUMNS)]
        if reference_rows.empty:
            return {}, 0.0, None

        reference_features = build_features(reference_rows).loc[:, list(_DRIFT_COLUMNS)]
        current_predictions = np.asarray(labeled["prediction"], dtype=np.float64)
        reference_predictions = self._reference_predictions(champion)
        if reference_predictions is None or reference_predictions.size == 0:
            # Feature drift is still computable; without a reference distribution the
            # prediction PSI is deliberately reported as missing instead of faked.
            summary = summarize_drift(
                reference_features,
                current_features,
                current_predictions,
                current_predictions,
            )
            prediction_psi: float | None = None
        else:
            summary = summarize_drift(
                reference_features,
                current_features,
                reference_predictions,
                current_predictions,
            )
            prediction_psi = summary.prediction_psi
        psi = {feature.feature: feature.psi for feature in summary.features}
        return psi, summary.numeric_drift_share, prediction_psi

    def _reference_predictions(self, champion: ModelVersion) -> np.ndarray | None:
        """Load the validation-time predictions logged with the champion's training run.

        The artifact is the primary source. Rebuilding them by re-scoring the champion's
        holdout is the fallback for registries populated outside `TrainCandidate` (older
        workspaces, contract fixtures), keeping monitoring usable without a backfill.
        """
        artifact = self._tracker.load_json(_REFERENCE_PREDICTIONS_ARTIFACT, champion.run_id)
        values = _artifact_values(artifact)
        if values is not None:
            return values
        if champion.holdout_window is None:
            return None
        rows = window_rows(
            self._labeled_data.training_frame(champion.holdout_window.end),
            champion.holdout_window,
        )
        if rows.empty:
            return None
        features = build_features(rows).loc[:, list(FEATURE_COLUMNS)]
        predictions = self._registry.load_predictor(champion.version).predict(features)
        return np.asarray(predictions, dtype=np.float64)


def _artifact_values(artifact: object) -> np.ndarray | None:
    """Extract the numeric prediction vector from a tracker JSON artifact."""
    if not isinstance(artifact, dict):
        return None
    values = artifact.get("values")
    if not isinstance(values, list) or not values:
        return None
    try:
        return np.asarray([float(value) for value in values], dtype=np.float64)
    except (TypeError, ValueError):
        return None


__all__ = ["ComputeSnapshot"]
