"""Train a challenger and register it as a candidate (S5, S6, S12 protocol).

The use case owns the *protocol*: which rows are the training frame, which are the
holdout, how the naive baseline is fitted on the same rows, and which lineage is logged.
The actual fitting is delegated to the `ModelTrainer` port and the artifact/registry
writes to the `ExperimentTracker`/`ModelRegistry` ports, so the same code runs against
MLflow + LightGBM, the in-memory fakes, and the drill.
"""

from __future__ import annotations

import numpy as np

from mlops_lifecycle_lab.application.dto import (
    TAG_DATA_FINGERPRINT,
    TAG_FEATURE_SPEC_VERSION,
    TAG_GIT_SHA,
    TAG_HOLDOUT_WINDOW,
    TAG_TRAIN_WINDOW,
    TAG_TRIGGER_REASON,
    TrainRequest,
    TrainResult,
    encode_version_tags,
)
from mlops_lifecycle_lab.application.ports import (
    Clock,
    ExperimentTracker,
    LabeledDataRepository,
    ModelRegistry,
    ModelTrainer,
)
from mlops_lifecycle_lab.application.services._data import (
    clip_predictions,
    data_fingerprint,
    feature_frame,
    require_rows,
    window_rows,
    window_tag,
)
from mlops_lifecycle_lab.domain.baseline import ProfileBaseline
from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.features import build_features
from mlops_lifecycle_lab.domain.lifecycle import ModelStage, ModelVersion
from mlops_lifecycle_lab.domain.metrics import compute_metrics
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS, FEATURE_SPEC_VERSION, TARGET_COLUMN
from mlops_lifecycle_lab.domain.validation import Severity, Verdict, validate_frame


class TrainCandidate:
    """Assemble, validate, train, log and register one candidate model."""

    def __init__(
        self,
        labeled_data: LabeledDataRepository,
        trainer: ModelTrainer,
        tracker: ExperimentTracker,
        registry: ModelRegistry,
        clock: Clock,
        *,
        seed: int = 42,
        model_name: str = "bike-demand",
        feature_spec_version: str = FEATURE_SPEC_VERSION,
        git_sha: str = "unknown",
    ) -> None:
        self._labeled_data = labeled_data
        self._trainer = trainer
        self._tracker = tracker
        self._registry = registry
        self._clock = clock
        self._seed = seed
        self._model_name = model_name
        self._feature_spec_version = feature_spec_version
        self._git_sha = git_sha

    def execute(self, request: TrainRequest) -> TrainResult:
        """Train on `request.train_window`, evaluate on its holdout and adopt the result.

        Raises:
            NotEnoughLabelsError: when either window has no labeled rows.
            ValidationFailedError: when the training frame fails the data contract; a
                failed frame is never used for training (S3).
        """
        rows = self._labeled_data.training_frame(request.holdout_window.end)
        train_rows = window_rows(rows, request.train_window)
        holdout_rows = window_rows(rows, request.holdout_window)
        require_rows(train_rows, "training window")
        require_rows(holdout_rows, "holdout window")

        validation = validate_frame(train_rows, context="training")
        if validation.verdict is Verdict.FAIL:
            failed = tuple(
                rule.name
                for rule in validation.rules
                if not rule.passed and rule.severity is Severity.ERROR
            )
            raise ValidationFailedError(
                failed_rules=failed,
                message=(
                    "training frame failed the data contract: "
                    f"{', '.join(failed) or 'unknown rules'}"
                ),
            )

        training_frame = feature_frame(train_rows)
        trained = self._trainer.fit(training_frame, self._seed)
        holdout_features = build_features(holdout_rows).loc[:, list(FEATURE_COLUMNS)]
        predictions = clip_predictions(trained.predictor.predict(holdout_features))
        actuals = np.asarray(holdout_rows[TARGET_COLUMN], dtype=np.float64)
        metrics = compute_metrics(actuals, predictions)

        baseline = ProfileBaseline.fit(train_rows)
        baseline_predictions = baseline.predict(holdout_rows)
        baseline_rmse = compute_metrics(actuals, baseline_predictions).rmse
        fingerprint = data_fingerprint(train_rows)

        tags = {
            TAG_TRAIN_WINDOW: window_tag(request.train_window),
            TAG_HOLDOUT_WINDOW: window_tag(request.holdout_window),
            TAG_DATA_FINGERPRINT: fingerprint,
            TAG_FEATURE_SPEC_VERSION: self._feature_spec_version,
            TAG_GIT_SHA: self._git_sha,
            TAG_TRIGGER_REASON: request.trigger_reason,
        }
        run_name = f"train-{request.trigger_reason}"
        if request.tick is not None:
            run_name = f"{run_name}-tick-{request.tick}"

        with self._tracker.run(run_name, tags) as run:
            run.log_params({**dict(trained.params), "seed": self._seed})
            run.log_metrics(
                {
                    "val_rmse": metrics.rmse,
                    "val_mae": metrics.mae,
                    "val_wape": metrics.wape,
                    "val_bias": metrics.bias,
                    "baseline_rmse": baseline_rmse,
                    "rows": float(len(train_rows)),
                }
            )
            run.log_json("validation_report.json", validation.to_dict())
            run.log_json("feature_columns.json", list(FEATURE_COLUMNS))
            run.log_json(
                "reference_predictions.json",
                {
                    "values": [float(value) for value in predictions],
                    "count": len(predictions),
                    "window": window_tag(request.holdout_window),
                },
            )
            logged = run.log_model(trained, input_example=training_frame[list(FEATURE_COLUMNS)])

        candidate = ModelVersion(
            name=self._model_name,
            version=logged.registered_version,
            stage=ModelStage.CANDIDATE,
            run_id=run.run_id,
            metrics={
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "wape": metrics.wape,
                "bias": metrics.bias,
            },
            train_window=request.train_window,
            holdout_window=request.holdout_window,
            data_fingerprint=fingerprint,
            feature_spec_version=self._feature_spec_version,
            created_at=self._clock.now(),
            trigger_reason=request.trigger_reason,
            tags={TAG_GIT_SHA: self._git_sha},
        )
        adopted = self._registry.adopt_candidate(
            logged.registered_version, encode_version_tags(candidate)
        )
        return TrainResult(
            version=adopted.version,
            model=adopted,
            metrics=metrics,
            baseline_rmse=baseline_rmse,
            data_fingerprint=fingerprint,
            run_id=run.run_id,
            training_rows=len(train_rows),
            holdout_rows=len(holdout_rows),
            validation=validation,
        )
