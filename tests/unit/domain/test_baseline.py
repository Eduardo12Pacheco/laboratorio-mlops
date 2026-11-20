"""Profile baseline: conditional means, fallback and the S5 reference score."""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset
from mlops_lifecycle_lab.domain.baseline import ProfileBaseline
from mlops_lifecycle_lab.domain.errors import ValidationFailedError
from mlops_lifecycle_lab.domain.metrics import compute_metrics


def make_frame() -> pd.DataFrame:
    """A tiny profile frame with repeated (workingday, hr) keys."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    datetime(2011, 1, 1, 8),
                    datetime(2011, 1, 1, 9),
                    datetime(2011, 1, 2, 8),
                    datetime(2011, 1, 2, 9),
                    datetime(2011, 1, 3, 8),
                ]
            ),
            "workingday": np.array([1, 1, 1, 0, 0], dtype=np.int64),
            "cnt": np.array([10, 30, 20, 5, 15], dtype=np.int64),
        }
    )


def frame_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def test_profile_is_the_mean_by_workingday_and_hour() -> None:
    baseline = ProfileBaseline.fit(make_frame())

    prediction = baseline.predict(
        pd.DataFrame({"workingday": [1, 0], "hr": [8, 8]}, dtype=np.int64)
    )

    assert prediction.tolist() == pytest.approx([15.0, 5.0])


def test_unseen_key_falls_back_to_the_global_mean() -> None:
    train = make_frame()
    baseline = ProfileBaseline.fit(train)

    prediction = baseline.predict(
        pd.DataFrame({"workingday": [1, 0], "hr": [23, 23]}, dtype=np.int64)
    )

    global_mean = float(train["cnt"].mean())
    assert prediction.tolist() == pytest.approx([global_mean, global_mean])


def test_hour_is_derived_from_the_timestamp_when_absent() -> None:
    baseline = ProfileBaseline.fit(make_frame())

    from_timestamp = baseline.predict(
        pd.DataFrame(
            {
                "workingday": [1],
                "timestamp": pd.to_datetime([datetime(2012, 5, 1, 8)]),
            }
        )
    )

    assert from_timestamp.tolist() == pytest.approx([15.0])


def test_accepts_a_feature_frame_with_hr_already_present() -> None:
    features = pd.DataFrame({"workingday": [1, 0, 1], "hr": [8, 8, 9]})

    prediction = ProfileBaseline.fit(make_frame()).predict(features)

    assert isinstance(prediction, np.ndarray)
    assert prediction.dtype == np.float64
    assert prediction.shape == (3,)


def test_predict_never_mutates_its_input() -> None:
    frame = pd.DataFrame({"workingday": [1, 0], "hr": [8, 9]})
    before = frame_hash(frame)

    ProfileBaseline.fit(make_frame()).predict(frame)

    assert frame_hash(frame) == before


def test_empty_prediction_frame_returns_an_empty_array() -> None:
    prediction = ProfileBaseline.fit(make_frame()).predict(
        pd.DataFrame({"workingday": pd.Series(dtype="int64"), "hr": pd.Series(dtype="int64")})
    )

    assert prediction.shape == (0,)


def test_fit_never_mutates_the_training_frame() -> None:
    frame = make_frame()
    before = frame_hash(frame)

    ProfileBaseline.fit(frame)

    assert frame_hash(frame) == before


@pytest.mark.parametrize("column", ["workingday", "cnt"])
def test_fit_requires_the_profile_columns(column: str) -> None:
    with pytest.raises(ValidationFailedError, match=column):
        ProfileBaseline.fit(make_frame().drop(columns=[column]))


def test_fit_requires_a_usable_hour_source() -> None:
    frame = make_frame().drop(columns=["timestamp"])

    with pytest.raises(ValidationFailedError, match="hr"):
        ProfileBaseline.fit(frame)


def test_reference_score_on_the_bundled_holdout_is_68() -> None:
    raw = BundledDataset().raw()
    train = raw.loc[
        (raw["timestamp"] >= pd.Timestamp("2011-01-01"))
        & (raw["timestamp"] < pd.Timestamp("2011-10-01"))
    ]
    holdout = raw.loc[
        (raw["timestamp"] >= pd.Timestamp("2011-10-01"))
        & (raw["timestamp"] < pd.Timestamp("2012-01-01"))
    ]

    prediction = ProfileBaseline.fit(train).predict(holdout)
    metrics = compute_metrics(holdout["cnt"], prediction)

    assert len(train) == 6442
    assert len(holdout) == 2203
    assert metrics.rmse == pytest.approx(68.00, abs=0.01)
    assert metrics.n == 2203
