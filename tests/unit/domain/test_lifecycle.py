"""Model lifecycle state machine: transition matrix and immutable model versions."""

from __future__ import annotations

import itertools
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from mlops_lifecycle_lab.domain.errors import IllegalTransitionError
from mlops_lifecycle_lab.domain.lifecycle import (
    ALLOWED,
    ModelStage,
    ModelVersion,
    allowed_targets,
    can_transition,
    transition,
)
from mlops_lifecycle_lab.domain.windows import TimeWindow

STAGES = tuple(ModelStage)

ALLOWED_PAIRS = {
    (ModelStage.CANDIDATE, ModelStage.CHAMPION),
    (ModelStage.CANDIDATE, ModelStage.REJECTED),
    (ModelStage.CHAMPION, ModelStage.ARCHIVED),
    (ModelStage.CHAMPION, ModelStage.ROLLED_BACK),
    (ModelStage.ARCHIVED, ModelStage.CHAMPION),
}


def make_version(stage: ModelStage = ModelStage.CANDIDATE, version: int = 1) -> ModelVersion:
    return ModelVersion(
        name="bike-demand",
        version=version,
        stage=stage,
        run_id="run-abc",
        metrics={"rmse": 50.88, "wape": 0.2273},
        train_window=TimeWindow(datetime(2011, 1, 1), datetime(2011, 10, 1)),
        holdout_window=TimeWindow(datetime(2011, 10, 1), datetime(2012, 1, 1)),
        data_fingerprint="abc123",
        feature_spec_version="1",
        created_at=datetime(2012, 1, 1),
        trigger_reason="BOOTSTRAP",
        tags={"mll.stage": stage.value},
    )


def test_the_allowed_map_has_exactly_the_five_documented_pairs() -> None:
    observed = {(source, target) for source, targets in ALLOWED.items() for target in targets}

    assert observed == ALLOWED_PAIRS
    assert ModelStage.CANDIDATE in ALLOWED
    assert len(ALLOWED) == len(STAGES)


@pytest.mark.parametrize(("source", "target"), list(itertools.product(STAGES, STAGES)))
def test_the_25_pair_transition_matrix(source: ModelStage, target: ModelStage) -> None:
    version = make_version(source)

    if (source, target) in ALLOWED_PAIRS:
        updated = transition(version, target)
        assert updated.stage is target
        assert updated is not version
        assert version.stage is source
        assert updated.run_id == version.run_id
        assert updated.metrics == version.metrics
        assert updated.data_fingerprint == version.data_fingerprint
    else:
        with pytest.raises(IllegalTransitionError, match=source.value):
            transition(version, target)
        assert version.stage is source


def test_can_transition_matches_the_allowed_table() -> None:
    for source, target in itertools.product(STAGES, STAGES):
        assert can_transition(source, target) is ((source, target) in ALLOWED_PAIRS)


def test_allowed_targets_are_returned_as_an_immutable_set() -> None:
    assert allowed_targets(ModelStage.CHAMPION) == frozenset(
        {ModelStage.ARCHIVED, ModelStage.ROLLED_BACK}
    )
    assert allowed_targets(ModelStage.REJECTED) == frozenset()
    assert allowed_targets(ModelStage.ROLLED_BACK) == frozenset()


def test_an_illegal_transition_names_both_stages_and_the_version() -> None:
    version = make_version(ModelStage.REJECTED, version=7)

    with pytest.raises(IllegalTransitionError) as raised:
        transition(version, ModelStage.CHAMPION)

    message = str(raised.value)
    assert "REJECTED" in message
    assert "CHAMPION" in message
    assert "7" in message


def test_model_versions_are_frozen() -> None:
    version = make_version()

    with pytest.raises(FrozenInstanceError):
        version.stage = ModelStage.CHAMPION  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "number"),
    [("", 1), ("bike-demand", 0), ("bike-demand", -1)],
)
def test_invalid_versions_are_rejected(name: str, number: int) -> None:
    with pytest.raises(ValueError, match=r"name|version"):
        make_version_and_override(name=name, number=number)


def make_version_and_override(name: str, number: int) -> ModelVersion:
    version = make_version()
    return ModelVersion(
        name=name,
        version=number,
        stage=version.stage,
        run_id=version.run_id,
        metrics=version.metrics,
        train_window=version.train_window,
        holdout_window=version.holdout_window,
        data_fingerprint=version.data_fingerprint,
        feature_spec_version=version.feature_spec_version,
        created_at=version.created_at,
        trigger_reason=version.trigger_reason,
        tags=version.tags,
    )


def test_stage_values_are_the_documented_tags() -> None:
    assert [stage.value for stage in STAGES] == [
        "CANDIDATE",
        "CHAMPION",
        "ARCHIVED",
        "REJECTED",
        "ROLLED_BACK",
    ]
