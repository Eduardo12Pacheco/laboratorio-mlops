"""Shared registry contract: run against the in-memory fake and, from T6, MLflow.

Every assertion uses only the `ModelRegistry` port, so a new adapter is covered by
adding its name to the fixture parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from mlops_lifecycle_lab.adapters.registry.mlflow_registry import MlflowRegistry
from mlops_lifecycle_lab.adapters.tracking.mlflow_tracker import MlflowTracker
from mlops_lifecycle_lab.application.dto import encode_version_tags
from mlops_lifecycle_lab.application.ports import ALIAS_CHAMPION, ModelRegistry, TrainedModel
from mlops_lifecycle_lab.domain.errors import (
    IllegalTransitionError,
    NoPreviousChampionError,
    UnknownModelVersionError,
)
from mlops_lifecycle_lab.domain.lifecycle import ModelStage, ModelVersion
from mlops_lifecycle_lab.domain.schema import FEATURE_COLUMNS
from mlops_lifecycle_lab.domain.windows import TimeWindow
from tests.fakes.mlflow_workspace import MlflowWorkspace
from tests.fakes.registry import InMemoryRegistry

CREATED_AT = datetime(2012, 1, 1)


class _RegistryHarness:
    """Contract wrapper that pre-registers the MLflow artifact of a version.

    The fake allocates versions itself, while a real registry can only adopt a version
    produced by a tracker run. The harness keeps that setup out of every shared assertion.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        prepare: Callable[[int], None],
        model_name: str,
    ) -> None:
        self._registry = registry
        self._prepare = prepare
        self.model_name = model_name

    def adopt_candidate(self, version: int, tags: Mapping[str, str]) -> ModelVersion:
        self._prepare(version)
        return self._registry.adopt_candidate(version, tags)

    def __getattr__(self, name: str) -> object:
        return getattr(self._registry, name)


@pytest.fixture(params=["memory", "mlflow"])
def registry(
    request: pytest.FixtureRequest,
    mlflow_workspace: MlflowWorkspace,
) -> Iterator[_RegistryHarness]:
    """Provide each registry backend under contract."""
    backend = str(request.param)
    if backend == "memory":
        yield _RegistryHarness(InMemoryRegistry(), lambda version: None, "bike-demand")
    elif backend == "mlflow":
        model_name = f"contract-{uuid4().hex}"
        tracker = MlflowTracker(
            tracking_uri=mlflow_workspace.uri,
            experiment_name="bike-demand",
            artifact_root=mlflow_workspace.root / "artifacts",
            model_name=model_name,
        )
        adapter = MlflowRegistry(tracking_uri=mlflow_workspace.uri, model_name=model_name)
        prepared: set[int] = set()
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({name: rng.random(12) for name in FEATURE_COLUMNS})

        def prepare(version: int) -> None:
            if version in prepared:
                return
            model = LGBMRegressor(n_estimators=2, n_jobs=1, verbose=-1).fit(
                frame, np.linspace(1, 10, len(frame))
            )
            with tracker.run("contract-register", {}) as run:
                logged = run.log_model(TrainedModel(model, {"algorithm": "lightgbm"}), frame)
            assert logged.registered_version == version
            prepared.add(version)

        yield _RegistryHarness(adapter, prepare, model_name)
    else:  # pragma: no cover - guards a typo in the parameter list
        raise AssertionError(f"unknown registry backend {backend!r}")


def make_version(version: int = 1, name: str = "bike-demand") -> ModelVersion:
    """Build a complete candidate version for the contract."""
    return ModelVersion(
        name=name,
        version=version,
        stage=ModelStage.CANDIDATE,
        run_id=f"run-{version}",
        metrics={"rmse": 50.0 + version, "mae": 30.0, "wape": 0.2, "bias": 1.0},
        train_window=TimeWindow(datetime(2011, 1, 1), datetime(2011, 10, 1)),
        holdout_window=TimeWindow(datetime(2011, 10, 1), datetime(2012, 1, 1)),
        data_fingerprint=f"fingerprint-{version}",
        feature_spec_version="1",
        created_at=CREATED_AT,
        trigger_reason="MANUAL",
        tags={"mll.git_sha": "deadbeef"},
    )


def expected(registry: _RegistryHarness, version: int) -> ModelVersion:
    """The exact version the adapter must rebuild from its tags."""
    return make_version(version, registry.model_name)


def adopt(registry: _RegistryHarness, version: int) -> ModelVersion:
    """Adopt a freshly registered version through the port."""
    name = registry.model_name
    return registry.adopt_candidate(version, encode_version_tags(make_version(version, name)))


def test_a_new_registry_has_no_versions_and_no_champion(registry: ModelRegistry) -> None:
    assert registry.list_versions() == []
    assert registry.get_champion() is None
    assert registry.get_alias(ALIAS_CHAMPION) is None
    assert registry.get_version(1) is None


def test_adopt_candidate_round_trips_the_lineage(registry: ModelRegistry) -> None:
    adopted = adopt(registry, 1)

    assert adopted == expected(registry, 1)
    assert adopted.stage is ModelStage.CANDIDATE
    assert registry.get_version(1) == adopted
    assert registry.list_versions() == [adopted]
    assert registry.get_alias("challenger") == adopted


def test_adopt_candidate_refuses_a_duplicate_version(registry: ModelRegistry) -> None:
    adopt(registry, 1)

    with pytest.raises(ValueError, match="already registered"):
        adopt(registry, 1)


def test_promote_sets_the_champion_alias(registry: ModelRegistry) -> None:
    adopt(registry, 1)

    promoted = registry.promote(1)

    assert promoted.stage is ModelStage.CHAMPION
    assert registry.get_champion() == promoted
    assert registry.get_alias(ALIAS_CHAMPION) == promoted


def test_promotion_archives_the_previous_champion(registry: ModelRegistry) -> None:
    adopt(registry, 1)
    registry.promote(1)
    adopt(registry, 2)

    promoted = registry.promote(2)

    archived = registry.get_version(1)
    assert archived is not None
    assert archived.stage is ModelStage.ARCHIVED
    previous = registry.get_alias("previous-champion")
    assert previous is not None
    assert previous.version == 1
    assert registry.get_champion() == promoted
    assert [model.version for model in registry.list_versions()] == [1, 2]


def test_rejecting_a_candidate_records_reason_and_failed_gates(
    registry: ModelRegistry,
) -> None:
    adopt(registry, 1)

    rejected = registry.reject(1, "worse than champion", ["IMPROVEMENT", "SEGMENT"])

    assert rejected.stage is ModelStage.REJECTED
    assert rejected.tags["mll.reject_reason"] == "worse than champion"
    assert rejected.tags["mll.failed_gates"] == "IMPROVEMENT,SEGMENT"
    assert registry.get_version(1) == rejected
    assert registry.get_alias("challenger") is None


def test_rejecting_a_champion_is_an_illegal_transition(registry: ModelRegistry) -> None:
    adopt(registry, 1)
    registry.promote(1)

    with pytest.raises(IllegalTransitionError):
        registry.reject(1, "too late", ["SEGMENT"])


def test_promoting_an_unknown_version_raises(registry: ModelRegistry) -> None:
    with pytest.raises(UnknownModelVersionError):
        registry.promote(99)


def test_promoting_the_current_champion_is_illegal(registry: ModelRegistry) -> None:
    adopt(registry, 1)
    registry.promote(1)

    with pytest.raises(IllegalTransitionError):
        registry.promote(1)


def test_rollback_without_a_previous_champion_raises(registry: ModelRegistry) -> None:
    adopt(registry, 1)
    registry.promote(1)

    with pytest.raises(NoPreviousChampionError):
        registry.rollback("nothing to restore")


def test_rollback_restores_the_previous_champion(registry: ModelRegistry) -> None:
    adopt(registry, 1)
    registry.promote(1)
    adopt(registry, 2)
    registry.promote(2)

    restored = registry.rollback("poisoned challenger")

    assert restored.version == 1
    assert restored.stage is ModelStage.CHAMPION
    bad = registry.get_version(2)
    assert bad is not None
    assert bad.stage is ModelStage.ROLLED_BACK
    assert registry.get_champion() == restored
    assert registry.get_alias("previous-champion") is None


def test_a_second_rollback_raises(registry: ModelRegistry) -> None:
    adopt(registry, 1)
    registry.promote(1)
    adopt(registry, 2)
    registry.promote(2)
    registry.rollback("first")

    with pytest.raises(NoPreviousChampionError):
        registry.rollback("second")


def test_loading_an_unknown_version_raises(registry: ModelRegistry) -> None:
    with pytest.raises(UnknownModelVersionError):
        registry.load_predictor(7)
