"""In-memory model registry implementing the tags-and-aliases contract (S7)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from mlops_lifecycle_lab.application.dto import (
    TAG_FAILED_GATES,
    TAG_REJECT_REASON,
    decode_version_tags,
)
from mlops_lifecycle_lab.application.ports import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    ALIAS_PREVIOUS_CHAMPION,
    Predictor,
)
from mlops_lifecycle_lab.domain.errors import (
    IllegalTransitionError,
    NoPreviousChampionError,
    UnknownModelVersionError,
)
from mlops_lifecycle_lab.domain.lifecycle import ModelStage, ModelVersion, transition


class InMemoryRegistry:
    """Registry with sequential versions, `mll.*` tags and the three aliases."""

    def __init__(self, name: str = "bike-demand") -> None:
        self._name = name
        self._versions: dict[int, ModelVersion] = {}
        self._aliases: dict[str, int] = {}
        self._predictors: dict[int, Predictor] = {}

    def set_predictor(self, version: int, predictor: Predictor) -> None:
        """Attach a predictor to a version (test helper; MLflow loads the artifact)."""
        self._predictors[version] = predictor

    def adopt_candidate(self, version: int, tags: Mapping[str, str]) -> ModelVersion:
        """Validate the tags and store the new candidate."""
        if version in self._versions:
            raise ValueError(f"model version {version} is already registered")
        model = decode_version_tags(self._name, version, tags)
        if model.stage is not ModelStage.CANDIDATE:
            raise ValueError(f"new versions must be CANDIDATE, got {model.stage.value}")
        self._versions[version] = model
        self._aliases[ALIAS_CHALLENGER] = version
        return model

    def get_champion(self) -> ModelVersion | None:
        """Return the version aliased `champion`."""
        return self.get_alias(ALIAS_CHAMPION)

    def get_version(self, version: int) -> ModelVersion | None:
        """Return one version or `None`."""
        return self._versions.get(version)

    def list_versions(self) -> list[ModelVersion]:
        """Every version ordered by number."""
        return [self._versions[number] for number in sorted(self._versions)]

    def get_alias(self, alias: str) -> ModelVersion | None:
        """Return the version an alias points at, or `None`."""
        version = self._aliases.get(alias)
        return None if version is None else self._versions.get(version)

    def promote(self, version: int) -> ModelVersion:
        """Archive the incumbent, move both aliases and mark the candidate champion."""
        candidate = self._require(version)
        current = self.get_champion()
        if current is not None:
            if current.version == version:
                raise IllegalTransitionError(
                    f"illegal transition for {self._name} v{version}: "
                    f"{ModelStage.CHAMPION.value} -> {ModelStage.CHAMPION.value}"
                )
            archived = transition(current, ModelStage.ARCHIVED)
            self._versions[archived.version] = archived
            self._aliases[ALIAS_PREVIOUS_CHAMPION] = archived.version
        promoted = transition(candidate, ModelStage.CHAMPION)
        self._versions[promoted.version] = promoted
        self._aliases[ALIAS_CHAMPION] = promoted.version
        return promoted

    def reject(self, version: int, reason: str, failed_gates: Sequence[str]) -> ModelVersion:
        """Reject a candidate and record why."""
        model = self._require(version)
        rejected = transition(model, ModelStage.REJECTED)
        rejected = replace(
            rejected,
            tags={
                **rejected.tags,
                TAG_REJECT_REASON: reason,
                TAG_FAILED_GATES: ",".join(failed_gates),
            },
        )
        self._versions[rejected.version] = rejected
        if self._aliases.get(ALIAS_CHALLENGER) == version:
            del self._aliases[ALIAS_CHALLENGER]
        return rejected

    def rollback(self, reason: str) -> ModelVersion:
        """Restore `previous-champion` and mark the bad champion `ROLLED_BACK`."""
        previous = self.get_alias(ALIAS_PREVIOUS_CHAMPION)
        if previous is None:
            raise NoPreviousChampionError("no previous champion alias to roll back to")
        current = self.get_champion()
        if current is None:
            raise NoPreviousChampionError("no current champion to roll back")
        rolled_back = transition(current, ModelStage.ROLLED_BACK)
        rolled_back = replace(rolled_back, tags={**rolled_back.tags, TAG_REJECT_REASON: reason})
        self._versions[rolled_back.version] = rolled_back
        restored = transition(previous, ModelStage.CHAMPION)
        self._versions[restored.version] = restored
        self._aliases[ALIAS_CHAMPION] = restored.version
        del self._aliases[ALIAS_PREVIOUS_CHAMPION]
        if self._aliases.get(ALIAS_CHALLENGER) == rolled_back.version:
            del self._aliases[ALIAS_CHALLENGER]
        return restored

    def load_predictor(self, version: int) -> Predictor:
        """Return the attached predictor, refusing unknown versions."""
        if version not in self._versions:
            raise UnknownModelVersionError(f"model version {version} is not registered")
        try:
            return self._predictors[version]
        except KeyError as exc:
            raise UnknownModelVersionError(
                f"model version {version} has no attached predictor"
            ) from exc

    def _require(self, version: int) -> ModelVersion:
        model = self.get_version(version)
        if model is None:
            raise UnknownModelVersionError(f"model version {version} is not registered")
        return model
