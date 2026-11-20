"""In-memory experiment tracker for application unit tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from mlops_lifecycle_lab.application.dto import JSONValue
from mlops_lifecycle_lab.application.ports import LoggedModel, TrainedModel

if TYPE_CHECKING:
    from tests.fakes.registry import InMemoryRegistry


@dataclass
class FakeRun:
    """One in-memory run; mirrors the `RunHandle` port and records every call."""

    run_id: str
    name: str
    tags: Mapping[str, str]
    tracker: InMemoryTracker
    params: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    jsons: dict[str, JSONValue] = field(default_factory=dict)
    models: list[tuple[TrainedModel, pd.DataFrame]] = field(default_factory=list)
    closed: bool = False

    def log_params(self, params: Mapping[str, object]) -> None:
        """Record parameters (later calls merge)."""
        self.params.update(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Record metrics (later calls merge)."""
        self.metrics.update(metrics)

    def log_json(self, name: str, obj: JSONValue) -> None:
        """Record a JSON artifact."""
        self.jsons[name] = obj

    def log_model(self, trained: TrainedModel, input_example: pd.DataFrame) -> LoggedModel:
        """Assign the next version and expose the predictor to a linked registry."""
        version = self.tracker.register(trained)
        self.models.append((trained, input_example.copy()))
        return LoggedModel(uri=f"memory://{self.run_id}/model", registered_version=version)


class InMemoryTracker:
    """Sequential in-memory tracker; optionally links predictors into a fake registry."""

    def __init__(
        self,
        registry: InMemoryRegistry | None = None,
        *,
        first_version: int = 1,
    ) -> None:
        self.registry = registry
        self.next_version = first_version
        self.runs: list[FakeRun] = []
        self._counter = 0

    @contextmanager
    def run(self, name: str, tags: Mapping[str, str]) -> Iterator[FakeRun]:
        """Open a fake run; the run is marked closed when the body exits."""
        self._counter += 1
        run = FakeRun(run_id=f"run-{self._counter}", name=name, tags=dict(tags), tracker=self)
        self.runs.append(run)
        try:
            yield run
        finally:
            run.closed = True

    def register(self, trained: TrainedModel) -> int:
        """Allocate the next version and attach its predictor to the registry."""
        version = self.next_version
        self.next_version += 1
        if self.registry is not None:
            self.registry.set_predictor(version, trained.predictor)
        return version

    def latest_run(self) -> FakeRun:
        """Return the most recent run."""
        return self.runs[-1]
