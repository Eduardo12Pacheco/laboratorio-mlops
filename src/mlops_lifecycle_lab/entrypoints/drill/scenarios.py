"""Scenario registry and typed intervention scripts for the drill (S15, S16).

A scenario is a name, a default number of weekly ticks and a frozen script. The runner
translates the script into `TickInterventions` and trainer arming; the lifecycle tick stays
agnostic of the drill (its hooks are typed and no-ops by default).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import pandas as pd

from mlops_lifecycle_lab.application.services.lifecycle import TickInterventions
from mlops_lifecycle_lab.entrypoints.drill.faults import celsius_scale

BatchFault = Callable[[pd.DataFrame], pd.DataFrame]
"""A pure transform applied to one weekly batch before validation."""


def _require_tick(tick: int) -> int:
    if tick < 1:
        raise ValueError(f"tick must be >= 1, got {tick}")
    return tick


@dataclass(frozen=True, slots=True)
class ScenarioScript:
    """Typed interventions at specific ticks; immutable builders return new scripts."""

    batch_faults: Mapping[int, BatchFault] = field(default_factory=dict)
    manual_retrain_ticks: frozenset[int] = frozenset()
    poisoned_promotions: Mapping[int, float] = field(default_factory=dict)

    def corrupt_batch_at(self, tick: int, fault: BatchFault) -> ScenarioScript:
        """Return a script that applies `fault` to the batch of `tick`."""
        tick = _require_tick(tick)
        return replace(self, batch_faults={**self.batch_faults, tick: fault})

    def manual_retrain_at(self, tick: int) -> ScenarioScript:
        """Return a script that requests a manual (but gated) retrain at `tick`."""
        tick = _require_tick(tick)
        return replace(self, manual_retrain_ticks=self.manual_retrain_ticks | {tick})

    def forced_poisoned_promotion_at(self, tick: int, label_scale: float = 0.3) -> ScenarioScript:
        """Return a script that trains a poisoned challenger and forces its promotion."""
        tick = _require_tick(tick)
        if not 0.0 < label_scale < 1.0:
            raise ValueError(f"label_scale must be in (0, 1), got {label_scale}")
        return replace(
            self,
            poisoned_promotions={**self.poisoned_promotions, tick: label_scale},
        )

    def interventions(self, tick: int) -> TickInterventions:
        """Translate the script into the typed hooks the lifecycle tick accepts."""
        fault = self.batch_faults.get(tick)
        manual = tick in self.manual_retrain_ticks
        forced_scale = self.poisoned_promotions.get(tick)

        def before_validate(
            _tick: int,
            _window: object,
            raw: pd.DataFrame,
        ) -> pd.DataFrame:
            if fault is None:
                return raw
            return fault(raw)

        decide: Callable[[object], None] | None = None
        if manual or forced_scale is not None:

            def before_decide(view: object) -> None:
                view.manual_trigger = True  # type: ignore[attr-defined]
                if forced_scale is not None:
                    view.force_promotion = True  # type: ignore[attr-defined]
                    view.promotion_reason = (  # type: ignore[attr-defined]
                        f"drill fault injection: poisoned challenger forced at tick {tick}"
                    )

            decide = before_decide

        return TickInterventions(
            before_validate=before_validate if fault is not None else None,
            before_decide=decide,
        )

    def trainer_label_scale(self, tick: int) -> float | None:
        """Label corruption scale to arm for `tick`, or `None` for a normal trainer."""
        return self.poisoned_promotions.get(tick)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One reproducible drill scenario."""

    name: str
    default_weeks: int
    script: ScenarioScript

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if self.default_weeks < 1:
            raise ValueError(f"default_weeks must be >= 1, got {self.default_weeks}")


SCENARIOS: Mapping[str, Scenario] = {
    "drift": Scenario("drift", 52, ScenarioScript()),
    "gate-reject": Scenario(
        "gate-reject",
        24,
        ScenarioScript().manual_retrain_at(23),
    ),
    "bad-batch": Scenario(
        "bad-batch",
        12,
        ScenarioScript().corrupt_batch_at(7, celsius_scale),
    ),
    "rollback": Scenario(
        "rollback",
        25,
        ScenarioScript().forced_poisoned_promotion_at(23),
    ),
}
"""The four scenarios and their S15/S16 default lengths."""


def get_scenario(name: str) -> Scenario:
    """Resolve a scenario name case-insensitively.

    Raises:
        ValueError: when the name is not one of the four scenarios.
    """
    try:
        return SCENARIOS[name.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"unknown drill scenario {name!r}; choose one of: {choices}") from exc


__all__ = ["SCENARIOS", "BatchFault", "Scenario", "ScenarioScript", "get_scenario"]
