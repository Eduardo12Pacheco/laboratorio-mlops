"""The drill engine: bootstrap once, replay weekly ticks, summarize and write (S15).

The engine is the only place that knows how to turn a scenario into a reproducible run.
It owns the simulated clock and the replay cursor (`replay_tick` in the KV store, the same
key the scheduler uses), arms the fault-injecting trainer only for the ticks the scenario
asks for, and closes the container even when a tick fails.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from mlops_lifecycle_lab.adapters.clock import SimulatedClock
from mlops_lifecycle_lab.adapters.models.lightgbm_trainer import LightGbmTrainer
from mlops_lifecycle_lab.application.ports import ModelTrainer
from mlops_lifecycle_lab.application.services.lifecycle import TickRequest
from mlops_lifecycle_lab.bootstrap.container import Container, build_container
from mlops_lifecycle_lab.config import Settings
from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent
from mlops_lifecycle_lab.domain.windows import tick_window
from mlops_lifecycle_lab.entrypoints.drill.faults import PoisonedTrainer
from mlops_lifecycle_lab.entrypoints.drill.report import (
    DrillResult,
    DrillSummary,
    GateReportRecord,
    TickRecord,
    summarize_performance,
)
from mlops_lifecycle_lab.entrypoints.drill.scenarios import (
    Scenario,
    get_scenario,
)

REPLAY_START = datetime(2012, 1, 1)
"""First instant of the replayed 2012 half (D3)."""

KEY_REPLAY_TICK = "replay_tick"
"""KV key shared with the scheduler so a drilled workspace can continue on schedule."""

ContainerFactory = Callable[[Settings, SimulatedClock, ModelTrainer], Container]
"""Builds a ready container from settings, the drill clock and the fault-injecting trainer."""


def _default_container_factory(
    settings: Settings,
    clock: SimulatedClock,
    trainer: ModelTrainer,
) -> Container:
    return build_container(settings, clock=clock, trainer=trainer)


@dataclass(frozen=True, slots=True)
class DrillRun:
    """The result plus the report files written by a run."""

    result: DrillResult
    json_path: Path
    markdown_path: Path


class DrillEngine:
    """Run one scenario against the real (or an injected) composition root."""

    def __init__(
        self,
        container_factory: ContainerFactory | None = None,
        *,
        base_trainer: ModelTrainer | None = None,
    ) -> None:
        self._container_factory = container_factory or _default_container_factory
        self._base_trainer = base_trainer or LightGbmTrainer()

    def run(
        self,
        scenario: str | Scenario,
        *,
        weeks: int | None = None,
        workspace: Path,
        seed: int = 42,
        via: Literal["inprocess", "http"] = "inprocess",
        report_dir: Path | None = None,
        fast: bool = False,
        api_url: str | None = None,
    ) -> DrillResult:
        """Execute a scenario and write `drill-<scenario>.{json,md}`.

        Args:
            scenario: scenario name or `Scenario`.
            weeks: number of weekly ticks; `None` uses the scenario default.
            workspace: operational SQLite/MLflow workspace (created if missing).
            seed: model seed and policy seed.
            via: score through the in-process gateway or the HTTP API.
            report_dir: destination directory (default `<workspace>/reports`).
            fast: use the Null drift-report renderer.
            api_url: override the HTTP API URL when `via="http"`.

        Raises:
            ValueError: for an unknown scenario or an invalid week count.
        """
        resolved = scenario if isinstance(scenario, Scenario) else get_scenario(scenario)
        total_weeks = resolved.default_weeks if weeks is None else weeks
        if total_weeks < 1:
            raise ValueError(f"weeks must be >= 1, got {total_weeks}")

        workspace = Path(workspace)
        destination = Path(report_dir) if report_dir is not None else workspace / "reports"
        overrides: dict[str, object] = {
            "workspace": workspace,
            "seed": seed,
            "fast": fast,
            "serving_mode": via,
        }
        if api_url is not None:
            overrides["api_url"] = api_url
        settings = Settings(**overrides)  # type: ignore[arg-type]

        clock = SimulatedClock(REPLAY_START)
        trainer = PoisonedTrainer(self._base_trainer, label_scale=0.3)
        container = self._container_factory(settings, clock, trainer)
        ticks: list[TickRecord] = []
        gate_reports: list[GateReportRecord] = []
        try:
            container.bootstrap.execute()
            for number in range(1, total_weeks + 1):
                window = tick_window(number, REPLAY_START, 7)
                clock.set(window.end)
                scale = resolved.script.trainer_label_scale(number)
                if scale is None:
                    trainer.disarm()
                else:
                    trainer.arm()
                report = container.lifecycle.execute(
                    TickRequest(
                        tick=number,
                        window=window,
                        interventions=resolved.script.interventions(number),
                    )
                )
                ticks.append(TickRecord.from_report(report))
                for event in report.events:
                    gate = GateReportRecord.from_event(event)
                    if gate is not None:
                        gate_reports.append(gate)
                container.kv.set(KEY_REPLAY_TICK, number)

            events = [event for tick in ticks for event in tick.events]
            as_served, frozen = summarize_performance(
                container.registry,
                container.prediction_log,
                REPLAY_START,
                tick_window(total_weeks, REPLAY_START, 7).end,
            )
            result = DrillResult(
                scenario=resolved.name,
                seed=seed,
                weeks=total_weeks,
                ticks=tuple(ticks),
                gate_reports=tuple(gate_reports),
                summary=DrillSummary(
                    as_served=as_served,
                    counterfactual_frozen_v1=frozen,
                    promotions=_count(events, EventType.CHALLENGER_PROMOTED),
                    rejections=_count(events, EventType.CHALLENGER_REJECTED),
                    rollbacks=_count(events, EventType.ROLLBACK_EXECUTED),
                    quarantined_batches=_count(events, EventType.BATCH_QUARANTINED),
                    versions=len(container.registry.list_versions()),
                ),
            )
            result.write(destination)
            return result
        finally:
            trainer.disarm()
            container.close()


def _count(events: Sequence[LifecycleEvent], event_type: EventType) -> int:
    return sum(1 for event in events if event.type is event_type)


__all__ = [
    "KEY_REPLAY_TICK",
    "REPLAY_START",
    "ContainerFactory",
    "DrillEngine",
    "DrillRun",
]
