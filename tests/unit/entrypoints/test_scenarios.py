"""The four drill scenarios and their typed intervention scripts (S15, S16)."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_lifecycle_lab.entrypoints.drill.faults import celsius_scale
from mlops_lifecycle_lab.entrypoints.drill.scenarios import (
    SCENARIOS,
    Scenario,
    ScenarioScript,
    get_scenario,
)
from tests.fakes.data import raw_frame


def test_scenario_registry_matches_the_plan() -> None:
    assert set(SCENARIOS) == {"drift", "gate-reject", "bad-batch", "rollback"}
    assert {name: scenario.default_weeks for name, scenario in SCENARIOS.items()} == {
        "drift": 52,
        "gate-reject": 24,
        "bad-batch": 12,
        "rollback": 25,
    }
    assert all(isinstance(scenario, Scenario) for scenario in SCENARIOS.values())


def test_get_scenario_is_case_insensitive_and_rejects_unknown_names() -> None:
    assert get_scenario("Drift").name == "drift"
    with pytest.raises(ValueError, match="unknown drill scenario"):
        get_scenario("nope")


def test_drift_script_has_no_interventions() -> None:
    script = ScenarioScript()

    interventions = script.interventions(13)

    assert interventions.before_validate is None
    assert interventions.before_decide is None
    assert script.trainer_label_scale(13) is None


def test_bad_batch_injects_a_celsius_batch_at_tick_seven() -> None:
    script = ScenarioScript().corrupt_batch_at(7, celsius_scale)
    frame = raw_frame(periods=3)

    interventions = script.interventions(7)
    assert interventions.before_validate is not None
    corrupted = interventions.before_validate(7, None, frame)  # type: ignore[arg-type]
    assert corrupted["temp"].max() > 1.0
    assert script.interventions(8).before_validate is None


def test_manual_retrain_sets_the_manual_trigger_before_the_decision() -> None:
    script = ScenarioScript().manual_retrain_at(23)

    interventions = script.interventions(23)
    assert interventions.before_decide is not None

    class View:
        manual_trigger = False
        force_promotion = False
        promotion_reason = ""

    view = View()
    interventions.before_decide(view)  # type: ignore[arg-type]
    assert view.manual_trigger is True
    assert view.force_promotion is False


def test_forced_poisoned_promotion_arms_the_trainer_only_at_that_tick() -> None:
    script = ScenarioScript().forced_poisoned_promotion_at(23, label_scale=0.3)

    class View:
        manual_trigger = False
        force_promotion = False
        promotion_reason = ""

    view = View()
    interventions = script.interventions(23)
    assert interventions.before_decide is not None
    interventions.before_decide(view)  # type: ignore[arg-type]
    assert view.manual_trigger is True
    assert view.force_promotion is True
    assert view.promotion_reason
    assert script.trainer_label_scale(23) == 0.3
    assert script.trainer_label_scale(22) is None
    assert script.interventions(22).before_decide is None


def test_script_builders_return_new_scripts() -> None:
    base = ScenarioScript()
    changed = base.manual_retrain_at(5)

    assert changed is not base
    assert base.interventions(5).before_decide is None
    assert changed.interventions(5).before_decide is not None


def test_corrupt_batch_at_rejects_invalid_ticks() -> None:
    with pytest.raises(ValueError, match="tick"):
        ScenarioScript().corrupt_batch_at(0, celsius_scale)
    with pytest.raises(ValueError, match="tick"):
        ScenarioScript().manual_retrain_at(-1)


def test_bad_batch_frame_is_still_a_dataframe() -> None:
    frame = raw_frame(periods=2)
    assert isinstance(celsius_scale(frame), pd.DataFrame)
