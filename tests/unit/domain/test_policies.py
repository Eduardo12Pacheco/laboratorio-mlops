"""Retrain trigger, promotion and probation policies: decision matrices and ordering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pytest

from mlops_lifecycle_lab.config import Settings
from mlops_lifecycle_lab.domain.gates import GATE_NAMES, GateContext
from mlops_lifecycle_lab.domain.policies import (
    PolicyConfig,
    ProbationPolicy,
    ProbationVerdict,
    PromotionPolicy,
    RetrainTriggerPolicy,
    TriggerOutcome,
    TriggerReason,
)


@dataclass(frozen=True)
class Snapshot:
    """Minimal structural stand-in for the application MonitoringSnapshot."""

    n_rows: int
    wape: float | None
    numeric_drift_share: float = 0.0


NOW = datetime(2012, 4, 1)


def trigger(
    snapshot: Snapshot,
    *,
    last_retrain_at: datetime | None = NOW - timedelta(days=28),
    now: datetime = NOW,
    in_probation: bool = False,
    manual: bool = False,
) -> TriggerOutcome:
    decision = RetrainTriggerPolicy().evaluate(
        snapshot,
        last_retrain_at=last_retrain_at,
        now=now,
        in_probation=in_probation,
        manual=manual,
    )
    return decision.outcome


def test_the_trigger_matrix_follows_the_documented_order() -> None:
    assert trigger(Snapshot(299, 0.0)) is TriggerOutcome.INSUFFICIENT_LABELS
    assert trigger(Snapshot(300, 0.2999)) is TriggerOutcome.HEALTHY
    assert trigger(Snapshot(300, 0.3000)) is TriggerOutcome.PERFORMANCE_SLO_BREACH
    assert trigger(Snapshot(300, 0.3001)) is TriggerOutcome.PERFORMANCE_SLO_BREACH
    assert (
        trigger(Snapshot(1000, 0.10), last_retrain_at=NOW - timedelta(days=27))
        is TriggerOutcome.COOLDOWN
    )
    assert trigger(Snapshot(1000, 0.10), in_probation=True) is TriggerOutcome.IN_PROBATION
    assert trigger(Snapshot(1000, 0.10), manual=True) is TriggerOutcome.MANUAL


def test_in_probation_wins_over_every_other_reason() -> None:
    assert (
        trigger(
            Snapshot(0, 0.99),
            in_probation=True,
            manual=True,
            last_retrain_at=NOW,
        )
        is TriggerOutcome.IN_PROBATION
    )


def test_manual_wins_over_insufficient_labels_and_cooldown() -> None:
    assert trigger(Snapshot(0, None), manual=True, last_retrain_at=NOW) is TriggerOutcome.MANUAL


def test_a_breach_inside_the_cooldown_is_suppressed() -> None:
    assert (
        trigger(Snapshot(671, 0.3086), last_retrain_at=NOW - timedelta(days=27))
        is TriggerOutcome.COOLDOWN
    )


def test_the_cooldown_expires_exactly_on_the_configured_day() -> None:
    assert (
        trigger(Snapshot(671, 0.3086), last_retrain_at=NOW - timedelta(days=28))
        is TriggerOutcome.PERFORMANCE_SLO_BREACH
    )


def test_no_previous_retrain_means_no_cooldown() -> None:
    assert (
        trigger(Snapshot(671, 0.3086), last_retrain_at=None)
        is TriggerOutcome.PERFORMANCE_SLO_BREACH
    )


def test_covariate_drift_alone_never_triggers_a_retrain() -> None:
    snapshot = Snapshot(n_rows=1000, wape=0.10, numeric_drift_share=1.0)

    decision = RetrainTriggerPolicy().evaluate(
        snapshot, last_retrain_at=None, now=NOW, in_probation=False
    )

    assert decision.outcome is TriggerOutcome.HEALTHY
    assert decision.trigger is None
    assert decision.should_retrain is False


def test_the_decision_carries_the_trigger_reason() -> None:
    policy = RetrainTriggerPolicy()

    breach = policy.evaluate(Snapshot(400, 0.35), last_retrain_at=None, now=NOW, in_probation=False)
    manual = policy.evaluate(
        Snapshot(400, 0.10), last_retrain_at=None, now=NOW, in_probation=False, manual=True
    )
    healthy = policy.evaluate(
        Snapshot(400, 0.10), last_retrain_at=None, now=NOW, in_probation=False
    )

    assert breach.trigger is TriggerReason.PERFORMANCE_SLO_BREACH
    assert breach.should_retrain is True
    assert manual.trigger is TriggerReason.MANUAL
    assert manual.should_retrain is True
    assert healthy.trigger is None
    assert healthy.should_retrain is False


def test_a_missing_wape_does_not_trigger() -> None:
    decision = RetrainTriggerPolicy().evaluate(
        Snapshot(400, None), last_retrain_at=None, now=NOW, in_probation=False
    )

    assert decision.outcome is TriggerOutcome.INSUFFICIENT_LABELS
    assert decision.trigger is None


def test_the_probation_matrix_follows_the_documented_rules() -> None:
    promoted_at = datetime(2012, 5, 20)
    policy = ProbationPolicy()

    assert (
        policy.evaluate(Snapshot(149, 0.31), promoted_at, promoted_at + timedelta(days=1))
        is ProbationVerdict.CONTINUE
    )
    assert (
        policy.evaluate(Snapshot(150, 0.31), promoted_at, promoted_at + timedelta(days=1))
        is ProbationVerdict.ROLLBACK
    )
    assert (
        policy.evaluate(Snapshot(150, 0.30), promoted_at, promoted_at + timedelta(days=1))
        is ProbationVerdict.CONTINUE
    )
    assert (
        policy.evaluate(Snapshot(500, 0.10), promoted_at, promoted_at + timedelta(days=20))
        is ProbationVerdict.CONTINUE
    )
    assert (
        policy.evaluate(Snapshot(500, 0.10), promoted_at, promoted_at + timedelta(days=21))
        is ProbationVerdict.PASSED
    )
    assert (
        policy.evaluate(Snapshot(500, 0.10), promoted_at, promoted_at + timedelta(days=30))
        is ProbationVerdict.PASSED
    )


def test_a_breach_on_the_last_probation_day_rolls_back() -> None:
    promoted_at = datetime(2012, 5, 20)

    verdict = ProbationPolicy().evaluate(
        Snapshot(500, 0.31), promoted_at, promoted_at + timedelta(days=21)
    )

    assert verdict is ProbationVerdict.ROLLBACK


def test_probation_passes_at_the_deadline_without_a_breach() -> None:
    promoted_at = datetime(2012, 5, 20)

    verdict = ProbationPolicy().evaluate(
        Snapshot(0, None), promoted_at, promoted_at + timedelta(days=21)
    )

    assert verdict is ProbationVerdict.PASSED


def test_probation_config_thresholds_are_honoured() -> None:
    config = PolicyConfig(probation_days=5, probation_min_rows=10, slo_wape=0.2)
    policy = ProbationPolicy(config)
    promoted_at = datetime(2012, 5, 20)

    assert (
        policy.evaluate(Snapshot(9, 0.5), promoted_at, promoted_at + timedelta(days=1))
        is ProbationVerdict.CONTINUE
    )
    assert (
        policy.evaluate(Snapshot(10, 0.5), promoted_at, promoted_at + timedelta(days=1))
        is ProbationVerdict.ROLLBACK
    )
    assert (
        policy.evaluate(Snapshot(500, 0.1), promoted_at, promoted_at + timedelta(days=5))
        is ProbationVerdict.PASSED
    )


def gate_context(
    *,
    champion_errors: list[float] | None,
    challenger_errors: list[float],
    actual: float = 100.0,
    baseline_rmse: float = 100.0,
    is_first_champion: bool = False,
) -> GateContext:
    count = len(challenger_errors)
    return GateContext(
        timestamps=[NOW + timedelta(hours=offset) for offset in range(count)],
        actuals=np.full(count, actual),
        champion_errors=None
        if champion_errors is None
        else np.asarray(champion_errors, dtype=np.float64),
        challenger_errors=np.asarray(challenger_errors, dtype=np.float64),
        segments=np.zeros(count, dtype=np.int64),
        baseline_rmse=baseline_rmse,
        training_verdict="PASS",
        is_first_champion=is_first_champion,
    )


def test_the_first_champion_is_evaluated_with_gates_one_to_three_only() -> None:
    context = gate_context(
        champion_errors=None,
        challenger_errors=[20.0] * 48,
        is_first_champion=True,
    )

    decision = PromotionPolicy().evaluate(context)

    assert [result.name for result in decision.report.results] == list(GATE_NAMES[:3])
    assert decision.approved is True


def test_a_later_challenger_is_evaluated_with_all_six_gates() -> None:
    context = gate_context(
        champion_errors=[40.0] * 48,
        challenger_errors=[20.0] * 48,
        baseline_rmse=100.0,
    )

    decision = PromotionPolicy().evaluate(context)

    assert [result.name for result in decision.report.results] == list(GATE_NAMES)
    assert decision.report.approved is True
    assert decision.reason == "all gates passed"


def test_a_rejected_challenger_lists_its_failing_gates_in_order() -> None:
    context = gate_context(
        champion_errors=[100.0] * 48,
        challenger_errors=[120.0] * 48,
        actual=1000.0,
        baseline_rmse=200.0,
    )

    decision = PromotionPolicy().evaluate(context)

    assert decision.approved is False
    assert decision.report.failed_gates == (
        "IMPROVEMENT",
        "SIGNIFICANCE",
        "SEGMENT",
    )
    assert decision.reason == "IMPROVEMENT, SIGNIFICANCE, SEGMENT"


def test_policy_config_defaults_mirror_the_settings_defaults() -> None:
    settings = Settings()
    config = PolicyConfig.from_settings(settings)

    assert config.slo_wape == settings.slo_wape
    assert config.abs_wape_max == settings.abs_wape_max
    assert config.baseline_max_ratio == settings.baseline_max_ratio
    assert config.improvement_min == settings.improvement_min
    assert config.segment_max_regression == settings.segment_max_regression
    assert config.bootstrap_resamples == settings.bootstrap_resamples
    assert config.min_labeled_rows == settings.min_labeled_rows
    assert config.cooldown_days == settings.cooldown_days
    assert config.holdout_days == settings.holdout_days
    assert config.probation_days == settings.probation_days
    assert config.probation_min_rows == settings.probation_min_rows
    assert config.seed == settings.seed


def test_policy_config_reads_overridden_settings() -> None:
    settings = Settings(
        slo_wape=0.25,
        min_labeled_rows=123,
        cooldown_days=7,
        seed=99,
    )
    config = PolicyConfig.from_settings(settings)

    assert config.slo_wape == 0.25
    assert config.min_labeled_rows == 123
    assert config.cooldown_days == 7
    assert config.seed == 99


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PolicyConfig(slo_wape=0.0),
        lambda: PolicyConfig(abs_wape_max=0.0),
        lambda: PolicyConfig(baseline_max_ratio=0.0),
        lambda: PolicyConfig(improvement_min=-0.1),
        lambda: PolicyConfig(segment_max_regression=-0.1),
        lambda: PolicyConfig(bootstrap_resamples=0),
        lambda: PolicyConfig(min_labeled_rows=0),
        lambda: PolicyConfig(cooldown_days=0),
        lambda: PolicyConfig(probation_days=0),
        lambda: PolicyConfig(probation_min_rows=0),
        lambda: PolicyConfig(holdout_days=0),
    ],
)
def test_invalid_policy_configuration_is_rejected(
    factory: Callable[[], PolicyConfig],
) -> None:
    with pytest.raises(ValueError, match="must be"):
        factory()
