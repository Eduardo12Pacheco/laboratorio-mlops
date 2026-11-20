"""Day-block paired bootstrap: block resampling, determinism and verdict behaviour."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from mlops_lifecycle_lab.domain.stats import day_block_bootstrap_ci


def timestamps(days: int, rows_per_day: int = 24) -> list[datetime]:
    start = datetime(2012, 1, 1)
    return [start + timedelta(hours=row) for row in range(days * rows_per_day)]


def test_identical_error_vectors_give_a_zero_interval() -> None:
    errors = np.linspace(0.0, 10.0, 48)

    lower, upper = day_block_bootstrap_ci(timestamps(2), errors, errors.copy())

    assert (lower, upper) == (0.0, 0.0)


def test_a_strictly_better_challenger_has_a_positive_lower_bound() -> None:
    champion = np.full(48, 4.0)
    challenger = np.full(48, 1.0)

    lower, upper = day_block_bootstrap_ci(timestamps(2), champion, challenger)

    assert lower > 0.0
    assert upper > lower


def test_same_seed_is_reproducible() -> None:
    rng = np.random.default_rng(7)
    champion = rng.uniform(0.0, 10.0, 240)
    challenger = rng.uniform(0.0, 10.0, 240)

    first = day_block_bootstrap_ci(timestamps(10), champion, challenger, seed=123)
    second = day_block_bootstrap_ci(timestamps(10), champion, challenger, seed=123)

    assert first == second
    assert first[0] <= first[1]


def test_a_different_seed_changes_the_interval_edges() -> None:
    rng = np.random.default_rng(11)
    champion = rng.uniform(0.0, 10.0, 480)
    challenger = rng.uniform(0.0, 10.0, 480)

    first = day_block_bootstrap_ci(timestamps(20), champion, challenger, seed=1)
    second = day_block_bootstrap_ci(timestamps(20), champion, challenger, seed=2)

    assert first != second


def test_whole_days_are_resampled_not_individual_rows() -> None:
    day_one = np.full(24, 10.0)
    day_two = np.full(24, -10.0)
    errors = np.concatenate([day_one, day_two])

    lower, upper = day_block_bootstrap_ci(
        timestamps(2), errors, np.zeros_like(errors), resamples=500, seed=42
    )

    # With two equal-sized blocks every resample averages whole days, so the interval
    # spans exactly the two day-level means.
    assert lower == pytest.approx(-10.0)
    assert upper == pytest.approx(10.0)


def test_a_single_day_collapses_to_the_mean_difference() -> None:
    champion = np.array([1.0, 2.0, 3.0, 4.0])
    challenger = np.zeros(4)

    lower, upper = day_block_bootstrap_ci(timestamps(1, rows_per_day=4), champion, challenger)

    assert lower == pytest.approx(2.5)
    assert upper == pytest.approx(2.5)


def test_confidence_widens_or_keeps_the_interval() -> None:
    rng = np.random.default_rng(3)
    champion = rng.uniform(0.0, 10.0, 720)
    challenger = rng.uniform(0.0, 10.0, 720)

    narrow = day_block_bootstrap_ci(timestamps(30), champion, challenger, confidence=0.80)
    wide = day_block_bootstrap_ci(timestamps(30), champion, challenger, confidence=0.99)

    assert wide[0] <= narrow[0]
    assert wide[1] >= narrow[1]


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        day_block_bootstrap_ci(timestamps(1), np.zeros(3), np.zeros(2))


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        day_block_bootstrap_ci([], [], [])


def test_invalid_confidence_and_resamples_raise() -> None:
    with pytest.raises(ValueError, match="confidence"):
        day_block_bootstrap_ci(timestamps(1), np.ones(24), np.zeros(24), confidence=1.5)
    with pytest.raises(ValueError, match="resamples"):
        day_block_bootstrap_ci(timestamps(1), np.ones(24), np.zeros(24), resamples=0)


def test_timestamps_may_be_strings() -> None:
    labels = [f"2012-01-0{day} 00:00:00" for day in (1, 2)]

    assert day_block_bootstrap_ci(labels, [1.0, 2.0], [1.0, 2.0]) == (0.0, 0.0)
