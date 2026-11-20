"""Day-block paired bootstrap for the promotion significance gate.

Hourly errors are autocorrelated, so resampling individual rows would overstate
significance (D6). The bootstrap resamples whole calendar days and recomputes the mean of
`champion_error - challenger_error` on each draw. Callers pass squared errors when they
want the RMSE-style comparison used by gate G5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def day_block_bootstrap_ci(
    timestamps: ArrayLike,
    errors_champion: ArrayLike,
    errors_challenger: ArrayLike,
    resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for the paired error difference.

    Args:
        timestamps: one timestamp per observation; only the calendar date is used.
        errors_champion: per-row error (or squared error) of the champion.
        errors_challenger: per-row error of the challenger, aligned with the champion.
        resamples: number of bootstrap draws.
        seed: RNG seed; the same seed always returns the same interval.
        confidence: two-sided confidence level in `(0, 1)`.

    Returns:
        `(lower, upper)` percentile bounds of the bootstrapped mean difference.

    Raises:
        ValueError: on empty or differently sized inputs and invalid parameters.
    """
    if resamples <= 0:
        raise ValueError(f"resamples must be positive, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    champion = np.asarray(errors_champion, dtype=np.float64)
    challenger = np.asarray(errors_challenger, dtype=np.float64)
    if champion.ndim != 1 or challenger.ndim != 1:
        raise ValueError("errors must be one-dimensional")
    if champion.size != challenger.size:
        raise ValueError(
            f"length mismatch: champion has {champion.size} values, "
            f"challenger has {challenger.size}"
        )
    if champion.size == 0:
        raise ValueError("cannot bootstrap empty error vectors")
    if not (np.isfinite(champion).all() and np.isfinite(challenger).all()):
        raise ValueError("errors must contain finite values")

    moments = pd.to_datetime(pd.Series(timestamps))
    if len(moments) != champion.size:
        raise ValueError(
            f"length mismatch: timestamps has {len(moments)} values, errors have {champion.size}"
        )

    frame = pd.DataFrame(
        {
            "day": moments.dt.normalize().to_numpy(),
            "difference": champion - challenger,
        }
    )
    grouped = frame.groupby("day", sort=True)["difference"].agg(["sum", "count"])
    day_sums = grouped["sum"].to_numpy(dtype=np.float64)
    day_counts = grouped["count"].to_numpy(dtype=np.float64)
    if day_sums.size == 0:
        raise ValueError("timestamps contain no usable calendar day")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, day_sums.size, size=(resamples, day_sums.size))
    means = day_sums[draws].sum(axis=1) / day_counts[draws].sum(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(lower), float(upper)
