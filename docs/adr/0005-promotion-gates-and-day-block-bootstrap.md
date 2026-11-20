# ADR-0005: Six promotion gates with a day-block paired bootstrap

## Context

A retrain trigger answers "the champion may be stale". It does not answer "is this
challenger actually better?". A single aggregate RMSE comparison is easy to game by a lucky
holdout, ignores uncertainty, and can hide a regression in an operationally important
segment. The project needs an explicit, inspectable decision that a reviewer can reproduce
from a committed report.

## Decision

At tick end `T`, set `cut = T − 14 days`. Train the challenger on every labelled row with
`ts < cut` (archive plus ingested production rows) and hold out `[cut, T)`. Re-score the
champion on exactly the same holdout and evaluate all gates, without short-circuiting:

1. `DATA_VALID` — the training frame verdict is not `FAIL`.
2. `ABS_WAPE` — challenger holdout WAPE ≤ 0.30.
3. `BEATS_BASELINE` — challenger RMSE ≤ 0.85 × the naive profile baseline RMSE on the same
   rows.
4. `IMPROVEMENT` — `(champion RMSE − challenger RMSE) / champion RMSE ≥ 0.05`.
5. `SIGNIFICANCE` — the lower bound of the 95% day-block paired bootstrap CI of
   `mean(champion SE − challenger SE)` is > 0, with 2000 resamples and seed 42.
6. `SEGMENT` — challenger RMSE is not worse than the champion by more than 10% in either
   `workingday` segment.

The first champion uses gates 1–3 only. Every gate result (observed value, threshold,
pass/fail, human-readable detail) is persisted as `gate_report.json` and as a
`GATES_EVALUATED` event. The promoted artifact is exactly the evaluated artifact: there is
no refit on the holdout.

The bootstrap resamples **whole calendar days**, not individual rows, because hourly errors
are autocorrelated. The statistic compares squared errors (the RMSE scale), and the domain
implementation is pure, seeded and property-tested.

## Consequences

- A challenger that wins on average but is noisy or regresses a segment is rejected.
- The decision is slower and more expensive than a single RMSE threshold, but it is
  explainable and the cost is trivial at this data size.
- The holdout is not refit, so reported numbers are honest but slightly pessimistic; this
  is documented as a limitation rather than hidden.
- A forced promotion remains possible for incident response, but requires a reason and
  emits `PROMOTION_FORCED`, so it is auditable.
- The gate report format is shared by the application, the drill report and the dashboard.

## Evidence

- The domain gate tests cover the exact boundaries: 0.0499 fails and 0.0500 passes for
  improvement, 0.1001 fails and 0.1000 passes for segment regression, 0.3001 fails and
  0.3000 passes for absolute WAPE, 0.8501 fails and 0.8500 passes for the baseline ratio,
  and a CI lower bound of exactly 0 fails significance.
- The drift replay promotes v2 at tick 13 with challenger RMSE 53.2673, improvement 0.5595,
  significance lower bound 9000.8429 and a worst-segment delta of −0.5528; the raw gate
  table is in [`../results/drill-drift.md`](../results/drill-drift.md).
- The gate-reject replay rejects with exactly `IMPROVEMENT`, `SIGNIFICANCE` and `SEGMENT`
  failing, proving the protocol does not hide a marginal challenger behind an average.
- Hypothesis tests prove `approved == all(results.passed)` and that every gate is always
  evaluated, so a future refactor cannot reintroduce short-circuiting.
