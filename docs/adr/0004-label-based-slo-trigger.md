# ADR-0004: Retraining is triggered by a label-based SLO, not by covariate drift

## Context

The system needs a defensible answer to "when do we retrain?". The obvious input is data
drift: compare this week's features with the champion's training window and retrain when
PSI crosses 0.2. That rule is attractive because it needs no labels, and labels are delayed
by a tick. But bike demand is strongly seasonal: temperature and season move every year,
and a model trained on 2011 will always look "drifted" in 2012 even while it predicts well.
A covariate-only rule would retrain almost every week, burning compute and producing noisy
champion/challenger comparisons.

## Decision

- Retraining is triggered by a **label-based** trailing-28-day WAPE breach
  (`WAPE >= MLL_SLO_WAPE`, default 0.30) on predictions made by the current champion, with
  at least 300 labelled rows and at least 28 days since the last retrain attempt.
- Probation suppresses retraining (rollback has priority), and a manual `MANUAL` trigger
  bypasses the SLO check but not the gates.
- Covariate PSI, numeric drift share and prediction PSI are computed, persisted and shown
  on the dashboard, but they never trigger a retrain. They are diagnostic evidence.
- The trigger outcome is a typed `TriggerDecision` (`PERFORMANCE_SLO_BREACH`, `MANUAL`,
  `HEALTHY`, `INSUFFICIENT_LABELS`, `COOLDOWN`, `IN_PROBATION`), evaluated in that order and
  unit-tested as a matrix.

## Consequences

- The automation reacts to what matters operationally (forecast error), not to the season.
- Retraining waits for labels; a fresh promotion cannot be judged until enough labels exist.
- The dashboard must separate the two questions ("is performance bad?" and "did inputs
  move?") instead of collapsing them into one red light.
- The policy needs a cooldown and a probation guard; without them a single noisy week could
  start a retrain storm.
- Drift reports remain useful for humans and for explaining *why* performance might change,
  but a healthy WAPE with `numeric_drift_share = 1.0` is explicitly a no-op in tests.

## Evidence

- The domain policy tests include a snapshot with `numeric_drift_share = 1.0` and healthy
  WAPE and assert that it never triggers.
- The 52-week drift replay shows `numeric_drift_share >= 0.5` in every weekly window of 2012
  (seasonality), while the first SLO breach happens only at tick 13 (end 2012-04-01) with
  WAPE 0.3086 on 671 labelled rows. The committed report is
  [`../results/drill-drift.json`](../results/drill-drift.json).
- The final tick breaches again (WAPE 0.3152) and the challenger is rejected by the gates,
  demonstrating that a trigger is a request to evaluate, not a promotion.
- The SLO of 0.30 leaves headroom over the bootstrap champion's validation WAPE while the
  frozen 2011 model crosses it in late March 2012, which is the intended failure curve.
