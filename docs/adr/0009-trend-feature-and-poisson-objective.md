# ADR-0009: A trend feature and a Poisson objective for growing demand

## Context

Bike demand grew substantially between 2011 (mean 143.8 rentals/hour) and 2012 (mean 234.7).
Calendar and weather features alone describe the shape of demand but not its level. Tree
models cannot extrapolate beyond the labels they have seen, so a model trained on 2011 will
systematically under-predict 2012 unless something restores the level. The planning
experiment compared a frozen model with a retrained model every 14 days under four
configurations; the choice had to be supported by evidence, not intuition.

## Decision

- Add exactly one time feature, `t_days` = fractional days since 2011-01-01 00:00, so the
  model can represent a level trend while still relying on trees for the calendar/weather
  shape. No `yr`, `instant`, `casual`, `registered` or `cnt` feature: those leak the target
  or are database identifiers.
- Use `LGBMRegressor(objective="poisson")` with the fixed hyper-parameters in
  [MODEL_CARD.md](../MODEL_CARD.md). Demand is a non-negative count; the Poisson objective
  matches the target and prevents negative predictions at the source, with a final clip at
  zero for serving.
- Fix the hyper-parameters and seed; no tuning, AutoML or hyper-parameter search. The
  lifecycle and honest retraining are the product, not a benchmark win.
- Keep the naive `(workingday, hour)` profile baseline as a deliberately weak floor for the
  `BEATS_BASELINE` gate, not as a competitive model.

## Consequences

- Retraining on recent labels is effective: the trend feature lets the model use the newer
  level instead of only learning seasonal shape.
- The trend is an assumption. A structural break (new pricing, a pandemic, a new city) can
  make `t_days` harmful; the SLO trigger and probation/rollback are safety nets, and the
  limitation is stated in the model card.
- Poisson predictions are always non-negative; the serving path still clips at zero as a
  defensive measure for adapter or artifact changes.
- Without the trend feature, retraining would barely help; a future change must re-run the
  drift drill and the quality gates, not just the unit tests.

## Evidence

- Planning experiment (28-day forward RMSE, frozen → retrained): plain features 131.6 → 125.3
  (ratio 0.95); plus `t_days` 109.3 → 67.6 (0.62); plus `t_days` and Poisson 114.0 → 64.6
  (0.57). Recency weighting did not help.
- The committed 52-week drift replay: as-served RMSE 70.7197 vs frozen-v1 109.9249 (ratio
  0.6433), as-served WAPE 0.1949 vs 0.3165, with exactly one promotion and one rejected
  final retrain. See [`../results/results-table.md`](../results/results-table.md).
- The bootstrap quality test exercises the real SQLite/MLflow/LightGBM stack and asserts the
  S5 bounds (RMSE, WAPE, baseline ratio and the ±2.0 reproduction tolerance), plus
  `np.array_equal` on two identical trainings.
- Feature parity is tested on all 17,379 rows: derived `mnth`, `hr` and `weekday` equal the
  dataset's original columns, and `build_features` never emits a leakage column.
