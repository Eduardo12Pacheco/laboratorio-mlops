# Model card — bike-demand champion

This card describes the production model family used by the lab: a LightGBM regressor
predicting hourly bike rentals (`cnt`) for Washington DC, plus the champion/challenger
lifecycle around it. It follows the spirit of the Model Cards for Model Reporting paper;
the lifecycle, not model sophistication, is the product.

## Intended use

- Demonstrate an end-to-end, local-first MLOps loop on a real temporal dataset.
- Serve hourly demand predictions through the FastAPI endpoints.
- Compare a champion and a challenger on a labelled holdout with explicit gates.
- Monitor trailing performance and covariate drift, and trigger a retrain on a
  label-based SLO breach.

Out of scope: real-time demand forecasting for operations, capacity planning, pricing,
person-level decisions, or any hosted/production service. The replay's "production" is the
2012 half of the bundled dataset ([ADR-0009](adr/0009-trend-feature-and-poisson-objective.md)).

## Data

| Item | Value |
|---|---|
| Dataset | UCI Bike Sharing Dataset (hourly), Fanaee-T 2013 |
| DOI | [10.24432/C5W894](https://doi.org/10.24432/C5W894) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Bundled artifact | `src/mlops_lifecycle_lab/_data/hour.csv.gz` |
| Rows / window | 17,379 rows, 2011-01-01 00:00 → 2012-12-31 23:00 |
| Target | `cnt` (casual + registered) |
| Attribution | `_data/ATTRIBUTION.txt` and [DATA.md](DATA.md) |

The data is bundled and validated offline. The only labels are historical `cnt`; the
replay scores a week before its labels arrive and then ingests them by `entity_id`. The
dataset is `WARN` (not `FAIL`) because 22 rows on 2011-03-10 have `hum = 0`; that warning
is visible in every validation report and never silently ignored.

## Features

The twelve feature columns, in order:

```text
season, mnth, hr, holiday, weekday, workingday, weathersit,
temp, atemp, hum, windspeed, t_days
```

`t_days` is fractional days since 2011-01-01 00:00 and captures the system's growth; without
it trees cannot extrapolate the 2012 demand level, and retraining barely helps. `yr`,
`instant`, `casual`, `registered` and `cnt` are leakage/time proxies and never become
features. `mnth`, `hr` and `weekday` are derived from the timestamp and proven equal to the
original UCI columns on all 17,379 rows. Training, batch scoring and the API share the same
pure `build_features` function, so train/serve parity is enforced by construction.

## Model and training protocol

```text
LGBMRegressor(
  objective="poisson", n_estimators=400, learning_rate=0.05, num_leaves=31,
  min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
  random_state=42, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1,
)
```

- **Bootstrap champion (v1):** train `2011-01-01 ≤ ts < 2011-10-01` (6,442 rows),
  validate `2011-10-01 ≤ ts < 2012-01-01` (2,203 rows). First-champion gates are
  `DATA_VALID`, `ABS_WAPE` and `BEATS_BASELINE`; the committed quality test
  (`tests/quality/test_bootstrap_champion.py`) checks RMSE, WAPE, the baseline ratio and
  the ±2.0 RMSE reproduction tolerance through the real adapters.
- **Challenger (retrain) at tick end `T`:** `cut = T − 14 days`; train on every labelled row
  with `ts < cut` (archive plus ingested production labels), hold out `[cut, T)`, and
  re-score the champion on the same holdout. Six gates are evaluated without
  short-circuiting; the promoted artifact is exactly the evaluated artifact (no refit).
- **Rollback/probation:** a promoted challenger is watched for 21 days; with at least 150
  labelled rows a trailing WAPE above the SLO triggers automatic rollback to
  `previous-champion`.
- **Trigger policy:** a label-based 28-day WAPE breach only. Covariate PSI is reported but
  never triggers a retrain ([ADR-0004](adr/0004-label-based-slo-trigger.md)).

The baseline used by `BEATS_BASELINE` is deliberately weak: the mean `cnt` by
`(workingday, hr)` over the same training rows. It is a floor, not a competitor.

## Metrics

The table below is generated from the four committed `drill-report/v1` documents by
`scripts/render_results_table.py`. "As-served" is what the system actually predicted over
each replay; "frozen-v1" is the counterfactual where the bootstrap champion is never
replaced. Ratios below 1.0 mean the lifecycle beat the never-retrained model.

<!-- results:start -->

| Scenario | Weeks | Versions | Promotions | Rejections | Rollbacks | Quarantined | As-served RMSE | As-served MAE | As-served WAPE | Frozen-v1 RMSE | Frozen-v1 WAPE | RMSE ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| drift | 52 | 3 | 1 | 1 | 0 | 0 | 70.7197 | 45.8888 | 0.1949 | 109.9249 | 0.3165 | 0.6433 |
| gate-reject | 24 | 3 | 1 | 1 | 0 | 0 | 66.8555 | 41.8600 | 0.1988 | 88.8282 | 0.2783 | 0.7526 |
| bad-batch | 12 | 1 | 0 | 0 | 0 | 1 | 70.3356 | 43.4195 | 0.2654 | 70.3356 | 0.2654 | 1.0000 |
| rollback | 25 | 3 | 2 | 0 | 1 | 0 | 84.8111 | 48.6398 | 0.2287 | 88.8369 | 0.2763 | 0.9547 |

_Generated by `uv run python scripts/render_results_table.py` from `docs/results/drill-*.json` (seed 42, schema `drill-report/v1`); do not edit by hand._

<!-- results:end -->

The drift scenario is the headline: 52 weekly ticks, one promotion at tick 13, one
rejection at tick 52 and no rollback. The full timeline, gate reports and per-tick metrics
are in [`results/drill-drift.md`](results/drill-drift.md); the raw JSON is the source of
truth. The two curated Evidently drift reports (trigger and final tick) are committed under
[`assets/reports/`](assets/reports/).

## Evaluation caveats

- Metrics come from a weekly replay, not independent production traffic. Windows overlap
  by design (trailing 28 days), so consecutive snapshots are correlated.
- Labels are delayed by one tick; a snapshot only evaluates labelled rows from the current
  champion. A newly promoted model needs some labels before its probation can conclude.
- The holdout is not refit after selection. This is honest but slightly pessimistic:
  light hyper-parameter adaptation to the holdout would improve reported numbers and was
  deliberately rejected.
- The trend feature extrapolates growth. A structural break (new city, new pricing,
  pandemic) would invalidate it; the SLO trigger and rollback are the safety net, not a
  guarantee.
- Cross-platform floating-point differences can move the retrain tick by one week; the
  golden tests assert event order and tolerances, not bit-equality across operating
  systems.

## Limitations

- No cloud, Kubernetes, managed registry or auth system beyond the optional admin API key.
- No live feed: "production" is the dataset's 2012 half replayed week by week.
- No feature store product, AutoML, hyper-parameter search or deep learning.
- Champion/challenger is evaluated offline on a labelled holdout; shadow/canary traffic
  splitting is out of scope.
- The Prometheus endpoint is exposed; scraping and Grafana are not bundled.
- Model artifacts are trusted local state: loading one executes pickle-based code. Only use
  registries and artifacts you created or trust. See [SECURITY.md](../SECURITY.md).

## Ethical considerations

The dataset is aggregate hourly bike-rental counts, with no personal data, no faces and no
individual trajectories. Predictions describe system demand; they should not be used to
rank, price or exclude people. The replay's historical demand reflects 2011–2012
Washington DC and may encode pre-existing service patterns; any real deployment would need
an equity and coverage review that this lab does not perform.

## Reproducibility

```bash
uv run mll data verify
uv run mll validate --bundled --json
uv run mll drill run --scenario drift --weeks 52 --workspace "$(mktemp -d)" \
  --report-dir "$(mktemp -d)"
uv run python scripts/render_results_table.py --check
uv run python scripts/check_docs.py
```
