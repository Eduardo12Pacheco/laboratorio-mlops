# ADR-0006: Native drift statistics as the contract, Evidently as a report

## Context

Monitoring needs two outputs: numbers the automation and tests can trust, and a
human-readable report for review. Evidently provides a polished HTML report and its own
PSI implementation, but its API is broad, heavyweight and sensitive to the pandas version.
If the lifecycle decision depended on an Evidently object, a library upgrade could change a
gate, and the pure domain would gain a large framework dependency. If the project only
shipped Evidently, reviewers would have no small, property-testable definition of the
statistic.

## Decision

- The **contract** is the project's own pure implementation in `domain/drift.py`:
  `psi_numeric` with 10 reference-quantile bins, outer ±inf edges and proportions clipped
  at 1e-4; `psi_categorical`; thresholds 0.1 (moderate) and 0.2 (drifted); a summary over
  `temp`, `atemp`, `hum`, `windspeed` (numeric) and `season`, `weathersit` (categorical),
  plus prediction-distribution PSI and `numeric_drift_share`.
- Quantile edges use `numpy.quantile(..., method="inverted_cdf")`, anchored on observed
  reference values. Linear interpolation invented interior edges from floating-point noise
  between repeated values and made PSI depend on representation rather than data.
- **Evidently is a renderer**: `Report([DataDriftPreset(method="psi")])` over exactly the
  monitored columns, saved as HTML at selected ticks for humans. `NullReportRenderer` is
  the Null Object used by `--fast` drills and tests.
- The pinned `pandas>=3.0.6,<3.1` range is part of this decision: Evidently 0.7.23 fails to
  import on pandas 3.1.x (`infer_dtype_from_object`). The pin is intentional and checked by
  `uv lock --check`, not a temporary workaround.
- A contract test renders the March-2012 window against the 2011 training reference and
  asserts that native PSI and Evidently agree on the most drifted feature.

## Consequences

- Drift decisions, thresholds and the dashboard are deterministic, fast and independent of
  Evidently; a failed Evidently import cannot block monitoring or training.
- The project owns two implementations of the same idea and must keep them honest; the
  agreement test is the guard.
- HTML reports are large (≈4 MB each). They are generated only for humans, never committed
  blindly, and the two curated copies share one extracted JavaScript bundle to stay inside
  the 5 MB documentation budget.
- Pandas upgrades are gated by Evidently's import compatibility until a newer Evidently
  release supports the newer pandas. See [DATA.md](../DATA.md) for the dependency pin.

## Evidence

- T7 integration: a March-2012 vs 2011-training report rendered 3,956,588 bytes; native
  `psi_numeric(temp) = 1.6952` and Evidently `ValueDrift(temp) = 1.7309` both flag `temp`.
- The drift replay reports a numeric drift share of 0.50–0.75 in every 2012 week while the
  model performs well, which is exactly why this signal is informational (see
  [ADR-0004](0004-label-based-slo-trigger.md)).
- PSI properties: `psi(x, x) = 0`, `psi >= 0`, affine invariance, and monotonic growth over
  mean shifts of 0.25/0.5/1.0/2.0 standard deviations on seeded normals.
- The curated bundle is 4,392,896 bytes: two HTML files plus the shared
  `evidently-dashboard.js`; `scripts/render_drift_reports.py` regenerates it from the
  bundled dataset and the two reference windows.
