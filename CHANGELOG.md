# Changelog

All notable changes to this project will be documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold: packaging with uv, lint/type/architecture gates, MIT license and
  community health files.
- Typed settings (`MLL_*` environment variables), structlog configuration and the
  `mll --version` entry point.
- Initial CI workflow running lint, types, architecture contracts and the test suite on
  Linux and Windows.
- Bundled UCI Bike Sharing dataset with a provenance manifest, CC BY 4.0 attribution and
  `mll data verify|info|fetch` (zip and CSV checksums verified before writing).
- Domain schema and half-open time windows, plus the pure data-contract validation engine
  exposed as `mll validate` (PASS/WARN/FAIL with per-rule violations).
- Shared feature pipeline (twelve columns including the `t_days` trend feature) and the
  regression metric contract (RMSE/MAE/WAPE/bias) used by every later report.
- Naive profile baseline (the promotion-gate floor), native PSI drift statistics for the
  monitored numeric/categorical features and the day-block paired bootstrap used by the
  significance gate.
- Model lifecycle state machine with five stages, a validated transition table and
  immutable model versions, plus the domain event audit log with JSON round-trip.
- Champion/challenger promotion gates G1-G6, the label-based retrain trigger policy
  (probation veto, manual override, cooldown and SLO breach) and the probation policy
  that decides between continuing, passing and rolling back.
- Application ports, frozen DTOs and the model-version tag codec shared with registry
  adapters, plus the retrying `Pipeline`/`Step` runner backed by a run ledger.
- Training, evaluation, promotion/rejection, rollback, bootstrap and scoring use cases:
  deterministic data fingerprints, S6 lineage, gate reports, probation state, idempotent
  prediction logging and the in-process scoring gateway.
- In-memory doubles for every port and shared registry/prediction-log contract suites
  that the SQLite and MLflow adapters will run unchanged.
- SQLite operational store with ordered migrations, WAL/busy-timeout pragmas, the
  idempotent prediction log, event/snapshot/key-value stores, the run ledger and the
  archive-plus-production labeled-data repository.
- LightGBM trainer with the fixed S5 Poisson hyper-parameters and deterministic fits,
  plus wall-clock and simulated replay clocks.
- MLflow experiment tracker (params, metrics, JSON artifacts and registered models) and
  the tags-and-aliases model registry; the shared registry/prediction-log contract suites
  now run against the real adapters, and the bootstrap quality gate proves the S5 numbers.
