# Changelog

All notable changes to this project will be documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Streamlit Control Room with five pages (Overview, Monitoring, Registry, Lifecycle,
  Drill), light/dark palettes that pass WCAG contrast checks, explicit loading/empty/error
  states and status marks that never rely on colour alone.
- Read-only dashboard view models over the existing ports (champion health, trailing
  metrics, PSI matrix, registry aliases and gate evidence, audit timeline, run ledger and
  committed drill reports) plus `mll dashboard` on host port 14501.
- APScheduler replay service: one interval job with `max_instances=1`, a persisted
  `replay_tick` cursor, self-stop at the replay length and clean SIGINT/SIGTERM handling.
- `mll scheduler start --mode replay|once` driving the same lifecycle tick as the drill,
  bootstrapping v1 when the workspace has no champion.
- `mll replay status|reset` to inspect or clear the replay cursor without building the
  full composition root.
- `mll doctor` with hard checks (Python 3.12, runtime imports, a real LightGBM fit,
  SQLite WAL, writable workspace and MLflow SQLite) plus informational Docker and host-port
  checks.
- `mll quickstart` (doctor-lite, bootstrap and the 16-week drift drill) and
  `mll mlflow-ui` on the documented host port.
- Reproducible drill engine for the four S15/S16 scenarios (`drift`, `gate-reject`,
  `bad-batch`, `rollback`) with a simulated clock, replay cursor, frozen-v1
  counterfactual, as-served metrics and byte-identical `drill-report/v1` JSON/Markdown.
- Fault-injection decorator (`PoisonedTrainer`) and Celsius-scale batch corruption,
  isolated from production paths and armed only for the scenario tick.
- Golden expectations with per-tick event sequences, relative metric tolerances and the
  documented ±1 week first-retrain slack; committed goldens for all four scenarios.
- `mll bootstrap`, `mll train`, `mll registry list|show|promote|rollback`,
  `mll monitor snapshot` and `mll drill run` (including `--check`).
- Serving API (S8): typed Pydantic v2 schemas with hourly timestamp and numeric-domain
  validation, `/healthz`, `/readyz`, `/v1/model`, `/v1/predict`, `/v1/predict/batch`,
  `/v1/labels`, `/v1/admin/reload` and Prometheus `/metrics`.
- Hot-reloading `ModelHolder`: the champion artifact is loaded before the atomic swap, a
  failing reload keeps the previous model and increments
  `mll_model_reload_failures_total`, and request ids are echoed and logged.
- `HttpScoringGateway` for HTTP serving mode: 1000-row chunking, exponential-backoff
  retries on connection/5xx errors and 422-to-`ValidationFailedError` mapping.
- `mll serve` and `mll score`, plus worker-thread-safe SQLite access for the API.
- Monitoring snapshot service: trailing labelled metrics for the current champion,
  native PSI drift against its training reference and prediction-distribution PSI from the
  `reference_predictions.json` artifact, persisted with a `SNAPSHOT_COMPUTED` event.
- Evidently PSI report adapter for human-readable drift HTML (Null Object in fast mode)
  and the replay batch source that keeps labels separate from the scoring frame.
- Lifecycle tick use case: batch validation/quarantine, scoring, label ingestion,
  monitoring, probation rollback/pass, retrain trigger, gates, promotion/rejection and a
  retrying run ledger with idempotent per-tick state.
- Typed composition root (`build_container`) wiring the SQLite stores, MLflow
  tracker/registry, LightGBM trainer, in-process gateway, policies and use cases.
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
