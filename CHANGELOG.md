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
