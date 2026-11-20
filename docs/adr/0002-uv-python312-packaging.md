# ADR-0002: uv, Python 3.12 and a src-layout package

## Context

The project must install reproducibly on Linux, Windows and macOS, ship an L1 CLI path
(`uv tool install`) and an L2 Docker Compose stack, and keep a lockfile that CI can verify
without resolving dependencies again. The ML stack (MLflow, LightGBM, Evidently) is
sensitive to interpreter and dependency versions; the planning probes pinned a working set
on Python 3.12.

## Decision

- Support only CPython 3.12 (`requires-python = ">=3.12,<3.13"`), recorded in
  `.python-version`. Other interpreters are rejected rather than half-supported.
- Use **uv** for environments, locking and builds. `uv.lock` is committed and CI runs
  `uv sync --locked` plus `uv lock --check`; the lock is never edited by hand.
- Use a `src/` layout with a single package `mlops_lifecycle_lab`, built by hatchling, and
  expose the CLI as `mll = "mlops_lifecycle_lab.entrypoints.cli.app:main"`.
- Pin the dependency ranges from the plan (e.g. `pandas>=3.0.6,<3.1` because Evidently
  0.7.23 fails to import on pandas 3.1; `apscheduler<4` because 4.x is alpha). Pins are
  documented next to the dependency so future upgrades are deliberate.
- Keep dev and capture tooling in dependency groups (`dev` default, `capture` on demand)
  so the runtime image and the L1 install stay lean.

## Consequences

- A single lockfile reproduces the same dependency graph for CI, Docker and local runs.
- The narrow Python range means new interpreters are adopted with an explicit lock update.
- `uv tool install <wheel>` works without a virtualenv for the L1 path, and the same wheel
  feeds the Compose image.
- `pandas-stubs` is a dev dependency only and cannot leak into runtime.

## Evidence

- Planning probe: a universal `uv lock` of the whole stack on Python 3.12 resolved 182
  packages with no conflicts; `uv sync --locked` plus MLflow/LightGBM/Evidently smoke
  calls passed on pandas 3.0.6 and failed at Evidently import on pandas 3.1.1.
- T1 acceptance: `uv sync --locked`, `uv lock --check` and `uv run mll --version` all
  succeed, and the wheel includes the `_data` package for later tasks.
