# Contributing

This repository is built in small, reviewable increments with a test-first workflow.
Every change must keep the documented quality gates green before it is pushed to `main`.

## Setup

```bash
uv sync --locked
uv run mll --version
uv run mll doctor
```

Python 3.12 is required. Linux and Windows are supported directly; on macOS install
LightGBM's OpenMP runtime once with `brew install libomp`. Docker is only needed for the
L2 Compose path.

## Installers

- `./install.sh --check` validates Docker >= 24, Compose v2, a reachable Linux engine,
  free host ports 14000/14080/14501 and at least 4 GB of free disk without side effects.
- `./install.sh` builds and starts the stack; `./install.sh --uninstall [--yes]` removes
  the stack, its volumes and local images but keeps `.env`.
- `./install.sh --check --l1` validates the Docker-free path (`uv`, Python 3.12 and
  `mll doctor`).
- `install.ps1` mirrors the same flags for Windows PowerShell 5.1 and PowerShell 7
  (`-Check`, `-L1`, `-Uninstall`, `-Yes`, `-NoStart`, `-ProjectName`).

Exit codes are shared: `0` ok, `1` prerequisites failed, `2` usage error, `3` install
failed, `4` uninstall aborted.

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run lint-imports
uv run pytest --cov=mlops_lifecycle_lab --cov-fail-under=85 -m "not docker and not ui" -q
uv lock --check
uvx --from shellcheck-py shellcheck install.sh scripts/*.sh
uvx check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/*.yml
```

The full suite includes the slow drill tests. `tests/installers` drives the shell
installer through `PATH` shims, so it never touches the real Docker daemon. The
PowerShell parse test runs when `pwsh` is installed and is skipped with a reason
otherwise; the static contract still runs everywhere.

## Conventions

- Conventional Commits in English, one coherent behaviour per commit, tests and docs with
  the behaviour. No AI attribution lines.
- Test-first: write the failing test, implement, then refactor. Never weaken or delete a
  test to get green.
- Keep the hexagonal dependency direction: `domain` → `application` → `adapters` →
  `bootstrap`/`config` → `entrypoints`. Import-linter enforces it.
- No secrets, no network access in tests, and no hand-edited lock files: run `uv lock`.
- Structured logs and typed settings (`MLL_*` environment variables) are the only
  configuration surface; document new variables in `docs/OPERATIONS.md`.

## Releases

`main` is always deployable. A release is a `v*` tag whose version matches
`pyproject.toml`. The `Release` workflow builds the wheel, the sdist and the reproducible
Compose bundle (`scripts/build_compose_bundle.py`), writes `SHA256SUMS`, generates notes
from Conventional Commits with git-cliff and publishes the GitHub release. Never move or
delete a published tag.
