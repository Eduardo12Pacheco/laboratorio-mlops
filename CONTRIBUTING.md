# Contributing

This repository is built in small, reviewable increments with a test-first workflow.

## Setup

```bash
uv sync --locked
uv run mll --version
```

## Quality gates

Every change must keep these commands green:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run lint-imports
uv run pytest -q
uv lock --check
```

## Conventions

- Conventional Commits in English, one coherent behaviour per commit.
- Tests first: write the failing test, implement, then refactor.
- Keep the hexagonal dependency direction: `domain` → `application` → `adapters` →
  `bootstrap`/`config` → `entrypoints`. Import-linter enforces it.
- No secrets, no network access in tests, no direct edits to generated lock files by hand:
  run `uv lock` instead.
