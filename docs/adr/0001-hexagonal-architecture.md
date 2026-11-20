# ADR-0001: Hexagonal architecture with enforced dependency direction

## Context

The lab is a lifecycle system, not a notebook: data validation, feature building, training,
registry access, serving, monitoring and scheduling all coexist. Without an explicit
boundary the ML code tends to leak framework imports (MLflow, FastAPI, LightGBM) into the
core, which makes the domain hard to test and easy to break when an adapter changes. The
project standards require ports & adapters with a test that enforces the dependency
direction.

## Decision

Use hexagonal architecture with five layers and a one-way dependency rule:

```
entrypoints  (CLI, API, dashboard, drill)
bootstrap | config | logging_setup   (composition root and cross-cutting wiring)
adapters     (MLflow, SQLite, LightGBM, FastAPI client, APScheduler, Evidently)
application  (ports, DTOs, use cases, pipeline)
domain       (pure: schema, features, metrics, drift, gates, policies, lifecycle)
```

- The domain layer imports only the standard library, `numpy`, `pandas` and `scipy`.
- Application services depend on Protocols ("ports"), never on concrete adapters.
- Adapters may depend on application and domain; adapter sub-packages are independent of
  each other (no adapter imports a sibling adapter).
- The composition root (`bootstrap`) is the only place that instantiates real adapters.
- The rule is executable: `import-linter` contracts live in `pyproject.toml` and run in
  CI through `uv run lint-imports`. `tests/architecture/test_layers.py` also proves the
  check fails on a deliberately planted violation, so the gate cannot silently rot.

## Consequences

- Domain and application tests run without MLflow, a database or network.
- Swapping adapters (e.g. an HTTP scoring gateway in Compose vs the in-process one in the
  drill) is a composition-root change only.
- Slightly more boilerplate: every side effect needs a port. This is accepted because the
  alternative — framework calls sprinkled through use cases — is untestable at this size.
- New cross-layer shortcuts fail CI instead of being discovered in review.

## Evidence

- `[tool.importlinter]` in `pyproject.toml` defines the layers, forbidden-external and
  adapter-independence contracts; T1's acceptance runs `uv run lint-imports` and expects
  `Contracts: 3 kept, 0 broken.`
- The negative test builds a throwaway package whose `domain` imports `adapters` and
  asserts import-linter reports the contract broken.
