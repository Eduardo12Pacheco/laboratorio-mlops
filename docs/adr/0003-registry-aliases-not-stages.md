# ADR-0003: Registry aliases and tags, not model stages

## Context

The model registry needs a lifecycle (`CANDIDATE`, `CHAMPION`, `ARCHIVED`, `REJECTED`,
`ROLLED_BACK`) and stable pointers for serving and rollback. MLflow used to express this
with model stages, but the official documentation states that stages are deprecated since
2.9.0 and will be removed in a future major release. Building the lifecycle on a deprecated
API would make the most visible part of the project fragile and would hide illegal
transitions inside MLflow.

## Decision

- The lifecycle stage is the tag `mll.stage`, with values from the domain
  `ModelStage` enum. `mll.run_id`, both windows, the data fingerprint, the feature-spec
  version, the creation time, the trigger reason and the four metrics travel as further
  `mll.*` tags.
- The moving pointers are MLflow **aliases**: `champion`, `previous-champion` and
  `challenger`. Promotion archives the incumbent, moves `previous-champion`, sets
  `champion` and clears `challenger`; rollback restores `previous-champion`.
- The domain state machine validates every transition before the adapter writes anything.
  The adapter reads with `search_model_versions` (filtered to versions carrying
  `mll.stage`) and never calls the deprecated stage APIs.
- Serving loads through `mlflow.pyfunc.load_model("models:/bike-demand/<version>")`, so the
  registry is flavor-agnostic and a future model flavour does not change the API.

## Consequences

- Promotion is a small sequence of tag/alias writes with a validated transition table; a
  corrupt or partial write is detectable by reading the tags back.
- `previous-champion` makes rollback explicit and testable. A second rollback without an
  intermediate promotion raises `NoPreviousChampionError` instead of guessing.
- The dashboard and CLI can list aliases without importing MLflow internals.
- MLflow UI users see aliases and tags rather than the familiar stage column; the docs and
  CLI explain the mapping.
- The registry contract suite runs against both `InMemoryRegistry` and `MlflowRegistry`, so
  the fake cannot promise semantics the real adapter lacks.

## Evidence

- A source-grep test asserts that `transition_model_version_stage` and `get_latest_versions`
  do not appear in the adapter; `uv run pytest tests/integration -q` is green.
- The registry contract covers promotion, rejection, rollback, illegal transitions,
  `previous-champion` and `list_versions` for both implementations.
- The drill shows the intended sequence in a real MLflow store: `CHALLENGER_PROMOTED` moves
  `champion` from v1 to v2, the final retrain is `REJECTED` without touching the champion,
  and the rollback scenario restores v2 while tagging v3 `ROLLED_BACK`.
- Planning probe on MLflow 3.18 verified alias set/get/delete and `pyfunc.load_model`; the
  registry contract reproduces it in the test suite. See the tasks table in the plan Log.
