# ADR-0007: APScheduler for replay and an in-house run ledger

## Context

One lifecycle tick per simulated week is all the scheduling this project needs. Airflow,
Prefect and Dagster each bring a database, a web server, workers, a UI and their own
concepts; adopting one would add more moving parts than the lifecycle itself and would make
the local-first Compose stack harder to reason about. The project still needs a durable
cursor, non-overlapping ticks, clean shutdown and an audit trail of every attempt.

## Decision

- Use APScheduler 3.11 with a `BackgroundScheduler` and one interval job configured with
  `max_instances=1`, `coalesce=True` and a `misfire_grace_time` derived from the interval.
  This gives serialized ticks without a message broker.
- The replay cursor lives in SQLite KV (`replay_tick`) and advances **only after** a tick
  completes. A restart continues at `cursor + 1`; a failed tick leaves the cursor untouched.
- Expose `mll scheduler start --mode replay|once`, share the same `RunLifecycleTick` use case
  with the drill (which runs it without sleeping), and stop cleanly on SIGINT/SIGTERM.
- Keep orchestration in-house and small: `Step`/`Pipeline` runs named steps with bounded
  retries and records a row per attempt in `runs`/`run_steps` (status, duration, error).
  The ledger is the audit source for the dashboard and the "never corrupt KV on failure"
  guarantee.
- Signals are installed only from the main thread, and a self-stop on the final tick uses
  `shutdown(wait=False)` so the worker does not join itself.

## Consequences

- No extra infrastructure: a single Python process and the same SQLite file as the rest of
  the operational state.
- APScheduler is a library, not a platform; complex DAGs, backfills and cross-job
  dependencies are out of scope by design.
- Every retry is visible as a separate ledger row (`FAILED, FAILED, SUCCEEDED`) instead of
  being flattened into one final status.
- A tick that exceeds the interval is skipped rather than overlapped; `coalesce=True`
  prevents a burst of catch-up runs after a long pause.
- The scheduler and drill share behaviour, so a bug found in one is fixed for both.

## Evidence

- Scheduler tests: a slow tick longer than the interval never runs twice concurrently; a
  second scheduler instance continues at `cursor + 1`; the cursor survives a restart; a
  failed tick is logged as `FAILED` and does not advance the cursor; `shutdown(wait=True)`
  returns in under two seconds; SIGTERM stops the process on POSIX.
- The Compose smoke runs `MLL_REPLAY_INTERVAL_SECONDS=1` and reaches champion v2 through
  the real API, which exercises the scheduler, the cursor, the API hot reload and the
  ledger together.
- `mll quickstart` leaves the same workspace at `tick 16 / 52`, and `mll replay reset`
  returns it to `tick 0 / 52`, proving the cursor is a normal, inspectable state value.
- See also [ARCHITECTURE.md](../ARCHITECTURE.md#one-lifecycle-tick) and
  [OPERATIONS.md](../OPERATIONS.md#replay-control).
