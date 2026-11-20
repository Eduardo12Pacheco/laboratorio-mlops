# ADR-0008: One SQLite operational store in WAL mode

## Context

The lifecycle needs a prediction log, delayed labels, an audit event stream, monitoring
snapshots, key/value state (replay cursor, probation) and a run ledger. A second database
engine would add a service, credentials and a migration tool to a local-first project whose
whole operational state is small. The API, scheduler, drill and dashboard also need to read
and write concurrently without corrupting state, including on Windows.

## Decision

- Use stdlib `sqlite3` and one file, `<workspace>/mll.db`, for all operational tables.
  MLflow keeps its own `mlflow.db` and artifact store; the two are never mixed.
- Open every connection with `journal_mode=WAL`, `busy_timeout=5000`,
  `foreign_keys=ON` and `row_factory=sqlite3.Row`. WAL gives readers a consistent snapshot
  while a writer commits, and the busy timeout absorbs normal short overlap.
- Version the schema with ordered migrations recorded in `schema_migrations`; the DDL lives
  in code, so a fresh workspace and an upgraded workspace converge deterministically.
- Use the repository pattern behind the application ports and a tiny unit of work
  (`with db.transaction():`) for multi-row writes. Timestamps are ISO-8601 UTC-naive text,
  matching the dataset.
- FastAPI runs sync handlers in worker threads, so the shared connection is opened with
  `check_same_thread=False` and guarded by an `RLock` around `transaction()`. This is a
  serving-specific concern; the WAL and busy-timeout contract is unchanged.

## Consequences

- Zero external services, trivial backup (copy the workspace), and SQL is inspectable with
  `sqlite3`. The dashboard and CLI read the same state the API writes.
- WAL on a Docker bind mount is unsafe on some hosts, so the Compose stack keeps SQLite on
  the named volume `mll-data` only.
- SQLite is a single-writer engine: very high write concurrency is out of scope. The API
  batches prediction writes per request and the scheduler has one tick at a time.
- Migrations are additive and in code; there is no downgrade path. A breaking schema change
  would require a new workspace or an explicit migration script.
- A stale `-wal`/`-shm` file is normal; do not delete it while the database is open.

## Evidence

- Integration tests assert `PRAGMA journal_mode` returns `wal` and that `busy_timeout` is
  applied; the prediction-log contract suite passes against both the in-memory fake and the
  SQLite adapter.
- The real-stack serving test exposed the cross-thread connection bug and now passes with
  the shared connection/RLock: online prediction, a 168-row batch, an idempotent re-send
  and a labelled join all run through uvicorn on a loopback port.
- The Compose smoke verifies the same code path inside the container with the volume-backed
  database, including the scheduler writing while the API reads.
- See [ARCHITECTURE.md](../ARCHITECTURE.md) for the storage ports and
  [OPERATIONS.md](../OPERATIONS.md#troubleshooting) for the lock troubleshooting entry.
