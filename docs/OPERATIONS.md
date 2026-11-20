# Operations

Everything in this repository is local-first. There is no cloud dependency, no account and
no mandatory key; the only network access is the optional `mll data fetch` adapter.

## Quick start

```bash
# L2: Docker Compose stack
./install.sh
# open http://127.0.0.1:14501 (dashboard), http://127.0.0.1:14000 (MLflow UI)

# L1: no Docker
uv sync --locked
uv run mll doctor
uv run mll quickstart --workspace .mll
uv run mll dashboard --workspace .mll
```

`install.sh --check` performs the prerequisite checks without side effects and exits 0/1.
`install.ps1 -Check` is the Windows equivalent; `--l1` / `-L1` validates the Docker-free
path. The stack publishes only loopback ports: MLflow `14000`, API `14080`, dashboard
`14501`. The Compose project name is `mlops-lifecycle-lab`.

## Start, stop, inspect

```bash
docker compose -p mlops-lifecycle-lab up -d --build --wait
docker compose -p mlops-lifecycle-lab ps
docker compose -p mlops-lifecycle-lab logs -f api scheduler
docker compose -p mlops-lifecycle-lab down          # keep the mll-data volume
./install.sh --uninstall --yes                      # down -v + local images, keeps .env
```

The stack has five long-running services (`mlflow`, `api`, `scheduler`, `dashboard` and the
one-shot `init`) plus the optional `cli` service under the `tools` profile. `init` is
idempotent and bootstrap the first champion on the replay clock; `scheduler` advances one
simulated week every `MLL_REPLAY_INTERVAL_SECONDS` (default 20 s) and scores through the
real API, so promotions are observed by hot reload.

For a one-off command in the same volume:

```bash
docker compose -p mlops-lifecycle-lab run --rm cli mll registry list
docker compose -p mlops-lifecycle-lab run --rm cli mll drill run --scenario all \
  --workspace /tmp/drill --report-dir /tmp/drill/reports
```

## Drills

```bash
uv run mll drill run --scenario drift --weeks 52 \
  --workspace "$(mktemp -d)" --report-dir "$(mktemp -d)"

uv run mll drill run --scenario gate-reject
uv run mll drill run --scenario bad-batch
uv run mll drill run --scenario rollback
uv run mll drill run --scenario all --check
uv run mll drill run --scenario drift --weeks 16 --fast --check   # CI smoke
```

`--fast` uses the Null Evidently renderer; it does not change lifecycle numbers. Reports
are `drill-<scenario>.json` and `.md` under `--report-dir` (default
`<workspace>/reports`). The committed copies and the generated summary live in
[`results/`](results/results-table.md).

All four scenarios are deterministic for a fixed seed. Two full drift runs produce
byte-identical JSON; the `--check` flag compares against
[`../tests/golden/`](../tests/golden/) with the documented ±1 week retrain slack and ±5%
metric tolerance. Regenerate goldens only with `scripts/regen_golden.py` and explain why in
the commit body.

## Dashboard

`mll dashboard [--host 127.0.0.1] [--port 14501] [--workspace DIR]` starts the Streamlit
Control Room. Pages:

- **Overview** — champion health, WAPE-28d vs the SLO, RMSE, labelled rows, probation.
- **Monitoring** — WAPE series, actual vs predicted, bias, PSI heatmap, validation strip.
- **Registry** — versions, aliases, metrics, gate report and lineage tags.
- **Lifecycle** — audit events and run-ledger rows with durations/failures.
- **Drill** — scrub a committed or workspace drill report tick by tick.

Every page has loading, empty and error states. The empty state prints the exact command
to run (`mll quickstart`). Overview and Monitoring refresh automatically; the other pages
refresh on demand. Status is always icon + label + colour, never colour alone.

## Registry operations

```bash
uv run mll registry list
uv run mll registry show 2
uv run mll registry promote 3 --force --reason "incident hotfix approved"
uv run mll registry rollback --reason "probation regression"
```

- A normal `promote` only works for a version whose last `GATES_EVALUATED` event approved
  it. `--force` bypasses that check, requires a non-empty `--reason`, and emits
  `PROMOTION_FORCED` in the audit log.
- `rollback` restores `previous-champion`; the bad version becomes `ROLLED_BACK` and the
  alias is removed. A second consecutive rollback with no previous champion raises
  `NoPreviousChampionError` and leaves state unchanged.
- Aliases are `champion`, `previous-champion`, `challenger`. MLflow stages are never used.

## Replay control

```bash
uv run mll replay status --workspace .mll   # tick N / 52
uv run mll replay reset --workspace .mll    # tick 0 / 52
uv run mll scheduler start --mode once      # one real lifecycle tick
uv run mll scheduler start --mode replay    # background weekly replay
```

The cursor is the SQLite KV key `replay_tick`. It only advances after a tick finishes; a
crash or `SIGTERM` leaves it at the last completed tick, so a restart continues at
`cursor + 1`. `mll quickstart` leaves the workspace at tick 16; `reset` clears the cursor
and does not delete models, predictions or events.

## Configuration

All application settings use the `MLL_` prefix and are validated by pydantic-settings.
Booleans accept the usual `true/false/1/0` forms; invalid values fail fast.

| Variable | Default | Purpose |
|---|---|---|
| `MLL_WORKSPACE` | `.mll` | SQLite `mll.db`, MLflow store and artifacts root |
| `MLL_MLFLOW_TRACKING_URI` | unset → workspace `mlflow.db` | explicit MLflow tracking URI |
| `MLL_MLFLOW_ARTIFACT_ROOT` | unset → `<workspace>/mlartifacts` | artifact destination |
| `MLL_MODEL_NAME` | `bike-demand` | registered model name |
| `MLL_EXPERIMENT_NAME` | `bike-demand` | MLflow experiment name |
| `MLL_SEED` | `42` | model and bootstrap seed |
| `MLL_SLO_WAPE` | `0.30` | monitored SLO and probation breach |
| `MLL_ABS_WAPE_MAX` | `0.30` | G2 absolute WAPE threshold |
| `MLL_BASELINE_MAX_RATIO` | `0.85` | G3 challenger/baseline RMSE ratio |
| `MLL_IMPROVEMENT_MIN` | `0.05` | G4 minimum RMSE improvement |
| `MLL_SEGMENT_MAX_REGRESSION` | `0.10` | G6 maximum per-segment regression |
| `MLL_BOOTSTRAP_RESAMPLES` | `2000` | G5 day-block resamples |
| `MLL_WINDOW_DAYS` | `28` | trailing monitoring window |
| `MLL_MIN_LABELED_ROWS` | `300` | minimum rows before a SLO breach can trigger |
| `MLL_COOLDOWN_DAYS` | `28` | cooldown after a retrain attempt |
| `MLL_HOLDOUT_DAYS` | `14` | challenger holdout ending at the tick end |
| `MLL_PROBATION_DAYS` | `21` | probation length |
| `MLL_PROBATION_MIN_ROWS` | `150` | minimum labelled rows for a probation verdict |
| `MLL_API_HOST` | `127.0.0.1` | `mll serve` bind address |
| `MLL_API_PORT` | `14080` | `mll serve` port |
| `MLL_API_URL` | `http://127.0.0.1:14080` | HTTP scoring gateway target |
| `MLL_ADMIN_API_KEY` | unset | optional `X-API-Key` for `/v1/admin/reload` |
| `MLL_RELOAD_INTERVAL_SECONDS` | `5.0` | champion alias poll interval |
| `MLL_REPLAY_START` | `2012-01-01` | first replay date |
| `MLL_REPLAY_INTERVAL_SECONDS` | `20.0` | simulated seconds per week |
| `MLL_REPLAY_WEEKS` | `52` | replay length |
| `MLL_SERVING_MODE` | `inprocess` | `inprocess` or `http` scoring |
| `MLL_FAST` | `false` | disable Evidently HTML rendering |
| `MLL_LOG_LEVEL` | `INFO` | structlog level |
| `MLL_LOG_FORMAT` | `console` | `console` or `json` (Compose sets `json`) |

Compose also forwards `MLFLOW_TRACKING_URI` for MLflow tooling and
`MLL_REPLAY_INTERVAL_SECONDS`/`MLL_ADMIN_API_KEY` from `.env`. Telemetry is disabled by
default through `apply_privacy_defaults()`: `MLFLOW_DISABLE_TELEMETRY`,
`DO_NOT_TRACK`, `ITERATIVE_DO_NOT_TRACK`, `EVIDENTLY_DISABLE_TELEMETRY`,
`STREAMLIT_BROWSER_GATHER_USAGE_STATS` and `MLFLOW_DISABLE_AGENT_HINT` are set unless the
operator overrides them.

## Observability

- Logs are structlog; Compose uses JSON with `request_id`, `run_id` and `tick` bindings.
- `GET /metrics` exposes `mll_predictions_total{model_version}`,
  `mll_prediction_latency_seconds`, `mll_model_info{version,stage}`,
  `mll_champion_version` and `mll_model_reload_failures_total`. The endpoint is unauthenticated
  and loopback-only by default; scraping and Grafana are intentionally not bundled.
- The run ledger (`runs`, `run_steps`) records every tick and step with status, duration
  and error. The Lifecycle page renders it; `sqlite3 .mll/mll.db` works for ad-hoc queries.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| MLflow returns 403 on the published UI | Host validation. Start MLflow with `--allowed-hosts` including `localhost:14000,127.0.0.1:14000,mlflow:5000` (Compose already does). |
| `ImportError: libgomp.so.1` / `libomp.dylib` | Install `libgomp1` in the Linux image, or `brew install libomp` on macOS; `mll doctor` prints the hint. |
| Evidently `ImportError: infer_dtype_from_object` | pandas 3.1+ is installed; the project pins `pandas>=3.0.6,<3.1`. Re-run `uv sync --locked`. |
| Dashboard cannot talk to MLflow | On Docker Desktop the engine must be in Linux-container mode; check `docker info --format '{{.OSType}}'`. |
| `sqlite3.OperationalError: database is locked` | Another process holds a long write. WAL + `busy_timeout=5000` handle normal overlap; stop the other writer and retry. |
| `/readyz` returns 503 | No champion is loaded yet. Run `mll bootstrap --if-missing` (or restart the `init` service). |
| Port 14000/14080/14501 already in use | Another local stack. Stop it or change `MLL_API_PORT`; `./install.sh --check` names the busy port. |
| `NoPreviousChampionError` on rollback | There is no `previous-champion` alias (already rolled back once). Promote a new champion first. |
| A drill `--check` fails with a metric mismatch | Investigate the implementation/window first; goldens may only be regenerated with `scripts/regen_golden.py` and an explanation in the commit. |
| Docs check fails after editing results | Re-run `uv run python scripts/render_results_table.py`; never hand-edit the generated table. |

## Backup and reset

The operational state is trusted local state and is not backed up by the lab. To keep a
snapshot, stop the stack and archive the named volume (`docker run --rm -v
mlops-lifecycle-lab_mll-data:/data -v "$PWD":/backup alpine tar czf
/backup/mll-data.tgz -C /data .`). To start clean, use `./install.sh --uninstall --yes`
(removes the volume) or a new `--workspace` directory for L1.
