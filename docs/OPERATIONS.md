# Operaciones

Todo en este repositorio es local-first. No hay dependencia de la nube, ni cuenta, ni clave obligatoria;
el único acceso a red es el adaptador opcional `mll data fetch`.

## Inicio rápido

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

`install.sh --check` realiza las comprobaciones de prerrequisitos sin efectos secundarios y termina con
código 0/1. `install.ps1 -Check` es el equivalente en Windows; `--l1` / `-L1` valida la ruta sin Docker.
El stack publica solo puertos de loopback: MLflow `14000`, API `14080`, dashboard `14501`. El nombre del
proyecto de Compose es `mlops-lifecycle-lab`.

## Iniciar, detener, inspeccionar

```bash
docker compose -p mlops-lifecycle-lab up -d --build --wait
docker compose -p mlops-lifecycle-lab ps
docker compose -p mlops-lifecycle-lab logs -f api scheduler
docker compose -p mlops-lifecycle-lab down          # keep the mll-data volume
./install.sh --uninstall --yes                      # down -v + local images, keeps .env
```

El stack tiene cinco servicios de larga duración (`mlflow`, `api`, `scheduler`, `dashboard` y el `init`
de una sola ejecución), más el servicio opcional `cli` bajo el perfil `tools`. `init` es idempotente y
arranca el primer campeón con el reloj del replay; `scheduler` avanza una semana simulada cada
`MLL_REPLAY_INTERVAL_SECONDS` (por defecto 20 s) y puntúa a través de la API real, de modo que las
promociones se observan mediante la recarga en caliente.

Para un comando puntual en el mismo volumen:

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

`--fast` usa el renderizador Null de Evidently; no cambia los números del ciclo de vida. Los reportes son
`drill-<scenario>.json` y `.md` bajo `--report-dir` (por defecto `<workspace>/reports`). Las copias
comprometidas y el resumen generado viven en [`results/`](results/results-table.md).

Los cuatro escenarios son deterministas para una semilla fija. Dos ejecuciones completas de drift producen
JSON idéntico byte a byte; el flag `--check` compara contra [`../tests/golden/`](../tests/golden/) con el
margen documentado de ±1 semana en el reentrenamiento y la tolerancia de ±5% en las métricas. Regenera
los goldens solo con `scripts/regen_golden.py` y explica el motivo en el cuerpo del commit.

## Dashboard

`mll dashboard [--host 127.0.0.1] [--port 14501] [--workspace DIR]` inicia el Control Room de Streamlit.
Páginas:

- **Overview** — salud del campeón, WAPE-28d frente al SLO, RMSE, filas etiquetadas, probation.
- **Monitoring** — serie de WAPE, real frente a predicho, sesgo, mapa de calor de PSI, franja de validación.
- **Registry** — versiones, aliases, métricas, reporte de compuertas y tags de linaje.
- **Lifecycle** — eventos de auditoría y filas del run ledger, con duraciones y fallos.
- **Drill** — recorrer un reporte de drill comprometido o del workspace, tick a tick.

Cada página tiene estados de carga, vacío y error. El estado vacío imprime el comando exacto a ejecutar
(`mll quickstart`). Overview y Monitoring se actualizan automáticamente; las demás páginas se actualizan
bajo demanda. El estado se muestra siempre como ícono + etiqueta + color, nunca solo con color.

## Operaciones del registro

```bash
uv run mll registry list
uv run mll registry show 2
uv run mll registry promote 3 --force --reason "incident hotfix approved"
uv run mll registry rollback --reason "probation regression"
```

- Un `promote` normal solo funciona para una versión cuyo último evento `GATES_EVALUATED` la aprobó.
  `--force` omite esa comprobación, exige un `--reason` no vacío y emite `PROMOTION_FORCED` en el log de
  auditoría.
- `rollback` restaura `previous-champion`; la versión mala pasa a `ROLLED_BACK` y se elimina el alias.
  Un segundo rollback consecutivo sin campeón previo lanza `NoPreviousChampionError` y deja el estado sin cambios.
- Los aliases son `champion`, `previous-champion` y `challenger`. Nunca se usan etapas de MLflow.

## Control del replay

```bash
uv run mll replay status --workspace .mll   # tick N / 52
uv run mll replay reset --workspace .mll    # tick 0 / 52
uv run mll scheduler start --mode once      # one real lifecycle tick
uv run mll scheduler start --mode replay    # background weekly replay
```

El cursor es la clave `replay_tick` del KV de SQLite. Solo avanza cuando un tick termina; un fallo o un
`SIGTERM` lo deja en el último tick completado, de modo que un reinicio continúa en `cursor + 1`.
`mll quickstart` deja el workspace en el tick 16; `reset` borra el cursor y no elimina modelos,
predicciones ni eventos.

## Configuración

Todas las configuraciones de la aplicación usan el prefijo `MLL_` y las valida pydantic-settings. Los
booleanos aceptan las formas habituales `true/false/1/0`; los valores inválidos fallan de inmediato.

| Variable | Valor por defecto | Propósito |
|---|---|---|
| `MLL_WORKSPACE` | `.mll` | SQLite `mll.db`, almacén de MLflow y raíz de artefactos |
| `MLL_MLFLOW_TRACKING_URI` | sin definir → `mlflow.db` del workspace | URI de tracking de MLflow explícita |
| `MLL_MLFLOW_ARTIFACT_ROOT` | sin definir → `<workspace>/mlartifacts` | destino de artefactos |
| `MLL_MODEL_NAME` | `bike-demand` | nombre del modelo registrado |
| `MLL_EXPERIMENT_NAME` | `bike-demand` | nombre del experimento de MLflow |
| `MLL_SEED` | `42` | semilla del modelo y del bootstrap |
| `MLL_SLO_WAPE` | `0.30` | SLO monitoreado y disparo de la probation |
| `MLL_ABS_WAPE_MAX` | `0.30` | umbral absoluto de WAPE de G2 |
| `MLL_BASELINE_MAX_RATIO` | `0.85` | razón RMSE retador/baseline de G3 |
| `MLL_IMPROVEMENT_MIN` | `0.05` | mejora mínima de RMSE de G4 |
| `MLL_SEGMENT_MAX_REGRESSION` | `0.10` | regresión máxima por segmento de G6 |
| `MLL_BOOTSTRAP_RESAMPLES` | `2000` | remuestreos por bloques de días de G5 |
| `MLL_WINDOW_DAYS` | `28` | ventana de monitoreo móvil |
| `MLL_MIN_LABELED_ROWS` | `300` | mínimo de filas antes de que una caída del SLO pueda disparar |
| `MLL_COOLDOWN_DAYS` | `28` | enfriamiento tras un intento de reentrenamiento |
| `MLL_HOLDOUT_DAYS` | `14` | holdout del retador que termina al final del tick |
| `MLL_PROBATION_DAYS` | `21` | duración de la probation |
| `MLL_PROBATION_MIN_ROWS` | `150` | mínimo de filas etiquetadas para un veredicto de probation |
| `MLL_API_HOST` | `127.0.0.1` | dirección de bind de `mll serve` |
| `MLL_API_PORT` | `14080` | puerto de `mll serve` |
| `MLL_API_URL` | `http://127.0.0.1:14080` | destino del gateway HTTP de scoring |
| `MLL_ADMIN_API_KEY` | sin definir | `X-API-Key` opcional para `/v1/admin/reload` |
| `MLL_RELOAD_INTERVAL_SECONDS` | `5.0` | intervalo de sondeo del alias del campeón |
| `MLL_REPLAY_START` | `2012-01-01` | fecha inicial del replay |
| `MLL_REPLAY_INTERVAL_SECONDS` | `20.0` | segundos simulados por semana |
| `MLL_REPLAY_WEEKS` | `52` | longitud del replay |
| `MLL_SERVING_MODE` | `inprocess` | scoring `inprocess` o `http` |
| `MLL_FAST` | `false` | desactiva el renderizado HTML de Evidently |
| `MLL_LOG_LEVEL` | `INFO` | nivel de structlog |
| `MLL_LOG_FORMAT` | `console` | `console` o `json` (Compose usa `json`) |

Compose también reenvía `MLFLOW_TRACKING_URI` para las herramientas de MLflow y
`MLL_REPLAY_INTERVAL_SECONDS`/`MLL_ADMIN_API_KEY` desde `.env`. La telemetría está desactivada por
defecto mediante `apply_privacy_defaults()`: `MLFLOW_DISABLE_TELEMETRY`, `DO_NOT_TRACK`,
`ITERATIVE_DO_NOT_TRACK`, `EVIDENTLY_DISABLE_TELEMETRY`, `STREAMLIT_BROWSER_GATHER_USAGE_STATS` y
`MLFLOW_DISABLE_AGENT_HINT` se establecen salvo que el operador los sobrescriba.

## Observabilidad

- Los logs son structlog; Compose usa JSON con los bindings `request_id`, `run_id` y `tick`.
- `GET /metrics` expone `mll_predictions_total{model_version}`, `mll_prediction_latency_seconds`,
  `mll_model_info{version,stage}`, `mll_champion_version` y `mll_model_reload_failures_total`. El endpoint
  no requiere autenticación y, por defecto, solo escucha en loopback; el scraping y Grafana no vienen
  incluidos a propósito.
- El run ledger (`runs`, `run_steps`) registra cada tick y cada paso con estado, duración y error. La
  página Lifecycle lo muestra; `sqlite3 .mll/mll.db` sirve para consultas ad hoc.

## Solución de problemas

| Síntoma | Causa / solución |
|---|---|
| MLflow devuelve 403 en la UI publicada | Validación de host. Inicia MLflow con `--allowed-hosts` que incluya `localhost:14000,127.0.0.1:14000,mlflow:5000` (Compose ya lo hace). |
| `ImportError: libgomp.so.1` / `libomp.dylib` | Instala `libgomp1` en la imagen de Linux, o `brew install libomp` en macOS; `mll doctor` muestra la sugerencia. |
| Evidently `ImportError: infer_dtype_from_object` | Está instalado pandas 3.1+; el proyecto fija `pandas>=3.0.6,<3.1`. Vuelve a ejecutar `uv sync --locked`. |
| El dashboard no puede comunicarse con MLflow | En Docker Desktop el motor debe estar en modo de contenedores Linux; verifica con `docker info --format '{{.OSType}}'`. |
| `sqlite3.OperationalError: database is locked` | Otro proceso mantiene una escritura larga. WAL + `busy_timeout=5000` absorben el solapamiento normal; detén el otro escritor y reintenta. |
| `/readyz` devuelve 503 | Todavía no hay campeón cargado. Ejecuta `mll bootstrap --if-missing` (o reinicia el servicio `init`). |
| Puerto 14000/14080/14501 ya en uso | Otro stack local. Deténlo o cambia `MLL_API_PORT`; `./install.sh --check` indica el puerto ocupado. |
| `NoPreviousChampionError` en rollback | No existe el alias `previous-champion` (ya se hizo un rollback). Promueve primero un nuevo campeón. |
| Un `--check` del drill falla por una discrepancia de métricas | Investiga primero la implementación o la ventana; los goldens solo se pueden regenerar con `scripts/regen_golden.py` y una explicación en el commit. |
| El check de docs falla tras editar resultados | Vuelve a ejecutar `uv run python scripts/render_results_table.py`; nunca edites a mano la tabla generada. |

## Respaldo y reinicio

El estado operativo es estado local de confianza y el laboratorio no lo respalda. Para conservar una
copia, detén el stack y archiva el volumen con nombre (`docker run --rm -v
mlops-lifecycle-lab_mll-data:/data -v "$PWD":/backup alpine tar czf
/backup/mll-data.tgz -C /data .`). Para empezar limpio, usa `./install.sh --uninstall --yes` (elimina el
volumen) o un nuevo directorio `--workspace` para L1.
