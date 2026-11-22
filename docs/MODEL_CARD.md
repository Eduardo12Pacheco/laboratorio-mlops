# Model card — campeón de bike-demand

Esta ficha describe la familia de modelos de producción que usa el laboratorio: un regresor LightGBM
que predice los alquileres horarios de bicicletas (`cnt`) en Washington D. C., junto con el ciclo de
vida campeón/retador que lo rodea. Sigue el espíritu del artículo *Model Cards for Model Reporting*;
el producto es el ciclo de vida, no la sofisticación del modelo.

## Uso previsto

- Demostrar un bucle MLOps completo y local-first sobre un dataset temporal real.
- Servir predicciones horarias de demanda a través de los endpoints de FastAPI.
- Comparar un campeón y un retador sobre un holdout etiquetado con compuertas explícitas.
- Monitorear el rendimiento móvil y el drift de covariables, y disparar un reentrenamiento ante una
  caída del SLO basada en etiquetas.

Fuera de alcance: pronóstico de demanda en tiempo real para operaciones, planificación de capacidad,
precios, decisiones a nivel de persona, o cualquier servicio alojado o de producción. La "producción"
del replay es la mitad de 2012 del dataset incluido ([ADR-0009](adr/0009-trend-feature-and-poisson-objective.md)).

## Datos

| Elemento | Valor |
|---|---|
| Dataset | UCI Bike Sharing Dataset (horario), Fanaee-T 2013 |
| DOI | [10.24432/C5W894](https://doi.org/10.24432/C5W894) |
| Licencia | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Artefacto incluido | `src/mlops_lifecycle_lab/_data/hour.csv.gz` |
| Filas / ventana | 17,379 filas, 2011-01-01 00:00 → 2012-12-31 23:00 |
| Target | `cnt` (casual + registered) |
| Atribución | `_data/ATTRIBUTION.txt` y [DATA.md](DATA.md) |

Los datos van incluidos y se validan sin conexión. Las únicas etiquetas son el `cnt` histórico; el
replay puntúa una semana antes de que lleguen sus etiquetas y luego las ingiere por `entity_id`. El
dataset es `WARN` (no `FAIL`) porque 22 filas del 2011-03-10 tienen `hum = 0`; esa advertencia aparece
en todos los reportes de validación y nunca se ignora en silencio.

## Features

Las doce columnas de features, en orden:

```text
season, mnth, hr, holiday, weekday, workingday, weathersit,
temp, atemp, hum, windspeed, t_days
```

`t_days` son los días fraccionarios desde 2011-01-01 00:00 y capturan el crecimiento del sistema; sin
esta columna los árboles no pueden extrapolar el nivel de demanda de 2012, y reentrenar apenas ayuda.
`yr`, `instant`, `casual`, `registered` y `cnt` son columnas de fuga o proxies temporales y nunca se
convierten en features. `mnth`, `hr` y `weekday` se derivan del timestamp y se comprobó que son iguales
a las columnas originales de UCI en las 17,379 filas. El entrenamiento, el scoring por lotes y la API
comparten la misma función pura `build_features`, de modo que la paridad entre entrenamiento y servicio
se garantiza por construcción.

## Modelo y protocolo de entrenamiento

```text
LGBMRegressor(
  objective="poisson", n_estimators=400, learning_rate=0.05, num_leaves=31,
  min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
  random_state=42, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1,
)
```

- **Campeón bootstrap (v1):** entrenar con `2011-01-01 ≤ ts < 2011-10-01` (6,442 filas) y validar con
  `2011-10-01 ≤ ts < 2012-01-01` (2,203 filas). Las compuertas del primer campeón son `DATA_VALID`,
  `ABS_WAPE` y `BEATS_BASELINE`; la prueba de calidad comprometida
  (`tests/quality/test_bootstrap_champion.py`) verifica el RMSE, el WAPE, la razón contra el baseline y
  la tolerancia de reproducción de RMSE de ±2.0 a través de los adaptadores reales.
- **Retador (reentrenamiento) al final del tick `T`:** `cut = T − 14 días`; entrenar con cada fila
  etiquetada con `ts < cut` (archivo más etiquetas de producción ingeridas), reservar `[cut, T)` como
  holdout y volver a puntuar al campeón sobre el mismo holdout. Se evalúan seis compuertas sin
  cortocircuitos; el artefacto promovido es exactamente el artefacto evaluado (sin reentrenar).
- **Rollback/probation:** un retador promovido se vigila durante 21 días; con al menos 150 filas
  etiquetadas, un WAPE móvil por encima del SLO dispara un rollback automático a `previous-champion`.
- **Política de disparo:** solo una caída del WAPE de 28 días basada en etiquetas. El PSI de
  covariables se reporta, pero nunca dispara un reentrenamiento ([ADR-0004](adr/0004-label-based-slo-trigger.md)).

El baseline que usa `BEATS_BASELINE` es deliberadamente débil: la media de `cnt` por `(workingday, hr)`
sobre las mismas filas de entrenamiento. Es un piso, no un competidor.

## Métricas

La tabla siguiente se genera a partir de los cuatro documentos `drill-report/v1` comprometidos mediante
`scripts/render_results_table.py`. "As-served" es lo que el sistema predijo realmente en cada replay;
"frozen-v1" es el contrafáctico en el que el campeón bootstrap nunca se reemplaza. Las razones menores
que 1.0 significan que el ciclo de vida superó al modelo que nunca se reentrenó.

<!-- results:start -->

| Scenario | Weeks | Versions | Promotions | Rejections | Rollbacks | Quarantined | As-served RMSE | As-served MAE | As-served WAPE | Frozen-v1 RMSE | Frozen-v1 WAPE | RMSE ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| drift | 52 | 3 | 1 | 1 | 0 | 0 | 70.7197 | 45.8888 | 0.1949 | 109.9249 | 0.3165 | 0.6433 |
| gate-reject | 24 | 3 | 1 | 1 | 0 | 0 | 66.8555 | 41.8600 | 0.1988 | 88.8282 | 0.2783 | 0.7526 |
| bad-batch | 12 | 1 | 0 | 0 | 0 | 1 | 70.3356 | 43.4195 | 0.2654 | 70.3356 | 0.2654 | 1.0000 |
| rollback | 25 | 3 | 2 | 0 | 1 | 0 | 84.8111 | 48.6398 | 0.2287 | 88.8369 | 0.2763 | 0.9547 |

_Generated by `uv run python scripts/render_results_table.py` from `docs/results/drill-*.json` (seed 42, schema `drill-report/v1`); do not edit by hand._

<!-- results:end -->

El escenario de drift es el principal: 52 ticks semanales, una promoción en el tick 13, un rechazo en
el tick 52 y ningún rollback. La línea de tiempo completa, los reportes de compuertas y las métricas
por tick están en [`results/drill-drift.md`](results/drill-drift.md); el JSON crudo es la fuente de
verdad. Los dos reportes curados de drift de Evidently (disparo y tick final) están comprometidos en
[`assets/reports/`](assets/reports/).

## Advertencias de evaluación

- Las métricas provienen de un replay semanal, no de tráfico de producción independiente. Las ventanas
  se solapan por diseño (28 días móviles), así que los snapshots consecutivos están correlacionados.
- Las etiquetas llegan con un tick de retraso; un snapshot solo evalúa filas etiquetadas del campeón
  actual. Un modelo recién promovido necesita algunas etiquetas antes de que su probation pueda concluir.
- El holdout no se reentrena después de la selección. Es honesto pero algo pesimista: un ajuste ligero
  de hiperparámetros sobre el holdout mejoraría los números reportados, y se rechazó deliberadamente.
- La feature de tendencia extrapola el crecimiento. Una ruptura estructural (nueva ciudad, nueva
  política de precios, pandemia) la invalidaría; el disparo por SLO y el rollback son la red de
  seguridad, no una garantía.
- Las diferencias de punto flotante entre plataformas pueden mover el tick de reentrenamiento una
  semana; las pruebas golden verifican el orden de los eventos y las tolerancias, no la igualdad de
  bits entre sistemas operativos.

## Limitaciones

- Sin nube, Kubernetes, registro gestionado ni sistema de auth más allá de la API key de administración opcional.
- Sin feed en vivo: la "producción" es la mitad de 2012 del dataset, reproducida semana a semana.
- Sin producto de feature store, AutoML, búsqueda de hiperparámetros ni deep learning.
- Campeón/retador se evalúa offline sobre un holdout etiquetado; el shadow/canary traffic splitting
  queda fuera de alcance.
- El endpoint de Prometheus está expuesto; el scraping y Grafana no vienen incluidos.
- Los artefactos del modelo son estado local de confianza: cargar uno ejecuta código basado en pickle.
  Usa solo registros y artefactos que hayas creado o en los que confíes. Ver [SECURITY.md](../SECURITY.md).

## Consideraciones éticas

El dataset son conteos horarios agregados de alquileres de bicicletas, sin datos personales, sin
rostros y sin trayectorias individuales. Las predicciones describen la demanda del sistema; no deben
usarse para clasificar, fijar precios ni excluir personas. La demanda histórica del replay refleja el
Washington D. C. de 2011–2012 y puede codificar patrones de servicio preexistentes; cualquier despliegue
real necesitaría una revisión de equidad y cobertura que este laboratorio no realiza.

## Reproducibilidad

```bash
uv run mll data verify
uv run mll validate --bundled --json
uv run mll drill run --scenario drift --weeks 52 --workspace "$(mktemp -d)" \
  --report-dir "$(mktemp -d)"
uv run python scripts/render_results_table.py --check
uv run python scripts/check_docs.py
```
