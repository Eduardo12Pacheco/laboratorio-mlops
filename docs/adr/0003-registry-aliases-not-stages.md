# ADR-0003: Aliases y tags del registro, no etapas de modelo

## Contexto

El registro de modelos necesita un ciclo de vida (`CANDIDATE`, `CHAMPION`, `ARCHIVED`, `REJECTED`,
`ROLLED_BACK`) y punteros estables para el servicio y el rollback. MLflow expresaba esto con
etapas de modelo, pero la documentación oficial indica que las etapas están obsoletas desde la
2.9.0 y se eliminarán en una futura versión mayor. Construir el ciclo de vida sobre una API
obsoleta volvería frágil la parte más visible del proyecto y ocultaría transiciones ilegales
dentro de MLflow.

## Decisión

- La etapa del ciclo de vida es el tag `mll.stage`, con valores del enum de dominio `ModelStage`.
  `mll.run_id`, ambas ventanas, la huella de los datos, la versión de la especificación de
  features, la hora de creación, el motivo del disparo y las cuatro métricas viajan como otros
  tags `mll.*`.
- Los punteros móviles son **aliases** de MLflow: `champion`, `previous-champion` y `challenger`.
  La promoción archiva al incumbente, mueve `previous-champion`, asigna `champion` y limpia
  `challenger`; el rollback restaura `previous-champion`.
- La máquina de estados del dominio valida cada transición antes de que el adaptador escriba
  nada. El adaptador lee con `search_model_versions` (filtrado a las versiones que llevan
  `mll.stage`) y nunca llama a las APIs de etapas obsoletas.
- El servicio carga mediante `mlflow.pyfunc.load_model("models:/bike-demand/<version>")`, de modo
  que el registro es independiente del flavor y un futuro flavor de modelo no cambia la API.

## Consecuencias

- La promoción es una pequeña secuencia de escrituras de tags/aliases con una tabla de transiciones
  validada; una escritura corrupta o parcial es detectable al releer los tags.
- `previous-champion` hace el rollback explícito y comprobable. Un segundo rollback sin una
  promoción intermedia lanza `NoPreviousChampionError` en lugar de adivinar.
- El dashboard y la CLI pueden listar aliases sin importar internals de MLflow.
- Los usuarios de la UI de MLflow ven aliases y tags en lugar de la columna de etapas conocida; la
  documentación y la CLI explican la correspondencia.
- La suite de contratos del registro se ejecuta contra `InMemoryRegistry` y `MlflowRegistry`, así
  el fake no puede prometer una semántica que el adaptador real no tiene.

## Evidencia

- Una prueba que busca en el código fuente afirma que `transition_model_version_stage` y
  `get_latest_versions` no aparecen en el adaptador; `uv run pytest tests/integration -q` está en
  verde.
- El contrato del registro cubre promoción, rechazo, rollback, transiciones ilegales,
  `previous-champion` y `list_versions` para ambas implementaciones.
- El drill muestra la secuencia prevista en un almacén real de MLflow: `CHALLENGER_PROMOTED` mueve
  `champion` de v1 a v2, el reentrenamiento final queda `REJECTED` sin tocar el campeón, y el
  escenario de rollback restaura v2 mientras marca v3 como `ROLLED_BACK`.
- Sonda de planificación con MLflow 3.18: verificó set/get/delete de aliases y `pyfunc.load_model`;
  el contrato del registro lo reproduce en la suite de pruebas. Ver la tabla de tareas en el Log
  del plan.
