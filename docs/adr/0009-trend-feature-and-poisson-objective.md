# ADR-0009: Una feature de tendencia y un objetivo Poisson para una demanda creciente

## Contexto

La demanda de bicicletas creció sustancialmente entre 2011 (media de 143.8 alquileres/hora) y 2012
(media de 234.7). Las features de calendario y clima por sí solas describen la forma de la demanda,
pero no su nivel. Los modelos de árboles no pueden extrapolar más allá de las etiquetas que han
visto, así que un modelo entrenado con datos de 2011 subestimará sistemáticamente 2012 a menos que
algo restaure el nivel. El experimento de planificación comparó un modelo congelado con uno
reentrenado cada 14 días bajo cuatro configuraciones; la elección tenía que estar respaldada por
evidencia, no por intuición.

## Decisión

- Añadir exactamente una feature temporal, `t_days` = días fraccionarios desde 2011-01-01 00:00,
  para que el modelo pueda representar una tendencia de nivel y siga apoyándose en los árboles para
  la forma de calendario/clima. No se usan `yr`, `instant`, `casual`, `registered` ni `cnt`: esas
  columnas filtran el target o son identificadores de la base de datos.
- Usar `LGBMRegressor(objective="poisson")` con los hiperparámetros fijos de
  [MODEL_CARD.md](../MODEL_CARD.md). La demanda es un conteo no negativo; el objetivo Poisson
  coincide con el target y evita predicciones negativas en origen, con un recorte final a cero para
  el servicio.
- Fijar los hiperparámetros y la semilla; sin tuning, AutoML ni búsqueda de hiperparámetros. El
  ciclo de vida y el reentrenamiento honesto son el producto, no una victoria en benchmark.
- Mantener el baseline de perfil ingenuo `(workingday, hour)` como un piso deliberadamente débil
  para la compuerta `BEATS_BASELINE`, no como un modelo competitivo.

## Consecuencias

- Reentrenar con etiquetas recientes es efectivo: la feature de tendencia permite al modelo usar el
  nuevo nivel en lugar de aprender solo la forma estacional.
- La tendencia es un supuesto. Una ruptura estructural (nueva política de precios, una pandemia, una
  nueva ciudad) puede volver dañino a `t_days`; el disparo por SLO y la probation/rollback son redes
  de seguridad, y la limitación se indica en la model card.
- Las predicciones Poisson son siempre no negativas; la ruta de servicio aun así recorta en cero
  como medida defensiva ante cambios del adaptador o del artefacto.
- Sin la feature de tendencia, reentrenar apenas ayudaría; un cambio futuro debe volver a ejecutar el
  drill de drift y las compuertas de calidad, no solo las pruebas unitarias.

## Evidencia

- Experimento de planificación (RMSE a 28 días hacia adelante, congelado → reentrenado): features
  simples 131.6 → 125.3 (ratio 0.95); más `t_days` 109.3 → 67.6 (0.62); más `t_days` y Poisson
  114.0 → 64.6 (0.57). La ponderación por recencia no ayudó.
- El replay de drift comprometido de 52 semanas: RMSE servido 70.7197 frente a 109.9249 de
  frozen-v1 (ratio 0.6433), WAPE servido 0.1949 frente a 0.3165, con exactamente una promoción y un
  reentrenamiento final rechazado. Ver [`../results/results-table.md`](../results/results-table.md).
- La prueba de calidad de bootstrap ejercita el stack real de SQLite/MLflow/LightGBM y afirma los
  límites S5 (RMSE, WAPE, ratio contra el baseline y la tolerancia de reproducción de ±2.0), además
  de `np.array_equal` sobre dos entrenamientos idénticos.
- La paridad de features se prueba en las 17,379 filas: `mnth`, `hr` y `weekday` derivados son
  iguales a las columnas originales del dataset, y `build_features` nunca emite una columna de fuga.
