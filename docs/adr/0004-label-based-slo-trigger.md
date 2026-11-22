# ADR-0004: El reentrenamiento se dispara por un SLO basado en etiquetas, no por drift de covariables

## Contexto

El sistema necesita una respuesta defendible a "¿cuándo reentrenamos?". La entrada obvia es el
drift de datos: comparar las features de esta semana con la ventana de entrenamiento del campeón y
reentrenar cuando el PSI supera 0.2. Esa regla resulta atractiva porque no necesita etiquetas, y
las etiquetas llegan con un tick de retraso. Pero la demanda de bicicletas es fuertemente
estacional: la temperatura y la temporada cambian cada año, y un modelo entrenado con datos de 2011
siempre parecerá "con drift" en 2012 aunque prediga bien. Una regla basada solo en covariables
reentrenaría casi cada semana, consumiendo cómputo y produciendo comparaciones campeón/retador
ruidosas.

## Decisión

- El reentrenamiento se dispara por una caída **basada en etiquetas** del WAPE de 28 días
  (`WAPE >= MLL_SLO_WAPE`, por defecto 0.30) sobre las predicciones hechas por el campeón actual,
  con al menos 300 filas etiquetadas y al menos 28 días desde el último intento de reentrenamiento.
- La probation suprime el reentrenamiento (el rollback tiene prioridad), y un disparo `MANUAL`
  omite la comprobación del SLO pero no las compuertas.
- El PSI de covariables, la proporción de drift numérico y el PSI de predicciones se calculan,
  persisten y se muestran en el dashboard, pero nunca disparan un reentrenamiento. Son evidencia
  diagnóstica.
- El resultado del disparo es un `TriggerDecision` tipado (`PERFORMANCE_SLO_BREACH`, `MANUAL`,
  `HEALTHY`, `INSUFFICIENT_LABELS`, `COOLDOWN`, `IN_PROBATION`), evaluado en ese orden y probado
  unitariamente como matriz.

## Consecuencias

- La automatización reacciona a lo que importa operativamente (el error de pronóstico), no a la
  temporada.
- El reentrenamiento espera etiquetas; una promoción reciente no puede juzgarse hasta que existan
  suficientes etiquetas.
- El dashboard debe separar las dos preguntas ("¿el rendimiento es malo?" y "¿se movieron las
  entradas?") en lugar de colapsarlas en una sola luz roja.
- La política necesita un cooldown y una guarda de probation; sin ellos, una sola semana ruidosa
  podría iniciar una tormenta de reentrenamientos.
- Los reportes de drift siguen siendo útiles para las personas y para explicar *por qué* podría
  cambiar el rendimiento, pero un WAPE sano con `numeric_drift_share = 1.0` es explícitamente un
  no-op en las pruebas.

## Evidencia

- Las pruebas de política del dominio incluyen un snapshot con `numeric_drift_share = 1.0` y WAPE
  sano, y afirman que nunca dispara.
- El replay de drift de 52 semanas muestra `numeric_drift_share >= 0.5` en cada ventana semanal de
  2012 (estacionalidad), mientras que la primera caída del SLO ocurre solo en el tick 13 (fin del
  2012-04-01) con WAPE 0.3086 sobre 671 filas etiquetadas. El reporte comprometido es
  [`../results/drill-drift.json`](../results/drill-drift.json).
- El último tick vuelve a superar el umbral (WAPE 0.3152) y el retador es rechazado por las
  compuertas, lo que demuestra que un disparo es una solicitud de evaluación, no una promoción.
- El SLO de 0.30 deja margen sobre el WAPE de validación del campeón bootstrap, mientras que el
  modelo congelado de 2011 lo cruza a fines de marzo de 2012: es la curva de fallo prevista.
