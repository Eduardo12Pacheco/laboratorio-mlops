# ADR-0005: Seis compuertas de promoción con bootstrap pareado por bloques de días

## Contexto

Un disparo de reentrenamiento responde "el campeón puede estar desactualizado". No responde "¿este
retador es realmente mejor?". Una comparación agregada de RMSE es fácil de manipular con un
holdout afortunado, ignora la incertidumbre y puede ocultar una regresión en un segmento
operativamente importante. El proyecto necesita una decisión explícita e inspeccionable que un
revisor pueda reproducir a partir de un reporte comprometido.

## Decisión

Al final del tick `T`, fijar `cut = T − 14 días`. Entrenar el retador con cada fila etiquetada con
`ts < cut` (archivo más filas de producción ingeridas) y reservar `[cut, T)` como holdout. Volver a
puntuar al campeón exactamente sobre el mismo holdout y evaluar todas las compuertas, sin
cortocircuitos:

1. `DATA_VALID` — el veredicto del frame de entrenamiento no es `FAIL`.
2. `ABS_WAPE` — el WAPE del holdout del retador ≤ 0.30.
3. `BEATS_BASELINE` — el RMSE del retador ≤ 0.85 × el RMSE del perfil ingenuo (baseline) sobre las
   mismas filas.
4. `IMPROVEMENT` — `(RMSE campeón − RMSE retador) / RMSE campeón ≥ 0.05`.
5. `SIGNIFICANCE` — la cota inferior del IC 95% del bootstrap pareado por bloques de días de
   `mean(SE campeón − SE retador)` es > 0, con 2000 remuestreos y semilla 42.
6. `SEGMENT` — el RMSE del retador no es peor que el del campeón en más de un 10% en ninguno de
   los segmentos `workingday`.

El primer campeón usa solo las compuertas 1–3. Cada resultado de compuerta (valor observado,
umbral, aprobado/rechazado, detalle legible) se persiste como `gate_report.json` y como un evento
`GATES_EVALUATED`. El artefacto promovido es exactamente el artefacto evaluado: no hay
reentrenamiento sobre el holdout.

El bootstrap remuestrea **días calendario completos**, no filas individuales, porque los errores
horarios están autocorrelacionados. El estadístico compara errores cuadráticos (la escala del
RMSE), y la implementación del dominio es pura, con semilla y con pruebas de propiedades.

## Consecuencias

- Un retador que gana en promedio pero es ruidoso o regresiona un segmento se rechaza.
- La decisión es más lenta y costosa que un umbral único de RMSE, pero es explicable y el costo es
  trivial a este tamaño de datos.
- El holdout no se reentrena, así que los números reportados son honestos pero algo pesimistas;
  esto se documenta como limitación en lugar de ocultarse.
- Una promoción forzada sigue siendo posible para la respuesta a incidentes, pero requiere un
  motivo y emite `PROMOTION_FORCED`, por lo que queda auditada.
- El formato del reporte de compuertas lo comparten la aplicación, el reporte del drill y el
  dashboard.

## Evidencia

- Las pruebas de compuertas del dominio cubren los límites exactos: 0.0499 falla y 0.0500 pasa
  para la mejora; 0.1001 falla y 0.1000 pasa para la regresión por segmento; 0.3001 falla y 0.3000
  pasa para el WAPE absoluto; 0.8501 falla y 0.8500 pasa para la razón contra el baseline; y una
  cota inferior del CI exactamente igual a 0 falla la significancia.
- El replay de drift promueve v2 en el tick 13 con RMSE del retador 53.2673, mejora 0.5595, cota
  inferior de significancia 9000.8429 y un delta del peor segmento de −0.5528; la tabla cruda de
  compuertas está en [`../results/drill-drift.md`](../results/drill-drift.md).
- El replay de gate-reject rechaza con exactamente `IMPROVEMENT`, `SIGNIFICANCE` y `SEGMENT`
  fallando, lo que demuestra que el protocolo no oculta a un retador marginal detrás de un
  promedio.
- Las pruebas de hipótesis demuestran que `approved == all(results.passed)` y que cada compuerta
  siempre se evalúa, de modo que una refactorización futura no pueda reintroducir cortocircuitos.
