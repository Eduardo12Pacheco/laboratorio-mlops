# ADR-0006: Estadísticos de drift nativos como contrato, Evidently como reporte

## Contexto

El monitoreo necesita dos salidas: números en los que el automatismo y las pruebas puedan confiar,
y un reporte legible para las personas que revisan. Evidently ofrece un reporte HTML pulido y su
propia implementación de PSI, pero su API es amplia, pesada y sensible a la versión de pandas. Si
la decisión del ciclo de vida dependiera de un objeto de Evidently, una actualización de la
biblioteca podría cambiar una compuerta, y el dominio puro ganaría una gran dependencia de
framework. Si el proyecto solo usara Evidently, los revisores no tendrían una definición pequeña y
comprobable por propiedades del estadístico.

## Decisión

- El **contrato** es la implementación pura propia del proyecto en `domain/drift.py`:
  `psi_numeric` con 10 bins de cuantiles de referencia, bordes externos ±inf y proporciones
  recortadas en 1e-4; `psi_categorical`; umbrales 0.1 (moderado) y 0.2 (drift); un resumen sobre
  `temp`, `atemp`, `hum`, `windspeed` (numéricas) y `season`, `weathersit` (categóricas), además del
  PSI de la distribución de predicciones y `numeric_drift_share`.
- Los bordes de cuantiles usan `numpy.quantile(..., method="inverted_cdf")`, anclados en valores de
  referencia observados. La interpolación lineal inventaba bordes internos a partir del ruido de
  punto flotante entre valores repetidos y hacía que el PSI dependiera de la representación y no de
  los datos.
- **Evidently es un renderizador**: `Report([DataDriftPreset(method="psi")])` sobre exactamente las
  columnas monitoreadas, guardado como HTML en ticks seleccionados para las personas.
  `NullReportRenderer` es el Null Object que usan los drills `--fast` y las pruebas.
- El rango fijado `pandas>=3.0.6,<3.1` forma parte de esta decisión: Evidently 0.7.23 no importa en
  pandas 3.1.x (`infer_dtype_from_object`). El pin es intencional y lo verifica `uv lock --check`;
  no es una solución temporal.
- Una prueba de contrato renderiza la ventana de marzo de 2012 contra la referencia de entrenamiento
  de 2011 y afirma que el PSI nativo y el de Evidently coinciden en la característica más
  desplazada.

## Consecuencias

- Las decisiones de drift, los umbrales y el dashboard son deterministas, rápidos e independientes
  de Evidently; una importación fallida de Evidently no puede bloquear el monitoreo ni el
  entrenamiento.
- El proyecto mantiene dos implementaciones de la misma idea y debe mantenerlas honestas; la prueba
  de acuerdo es la garantía.
- Los reportes HTML son grandes (≈4 MB cada uno). Se generan solo para las personas, nunca se
  comprometen a ciegas, y las dos copias curadas comparten un único bundle de JavaScript extraído
  para mantenerse dentro del presupuesto de documentación de 5 MB.
- Las actualizaciones de pandas quedan condicionadas a la compatibilidad de importación de Evidently
  hasta que una versión más nueva de Evidently soporte la versión nueva de pandas. Ver
  [DATA.md](../DATA.md) para el pin de dependencias.

## Evidencia

- Integración T7: un reporte de marzo de 2012 contra el entrenamiento de 2011 ocupó 3,956,588
  bytes; el `psi_numeric(temp) = 1.6952` nativo y el `ValueDrift(temp) = 1.7309` de Evidently
  marcan ambos `temp`.
- El replay de drift reporta una proporción de drift numérico de 0.50–0.75 en cada semana de 2012
  mientras el modelo rinde bien, justo por lo que esta señal es informativa (ver
  [ADR-0004](0004-label-based-slo-trigger.md)).
- Propiedades del PSI: `psi(x, x) = 0`, `psi >= 0`, invariancia afín y crecimiento monótono ante
  desplazamientos de media de 0.25/0.5/1.0/2.0 desviaciones estándar sobre normales con semilla.
- El bundle curado pesa 4,392,896 bytes: dos archivos HTML más el `evidently-dashboard.js`
  compartido; `scripts/render_drift_reports.py` lo regenera a partir del dataset incluido y de las
  dos ventanas de referencia.
