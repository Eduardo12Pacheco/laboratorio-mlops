# Datos

Un problema, un dataset: alquileres horarios de bicicletas en Washington D. C., 2011–2012, del
UCI Machine Learning Repository. El dataset va incluido en el wheel para que el proyecto funcione
por completo sin conexión; el comando de adquisición opcional existe solo para reproducir el bundle.

## Procedencia

| Campo | Valor |
|---|---|
| Dataset | Bike Sharing Dataset (horario) |
| Autor | Hadi Fanaee-T (2013) |
| Página del dataset | <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset> |
| DOI | [10.24432/C5W894](https://doi.org/10.24432/C5W894) |
| Licencia | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Archivo fijado | <https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip> |
| Archivo incluido | `src/mlops_lifecycle_lab/_data/hour.csv.gz` |
| Manifiesto | `src/mlops_lifecycle_lab/_data/manifest.json` |
| Atribución | `src/mlops_lifecycle_lab/_data/ATTRIBUTION.txt` |

El `.csv.gz` incluido es una compresión gzip sin modificaciones de `hour.csv` del archivo fijado.
Los bytes descomprimidos son la identidad que importa, y son los que verifica la CLI.

## Sumas de verificación

| Artefacto | sha256 | Tamaño |
|---|---|---|
| ZIP de UCI | `b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401` | 279,992 B |
| `hour.csv` (descomprimido) | `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f` | 1,156,736 B |
| `hour.csv.gz` (incluido) | derivado por gzip; no es una identidad de procedencia | 256,659 B |

`day.csv` (731 filas) y `Readme.txt` están presentes en el archivo pero no se usan. El `Readme.txt`
del archivo es la fuente de las constantes de normalización (`temp` usa −8…+39 °C, `atemp` usa
−16…+50 °C, y `hum` y `windspeed` se dividen entre 100 y 67, respectivamente).

## Esquema

Contrato raw (`mll validate`, fuente de lotes y API de serving), en orden:

```text
entity_id, timestamp, season, holiday, workingday, weathersit,
temp, atemp, hum, windspeed
```

Etiquetas (solo archivo y entrenamiento): `casual, registered, cnt`. El target es `cnt`, y
`cnt == casual + registered` en cada fila. `timestamp` se deriva de las columnas `dteday` + `hr` de
UCI, y `entity_id` es el `instant` de UCI. El contrato de features (`FEATURE_COLUMNS`) se define una
sola vez en `domain/schema.py`; `yr`, `instant`, `casual`, `registered` y `cnt` son columnas de fuga
o proxies temporales y nunca se convierten en features.

## Linaje

```text
UCI archive zip (pinned sha256)
  └─ hour.csv (decompressed sha256)
       └─ _data/hour.csv.gz + _data/manifest.json (committed, shipped in the wheel)
            ├─ mll data verify            -> provenance check (exit 0/2)
            ├─ mll data info --json       -> rows, window, checksum, license
            ├─ BundledDataset.raw()       -> raw schema -> validation / features
            ├─ BundledDataset.archive()   -> 2011 rows for the bootstrap champion
            └─ BundledDataset.window()    -> 2012 weekly replay batches
```

`mll data fetch --source uci --dest DIR` vuelve a descargar el ZIP fijado, verifica el sha256 del ZIP
y luego el sha256 del `hour.csv` interno antes de escribir nada. Es opcional, nunca lo usan las
pruebas ni la ruta de ejecución por defecto, y rechaza las discrepancias con `ChecksumMismatchError`
(código de salida 2 de la CLI).

## Reglas de validación

`mll validate` implementa el contrato de datos de `domain/validation.py`. Es una función pura: todas
las reglas se evalúan (sin cortocircuito) y el frame de entrada nunca se modifica. Una regla `ERROR`
fallida hace que el veredicto sea `FAIL`; una regla `WARN` fallida lo convierte en `WARN`; en otro
caso es `PASS`. Un lote `FAIL` se pone en cuarentena en el tick del ciclo de vida: nunca se puntúa,
nunca se registra y nunca se entrena con él.

| Regla | Contexto | Severidad | Tolerancia |
|---|---|---|---|
| `required_columns` | ambos | ERROR | 0 |
| `no_nulls` | ambos | ERROR | 0 |
| `timestamp_hourly` | ambos | ERROR | 0 |
| `timestamp_unique` | ambos | ERROR | 0 |
| `timestamp_monotonic` | ambos | ERROR | 0 |
| `season_domain` (1–4) | ambos | ERROR | 0 |
| `binary_domain` (`holiday`, `workingday`) | ambos | ERROR | 0 |
| `weathersit_domain` (1–4) | ambos | ERROR | 0 |
| `numeric_range` (`temp/atemp/hum/windspeed` ∈ [0,1]) | ambos | ERROR | 0 |
| `hum_positive` | ambos | WARN | 0 |
| `gap_hours` (hueco máximo ≤ 48 h) | ambos | WARN | 0 |
| `label_consistency` (`cnt == casual + registered`, `cnt ≥ 0`) | entrenamiento | ERROR | 0 |
| `batch_size` (1…1000 filas) | lote | ERROR | 0 |

El dataset incluido se valida como `WARN` con exactamente una regla fallida: `hum_positive`, 22 filas,
todas del 2011-03-10. No hay reglas `ERROR` fallidas. El plan mencionaba una tolerancia de 0.5% para
`hum_positive`; S3 y la aceptación de T2 exigen que esas 22 filas se reporten como la única
advertencia, así que la tolerancia implementada es cero y cualquier humedad nula queda visible.

## Particularidades conocidas

- La página de UCI indica 17,389 instancias; el `hour.csv` real tiene 17,379 filas.
- Faltan 165 horas porque no hubo alquileres en esas horas (0 nulos, 0 marcas de tiempo duplicadas).
  El mayor hueco entre filas consecutivas es de 1 día 13 horas.
- `windspeed = 0` en 2,180 filas; `weathersit = 4` solo en 3 filas.
- `hum = 0` en 22 filas, todas del 2011-03-10; por eso el dataset es `WARN` y no `PASS`.
- El gzip comprometido se produce con `gzip -9 -n` (sin nombre de archivo ni mtime en el encabezado),
  de ahí los 256,659 bytes; la nota de planificación decía 256,668 bytes porque provenía de una
  ejecución de gzip que guardó metadatos del encabezado. El sha256 descomprimido es la identidad
  vinculante.
- El replay trata la mitad de 2012 como producción. Las etiquetas llegan con un tick de retraso por
  diseño; los datos en sí no tienen etiquetas faltantes.

## Artefactos de resultados

Los resultados comprometidos del drill se generan con:

```bash
uv run mll drill run --scenario all \
  --report-dir docs/results --workspace "$(mktemp -d)"
uv run python scripts/render_results_table.py
```

`docs/results/drill-*.json` es la fuente de verdad reproducible (`schema: drill-report/v1`, semilla
42); los archivos `.md` se renderizan a partir de él, y
[`results-table.md`](results/results-table.md) se genera a partir de los cuatro reportes JSON con
`scripts/render_results_table.py --check` en la compuerta de documentación. Los dos reportes curados
de Evidently en `assets/reports/` usan las ventanas del dataset descritas allí y comparten un único
bundle de JavaScript extraído, para que el directorio se mantenga por debajo de 5 MB. Para
regenerarlos, ejecuta `uv run python scripts/render_drift_reports.py`.

## Reproducir el bundle

```bash
tmp="$(mktemp -d)"
uv run mll data fetch --source uci --dest "$tmp"
gzip -9 -n -c "$tmp/hour.csv" > "$tmp/hour.csv.gz"
uv run mll data verify --path "$tmp/hour.csv.gz"
uv run mll validate --bundled --json
```
