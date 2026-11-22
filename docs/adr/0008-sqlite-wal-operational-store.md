# ADR-0008: Un único almacén operativo SQLite en modo WAL

## Contexto

El ciclo de vida necesita un log de predicciones, etiquetas tardías, un stream de eventos de
auditoría, snapshots de monitoreo, estado clave/valor (cursor del replay, probation) y un run
ledger. Un segundo motor de base de datos añadiría un servicio, credenciales y una herramienta de
migración a un proyecto local-first cuyo estado operativo completo es pequeño. La API, el
scheduler, el drill y el dashboard también necesitan leer y escribir concurrentemente sin corromper
el estado, incluido en Windows.

## Decisión

- Usar `sqlite3` de la biblioteca estándar y un único archivo, `<workspace>/mll.db`, para todas las
  tablas operativas. MLflow mantiene su propio `mlflow.db` y su almacén de artefactos; nunca se
  mezclan los dos.
- Abrir cada conexión con `journal_mode=WAL`, `busy_timeout=5000`, `foreign_keys=ON` y
  `row_factory=sqlite3.Row`. WAL da a los lectores una vista consistente mientras un escritor
  confirma, y el busy timeout absorbe el solapamiento corto normal.
- Versionar el esquema con migraciones ordenadas registradas en `schema_migrations`; el DDL vive en
  el código, de modo que un workspace nuevo y uno actualizado converjan de forma determinista.
- Usar el patrón repositorio detrás de los puertos de aplicación y una unidad de trabajo mínima
  (`with db.transaction():`) para las escrituras de varias filas. Las marcas de tiempo son texto
  ISO-8601 UTC sin zona, igual que el dataset.
- FastAPI ejecuta los handlers síncronos en hilos de trabajo, así que la conexión compartida se abre
  con `check_same_thread=False` y se protege con un `RLock` alrededor de `transaction()`. Es una
  preocupación específica del servicio; el contrato de WAL y de busy timeout no cambia.

## Consecuencias

- Cero servicios externos, respaldo trivial (copiar el workspace), y el SQL es inspeccionable con
  `sqlite3`. El dashboard y la CLI leen el mismo estado que escribe la API.
- WAL sobre un bind mount de Docker no es seguro en algunos hosts, por lo que el stack de Compose
  mantiene SQLite solo en el volumen con nombre `mll-data`.
- SQLite es un motor de un solo escritor: una concurrencia de escritura muy alta queda fuera de
  alcance. La API agrupa las escrituras de predicciones por petición y el scheduler tiene un tick a
  la vez.
- Las migraciones son aditivas y están en el código; no hay ruta de downgrade. Un cambio de esquema
  incompatible requeriría un workspace nuevo o un script de migración explícito.
- Un archivo `-wal`/`-shm` residual es normal; no hay que borrarlo mientras la base de datos está
  abierta.

## Evidencia

- Las pruebas de integración afirman que `PRAGMA journal_mode` devuelve `wal` y que se aplica
  `busy_timeout`; la suite de contratos del log de predicciones pasa tanto contra el fake en memoria
  como contra el adaptador SQLite.
- La prueba de serving con stack real expuso el error de conexión entre hilos y ahora pasa con la
  conexión compartida/RLock: predicción online, un lote de 168 filas, un reenvío idempotente y un
  join con etiquetas, todo ejecutado a través de uvicorn en un puerto de loopback.
- La prueba de humo de Compose verifica la misma ruta de código dentro del contenedor con la base de
  datos respaldada por volumen, incluido el scheduler escribiendo mientras la API lee.
- Ver [ARCHITECTURE.md](../ARCHITECTURE.md) para los puertos de almacenamiento y
  [OPERATIONS.md](../OPERATIONS.md#solución-de-problemas) para la entrada de solución de problemas
  sobre locks.
