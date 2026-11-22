# ADR-0007: APScheduler para el replay y un run ledger propio

## Contexto

Un tick del ciclo de vida por semana simulada es todo el scheduling que necesita este proyecto.
Airflow, Prefect y Dagster traen cada uno una base de datos, un servidor web, workers, una UI y sus
propios conceptos; adoptar uno añadiría más piezas móviles que el propio ciclo de vida y haría más
difícil razonar sobre el stack Compose local-first. El proyecto aún necesita un cursor durable,
ticks que no se solapen, un apagado limpio y una pista de auditoría de cada intento.

## Decisión

- Usar APScheduler 3.11 con un `BackgroundScheduler` y un job de intervalo configurado con
  `max_instances=1`, `coalesce=True` y un `misfire_grace_time` derivado del intervalo. Así se
  obtienen ticks serializados sin un broker de mensajes.
- El cursor del replay vive en el KV de SQLite (`replay_tick`) y avanza **solo después** de que un
  tick se completa. Un reinicio continúa en `cursor + 1`; un tick fallido deja el cursor intacto.
- Exponer `mll scheduler start --mode replay|once`, compartir el mismo caso de uso
  `RunLifecycleTick` con el drill (que lo ejecuta sin dormir) y detenerse limpiamente con
  SIGINT/SIGTERM.
- Mantener la orquestación propia y pequeña: `Step`/`Pipeline` ejecuta pasos con nombre con
  reintentos acotados y registra una fila por intento en `runs`/`run_steps` (estado, duración,
  error). El ledger es la fuente de auditoría para el dashboard y para la garantía de "nunca
  corromper el KV ante un fallo".
- Las señales solo se instalan desde el hilo principal, y una autodetención en el último tick usa
  `shutdown(wait=False)` para que el worker no se una a sí mismo.

## Consecuencias

- Sin infraestructura adicional: un único proceso Python y el mismo archivo SQLite que el resto del
  estado operativo.
- APScheduler es una biblioteca, no una plataforma; los DAG complejos, los backfills y las
  dependencias entre jobs quedan fuera de alcance por diseño.
- Cada reintento es visible como una fila separada del ledger (`FAILED, FAILED, SUCCEEDED`) en
  lugar de aplanarse en un único estado final.
- Un tick que excede el intervalo se omite en lugar de solaparse; `coalesce=True` evita una ráfaga
  de ejecuciones de recuperación tras una pausa larga.
- El scheduler y el drill comparten comportamiento, así que un error encontrado en uno se corrige
  para ambos.

## Evidencia

- Pruebas del scheduler: un tick lento más largo que el intervalo nunca se ejecuta dos veces en
  paralelo; una segunda instancia del scheduler continúa en `cursor + 1`; el cursor sobrevive a un
  reinicio; un tick fallido se registra como `FAILED` y no avanza el cursor; `shutdown(wait=True)`
  retorna en menos de dos segundos; SIGTERM detiene el proceso en POSIX.
- La prueba de humo de Compose ejecuta `MLL_REPLAY_INTERVAL_SECONDS=1` y alcanza el campeón v2 a
  través de la API real, lo que ejercita en conjunto el scheduler, el cursor, la recarga en caliente
  de la API y el ledger.
- `mll quickstart` deja el mismo workspace en `tick 16 / 52`, y `mll replay reset` lo devuelve a
  `tick 0 / 52`, lo que demuestra que el cursor es un valor de estado normal e inspeccionable.
- Ver también [ARCHITECTURE.md](../ARCHITECTURE.md#un-tick-del-ciclo-de-vida) y
  [OPERATIONS.md](../OPERATIONS.md#control-del-replay).
