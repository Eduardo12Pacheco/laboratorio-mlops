# ADR-0002: uv, Python 3.12 y paquete con layout src

## Contexto

El proyecto debe instalarse de forma reproducible en Linux, Windows y macOS, ofrecer una ruta CLI
L1 (`uv tool install`) y un stack Docker Compose L2, y mantener un lockfile que CI pueda verificar
sin volver a resolver dependencias. La pila de ML (MLflow, LightGBM, Evidently) es sensible a las
versiones del intérprete y de las dependencias; las sondas de planificación fijaron un conjunto
funcional sobre Python 3.12.

## Decisión

- Soportar solo CPython 3.12 (`requires-python = ">=3.12,<3.13"`), registrado en
  `.python-version`. Otros intérpretes se rechazan en lugar de soportarse a medias.
- Usar **uv** para entornos, locking y builds. `uv.lock` se versiona y CI ejecuta
  `uv sync --locked` más `uv lock --check`; el lock nunca se edita a mano.
- Usar un layout `src/` con un único paquete `mlops_lifecycle_lab`, construido con hatchling, y
  exponer la CLI como `mll = "mlops_lifecycle_lab.entrypoints.cli.app:main"`.
- Fijar los rangos de dependencias del plan (por ejemplo, `pandas>=3.0.6,<3.1` porque Evidently
  0.7.23 no importa en pandas 3.1; `apscheduler<4` porque 4.x es alpha). Los pins se documentan
  junto a la dependencia para que las actualizaciones futuras sean deliberadas.
- Mantener las herramientas de desarrollo y captura en grupos de dependencias (`dev` por defecto,
  `capture` bajo demanda) para que la imagen de runtime y la instalación L1 sigan siendo livianas.

## Consecuencias

- Un único lockfile reproduce el mismo grafo de dependencias para CI, Docker y las ejecuciones
  locales.
- El rango estrecho de Python implica que los nuevos intérpretes se adoptan con una actualización
  explícita del lock.
- `uv tool install <wheel>` funciona sin virtualenv para la ruta L1, y el mismo wheel alimenta la
  imagen de Compose.
- `pandas-stubs` es solo una dependencia de desarrollo y no puede filtrarse al runtime.

## Evidencia

- Sonda de planificación: un `uv lock` universal de toda la pila en Python 3.12 resolvió 182
  paquetes sin conflictos; `uv sync --locked` más las llamadas de humo a MLflow/LightGBM/Evidently
  pasaron con pandas 3.0.6 y fallaron en la importación de Evidently con pandas 3.1.1.
- Aceptación de T1: `uv sync --locked`, `uv lock --check` y `uv run mll --version` terminan con
  éxito, y el wheel incluye el paquete `_data` para tareas posteriores.
