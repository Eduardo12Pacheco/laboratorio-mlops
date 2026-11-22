# ADR-0001: Arquitectura hexagonal con dirección de dependencias forzada

## Contexto

El laboratorio es un sistema de ciclo de vida, no un cuaderno: validación de datos, construcción
de features, entrenamiento, acceso al registro, servicio, monitoreo y scheduling coexisten. Sin
un límite explícito, el código de ML tiende a filtrar imports de frameworks (MLflow, FastAPI,
LightGBM) hacia el núcleo, lo que hace que el dominio sea difícil de probar y fácil de romper
cuando cambia un adaptador. Los estándares del proyecto exigen puertos y adaptadores con una
prueba que haga cumplir la dirección de dependencias.

## Decisión

Usar arquitectura hexagonal con cinco capas y una regla de dependencia unidireccional:

```
entrypoints  (CLI, API, dashboard, drill)
bootstrap | config | logging_setup   (composition root and cross-cutting wiring)
adapters     (MLflow, SQLite, LightGBM, FastAPI client, APScheduler, Evidently)
application  (ports, DTOs, use cases, pipeline)
domain       (pure: schema, features, metrics, drift, gates, policies, lifecycle)
```

- La capa de dominio importa solo la biblioteca estándar, `numpy`, `pandas` y `scipy`.
- Los servicios de aplicación dependen de Protocols ("puertos"), nunca de adaptadores concretos.
- Los adaptadores pueden depender de application y domain; los subpaquetes de adaptadores son
  independientes entre sí (ningún adaptador importa a otro hermano).
- La raíz de composición (`bootstrap`) es el único lugar que instancia adaptadores reales.
- La regla es ejecutable: los contratos de `import-linter` viven en `pyproject.toml` y se ejecutan
  en CI mediante `uv run lint-imports`. `tests/architecture/test_layers.py` también demuestra que
  la verificación falla ante una violación plantada a propósito, para que la compuerta no se
  degrade en silencio.

## Consecuencias

- Las pruebas de dominio y de aplicación se ejecutan sin MLflow, base de datos ni red.
- Cambiar adaptadores (por ejemplo, un gateway HTTP de scoring en Compose frente al de proceso
  único del drill) es un cambio solo en la raíz de composición.
- Algo más de código repetitivo: cada efecto secundario necesita un puerto. Se acepta porque la
  alternativa —llamadas al framework dispersas en los casos de uso— no es probable de probar a
  este tamaño.
- Los atajos nuevos entre capas fallan en CI en lugar de descubrirse en la revisión.

## Evidencia

- `[tool.importlinter]` en `pyproject.toml` define las capas y los contratos de dependencias
  externas prohibidas e independencia de adaptadores; la aceptación de T1 ejecuta
  `uv run lint-imports` y espera `Contracts: 3 kept, 0 broken.`
- La prueba negativa construye un paquete descartable cuyo `domain` importa `adapters` y verifica
  que import-linter reporta el contrato roto.
