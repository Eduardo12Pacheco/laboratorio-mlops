# Contribuir

Este repositorio se construye en incrementos pequeños y revisables, con un flujo test-first. Todo cambio
debe mantener verdes las compuertas de calidad documentadas antes de subirse a `main`.

## Configuración

```bash
uv sync --locked
uv run mll --version
uv run mll doctor
```

Se requiere Python 3.12. Linux y Windows son compatibles directamente; en macOS instala una vez el runtime
de OpenMP de LightGBM con `brew install libomp`. Docker solo se necesita para la ruta L2 con Compose.

## Instaladores

- `./install.sh --check` valida Docker >= 24, Compose v2, un motor Linux accesible, los puertos libres del
  host 14000/14080/14501 y al menos 4 GB de disco libres, sin efectos secundarios.
- `./install.sh` construye e inicia el stack; `./install.sh --uninstall [--yes]` elimina el stack, sus
  volúmenes y las imágenes locales, pero conserva `.env`.
- `./install.sh --check --l1` valida la ruta sin Docker (`uv`, Python 3.12 y `mll doctor`).
- `install.ps1` replica las mismas opciones para Windows PowerShell 5.1 y PowerShell 7 (`-Check`, `-L1`,
  `-Uninstall`, `-Yes`, `-NoStart`, `-ProjectName`).

Los códigos de salida son compartidos: `0` correcto, `1` fallaron los prerrequisitos, `2` error de uso,
`3` la instalación falló, `4` desinstalación abortada.

## Compuertas de calidad

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run lint-imports
uv run pytest --cov=mlops_lifecycle_lab --cov-fail-under=85 -m "not docker and not ui" -q
uv lock --check
uv run python scripts/check_docs.py
uv run python scripts/render_results_table.py --check
uvx --from shellcheck-py shellcheck install.sh scripts/*.sh
uvx check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/*.yml
```

## Documentación

- `docs/ARCHITECTURE.md`, `docs/DATA.md`, `docs/MODEL_CARD.md` y `docs/OPERATIONS.md` describen el
  sistema; los nueve ADR de `docs/adr/` mantienen la forma Contexto / Decisión / Consecuencias / Evidencia.
- Cada número de la documentación proviene de un `docs/results/drill-*.json` comprometido o de un comando
  documentado. Nunca edites a mano `docs/results/results-table.md`; vuelve a generarlo con
  `uv run python scripts/render_results_table.py`.
- `uv run python scripts/check_docs.py` valida los enlaces relativos, la estructura de los ADR, el esquema
  JSON de los drills, la tabla generada, los bloques de resultados del README y de la model card, y el
  tamaño del bundle de reportes. Imprime `docs OK` si todo está bien.
- Para regenerar los resultados comprometidos del drill:
  `uv run mll drill run --scenario all --report-dir docs/results --workspace "$(mktemp -d)"`.
  Regenerar los goldens es un paso aparte y explícito (`scripts/regen_golden.py`) y debe justificarse en
  el cuerpo del commit.
- Los reportes curados de Evidently en `docs/assets/reports/` se regeneran con
  `uv run python scripts/render_drift_reports.py`; comparten un único bundle de JavaScript extraído y deben
  mantenerse dentro del presupuesto de 5 MB.

La suite completa incluye las pruebas lentas del drill. `tests/installers` ejecuta el instalador de shell
mediante shims en `PATH`, así que nunca toca el daemon real de Docker. La prueba de parseo de PowerShell se
ejecuta cuando `pwsh` está instalado y se omite con un motivo en caso contrario; el contrato estático
siempre se ejecuta.

## Convenciones

- Conventional Commits en inglés, un comportamiento coherente por commit, con pruebas y documentación junto
  al comportamiento. Sin líneas de atribución de IA.
- Test-first: escribe primero la prueba que falla, implementa y luego refactoriza. Nunca debilites ni
  elimines una prueba para que todo esté en verde.
- Mantén la dirección de dependencias hexagonal: `domain` → `application` → `adapters` →
  `bootstrap`/`config` → `entrypoints`. Import-linter lo verifica.
- Sin secretos, sin acceso a red en las pruebas y sin archivos de lock editados a mano: ejecuta `uv lock`.
- Los logs estructurados y los settings tipados (variables de entorno `MLL_*`) son la única superficie de
  configuración; documenta las nuevas variables en `docs/OPERATIONS.md`.

## Releases

`main` siempre es desplegable. Un release es un tag `v*` cuya versión coincide con `pyproject.toml`. El
workflow `Release` construye el wheel, el sdist y el bundle reproducible de Compose
(`scripts/build_compose_bundle.py`), escribe `SHA256SUMS`, genera las notas a partir de Conventional Commits
con git-cliff y publica el release en GitHub. Nunca muevas ni elimines un tag ya publicado.
