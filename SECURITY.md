# Security policy

## Supported versions

The latest release and the current `main` branch are supported. This is a local-first
educational lab, not a hosted service.

## Reporting a vulnerability

Open a private security advisory on GitHub (Security → Advisories) or contact the
maintainer through the GitHub profile. Please do not open a public issue with exploit
details before a fix is available. Expect an initial response within a few days.

## Trust model

This is a local-first demo. Model artifacts stored in MLflow are treated as trusted local
state: loading a model executes pickle-based code. Run the stack only against registries
and artifacts you created or trust, and never point `MLL_MLFLOW_TRACKING_URI` at an
untrusted server. The optional `MLL_ADMIN_API_KEY` only guards `POST /v1/admin/reload`;
it is not an authentication system for the rest of the API, which binds to loopback by
default.

## Secrets and accounts

- Zero mandatory keys or accounts. Everything runs offline with the bundled dataset and
  deterministic tests.
- `.env.example` is the only environment template committed; `.env` is gitignored. Never
  commit a real `MLL_ADMIN_API_KEY` or a registry credential.
- The API exposes `GET /metrics` without authentication, like the rest of the local
  endpoints; do not publish it on a public interface.
- Telemetry is disabled by default for MLflow, Evidently/iterative-telemetry and
  Streamlit, so the lab does not phone home.

## Supply chain and container posture

- Runtime and development dependencies are locked in `uv.lock`; CI runs
  `uv lock --check`, `pip-audit` over the exported runtime requirements and a gitleaks
  history scan.
- The container image runs as the non-root user `mll` (uid 10001). Every Compose service
  drops all capabilities, sets `no-new-privileges`, uses a read-only root filesystem with
  a `/tmp` tmpfs, and publishes ports on `127.0.0.1` only.
- SQLite operational state and MLflow artifacts live on the named volume `mll-data`;
  they are trusted local state and are not backed up by the lab.
