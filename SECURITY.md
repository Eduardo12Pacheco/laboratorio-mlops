# Security policy

## Supported versions

The latest release on `main` is supported.

## Reporting a vulnerability

Open a private security advisory on GitHub (Security → Advisories) or contact the
maintainer through the GitHub profile. Please do not open a public issue with exploit
details before a fix is available.

## Trust model

This is a local-first demo. Model artifacts stored in MLflow are treated as trusted local
state: loading a model executes pickle-based code. Run the stack only against registries
and artifacts you created or trust. No secret or API key is required to run the lab; the
optional admin API key (`MLL_ADMIN_API_KEY`) only guards the reload endpoint.
