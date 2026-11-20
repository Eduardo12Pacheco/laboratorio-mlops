"""mlops-lifecycle-lab: a local-first MLOps lifecycle lab."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from mlops_lifecycle_lab.config import apply_privacy_defaults

# MLflow decides whether to create its telemetry client while it is being imported, so the
# offline defaults (D22) must be in the environment before any adapter pulls it in. Every
# entrypoint (CLI, API factory, Streamlit app) imports this package first.
apply_privacy_defaults()

DISTRIBUTION_NAME = "mlops-lifecycle-lab"

try:
    __version__ = version(DISTRIBUTION_NAME)
except PackageNotFoundError:  # pragma: no cover - only when running from a bare tree
    __version__ = "0.0.0"

__all__ = ["DISTRIBUTION_NAME", "__version__"]
