"""mlops-lifecycle-lab: a local-first MLOps lifecycle lab."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION_NAME = "mlops-lifecycle-lab"

try:
    __version__ = version(DISTRIBUTION_NAME)
except PackageNotFoundError:  # pragma: no cover - only when running from a bare tree
    __version__ = "0.0.0"

__all__ = ["DISTRIBUTION_NAME", "__version__"]
