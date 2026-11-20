"""Allow ``python -m mlops_lifecycle_lab`` to behave like the ``mll`` command."""

from __future__ import annotations

from mlops_lifecycle_lab.entrypoints.cli.app import main

if __name__ == "__main__":
    main()
