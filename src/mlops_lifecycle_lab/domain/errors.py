"""Domain error hierarchy shared by every layer.

The errors are deliberately small and dependency-free: adapters translate their own
failures into these types so the application and CLI never leak framework exceptions.
"""

from __future__ import annotations

from collections.abc import Sequence


class DomainError(Exception):
    """Base class for every expected, user-facing failure of the lab."""


class ChecksumMismatchError(DomainError):
    """A downloaded or bundled artifact does not match its pinned checksum."""

    def __init__(self, source: str, expected: str, actual: str) -> None:
        self.source = source
        self.expected = expected
        self.actual = actual
        super().__init__(f"{source} checksum mismatch: expected {expected}, got {actual}")


class ValidationFailedError(DomainError):
    """A frame failed the data contract and must not be scored or trained on."""

    def __init__(
        self,
        failed_rules: Sequence[str] | None = None,
        message: str | None = None,
    ) -> None:
        self.failed_rules = tuple(failed_rules or ())
        if message is None:
            detail = ", ".join(self.failed_rules) or "unknown rules"
            message = f"data validation failed: {detail}"
        super().__init__(message)


class IllegalTransitionError(DomainError):
    """A model lifecycle state machine transition is not allowed."""


class NoPreviousChampionError(DomainError):
    """A rollback was requested but there is no previous champion to restore."""


class NoChampionError(DomainError):
    """Serving was requested but the registry has no champion to load."""


class UnknownModelVersionError(DomainError):
    """A model version referenced by a use case does not exist in the registry."""


class NotEnoughLabelsError(DomainError):
    """A metric or decision needs more labeled rows than are available."""


class GateEvaluationError(DomainError):
    """A promotion gate could not be evaluated from the given evidence."""
