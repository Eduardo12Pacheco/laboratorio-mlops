"""Single structlog configuration for every entry point (D21)."""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog

LogFormat = Literal["json", "console"]


def configure_logging(level: str = "INFO", fmt: LogFormat = "console") -> None:
    """Configure structlog deterministically; repeated calls never stack processors.

    Console rendering is the developer default; containers use JSON so logs stay
    machine-readable. Log records carry the context-vars bound with
    :func:`bind_context` (``request_id``, ``run_id``, ``tick``).
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if fmt == "json":
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [*shared_processors, structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_level_value(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def bind_context(**kwargs: object) -> None:
    """Bind structured fields to every log record emitted in the current context."""
    structlog.contextvars.bind_contextvars(**kwargs)


def _level_value(level: str) -> int:
    return logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
