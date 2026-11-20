from __future__ import annotations

import json

import pytest
import structlog

from mlops_lifecycle_lab.logging_setup import bind_context, configure_logging


@pytest.fixture(autouse=True)
def clear_structlog_context() -> None:
    structlog.contextvars.clear_contextvars()


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO", "console")
    first_processors = len(structlog.get_config()["processors"])

    configure_logging("DEBUG", "console")

    assert len(structlog.get_config()["processors"]) == first_processors


def test_json_format_emits_structured_event(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    logger = structlog.get_logger("test")

    logger.info("hello", request_id="req-1")

    captured = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(captured)
    assert payload["event"] == "hello"
    assert payload["request_id"] == "req-1"
    assert payload["level"] == "info"


def test_bind_context_adds_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    logger = structlog.get_logger("test")

    bind_context(request_id="req-2", run_id="run-9")
    logger.info("bound")

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["request_id"] == "req-2"
    assert payload["run_id"] == "run-9"


def test_console_format_still_logs(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "console")

    structlog.get_logger("test").warning("visible-message")

    assert "visible-message" in capsys.readouterr().out
