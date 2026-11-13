import json
import logging

from codeguard.core.logging_config import (
    JsonFormatter,
    configure_logging,
)


def test_json_formatter_creates_valid_json() -> None:
    record = logging.LogRecord(
        name="codeguard.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="test_event",
        args=(),
        exc_info=None,
    )

    result = JsonFormatter().format(record)
    data = json.loads(result)

    assert data["level"] == "INFO"
    assert data["logger"] == "codeguard.test"
    assert data["message"] == "test_event"


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.LogRecord(
        name="codeguard.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="application_created",
        args=(),
        exc_info=None,
    )
    record.environment = "development"

    data = json.loads(JsonFormatter().format(record))

    assert data["environment"] == "development"


def test_configure_logging_rejects_invalid_level() -> None:
    try:
        configure_logging("INVALID")
    except ValueError as exc:
        assert "Invalid logging level" in str(exc)
    else:
        raise AssertionError("Invalid logging level must be rejected.")
