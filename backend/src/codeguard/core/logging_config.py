import json
import logging
from datetime import UTC, datetime
from typing import Any

STANDARD_LOG_FIELDS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": (
                datetime.fromtimestamp(
                    record.created,
                    tz=UTC,
                ).isoformat()
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in STANDARD_LOG_FIELDS:
                log_data[key] = value

        return json.dumps(log_data, default=str)


def configure_logging(log_level: str) -> logging.Logger:
    """Configure the CodeGuard application logger."""

    level = getattr(logging, log_level.upper(), None)

    if not isinstance(level, int):
        raise ValueError(f"Invalid logging level: {log_level}")

    logger = logging.getLogger("codeguard")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger
