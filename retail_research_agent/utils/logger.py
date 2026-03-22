"""Structured logging for pipeline steps."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


def get_logger(name: str = "retail_research_agent") -> logging.Logger:
    """Return a logger that emits JSON lines to stderr for easy parsing."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class _JsonFormatter(logging.Formatter):
    """Minimal structured formatter (one JSON object per line)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "structured", None)
        if isinstance(extra, Mapping):
            payload["extra"] = dict(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def log_step(
    logger: logging.Logger,
    step: str,
    *,
    status: str = "ok",
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    """Log a named pipeline step with optional structured details."""
    structured: dict[str, Any] = {"step": step, "status": status}
    if details:
        structured.update(dict(details))
    logger.info("pipeline_step", extra={"structured": structured})
