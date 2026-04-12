"""
scripts/core/logging_utils.py — Structured JSON logging with request ID tracing.

Provides a StructuredLogger class that wraps Python's logging module with:
- Automatic JSON serialization
- Request-scoped trace IDs via contextvars
- Job ID propagation
- Human-readable console output in dev mode
- JSON-formatted output in production

Usage:
    from scripts.core.logging_utils import get_logger

    logger = get_logger("mymodule")
    logger.info("Processing started", extra={"user_id": "12345"})

    # Inside a request context (set via middleware):
    logger.error("Job failed", extra={"job_id": job_id, "error_code": 500})
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Context variables for request and job scoping
# ---------------------------------------------------------------------------

_request_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)
_job_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "job_id", default=None
)


def set_request_id(request_id: str) -> None:
    """Set the request ID for the current context (e.g., in middleware)."""
    _request_id_context.set(request_id)


def get_request_id() -> Optional[str]:
    """Retrieve the current request ID, or None if not in a request context."""
    return _request_id_context.get()


def set_job_id(job_id: str) -> None:
    """Set the job ID for the current context."""
    _job_id_context.set(job_id)


def get_job_id() -> Optional[str]:
    """Retrieve the current job ID, or None if not in a job context."""
    return _job_id_context.get()


# ---------------------------------------------------------------------------
# JSON Formatter for production logging
# ---------------------------------------------------------------------------


class StructuredJsonFormatter(logging.Formatter):
    """
    Log formatter that outputs structured JSON with all relevant metadata.

    Output format:
    {
        "timestamp": "2024-03-22T14:35:22.123456Z",
        "level": "INFO",
        "logger_name": "scripts.api.routes",
        "message": "Remix task started",
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "extra_fields": {
            "user_id": "12345",
            "duration_ms": 1234
        }
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        # Build base log object
        log_obj: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
        }

        # Add context variables if present
        request_id = get_request_id()
        if request_id:
            log_obj["request_id"] = request_id

        job_id = get_job_id()
        if job_id:
            log_obj["job_id"] = job_id

        # Add any extra fields passed via the extra dict
        # Exclude standard logging fields to avoid duplication
        extra_keys = {
            "message",
            "asctime",
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
            "request_id",
            "job_id",
        }
        extra_fields = {k: v for k, v in record.__dict__.items() if k not in extra_keys}

        if extra_fields:
            log_obj["extra"] = extra_fields

        # Capture exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": record.exc_text
                or (
                    "".join(traceback.format_exception(*record.exc_info))
                    if record.exc_info[0]
                    else None
                ),
            }

        return json.dumps(log_obj)


# ---------------------------------------------------------------------------
# Human-readable formatter for development
# ---------------------------------------------------------------------------


class StructuredConsoleFormatter(logging.Formatter):
    """
    Log formatter with colors and readable structure for development.
    Includes context variables inline for easy tracing.
    """

    # ANSI color codes
    _COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with colors and context."""
        # Get color for this level
        color = self._COLORS.get(record.levelname, "")

        # Build context suffix
        context_parts = []
        request_id = get_request_id()
        if request_id:
            context_parts.append(f"req={request_id[:8]}")

        job_id = get_job_id()
        if job_id:
            context_parts.append(f"job={job_id[:8]}")

        context_suffix = f" [{', '.join(context_parts)}]" if context_parts else ""

        # Format timestamp
        timestamp = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).strftime("%H:%M:%S")

        # Build the formatted string
        formatted = (
            f"{color}{timestamp} [{record.levelname:8}]{self._RESET} "
            f"{record.name}: {record.getMessage()}{context_suffix}"
        )

        # Include exception traceback if present
        if record.exc_info:
            formatted += "\n" + (
                record.exc_text
                or "".join(traceback.format_exception(*record.exc_info))
            )

        return formatted


# ---------------------------------------------------------------------------
# Structured Logger class
# ---------------------------------------------------------------------------


class StructuredLogger:
    """
    Wrapper around Python's logging module that provides structured logging
    with context-aware request and job ID propagation.

    Features:
    - Automatic JSON serialization in production
    - Human-readable console output in dev
    - Request ID and job ID from context variables
    - Extra fields support for custom metadata
    """

    def __init__(self, name: str, logger: logging.Logger):
        self._name = name
        self._logger = logger

    def _log(self, level: int, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Internal method to log with extra context."""
        extra_dict = extra or {}
        self._logger.log(level, msg, extra=extra_dict)

    def debug(self, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, msg, extra)

    def info(self, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log an info message."""
        self._log(logging.INFO, msg, extra)

    def warning(self, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, msg, extra)

    def warn(self, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Alias for warning()."""
        self.warning(msg, extra)

    def error(self, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log an error message."""
        self._log(logging.ERROR, msg, extra)

    def critical(self, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log a critical message."""
        self._log(logging.CRITICAL, msg, extra)

    def exception(self, msg: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log an exception with traceback."""
        extra_dict = extra or {}
        self._logger.exception(msg, extra=extra_dict)


# ---------------------------------------------------------------------------
# Global logger factory and initialization
# ---------------------------------------------------------------------------

_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """
    Factory function to get or create a StructuredLogger by name.

    Args:
        name: Module name (typically __name__)

    Returns:
        A StructuredLogger instance suitable for use within the module.

    Example:
        from scripts.core.logging_utils import get_logger

        logger = get_logger(__name__)
        logger.info("Application started")
    """
    if name not in _loggers:
        python_logger = logging.getLogger(name)
        _loggers[name] = StructuredLogger(name, python_logger)
    return _loggers[name]


def configure_structured_logging(
    level: str = "INFO",
    json_output: Optional[bool] = None,
) -> None:
    """
    Configure structured logging for the entire application.

    This function sets up the root logger and all future loggers with
    the appropriate formatter and handlers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, use JSON formatter; if False, use console;
                     if None (default), auto-detect based on isatty() and env.

    Environment variables:
        REMIXMATE_LOG_JSON: Set to 1/true to force JSON output
        REMIXMATE_LOG_LEVEL: Override the level parameter
    """
    # Allow environment overrides
    log_level = os.getenv("REMIXMATE_LOG_LEVEL", level).upper()
    logging_level = getattr(logging, log_level, logging.INFO)

    # Auto-detect output mode if not specified
    if json_output is None:
        # Force JSON if not a TTY or if env var is set
        force_json = os.getenv("REMIXMATE_LOG_JSON", "").lower() in ("1", "true", "yes")
        is_tty = sys.stderr.isatty()
        json_output = force_json or not is_tty

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging_level)

    # Set formatter based on output mode
    if json_output:
        formatter = StructuredJsonFormatter()
    else:
        formatter = StructuredConsoleFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging_level)
