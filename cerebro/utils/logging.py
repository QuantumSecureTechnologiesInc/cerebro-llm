"""Structured logging for Cerebro.

Provides structured JSON logging with:
- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Request tracing with correlation IDs
- OpenTelemetry-compatible span contexts
- Console and file output
- Train step-aware formatting
"""

from __future__ import annotations

import json
import sys
import time
import uuid
import logging
from contextvars import ContextVar
from typing import Optional

# Task-local correlation ID for request tracing
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_current_span: ContextVar[dict] = ContextVar("current_span", default={})


def set_correlation_id(cid: str | None = None) -> str:
    """Set or generate a correlation ID for the current task."""
    if cid is None:
        cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get()


class JsonFormatter(logging.Formatter):
    """JSON-structured log formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S.{}Z".format(
                    str(int(time.time() * 1000) % 1000).zfill(3)
                ),
                time.gmtime(record.created),
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # Include correlation ID if available
        cid = _correlation_id.get()
        if cid:
            log_entry["correlation_id"] = cid

        # Include extra fields
        if hasattr(record, "step"):
            log_entry["step"] = record.step
        if hasattr(record, "loss"):
            log_entry["loss"] = record.loss
        if hasattr(record, "lr"):
            log_entry["lr"] = record.lr

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_format: bool = True,
) -> None:
    """Configure structured logging for Cerebro.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for persistent logs.
        json_format: If True, use JSON-structured output (production).
                     If False, use human-readable format (development).
    """
    root = logging.getLogger("cerebro")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    if json_format:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file)
        if json_format:
            fh.setFormatter(JsonFormatter())
        else:
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(f"cerebro.{name}")


class TrainLogger:
    """Training-specific logger with step-aware metrics."""

    def __init__(self, log_interval: int = 10) -> None:
        self.logger = get_logger("trainer")
        self.log_interval = log_interval
        self._step = 0

    def log_metrics(self, step: int, metrics: dict, prefix: str = "train") -> None:
        """Log training metrics at the given step.

        Args:
            step: Current training step.
            metrics: Dict of metric_name -> value.
            prefix: Metric prefix (e.g., 'train', 'eval').
        """
        self._step = step
        msg_parts = [f"{prefix}/step={step}"]
        for k, v in metrics.items():
            msg_parts.append(f"{prefix}/{k}={v:.4f}")
        self.logger.info(" ".join(msg_parts))

    def log_step(self, step: int, loss: float, lr: float, tok_per_s: float) -> None:
        """Log a single training step.

        Args:
            step: Current step number.
            loss: Current loss value.
            lr: Current learning rate.
            tok_per_s: Tokens per second throughput.
        """
        self._step = step
        self.logger.info(
            "train/step=%d train/loss=%.4f train/lr=%.2e train/tok_per_s=%.0f",
            step, loss, lr, tok_per_s,
        )