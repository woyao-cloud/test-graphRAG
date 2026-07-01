"""Structured logging setup for GraphRAG-KG.

Provides configured loggers with Rich console output and optional
file-based logging for pipeline observability.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    module_filter: Optional[str] = None,
) -> logging.Logger:
    """Configure structured logging for GraphRAG-KG.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional file path for persistent logging.
        module_filter: Optional module name filter (e.g., "graphrag_kg").

    Returns:
        Configured root logger for graphrag_kg.
    """
    logger = logging.getLogger("graphrag_kg")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "[%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler if requested
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s:%(lineno)d: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    # Add module filter if specified
    if module_filter:
        logger.addFilter(logging.Filter(module_filter))

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger under graphrag_kg."""
    return logging.getLogger(f"graphrag_kg.{name}")
