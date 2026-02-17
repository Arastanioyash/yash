"""Logging setup utilities."""

from __future__ import annotations

import logging


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once, or refresh the root level if already configured."""
    resolved_level = level
    if isinstance(level, str):
        resolved_level = logging.getLevelName(level.upper())
        if not isinstance(resolved_level, int):
            resolved_level = logging.INFO

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
        for handler in root_logger.handlers:
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
        return

    logging.basicConfig(level=resolved_level, format=LOG_FORMAT)
