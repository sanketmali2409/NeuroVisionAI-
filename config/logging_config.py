"""
Central logging configuration.

Every module in the project should obtain its logger via
`get_logger(__name__)` rather than calling `logging.getLogger` directly or
using `print`. This guarantees a consistent log format, consistent log
level (driven by `settings.LOG_LEVEL`), and that logs are written both to
the console and to a rotating file under `logs/`.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    """Attach console + rotating-file handlers to the root logger exactly once."""
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL.upper())

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = settings.LOGS_DIR / "neurovision.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Third-party libraries are noisy at INFO/DEBUG; keep them at WARNING
    # unless the app itself is running in DEBUG mode.
    if not settings.DEBUG:
        for noisy_logger in ("urllib3", "httpx", "matplotlib", "PIL"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the project's standard configuration applied.

    Args:
        name: Usually `__name__` of the calling module, so log lines show
            exactly which file emitted them.
    """
    _configure_root_logger()
    return logging.getLogger(name)
