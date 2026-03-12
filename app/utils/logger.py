"""
Structured Logging Utility (Phase 7)

WHAT IS THIS FILE?
------------------
A single place to configure Python's built-in logging module for the entire
DocuMind application.

WHY REPLACE print()?
--------------------
print() is fine for quick scripts, but has no:
  - Severity levels (INFO vs ERROR vs WARNING)
  - Timestamps (can't tell when something happened)
  - Module names (can't tell where a log came from)
  - Output control (can't silence debug logs in production)

Python's `logging` module gives all of this for free.

HOW TO USE:
-----------
    from app.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Processing document: report.pdf")
    logger.warning("No text extracted from page 3")
    logger.error("FAISS index not found")

LOG FORMAT:
-----------
    2024-01-01 12:00:00 | INFO     | document_service | Processing report.pdf
    2024-01-01 12:00:01 | WARNING  | retrieval_service | Low similarity score: 0.09
    2024-01-01 12:00:02 | ERROR    | llm_service | Groq API call failed

LOG LEVELS (lowest → highest severity):
    DEBUG    - Detailed diagnostic info (disabled in production)
    INFO     - Normal operation events
    WARNING  - Something unexpected, but recoverable
    ERROR    - A failure that needs attention
    CRITICAL - System-level failure
"""

import logging
import sys
from app.config.settings import settings


# ---------------------------------------------------------------------------
# Log format
# ---------------------------------------------------------------------------
# %(asctime)s      → timestamp: "2024-01-01 12:00:00"
# %(levelname)-8s  → level, padded to 8 chars: "INFO    " or "WARNING "
# %(name)-20s      → logger name (module), padded to 20 chars
# %(message)s      → the actual log message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Root logger config (runs once when this module is first imported)
# ---------------------------------------------------------------------------
def _configure_root_logger() -> None:
    """
    Set up the root logger with a console (stdout) handler.

    Called automatically when logger.py is first imported.
    All child loggers (created by get_logger) inherit this config.

    WHY stdout instead of stderr?
    ------------------------------
    By convention, application logs go to stdout so they appear in
    Docker container logs, Render logs, and uvicorn output cleanly.
    Errors still surface — they just also go to stdout, not just stderr.
    """
    root = logging.getLogger()

    # Don't re-configure if already set up (e.g. in tests)
    if root.handlers:
        return

    # Set level: DEBUG in dev, INFO in production
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    root.setLevel(level)

    # Console handler → writes to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root.addHandler(handler)

    # Silence noisy third-party debug logs
    logging.getLogger("pdfminer").setLevel(logging.WARNING)


# Run once on import
_configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a specific module.

    USAGE:
        logger = get_logger(__name__)

    Passing __name__ automatically uses the module's dotted name,
    e.g. "app.services.document_service" — this appears in the log output
    so you always know which file produced each log line.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        A configured Logger instance
    """
    return logging.getLogger(name)
