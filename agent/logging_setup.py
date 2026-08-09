"""Logging configuration for WhatsApp Leo agent."""

import logging
import logging.handlers
import os
import time
from collections import deque

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Optional file log. Rotated so a long-running instance cannot fill the disk.
LOG_FILE = os.getenv("LOG_FILE", "")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Configure root logging
logging.basicConfig(level=LOG_LEVEL, format=_FORMAT, datefmt=_DATEFMT)

logger = logging.getLogger("AgentServer")

if LOG_FILE:
    _file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    _file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logging.getLogger().addHandler(_file_handler)

# Deque for test mode live log streaming
log_deque = deque(maxlen=500)

# Recent warnings and errors, for #status. (timestamp, level, logger, message)
error_deque: deque[tuple[float, str, str, str]] = deque(maxlen=20)

# Process start time, for uptime reporting.
STARTED_AT = time.time()


class DequeLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_deque.append(log_entry)


class ErrorTracker(logging.Handler):
    """Keep the most recent problems so #status can report them."""

    def emit(self, record):
        if record.levelno < logging.WARNING:
            return
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            message = str(record.msg)
        error_deque.append(
            (record.created, record.levelname, record.name, message[:300])
        )


deque_handler = DequeLogHandler()
deque_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
)
# Attach to root logger to capture ALL module logs (e.g. httpx, tools, etc.)
logging.getLogger().addHandler(deque_handler)
logging.getLogger().addHandler(ErrorTracker())
