"""Logging configuration for WhatsApp Leo agent."""

import logging
from collections import deque

# Configure root logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("AgentServer")

# Deque for test mode live log streaming
log_deque = deque(maxlen=500)


class DequeLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_deque.append(log_entry)


deque_handler = DequeLogHandler()
deque_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
)
# Attach to root logger to capture ALL module logs (e.g. httpx, tools, etc.)
logging.getLogger().addHandler(deque_handler)
