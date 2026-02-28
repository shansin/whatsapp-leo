#!/usr/bin/env python3
"""WhatsApp Leo Agent — entry point.

Dispatches to either the Unix socket server (production) or the Gradio
test UI based on the IS_TEST_MODE environment variable.
"""

import os
import asyncio

from config import SOCKET_PATH
from logging_setup import logger


def main():
    """Entry point: start production server or test UI."""
    is_test_mode = os.getenv("IS_TEST_MODE", "false").lower() == "true"

    if is_test_mode:
        from test_ui import start_test_ui

        start_test_ui()
    else:
        from server import main as server_main

        asyncio.run(server_main())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down Agent Server...")
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
