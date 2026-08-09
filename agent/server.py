"""Unix domain socket server for WhatsApp Leo agent."""

import os
import asyncio

import orjson

from config import SOCKET_PATH, MAX_MESSAGE_SIZE, prewarm_whisper
from message_handler import process_message
from briefing_executor import execute_briefing_prompt
from logging_setup import logger
from whatsapp import send_message as whatsapp_send_message
from reminder import ReminderScheduler, RecurringReminderScheduler
from briefing import BriefingScheduler
from hooks import init_hooks, cleanup_hooks
from mcp_pool import mcp_pool


async def handle_client(reader, writer):
    """Handle a single client connection."""
    try:
        # Read to EOF, then parse once. Parsing after every chunk made this
        # O(n²) in the payload size.
        chunks = bytearray()
        message_data = None
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            chunks.extend(chunk)
            # Check message size limit
            if len(chunks) > MAX_MESSAGE_SIZE:
                logger.error(
                    "Message too large (%d bytes), dropping connection", len(chunks)
                )
                writer.write(orjson.dumps({"error": "Message too large"}))
                await writer.drain()
                return

        if not chunks:
            return

        try:
            message_data = orjson.loads(chunks)
        except orjson.JSONDecodeError as e:
            logger.warning("Discarding malformed payload (%d bytes): %s", len(chunks), e)
            message_data = None

        if message_data is not None:
            # Process immediately in background task
            asyncio.create_task(process_message(message_data))

            response = orjson.dumps(
                {"status": "processing", "message": "Message received"}
            )
            writer.write(response)
            await writer.drain()
        else:
            writer.write(b'{"error": "Invalid JSON"}')
            await writer.drain()

    except (ConnectionResetError, BrokenPipeError):
        # Client disconnected during processing - this is expected behavior
        pass
    except Exception as e:
        logger.error(f"Error handling client: {e}")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.warning(f"Error closing writer: {e}")


async def main():
    """Start the Unix domain socket server."""
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    # Bring up the MCP servers once for the process lifetime. Done in the
    # background so the socket starts accepting immediately; the first message
    # awaits ensure_started() and reuses whatever this task set up.
    asyncio.create_task(mcp_pool.ensure_started())

    # Load the transcription model now rather than making the first voice note
    # wait for a model download plus load. Threaded: it is CPU/IO-bound.
    asyncio.create_task(asyncio.to_thread(prewarm_whisper))

    # Start the reminder scheduler in the background
    scheduler = ReminderScheduler(send_fn=whatsapp_send_message)
    asyncio.create_task(scheduler.run())

    # Start the briefing scheduler in the background
    briefing_scheduler = BriefingScheduler(
        execute_fn=execute_briefing_prompt,
        send_fn=whatsapp_send_message,
    )
    asyncio.create_task(briefing_scheduler.run())

    # Start the recurring reminder scheduler in the background
    recurring_scheduler = RecurringReminderScheduler(send_fn=whatsapp_send_message)
    asyncio.create_task(recurring_scheduler.run())

    # Initialise hook FIFOs (no-op if IS_HOOK_ENABLED is false)
    init_hooks()

    server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
    # 0666 let any local user post a message with a spoofed phone_number from
    # ALLOWED_SENDERS, which grants Gmail/Calendar/memory access. The bridge
    # runs as the same user, so owner-only is all it needs.
    os.chmod(SOCKET_PATH, 0o600)

    logger.info(f"Unix domain socket Agent Server running at {SOCKET_PATH}")

    try:
        async with server:
            await server.serve_forever()
    finally:
        cleanup_hooks()
        await mcp_pool.stop()
