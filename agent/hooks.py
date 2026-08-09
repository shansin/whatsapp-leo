"""Hooks system for bidirectional communication with external programs.

Each hook gets two named pipes (FIFOs):
  • {name}-in.fifo   — WhatsApp → external program (program reads from this)
  • {name}-out.fifo  — external program → WhatsApp (program writes to this)

Inbound WhatsApp messages matching #hook-name or @hook-name are intercepted
(Leo won't process them) and the stripped body is written to the -in FIFO.

A background task continuously reads from each -out FIFO and sends the
content as "hook_name: message" to all ALLOWED_SENDERS via WhatsApp.
"""

import os
import asyncio
import stat

from config import HOOKS, ALLOWED_SENDERS, IS_HOOK_ENABLED, INSTANCE_GUID
from logging_setup import logger

# Track background reader tasks so we can cancel on cleanup
_reader_tasks: list[asyncio.Task] = []

# Active hook sessions: chat_jid → hook_name
_active_sessions: dict[str, str] = {}


def _fifo_path(hook_name: str, direction: str) -> str:
    """Return the FIFO path for a given hook name and direction (in/out)."""
    return f"/tmp/whatsapp-leo-hook-{INSTANCE_GUID}-{hook_name}-{direction}.fifo"


def _ensure_fifo(path: str, label: str) -> None:
    """Create a FIFO at path if it doesn't already exist."""
    if os.path.exists(path):
        if not stat.S_ISFIFO(os.stat(path).st_mode):
            os.unlink(path)
            os.mkfifo(path)
            logger.info(f"Recreated FIFO {label} at {path}")
        else:
            logger.info(f"FIFO already exists {label} at {path}")
    else:
        os.mkfifo(path)
        logger.info(f"Created FIFO {label} at {path}")


def init_hooks() -> None:
    """Create FIFOs for all configured hooks and start outbound readers."""
    if not IS_HOOK_ENABLED or not HOOKS:
        return

    for name in HOOKS:
        _ensure_fifo(_fifo_path(name, "in"), f"'{name}' inbound")
        _ensure_fifo(_fifo_path(name, "out"), f"'{name}' outbound")

    # Start background readers for outbound FIFOs
    for name in HOOKS:
        task = asyncio.create_task(_read_outbound_fifo(name))
        _reader_tasks.append(task)

    logger.info(f"Hooks initialised: {', '.join(HOOKS)}")


def cleanup_hooks() -> None:
    """Cancel readers and remove FIFOs for all configured hooks."""
    if not IS_HOOK_ENABLED or not HOOKS:
        return

    # Cancel background reader tasks
    for task in _reader_tasks:
        task.cancel()
    _reader_tasks.clear()

    for name in HOOKS:
        for direction in ("in", "out"):
            path = _fifo_path(name, direction)
            if os.path.exists(path):
                os.unlink(path)
        logger.info(f"Removed FIFOs for hook '{name}'")


async def _read_outbound_fifo(hook_name: str) -> None:
    """Background task: read from the outbound FIFO and send via WhatsApp.

    Opens the FIFO in a loop. Each line written by an external program
    becomes a WhatsApp message sent as "hook_name: line" to ALLOWED_SENDERS.
    """
    from whatsapp import send_message as whatsapp_send_message

    path = _fifo_path(hook_name, "out")
    logger.info(f"Hook '{hook_name}' outbound reader started on {path}")

    while True:
        try:
            # _blocking_read runs entirely in a thread so the event loop isn't blocked
            lines = await asyncio.to_thread(_blocking_read_lines, path)
            if not lines:
                continue  # Writer closed without content, re-open

            text = f"_*({hook_name})*_ : {chr(10).join(lines)}"
            for sender in ALLOWED_SENDERS:
                success, result = await asyncio.to_thread(
                    whatsapp_send_message, sender, text
                )
                if success:
                    logger.info(f"Hook '{hook_name}' sent to {sender}: {text[:80]}")
                else:
                    logger.error(
                        f"Hook '{hook_name}' send failed to {sender}: {result}"
                    )
        except asyncio.CancelledError:
            logger.info(f"Hook '{hook_name}' outbound reader stopped")
            return
        except Exception as e:
            logger.error(f"Hook '{hook_name}' outbound error: {e}")
            await asyncio.sleep(1)


def _blocking_read_lines(path: str) -> list[str]:
    """Blocking read of all non-empty lines from a FIFO. Runs in a thread.

    Opens the FIFO (blocks until a writer connects), reads every line
    until the writer closes (EOF), then returns them all. Returns an
    empty list when the writer closes without writing any content.

    IMPORTANT: Uses readline() not 'for line in file' — Python's file
    iterator uses a hidden read-ahead buffer that breaks on FIFOs.
    """
    logger.debug(f"FIFO reader: waiting for writer on {path}")
    lines: list[str] = []
    with open(path) as fifo:
        while True:
            line = fifo.readline()
            if not line:
                return lines  # EOF — writer closed
            stripped = line.strip()
            if stripped:
                lines.append(stripped)


async def write_to_hook(hook_name: str, message: str) -> None:
    """Write a message to the hook's inbound FIFO for external programs to read.

    Uses O_NONBLOCK|O_WRONLY so we never block the event loop if nobody is
    reading from the pipe. If the pipe has no reader, the message is dropped
    with a debug log.
    """
    path = _fifo_path(hook_name, "in")
    if not os.path.exists(path):
        logger.warning(f"FIFO for hook '{hook_name}' not found at {path}")
        return

    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, (message + "\n").encode())
            logger.info(f"Hook '{hook_name}' received: {message[:80]}")
        finally:
            os.close(fd)
    except OSError as e:
        # ENXIO (6) = no reader on the other end
        if e.errno == 6:
            logger.debug(f"No reader for hook '{hook_name}', dropping message")
        else:
            logger.error(f"Error writing to hook '{hook_name}': {e}")


def match_hook(content: str) -> tuple[str, str] | None:
    """Check if message content matches a hook prefix.

    Matches '#hook-name ...' or '@hook-name ...'.
    Returns (hook_name, stripped_body) or None.
    """
    if not IS_HOOK_ENABLED or not HOOKS:
        return None

    stripped = content.strip()
    for name in HOOKS:
        for prefix_char in ("#", "@"):
            prefix = f"{prefix_char}{name}"
            if stripped.lower().startswith(prefix.lower()):
                rest = stripped[len(prefix):]
                if not rest or rest[0].isspace():
                    return name, rest.strip()

    return None


def match_hook_session_command(content: str) -> tuple[str, str] | None:
    """Check if message is a hook session start/stop command.

    Matches '#hook-name #start' or '#hook-name #stop' (also with @).
    Returns (hook_name, "start"|"stop") or None.
    """
    if not IS_HOOK_ENABLED or not HOOKS:
        return None

    stripped = content.strip()
    for name in HOOKS:
        for prefix_char in ("#", "@"):
            prefix = f"{prefix_char}{name}"
            if stripped.lower().startswith(prefix.lower()):
                rest = stripped[len(prefix):].strip().lower()
                if rest == "#start":
                    return name, "start"
                if rest == "#stop":
                    return name, "stop"

    return None


def start_hook_session(chat_jid: str, hook_name: str) -> None:
    """Enter hook session mode for a chat — all messages forwarded to hook."""
    _active_sessions[chat_jid] = hook_name
    logger.info(f"Hook session started: chat={chat_jid} hook={hook_name}")


def stop_hook_session(chat_jid: str) -> str | None:
    """Exit hook session mode for a chat. Returns the hook name or None."""
    hook_name = _active_sessions.pop(chat_jid, None)
    if hook_name:
        logger.info(f"Hook session stopped: chat={chat_jid} hook={hook_name}")
    return hook_name


def get_hook_session(chat_jid: str) -> str | None:
    """Return the active hook name for a chat, or None."""
    return _active_sessions.get(chat_jid)
