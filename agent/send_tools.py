"""Tools that let Leo send files back into the chat.

Deliberately narrow. A tool that sends "any path" to WhatsApp is an
exfiltration primitive: one prompt injection in a quoted message or a web
result and .env, the session database or a private key leaves the machine.
Sends are therefore restricted to an allowlist of directories, resolved
through symlinks, and only wired up for privileged senders.
"""

import asyncio
import os

from agents import function_tool

from config import STORE_DIR
from logging_setup import logger
from whatsapp import send_file

# Directories Leo may send from. STORE_DIR holds downloaded chat media (so Leo
# can forward an image you sent it); SHARE_DIR is an opt-in drop box.
SHARE_DIR = os.getenv("SHARE_DIR", "")

# Media too large for WhatsApp is rejected upstream anyway; fail early.
MAX_SEND_BYTES = int(os.getenv("MAX_SEND_BYTES", str(64 * 1024 * 1024)))


def allowed_roots() -> list[str]:
    roots = [os.path.realpath(STORE_DIR)]
    if SHARE_DIR:
        roots.append(os.path.realpath(os.path.expanduser(SHARE_DIR)))
    return roots


def resolve_sendable(path: str) -> str:
    """Return the real path of a sendable file, or raise ValueError.

    Resolves symlinks *before* checking, so a link planted inside the store
    cannot point at /etc/shadow.
    """
    if not path:
        raise ValueError("No file path given.")

    real = os.path.realpath(os.path.expanduser(path))
    roots = allowed_roots()
    if not any(
        real == root or real.startswith(root + os.sep) for root in roots
    ):
        raise ValueError(
            "That file is outside the directories Leo may send from "
            f"({', '.join(roots)})."
        )
    if not os.path.isfile(real):
        raise ValueError(f"No such file: {path}")
    if os.path.getsize(real) > MAX_SEND_BYTES:
        raise ValueError("That file is too large to send.")
    return real


def make_send_tools(chat_jid: str) -> list:
    """Create file-sending tools bound to the current chat."""

    @function_tool
    async def send_file_to_chat(file_path: str) -> str:
        """Send a file (image, document, audio) into this chat.

        Only files under Leo's own store or share directory can be sent.

        Args:
            file_path: path to the file to send
        """
        try:
            real = resolve_sendable(file_path)
        except ValueError as e:
            logger.warning(f"Refused to send {file_path!r}: {e}")
            return str(e)

        ok, detail = await asyncio.to_thread(send_file, chat_jid, real)
        if ok:
            logger.info(f"Sent file {real} to {chat_jid}")
            return f"Sent {os.path.basename(real)}."
        return f"Could not send the file: {detail}"

    return [send_file_to_chat]
