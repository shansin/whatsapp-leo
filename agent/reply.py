"""Outbound reply shaping: splitting, and optional voice replies.

A long model answer used to go out as one wall of text. WhatsApp will carry it,
but it reads badly on a phone, so replies are split at natural boundaries —
paragraphs first, then sentences, then words, and never mid-code-fence.
"""

import asyncio
import os
import re
import shlex
import tempfile

from config import STORE_DIR
from logging_setup import logger
from models import ReceivedMessage
from whatsapp import send_message as whatsapp_send_message

# Practical readability limit, not WhatsApp's hard cap.
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "3500"))

# Optional TTS for voice replies, e.g.
#   TTS_COMMAND="piper --model en_US-amy-medium.onnx --output_file {out}"
# The text is written to stdin and the command must produce audio at {out}.
TTS_COMMAND = os.getenv("TTS_COMMAND", "")
VOICE_REPLIES = os.getenv("VOICE_REPLIES", "false").lower() == "true"

_CODE_FENCE = re.compile(r"```")


def _split_chunk(text: str, limit: int) -> tuple[str, str]:
    """Split ``text`` once at the best boundary at or before ``limit``."""
    window = text[:limit]

    for separator in ("\n\n", "\n", ". ", " "):
        cut = window.rfind(separator)
        if cut > limit // 4:  # avoid a pathologically short piece
            keep = cut + (len(separator) if separator == ". " else 0)
            return text[:keep].rstrip(), text[keep:].lstrip()

    return window, text[limit:]


def _balance_fences(parts: list[str]) -> list[str]:
    """Close and reopen code fences so no part renders as broken markup.

    A code block longer than the limit has to be cut somewhere; cutting it
    without closing the fence turns the rest of the conversation into code.
    """
    balanced = []
    inside = False
    for part in parts:
        if inside:
            part = "```\n" + part
        if len(_CODE_FENCE.findall(part)) % 2 == 1:
            part = part.rstrip() + "\n```"
            inside = True
        else:
            inside = False
        balanced.append(part)
    return balanced


def split_reply(text: str, limit: int | None = None) -> list[str]:
    """Split a reply into WhatsApp-friendly pieces, preserving structure."""
    limit = limit or MAX_REPLY_CHARS
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts = []
    remaining = text
    while len(remaining) > limit:
        head, remaining = _split_chunk(remaining, limit)
        if not head:  # no progress possible; hard-cut to stay terminating
            head, remaining = remaining[:limit], remaining[limit:]
        parts.append(head)
    if remaining:
        parts.append(remaining)
    return _balance_fences(parts)


async def reply_to(message: ReceivedMessage, text: str) -> None:
    """Send a WhatsApp reply quoting the originating message."""
    await asyncio.to_thread(
        whatsapp_send_message,
        message.chat_jid,
        text,
        reply_to=message.id,
        reply_to_sender=message.sender_jid,
    )


async def send_reply(send_fn, chat_jid: str, text: str) -> bool:
    """Send a reply, split across messages if needed. True if all parts sent."""
    parts = split_reply(text)
    if not parts:
        return False

    all_ok = True
    for index, part in enumerate(parts):
        ok, detail = await asyncio.to_thread(send_fn, chat_jid, part)
        if ok:
            continue
        all_ok = False
        logger.error(
            f"Failed to send reply part {index + 1}/{len(parts)} to {chat_jid}: {detail}"
        )
        break  # a truncated reply beats an out-of-order one
    return all_ok


# ── Voice replies ────────────────────────────────────────────────────────────


def _synthesize(text: str) -> str | None:
    """Run the configured TTS command. Returns an audio path, or None."""
    if not TTS_COMMAND:
        return None

    out_dir = os.path.join(STORE_DIR, "tts")
    os.makedirs(out_dir, exist_ok=True)
    fd, out_path = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)

    command = [
        arg.format(out=out_path) for arg in shlex.split(TTS_COMMAND)
    ]
    try:
        import subprocess

        result = subprocess.run(
            command,
            input=text.encode(),
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning(
                f"TTS command failed ({result.returncode}): "
                f"{result.stderr.decode(errors='replace')[:200]}"
            )
            os.unlink(out_path)
            return None
        if not os.path.getsize(out_path):
            logger.warning("TTS command produced an empty file")
            os.unlink(out_path)
            return None
        return out_path
    except Exception as e:
        logger.warning(f"TTS failed: {e}")
        if os.path.exists(out_path):
            os.unlink(out_path)
        return None


async def send_voice_reply(send_audio_fn, chat_jid: str, text: str) -> bool:
    """Speak a reply as a voice note. False if TTS is unavailable or fails.

    Callers must fall back to text — this is best-effort by design.
    """
    if not (VOICE_REPLIES and TTS_COMMAND):
        return False

    audio_path = await asyncio.to_thread(_synthesize, text)
    if not audio_path:
        return False

    try:
        ok, detail = await asyncio.to_thread(send_audio_fn, chat_jid, audio_path)
        if not ok:
            logger.warning(f"Could not send voice reply to {chat_jid}: {detail}")
        return ok
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
