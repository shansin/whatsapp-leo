"""Per-chat message debouncing.

People send a thought as three or four rapid messages. Answering each one
separately wastes model runs and produces replies that address a fragment
instead of the whole point. This holds a short window open per chat and merges
whatever arrives into a single prompt.

Only the last message in a burst proceeds; the earlier ones return None and
their tasks exit quietly.
"""

import asyncio
import os

from logging_setup import logger

DEBOUNCE_SECONDS = float(os.getenv("DEBOUNCE_SECONDS", "2.5"))
# A burst is a burst, not a filibuster.
MAX_BURST_MESSAGES = int(os.getenv("MAX_BURST_MESSAGES", "10"))


class Debouncer:
    """Collects rapid consecutive messages per chat into one turn."""

    def __init__(self, window: float | None = None):
        self._window = DEBOUNCE_SECONDS if window is None else window
        self._pending: dict[str, list] = {}

    async def collect(self, chat_jid: str, message) -> list | None:
        """Wait out the burst window.

        Returns every message in the burst (oldest first) for the last caller,
        and None for callers that a later message superseded.
        """
        if self._window <= 0:
            return [message]

        burst = self._pending.setdefault(chat_jid, [])
        burst.append(message)

        # A long burst is flushed immediately rather than extending forever.
        if len(burst) >= MAX_BURST_MESSAGES:
            return self._flush(chat_jid)

        await asyncio.sleep(self._window)

        burst = self._pending.get(chat_jid) or []
        if not burst or burst[-1] is not message:
            # Someone else arrived after us; they will send the merged turn.
            return None
        return self._flush(chat_jid)

    def _flush(self, chat_jid: str) -> list:
        burst = self._pending.pop(chat_jid, [])
        if len(burst) > 1:
            logger.info(f"Merged {len(burst)} rapid messages in {chat_jid}")
        return burst


def merge_content(messages: list) -> str:
    """Join a burst's text into one prompt, dropping empties and duplicates."""
    seen = []
    for message in messages:
        text = (message.content or "").strip()
        if text and text not in seen:
            seen.append(text)
    return "\n".join(seen)


# Process-wide instance.
debouncer = Debouncer()
