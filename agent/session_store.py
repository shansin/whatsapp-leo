"""Durable, bounded conversation sessions.

``SQLiteSession`` defaults to ``db_path=":memory:"``, so history died on every
process restart *and* whenever the agent cache evicted an entry — silent
context loss mid-conversation. Nothing trimmed it either, so a long-running
chat grew until it overflowed the local model's context window, which degrades
quality invisibly.

Sessions here live in ``store/sessions.db``, are keyed by chat only (so they
outlive the per-(chat, model) agent cache and are shared by the text and vision
agents), and are trimmed to a rolling window after each run.
"""

import asyncio
import json
import os
import sqlite3

from agents import SQLiteSession

from logging_setup import logger

SESSIONS_DB_PATH = os.getenv("SESSIONS_DB_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "store", "sessions.db"
)

# Rolling window of conversation items kept per chat. 0 disables trimming.
MAX_SESSION_ITEMS = int(os.getenv("MAX_SESSION_ITEMS", "40"))


def _is_turn_boundary(item) -> bool:
    """True if history can safely start at ``item``.

    Cutting mid-turn can strip a tool call away from its result (or vice
    versa), which most chat APIs reject. A user message is always a safe start.
    """
    return isinstance(item, dict) and item.get("role") == "user"


class TrimmedSQLiteSession(SQLiteSession):
    """SQLiteSession that can drop its oldest items.

    ``pop_item()`` removes the *newest* item, so it cannot implement a rolling
    window; this deletes from the front instead.
    """

    async def trim(self, max_items: int) -> int:
        """Keep roughly the newest ``max_items`` items. Returns rows deleted."""
        if max_items <= 0:
            return 0
        return await asyncio.to_thread(self._trim_sync, max_items)

    def _trim_sync(self, max_items: int) -> int:
        conn = self._get_connection()
        rows = conn.execute(
            f"""
            SELECT id, message_data FROM {self.messages_table}
            WHERE session_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (self.session_id,),
        ).fetchall()

        if len(rows) <= max_items:
            return 0

        cutoff = len(rows) - max_items
        # Walk forward to the next clean turn boundary so we never orphan a
        # tool result. If there is none, keep everything rather than corrupt it.
        while cutoff < len(rows):
            try:
                item = json.loads(rows[cutoff][1])
            except json.JSONDecodeError:
                break  # unparseable row: dropping it is fine
            if _is_turn_boundary(item):
                break
            cutoff += 1

        if cutoff >= len(rows):
            logger.debug(
                f"Session {self.session_id}: no safe trim boundary, keeping all items"
            )
            return 0

        doomed = [row[0] for row in rows[:cutoff]]
        placeholders = ",".join("?" * len(doomed))
        conn.execute(
            f"DELETE FROM {self.messages_table} WHERE id IN ({placeholders})",
            doomed,
        )
        conn.commit()
        return len(doomed)


_sessions: dict[str, TrimmedSQLiteSession] = {}


def get_session(chat_jid: str) -> TrimmedSQLiteSession:
    """Return the persistent session for a chat, creating it on first use."""
    session = _sessions.get(chat_jid)
    if session is None:
        os.makedirs(os.path.dirname(os.path.abspath(SESSIONS_DB_PATH)), exist_ok=True)
        session = TrimmedSQLiteSession(chat_jid, db_path=SESSIONS_DB_PATH)
        _sessions[chat_jid] = session
    return session


async def trim_session(session: TrimmedSQLiteSession, chat_jid: str = "") -> None:
    """Apply the rolling window; never let a trim failure break a reply."""
    try:
        removed = await session.trim(MAX_SESSION_ITEMS)
        if removed:
            logger.info(
                f"Trimmed {removed} old item(s) from session "
                f"{chat_jid or session.session_id}"
            )
    except sqlite3.Error as e:
        logger.warning(f"Could not trim session {session.session_id}: {e}")
