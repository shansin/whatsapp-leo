"""Reminder module for WhatsApp Leo — validate, persist, and schedule reminders."""

import sqlite3
import asyncio
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("Reminder")

TZ = ZoneInfo("America/Los_Angeles")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "store", "reminders.db")


# ── Validation ───────────────────────────────────────────────────────────────

def validate_reminder_time(dt: datetime) -> None:
    """Raise ValueError if the reminder time is in the past."""
    now = datetime.now(TZ)
    if dt <= now:
        raise ValueError("The reminder time is in the past.")


# ── Persistence ──────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_jid TEXT NOT NULL,
            message TEXT NOT NULL,
            remind_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            fired INTEGER NOT NULL DEFAULT 0,
            message_id TEXT,
            sender_jid TEXT
        )
    """)
    conn.commit()
    # Migrate: add columns if missing (for DBs created before these columns existed)
    for col in ["message_id TEXT", "sender_jid TEXT"]:
        try:
            conn.execute(f"ALTER TABLE reminders ADD COLUMN {col}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    return conn


def store_reminder(chat_jid: str, message: str, remind_at: datetime, message_id: str = "", sender_jid: str = "") -> int:
    """Insert a new reminder. Returns its row id."""
    conn = _get_db()
    cur = conn.execute(
        "INSERT INTO reminders (chat_jid, message, remind_at, created_at, message_id, sender_jid) VALUES (?, ?, ?, ?, ?, ?)",
        (chat_jid, message, remind_at.isoformat(), datetime.now(TZ).isoformat(), message_id, sender_jid),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_due_reminders() -> list:
    """Return all unfired reminders whose remind_at <= now."""
    conn = _get_db()
    now_iso = datetime.now(TZ).isoformat()
    rows = conn.execute(
        "SELECT id, chat_jid, message, remind_at, message_id, sender_jid FROM reminders WHERE fired = 0 AND remind_at <= ?",
        (now_iso,),
    ).fetchall()
    conn.close()
    return rows


def mark_fired(reminder_id: int) -> None:
    conn = _get_db()
    conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()


# ── Scheduler ────────────────────────────────────────────────────────────────

class ReminderScheduler:
    """Async background loop that fires due reminders."""

    def __init__(self, send_fn):
        """send_fn must be a callable(chat_jid, message, reply_to, reply_to_sender) -> (bool, Any)."""
        self._send_fn = send_fn

    async def run(self):
        """Poll every 60 seconds for due reminders and fire them."""
        logger.info("Reminder scheduler started")
        while True:
            try:
                due = get_due_reminders()
                for rid, chat_jid, message, remind_at, message_id, sender_jid in due:
                    text = f"⏰ *Reminder*\n\n{message}"
                    success, result = self._send_fn(
                        chat_jid, text,
                        reply_to=message_id or None,
                        reply_to_sender=sender_jid or None,
                    )
                    if success:
                        mark_fired(rid)
                        logger.info(f"Fired reminder {rid} to {chat_jid}")
                    else:
                        logger.error(f"Failed to fire reminder {rid}: {result}")
            except Exception:
                logger.exception("Error in reminder scheduler loop")
            await asyncio.sleep(60)
