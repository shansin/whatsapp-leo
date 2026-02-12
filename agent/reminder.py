"""Reminder module for WhatsApp Leo — parse, validate, persist, and schedule reminders."""

import re
import sqlite3
import asyncio
import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple

logger = logging.getLogger("Reminder")

TZ = ZoneInfo("America/Los_Angeles")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "store", "reminders.db")

# ── Relative-time patterns ───────────────────────────────────────────────────
_RELATIVE_PATTERN = re.compile(
    r"in\s+(\d+)\s+(minute|minutes|min|mins|hour|hours|hr|hrs|day|days|week|weeks)",
    re.IGNORECASE,
)

_UNIT_MAP = {
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours",
    "day": "days", "days": "days",
    "week": "weeks", "weeks": "weeks",
}

# ── #remindme regex ──────────────────────────────────────────────────────────
_REMINDME_PATTERN = re.compile(r"#remindme\s+(.+)", re.IGNORECASE)


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_remindme(content: str) -> Optional[Tuple[datetime, str]]:
    """Parse a message for #remindme <time>.

    Returns (remind_at_datetime, original_message_text) or None if no #remindme found.
    Raises ValueError if the time expression cannot be parsed.
    """
    match = _REMINDME_PATTERN.search(content)
    if not match:
        return None

    time_str = match.group(1).strip()
    # The "original message" is everything except the #remindme ... portion
    original = _REMINDME_PATTERN.sub("", content).strip() or time_str

    now = datetime.now(TZ)

    # Try relative time first ("in 30 minutes", "in 2 hours", etc.)
    rel = _RELATIVE_PATTERN.match(time_str)
    if rel:
        amount = int(rel.group(1))
        unit = _UNIT_MAP[rel.group(2).lower()]
        delta = timedelta(**{unit: amount})
        return (now + delta, original)

    # Fall back to dateutil for absolute times
    from dateutil import parser as dateutil_parser

    try:
        parsed = dateutil_parser.parse(time_str, fuzzy=True)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Could not understand the time: {time_str}") from exc

    # If no timezone was provided, assume our local TZ
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ)

    return (parsed, original)


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
