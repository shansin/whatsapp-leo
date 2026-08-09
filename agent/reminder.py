"""Reminder module for WhatsApp Leo — validate, persist, and schedule reminders."""

import sqlite3
import asyncio
import logging
import os
from datetime import datetime
from typing import Any
from collections.abc import Callable

from croniter import croniter

from config import TZ
from sqlite_store import PollingScheduler, SqliteStore
from timeutil import normalize_columns, now_db, resolve_tz, to_db

logger = logging.getLogger("Reminder")

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "store", "reminders.db"
)
POLL_INTERVAL = int(os.getenv("REMINDER_POLL_INTERVAL", "60"))


# ── Validation ───────────────────────────────────────────────────────────────


def validate_reminder_time(dt: datetime) -> None:
    """Raise ValueError if the reminder time is in the past."""
    now = datetime.now(TZ)
    if dt <= now:
        raise ValueError("The reminder time is in the past.")


# ── One-Shot Reminder Persistence ────────────────────────────────────────────


def _migrate_reminders(conn: sqlite3.Connection) -> None:
    """Create the one-shot reminder schema and bring old rows up to date."""
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
    for col in ["message_id TEXT", "sender_jid TEXT"]:
        try:
            conn.execute(f"ALTER TABLE reminders ADD COLUMN {col}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_reminders_fired_remind_at
        ON reminders(fired, remind_at)
    """)
    conn.commit()
    # Existing rows may hold local-offset timestamps; see timeutil.
    normalize_columns(conn, "reminders", ["remind_at"])


_store = SqliteStore(lambda: DB_PATH, _migrate_reminders)


def _get_db() -> sqlite3.Connection:
    return _store.connect()


def store_reminder(
    chat_jid: str,
    message: str,
    remind_at: datetime,
    message_id: str = "",
    sender_jid: str = "",
) -> int:
    """Insert a new reminder. Returns its row id."""
    conn = _get_db()
    cur = conn.execute(
        "INSERT INTO reminders (chat_jid, message, remind_at, created_at, message_id, sender_jid) VALUES (?, ?, ?, ?, ?, ?)",
        (
            chat_jid,
            message,
            to_db(remind_at),
            to_db(datetime.now(TZ)),
            message_id,
            sender_jid,
        ),
    )
    conn.commit()
    row_id: int | None = cur.lastrowid
    if row_id is None:
        raise RuntimeError("Failed to get row id after insert")
    return row_id


def get_due_reminders() -> list:
    """Return all unfired reminders whose remind_at <= now."""
    conn = _get_db()
    now_iso = now_db()
    rows = conn.execute(
        "SELECT id, chat_jid, message, remind_at, message_id, sender_jid FROM reminders WHERE fired = 0 AND remind_at <= ?",
        (now_iso,),
    ).fetchall()
    return rows


def get_pending_reminders(chat_jid: str) -> list:
    """Return unfired reminders for a chat, soonest first."""
    conn = _get_db()
    return conn.execute(
        "SELECT id, message, remind_at FROM reminders "
        "WHERE fired = 0 AND chat_jid = ? ORDER BY remind_at",
        (chat_jid,),
    ).fetchall()


def cancel_reminder(reminder_id: int, chat_jid: str) -> bool:
    """Delete an unfired reminder. Returns True if one was removed."""
    conn = _get_db()
    cur = conn.execute(
        "DELETE FROM reminders WHERE id = ? AND chat_jid = ? AND fired = 0",
        (reminder_id, chat_jid),
    )
    conn.commit()
    return cur.rowcount > 0


def mark_fired(reminder_id: int) -> None:
    conn = _get_db()
    conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
    conn.commit()


# ── One-Shot Reminder Scheduler ──────────────────────────────────────────────


class ReminderScheduler(PollingScheduler):
    """Fires one-shot reminders whose time has come."""

    name = "Reminder scheduler"

    def __init__(self, send_fn):
        """send_fn(chat_jid, message, reply_to, reply_to_sender) -> (bool, Any)."""
        super().__init__(_store, POLL_INTERVAL)
        self._send_fn = send_fn

    @staticmethod
    def _mark_fired(conn: sqlite3.Connection, reminder_id: int) -> None:
        conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
        conn.commit()

    async def poll(self) -> None:
        conn = self.conn()
        now_iso = now_db()
        rows = await asyncio.to_thread(
            lambda: conn.execute(
                "SELECT id, chat_jid, message, remind_at, message_id, sender_jid "
                "FROM reminders WHERE fired = 0 AND remind_at <= ?",
                (now_iso,),
            ).fetchall()
        )
        for rid, chat_jid, message, _remind_at, message_id, sender_jid in rows:
            text = f"⏰ *Reminder*\n\n{message}"
            # send_fn is HTTP-over-unix-socket with a 30s timeout; calling it
            # inline froze all message processing for its duration.
            success, result = await asyncio.to_thread(
                self._send_fn,
                chat_jid,
                text,
                reply_to=message_id or None,
                reply_to_sender=sender_jid or None,
            )
            if success:
                await asyncio.to_thread(self._mark_fired, conn, rid)
                logger.info(f"Fired reminder {rid} to {chat_jid}")
            else:
                logger.error(f"Failed to fire reminder {rid}: {result}")


# ── Recurring Reminder Persistence ───────────────────────────────────────────


def _migrate_recurring(conn: sqlite3.Connection) -> None:
    """Create the recurring reminder schema and bring old rows up to date."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recurring_reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            schedule_cron TEXT NOT NULL,
            chat_jid TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            last_run_at TEXT,
            next_run_at TEXT NOT NULL,
            tz TEXT
        )
    """)
    conn.commit()
    # Older databases predate the tz column.
    try:
        conn.execute("ALTER TABLE recurring_reminders ADD COLUMN tz TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_recurring_reminders_enabled_next_run
        ON recurring_reminders(enabled, next_run_at)
    """)
    conn.commit()
    normalize_columns(
        conn, "recurring_reminders", ["next_run_at", "last_run_at"]
    )


_recurring_store = SqliteStore(lambda: DB_PATH, _migrate_recurring)


def _get_recurring_db() -> sqlite3.Connection:
    return _recurring_store.connect()


def store_recurring_reminder(
    message: str,
    schedule_cron: str,
    chat_jid: str,
    next_run_at: datetime,
    tz_name: str = "",
) -> int:
    """Insert a new recurring reminder. Returns its row id.

    ``tz_name`` is the creator's timezone: "9pm every day" has to keep meaning
    9pm where they are, not on the server.
    """
    conn = _get_recurring_db()
    cur = conn.execute(
        "INSERT INTO recurring_reminders (message, schedule_cron, chat_jid, enabled, created_at, next_run_at, tz) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            message,
            schedule_cron,
            chat_jid,
            1,
            to_db(datetime.now(TZ)),
            to_db(next_run_at),
            tz_name or str(TZ),
        ),
    )
    conn.commit()
    row_id: int | None = cur.lastrowid
    if row_id is None:
        raise RuntimeError("Failed to get row id after insert")
    return row_id


def get_all_recurring_reminders() -> list:
    """Return all recurring reminders."""
    conn = _get_recurring_db()
    rows = conn.execute(
        "SELECT id, message, schedule_cron, chat_jid, enabled, created_at, last_run_at, next_run_at, tz FROM recurring_reminders ORDER BY id"
    ).fetchall()
    return rows


def delete_recurring_reminder(reminder_id: int) -> bool:
    """Delete a recurring reminder by id. Returns True if deleted."""
    conn = _get_recurring_db()
    cur = conn.execute("DELETE FROM recurring_reminders WHERE id = ?", (reminder_id,))
    conn.commit()
    return cur.rowcount > 0


def delete_all_recurring_reminders() -> int:
    """Remove all recurring reminders. Returns the number deleted."""
    conn = _get_recurring_db()
    cur = conn.execute("DELETE FROM recurring_reminders")
    conn.commit()
    return cur.rowcount


# ── Recurring Reminder Scheduler ─────────────────────────────────────────────


class RecurringReminderScheduler(PollingScheduler):
    """Fires recurring reminders and advances them to their next occurrence."""

    name = "Recurring reminder scheduler"

    def __init__(
        self,
        send_fn: Callable[[str, str, str | None, str | None], tuple[bool, Any]],
    ):
        """send_fn(chat_jid, message, reply_to, reply_to_sender) -> (bool, Any)."""
        super().__init__(_recurring_store, POLL_INTERVAL)
        self._send_fn = send_fn

    @staticmethod
    def _record_run(
        conn: sqlite3.Connection, rid: int, now: datetime, next_run: datetime
    ) -> None:
        conn.execute(
            "UPDATE recurring_reminders SET last_run_at = ?, next_run_at = ? WHERE id = ?",
            (to_db(now), to_db(next_run), rid),
        )
        conn.commit()

    async def poll(self) -> None:
        conn = self.conn()
        now_iso = now_db()
        rows = await asyncio.to_thread(
            lambda: conn.execute(
                "SELECT id, message, schedule_cron, chat_jid, tz FROM recurring_reminders "
                "WHERE enabled = 1 AND next_run_at <= ?",
                (now_iso,),
            ).fetchall()
        )
        for rid, message, schedule_cron, chat_jid, tz_name in rows:
            text = f"⏰ *Reminder*\n\n{message}"
            # send_fn blocks on a socket with a 30s timeout — keep it off the
            # event loop or every message stalls behind it.
            success, result = await asyncio.to_thread(
                self._send_fn, chat_jid, text, None, None
            )
            if not success:
                logger.error(f"Failed to fire recurring reminder {rid}: {result}")
                continue

            # Advance the schedule in the timezone it was written in.
            tz = resolve_tz(tz_name)
            now = datetime.now(tz)
            next_run = croniter(schedule_cron, now).get_next(datetime)
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=tz)
            await asyncio.to_thread(self._record_run, conn, rid, now, next_run)
            logger.info(
                f"Fired recurring reminder {rid} to {chat_jid}, next run: {next_run}"
            )
