"""Briefing module for WhatsApp Leo — schedule automated AI-driven briefings."""

import sqlite3
import asyncio
import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Callable, Any
from croniter import croniter

from dotenv import load_dotenv

logger = logging.getLogger("Briefing")

load_dotenv(override=True)

_db_migrated = False

TZ = ZoneInfo("America/Los_Angeles")
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "store", "briefings.db"
)
POLL_INTERVAL = int(os.getenv("BRIEFING_POLL_INTERVAL", "60"))

# ── Persistence ──────────────────────────────────────────────────────────────


def _get_db() -> sqlite3.Connection:
    """Return a connection, running schema migration only on first call."""
    global _db_migrated
    conn = sqlite3.connect(DB_PATH)
    if not _db_migrated:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS briefings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                schedule_cron TEXT NOT NULL,
                chat_jid TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_run_at TEXT,
                next_run_at TEXT NOT NULL
            )
        """)
        conn.commit()
        # Performance indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_briefings_enabled_next_run 
            ON briefings(enabled, next_run_at)
        """)
        conn.commit()
        _db_migrated = True
    return conn


def store_briefing(
    name: str,
    prompt: str,
    schedule_cron: str,
    chat_jid: str,
    next_run_at: datetime,
) -> int:
    """Insert a new briefing. Returns its row id."""
    conn = _get_db()
    cur = conn.execute(
        "INSERT INTO briefings (name, prompt, schedule_cron, chat_jid, enabled, created_at, next_run_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            name,
            prompt,
            schedule_cron,
            chat_jid,
            1,
            datetime.now(TZ).isoformat(),
            next_run_at.isoformat(),
        ),
    )
    conn.commit()
    row_id: int | None = cur.lastrowid
    if row_id is None:
        raise RuntimeError("Failed to get row id after insert")
    return row_id


def get_due_briefings() -> list:
    """Return all enabled briefings whose next_run_at <= now."""
    conn = _get_db()
    now_iso = datetime.now(TZ).isoformat()
    rows = conn.execute(
        "SELECT id, name, prompt, schedule_cron, chat_jid, last_run_at, next_run_at FROM briefings WHERE enabled = 1 AND next_run_at <= ?",
        (now_iso,),
    ).fetchall()
    return rows


def update_briefing_schedule(
    briefing_id: int, last_run_at: datetime, next_run_at: datetime
) -> None:
    """Update the last_run_at and next_run_at for a briefing."""
    conn = _get_db()
    conn.execute(
        "UPDATE briefings SET last_run_at = ?, next_run_at = ? WHERE id = ?",
        (last_run_at.isoformat(), next_run_at.isoformat(), briefing_id),
    )
    conn.commit()


def get_all_briefings() -> list:
    """Return all briefings."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT id, name, prompt, schedule_cron, chat_jid, enabled, created_at, last_run_at, next_run_at FROM briefings ORDER BY id"
    ).fetchall()
    return rows


def delete_briefing(briefing_id: int) -> bool:
    """Delete a briefing by id. Returns True if deleted."""
    conn = _get_db()
    cur = conn.execute("DELETE FROM briefings WHERE id = ?", (briefing_id,))
    conn.commit()
    return cur.rowcount > 0


def toggle_briefing(briefing_id: int, enabled: bool) -> bool:
    """Enable/disable a briefing. Returns True if updated."""
    conn = _get_db()
    cur = conn.execute(
        "UPDATE briefings SET enabled = ? WHERE id = ?",
        (1 if enabled else 0, briefing_id),
    )
    conn.commit()
    return cur.rowcount > 0


# ── Schedule Parsing ─────────────────────────────────────────────────────────


def parse_schedule_to_cron(schedule_str: str) -> str:
    """
    Parse natural language schedule into cron expression.

    Examples:
    - "9am everyday" -> "0 9 * * *"
    - "every day at 9am" -> "0 9 * * *"
    - "every morning" -> "0 9 * * *"
    - "9am monday" -> "0 9 * * 1"
    - "weekly on friday at 5pm" -> "0 17 * * 5"
    """
    schedule_lower = schedule_str.lower().strip()

    # Time extraction
    import re

    # Extract hour and minute
    time_match = re.search(r"(\d+)(?::(\d+))?\s*(am|pm)?", schedule_lower)
    hour = 9  # default
    minute = 0

    if time_match:
        hour = int(time_match.group(1))
        if time_match.group(2):
            minute = int(time_match.group(2))
        ampm = time_match.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

    # Day of week mapping
    days = {
        "sunday": 0,
        "sun": 0,
        "monday": 1,
        "mon": 1,
        "tuesday": 2,
        "tue": 2,
        "tues": 2,
        "wednesday": 3,
        "wed": 3,
        "thursday": 4,
        "thu": 4,
        "thurs": 4,
        "friday": 5,
        "fri": 5,
        "saturday": 6,
        "sat": 6,
    }

    # Determine day of week
    day_of_week = "*"
    for day_name, day_num in days.items():
        if day_name in schedule_lower:
            day_of_week = str(day_num)
            break

    # Check for daily/everyday patterns
    if any(
        x in schedule_lower
        for x in ["everyday", "every day", "daily", "every morning", "every evening"]
    ):
        day_of_week = "*"

    # Check for weekly pattern (if specific day mentioned without "every")
    if "weekly" in schedule_lower:
        # day_of_week already set above if specific day mentioned
        pass

    return f"{minute} {hour} * * {day_of_week}"


def get_next_run_from_cron(
    cron_expr: str, base_time: datetime | None = None
) -> datetime:
    """Calculate the next run time from a cron expression."""
    if base_time is None:
        base_time = datetime.now(TZ)

    itr = croniter(cron_expr, base_time)
    next_run = itr.get_next(datetime)

    # Ensure timezone aware
    if next_run.tzinfo is None:
        next_run = next_run.replace(tzinfo=TZ)

    return next_run


# ── Scheduler ─────────────────────────────────────────────────────────────────


class BriefingScheduler:
    """Async background loop that executes due briefings."""

    def __init__(
        self,
        execute_fn: Callable[[str, str, str], Any],
        send_fn: Callable[[str, str, str | None, str | None], tuple[bool, Any]],
    ):
        """
        execute_fn: async function(prompt, chat_jid, briefing_name) -> result_text
        send_fn: function(chat_jid, message, reply_to, reply_to_sender) -> (bool, result)
        """
        self._execute_fn = execute_fn
        self._send_fn = send_fn
        self._conn: sqlite3.Connection | None = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _get_db()
        return self._conn

    async def run(self):
        """Poll every 60 seconds for due briefings and execute them."""
        logger.info("Briefing scheduler started")

        # Seed initial briefings if database is empty
        await self._seed_default_briefings()

        while True:
            try:
                conn = self._ensure_conn()
                now_iso = datetime.now(TZ).isoformat()
                rows = conn.execute(
                    "SELECT id, name, prompt, schedule_cron, chat_jid, last_run_at, next_run_at FROM briefings WHERE enabled = 1 AND next_run_at <= ?",
                    (now_iso,),
                ).fetchall()

                for (
                    briefing_id,
                    name,
                    prompt,
                    schedule_cron,
                    chat_jid,
                    last_run_at,
                    next_run_at,
                ) in rows:
                    logger.info(f"Executing briefing '{name}' for {chat_jid}")

                    try:
                        # Execute the briefing prompt through the AI
                        result_text = await self._execute_fn(prompt, chat_jid, name)

                        # Send the result
                        header = f"📋 *{name}*\n\n"
                        full_message = header + result_text

                        success, send_result = self._send_fn(
                            chat_jid,
                            full_message,
                            None,
                            None,
                        )

                        if success:
                            # Calculate next run time
                            now = datetime.now(TZ)
                            next_run = get_next_run_from_cron(schedule_cron, now)

                            conn.execute(
                                "UPDATE briefings SET last_run_at = ?, next_run_at = ? WHERE id = ?",
                                (now.isoformat(), next_run.isoformat(), briefing_id),
                            )
                            conn.commit()
                            logger.info(
                                f"Briefing '{name}' executed and next run scheduled for {next_run}"
                            )
                        else:
                            logger.error(
                                f"Failed to send briefing '{name}': {send_result}"
                            )

                    except Exception as exec_err:
                        logger.exception(
                            f"Error executing briefing '{name}': {exec_err}"
                        )

            except sqlite3.OperationalError:
                logger.warning(
                    "DB connection error in briefing scheduler, reconnecting"
                )
                self._conn = None
            except Exception:
                logger.exception("Error in briefing scheduler loop")

            await asyncio.sleep(POLL_INTERVAL)

    async def _seed_default_briefings(self):
        """Seed default briefings if none exist."""
        try:
            conn = self._ensure_conn()
            count = conn.execute("SELECT COUNT(*) FROM briefings").fetchone()[0]

            if count == 0:
                logger.info(
                    "No existing briefings found - user can create them with #briefing command"
                )
                # Note: We don't seed default briefings automatically
                # Users should create their own briefings using #briefing add

        except Exception as e:
            logger.error(f"Error checking briefings: {e}")


# ── Briefing Management Functions ───────────────────────────────────────────


def add_briefing(
    name: str,
    prompt: str,
    schedule: str,
    chat_jid: str,
) -> tuple[int, str]:
    """
    Add a new briefing.

    Args:
        name: Name of the briefing (e.g., "Morning Brief")
        prompt: The prompt to execute
        schedule: Natural language schedule (e.g., "9am everyday")
        chat_jid: JID to send briefing to

    Returns:
        (briefing_id, cron_expression)
    """
    try:
        cron_expr = parse_schedule_to_cron(schedule)
        next_run = get_next_run_from_cron(cron_expr)
        briefing_id = store_briefing(name, prompt, cron_expr, chat_jid, next_run)
        return (briefing_id, cron_expr)
    except Exception as e:
        raise ValueError(f"Failed to create briefing: {e}") from e


def list_briefings() -> list[dict]:
    """Return all briefings as list of dicts."""
    rows = get_all_briefings()
    return [
        {
            "id": row[0],
            "name": row[1],
            "prompt": row[2],
            "schedule_cron": row[3],
            "chat_jid": row[4],
            "enabled": bool(row[5]),
            "created_at": row[6],
            "last_run_at": row[7],
            "next_run_at": row[8],
        }
        for row in rows
    ]


def remove_briefing(briefing_id: int) -> bool:
    """Remove a briefing by id."""
    return delete_briefing(briefing_id)


def remove_all_briefings() -> int:
    """Remove all briefings. Returns the number deleted."""
    conn = _get_db()
    cur = conn.execute("DELETE FROM briefings")
    conn.commit()
    return cur.rowcount
