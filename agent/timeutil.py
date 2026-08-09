"""Storage format for scheduled times.

Schedulers select due rows with a string comparison (``next_run_at <= now``).
Local ISO timestamps carry a UTC offset that changes across DST, so
``2026-11-01T01:30:00-08:00`` sorts *before* ``2026-11-01T01:15:00-07:00``
even though it happens later — reminders around a DST switch fired in the
wrong order or not at all.

Everything scheduled is therefore stored as normalized UTC ISO. Those strings
all share one offset and a fixed width, so lexicographic order is chronological
order.
"""

from datetime import datetime, UTC
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as dateutil_parser

from config import TZ

_UTC_SUFFIX = "+00:00"


def resolve_tz(tz_name: str | None):
    """Return the named timezone, falling back to the instance default.

    Used for rows that recorded the timezone their schedule was written in.
    """
    if not tz_name:
        return TZ
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        return TZ


def to_db(dt: datetime) -> str:
    """Serialize a datetime for storage in a scheduling column."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(UTC).isoformat()


def now_db() -> str:
    """Current time in the storage format."""
    return datetime.now(UTC).isoformat()


def from_db(value: str) -> datetime:
    """Parse a stored timestamp back into local time, for display.

    Tolerates rows written before normalization (local offsets, or none).
    """
    dt = dateutil_parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ)


def normalize_columns(conn, table: str, columns: list[str]) -> int:
    """Rewrite any non-UTC values in ``columns`` in place. Idempotent.

    Runs at startup so databases written before normalization keep working.
    """
    quoted = ", ".join(columns)
    rows = conn.execute(f"SELECT id, {quoted} FROM {table}").fetchall()

    updates = []
    for row in rows:
        row_id, values = row[0], row[1:]
        fixed = []
        changed = False
        for value in values:
            if value and not str(value).endswith(_UTC_SUFFIX):
                try:
                    fixed.append(to_db(dateutil_parser.parse(value)))
                    changed = True
                    continue
                except (ValueError, OverflowError):
                    pass  # unparseable: leave it alone
            fixed.append(value)
        if changed:
            updates.append((*fixed, row_id))

    if updates:
        assignments = ", ".join(f"{col} = ?" for col in columns)
        conn.executemany(
            f"UPDATE {table} SET {assignments} WHERE id = ?", updates
        )
        conn.commit()
    return len(updates)
