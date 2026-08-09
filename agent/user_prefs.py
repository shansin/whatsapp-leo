"""Per-user preferences, keyed by phone number.

Timezone was hardcoded to America/Los_Angeles (and the prompt said "PST"
outright), so reminders and briefings fired at the wrong local time for anyone
else, and Leo told them the wrong time of day.

Stored as one small JSON file rather than another SQLite table — it is read on
every message and written approximately never.
"""

import json
import os
import threading
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config import STORE_DIR, TZ
from logging_setup import logger

PREFS_PATH = os.getenv("USER_PREFS_PATH") or os.path.join(STORE_DIR, "user_prefs.json")

_lock = threading.Lock()
_cache: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(PREFS_PATH) as fh:
            loaded = json.load(fh)
        _cache = loaded if isinstance(loaded, dict) else {}
    except FileNotFoundError:
        _cache = {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read {PREFS_PATH}: {e}")
        _cache = {}
    return _cache


def _save(prefs: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(PREFS_PATH)), exist_ok=True)
    tmp = f"{PREFS_PATH}.tmp"
    with open(tmp, "w") as fh:
        json.dump(prefs, fh, indent=2, sort_keys=True)
    os.replace(tmp, PREFS_PATH)  # atomic
    os.chmod(PREFS_PATH, 0o600)


def get(phone: str, key: str, default=None):
    """Read one preference for a user."""
    return _load().get(phone, {}).get(key, default)


def set(phone: str, key: str, value) -> None:
    """Write one preference for a user."""
    with _lock:
        prefs = _load()
        prefs.setdefault(phone, {})[key] = value
        _save(prefs)


def get_tz(phone: str) -> ZoneInfo:
    """Return the user's timezone, falling back to the instance default."""
    name = get(phone, "tz")
    if not name:
        return TZ
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(f"Invalid stored timezone {name!r} for {phone}; using default")
        return TZ


def set_tz(phone: str, tz_name: str) -> ZoneInfo:
    """Validate and store a user's timezone. Raises ValueError if unknown."""
    try:
        zone = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Unknown timezone: {tz_name}. Use an IANA name like Europe/London."
        ) from exc
    set(phone, "tz", tz_name)
    return zone
