"""Instance-wide model override, persisted so a `#model` switch survives restart.

Deliberately separate from user_prefs: this is one setting for the whole
instance (chats, briefings, reminder parsing all run on it), not a per-phone
preference. Same JSON-file shape though — read once at import, written
approximately never.

    {"text": "qwen3:14b", "vision": "gemma3:27b"}
"""

from __future__ import annotations

import json
import os
import threading

from config import STORE_DIR
from logging_setup import logger

OVERRIDE_PATH = os.getenv("MODEL_OVERRIDE_PATH") or os.path.join(
    STORE_DIR, "model_override.json"
)

_lock = threading.Lock()


def load() -> dict:
    """Read the stored override. Never raises — a bad file just means no override."""
    try:
        with open(OVERRIDE_PATH) as fh:
            loaded = json.load(fh)
        return loaded if isinstance(loaded, dict) else {}
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read {OVERRIDE_PATH}: {e}")
        return {}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(OVERRIDE_PATH)), exist_ok=True)
    tmp = f"{OVERRIDE_PATH}.tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, OVERRIDE_PATH)  # atomic
    os.chmod(OVERRIDE_PATH, 0o600)


def _set(key: str, name: str) -> None:
    with _lock:
        data = load()
        data[key] = name
        _save(data)


def set_text(name: str) -> None:
    """Persist the main text model override."""
    _set("text", name)


def set_vision(name: str) -> None:
    """Persist the vision model override."""
    _set("vision", name)


def clear() -> None:
    """Drop the override so the next start uses the .env defaults."""
    with _lock:
        try:
            os.remove(OVERRIDE_PATH)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Could not remove {OVERRIDE_PATH}: {e}")
