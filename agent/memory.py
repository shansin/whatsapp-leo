"""Per-user persistent memory stored as .md files."""

import os

from agents import function_tool
from logging_setup import logger

MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "store", "memory"
)


def _sanitize_phone(phone: str) -> str:
    """Make a phone number safe for use as a filename."""
    return phone.replace("+", "_").replace("@", "_").replace(".", "_")


def get_memory_path(phone: str) -> str:
    """Return the .md file path for a given phone number."""
    return os.path.join(MEMORY_DIR, f"{_sanitize_phone(phone)}.md")


def load_memory(phone: str) -> str:
    """Read the memory file for a phone number. Returns empty string if missing."""
    path = get_memory_path(phone)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to read memory file {path}: {e}")
        return ""


def make_memory_tools(phone: str) -> list:
    """Create memory tools bound to a specific phone number."""
    memory_path = get_memory_path(phone)

    @function_tool
    def save_memory(content: str) -> str:
        """Replace the entire memory file with new content. Use this when you need to reorganize, edit, or remove specific entries from the user's memory."""
        os.makedirs(MEMORY_DIR, exist_ok=True)
        with open(memory_path, "w") as f:
            f.write(content)
        logger.info(f"Memory saved for {phone} ({len(content)} chars)")
        return "Memory saved successfully."

    @function_tool
    def append_memory(entry: str) -> str:
        """Append a new entry to the user's memory file. Use this for simple 'remember that...' additions."""
        os.makedirs(MEMORY_DIR, exist_ok=True)
        with open(memory_path, "a") as f:
            f.write(f"\n- {entry}\n")
        logger.info(f"Memory appended for {phone}: {entry[:80]}")
        return "Memory entry added successfully."

    @function_tool
    def delete_memory() -> str:
        """Delete all memories for this user. Only use when the user explicitly asks to forget everything or clear all memories."""
        if os.path.exists(memory_path):
            os.remove(memory_path)
            logger.info(f"Memory deleted for {phone}")
            return "All memories cleared."
        return "No memories to clear."

    return [append_memory, save_memory, delete_memory]
