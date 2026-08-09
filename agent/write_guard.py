"""Code-enforced confirmation for write-capable tool calls.

Quoted messages, voice transcripts, web-search results and — once browsing is
enabled — the full text of any page Leo opens all flow into a privileged agent
that can create calendar events, delete them, draft mail, run JavaScript in a
logged-in browser and upload local files to a web form. The system prompt asks
for a preview before writes, but a prompt is not a control: injected text can
just as easily tell the model to skip it.

When ``REQUIRE_WRITE_CONFIRMATION`` is on, a mutating tool call is refused at
the MCP boundary unless the user's *current* message is an explicit
confirmation. The refusal is returned to the model as a tool error, so it
naturally asks the user first and succeeds on the next turn.

Off by default: it costs an extra round-trip on every write.
"""

import os
import re
from contextvars import ContextVar

from logging_setup import logger

REQUIRE_WRITE_CONFIRMATION = (
    os.getenv("REQUIRE_WRITE_CONFIRMATION", "false").lower() == "true"
)

# Tools that change state somewhere the user cares about, per MCP server.
# Read-only tools are deliberately absent — this is an allowlist of danger.
WRITE_TOOLS: dict[str, set[str]] = {
    "workspace": {
        "calendar.createEvent",
        "calendar.updateEvent",
        "calendar.deleteEvent",
        "calendar.respondToEvent",
        "gmail.createDraft",
        "gmail.downloadAttachment",
        "auth.clear",
        "docs.create",
        "docs.appendText",
        "docs.replaceText",
        "docs.insertText",
        "docs.move",
        "drive.createFolder",
        "drive.downloadFile",
    },
    # Browser tools that act rather than read. Navigating and reading
    # (browser_navigate, browser_snapshot, browser_take_screenshot,
    # browser_wait_for, browser_tabs, browser_navigate_back, browser_resize,
    # browser_console_messages, browser_network_requests, browser_hover,
    # browser_drag, browser_close) are deliberately absent: they change nothing
    # on the user's behalf, and gating them would demand a "yes" per page.
    "playwright": {
        "browser_evaluate",       # arbitrary JS in the page
        "browser_run_code",       # arbitrary Playwright code
        "browser_file_upload",    # local file -> remote site
        "browser_click",
        "browser_type",
        "browser_fill_form",
        "browser_press_key",
        "browser_select_option",
        "browser_handle_dialog",
    },
}

_CONFIRMATION = re.compile(
    r"^\s*(yes|yep|yeah|yup|confirm|confirmed|approve|approved|do it|go ahead|"
    r"send it|ok|okay|sure|please do)\b",
    re.IGNORECASE,
)

# The chat whose run is currently executing. Runs are serialized per chat and
# asyncio tasks inherit the context, so this stays correct under concurrency.
current_chat: ContextVar[str] = ContextVar("current_chat", default="")

# Chats whose current turn carries an explicit confirmation.
_confirmed: set[str] = set()


def is_confirmation(content: str) -> bool:
    """True if a message reads as the user approving a pending action."""
    return bool(_CONFIRMATION.match(content or ""))


def begin_turn(chat_jid: str, content: str) -> None:
    """Record whether this turn is allowed to perform writes."""
    if is_confirmation(content):
        _confirmed.add(chat_jid)
        logger.info(f"Write operations confirmed for this turn in {chat_jid}")
    else:
        _confirmed.discard(chat_jid)


def end_turn(chat_jid: str) -> None:
    """Confirmation applies to one turn only."""
    _confirmed.discard(chat_jid)


def is_write_tool(server_name: str, tool_name: str) -> bool:
    return tool_name in WRITE_TOOLS.get(server_name, ())


def refusal(server_name: str, tool_name: str) -> str | None:
    """Return a refusal message if this call must be confirmed first."""
    if not REQUIRE_WRITE_CONFIRMATION:
        return None
    if not is_write_tool(server_name, tool_name):
        return None

    chat_jid = current_chat.get()
    if chat_jid and chat_jid in _confirmed:
        return None

    logger.warning(f"Blocked unconfirmed write tool {server_name}.{tool_name}")
    return (
        f"BLOCKED: '{tool_name}' changes the user's data and has not been "
        "confirmed. Do not retry. Show the user exactly what you intend to do "
        "and ask them to reply 'yes' to confirm; then run it on that reply."
    )
