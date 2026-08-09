"""Read-only chat-history tools.

The bridge already persists every message in messages.db and whatsapp.py has
rich query helpers, but the agent was never given any of them — so Leo could
search the web but not the conversation it was sitting in. These expose that
history as tools: "what did Sam say about the trip?", "summarise this group
today".

Read-only by construction, and only attached for privileged senders — this is
the user's entire message archive.
"""

import asyncio
from datetime import datetime, timedelta

from agents import function_tool

import user_prefs
from logging_setup import logger
from whatsapp import list_chats, list_messages, search_contacts

# Keep results well inside the local model's context window.
MAX_RESULTS = 30
SNIPPET_CHARS = 400


def _format(messages, tz) -> str:
    """Render messages as compact, model-friendly lines."""
    if not messages:
        return "No matching messages."

    lines = []
    for msg in messages:
        when = msg.timestamp
        if isinstance(when, datetime):
            when = when.astimezone(tz).strftime("%b %d %H:%M")
        who = "Me" if msg.is_from_me else (msg.sender or "unknown")
        body = (msg.content or "").replace("\n", " ").strip()
        if msg.media_type and not body:
            body = f"[{msg.media_type}]"
        if len(body) > SNIPPET_CHARS:
            body = body[:SNIPPET_CHARS] + "…"
        chat = f" ({msg.chat_name})" if msg.chat_name else ""
        lines.append(f"[{when}]{chat} {who}: {body}")
    return "\n".join(lines)


def make_history_tools(chat_jid: str, phone: str) -> list:
    """Create history tools bound to the current chat and user."""
    tz = user_prefs.get_tz(phone)

    def _clamp(limit: int) -> int:
        return max(1, min(int(limit or 20), MAX_RESULTS))

    @function_tool
    async def search_chat_history(
        query: str, days: int = 30, limit: int = 20, this_chat_only: bool = True
    ) -> str:
        """Search past WhatsApp messages for text.

        Args:
            query: words to look for in message content
            days: how far back to search (default 30)
            limit: maximum messages to return (max 30)
            this_chat_only: restrict to the current conversation
        """
        after = (datetime.now(tz) - timedelta(days=max(1, days))).isoformat()
        try:
            messages = await asyncio.to_thread(
                list_messages,
                after=after,
                chat_jid=chat_jid if this_chat_only else None,
                query=query,
                limit=_clamp(limit),
                include_context=False,
            )
        except Exception as e:
            logger.error(f"search_chat_history failed: {e}", exc_info=True)
            return f"Could not search history: {e}"
        return _format(messages, tz)

    @function_tool
    async def recent_chat_messages(limit: int = 20, this_chat_only: bool = True) -> str:
        """Read the most recent messages, newest last.

        Use this to summarise what has been said recently.

        Args:
            limit: maximum messages to return (max 30)
            this_chat_only: restrict to the current conversation
        """
        try:
            messages = await asyncio.to_thread(
                list_messages,
                chat_jid=chat_jid if this_chat_only else None,
                limit=_clamp(limit),
                include_context=False,
            )
        except Exception as e:
            logger.error(f"recent_chat_messages failed: {e}", exc_info=True)
            return f"Could not read history: {e}"
        return _format(messages, tz)

    @function_tool
    async def messages_from_person(
        name_or_number: str, days: int = 30, limit: int = 20
    ) -> str:
        """Find what a specific person said, across all chats.

        Args:
            name_or_number: contact name or phone number
            days: how far back to search (default 30)
            limit: maximum messages to return (max 30)
        """
        try:
            contacts = await asyncio.to_thread(search_contacts, name_or_number)
        except Exception as e:
            logger.error(f"messages_from_person lookup failed: {e}", exc_info=True)
            return f"Could not look up that contact: {e}"

        if not contacts:
            return f"No contact matching '{name_or_number}'."

        after = (datetime.now(tz) - timedelta(days=max(1, days))).isoformat()
        collected = []
        for contact in contacts[:3]:  # a name can match a few numbers
            try:
                collected += await asyncio.to_thread(
                    list_messages,
                    after=after,
                    sender_phone_number=contact.phone_number,
                    limit=_clamp(limit),
                    include_context=False,
                )
            except Exception as e:
                logger.warning(f"messages_from_person query failed: {e}")

        collected.sort(key=lambda m: m.timestamp)
        return _format(collected[-_clamp(limit):], tz)

    @function_tool
    async def find_chats(query: str = "", limit: int = 20) -> str:
        """List WhatsApp chats, optionally filtered by name.

        Args:
            query: text to match against chat names
            limit: maximum chats to return (max 30)
        """
        try:
            chats = await asyncio.to_thread(
                list_chats, query=query or None, limit=_clamp(limit)
            )
        except Exception as e:
            logger.error(f"find_chats failed: {e}", exc_info=True)
            return f"Could not list chats: {e}"

        if not chats:
            return "No matching chats."
        lines = []
        for chat in chats:
            kind = "group" if chat.is_group else "direct"
            last = ""
            if chat.last_message_time:
                last = f" — last active {chat.last_message_time.astimezone(tz):%b %d %H:%M}"
            lines.append(f"{chat.name or chat.jid} ({kind}){last}")
        return "\n".join(lines)

    return [
        search_chat_history,
        recent_chat_messages,
        messages_from_person,
        find_chats,
    ]
