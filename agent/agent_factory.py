"""Agent factory with LRU caching and TTL, plus reminder parsing agent."""

import asyncio
import time
from collections import OrderedDict
from datetime import datetime

from dateutil import parser as dateutil_parser
from agents import Agent, Runner
from agents.mcp import MCPServer

from config import MAX_AGENTS, TTL_SECONDS, TZ, _cached_model, _model_settings
from instructions import REMINDER_INSTRUCTIONS_TEMPLATE
from models import ReminderParsed
from session_store import TrimmedSQLiteSession, get_session
from logging_setup import logger


class AgentFactory:
    """Factory for creating and caching Agent instances with LRU eviction and TTL."""

    def __init__(self):
        # OrderedDict keyed by (chat_jid, model_name) to maintain LRU order
        self._agents: OrderedDict[tuple[str, str], tuple[Agent, float]] = OrderedDict()
        # One lock per chat. Runs in the same chat must not interleave: they
        # share one Agent object (whose mcp_servers/tools/instructions are
        # rebound per message) and one session, and concurrent Runner.run calls
        # against a single session interleave its history.
        self._locks: dict[str, asyncio.Lock] = {}

    def lock_for(self, chat_jid: str) -> asyncio.Lock:
        """Return the serialization lock for a chat."""
        lock = self._locks.get(chat_jid)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_jid] = lock
        return lock

    def clear(self) -> None:
        """Drop every cached agent, e.g. after a `#model` switch.

        Locks are deliberately kept: they are cheap, and dropping one while a
        run holds it would let the next message in that chat interleave with it.
        Sessions live in session_store keyed by chat, so history is unaffected.
        """
        count = len(self._agents)
        self._agents.clear()
        logger.info(f"Cleared agent cache ({count} entries)")

    def _is_expired(self, last_used: float) -> bool:
        """Check if an entry has exceeded the TTL."""
        return (time.time() - last_used) > TTL_SECONDS

    async def get_agent(
        self, chat_jid: str, mcp_servers: list[MCPServer], model, instructions: str,
        tools: list | None = None,
    ) -> tuple[Agent, TrimmedSQLiteSession]:
        """Get or create an Agent for the given chat_jid and model.

        Callers must hold ``lock_for(chat_jid)``: the returned agent is shared
        and its per-message fields are rebound here.

        The session is deliberately *not* cached alongside the agent — it is
        keyed by chat alone, so history survives cache eviction and is shared by
        the text and vision agents for the same chat.
        """
        current_time = time.time()
        model_name = getattr(model, "model", None) or str(model)
        cache_key = (chat_jid, model_name)
        session = get_session(chat_jid)

        if cache_key in self._agents:
            agent, last_used = self._agents[cache_key]

            # Check if expired (TTL exceeded)
            if self._is_expired(last_used):
                del self._agents[cache_key]
                logger.info(f"Agent expired for {chat_jid}/{model_name} (TTL exceeded)")
            else:
                # Move to end (most recently used)
                self._agents.move_to_end(cache_key)
                agent.mcp_servers = mcp_servers
                agent.tools = tools or []
                agent.instructions = instructions
                self._agents[cache_key] = (agent, current_time)
                logger.info(
                    f"Reusing agent for {chat_jid}/{model_name} (cache: {len(self._agents)})"
                )
                return agent, session

        # Evict least recently used if at capacity
        if len(self._agents) >= MAX_AGENTS:
            (oldest_jid, oldest_model), _ = self._agents.popitem(last=False)
            logger.info(f"Evicting LRU agent for {oldest_jid}/{oldest_model}")

        agent = Agent(
            name="Leo", instructions=instructions, mcp_servers=mcp_servers, model=model,
            model_settings=_model_settings, tools=tools or [],
        )
        self._agents[cache_key] = (agent, current_time)
        logger.info(f"Created new agent for {chat_jid}/{model_name} (cache: {len(self._agents)})")
        return agent, session


# Global agent factory instance
agent_factory = AgentFactory()


# ── Reminder parsing agent ──────────────────────────────────────────────────

# Template agent — never run directly. Each call clones it with its own
# instructions, because two concurrent #remindme messages sharing one agent
# would race and one could be parsed against the other's timestamp.
_reminder_parser_agent = Agent(
    name="ReminderParser",
    instructions="",  # set per call, on the clone
    model=_cached_model,
    model_settings=_model_settings,
    output_type=ReminderParsed,
)


async def parse_remindme_with_agent(
    content: str, tz=None
) -> tuple[datetime, str]:
    """Use an OpenAI agent to parse a #remindme message into (remind_at, message).

    ``tz`` is the requesting user's timezone; "in 30 minutes" and "9am
    tomorrow" are both meaningless without it.

    Returns (remind_at_datetime, reminder_message_text).
    Raises ValueError if parsing fails.
    """
    tz = tz or TZ
    now = datetime.now(tz)
    current_time = now.strftime("%I:%M %p %Z, %A %B %d, %Y")

    parser_agent = _reminder_parser_agent.clone(
        instructions=REMINDER_INSTRUCTIONS_TEMPLATE.format(current_time=current_time)
    )

    result = await Runner.run(parser_agent, content)
    parsed: ReminderParsed = result.final_output

    try:
        remind_at = dateutil_parser.parse(parsed.remind_at, fuzzy=True)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Could not understand the time: {parsed.remind_at}") from exc

    # If no timezone was provided, assume the user's
    if remind_at.tzinfo is None:
        remind_at = remind_at.replace(tzinfo=tz)

    return (remind_at, parsed.reminder_message)
