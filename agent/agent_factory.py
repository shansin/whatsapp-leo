"""Agent factory with LRU caching and TTL, plus reminder parsing agent."""

import time
from collections import OrderedDict
from datetime import datetime

from dateutil import parser as dateutil_parser
from agents import Agent, Runner, SQLiteSession
from agents.mcp import MCPServerStdio

from config import MAX_AGENTS, TTL_SECONDS, TZ, _cached_model, _model_settings
from instructions import REMINDER_INSTRUCTIONS_TEMPLATE
from models import ReminderParsed
from logging_setup import logger


class AgentFactory:
    """Factory for creating and caching Agent instances with LRU eviction and TTL."""

    def __init__(self):
        # OrderedDict to maintain LRU order: most recently used at the end
        self._agents: OrderedDict[
            str, tuple[Agent, list[MCPServerStdio], SQLiteSession, float]
        ] = OrderedDict()

    def _is_expired(self, last_used: float) -> bool:
        """Check if an entry has exceeded the TTL."""
        return (time.time() - last_used) > TTL_SECONDS

    async def get_agent(
        self, chat_jid: str, mcp_servers: list[MCPServerStdio], model, instructions: str
    ) -> tuple[Agent, SQLiteSession]:
        """Get or create an Agent for the given chat_jid."""
        current_time = time.time()

        if chat_jid in self._agents:
            agent, _, session, last_used = self._agents[chat_jid]

            # Check if expired (TTL exceeded)
            if self._is_expired(last_used):
                del self._agents[chat_jid]
                logger.info(f"Agent expired for {chat_jid} (TTL exceeded)")
            else:
                # Move to end (most recently used)
                self._agents.move_to_end(chat_jid)
                agent.mcp_servers = mcp_servers
                self._agents[chat_jid] = (agent, mcp_servers, session, current_time)
                logger.info(
                    f"Reusing agent for {chat_jid} (cache: {len(self._agents)})"
                )
                return agent, session

        # Evict least recently used if at capacity
        if len(self._agents) >= MAX_AGENTS:
            oldest_jid, _ = self._agents.popitem(last=False)
            logger.info(f"Evicting LRU agent for {oldest_jid}")

        # Create new agent and session
        agent = Agent(
            name="Leo", instructions=instructions, mcp_servers=mcp_servers, model=model, model_settings=_model_settings
        )
        session = SQLiteSession(chat_jid)
        self._agents[chat_jid] = (agent, mcp_servers, session, current_time)
        logger.info(f"Created new agent for {chat_jid} (cache: {len(self._agents)})")
        return agent, session


# Global agent factory instance
agent_factory = AgentFactory()


# ── Reminder parsing agent ──────────────────────────────────────────────────

# Cached ReminderParser agent (instructions are updated dynamically per call)
_reminder_parser_agent = Agent(
    name="ReminderParser",
    instructions="",  # set dynamically before each run
    model=_cached_model,
    model_settings=_model_settings,
    output_type=ReminderParsed,
)


async def parse_remindme_with_agent(content: str) -> tuple[datetime, str]:
    """Use an OpenAI agent to parse a #remindme message into (remind_at, message).

    Returns (remind_at_datetime, reminder_message_text).
    Raises ValueError if parsing fails.
    """
    now = datetime.now(TZ)
    current_time = now.strftime("%I:%M %p %Z, %A %B %d, %Y")

    _reminder_parser_agent.instructions = REMINDER_INSTRUCTIONS_TEMPLATE.format(
        current_time=current_time
    )

    result = await Runner.run(_reminder_parser_agent, content)
    parsed: ReminderParsed = result.final_output

    try:
        remind_at = dateutil_parser.parse(parsed.remind_at, fuzzy=True)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Could not understand the time: {parsed.remind_at}") from exc

    # If no timezone was provided, assume our local TZ
    if remind_at.tzinfo is None:
        remind_at = remind_at.replace(tzinfo=TZ)

    return (remind_at, parsed.reminder_message)
