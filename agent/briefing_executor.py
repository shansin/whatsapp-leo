"""Briefing execution through the AI agent pipeline with retry logic."""

import asyncio
from datetime import datetime

from agents import Agent, Runner, trace, SQLiteSession
from agents.exceptions import AgentsException, ModelBehaviorError
from openai import APIConnectionError, APIStatusError, APITimeoutError

from config import TRACING_ENABLED, _cached_model
from mcp_pool import mcp_pool
from instructions import INSTRUCTIONS_PRIVILEGED_TEMPLATE
from agent_factory import TZ
from logging_setup import logger

MAX_BRIEFING_RETRIES = 3


def _is_retryable(exc: BaseException) -> bool:
    """True for transient failures worth another attempt.

    Matching on ``"500" in str(e)`` used to also catch, say, a briefing that
    merely mentioned the number 500 in an error message.
    """
    if isinstance(
        exc,
        (APIConnectionError, APITimeoutError, asyncio.TimeoutError, ModelBehaviorError),
    ):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code is not None and exc.status_code >= 500
    # MCP tool plumbing wraps transport failures in AgentsException.
    if isinstance(exc, AgentsException):
        return True
    return False


async def execute_briefing_prompt(
    prompt: str, chat_jid: str, briefing_name: str
) -> str:
    """
    Execute a briefing prompt through the AI agent.

    This function runs the briefing prompt through the full AI pipeline
    with access to all privileged MCP servers (workspace, garmin, etc.).
    Retries up to MAX_BRIEFING_RETRIES times on transient LLM errors (e.g.
    malformed tool-call JSON causing 500s).
    """
    now = datetime.now(TZ)
    current_time = now.strftime("%I:%M %p %Z, %A %B %d, %Y")

    # Use privileged instructions for briefings (they run as system tasks)
    # Add explicit briefing output instructions
    briefing_output_rule = """
**BRIEFING OUTPUT RULE**: This is an automated briefing. Return ONLY plain text formatted for WhatsApp.
NO JSON, NO XML, NO code blocks, NO raw API responses. Use emojis, bullet points (* ), bold (*text*), and clear formatting.
If any tool call fails or returns an error, skip that section gracefully and continue with the rest of the briefing.

**TOOL USAGE RULES FOR BRIEFINGS** (you MUST follow these):
- Call tools ONE AT A TIME. Do NOT make parallel or batch tool calls.
- Use ONLY the required parameters for each tool call. Do NOT include optional parameters unless absolutely necessary.
- For calendar.listEvents: only pass calendarId, timeMin, and timeMax. Do NOT pass attendeeResponseStatus or any other optional parameters.
- For calendar.createEvent: pass calendarId, summary, start, and end. Only add attendees if explicitly requested.
- Keep tool call arguments as simple as possible. Prefer simple string values over complex nested objects.
- If a tool call fails, do NOT retry it. Skip that data and move on to the next section.
"""
    instructions = (
        INSTRUCTIONS_PRIVILEGED_TEMPLATE.format(current_time=current_time)
        + briefing_output_rule
    )

    last_error = None
    for attempt in range(1, MAX_BRIEFING_RETRIES + 1):
        try:
            await mcp_pool.ensure_started()

            # Create a fresh agent for each attempt (avoids poisoned conversation state)
            briefing_agent = Agent(
                name=f"LeoBriefing-{briefing_name}",
                instructions=instructions,
                mcp_servers=mcp_pool.servers(is_privileged=True),
                model=_cached_model,
            )

            # Fresh session per attempt so retries don't replay the broken tool
            # call. Deliberately in-memory: a briefing carries no history worth
            # keeping, and persisting one row set per attempt would only grow
            # sessions.db forever.
            session = SQLiteSession(f"briefing:{briefing_name}:{attempt}")

            with trace("LeoBriefing", disabled=not TRACING_ENABLED):
                result = await Runner.run(briefing_agent, prompt, session=session)

            # Extract the final output
            if result.final_output is None:
                return "No briefing content generated."

            output = result.final_output
            if not isinstance(output, str):
                if hasattr(output, "model_dump"):
                    output = str(output.model_dump())
                elif hasattr(output, "__dict__"):
                    output = str(output.__dict__)
                else:
                    output = str(output)

            return output

        except Exception as e:
            last_error = e
            if _is_retryable(e) and attempt < MAX_BRIEFING_RETRIES:
                wait = 2 ** attempt  # 2s, 4s
                logger.warning(
                    f"Briefing '{briefing_name}' attempt {attempt}/{MAX_BRIEFING_RETRIES} "
                    f"failed with retryable error, retrying in {wait}s: {e}"
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    f"Briefing '{briefing_name}' failed after {attempt} attempt(s): {e}",
                    exc_info=True,
                )
                return f"❌ Error generating briefing: {str(e)}"

    # Should not reach here, but just in case
    return f"❌ Error generating briefing after {MAX_BRIEFING_RETRIES} attempts: {str(last_error)}"
