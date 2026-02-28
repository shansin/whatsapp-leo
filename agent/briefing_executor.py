"""Briefing execution through the AI agent pipeline with retry logic."""

import os
import asyncio
from contextlib import AsyncExitStack
from datetime import datetime

from agents import Agent, Runner, trace, SQLiteSession
from agents.mcp import MCPServerStdio

from config import (
    WORKSPACE_MCP_PATH,
    _cached_model,
    _brave_mcp_params,
    _workspace_mcp_params,
    _garmin_mcp_params,
)
from instructions import INSTRUCTIONS_PRIVILEGED_TEMPLATE
from agent_factory import TZ
from logging_setup import logger

MAX_BRIEFING_RETRIES = 3


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
    current_time = now.strftime("%I:%M %p PST, %B %d, %Y")

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
            async with AsyncExitStack() as stack:
                # Start all MCP servers
                brave_mcp_server = await stack.enter_async_context(
                    MCPServerStdio(
                        params=_brave_mcp_params, client_session_timeout_seconds=30
                    )
                )
                mcp_servers = [brave_mcp_server]

                if os.path.exists(WORKSPACE_MCP_PATH):
                    workspace_mcp_server = await stack.enter_async_context(
                        MCPServerStdio(
                            params=_workspace_mcp_params,
                            client_session_timeout_seconds=300,
                        )
                    )
                    mcp_servers.append(workspace_mcp_server)
                else:
                    logger.warning(f"Workspace MCP not found at {WORKSPACE_MCP_PATH}")
                garmin_mcp_server = await stack.enter_async_context(
                    MCPServerStdio(
                        params=_garmin_mcp_params,
                        client_session_timeout_seconds=120,
                    )
                )
                mcp_servers.append(garmin_mcp_server)

                # Create a fresh agent for each attempt (avoids poisoned conversation state)
                briefing_agent = Agent(
                    name=f"LeoBriefing-{briefing_name}",
                    instructions=instructions,
                    mcp_servers=mcp_servers,
                    model=_cached_model,
                )

                # Fresh session per attempt so retries don't replay the broken tool call
                session = SQLiteSession(f"briefing:{briefing_name}:{attempt}")

                with trace("LeoBriefing"):
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
            is_retryable = "500" in str(e) or "parsing" in str(e).lower()
            if is_retryable and attempt < MAX_BRIEFING_RETRIES:
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
