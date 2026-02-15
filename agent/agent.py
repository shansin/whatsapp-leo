#!/usr/bin/env python3
"""Unix domain socket server for receiving WhatsApp messages from Go bridge."""

from contextlib import AsyncExitStack
from dataclasses import dataclass, asdict
from collections import OrderedDict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import orjson
import re
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
import sys
import asyncio
import logging
from dateutil import parser as dateutil_parser
from pydantic import BaseModel
from agents import Agent, Runner, trace, OpenAIChatCompletionsModel, SQLiteSession
from agents.mcp import MCPServerStdio

# Add whatsapp-mcp-server to path for direct imports
WHATSAPP_MCP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "whatsapp-mcp",
    "whatsapp-mcp-server",
)
sys.path.insert(0, WHATSAPP_MCP_DIR)
from whatsapp import send_message as whatsapp_send_message
from reminder import validate_reminder_time, store_reminder, ReminderScheduler
from briefing import (
    BriefingScheduler,
    add_briefing,
    list_briefings,
    remove_briefing,
    remove_all_briefings,
    parse_schedule_to_cron,
    get_next_run_from_cron,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AgentServer")

load_dotenv(override=True)

# Get instance GUID for multi-instance support
INSTANCE_GUID = os.getenv("INSTANCE_GUID", "default")

# Socket path for Unix domain socket (supports multi-instance via INSTANCE_GUID)
SOCKET_PATH = os.getenv("AGENT_SOCKET_PATH", f"/tmp/whatsapp-leo-{INSTANCE_GUID}.sock")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_AGENTS = int(os.getenv("MAX_AGENTS", "20"))
TTL_SECONDS = int(os.getenv("TTL_SECONDS", "1800"))
ALLOWED_SENDERS = [
    s.strip() for s in os.getenv("ALLOWED_SENDERS", "").split(",") if s.strip()
]
LEO_MENTION_ID = os.getenv("LEO_MENTION_ID", "@23833461416078")
IS_DEDICATED_NUMBER = os.getenv("IS_DEDICATED_NUMBER", "false").lower() == "true"

# Maximum message size to prevent memory exhaustion (10MB)
MAX_MESSAGE_SIZE = int(os.getenv("MAX_MESSAGE_SIZE", "10485760"))

# MCP Server Paths
WORKSPACE_MCP_PATH = os.getenv(
    "WORKSPACE_MCP_PATH",
    "/home/shsin/git_linux/workspace/workspace-server/dist/index.js",
)

# ── Cached singletons (avoid re-creation per message) ───────────────────────
_openai_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
_cached_model = OpenAIChatCompletionsModel(
    model=MODEL_NAME, openai_client=_openai_client
)

# Shared env copy (avoids copying 100+ vars per message)
_shared_env = os.environ.copy()
_shared_env["GEMINI_CLI_WORKSPACE_FORCE_FILE_STORAGE"] = "true"

# Pre-built static MCP param dicts
_workspace_mcp_params = {
    "command": "node",
    "args": [WORKSPACE_MCP_PATH, "--use-dot-names"],
    "env": _shared_env,
}
_brave_mcp_params = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
    "env": _shared_env,
}
_garmin_mcp_params = {
    "command": "uvx",
    "args": ["git+https://github.com/Taxuspt/garmin_mcp"],
}


# ── Pre-built instruction fragments (loaded from instructions.txt) ──────────
def _load_instructions():
    instr_path = os.path.join(os.path.dirname(__file__), "instructions.txt")
    if not os.path.exists(instr_path):
        logger.warning(f"instructions.txt not found at {instr_path}")
        return "", "", ""

    with open(instr_path, "r") as f:
        content = f.read()

    sections = {}
    current_section = None
    lines = []

    for line in content.splitlines():
        if line.startswith("[") and line.endswith("]"):
            if current_section:
                sections[current_section] = "\n".join(lines).strip()
            current_section = line[1:-1]
            lines = []
        else:
            lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(lines).strip()

    return (
        sections.get("BASE_INSTRUCTIONS", "") + "\n",
        "\n" + sections.get("PRIVILEDGED_INSTRUCTIONS", "") + "\n",
        "\n" + sections.get("COMMON_RULES", ""),
        sections.get("REMINDER_INSTRUCTIONS", ""),
    )


(
    _BASE_INSTRUCTION_TEMPLATE,
    _PRIVILEDGED_INSTRUCTIONS,
    _COMMON_RULES,
    _REMINDER_INSTRUCTIONS_TEMPLATE,
) = _load_instructions()

# Pre-built instruction templates (only {current_time} needs filling at message time)
_INSTRUCTIONS_PRIVILEGED_TEMPLATE = (
    _BASE_INSTRUCTION_TEMPLATE + _PRIVILEDGED_INSTRUCTIONS + _COMMON_RULES
)
_INSTRUCTIONS_BASIC_TEMPLATE = _BASE_INSTRUCTION_TEMPLATE + _COMMON_RULES


def format_leo_response(text: str) -> str:
    return f"_*(Leo)*_ {text}" if not IS_DEDICATED_NUMBER else text


async def _reply(message: "ReceivedMessage", text: str) -> None:
    """Send a WhatsApp reply to the originating message (non-blocking)."""
    await asyncio.to_thread(
        whatsapp_send_message,
        message.chat_jid,
        text,
        reply_to=message.id,
        reply_to_sender=message.sender_jid,
    )


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
            name="Leo", instructions=instructions, mcp_servers=mcp_servers, model=model
        )
        session = SQLiteSession(chat_jid)
        self._agents[chat_jid] = (agent, mcp_servers, session, current_time)
        logger.info(f"Created new agent for {chat_jid} (cache: {len(self._agents)})")
        return agent, session


# Global agent factory instance
agent_factory = AgentFactory()

TZ = ZoneInfo("America/Los_Angeles")


# ── Structured output model for reminder parsing ────────────────────────────


class ReminderParsed(BaseModel):
    reminder_message: str
    remind_at: str


# Cached ReminderParser agent (instructions are updated dynamically per call)
_reminder_parser_agent = Agent(
    name="ReminderParser",
    instructions="",  # set dynamically before each run
    model=_cached_model,
    output_type=ReminderParsed,
)


async def parse_remindme_with_agent(content: str) -> tuple[datetime, str]:
    """Use an OpenAI agent to parse a #remindme message into (remind_at, message).

    Returns (remind_at_datetime, reminder_message_text).
    Raises ValueError if parsing fails.
    """
    now = datetime.now(TZ)
    current_time = now.strftime("%I:%M %p %Z, %A %B %d, %Y")

    _reminder_parser_agent.instructions = _REMINDER_INSTRUCTIONS_TEMPLATE.format(
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


@dataclass
class ReceivedMessage:
    """Data structure for incoming WhatsApp messages."""

    chat_jid: str
    chat_name: str
    content: str
    file_length: int
    filename: str
    id: str
    is_from_me: bool
    media_type: str
    phone_number: str
    sender: str
    sender_jid: str
    timestamp: str
    url: str

    @classmethod
    def from_dict(cls, data: dict) -> "ReceivedMessage":
        return cls(
            chat_jid=data.get("chat_jid", ""),
            chat_name=data.get("chat_name", ""),
            content=data.get("content", ""),
            file_length=data.get("file_length", 0),
            filename=data.get("filename", ""),
            id=data.get("id", ""),
            is_from_me=data.get("is_from_me", False),
            media_type=data.get("media_type", ""),
            phone_number=data.get("phone_number", ""),
            sender=data.get("sender", ""),
            sender_jid=data.get("sender_jid", ""),
            timestamp=data.get("timestamp", ""),
            url=data.get("url", ""),
        )


async def handle_briefing_command(message: ReceivedMessage):
    """Handle #briefing commands for managing briefings."""
    content = message.content.strip()
    parts = content.split(maxsplit=2)

    if len(parts) < 2:
        await send_briefing_help(message)
        return

    subcommand = parts[1].lower()

    try:
        if subcommand == "add":
            # Format: #briefing add <name> <schedule> <prompt>
            # Example: #briefing add "Morning Brief" "9am everyday" "Get my sleep data and calendar events"
            await handle_briefing_add(message, parts)
        elif subcommand == "list":
            await handle_briefing_list(message)
        elif subcommand == "remove":
            # Format: #briefing remove <id>
            await handle_briefing_remove(message, parts)
        elif subcommand == "remove-all":
            await handle_briefing_remove_all(message)
        elif subcommand == "help":
            await send_briefing_help(message)
        else:
            await _reply(message, f"❌ Unknown briefing command: {subcommand}\n\nUse #briefing help for usage.")
    except Exception as e:
        logger.error(f"Error handling briefing command: {e}", exc_info=True)
        await _reply(message, f"❌ Error: {str(e)}")


async def handle_briefing_add(message: ReceivedMessage, parts: list):
    """Handle #briefing add command."""
    content = message.content.strip()

    # Extract quoted strings
    quoted = re.findall(r'"([^"]*)"', content)

    if len(quoted) < 2:
        await _reply(
            message,
            '❌ Usage: #briefing add "Name" "Schedule" Prompt text...\n\nExample:\n#briefing add "Morning Brief" "9am everyday" Get my sleep data from Garmin and calendar events for today',
        )
        return

    name = quoted[0]
    schedule = quoted[1]

    # Get prompt (everything after the second quoted string)
    prompt_start = content.find(f'"{schedule}"') + len(f'"{schedule}"')
    prompt = content[prompt_start:].strip()

    if not prompt:
        await _reply(message, "❌ Please provide a prompt for the briefing.")
        return

    try:
        briefing_id, cron_expr = add_briefing(name, prompt, schedule, message.chat_jid)
        next_run = get_next_run_from_cron(cron_expr)
        next_run_str = next_run.strftime("%b %d, %I:%M %p %Z")

        await _reply(
            message,
            f"📋 Briefing created!\n\n*Name:* {name}\n*Schedule:* {schedule}\n*Cron:* {cron_expr}\n*Next run:* {next_run_str}\n*ID:* {briefing_id}",
        )
        logger.info(f"Briefing '{name}' created with ID {briefing_id}")
    except ValueError as e:
        await _reply(message, f"❌ {e}")


async def handle_briefing_list(message: ReceivedMessage):
    """Handle #briefing list command."""
    briefings = list_briefings()

    if not briefings:
        await _reply(message, "📋 No briefings configured.\n\nUse #briefing add to create one.")
        return

    lines = ["📋 *Configured Briefings:*\n"]
    for b in briefings:
        status = "✅" if b["enabled"] else "⏸️"
        next_run = b["next_run_at"]
        if next_run:
            try:
                next_dt = datetime.fromisoformat(next_run)
                next_str = next_dt.strftime("%b %d, %I:%M %p")
            except (ValueError, TypeError):
                next_str = next_run
        else:
            next_str = "Not scheduled"
        lines.append(f"{status} *ID {b['id']}:* {b['name']}")
        lines.append(f"   Schedule: {b['schedule_cron']}")
        lines.append(f"   Next: {next_str}\n")

    await _reply(message, "\n".join(lines))


async def handle_briefing_remove(message: ReceivedMessage, parts: list):
    """Handle #briefing remove command."""
    if len(parts) < 3:
        await _reply(message, "❌ Usage: #briefing remove <id>")
        return

    try:
        briefing_id = int(parts[2])
        if remove_briefing(briefing_id):
            await _reply(message, f"✅ Briefing {briefing_id} removed.")
            logger.info(f"Briefing {briefing_id} removed")
        else:
            await _reply(message, f"❌ Briefing {briefing_id} not found.")
    except ValueError:
        await _reply(message, "❌ Please provide a valid briefing ID number.")


async def handle_briefing_remove_all(message: ReceivedMessage):
    """Handle #briefing remove-all command."""
    count = remove_all_briefings()
    await _reply(message, f"✅ Removed all briefings ({count} deleted).")
    logger.info(f"All briefings removed ({count} deleted)")


async def send_briefing_help(message: ReceivedMessage):
    """Send briefing help message."""
    help_text = """📋 *Briefing Commands*

Create automated AI briefings that run on a schedule.

*Commands:*
• #briefing add "Name" "Schedule" Prompt
  _Create a new briefing_
  Example: #briefing add "Morning Brief" "9am everyday" Get my sleep and calendar

• #briefing list
  _Show all briefings_

• #briefing remove <id>
  _Remove a briefing by ID_

• #briefing remove-all
  _Remove all briefings_

• #briefing help
  _Show this help_

*Schedule formats:*
• "9am everyday" - Daily at 9 AM
• "8am monday" - Mondays at 8 AM
• "5pm friday" - Fridays at 5 PM
• "every morning" - Daily at 9 AM
"""
    await _reply(message, help_text)


async def process_message(data: dict):
    """Process a single message asynchronously."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Full message payload: %s", orjson.dumps(data).decode())
    message = ReceivedMessage.from_dict(data)

    is_leo_mentioned = (
        "#leo" in message.content.lower() or "@leo" in message.content.lower()
    )

    # ── Handle #remindme ─────────────────────────
    if (
        IS_DEDICATED_NUMBER
        and ("#remindme" in message.content.lower())
        and (message.phone_number in ALLOWED_SENDERS)
    ):
        try:
            remind_at, original_msg = await parse_remindme_with_agent(message.content)
            validate_reminder_time(remind_at)
            store_reminder(
                message.chat_jid,
                original_msg,
                remind_at,
                message_id=message.id,
                sender_jid=message.sender_jid,
            )
            confirm_time = remind_at.strftime("%b %d, %I:%M %p %Z")
            await _reply(message, f"⏰ Reminder set for *{confirm_time}*!")
            logger.info(f"Reminder set for {confirm_time} in {message.chat_jid}")
            return
        except ValueError as e:
            await _reply(message, f"❌ {e}")
            return
        except Exception as e:
            logger.error(f"Error handling #remindme: {e}", exc_info=True)
            await _reply(message, "❌ Something went wrong setting the reminder.")
            return

    if "#remindme" in message.content.lower():
        return

    # ── Handle #briefing commands ─────────────────────────
    if (
        IS_DEDICATED_NUMBER
        and "#briefing" in message.content.lower()
        and message.phone_number in ALLOWED_SENDERS
    ):
        await handle_briefing_command(message)
        return

    should_leo_respond = False

    if IS_DEDICATED_NUMBER:
        # Respond if: DM (ends with @lid) OR group mention (@g in jid AND @23833461416078 in content)
        is_dm = message.chat_jid.endswith("@lid")
        is_group_mention = (
            "@g" in message.chat_jid and LEO_MENTION_ID in message.content
        )
        should_leo_respond = is_dm or is_group_mention
    else:
        should_leo_respond = is_leo_mentioned

    if should_leo_respond:
        logger.info(f"Leo mentioned by {message.sender}! Processing...")

        # Check if sender is allowed to use privileged feature
        is_allowed = message.phone_number in ALLOWED_SENDERS

        try:
            now = datetime.now(TZ)
            current_time = now.strftime("%I:%M %p PST, %B %d, %Y")

            # Use pre-built template — only interpolate the timestamp
            template = (
                _INSTRUCTIONS_PRIVILEGED_TEMPLATE
                if is_allowed
                else _INSTRUCTIONS_BASIC_TEMPLATE
            )
            instructions = template.format(current_time=current_time)

            # MCP servers use pre-built param dicts from module level

            async with AsyncExitStack() as stack:
                # Start Brave MCP server (WhatsApp is handled via direct function calls)
                brave_mcp_server = await stack.enter_async_context(
                    MCPServerStdio(
                        params=_brave_mcp_params, client_session_timeout_seconds=30
                    )
                )

                mcp_servers = [brave_mcp_server]

                # Conditionally start privileged MCPs
                if is_allowed:
                    workspace_mcp_server = await stack.enter_async_context(
                        MCPServerStdio(
                            params=_workspace_mcp_params,
                            client_session_timeout_seconds=300,
                        )
                    )
                    mcp_servers.append(workspace_mcp_server)

                    garmin_mcp_server = await stack.enter_async_context(
                        MCPServerStdio(
                            params=_garmin_mcp_params,
                            client_session_timeout_seconds=120,
                        )
                    )
                    mcp_servers.append(garmin_mcp_server)

                agent, session = await agent_factory.get_agent(
                    chat_jid=message.chat_jid,
                    mcp_servers=mcp_servers,
                    model=_cached_model,
                    instructions=instructions,
                )

                with trace("LeoWhatsappAssistant"):
                    result = await Runner.run(
                        agent, orjson.dumps(asdict(message)).decode(), session=session
                    )

                logger.info(f"Agent execution completed. Result: {result.final_output}")

                # Send the agent's response directly via WhatsApp
                if result.final_output:
                    success, send_result = await asyncio.to_thread(
                        whatsapp_send_message,
                        message.chat_jid,
                        format_leo_response(result.final_output),
                    )
                    if success:
                        logger.info(f"Message sent successfully to {message.chat_jid}")
                    else:
                        logger.error(
                            f"Failed to send message to {message.chat_jid}: {send_result}"
                        )

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)


async def handle_client(reader, writer):
    """Handle a single client connection."""
    try:
        chunks = bytearray()
        message_data = None
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            chunks.extend(chunk)
            # Check message size limit
            if len(chunks) > MAX_MESSAGE_SIZE:
                logger.error(
                    "Message too large (%d bytes), dropping connection", len(chunks)
                )
                writer.write(orjson.dumps({"error": "Message too large"}))
                await writer.drain()
                return
            try:
                message_data = orjson.loads(chunks)
                break
            except orjson.JSONDecodeError:
                continue

        if not chunks:
            return

        if message_data is not None:
            # Process immediately in background task
            asyncio.create_task(process_message(message_data))

            response = orjson.dumps(
                {"status": "processing", "message": "Message received"}
            )
            writer.write(response)
            await writer.drain()
        else:
            writer.write(b'{"error": "Invalid JSON"}')
            await writer.drain()

    except (ConnectionResetError, BrokenPipeError):
        # Client disconnected during processing - this is expected behavior for some clients
        pass
    except Exception as e:
        logger.error(f"Error handling client: {e}")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.warning(f"Error closing writer: {e}")


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
    MAX_BRIEFING_RETRIES = 3

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
        _INSTRUCTIONS_PRIVILEGED_TEMPLATE.format(current_time=current_time)
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
                workspace_mcp_server = await stack.enter_async_context(
                    MCPServerStdio(
                        params=_workspace_mcp_params,
                        client_session_timeout_seconds=300,
                    )
                )
                garmin_mcp_server = await stack.enter_async_context(
                    MCPServerStdio(
                        params=_garmin_mcp_params,
                        client_session_timeout_seconds=120,
                    )
                )

                mcp_servers = [brave_mcp_server, workspace_mcp_server, garmin_mcp_server]

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


async def main():
    """Start the Unix domain socket server."""
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    # Start the reminder scheduler in the background
    scheduler = ReminderScheduler(send_fn=whatsapp_send_message)
    asyncio.create_task(scheduler.run())

    # Start the briefing scheduler in the background
    briefing_scheduler = BriefingScheduler(
        execute_fn=execute_briefing_prompt,
        send_fn=whatsapp_send_message,
    )
    asyncio.create_task(briefing_scheduler.run())

    server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o666)

    logger.info(f"Unix domain socket Agent Server running at {SOCKET_PATH}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Agent Server...")
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
