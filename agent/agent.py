#!/usr/bin/env python3
"""Unix domain socket server for receiving WhatsApp messages from Go bridge."""

from contextlib import AsyncExitStack
from dataclasses import dataclass, asdict
from collections import OrderedDict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
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
from typing import Dict, List, Tuple

# Add whatsapp-mcp-server to path for direct imports
WHATSAPP_MCP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'whatsapp-mcp', 'whatsapp-mcp-server')
sys.path.insert(0, WHATSAPP_MCP_DIR)
from whatsapp import send_message as whatsapp_send_message
from reminder import validate_reminder_time, store_reminder, ReminderScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
ALLOWED_SENDERS = [s.strip() for s in os.getenv("ALLOWED_SENDERS", "").split(",") if s.strip()]
LEO_MENTION_ID = os.getenv("LEO_MENTION_ID", "@23833461416078")
IS_DEDICATED_NUMBER = os.getenv("IS_DEDICATED_NUMBER", "false").lower() == "true"

# MCP Server Paths
WORKSPACE_MCP_PATH = os.getenv("WORKSPACE_MCP_PATH", "/home/shsin/git_linux/workspace/workspace-server/dist/index.js")

# ── Cached singletons (avoid re-creation per message) ───────────────────────
_openai_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
_cached_model = OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=_openai_client)

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
        "\n" + sections.get("COMMON_RULES", "")
    )

_BASE_INSTRUCTION_TEMPLATE, _PRIVILEDGED_INSTRUCTIONS, _COMMON_RULES = _load_instructions()

def format_leo_response(text: str) -> str:
    return f"_*(Leo)*_ {text}" if not IS_DEDICATED_NUMBER else text

class AgentFactory:
    """Factory for creating and caching Agent instances with LRU eviction and TTL."""

    def __init__(self):
        # OrderedDict to maintain LRU order: most recently used at the end
        self._agents: OrderedDict[str, Tuple[Agent, List[MCPServerStdio], SQLiteSession, float]] = OrderedDict()
    
    def _is_expired(self, last_used: float) -> bool:
        """Check if an entry has exceeded the TTL."""
        return (time.time() - last_used) > TTL_SECONDS
    
    async def get_agent(self, chat_jid: str, mcp_servers: List[MCPServerStdio], model, instructions: str) -> Tuple[Agent, SQLiteSession]:
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
                logger.info(f"Reusing agent for {chat_jid} (cache: {len(self._agents)})")
                return agent, session
        
        # Evict least recently used if at capacity
        if len(self._agents) >= MAX_AGENTS:
            oldest_jid, _ = self._agents.popitem(last=False)
            logger.info(f"Evicting LRU agent for {oldest_jid}")
        
        # Create new agent and session
        agent = Agent(
            name="Leo",
            instructions=instructions,
            mcp_servers=mcp_servers,
            model=model
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


async def parse_remindme_with_agent(content: str) -> tuple[datetime, str]:
    """Use an OpenAI agent to parse a #remindme message into (remind_at, message).

    Returns (remind_at_datetime, reminder_message_text).
    Raises ValueError if parsing fails.
    """
    now = datetime.now(TZ)
    current_time = now.strftime("%I:%M %p %Z, %A %B %d, %Y")

    agent = Agent(
        name="ReminderParser",
        instructions=(
            f"The current date and time is {current_time}.\n"
            "You are a reminder parser. The user will give you a message that contains "
            "a reminder request (tagged with #remindme). Your job is to extract:\n"
            "1. `reminder_message` — what the user wants to be reminded about. "
            "Keep it concise and natural. Do NOT include the #remindme tag or the time expression.\n"
            "2. `remind_at` — the exact date and time the reminder should fire, as a "
            "human-readable datetime string that dateutil can parse. Examples:\n"
            "   - 'in 30 minutes' → compute the exact time and output e.g. '2:30 PM Feb 12, 2026'\n"
            "   - 'at 5pm' → '5:00 PM Feb 12, 2026'\n"
            "   - 'tomorrow at 9am' → '9:00 AM Feb 13, 2026'\n"
            "Always include the full date and time. Use 12-hour format with AM/PM.\n"
        ),
        model=_cached_model,
        output_type=ReminderParsed,
    )

    result = await Runner.run(agent, content)
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

async def process_message(data: dict):
    """Process a single message asynchronously."""
    logger.info(f"Full message payload: {json.dumps(data, indent=2)}")
    message = ReceivedMessage.from_dict(data)

    is_leo_mentioned = "#leo" in message.content.lower() or "@leo" in message.content.lower()

    # ── Handle #remindme ─────────────────────────
    if (not IS_DEDICATED_NUMBER) and ("#remindme" in message.content.lower()) and (message.phone_number in ALLOWED_SENDERS):
        try:
            remind_at, original_msg = await parse_remindme_with_agent(message.content)
            validate_reminder_time(remind_at)
            store_reminder(message.chat_jid, original_msg, remind_at, message_id=message.id, sender_jid=message.sender_jid)
            confirm_time = remind_at.strftime("%b %d, %I:%M %p %Z")
            whatsapp_send_message(
                message.chat_jid,
                f"⏰ Reminder set for *{confirm_time}*!",
                reply_to=message.id,
                reply_to_sender=message.sender_jid,
            )
            logger.info(f"Reminder set for {confirm_time} in {message.chat_jid}")
            return
        except ValueError as e:
            whatsapp_send_message(message.chat_jid, f"❌ {e}", reply_to=message.id, reply_to_sender=message.sender_jid)
            return
        except Exception as e:
            logger.error(f"Error handling #remindme: {e}", exc_info=True)
            whatsapp_send_message(message.chat_jid, "❌ Something went wrong setting the reminder.", reply_to=message.id, reply_to_sender=message.sender_jid)
            return

    should_leo_respond = False

    if(IS_DEDICATED_NUMBER):
        # Respond if: DM (ends with @lid) OR group mention (@g in jid AND @23833461416078 in content)
        is_dm = message.chat_jid.endswith("@lid")
        is_group_mention = "@g" in message.chat_jid and LEO_MENTION_ID in message.content
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
            
            # Build instructions with timestamp interpolation only
            base_instruction = _BASE_INSTRUCTION_TEMPLATE.format(current_time=current_time)

            # Compose final instructions based on permission
            if is_allowed:
                Instruction = base_instruction + _PRIVILEDGED_INSTRUCTIONS + _COMMON_RULES
            else:
                Instruction = base_instruction + _COMMON_RULES

            # MCP servers use pre-built param dicts from module level
            
            async with AsyncExitStack() as stack:
                # Start Brave MCP server (WhatsApp is handled via direct function calls)
                brave_mcp_server = await stack.enter_async_context(MCPServerStdio(params=_brave_mcp_params, client_session_timeout_seconds=30))
                
                mcp_servers = [brave_mcp_server]
                
                # Conditionally start privileged MCPs
                if is_allowed:
                    workspace_mcp_server = await stack.enter_async_context(MCPServerStdio(params=_workspace_mcp_params, client_session_timeout_seconds=300))
                    mcp_servers.append(workspace_mcp_server)

                    garmin_mcp_server = await stack.enter_async_context(MCPServerStdio(params=_garmin_mcp_params, client_session_timeout_seconds=120))
                    mcp_servers.append(garmin_mcp_server)

                agent, session = await agent_factory.get_agent(
                    chat_jid=message.chat_jid,
                    mcp_servers=mcp_servers,
                    model=_cached_model,
                    instructions=Instruction
                )

                with trace("LeoWhatsappAssistant"):
                    result = await Runner.run(agent, json.dumps(asdict(message)), session=session)

                logger.info(f"Agent execution completed. Result: {result.final_output}")
                
                # Send the agent's response directly via WhatsApp
                if result.final_output:
                    success, send_result = whatsapp_send_message(message.chat_jid, format_leo_response(result.final_output))
                    if success:
                        logger.info(f"Message sent successfully to {message.chat_jid}")
                    else:
                        logger.error(f"Failed to send message to {message.chat_jid}: {send_result}")

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

async def handle_client(reader, writer):
    """Handle a single client connection."""
    try:
        data = b""
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                break
            data += chunk
            try:
                json.loads(data.decode())
                break
            except json.JSONDecodeError:
                continue
        
        if not data:
            return

        try:
            message_data = json.loads(data.decode())
            # Process immediately in background task
            asyncio.create_task(process_message(message_data))
            
            response = json.dumps({
                "status": "processing",
                "message": "Message received"
            })
            writer.write(response.encode())
            await writer.drain()
            
        except json.JSONDecodeError:
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

async def main():
    """Start the Unix domain socket server."""
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    
    # Start the reminder scheduler in the background
    scheduler = ReminderScheduler(send_fn=whatsapp_send_message)
    asyncio.create_task(scheduler.run())
    
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
