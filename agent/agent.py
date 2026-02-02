#!/usr/bin/env python3
"""Unix domain socket server for receiving WhatsApp messages from Go bridge."""

from dataclasses import dataclass, asdict
from collections import OrderedDict
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
import asyncio
import logging
from agents import Agent, Runner, trace, OpenAIChatCompletionsModel, SQLiteSession
from agents.mcp import MCPServerStdio
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AgentServer")

# Socket path for Unix domain socket
SOCKET_PATH = "/tmp/whatsapp-leo.sock"

load_dotenv(override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_AGENTS = int(os.getenv("MAX_AGENTS", "20"))
TTL_SECONDS = int(os.getenv("TTL_SECONDS", "1800"))
ALLOWED_SENDERS = [s.strip() for s in os.getenv("ALLOWED_SENDERS", "").split(",") if s.strip()]

# MCP Server Paths
WORKSPACE_MCP_PATH = os.getenv("WORKSPACE_MCP_PATH", "/home/shant/git_linux/workspace/workspace-server/dist/index.js")
WHATSAPP_MCP_PATH = os.getenv("WHATSAPP_MCP_PATH", "/home/shant/git_linux/whatsapp-leo/whatsapp-mcp/whatsapp-mcp-server/main.py")

class AgentFactory:
    """Factory for creating and caching Agent instances with LRU eviction and TTL."""

    def __init__(self):
        # OrderedDict to maintain LRU order: most recently used at the end
        self._agents: OrderedDict[str, Tuple[Agent, List[MCPServerStdio], SQLiteSession, float]] = OrderedDict()
    
    def _is_expired(self, last_used: float) -> bool:
        """Check if an entry has exceeded the TTL."""
        import time
        return (time.time() - last_used) > TTL_SECONDS
    
    async def get_agent(self, chat_jid: str, mcp_servers: List[MCPServerStdio], model, instructions: str) -> Tuple[Agent, SQLiteSession]:
        """Get or create an Agent for the given chat_jid."""
        import time
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
            timestamp=data.get("timestamp", ""),
            url=data.get("url", ""),
        )

async def process_message(data: dict):
    """Process a single message asynchronously."""
    message = ReceivedMessage.from_dict(data)
    logger.info(f"Received message from {message.sender}: {message.content[:50]}...")

    is_leo_mentioned = "#leo" in message.content.lower() or "@leo" in message.content.lower()
    if is_leo_mentioned and message.phone_number in ALLOWED_SENDERS:
        logger.info("Leo mentioned! Processing...")
        
        try:
            client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            model = OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client)
            
            from datetime import datetime
            from zoneinfo import ZoneInfo
        
            now = datetime.now(ZoneInfo("America/Los_Angeles"))
            current_time = now.strftime("%I:%M %p PST, %B %d, %Y")
            
            Instruction = f"""Date and time right now is {current_time}. 
        
            You are a powerful assistant called Leo. You MUST interact with user through whatsapp-mcp-server. 
            
            You can help with:
            - **General topics and queries**: from your knowledge base
            - **Web search**: using brave search
            - **Google Docs**: Create, read, find, update (append/replace/insert), and move documents.
            - **Google Drive**: Find/create folders, search for files, and download files.
            - **Google Calendar**: List calendars, view events, create/update/delete events, respond to invites, and find free time.
            - **Google Sheets**: Read content, get ranges, find spreadsheets, and get metadata.
            - **Google Slides**: Read text, find presentations, and get metadata.
            - **Gmail**: Search threads, draft/send emails, manage labels.
        
            **Important Rules:**
            1. **User Interaction**:
               - You MUST respond to user's message using send_message() function from whatsapp-mcp-server. 
               - Use the FULL chat_jid value ({message.chat_jid}) as the recipient parameter. 
               - Do not ask followup questions. Just answer and finish.
            2. **Safety**: 
               - Always PREVIEW write operations (creating events, sending emails, editing docs) before executing them. 
               - Ask for explicit user confirmation for destructive actions or sending messages.
            3. **Be concise, helpful, and professional.
            """

            # MCP Server Configuration
            workspace_mcp_server_params = {
                "command": "node",
                "args": [WORKSPACE_MCP_PATH, "--use-dot-names"],
                "env": {
                    "GEMINI_CLI_WORKSPACE_FORCE_FILE_STORAGE": "true",
                    "PATH": os.environ["PATH"]
                }
            }

            whatsapp_mcp_server_params = {
                "command": "uv", 
                "args": ["run", WHATSAPP_MCP_PATH] # Assuming run inside uv venv or with uv run script
            }
            # Fix params for UV run if needed based on original file:
            # original: ["--directory", "...", "run", "main.py"]
            # simplified to use the path we got from env or default
            whatsapp_dir = os.path.dirname(WHATSAPP_MCP_PATH)
            whatsapp_script = os.path.basename(WHATSAPP_MCP_PATH)
            whatsapp_mcp_server_params = {
                "command": "uv",
                "args": [
                    "--directory",
                    whatsapp_dir,
                    "run",
                    whatsapp_script
                ]
            }

            brave_env = os.environ.copy()
            brave_params = {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"], "env": brave_env}
            
            async with MCPServerStdio(params=brave_params, client_session_timeout_seconds=30) as brave_mcp_server:
                async with MCPServerStdio(params=workspace_mcp_server_params, client_session_timeout_seconds=300) as workspace_mcp_server:
                    async with MCPServerStdio(params=whatsapp_mcp_server_params, client_session_timeout_seconds=120) as whatsapp_mcp_server:
                        agent, session = await agent_factory.get_agent(
                            chat_jid=message.chat_jid,
                            mcp_servers=[whatsapp_mcp_server, brave_mcp_server, workspace_mcp_server],
                            model=model,
                            instructions=Instruction
                        )

                        with trace("LeoWhatsappAssistant"):
                            result = await Runner.run(agent, json.dumps(asdict(message)), session=session)

                        logger.info(f"Agent execution completed. Result: {result.final_output}")

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
