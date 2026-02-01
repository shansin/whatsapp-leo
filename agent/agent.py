#!/usr/bin/env python3
"""Unix domain socket server for receiving WhatsApp messages from Go bridge."""

from dataclasses import dataclass
from collections import OrderedDict
import json
from dotenv import load_dotenv
from openai import OpenAI
import os
import socket
from agents import Agent, Runner, trace, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput, SQLiteSession
from pydantic import BaseModel
from openai import AsyncOpenAI
from agents import function_tool
from typing import Dict, Tuple
import asyncio
from agents.mcp import MCPServerStdio
import threading
import queue

# Socket path for Unix domain socket
SOCKET_PATH = "/tmp/whatsapp-leo.sock"

load_dotenv(override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

MAX_AGENTS = int(os.getenv("MAX_AGENTS"))
TTL_SECONDS = int(os.getenv("TTL_SECONDS"))


class AgentFactory:
    """Factory for creating and caching Agent instances with LRU eviction and TTL.
    this is needed for agent to maintain conversation history """

    def __init__(self):
        # OrderedDict to maintain LRU order: most recently used at the end
        # Each entry stores: (Agent, MCPServerStdio, SQLiteSession, last_used_timestamp)
        self._agents: OrderedDict[str, Tuple[Agent, MCPServerStdio, SQLiteSession, float]] = OrderedDict()
        self._lock = threading.Lock()
    
    def _is_expired(self, last_used: float) -> bool:
        """Check if an entry has exceeded the TTL."""
        import time
        return (time.time() - last_used) > TTL_SECONDS
    
    async def get_agent(self, chat_jid: str, mcp_server: MCPServerStdio, model, instructions: str) -> Tuple[Agent, SQLiteSession]:
        """
        Get or create an Agent for the given chat_jid.
        
        If an agent exists for this chat_jid, it is reused and marked as recently used.
        The agent's mcp_servers is always updated with the current (active) MCP server.
        If a new agent is needed and capacity is exceeded, the least recently used agent is evicted.
        Agents not used within 20 minutes are discarded and recreated.
        
        Args:
            chat_jid: The WhatsApp chat JID to identify the agent
            mcp_server: The MCP server instance to use
            model: The model to use for the agent
            instructions: The instructions for the agent
            
        Returns:
            A tuple of (Agent, SQLiteSession) for the given chat_jid
        """
        import time
        current_time = time.time()
        
        with self._lock:
            if chat_jid in self._agents:
                agent, _, session, last_used = self._agents[chat_jid]
                
                # Check if expired (TTL exceeded)
                if self._is_expired(last_used):
                    # Remove expired entry, will create fresh one below
                    del self._agents[chat_jid]
                    print(f"[AgentFactory] Agent expired for chat_jid: {chat_jid} (TTL exceeded)")
                else:
                    # Move to end (most recently used) and update timestamp
                    self._agents.move_to_end(chat_jid)
                    # Update the agent's mcp_servers with the new active server
                    # The old server connection is closed after each async with block
                    agent.mcp_servers = [mcp_server]
                    self._agents[chat_jid] = (agent, mcp_server, session, current_time)
                    print(f"[AgentFactory] Reusing agent for chat_jid: {chat_jid} (cache size: {len(self._agents)})")
                    return agent, session
            
            # Evict least recently used if at capacity
            if len(self._agents) >= MAX_AGENTS:
                oldest_jid, (_, old_server, _, _) = self._agents.popitem(last=False)
                print(f"[AgentFactory] Evicting LRU agent for chat_jid: {oldest_jid}")
            
            # Create new agent and session
            agent = Agent(
                name="Leo",
                instructions=instructions,
                mcp_servers=[mcp_server],
                model=model
            )
            # Create a unique session for this chat_jid to maintain conversation history
            session = SQLiteSession(chat_jid)
            self._agents[chat_jid] = (agent, mcp_server, session, current_time)
            print(f"[AgentFactory] Created new agent for chat_jid: {chat_jid} (cache size: {len(self._agents)})")
            return agent, session
    
    def get_cache_size(self) -> int:
        """Return the current number of cached agents."""
        with self._lock:
            return len(self._agents)


# Global agent factory instance
agent_factory = AgentFactory()

# Task queue for background processing
task_queue = queue.Queue()

def worker():
    """Background worker that processes tasks from the queue."""
    while True:
        try:
            data = task_queue.get()
            if data is None:  # Shutdown signal
                break
            print(f"[Agent Worker] Processing task from queue (queue size: {task_queue.qsize()})")
            asyncio.run(reply_to_message(data))
            task_queue.task_done()
            print(f"[Agent Worker] Task completed (queue size: {task_queue.qsize()})")
        except Exception as e:
            print(f"[Agent Worker] Error processing task: {e}")
            task_queue.task_done()

# Start the worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

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
        """Create a ReceivedMessage from a dictionary."""
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

async def reply_to_message(data: dict) -> dict:
    """
    Process the incoming request data.
    
    Args:
        data: The JSON payload from the POST request
        
    Returns:
        A response dictionary
    """
    # Parse incoming data into ReceivedMessage structure
    message = ReceivedMessage.from_dict(data)
    print(f"[Agent Server] Received message: {message}")

    if("#leo" in message.content.lower()):
        print("[Agent Server] Leo mentioned!")
        client = AsyncOpenAI(base_url= OLLAMA_BASE_URL, api_key="ollama")
        model = OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client)
        
        from datetime import datetime
        from zoneinfo import ZoneInfo
    
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
        current_time = now.strftime("%I:%M %p PST, %B %d, %Y")
    
        Instruction = f"Date and time right now is {current_time}. You are a helpful assistant called #leo. Please respond to user's message in a helpful and concise manner using send_message() function from whatsapp-mcp-server. You must use send_message() function to send the response. Use the FULL chat_jid value (including the @lid or @s.whatsapp.net suffix) as the recipient parameter. Do not ask followup questions. Just answer and finish."

        #Push MCP Server instantiation
        whatsapp_mcp_server_params = {"command": "uv", "args": [
        "--directory",
        "/home/shant/git_linux/whatsapp-leo/whatsapp-mcp/whatsapp-mcp-server",
        "run", 
        "main.py"]
        }
        async with MCPServerStdio(params=whatsapp_mcp_server_params, client_session_timeout_seconds=120) as whatsapp_mcp_server:
            agent, session = await agent_factory.get_agent(
                chat_jid=message.chat_jid,
                mcp_server=whatsapp_mcp_server,
                model=model,
                instructions=Instruction
            )

            with trace("LeoWhatsappAssistant"):
                from dataclasses import asdict
                result = await Runner.run(agent, json.dumps(asdict(message)), session=session)

            print(f"[Agent Server] Result: {result.final_output}")
    
    return {"status": "success", "message": "Action taken", "received": data}


def handle_client(client_socket: socket.socket):
    """Handle a single client connection."""
    try:
        # Receive data from the client
        data = b""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            data += chunk
            # Check if we've received a complete JSON message
            try:
                json.loads(data.decode())
                break  # Valid JSON, we're done receiving
            except json.JSONDecodeError:
                continue  # Keep receiving
        
        if not data:
            return
        
        try:
            message_data = json.loads(data.decode())
        except json.JSONDecodeError:
            response = json.dumps({"error": "Invalid JSON"})
            client_socket.sendall(response.encode())
            return
        
        # Queue the task for background processing
        task_queue.put(message_data)
        print(f"[Agent Server] Task queued (queue size: {task_queue.qsize()})")
        
        # Send immediate response
        response = json.dumps({
            "status": "queued",
            "message": "Task queued for processing",
            "queue_size": task_queue.qsize()
        })
        client_socket.sendall(response.encode())
        
    except Exception as e:
        print(f"[Agent Server] Error handling client: {e}")
    finally:
        client_socket.close()


def run_server():
    """Start the Unix domain socket server."""
    # Remove existing socket file if it exists
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    
    # Create Unix domain socket
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(SOCKET_PATH)
    server_socket.listen(5)
    
    # Set socket permissions so other processes can connect
    os.chmod(SOCKET_PATH, 0o666)
    
    print(f"Unix domain socket Agent Server running at {SOCKET_PATH}")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            client_socket, _ = server_socket.accept()
            # Handle each client in a separate thread
            client_thread = threading.Thread(
                target=handle_client, 
                args=(client_socket,),
                daemon=True
            )
            client_thread.start()
    except KeyboardInterrupt:
        print("\nShutting down Agent Server...")
    finally:
        server_socket.close()
        # Clean up socket file
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)


if __name__ == "__main__":
    run_server()
