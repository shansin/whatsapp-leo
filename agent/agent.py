#!/usr/bin/env python3
"""Unix domain socket server for receiving WhatsApp messages from Go bridge."""

from dataclasses import dataclass
import json
from dotenv import load_dotenv
from openai import OpenAI
import os
import socket
from agents import Agent, Runner, trace, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput
from pydantic import BaseModel
from openai import AsyncOpenAI
from agents import function_tool
from typing import Dict
import asyncio
from agents.mcp import MCPServerStdio
import threading
import queue

# Socket path for Unix domain socket
SOCKET_PATH = "/tmp/whatsapp-leo.sock"

load_dotenv(override=True)

ollama_base_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("MODEL_NAME")

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
        client = AsyncOpenAI(base_url= ollama_base_url, api_key="ollama")
        model = OpenAIChatCompletionsModel(model=model_name, openai_client=client)
        
        from datetime import datetime
        from zoneinfo import ZoneInfo
    
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
        current_time = now.strftime("%I:%M %p PST, %B %d, %Y")
    
        Instruction = f"Date and time right now is {current_time}. You are a helpful assistant called #leo. You are able interact with whatsapp though whatsapp mcp server. Please respond to user's message through whatsapp mcp server in helpful and concise manner. Do not ask followup questions. Just do the task and finish."

        #Push MCP Server instantiation
        whatsapp_mcp_server_params = {"command": "uv", "args": [
        "--directory",
        "/home/shant/git_linux/whatsapp-leo/whatsapp-mcp/whatsapp-mcp-server",
        "run", 
        "main.py"]
        }
        async with MCPServerStdio(params=whatsapp_mcp_server_params, client_session_timeout_seconds=120) as whatsapp_mcp_server:
            agent = Agent(name="Leo", 
                instructions=Instruction,
                mcp_servers=[whatsapp_mcp_server],
                model=model)

            with trace("LeoWhatsappAssistant"):
                from dataclasses import asdict
                result = await Runner.run(agent, json.dumps(asdict(message)))

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
