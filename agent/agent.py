#!/usr/bin/env python3
"""Simple HTTP server that listens on port 8081 for POST requests."""

from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from dotenv import load_dotenv
from openai import OpenAI
import os
from agents import Agent, Runner, trace, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput
from pydantic import BaseModel
from openai import AsyncOpenAI
from agents import function_tool
from typing import Dict
import asyncio
from agents.mcp import MCPServerStdio

load_dotenv(override=True)

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

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


async def take_action(data: dict) -> dict:
    """
    Process the incoming request data.
    
    Args:
        data: The JSON payload from the POST request
        
    Returns:
        A response dictionary
    """
    # Parse incoming data into ReceivedMessage structure
    message = ReceivedMessage.from_dict(data)
    print(f"Received message: {message}")

    if("#leo" in message.content.lower()):
        print("Leo mentioned!")
        client = AsyncOpenAI(base_url= ollama_base_url, api_key="ollama")
        model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=client)
        
        from datetime import datetime
        from zoneinfo import ZoneInfo
    
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
        current_time = now.strftime("%I:%M %p PST, %B %d, %Y")
    
        Instruction = f"Date and time right now is {current_time}. You are a helpful assistant called #leo. You are able interact with whatsapp though whatsapp mcp server. Please respond to user's query through whatsapp mcp server in helpful and concise manner. Do not ask followup questions. Just do the task and finish."

        #Push MCP Server instantiation
        whatsapp_mcp_server_params = {"command": "uv", "args": [
        "--directory",
        "/home/shant/git_linux/whatsapp-mcp/whatsapp-mcp-server",
        "run", 
        "main.py"]
        }
        async with MCPServerStdio(params=whatsapp_mcp_server_params, client_session_timeout_seconds=120) as whatsapp_mcp_server:
            agent = Agent(name="Leo", 
            instructions=Instruction,
            mcp_servers=[whatsapp_mcp_server],
            model=model)

            with trace("Leo"):
                from dataclasses import asdict
                result = await Runner.run(agent, json.dumps(asdict(message)))
            print(result.final_output)

    
    return {"status": "success", "message": "Action taken", "received": data}


class RequestHandler(BaseHTTPRequestHandler):
    """Handle incoming HTTP requests."""
    
    def do_POST(self):
        """Handle POST requests by calling take_action."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return
        
        # Call the take_action function (async)
        result = asyncio.run(take_action(data))
        
        # Send response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    
    def log_message(self, format, *args):
        """Override to customize logging."""
        print(f"[{self.log_date_time_string()}] {args[0]}")


def run_server(port: int = 8081):
    """Start the HTTP server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Server running on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    run_server()
