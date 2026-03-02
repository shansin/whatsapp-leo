"""Environment configuration and cached singletons for WhatsApp Leo agent."""

import os
import sys
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

# Ensure logging is configured before anything else
from logging_setup import logger  # noqa: F401

load_dotenv(override=True)

TZ = ZoneInfo("America/Los_Angeles")

# Add whatsapp-mcp-server to path for direct imports
WHATSAPP_MCP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "whatsapp-mcp",
    "whatsapp-mcp-server",
)
sys.path.insert(0, WHATSAPP_MCP_DIR)

# ── Instance & socket configuration ─────────────────────────────────────────
INSTANCE_GUID = os.getenv("INSTANCE_GUID", "default")
SOCKET_PATH = os.getenv(
    "AGENT_SOCKET_PATH", f"/tmp/whatsapp-leo-{INSTANCE_GUID}.sock"
)

# ── Model & agent configuration ─────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_AGENTS = int(os.getenv("MAX_AGENTS", "20"))
TTL_SECONDS = int(os.getenv("TTL_SECONDS", "1800"))

# ── Sender / access control ─────────────────────────────────────────────────
ALLOWED_SENDERS = [
    s.strip() for s in os.getenv("ALLOWED_SENDERS", "").split(",") if s.strip()
]
LEO_MENTION_ID = os.getenv("LEO_MENTION_ID", "@23833461416078")
IS_DEDICATED_NUMBER = os.getenv("IS_DEDICATED_NUMBER", "false").lower() == "true"

# ── Hooks ────────────────────────────────────────────────────────────────────
IS_HOOK_ENABLED = os.getenv("IS_HOOK_ENABLED", "false").lower() == "true"
HOOKS = [h.strip() for h in os.getenv("HOOKS", "").split(",") if h.strip()]

# Maximum message size to prevent memory exhaustion (10MB)
MAX_MESSAGE_SIZE = int(os.getenv("MAX_MESSAGE_SIZE", "10485760"))

# MCP Server Paths
WORKSPACE_MCP_PATH = os.getenv("WORKSPACE_MCP_PATH")

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


@asynccontextmanager
async def mcp_stack(is_privileged: bool = False):
    """Async context manager that starts and yields configured MCP servers.

    Always starts Brave search MCP. If is_privileged, also starts workspace
    MCP (if available) and Garmin MCP.
    """
    from contextlib import AsyncExitStack
    from agents.mcp import MCPServerStdio

    async with AsyncExitStack() as stack:
        brave = await stack.enter_async_context(
            MCPServerStdio(params=_brave_mcp_params, client_session_timeout_seconds=30)
        )
        servers = [brave]

        if is_privileged:
            if WORKSPACE_MCP_PATH and os.path.exists(WORKSPACE_MCP_PATH):
                ws = await stack.enter_async_context(
                    MCPServerStdio(
                        params=_workspace_mcp_params,
                        client_session_timeout_seconds=300,
                    )
                )
                servers.append(ws)
            elif WORKSPACE_MCP_PATH:
                logger.warning(f"Workspace MCP not found at {WORKSPACE_MCP_PATH}")

            garmin = await stack.enter_async_context(
                MCPServerStdio(
                    params=_garmin_mcp_params,
                    client_session_timeout_seconds=120,
                )
            )
            servers.append(garmin)

        yield servers
