"""Environment configuration and cached singletons for WhatsApp Leo agent."""

import os
import sys
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, ModelSettings

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
CONTEXT_SIZE = os.getenv("CONTEXT_SIZE")

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
X_MCP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "x-mcp",
    "x-mcp-server",
    "main.py",
)

# ── Cached singletons (avoid re-creation per message) ───────────────────────
_openai_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

_model_settings = ModelSettings()
if CONTEXT_SIZE:
    _model_settings.extra_body = {"num_ctx": int(CONTEXT_SIZE)}

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
_x_mcp_params = {
    "command": "uv",
    "args": ["run", "python", X_MCP_PATH],
    "env": _shared_env,
}

# ── MCP registry ─────────────────────────────────────────────────────────────
# Single source of truth for all configured MCP servers.
# Used by mcp_stack (runtime) and update_tools_config.py (tool discovery).
# Add new servers here when you add them to mcp_stack below.
MCP_REGISTRY: dict[str, dict] = {
    "workspace": {"params": _workspace_mcp_params, "timeout": 300},
    "garmin":    {"params": _garmin_mcp_params,    "timeout": 120},
    "x":         {"params": _x_mcp_params,         "timeout": 60},
    "brave":     {"params": _brave_mcp_params,     "timeout": 30},
}


@asynccontextmanager
async def mcp_stack(is_privileged: bool = False):
    """Async context manager that starts and yields configured MCP servers.

    Always starts Brave search MCP. If is_privileged, also starts workspace
    MCP (if available) and Garmin MCP.

    Tool exposure is controlled by tools_config.py — edit the allowlists there
    to change which tools each server exposes to the LLM.
    """
    from contextlib import AsyncExitStack
    from agents.mcp import MCPServerStdio
    from tools_config import make_tool_filter

    async with AsyncExitStack() as stack:
        servers = []

        if is_privileged:
            if WORKSPACE_MCP_PATH and os.path.exists(WORKSPACE_MCP_PATH):
                ws = await stack.enter_async_context(
                    MCPServerStdio(
                        params=_workspace_mcp_params,
                        client_session_timeout_seconds=300,
                        tool_filter=make_tool_filter("workspace"),
                        cache_tools_list=True,
                    )
                )
                servers.append(ws)
            elif WORKSPACE_MCP_PATH:
                logger.warning(f"Workspace MCP not found at {WORKSPACE_MCP_PATH}")

            garmin = await stack.enter_async_context(
                MCPServerStdio(
                    params=_garmin_mcp_params,
                    client_session_timeout_seconds=120,
                    tool_filter=make_tool_filter("garmin"),
                    cache_tools_list=True,
                )
            )
            servers.append(garmin)

            x = await stack.enter_async_context(
                MCPServerStdio(
                    params=_x_mcp_params,
                    client_session_timeout_seconds=60,
                    tool_filter=make_tool_filter("x"),
                    cache_tools_list=True,
                )
            )
            servers.append(x)

        # Brave comes last so the model's recency bias works in our favor
        # (with many tools from workspace+garmin, models tend to ignore early-listed tools)
        brave = await stack.enter_async_context(
            MCPServerStdio(
                params=_brave_mcp_params,
                client_session_timeout_seconds=30,
                tool_filter=make_tool_filter("brave"),
                cache_tools_list=True,
            )
        )
        servers.append(brave)

        yield servers
