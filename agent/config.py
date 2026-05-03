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

# ── Sender / access control ─────────────────────────────────────────────────
ALLOWED_SENDERS = [
    s.strip() for s in os.getenv("ALLOWED_SENDERS", "").split(",") if s.strip()
]
LEO_MENTION_ID = os.getenv("LEO_MENTION_ID", "@23833461416078")
IS_DEDICATED_NUMBER = os.getenv("IS_DEDICATED_NUMBER", "false").lower() == "true"

# ── Hooks ────────────────────────────────────────────────────────────────────
IS_HOOK_ENABLED = os.getenv("IS_HOOK_ENABLED", "false").lower() == "true"
HOOKS = [h.strip() for h in os.getenv("HOOKS", "").split(",") if h.strip()]
PLAYWRIGHT_ENABLED = os.getenv("PLAYWRIGHT_ENABLED", "false").lower() == "true"

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

_cached_model = OpenAIChatCompletionsModel(
    model=MODEL_NAME, openai_client=_openai_client
)

# ── Vision model (for image processing) ────────────────────────────────────
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "gemma3:27b")
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1280"))
_cached_vision_model = OpenAIChatCompletionsModel(
    model=VISION_MODEL_NAME, openai_client=_openai_client
)

# ── Audio transcription (faster-whisper, lazy-loaded) ──────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
_whisper_model = None


def get_whisper_model():
    """Lazy-load the faster-whisper model on first use."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="auto")
        logger.info(f"Loaded faster-whisper model: {WHISPER_MODEL_SIZE}")
    return _whisper_model


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
_playwright_user_data_dir = os.path.expanduser(
    os.getenv(
        "PLAYWRIGHT_USER_DATA_DIR",
        "~/.cache/whatsapp-leo/playwright-profile",
    )
)
_playwright_browser = os.getenv("PLAYWRIGHT_BROWSER", "chrome")
_playwright_headless = os.getenv("PLAYWRIGHT_HEADLESS", "false").lower() == "true"
_playwright_viewport = os.getenv("PLAYWRIGHT_VIEWPORT", "1280x800")

_playwright_args = [
    "@playwright/mcp@0.0.70",
    "--browser", _playwright_browser,
    "--user-data-dir", _playwright_user_data_dir,
    "--viewport-size", _playwright_viewport,
]
if _playwright_headless:
    _playwright_args.append("--headless")

_playwright_mcp_params = {
    "command": "npx",
    "args": _playwright_args,
}

# ── MCP registry ─────────────────────────────────────────────────────────────
# Single source of truth for all configured MCP servers.
# Used by mcp_stack (runtime) and update_tools_config.py (tool discovery).
#
# Fields:
#   params   — MCPServerStdio constructor params (command, args, env)
#   timeout  — client_session_timeout_seconds
#   gate     — optional callable; server is skipped when it returns False
#   privileged — if True, only started for allowed senders (default True)
MCP_REGISTRY: dict[str, dict] = {
    "workspace": {
        "params": _workspace_mcp_params, "timeout": 300,
        "gate": lambda: WORKSPACE_MCP_PATH and os.path.exists(WORKSPACE_MCP_PATH),
    },
    "garmin":     {"params": _garmin_mcp_params,     "timeout": 120},
    "x":          {"params": _x_mcp_params,          "timeout": 60},
    "playwright": {"params": _playwright_mcp_params, "timeout": 120,
                   "gate": lambda: PLAYWRIGHT_ENABLED},
    "brave":      {"params": _brave_mcp_params,      "timeout": 30, "privileged": False},
}

# Servers to start in order. Brave last so the model's recency bias keeps
# web-search tools visible even with many tools from other servers.
_PRIVILEGED_SERVERS = ["workspace", "garmin", "x", "playwright"]
_ALWAYS_SERVERS = ["brave"]


@asynccontextmanager
async def mcp_stack(is_privileged: bool = False):
    """Async context manager that starts and yields configured MCP servers.

    Iterates MCP_REGISTRY, respecting each entry's ``privileged`` flag and
    optional ``gate`` callable.  Tool exposure is controlled by tools_config.py.
    """
    from contextlib import AsyncExitStack
    from agents.mcp import MCPServerStdio
    from tools_config import make_tool_filter

    async with AsyncExitStack() as stack:
        servers = []

        order = (_PRIVILEGED_SERVERS if is_privileged else []) + _ALWAYS_SERVERS
        for name in order:
            entry = MCP_REGISTRY[name]
            gate = entry.get("gate")
            if gate and not gate():
                if name == "workspace" and WORKSPACE_MCP_PATH:
                    logger.warning(f"Workspace MCP not found at {WORKSPACE_MCP_PATH}")
                continue

            srv = await stack.enter_async_context(
                MCPServerStdio(
                    params=entry["params"],
                    client_session_timeout_seconds=entry["timeout"],
                    tool_filter=make_tool_filter(name),
                    cache_tools_list=True,
                )
            )
            servers.append(srv)

        yield servers
