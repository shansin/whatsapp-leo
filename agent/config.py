"""Environment configuration and cached singletons for WhatsApp Leo agent."""

import os
import shutil
import sys
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, ModelSettings, set_tracing_disabled

# Ensure logging is configured before anything else
from logging_setup import logger  # noqa: F401

load_dotenv(override=True)

# ── Tracing ──────────────────────────────────────────────────────────────────
# The Agents SDK batches run traces (message content included) and POSTs them
# to api.openai.com. We talk to a local Ollama with a dummy key, so that is a
# privacy leak plus a stream of background 401 retries. Opt in explicitly via
# AGENTS_TRACING_ENABLED=true if you ever point this at a real trace backend.
TRACING_ENABLED = os.getenv("AGENTS_TRACING_ENABLED", "false").lower() == "true"
set_tracing_disabled(not TRACING_ENABLED)
if TRACING_ENABLED:
    logger.info("Agents SDK tracing export is ENABLED")

# Instance default timezone. Individual users can override it with `#tz`
# (see user_prefs); this is the fallback and what schedulers use for logging.
TZ = ZoneInfo(os.getenv("DEFAULT_TZ", "America/Los_Angeles"))

# Add whatsapp-mcp-server to path for direct imports
WHATSAPP_MCP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "whatsapp-mcp",
    "whatsapp-mcp-server",
)
sys.path.insert(0, WHATSAPP_MCP_DIR)

# ── Shared storage paths ────────────────────────────────────────────────────
# One value for every component. The Go bridge writes messages.db relative to
# its own working directory, so this must agree with it in all launch modes —
# start_services.sh exports MESSAGES_DB_PATH to keep them in sync.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.getenv("STORE_DIR") or os.path.join(_REPO_ROOT, "store")
MESSAGES_DB_PATH = os.getenv("MESSAGES_DB_PATH") or os.path.join(
    STORE_DIR, "messages.db"
)

# ── Instance & socket configuration ─────────────────────────────────────────
INSTANCE_GUID = os.getenv("INSTANCE_GUID", "default")


def runtime_dir() -> str:
    """Directory for Unix sockets.

    $XDG_RUNTIME_DIR is per-user and mode 700; /tmp is world-writable, which
    matters because anything that can write to the agent socket can spoof a
    phone_number from ALLOWED_SENDERS and get privileged access.
    Mirrored in the Go bridge (getRuntimeDir) and start_services.sh.
    """
    xdg = os.getenv("XDG_RUNTIME_DIR", "").strip("\"'")
    if xdg and os.path.isdir(xdg) and os.access(xdg, os.W_OK):
        return xdg
    return "/tmp"


SOCKET_PATH = os.getenv("AGENT_SOCKET_PATH") or os.path.join(
    runtime_dir(), f"whatsapp-leo-{INSTANCE_GUID}.sock"
)

# ── Model & agent configuration ─────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_AGENTS = int(os.getenv("MAX_AGENTS", "20"))
TTL_SECONDS = int(os.getenv("TTL_SECONDS", "1800"))

# Backup Ollama (degraded fallback when primary fails or stalls)
OLLAMA_BACKUP_BASE_URL = os.getenv("OLLAMA_BACKUP_BASE_URL") or None
OLLAMA_BACKUP_MODEL_NAME = os.getenv("OLLAMA_BACKUP_MODEL_NAME") or None
OLLAMA_BACKUP_VISION_MODEL_NAME = (
    os.getenv("OLLAMA_BACKUP_VISION_MODEL_NAME") or OLLAMA_BACKUP_MODEL_NAME
)
OLLAMA_PRIMARY_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_PRIMARY_TIMEOUT_SECONDS", "120"))
OLLAMA_FALLBACK_STICKY_SECONDS = float(os.getenv("OLLAMA_FALLBACK_STICKY_SECONDS", "300"))

# ── Sender / access control ─────────────────────────────────────────────────
ALLOWED_SENDERS = [
    s.strip() for s in os.getenv("ALLOWED_SENDERS", "").split(",") if s.strip()
]
LEO_MENTION_ID = os.getenv("LEO_MENTION_ID", "@23833461416078")
IS_DEDICATED_NUMBER = os.getenv("IS_DEDICATED_NUMBER", "false").lower() == "true"

# ── Presence (typing indicator) ─────────────────────────────────────────────
# WhatsApp clients drop the "typing…" bubble after ~10s, so it has to be
# refreshed while a slow local model is running.
PRESENCE_ENABLED = os.getenv("PRESENCE_ENABLED", "true").lower() == "true"
PRESENCE_REFRESH_SECONDS = float(os.getenv("PRESENCE_REFRESH_SECONDS", "8"))

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

_primary_text_model = OpenAIChatCompletionsModel(
    model=MODEL_NAME, openai_client=_openai_client
)

# ── Vision model (for image processing) ────────────────────────────────────
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "gemma3:27b")
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1280"))
_primary_vision_model = OpenAIChatCompletionsModel(
    model=VISION_MODEL_NAME, openai_client=_openai_client
)

# ── Backup wiring ────────────────────────────────────────────────────────────
from fallback_model import FallbackModel, FallbackRouter

_fallback_enabled = bool(OLLAMA_BACKUP_BASE_URL and OLLAMA_BACKUP_MODEL_NAME)
_fallback_router = FallbackRouter(sticky_seconds=OLLAMA_FALLBACK_STICKY_SECONDS)

if _fallback_enabled:
    _backup_openai_client = AsyncOpenAI(
        base_url=OLLAMA_BACKUP_BASE_URL, api_key="ollama"
    )
    _backup_text_model = OpenAIChatCompletionsModel(
        model=OLLAMA_BACKUP_MODEL_NAME, openai_client=_backup_openai_client
    )
    _backup_vision_model = OpenAIChatCompletionsModel(
        model=OLLAMA_BACKUP_VISION_MODEL_NAME, openai_client=_backup_openai_client
    )
    logger.info(
        f"Ollama fallback enabled: backup={OLLAMA_BACKUP_BASE_URL} "
        f"model={OLLAMA_BACKUP_MODEL_NAME} "
        f"timeout={OLLAMA_PRIMARY_TIMEOUT_SECONDS}s "
        f"sticky={OLLAMA_FALLBACK_STICKY_SECONDS}s"
    )
else:
    _backup_text_model = None
    _backup_vision_model = None

_cached_model = FallbackModel(
    primary=_primary_text_model,
    backup=_backup_text_model,
    router=_fallback_router,
    primary_timeout_seconds=OLLAMA_PRIMARY_TIMEOUT_SECONDS,
)
_cached_vision_model = FallbackModel(
    primary=_primary_vision_model,
    backup=_backup_vision_model,
    router=_fallback_router,
    primary_timeout_seconds=OLLAMA_PRIMARY_TIMEOUT_SECONDS,
)

# ── Audio transcription (faster-whisper) ───────────────────────────────────
# distil-medium.en is several times faster than `medium` on CPU at comparable
# quality for English. Set WHISPER_MODEL_SIZE=medium for multilingual notes.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "distil-medium.en")
# beam_size=1 (greedy) is much faster than 5 and rarely worse on short voice
# notes; VAD skips silence, which is most of the cost on a rambling recording.
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
# Load the model at startup instead of on the first voice note, which
# otherwise pays download + load time while the user waits.
WHISPER_PREWARM = os.getenv("WHISPER_PREWARM", "true").lower() == "true"

_whisper_model = None


def get_whisper_model():
    """Load the faster-whisper model, reusing it across calls."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="auto")
        logger.info(f"Loaded faster-whisper model: {WHISPER_MODEL_SIZE}")
    return _whisper_model


def prewarm_whisper() -> None:
    """Warm the transcription model. Safe to call from a background task."""
    if not WHISPER_PREWARM:
        return
    try:
        get_whisper_model()
    except Exception as e:
        # Not fatal: the next voice note retries the lazy path.
        logger.warning(f"Could not pre-warm faster-whisper: {e}")


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
# Garmin: prefer a locally installed `garmin-mcp` (setup.sh runs
# `uv tool install`) so startup never depends on GitHub being reachable.
# Falls back to uvx against a pinned commit, which is at least reproducible.
# garmin_mcp declares an unpinned `mcp>=1.28.1` dependency and imports
# mcp.server.fastmcp, which the mcp 2.0.0 release removed, hence `mcp<2.0.0`.
GARMIN_MCP_REF = os.getenv(
    "GARMIN_MCP_REF", "a16f05770291fd25e35c58db17f9e77e70facbc2"
)
GARMIN_MCP_COMMAND = os.getenv("GARMIN_MCP_COMMAND") or shutil.which("garmin-mcp")

if GARMIN_MCP_COMMAND:
    _garmin_mcp_params = {"command": GARMIN_MCP_COMMAND, "args": []}
else:
    _garmin_mcp_params = {
        "command": "uvx",
        "args": [
            "--with", "mcp<2.0.0",
            f"git+https://github.com/Taxuspt/garmin_mcp@{GARMIN_MCP_REF}",
        ],
    }
_x_mcp_params = {
    "command": "uv",
    "args": ["run", "python", X_MCP_PATH],
    "env": _shared_env,
}
# Chrome locks its user-data-dir: a desktop Chrome pointed at the same profile
# and Leo cannot run at the same time. Keep this dir dedicated to Leo, and log
# it into the sites Leo should reach by running the MCP headed once by hand
# (see the Playwright section of .env_example).
_playwright_user_data_dir = os.path.expanduser(
    os.getenv(
        "PLAYWRIGHT_USER_DATA_DIR",
        "~/.cache/whatsapp-leo/playwright-profile",
    )
)
_playwright_browser = os.getenv("PLAYWRIGHT_BROWSER", "chrome")
# Headless by default: Leo normally runs as a systemd user unit, which sets no
# DISPLAY. Set false only where a display (or Xvfb) is actually available.
_playwright_headless = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true"
_playwright_viewport = os.getenv("PLAYWRIGHT_VIEWPORT", "1280x800")
# Recent kernels restrict unprivileged user namespaces via AppArmor, which
# breaks Chrome's own sandbox under a user unit. Turn this off where the
# sandbox works — it is a real defence layer, just an unavailable one here.
_playwright_no_sandbox = os.getenv("PLAYWRIGHT_NO_SANDBOX", "true").lower() == "true"

_playwright_args = [
    "@playwright/mcp@0.0.70",
    "--browser", _playwright_browser,
    "--user-data-dir", _playwright_user_data_dir,
    "--viewport-size", _playwright_viewport,
]
if _playwright_headless:
    _playwright_args.append("--headless")
if _playwright_no_sandbox:
    _playwright_args.append("--no-sandbox")

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
# The servers themselves are started once per process by mcp_pool.MCPPool.
PRIVILEGED_SERVERS = ["workspace", "garmin", "x", "playwright"]
ALWAYS_SERVERS = ["brave"]
