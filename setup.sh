#!/bin/bash

# setup.sh — One-shot setup for WhatsApp Leo on a fresh machine
# Run: chmod +x setup.sh && ./setup.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# Prompt with default — empty input accepts the default
# Usage: ask "Prompt text" "default_value"
# Sets $REPLY to the answer
ask() {
    local prompt="$1" default="$2"
    if [ -n "$default" ]; then
        echo -en "  ${BOLD}$prompt${NC} [${default}]: "
    else
        echo -en "  ${BOLD}$prompt${NC}: "
    fi
    read -r REPLY
    REPLY="${REPLY:-$default}"
}

# Yes/no prompt — returns 0 for yes, 1 for no
# Usage: ask_yn "Do something?" "y"   (default yes)
ask_yn() {
    local prompt="$1" default="${2:-n}"
    local hint
    if [ "$default" = "y" ]; then hint="[Y/n]"; else hint="[y/N]"; fi
    echo -en "  ${BOLD}$prompt${NC} $hint: "
    read -r REPLY
    REPLY="${REPLY:-$default}"
    [[ "$REPLY" =~ ^[Yy] ]]
}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "================================================"
echo "  WhatsApp Leo — One-Shot Setup"
echo "================================================"
echo ""
echo "  This script will install all dependencies, build"
echo "  components, and configure Leo interactively."
echo "  You can re-run it safely — it skips completed steps."
echo "  Any value can be changed later in .env."
echo ""

# ── 1. Check system dependencies ─────────────────────────────────────────────

info "Checking system dependencies..."

MISSING=()

# Python 3.13+
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 13 ]; then
        ok "Python $PY_VER"
    else
        fail "Python $PY_VER found, but >=3.13 is required"
        MISSING+=("python>=3.13")
    fi
else
    fail "Python 3 not found"
    MISSING+=("python>=3.13")
fi

# uv
if command -v uv &>/dev/null; then
    ok "uv $(uv --version 2>/dev/null | head -1)"
else
    fail "uv not found"
    MISSING+=("uv")
fi

# Go 1.24+
if command -v go &>/dev/null; then
    GO_VER=$(go version | grep -oP 'go\K[0-9]+\.[0-9]+')
    ok "Go $GO_VER"
else
    fail "Go not found"
    MISSING+=("go>=1.24")
fi

# Node.js
if command -v node &>/dev/null; then
    ok "Node.js $(node --version)"
else
    fail "Node.js not found"
    MISSING+=("node")
fi

# npm
if command -v npm &>/dev/null; then
    ok "npm $(npm --version)"
else
    fail "npm not found"
    MISSING+=("npm")
fi

# Ollama (offer to install if missing)
if command -v ollama &>/dev/null; then
    ok "Ollama installed"
else
    warn "Ollama not found"
    if ask_yn "Install Ollama now?" "y"; then
        info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        if command -v ollama &>/dev/null; then
            ok "Ollama installed"
        else
            fail "Ollama installation failed"
            MISSING+=("ollama")
        fi
    else
        fail "Ollama not found"
        MISSING+=("ollama")
    fi
fi

echo ""

if [ ${#MISSING[@]} -gt 0 ]; then
    fail "Missing dependencies: ${MISSING[*]}"
    echo ""
    echo "  Quick install hints:"
    echo "    uv:      curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "    ollama:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "    node:    https://nodejs.org/ or use nvm"
    echo "    go:      https://go.dev/dl/"
    echo "    python:  https://www.python.org/downloads/ or use pyenv"
    echo ""
    echo "  Install the missing dependencies and re-run this script."
    exit 1
fi

ok "All system dependencies found"
echo ""

# ── 2. Install Python dependencies ───────────────────────────────────────────

info "Installing Python dependencies with uv..."
uv sync
ok "Python dependencies installed"
echo ""

# ── 3. Build Go WhatsApp bridge ──────────────────────────────────────────────

info "Building Go WhatsApp bridge..."
cd "$PROJECT_DIR/whatsapp-mcp/whatsapp-bridge"
go build -o whatsapp-bridge .
ok "Go bridge built"
cd "$PROJECT_DIR"
echo ""

# ── 4. Install Brave Search MCP ──────────────────────────────────────────────

info "Pre-fetching Brave Search MCP (npx will cache it)..."
npx -y @modelcontextprotocol/server-brave-search --help &>/dev/null || true
ok "Brave Search MCP cached"
echo ""

# ── 5. Set up Google Workspace MCP ───────────────────────────────────────────

WORKSPACE_DIR="$PROJECT_DIR/../linux.google-workspace-extension"

if [ -f "$WORKSPACE_DIR/dist/index.js" ]; then
    ok "Google Workspace MCP already installed at $WORKSPACE_DIR"
else
    info "Google Workspace MCP enables Calendar, Gmail, and Drive integration."
    if ask_yn "Download and install Google Workspace MCP?" "y"; then
        # Detect platform
        case "$(uname -s)" in
            Linux*)  PLATFORM="linux" ;;
            Darwin*) PLATFORM="darwin" ;;
            MINGW*|MSYS*|CYGWIN*) PLATFORM="win32" ;;
            *) PLATFORM="linux" ;;
        esac

        RELEASE_URL="https://github.com/gemini-cli-extensions/workspace/releases/latest/download/${PLATFORM}.google-workspace-extension.tar.gz"

        info "Downloading from $RELEASE_URL..."
        mkdir -p "$WORKSPACE_DIR"
        curl -fsSL "$RELEASE_URL" | tar xz -C "$WORKSPACE_DIR"

        if [ -f "$WORKSPACE_DIR/dist/index.js" ]; then
            # Install Node.js dependencies
            info "Installing Workspace MCP Node.js dependencies..."
            cd "$WORKSPACE_DIR"
            npm install
            cd "$PROJECT_DIR"
            ok "Google Workspace MCP installed"
        else
            fail "Download succeeded but dist/index.js not found — check $WORKSPACE_DIR"
        fi
    else
        warn "Skipping Google Workspace MCP (can set up later)"
    fi
fi

# Resolve to absolute path for .env
if [ -f "$WORKSPACE_DIR/dist/index.js" ]; then
    WORKSPACE_MCP_PATH_RESOLVED="$(cd "$WORKSPACE_DIR" && pwd)/dist/index.js"
else
    WORKSPACE_MCP_PATH_RESOLVED=""
fi
echo ""

# ── 6. Create store directory ─────────────────────────────────────────────────

mkdir -p "$PROJECT_DIR/store"
ok "store/ directory ready"
echo ""

# ── 8. Interactive .env configuration ─────────────────────────────────────────

echo "================================================"
echo "  Configure Leo"
echo "================================================"
echo ""
echo "  Press Enter to accept the [default] value."
echo "  All values can be changed later in .env."
echo ""

# Load existing .env (if any) and track which keys were defined there,
# so we only prompt for values that are missing.
ENV_FILE="$PROJECT_DIR/.env"
declare -A ENV_DEFINED=()
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)= ]]; then
            ENV_DEFINED[${BASH_REMATCH[1]}]=1
        fi
    done < "$ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    info "Found existing .env — keeping ${#ENV_DEFINED[@]} defined value(s); will only prompt for missing ones"
    echo ""
fi

# ask_unset ENV_VAR CFG_VAR "Prompt" "default"
# If ENV_VAR was defined in .env, copy its current value into CFG_VAR (no prompt).
# Otherwise prompt with the given default.
ask_unset() {
    local envvar="$1" cfgvar="$2" prompt="$3" default="$4"
    if [ -n "${ENV_DEFINED[$envvar]:-}" ]; then
        printf -v "$cfgvar" '%s' "${!envvar-}"
    else
        ask "$prompt" "$default"
        printf -v "$cfgvar" '%s' "$REPLY"
    fi
}

# ask_yn_unset ENV_VAR CFG_VAR "Prompt" "y|n"
ask_yn_unset() {
    local envvar="$1" cfgvar="$2" prompt="$3" default="$4"
    if [ -n "${ENV_DEFINED[$envvar]:-}" ]; then
        printf -v "$cfgvar" '%s' "${!envvar-}"
    else
        if ask_yn "$prompt" "$default"; then
            printf -v "$cfgvar" '%s' "true"
        else
            printf -v "$cfgvar" '%s' "false"
        fi
    fi
}

# -- Core --
echo -e "  ${CYAN}── Core ──${NC}"
ask_unset INSTANCE_GUID       CFG_INSTANCE_GUID       "Instance GUID (for running multiple instances)" "default"
ask_unset OLLAMA_BASE_URL     CFG_OLLAMA_BASE_URL     "Ollama base URL" "http://localhost:11434/v1"
ask_unset MODEL_NAME          CFG_MODEL_NAME          "Text model name" "qwen3.5:35b"
ask_unset VISION_MODEL_NAME   CFG_VISION_MODEL_NAME   "Vision model name" "gemma3:27b"
ask_unset MAX_IMAGE_DIMENSION CFG_MAX_IMAGE_DIMENSION "Max image dimension (px)" "1280"
ask_unset WHISPER_MODEL_SIZE  CFG_WHISPER_MODEL_SIZE  "Whisper model size (tiny/base/small/medium/large)" "medium"
ask_unset MAX_AGENTS          CFG_MAX_AGENTS          "Max cached agents" "20"
ask_unset TTL_SECONDS         CFG_TTL_SECONDS         "Agent cache TTL (seconds)" "1800"

echo ""

# -- Access control --
echo -e "  ${CYAN}── Access Control ──${NC}"
ask_yn_unset IS_DEDICATED_NUMBER CFG_IS_DEDICATED_NUMBER "Is this a dedicated phone number for Leo?" "y"
ask_unset    ALLOWED_SENDERS     CFG_ALLOWED_SENDERS     "Allowed sender phone numbers (comma-separated, e.g. 1234567890)" ""
ask_unset    LEO_MENTION_ID      CFG_LEO_MENTION_ID      "Leo mention ID (for group triggers, e.g. @12345)" ""

echo ""

# -- API keys --
echo -e "  ${CYAN}── API Keys ──${NC}"
ask_unset OPENAI_API_KEY CFG_OPENAI_API_KEY "OpenAI API key (required by SDK, can be a dummy value for Ollama)" "ollama"
ask_unset BRAVE_API_KEY  CFG_BRAVE_API_KEY  "Brave Search API key (get one at https://brave.com/search/api/)" ""
if [ -z "$CFG_BRAVE_API_KEY" ]; then
    warn "No Brave API key — web search won't work until you add it to .env"
fi

echo ""

# -- Hooks --
echo -e "  ${CYAN}── Hooks ──${NC}"
ask_yn_unset IS_HOOK_ENABLED CFG_IS_HOOK_ENABLED "Enable hooks (bidirectional pipes to external programs)?" "n"
if [ "$CFG_IS_HOOK_ENABLED" = "true" ]; then
    ask_unset HOOKS CFG_HOOKS "Hook names (comma-separated, e.g. claude,claude-session)" "claude,claude-session"
else
    CFG_HOOKS="${ENV_DEFINED[HOOKS]:+${HOOKS-}}"
fi

echo ""

# -- Reminders --
echo -e "  ${CYAN}── Reminders ──${NC}"
ask_unset REMINDER_POLL_INTERVAL CFG_REMINDER_POLL_INTERVAL "Reminder poll interval (seconds)" "30"

echo ""

# -- X/Twitter --
echo -e "  ${CYAN}── X (Twitter) ──${NC}"
ask_unset X_COOKIE_PATH CFG_X_COOKIE_PATH "X cookie file path" "/tmp/x_cookies.json"

echo ""

# -- Write .env --
info "Writing .env..."

cat > "$PROJECT_DIR/.env" << ENVEOF
# WhatsApp Leo configuration — generated by setup.sh

# Unique identifier for this instance (allows running multiple instances)
INSTANCE_GUID="${CFG_INSTANCE_GUID}"
OPENAI_AGENTS_DISABLE_TRACING=1
IS_DEDICATED_NUMBER=${CFG_IS_DEDICATED_NUMBER}
IS_TEST_MODE=false

OLLAMA_BASE_URL="${CFG_OLLAMA_BASE_URL}"
MODEL_NAME="${CFG_MODEL_NAME}"
VISION_MODEL_NAME="${CFG_VISION_MODEL_NAME}"
MAX_IMAGE_DIMENSION="${CFG_MAX_IMAGE_DIMENSION}"
WHISPER_MODEL_SIZE="${CFG_WHISPER_MODEL_SIZE}"

MAX_AGENTS="${CFG_MAX_AGENTS}"
TTL_SECONDS="${CFG_TTL_SECONDS}"
ALLOWED_SENDERS=${CFG_ALLOWED_SENDERS}
LEO_MENTION_ID=${CFG_LEO_MENTION_ID}

OPENAI_API_KEY=${CFG_OPENAI_API_KEY}
BRAVE_API_KEY=${CFG_BRAVE_API_KEY}

REMINDER_POLL_INTERVAL="${CFG_REMINDER_POLL_INTERVAL}"

IS_HOOK_ENABLED=${CFG_IS_HOOK_ENABLED}
HOOKS="${CFG_HOOKS}"

# Google Workspace MCP (Calendar, Gmail, Drive)
# Auth: on first use, opens a browser for Google OAuth consent
WORKSPACE_MCP_PATH="${WORKSPACE_MCP_PATH_RESOLVED}"

# X (Twitter) MCP — run setup to authenticate:
#   uv run python x-mcp/x-mcp-server/setup.py
X_COOKIE_PATH="${CFG_X_COOKIE_PATH}"
ENVEOF

ok ".env written"
echo ""

# ── 9. Pull Ollama models ────────────────────────────────────────────────────

echo "================================================"
echo "  Pull Ollama Models"
echo "================================================"
echo ""

# Skip pulling when the configured endpoint isn't on this machine —
# the remote daemon is responsible for having the models.
OLLAMA_HOST_FROM_URL="$(echo "$CFG_OLLAMA_BASE_URL" | sed -E 's#^[a-z]+://([^:/]+).*#\1#')"
case "$OLLAMA_HOST_FROM_URL" in
    localhost|127.0.0.1|::1|"")
        OLLAMA_IS_LOCAL=1 ;;
    *)
        OLLAMA_IS_LOCAL=0 ;;
esac

if [ "$OLLAMA_IS_LOCAL" = "0" ]; then
    info "Ollama endpoint is remote ($OLLAMA_HOST_FROM_URL) — skipping local model pull"
    info "Ensure these models exist on the remote host:"
    echo "    $CFG_MODEL_NAME"
    [ "$CFG_VISION_MODEL_NAME" != "$CFG_MODEL_NAME" ] && echo "    $CFG_VISION_MODEL_NAME"
    echo ""
else

info "Checking Ollama service..."
if ! ollama list &>/dev/null; then
    warn "Ollama is not running. Starting it..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    sleep 3
    if ollama list &>/dev/null; then
        ok "Ollama started (PID: $OLLAMA_PID)"
    else
        warn "Could not start Ollama. Start it manually with: ollama serve"
        warn "Then pull models with:"
        echo "    ollama pull $CFG_MODEL_NAME"
        echo "    ollama pull $CFG_VISION_MODEL_NAME"
        echo ""
        OLLAMA_PID=""
    fi
else
    ok "Ollama is running"
    OLLAMA_PID=""
fi

if ollama list &>/dev/null; then
    info "Pulling text model ($CFG_MODEL_NAME)... (this may take a while)"
    ollama pull "$CFG_MODEL_NAME"
    ok "$CFG_MODEL_NAME ready"

    if [ "$CFG_VISION_MODEL_NAME" != "$CFG_MODEL_NAME" ]; then
        info "Pulling vision model ($CFG_VISION_MODEL_NAME)... (this may take a while)"
        ollama pull "$CFG_VISION_MODEL_NAME"
        ok "$CFG_VISION_MODEL_NAME ready"
    fi

    # Stop Ollama if we started it
    if [ -n "$OLLAMA_PID" ]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
        info "Stopped temporary Ollama process"
    fi
fi

fi  # end OLLAMA_IS_LOCAL guard
echo ""

# ── 10. Authenticate MCP services ────────────────────────────────────────────

echo "================================================"
echo "  Authenticate Services"
echo "================================================"
echo ""

# -- X/Twitter auth --
echo -e "  ${CYAN}── X (Twitter) ──${NC}"
echo "  Reads cookies from your Firefox or Chrome browser."
echo "  You must be logged in to x.com in the browser first."
echo ""
if ask_yn "Authenticate X/Twitter now?" "n"; then
    echo ""
    uv run python x-mcp/x-mcp-server/setup.py || warn "X auth failed — you can retry later with: uv run python x-mcp/x-mcp-server/setup.py"
    echo ""
else
    info "Skipping — authenticate later with: uv run python x-mcp/x-mcp-server/setup.py"
fi
echo ""

# -- Garmin auth --
echo -e "  ${CYAN}── Garmin Connect ──${NC}"
echo "  Authenticates with your Garmin account and stores OAuth"
echo "  tokens in ~/.garminconnect/ so Leo can pull fitness data"
echo "  without prompting at runtime. Supports MFA."
echo ""
if [ -d "$HOME/.garminconnect" ] && [ -n "$(ls -A "$HOME/.garminconnect" 2>/dev/null)" ]; then
    ok "Garmin tokens already present in ~/.garminconnect/"
    if ask_yn "Re-authenticate Garmin?" "n"; then
        DO_GARMIN_AUTH=1
    else
        DO_GARMIN_AUTH=0
    fi
else
    if ask_yn "Authenticate Garmin now?" "y"; then
        DO_GARMIN_AUTH=1
    else
        DO_GARMIN_AUTH=0
        info "Skipping — Garmin will prompt on first use at runtime"
    fi
fi

if [ "$DO_GARMIN_AUTH" = "1" ]; then
    info "Launching garmin-mcp-auth (interactive — will prompt for email, password, and MFA)..."
    info "Tokens will be saved to ~/.garminconnect/ in the format the MCP expects."
    echo ""
    if uvx --refresh --from "git+https://github.com/Taxuspt/garmin_mcp" garmin-mcp-auth; then
        if [ -d "$HOME/.garminconnect" ] && [ -n "$(ls -A "$HOME/.garminconnect" 2>/dev/null)" ]; then
            ok "Garmin authenticated — tokens stored in ~/.garminconnect/"
        else
            warn "garmin-mcp-auth exited 0 but ~/.garminconnect/ is empty"
        fi
    else
        warn "Garmin auth failed — Leo will prompt on first use (or re-run: uvx --from git+https://github.com/Taxuspt/garmin_mcp garmin-mcp-auth)"
    fi
fi
echo ""

# -- Google Workspace auth --
if [ -n "$WORKSPACE_MCP_PATH_RESOLVED" ]; then
    echo -e "  ${CYAN}── Google Workspace ──${NC}"
    echo "  On first use, the Workspace MCP will open your browser"
    echo "  for Google OAuth consent. Tokens are stored locally in:"
    echo "    $(dirname "$WORKSPACE_MCP_PATH_RESOLVED")/gemini-cli-workspace-token.json"
    echo ""
    ok "No action needed now — Google will prompt on first use"
    echo ""
fi

# -- WhatsApp auth --
echo -e "  ${CYAN}── WhatsApp ──${NC}"
echo "  The bridge connects via QR code — no phone number or SIM needed."
echo "  It will print a QR code; scan it with:"
echo "    WhatsApp > Settings > Linked Devices > Link a Device"
echo "  The session persists in SQLite so you only scan once."
echo ""
if ask_yn "Link WhatsApp now?" "y"; then
    echo ""
    info "Starting WhatsApp bridge for QR code pairing..."
    info "Once paired, press Ctrl+C to continue setup."
    echo ""
    cd "$PROJECT_DIR/whatsapp-mcp/whatsapp-bridge"
    ./whatsapp-bridge || true
    cd "$PROJECT_DIR"
    echo ""
    ok "WhatsApp pairing complete"
else
    info "Skipping — the QR code will appear when you run ./start_services.sh"
fi
echo ""

# ── 11. Summary ───────────────────────────────────────────────────────────────

echo "================================================"
echo -e "  ${GREEN}Setup complete!${NC}"
echo "================================================"
echo ""
echo "  Start Leo:"
echo "    ./start_services.sh"
echo ""
echo "  Test mode (Gradio UI, no WhatsApp):"
echo "    IS_TEST_MODE=true ./start_services.sh"
echo ""
echo "  Edit configuration anytime:"
echo "    \$EDITOR .env"
echo ""
