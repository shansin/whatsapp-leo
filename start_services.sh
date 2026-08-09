#!/bin/bash

# Start Services Script
# Starts both the Go WhatsApp bridge server and the Python agent server
# Communication uses Unix domain sockets with configurable paths via INSTANCE_GUID

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables from .env file if it exists
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a  # automatically export all variables
    source "$PROJECT_DIR/.env"
    set +a
fi

# Print hook FIFO paths if enabled
if [ "${IS_HOOK_ENABLED:-false}" = "true" ] && [ -n "$HOOKS" ]; then
    echo ""
    echo "🪝 Hooks enabled. FIFOs:"
    IFS=',' read -ra HOOK_NAMES <<< "$HOOKS"
    for hook in "${HOOK_NAMES[@]}"; do
        hook=$(echo "$hook" | xargs)  # trim whitespace
        echo "  ${hook}:"
        echo "    read  ← /tmp/whatsapp-leo-hook-${INSTANCE_GUID:-default}-${hook}-in.fifo"
        echo "    write → /tmp/whatsapp-leo-hook-${INSTANCE_GUID:-default}-${hook}-out.fifo"
    done
    echo ""
fi

# Get instance GUID (default to "default" if not set)
INSTANCE_GUID="${INSTANCE_GUID:-default}"

# Sockets live in $XDG_RUNTIME_DIR (per-user, mode 700) rather than
# world-writable /tmp — anything that can write to the agent socket can spoof an
# allowed sender. Mirrored in agent/config.py and the Go bridge.
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
[ -d "$RUNTIME_DIR" ] && [ -w "$RUNTIME_DIR" ] || RUNTIME_DIR="/tmp"

# Set socket paths based on env vars or defaults with INSTANCE_GUID.
# Exported so the agent and the bridge cannot disagree about them.
export AGENT_SOCKET_PATH="${AGENT_SOCKET_PATH:-$RUNTIME_DIR/whatsapp-leo-${INSTANCE_GUID}.sock}"
export BRIDGE_SOCKET_PATH="${BRIDGE_SOCKET_PATH:-$RUNTIME_DIR/whatsapp-bridge-${INSTANCE_GUID}.sock}"

# One messages.db for the bridge (writer) and the Python side (readers).
export MESSAGES_DB_PATH="${MESSAGES_DB_PATH:-$PROJECT_DIR/store/messages.db}"

echo "Starting WhatsApp Leo services (Instance: $INSTANCE_GUID)..."

# Flag to prevent cleanup from running twice
CLEANUP_DONE=0

# Function to cleanup background processes on exit
cleanup() {
    if [ $CLEANUP_DONE -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1
    echo ""
    echo "Shutting down services..."
    kill ${GO_PID:-} ${AGENT_PID:-} 2>/dev/null || true
    wait ${GO_PID:-} ${AGENT_PID:-} 2>/dev/null || true
    # Clean up socket files
    rm -f "$AGENT_SOCKET_PATH" "$BRIDGE_SOCKET_PATH"
    echo "Services stopped."
}

trap cleanup EXIT INT TERM

# Start the Python agent server
echo "[1/2] Starting Python agent server..."
cd "$PROJECT_DIR"
uv run python agent/agent.py &
AGENT_PID=$!
echo "      Agent server started (PID: $AGENT_PID)"

# Wait for the agent socket to actually appear rather than guessing with
# `sleep 2` — the bridge drops messages if it starts first.
echo "      Waiting for agent socket..."
for _ in $(seq 1 60); do
    [ -S "$AGENT_SOCKET_PATH" ] && break
    if ! kill -0 "$AGENT_PID" 2>/dev/null; then
        echo "      Agent exited before creating its socket; aborting." >&2
        exit 1
    fi
    sleep 0.5
done
if [ ! -S "$AGENT_SOCKET_PATH" ]; then
    echo "      Agent socket did not appear at $AGENT_SOCKET_PATH; aborting." >&2
    exit 1
fi
echo "      Agent socket ready"

# Start the Go WhatsApp bridge server
if [ "$IS_TEST_MODE" != "true" ]; then
    echo "[2/2] Starting Go WhatsApp bridge server..."
    cd "$PROJECT_DIR/whatsapp-mcp/whatsapp-bridge"
    # Build first to ensure we have the latest binary
    go build -o whatsapp-bridge .
    ./whatsapp-bridge &
    GO_PID=$!
    echo "      Go server started (PID: $GO_PID)"

    echo ""
    echo "✓ All services started!"
    echo "  - Go server (WhatsApp bridge): $BRIDGE_SOCKET_PATH (Unix socket)"
    echo "  - Agent server: $AGENT_SOCKET_PATH (Unix socket)"
    echo ""
else
    echo "[2/2] Skipping Go WhatsApp bridge start due to IS_TEST_MODE=true"
    echo ""
    echo "✓ Test Mode services started!"
    echo "  - Agent server (Gradio UI): http://127.0.0.1:7860"
    echo ""
    GO_PID=""
fi

echo "Press Ctrl+C to stop all services"

# Wait for both processes
if [ -n "$GO_PID" ]; then
    wait $GO_PID $AGENT_PID
else
    wait $AGENT_PID
fi
