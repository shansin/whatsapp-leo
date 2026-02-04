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

# Get instance GUID (default to "default" if not set)
INSTANCE_GUID="${INSTANCE_GUID:-default}"

# Set socket paths based on env vars or defaults with INSTANCE_GUID
AGENT_SOCKET_PATH="${AGENT_SOCKET_PATH:-/tmp/whatsapp-leo-${INSTANCE_GUID}.sock}"
BRIDGE_SOCKET_PATH="${BRIDGE_SOCKET_PATH:-/tmp/whatsapp-bridge-${INSTANCE_GUID}.sock}"

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
    kill $GO_PID $AGENT_PID 2>/dev/null || true
    wait $GO_PID $AGENT_PID 2>/dev/null || true
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

# Give the agent server a moment to start
sleep 2

# Start the Go WhatsApp bridge server
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
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait $GO_PID $AGENT_PID
