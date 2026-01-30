#!/bin/bash

# Start Services Script
# Starts both the Go WhatsApp bridge server and the Python agent server

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting WhatsApp Leo services..."

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $GO_PID $AGENT_PID 2>/dev/null || true
    wait $GO_PID $AGENT_PID 2>/dev/null || true
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
go run . &
GO_PID=$!
echo "      Go server started (PID: $GO_PID)"

echo ""
echo "✓ All services started!"
echo "  - Go server (WhatsApp bridge): http://localhost:8080"
echo "  - Agent server: http://localhost:8081"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait $GO_PID $AGENT_PID
