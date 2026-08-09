"""Shared pytest setup: put the agent package and its deps on sys.path."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for path in (
    os.path.join(REPO_ROOT, "agent"),
    os.path.join(REPO_ROOT, "whatsapp-mcp", "whatsapp-mcp-server"),
):
    if path not in sys.path:
        sys.path.insert(0, path)
