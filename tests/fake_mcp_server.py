"""Minimal stdio MCP server used by the MCP pool tests.

Exposes an echo tool, a way to identify the process, and a way to make the
process die mid-request so restart handling can be exercised.
"""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake")


@mcp.tool()
def echo(text: str) -> str:
    """Return the input back."""
    return f"echo:{text}"


@mcp.tool()
def whoami() -> str:
    """Return this server process's pid."""
    return str(os.getpid())


@mcp.tool()
def die() -> str:
    """Kill the server process without a clean shutdown."""
    os._exit(1)


@mcp.tool()
def die_once() -> str:
    """Kill the process the first time only, then answer normally.

    The marker file at $FAKE_MARKER persists across restarts of this process,
    so the second (retried) call is served by the respawned server.
    """
    marker = os.environ.get("FAKE_MARKER")
    if marker and not os.path.exists(marker):
        with open(marker, "w") as fh:
            fh.write("died")
        os._exit(1)
    return f"survived:{os.getpid()}"


if __name__ == "__main__":
    mcp.run()
