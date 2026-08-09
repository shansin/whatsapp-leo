"""Tests for the long-lived MCP server pool."""

import asyncio
import os
import sys

import pytest
from mcp.shared.exceptions import McpError

import mcp_pool
from mcp_pool import MCPPool

pytestmark = pytest.mark.asyncio

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_SERVER = os.path.join(HERE, "fake_mcp_server.py")


def _fake_entry(args=None):
    return {
        "params": {"command": sys.executable, "args": args or [FAKE_SERVER]},
        "timeout": 15,
    }


@pytest.fixture
def registry(monkeypatch):
    """Point the pool at fake servers instead of the real registry."""

    def configure(entries: dict, privileged: list[str], always: list[str]):
        monkeypatch.setattr(mcp_pool, "MCP_REGISTRY", entries)
        monkeypatch.setattr(mcp_pool, "PRIVILEGED_SERVERS", privileged)
        monkeypatch.setattr(mcp_pool, "ALWAYS_SERVERS", always)

    return configure


async def _text(result):
    return result.content[0].text


async def test_servers_are_shared_across_calls(registry):
    """The whole point: one process serves every message, not one per message."""
    registry({"fake": _fake_entry()}, privileged=["fake"], always=[])
    pool = MCPPool()
    try:
        await pool.ensure_started()
        servers = pool.servers(is_privileged=True)
        assert len(servers) == 1

        first = await _text(await servers[0].call_tool("whoami", {}))
        await pool.ensure_started()  # second "message"
        second = await _text(await pool.servers(is_privileged=True)[0].call_tool("whoami", {}))

        assert first == second
    finally:
        await pool.stop()


async def test_privilege_gating(registry):
    registry(
        {"priv": _fake_entry(), "open": _fake_entry()},
        privileged=["priv"],
        always=["open"],
    )
    pool = MCPPool()
    try:
        await pool.ensure_started()
        assert [s.name for s in pool.servers(is_privileged=True)] == ["priv", "open"]
        assert [s.name for s in pool.servers(is_privileged=False)] == ["open"]
    finally:
        await pool.stop()


async def test_tool_calls_and_filtering(registry):
    registry({"fake": _fake_entry()}, privileged=[], always=["fake"])
    pool = MCPPool()
    try:
        await pool.ensure_started()
        server = pool.servers()[0]
        tools = await server.list_tools()
        assert {"echo", "whoami", "die"} <= {t.name for t in tools}
        assert await _text(await server.call_tool("echo", {"text": "hi"})) == "echo:hi"
    finally:
        await pool.stop()


async def test_in_flight_call_is_retried_after_restart(registry, tmp_path):
    """A call that kills the server once is retried against the new process."""
    marker = tmp_path / "died"
    entry = _fake_entry()
    entry["params"]["env"] = {**os.environ, "FAKE_MARKER": str(marker)}
    registry({"fake": entry}, privileged=[], always=["fake"])

    pool = MCPPool()
    try:
        await pool.ensure_started()
        server = pool.servers()[0]
        before = await _text(await server.call_tool("whoami", {}))

        result = await asyncio.wait_for(server.call_tool("die_once", {}), timeout=60)

        assert marker.exists()  # the first attempt really did kill the server
        answer = await _text(result)
        assert answer.startswith("survived:")
        assert answer != f"survived:{before}"  # answered by a fresh process
        assert server.healthy
    finally:
        await pool.stop()


async def test_crashed_server_recovers_for_later_calls(registry):
    """Even a call that can never succeed leaves the server usable afterwards."""
    registry({"fake": _fake_entry()}, privileged=[], always=["fake"])
    pool = MCPPool()
    try:
        await pool.ensure_started()
        server = pool.servers()[0]
        before = await _text(await server.call_tool("whoami", {}))

        with pytest.raises(McpError):
            await asyncio.wait_for(server.call_tool("die", {}), timeout=60)

        assert server.healthy
        assert await _text(await server.call_tool("whoami", {})) != before
    finally:
        await pool.stop()


async def test_broken_server_is_excluded_but_others_start(registry):
    """One server failing to launch must not take the pool down."""
    registry(
        {
            "broken": {
                "params": {"command": sys.executable, "args": ["-c", "raise SystemExit(1)"]},
                "timeout": 5,
            },
            "fake": _fake_entry(),
        },
        privileged=["broken"],
        always=["fake"],
    )
    pool = MCPPool()
    try:
        await pool.ensure_started()
        assert [s.name for s in pool.servers(is_privileged=True)] == ["fake"]
    finally:
        await pool.stop()


async def test_gate_skips_server(registry, monkeypatch):
    entry = _fake_entry()
    entry["gate"] = lambda: False
    registry({"gated": entry, "fake": _fake_entry()}, privileged=[], always=["gated", "fake"])
    pool = MCPPool()
    try:
        await pool.ensure_started()
        assert [s.name for s in pool.servers()] == ["fake"]
    finally:
        await pool.stop()
