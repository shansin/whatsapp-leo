#!/usr/bin/env python3
"""Sync tools_config.py with the live tool lists from every MCP server in MCP_REGISTRY.

Usage:
    uv run scripts/update_tools_config.py            # dry-run: show diff only
    uv run scripts/update_tools_config.py --write    # write changes to tools_config.py
    uv run scripts/update_tools_config.py --add-new  # also add newly available tools to existing allowlists
    uv run scripts/update_tools_config.py --write --add-new
    uv run scripts/update_tools_config.py -s playwright --write --add-new  # only query one server

Merge rules (safe by default):
  • New server in registry → added to TOOL_CONFIG with None (all tools enabled).
  • Existing server, config = None → unchanged (user wants all tools).
  • Existing server, config = list → stale tools pruned; new tools reported but NOT added
    unless --add-new is passed (prevents accidental tool exposure on updates).
  • Server in TOOL_CONFIG but removed from registry → reported, left in file untouched.

Tool descriptions are written as inline comments, e.g.:
    "calendar.listEvents",  # List events from a calendar within a time range
"""

import argparse
import asyncio
import os
import re
import sys

# ── Bootstrap path so we can import from agent/ ──────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(REPO_ROOT, "agent")
sys.path.insert(0, AGENT_DIR)

os.chdir(AGENT_DIR)  # needed for relative paths inside config.py

from dotenv import load_dotenv

load_dotenv(os.path.join(REPO_ROOT, ".env"), override=True)

from config import MCP_REGISTRY  # noqa: E402 (after sys.path setup)
from tools_config import TOOL_CONFIG  # noqa: E402

TOOLS_CONFIG_PATH = os.path.join(AGENT_DIR, "tools_config.py")

# Type alias: maps tool name → description (first line only, stripped)
Descriptions = dict[str, str]


# ── Tool discovery ────────────────────────────────────────────────────────────

async def fetch_tools(name: str, entry: dict) -> tuple[list[str], Descriptions]:
    """Start the MCP server and return (tool_names, {name: description})."""
    from agents.mcp import MCPServerStdio
    print(f"  Connecting to '{name}'...", flush=True)
    async with MCPServerStdio(
        params=entry["params"],
        client_session_timeout_seconds=entry["timeout"],
    ) as server:
        tools = await server.list_tools()
        names = [t.name for t in tools]
        descs = {t.name: _first_line(t.description or "") for t in tools}
        return names, descs


def _first_line(text: str) -> str:
    """Return the first non-empty line of a description, stripped."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


async def discover_all(servers: list[str] | None = None) -> dict[str, tuple[list[str], Descriptions] | None]:
    """Fetch live tool lists for MCP servers sequentially.

    If servers is given, only query those; otherwise query all in MCP_REGISTRY.
    Returns {server_name: (tool_names, descriptions)} or None if unreachable.
    """
    targets = {n: MCP_REGISTRY[n] for n in servers} if servers else MCP_REGISTRY
    result = {}
    for name, entry in targets.items():
        try:
            result[name] = await fetch_tools(name, entry)
        except Exception as exc:
            print(f"  WARNING: could not connect to '{name}': {exc}", flush=True)
            result[name] = None
    return result


# ── Merge logic ───────────────────────────────────────────────────────────────

def merge(
    current: dict[str, list[str] | None],
    live: dict[str, tuple[list[str], Descriptions] | None],
    add_new_tools: bool,
) -> tuple[dict[str, list[str] | None], list[str]]:
    """
    Merge live tool lists into the current config.
    Returns (updated_config, list_of_human_readable_changes).
    """
    updated = dict(current)
    changes = []

    # 1. New servers (in registry but not yet in TOOL_CONFIG)
    for name, live_data in live.items():
        if live_data is None:
            continue  # server unreachable — skip
        live_tools, _ = live_data
        if name not in current:
            updated[name] = None  # expose everything by default
            changes.append(f"[NEW]    '{name}' added with all {len(live_tools)} tools enabled (None).")

    # 2. Existing servers — sync allowlists
    for name, current_val in current.items():
        live_data = live.get(name)
        if live_data is None:
            if name not in live:
                changes.append(f"[WARN]   '{name}' is in TOOL_CONFIG but not in MCP_REGISTRY.")
            else:
                changes.append(f"[SKIP]   '{name}' was unreachable — config unchanged.")
            continue

        live_tools, _ = live_data
        live_set = set(live_tools)

        if current_val is None:
            continue  # user wants all tools — nothing to do

        current_set = set(current_val)
        removed = current_set - live_set
        new_available = live_set - current_set

        new_list = [t for t in current_val if t in live_set]  # preserve order, prune stale

        if removed:
            changes.append(
                f"[PRUNED] '{name}': removed {len(removed)} stale tool(s): {sorted(removed)}"
            )

        if new_available:
            if add_new_tools:
                new_list += sorted(new_available)
                changes.append(
                    f"[ADDED]  '{name}': added {len(new_available)} new tool(s): {sorted(new_available)}"
                )
            else:
                changes.append(
                    f"[INFO]   '{name}': {len(new_available)} new tool(s) available but not added "
                    f"(use --add-new to include): {sorted(new_available)}"
                )

        if new_list != current_val:
            updated[name] = new_list

    return updated, changes


# ── Code generation ───────────────────────────────────────────────────────────

def _format_tool_line(name: str, desc: str, indent: str = "        ") -> str:
    """Format a single tool entry with its description as an inline comment."""
    base = f'{indent}"{name}",'
    if desc:
        # Align comments at column 48 (relative to the indent start)
        padding = max(1, 48 - len(base))
        return f"{base}{' ' * padding}# {desc}"
    return base


def generate_tool_config_block(
    config: dict[str, list[str] | None],
    all_descs: dict[str, Descriptions],
) -> str:
    """Render the TOOL_CONFIG dict body with inline description comments."""
    parts = []
    for server_name, tools in config.items():
        if tools is None:
            parts.append(f'    "{server_name}": None,')
        else:
            server_descs = all_descs.get(server_name, {})
            lines = [
                _format_tool_line(t, server_descs.get(t, ""))
                for t in tools
            ]
            inner = "\n".join(lines)
            parts.append(f'    "{server_name}": [\n{inner}\n    ],')
    return "\n\n".join(parts)


def rewrite_tools_config(
    path: str,
    new_config: dict[str, list[str] | None],
    all_descs: dict[str, Descriptions],
) -> None:
    """Replace the TOOL_CONFIG dict in tools_config.py, preserving the rest of the file."""
    with open(path) as f:
        source = f.read()

    new_body = generate_tool_config_block(new_config, all_descs)
    new_dict = f"TOOL_CONFIG: dict[str, list[str] | None] = {{\n{new_body}\n}}"

    pattern = r"TOOL_CONFIG:\s*dict\[.*?\]\s*=\s*\{.*?\n\}"
    updated = re.sub(pattern, new_dict, source, flags=re.DOTALL)

    if updated == source:
        print("WARNING: could not locate TOOL_CONFIG in the file — nothing written.")
        return

    with open(path, "w") as f:
        f.write(updated)


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main(write: bool, add_new_tools: bool, servers: list[str] | None = None) -> None:
    if servers:
        unknown = [s for s in servers if s not in MCP_REGISTRY]
        if unknown:
            print(f"ERROR: unknown server(s): {unknown}")
            print(f"Available: {list(MCP_REGISTRY.keys())}")
            sys.exit(1)
        print(f"Discovering tools from: {', '.join(servers)}...")
    else:
        print("Discovering tools from MCP servers...")
    live = await discover_all(servers)
    print()

    # Descriptions are written alongside the tool names as comments
    all_descs: dict[str, Descriptions] = {
        name: data[1] for name, data in live.items() if data
    }

    updated, changes = merge(TOOL_CONFIG, live, add_new_tools)

    # Always rewrite when descriptions may have changed, even if tool lists are identical
    desc_write = write and not changes

    if not changes and not desc_write:
        # Check if any descriptions differ from what's already in the file
        # (simplest approach: always rewrite if --write is passed)
        if not write:
            print("Everything is up to date — no changes needed.")
            print("(Pass --write to refresh descriptions even when lists are unchanged.)")
            return

    if changes:
        print("Changes detected:")
        for c in changes:
            print(f"  {c}")
        print()

    if write:
        rewrite_tools_config(TOOLS_CONFIG_PATH, updated, all_descs)
        print("tools_config.py updated.")
    else:
        print("Dry run — pass --write to apply changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="Write changes to tools_config.py")
    parser.add_argument("--add-new", action="store_true", dest="add_new_tools",
                        help="Also add newly available tools to existing allowlists")
    parser.add_argument("--server", "-s", action="append", dest="servers", metavar="NAME",
                        help="Only query specific server(s). Can be repeated: -s brave -s playwright")
    args = parser.parse_args()

    asyncio.run(main(write=args.write, add_new_tools=args.add_new_tools, servers=args.servers))
