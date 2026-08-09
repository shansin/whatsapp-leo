"""Tests for Chrome browsing: which browser tools the write guard gates, and
how the Playwright server is gated onto privileged senders only.

Browsing is the one capability that pipes attacker-authored text straight into
a privileged agent, so the split between reading a page (never gated) and
acting on one (gated) is worth pinning down.
"""

import pytest

import config
import mcp_pool
import write_guard

pytestmark = pytest.mark.asyncio


@pytest.fixture
def guard_on(monkeypatch):
    monkeypatch.setattr(write_guard, "REQUIRE_WRITE_CONFIRMATION", True)
    monkeypatch.setattr(write_guard, "_confirmed", set())
    write_guard.current_chat.set("chat@lid")


# ── Which browser tools are gated ────────────────────────────────────────────

# Acting on a page, or executing code in it, on the user's behalf.
ACTING_TOOLS = [
    "browser_evaluate",
    "browser_run_code",
    "browser_file_upload",
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_press_key",
    "browser_select_option",
    "browser_handle_dialog",
]

# Reaching a page and reading it. Gating these would demand a "yes" per page.
READING_TOOLS = [
    "browser_navigate",
    "browser_navigate_back",
    "browser_snapshot",
    "browser_take_screenshot",
    "browser_wait_for",
    "browser_tabs",
    "browser_resize",
    "browser_hover",
    "browser_drag",
    "browser_console_messages",
    "browser_network_requests",
    "browser_close",
]


@pytest.mark.parametrize("tool", ACTING_TOOLS)
async def test_acting_browser_tools_are_blocked_without_confirmation(guard_on, tool):
    write_guard.begin_turn("chat@lid", "log into my bank and move some money")
    assert write_guard.refusal("playwright", tool)


@pytest.mark.parametrize("tool", READING_TOOLS)
async def test_reading_browser_tools_are_never_blocked(guard_on, tool):
    write_guard.begin_turn("chat@lid", "what's on the front page of HN?")
    assert write_guard.refusal("playwright", tool) is None


async def test_confirmation_turn_allows_the_browser_action(guard_on):
    write_guard.begin_turn("chat@lid", "yes")
    assert write_guard.refusal("playwright", "browser_evaluate") is None


async def test_browser_confirmation_does_not_carry_to_the_next_turn(guard_on):
    write_guard.begin_turn("chat@lid", "yes")
    write_guard.end_turn("chat@lid")
    write_guard.begin_turn("chat@lid", "now open the next result")
    assert write_guard.refusal("playwright", "browser_click")


async def test_browser_tools_are_ungated_when_the_guard_is_off(monkeypatch):
    monkeypatch.setattr(write_guard, "REQUIRE_WRITE_CONFIRMATION", False)
    monkeypatch.setattr(write_guard, "_confirmed", set())
    assert write_guard.refusal("playwright", "browser_run_code") is None


# ── Server gating ────────────────────────────────────────────────────────────


async def test_playwright_is_gated_on_the_enabled_flag(monkeypatch):
    monkeypatch.setattr(config, "PLAYWRIGHT_ENABLED", False)
    assert mcp_pool._gate_open("playwright") is False

    monkeypatch.setattr(config, "PLAYWRIGHT_ENABLED", True)
    assert mcp_pool._gate_open("playwright") is True


async def test_playwright_is_privileged_only():
    """A stranger messaging Leo must not get a browser that holds the user's
    logged-in sessions."""
    assert "playwright" in config.PRIVILEGED_SERVERS
    assert "playwright" not in config.ALWAYS_SERVERS


async def test_launch_args_track_the_headless_and_sandbox_flags():
    """Defaults suit a systemd user unit: no DISPLAY, and a kernel that blocks
    Chrome's own sandbox. A local .env may override either, so assert the args
    follow the flags rather than hardcoding the outcome."""
    args = config.MCP_REGISTRY["playwright"]["params"]["args"]
    assert "--browser" in args
    assert "--user-data-dir" in args
    assert ("--headless" in args) is config._playwright_headless
    assert ("--no-sandbox" in args) is config._playwright_no_sandbox
