"""Tests for the WhatsApp typing indicator wrapper."""

import asyncio

import pytest

import message_handler

pytestmark = pytest.mark.asyncio


@pytest.fixture
def presence_calls(monkeypatch):
    """Record every presence call instead of hitting the bridge."""
    calls: list[tuple[str, str]] = []

    def fake_send_presence(chat_jid, state="composing", media=""):
        calls.append((chat_jid, state))
        return True, "ok"

    monkeypatch.setattr(message_handler, "send_presence", fake_send_presence)
    monkeypatch.setattr(message_handler, "PRESENCE_ENABLED", True)
    monkeypatch.setattr(message_handler, "PRESENCE_REFRESH_SECONDS", 0.01)
    return calls


async def test_typing_refreshes_then_pauses(presence_calls):
    async with message_handler.typing("chat@lid"):
        await asyncio.sleep(0.05)

    states = [state for _, state in presence_calls]
    assert states.count("composing") >= 2, "indicator must be refreshed while running"
    assert states[-1] == "paused", "indicator must be cleared when the run ends"
    assert {jid for jid, _ in presence_calls} == {"chat@lid"}


async def test_typing_pauses_when_body_raises(presence_calls):
    with pytest.raises(ValueError):
        async with message_handler.typing("chat@lid"):
            raise ValueError("boom")

    assert presence_calls[-1] == ("chat@lid", "paused")


async def test_typing_stops_when_bridge_rejects(monkeypatch):
    """An unsupported/failing bridge must not spin, and must not raise."""
    calls: list[str] = []

    def failing(chat_jid, state="composing", media=""):
        calls.append(state)
        return False, "Not connected to WhatsApp"

    monkeypatch.setattr(message_handler, "send_presence", failing)
    monkeypatch.setattr(message_handler, "PRESENCE_ENABLED", True)
    monkeypatch.setattr(message_handler, "PRESENCE_REFRESH_SECONDS", 0.01)

    async with message_handler.typing("chat@lid"):
        await asyncio.sleep(0.05)

    assert calls.count("composing") == 1  # gave up after the first failure


async def test_typing_disabled_is_a_noop(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        message_handler, "send_presence", lambda *a, **k: calls.append("x") or (True, "")
    )
    monkeypatch.setattr(message_handler, "PRESENCE_ENABLED", False)

    async with message_handler.typing("chat@lid"):
        await asyncio.sleep(0.02)

    assert calls == []
