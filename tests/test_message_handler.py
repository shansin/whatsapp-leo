"""Tests for the user-visible feedback paths in process_message."""

import pytest

import message_handler
import reply

pytestmark = pytest.mark.asyncio


def _payload(**overrides) -> dict:
    data = {
        "chat_jid": "12345@lid",
        "chat_name": "Test",
        "content": "#leo hello",
        "id": "MSGID1",
        "is_from_me": False,
        "media_type": "",
        "phone_number": "15550000000",
        "sender": "Tester",
        "sender_jid": "15550000000@s.whatsapp.net",
        "timestamp": "2026-08-02T10:00:00Z",
    }
    data.update(overrides)
    return data


class _StubPool:
    async def ensure_started(self):
        return None

    def servers(self, is_privileged=False):
        return []


class _StubSession:
    async def trim(self, max_items):
        return 0


@pytest.fixture
def sent(monkeypatch):
    """Capture outbound WhatsApp messages; stub out everything external."""
    messages: list[tuple[str, str]] = []

    def fake_send(recipient, text, reply_to=None, reply_to_sender=None):
        messages.append((recipient, text))
        return True, "sent"

    monkeypatch.setattr(message_handler, "whatsapp_send_message", fake_send)
    monkeypatch.setattr(reply, "whatsapp_send_message", fake_send)
    monkeypatch.setattr(message_handler, "send_presence", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(message_handler, "PRESENCE_ENABLED", False)
    monkeypatch.setattr(message_handler, "IS_DEDICATED_NUMBER", False)
    monkeypatch.setattr(message_handler, "mcp_pool", _StubPool())
    # Keep the real factory (and its per-chat locks); only stub agent creation.
    async def fake_get_agent(**kwargs):
        return object(), _StubSession()

    monkeypatch.setattr(message_handler.agent_factory, "get_agent", fake_get_agent)
    monkeypatch.setattr(message_handler.agent_factory, "_locks", {})
    # Debouncing is exercised in test_phase6; keep it out of the way here.
    from debounce import Debouncer

    monkeypatch.setattr(message_handler, "debouncer", Debouncer(window=0))
    monkeypatch.setattr(message_handler, "load_memory", lambda phone: "")
    monkeypatch.setattr(message_handler, "make_memory_tools", lambda phone: [])
    return messages


def _stub_runner(monkeypatch, *, output=None, error=None):
    class FakeResult:
        final_output = output

    class FakeRunner:
        @staticmethod
        async def run(agent, input, session=None, run_config=None):
            if error is not None:
                raise error
            return FakeResult()

    monkeypatch.setattr(message_handler, "Runner", FakeRunner)


async def test_successful_run_replies(sent, monkeypatch):
    _stub_runner(monkeypatch, output="Hi there")
    await message_handler.process_message(_payload())
    assert len(sent) == 1
    assert "Hi there" in sent[0][1]


async def test_failure_tells_the_user(sent, monkeypatch):
    """An exception used to be logged and silently dropped."""
    _stub_runner(monkeypatch, error=RuntimeError("ollama exploded"))
    await message_handler.process_message(_payload())
    assert len(sent) == 1, "user must get an error reply, not silence"
    assert "❌" in sent[0][1]


async def test_empty_output_tells_the_user(sent, monkeypatch):
    _stub_runner(monkeypatch, output="   ")
    await message_handler.process_message(_payload())
    assert len(sent) == 1, "empty model output must not send nothing"
    assert sent[0][1].strip() != ""


async def test_unreadable_image_is_flagged(sent, monkeypatch):
    """Vision download failure previously answered from text with no explanation."""
    _stub_runner(monkeypatch, output="Some answer")

    async def failed_vision(*args, **kwargs):
        return None

    monkeypatch.setattr(message_handler, "_build_vision_input", failed_vision)

    await message_handler.process_message(_payload(media_type="image"))
    assert len(sent) == 1
    body = sent[0][1]
    assert "Some answer" in body
    assert "couldn't read that image" in body


async def test_unrelated_message_is_ignored(sent, monkeypatch):
    _stub_runner(monkeypatch, output="should not be used")
    await message_handler.process_message(_payload(content="just chatting"))
    assert sent == []


async def test_own_message_on_a_shared_number_still_gets_answered(sent, monkeypatch):
    """Regression: on a shared number the user's own messages are is_from_me.

    A blanket `if message.is_from_me: return` silently dropped every command
    and every #leo mention on such an instance.
    """
    _stub_runner(monkeypatch, output="Life is good")
    await message_handler.process_message(_payload(is_from_me=True))
    assert len(sent) == 1
    assert "Life is good" in sent[0][1]


async def test_leos_own_reply_does_not_loop(sent, monkeypatch):
    """Leo's own replies carry no #leo mention, so they are not addressed to it."""
    _stub_runner(monkeypatch, output="should never run")
    await message_handler.process_message(
        _payload(is_from_me=True, content="_*(Leo)*_ Life is good")
    )
    assert sent == []


async def test_runs_in_one_chat_do_not_interleave(sent, monkeypatch):
    """Concurrent runs shared one agent and one session — they must serialize."""
    import asyncio

    concurrent = 0
    peak = 0

    class FakeResult:
        final_output = "ok"

    class FakeRunner:
        @staticmethod
        async def run(agent, input, session=None, run_config=None):
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0.02)
            concurrent -= 1
            return FakeResult()

    monkeypatch.setattr(message_handler, "Runner", FakeRunner)

    await asyncio.gather(
        message_handler.process_message(_payload(id="A")),
        message_handler.process_message(_payload(id="B")),
    )

    assert peak == 1, "two runs in the same chat overlapped"
    assert len(sent) == 2


async def test_different_chats_still_run_in_parallel(sent, monkeypatch):
    import asyncio

    concurrent = 0
    peak = 0

    class FakeResult:
        final_output = "ok"

    class FakeRunner:
        @staticmethod
        async def run(agent, input, session=None, run_config=None):
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0.02)
            concurrent -= 1
            return FakeResult()

    monkeypatch.setattr(message_handler, "Runner", FakeRunner)

    await asyncio.gather(
        message_handler.process_message(_payload(chat_jid="one@lid")),
        message_handler.process_message(_payload(chat_jid="two@lid")),
    )

    assert peak == 2, "the per-chat lock must not serialize unrelated chats"
