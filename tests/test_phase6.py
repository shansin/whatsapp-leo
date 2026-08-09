"""Tests for reply splitting, file-send guards, debouncing, and #status."""

import asyncio
import os

import pytest

import command_handlers
import debounce
import reply
import send_tools
from debounce import Debouncer, merge_content
from reply import split_reply
from send_tools import resolve_sendable

pytestmark = pytest.mark.asyncio


# ── Reply splitting (item 17) ────────────────────────────────────────────────


async def test_short_replies_are_not_split():
    assert split_reply("hello") == ["hello"]
    assert split_reply("") == []
    assert split_reply("   ") == []


async def test_long_replies_split_at_paragraphs():
    paragraphs = [f"Paragraph {n}. " + "word " * 30 for n in range(20)]
    text = "\n\n".join(paragraphs)

    parts = split_reply(text, limit=500)

    assert len(parts) > 1
    assert all(len(p) <= 500 for p in parts)
    # Nothing lost, nothing duplicated.
    assert " ".join(parts).split() == text.split()


async def test_split_never_cuts_inside_a_code_fence():
    text = "Here is some code:\n\n```\n" + "x = 1\n" * 60 + "```\n\nDone."
    parts = split_reply(text, limit=200)
    for part in parts:
        assert part.count("```") % 2 == 0, f"unbalanced fence in: {part[:60]!r}"


async def test_split_terminates_on_unbreakable_text():
    text = "x" * 5000
    parts = split_reply(text, limit=100)
    assert len(parts) == 50
    assert "".join(parts) == text


async def test_send_reply_sends_every_part(monkeypatch):
    monkeypatch.setattr(reply, "MAX_REPLY_CHARS", 300)
    sent = []

    def send(chat_jid, text):
        sent.append(text)
        return True, "ok"

    text = "\n\n".join(f"Para {n} " + "word " * 50 for n in range(10))
    assert await reply.send_reply(send, "c@lid", text)
    assert len(sent) > 1


async def test_send_reply_stops_on_failure(monkeypatch):
    """A truncated reply beats one delivered out of order."""
    monkeypatch.setattr(reply, "MAX_REPLY_CHARS", 300)
    sent = []

    def send(chat_jid, text):
        sent.append(text)
        return (False, "boom") if len(sent) == 2 else (True, "ok")

    text = "\n\n".join(f"Para {n} " + "word " * 50 for n in range(10))
    assert not await reply.send_reply(send, "c@lid", text)
    assert len(sent) == 2


async def test_voice_reply_is_off_without_configuration(monkeypatch):
    monkeypatch.setattr(reply, "VOICE_REPLIES", False)
    assert not await reply.send_voice_reply(lambda *a: (True, ""), "c@lid", "hi")

    monkeypatch.setattr(reply, "VOICE_REPLIES", True)
    monkeypatch.setattr(reply, "TTS_COMMAND", "")
    assert not await reply.send_voice_reply(lambda *a: (True, ""), "c@lid", "hi")


# ── File sending guard (item 17) ─────────────────────────────────────────────


@pytest.fixture
def share(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    (store / "photo.jpg").write_bytes(b"jpegdata")
    monkeypatch.setattr(send_tools, "STORE_DIR", str(store))
    monkeypatch.setattr(send_tools, "SHARE_DIR", "")
    return store


async def test_files_in_the_store_can_be_sent(share):
    assert resolve_sendable(str(share / "photo.jpg")) == str(
        (share / "photo.jpg").resolve()
    )


async def test_files_outside_the_store_are_refused(share, tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("BRAVE_API_KEY=hunter2")
    with pytest.raises(ValueError, match="outside"):
        resolve_sendable(str(secret))


async def test_traversal_is_refused(share):
    with pytest.raises(ValueError):
        resolve_sendable(str(share / ".." / ".env"))


async def test_symlinks_out_of_the_store_are_refused(share, tmp_path):
    """A link planted in the store must not become an exfiltration path."""
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    link = share / "innocent.txt"
    os.symlink(secret, link)

    with pytest.raises(ValueError, match="outside"):
        resolve_sendable(str(link))


async def test_missing_and_oversized_files_are_refused(share, monkeypatch):
    with pytest.raises(ValueError, match="No such file"):
        resolve_sendable(str(share / "nope.jpg"))

    monkeypatch.setattr(send_tools, "MAX_SEND_BYTES", 1)
    with pytest.raises(ValueError, match="too large"):
        resolve_sendable(str(share / "photo.jpg"))


async def test_share_dir_is_also_allowed(share, tmp_path, monkeypatch):
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    (outbox / "report.pdf").write_bytes(b"pdf")
    monkeypatch.setattr(send_tools, "SHARE_DIR", str(outbox))

    assert resolve_sendable(str(outbox / "report.pdf"))


# ── Debouncing (item 18) ─────────────────────────────────────────────────────


class _Msg:
    def __init__(self, content):
        self.content = content


async def test_rapid_messages_merge_into_one_turn():
    d = Debouncer(window=0.05)

    results = await asyncio.gather(
        d.collect("c@lid", _Msg("I was thinking")),
        _delayed(d.collect("c@lid", _Msg("about the trip")), 0.01),
        _delayed(d.collect("c@lid", _Msg("next month")), 0.02),
    )

    delivered = [r for r in results if r is not None]
    assert len(delivered) == 1, "only the last message should proceed"
    assert [m.content for m in delivered[0]] == [
        "I was thinking",
        "about the trip",
        "next month",
    ]


async def _delayed(coro, delay):
    await asyncio.sleep(delay)
    return await coro


async def test_separate_chats_do_not_merge():
    d = Debouncer(window=0.02)
    a, b = await asyncio.gather(
        d.collect("one@lid", _Msg("hello")), d.collect("two@lid", _Msg("hi"))
    )
    assert [m.content for m in a] == ["hello"]
    assert [m.content for m in b] == ["hi"]


async def test_slow_messages_are_separate_turns():
    d = Debouncer(window=0.02)
    first = await d.collect("c@lid", _Msg("one"))
    second = await d.collect("c@lid", _Msg("two"))
    assert [m.content for m in first] == ["one"]
    assert [m.content for m in second] == ["two"]


async def test_zero_window_disables_debouncing():
    d = Debouncer(window=0)
    assert len(await d.collect("c@lid", _Msg("x"))) == 1


async def test_long_bursts_flush_immediately(monkeypatch):
    """A burst must not extend the window indefinitely."""
    monkeypatch.setattr(debounce, "MAX_BURST_MESSAGES", 3)
    d = Debouncer(window=60)  # never reached: the cap should flush first

    # The first two park on the (long) window; the third trips the cap.
    waiting = [
        asyncio.create_task(d.collect("c@lid", _Msg("a"))),
        asyncio.create_task(d.collect("c@lid", _Msg("b"))),
    ]
    await asyncio.sleep(0.01)

    burst = await asyncio.wait_for(d.collect("c@lid", _Msg("c")), timeout=1)

    assert [m.content for m in burst] == ["a", "b", "c"]
    for task in waiting:
        task.cancel()


async def test_merge_content_drops_blanks_and_duplicates():
    merged = merge_content([_Msg("one"), _Msg("  "), _Msg("one"), _Msg("two")])
    assert merged == "one\ntwo"


# ── #status (item 19) ────────────────────────────────────────────────────────


async def test_status_reports_model_and_errors(monkeypatch):
    from logging_setup import error_deque
    from models import ReceivedMessage

    sent = []
    monkeypatch.setattr(
        reply, "whatsapp_send_message", lambda r, t, **k: sent.append(t) or (True, "ok")
    )
    error_deque.clear()
    error_deque.append((1_800_000_000.0, "ERROR", "AgentServer", "something broke"))

    message = ReceivedMessage.from_dict(
        {"content": "#status", "phone_number": "111", "chat_jid": "c@lid"}
    )
    await command_handlers.handle_status_command(message)

    text = sent[-1]
    assert "Leo Status" in text
    assert "Model:" in text
    assert "Uptime:" in text
    assert "something broke" in text


async def test_status_says_so_when_clean(monkeypatch):
    from logging_setup import error_deque
    from models import ReceivedMessage

    sent = []
    monkeypatch.setattr(
        reply, "whatsapp_send_message", lambda r, t, **k: sent.append(t) or (True, "ok")
    )
    error_deque.clear()

    await command_handlers.handle_status_command(
        ReceivedMessage.from_dict(
            {"content": "#status", "phone_number": "111", "chat_jid": "c@lid"}
        )
    )
    assert "No warnings or errors" in sent[-1]


async def test_uptime_formatting():
    assert command_handlers._format_uptime(90) == "1m"
    assert command_handlers._format_uptime(3700) == "1h 1m"
    assert command_handlers._format_uptime(90000) == "1d 1h 0m"
