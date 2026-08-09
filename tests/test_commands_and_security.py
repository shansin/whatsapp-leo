"""Tests for command matching, timestamp storage, and the write guard."""

from datetime import datetime, UTC

import pytest

import message_handler
import timeutil
import write_guard
from config import TZ
from message_handler import match_command
from models import ReceivedMessage

pytestmark = pytest.mark.asyncio


# ── Command matching (item 7) ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content,expected",
    [
        ("#remindme in 5 minutes call mum", "#remindme"),
        ("#reminder list", "#reminder"),
        ("  #BRIEFING add x  ", "#briefing"),
        ("#briefing", "#briefing"),
        # These used to be hijacked (and silently dropped for non-owners).
        ("did you see the #reminder I set?", None),
        ("what does #briefing do?", None),
        ("#remindmenot", None),
        ("hey #leo, remind me about the #briefing", None),
        ("", None),
    ],
)
async def test_command_matching_is_prefix_only(content, expected):
    assert match_command(content) == expected


# ── Storage timestamps (item 8) ──────────────────────────────────────────────


async def test_stored_times_sort_chronologically_across_dst():
    """Local-offset ISO strings mis-sort when the UTC offset changes.

    On 2026-11-01 the 1am hour happens twice in America/Los_Angeles: once at
    -07:00 (PDT) and again at -08:00 (PST). 01:30 PDT is 45 minutes *earlier*
    than 01:15 PST, but its ISO string sorts later.
    """
    earlier = datetime(2026, 11, 1, 1, 30, tzinfo=TZ, fold=0)  # PDT, -07:00
    later = datetime(2026, 11, 1, 1, 15, tzinfo=TZ, fold=1)  # PST, -08:00

    # Same-zone datetime comparison is wall-clock based (PEP 495), so compare
    # the actual instants.
    assert earlier.astimezone(UTC) < later.astimezone(UTC)
    # `remind_at <= now` on these strings gets it backwards:
    assert earlier.isoformat() > later.isoformat()
    # The storage format does not:
    assert timeutil.to_db(earlier) < timeutil.to_db(later)


async def test_roundtrip_preserves_the_instant():
    dt = datetime.now(TZ).replace(microsecond=0)
    assert timeutil.from_db(timeutil.to_db(dt)) == dt


async def test_naive_times_are_treated_as_local():
    naive = datetime(2026, 6, 1, 12, 0)
    assert timeutil.to_db(naive) == timeutil.to_db(naive.replace(tzinfo=TZ))


async def test_now_db_is_utc():
    assert timeutil.now_db().endswith("+00:00")
    parsed = datetime.fromisoformat(timeutil.now_db())
    assert abs((parsed - datetime.now(UTC)).total_seconds()) < 5


async def test_normalize_columns_migrates_legacy_rows(tmp_path):
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "t.db"))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, when_at TEXT)")
    legacy = datetime(2026, 6, 1, 12, 0, tzinfo=TZ).isoformat()
    conn.execute("INSERT INTO t (when_at) VALUES (?)", (legacy,))
    conn.commit()

    assert timeutil.normalize_columns(conn, "t", ["when_at"]) == 1
    stored = conn.execute("SELECT when_at FROM t").fetchone()[0]
    assert stored.endswith("+00:00")
    assert timeutil.from_db(stored) == datetime(2026, 6, 1, 12, 0, tzinfo=TZ)

    # Idempotent: a second run changes nothing.
    assert timeutil.normalize_columns(conn, "t", ["when_at"]) == 0
    conn.close()


# ── is_from_me guard (item 8) ────────────────────────────────────────────────


def _capture(monkeypatch) -> list:
    """Record every outbound message instead of sending it."""
    calls: list = []
    import reply

    def fake_send(*args, **kwargs):
        calls.append(args)
        return True, "ok"

    monkeypatch.setattr(message_handler, "whatsapp_send_message", fake_send)
    monkeypatch.setattr(reply, "whatsapp_send_message", fake_send)
    return calls


async def test_leos_own_replies_do_not_loop_on_a_shared_number(monkeypatch):
    """Leo's replies carry no #leo mention, so _should_respond ignores them."""
    calls = _capture(monkeypatch)
    monkeypatch.setattr(message_handler, "IS_DEDICATED_NUMBER", False)

    await message_handler.process_message(
        {
            "chat_jid": "c@lid",
            "content": "_*(Leo)*_ Life is good, thanks for asking!",
            "is_from_me": True,
        }
    )
    assert calls == []


async def test_own_messages_are_ignored_on_a_dedicated_number(monkeypatch):
    """Leo owns the account, so anything from it is its own output."""
    calls = _capture(monkeypatch)
    monkeypatch.setattr(message_handler, "IS_DEDICATED_NUMBER", True)

    await message_handler.process_message(
        {"chat_jid": "c@lid", "content": "#leo hi", "is_from_me": True}
    )
    assert calls == []


@pytest.mark.parametrize(
    "dedicated,from_me,expected",
    [
        # Shared number: the user types from this very device, so their own
        # messages are is_from_me and must still be processed. The #leo
        # mention is what separates them from Leo's replies.
        (False, True, False),
        (False, False, False),
        # Dedicated number: the user is on another number entirely, so
        # anything from us is our own output.
        (True, True, True),
        (True, False, False),
    ],
)
async def test_own_output_detection(monkeypatch, dedicated, from_me, expected):
    monkeypatch.setattr(message_handler, "IS_DEDICATED_NUMBER", dedicated)
    message = ReceivedMessage.from_dict(
        {"content": "#leo how's life", "is_from_me": from_me}
    )
    assert message_handler.is_own_output(message) is expected


# ── Write guard (item 10) ────────────────────────────────────────────────────


@pytest.fixture
def guard_on(monkeypatch):
    monkeypatch.setattr(write_guard, "REQUIRE_WRITE_CONFIRMATION", True)
    monkeypatch.setattr(write_guard, "_confirmed", set())
    write_guard.current_chat.set("chat@lid")


async def test_write_tool_is_blocked_without_confirmation(guard_on):
    write_guard.begin_turn("chat@lid", "delete my 3pm meeting")
    assert write_guard.refusal("workspace", "calendar.deleteEvent")


async def test_read_tool_is_never_blocked(guard_on):
    write_guard.begin_turn("chat@lid", "what's on my calendar?")
    assert write_guard.refusal("workspace", "calendar.listEvents") is None
    assert write_guard.refusal("brave", "brave_web_search") is None


async def test_confirmation_turn_allows_the_write(guard_on):
    write_guard.begin_turn("chat@lid", "yes, go ahead")
    assert write_guard.refusal("workspace", "calendar.deleteEvent") is None


async def test_confirmation_does_not_carry_to_the_next_turn(guard_on):
    write_guard.begin_turn("chat@lid", "yes")
    write_guard.end_turn("chat@lid")
    write_guard.begin_turn("chat@lid", "and what about tomorrow?")
    assert write_guard.refusal("workspace", "calendar.createEvent")


async def test_confirmation_is_scoped_to_one_chat(guard_on):
    write_guard.begin_turn("chat@lid", "yes")
    write_guard.current_chat.set("other@lid")
    write_guard.begin_turn("other@lid", "book the flight")
    assert write_guard.refusal("workspace", "calendar.createEvent")


async def test_guard_is_off_by_default(monkeypatch):
    monkeypatch.setattr(write_guard, "REQUIRE_WRITE_CONFIRMATION", False)
    monkeypatch.setattr(write_guard, "_confirmed", set())
    assert write_guard.refusal("workspace", "calendar.deleteEvent") is None
