"""Tests for #help/commands, per-user timezone, and history tools."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import command_handlers
import message_handler
import reply
import reminder
import user_prefs
from config import TZ
from message_handler import match_command, parse_snooze

pytestmark = pytest.mark.asyncio


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    monkeypatch.setattr(user_prefs, "PREFS_PATH", str(tmp_path / "prefs.json"))
    monkeypatch.setattr(user_prefs, "_cache", None)
    return user_prefs


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the reminder module at a throwaway database."""
    monkeypatch.setattr(reminder, "DB_PATH", str(tmp_path / "reminders.db"))
    reminder._store.reset()
    reminder._recurring_store.reset()
    yield
    reminder._store.reset()
    reminder._recurring_store.reset()


@pytest.fixture
def replies(monkeypatch):
    """Capture replies from both reply helpers."""
    sent: list[str] = []

    def fake_send(recipient, text, reply_to=None, reply_to_sender=None):
        sent.append(text)
        return True, "ok"

    monkeypatch.setattr(reply, "whatsapp_send_message", fake_send)
    monkeypatch.setattr(message_handler, "whatsapp_send_message", fake_send)
    return sent


def _msg(content="", phone="15550000000", chat="c@lid", **kw):
    from models import ReceivedMessage

    data = {
        "content": content,
        "phone_number": phone,
        "chat_jid": chat,
        "id": "M1",
        "sender_jid": f"{phone}@s.whatsapp.net",
    }
    data.update(kw)
    return ReceivedMessage.from_dict(data)


# ── Per-user timezone (item 15) ──────────────────────────────────────────────


async def test_default_timezone_is_the_instance_default(prefs):
    assert prefs.get_tz("15550000000") == TZ


async def test_timezone_is_stored_per_user(prefs):
    prefs.set_tz("111", "Europe/London")
    prefs.set_tz("222", "Asia/Kolkata")

    assert prefs.get_tz("111") == ZoneInfo("Europe/London")
    assert prefs.get_tz("222") == ZoneInfo("Asia/Kolkata")
    assert prefs.get_tz("333") == TZ


async def test_timezone_survives_a_reload(prefs, monkeypatch):
    prefs.set_tz("111", "Europe/London")
    monkeypatch.setattr(user_prefs, "_cache", None)  # simulate a restart
    assert prefs.get_tz("111") == ZoneInfo("Europe/London")


async def test_invalid_timezone_is_rejected(prefs):
    with pytest.raises(ValueError):
        prefs.set_tz("111", "Mars/Olympus_Mons")
    assert prefs.get_tz("111") == TZ


async def test_tz_command_shows_and_sets(prefs, replies):
    await command_handlers.handle_tz_command(_msg("#tz", phone="111"))
    assert str(TZ) in replies[-1]

    await command_handlers.handle_tz_command(_msg("#tz Europe/London", phone="111"))
    assert "Europe/London" in replies[-1]
    assert prefs.get_tz("111") == ZoneInfo("Europe/London")

    await command_handlers.handle_tz_command(_msg("#tz Nowhere/Fake", phone="111"))
    assert "❌" in replies[-1]


# ── Commands (item 14) ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content,expected",
    [("#help", "#help"), ("#tz Europe/London", "#tz"), ("#helpful hint", None)],
)
async def test_new_commands_are_matched(content, expected):
    assert match_command(content) == expected


async def test_help_lists_every_command(replies):
    await command_handlers.handle_help_command(_msg("#help"))
    text = replies[-1]
    for expected in ("#remindme", "#reminder", "#briefing", "#tz", "snooze"):
        assert expected in text


async def test_help_includes_configured_hooks(replies):
    await command_handlers.handle_help_command(_msg("#help"), hook_names=["claude"])
    assert "#claude" in replies[-1]


async def test_remindme_list_and_cancel(db, prefs, replies):
    from config import TZ as default_tz

    rid = reminder.store_reminder(
        "c@lid", "call the dentist", datetime.now(default_tz) + timedelta(hours=2)
    )
    reminder.store_reminder(
        "other@lid", "not mine", datetime.now(default_tz) + timedelta(hours=2)
    )

    await command_handlers.handle_remindme_list(_msg("#remindme list"))
    assert "call the dentist" in replies[-1]
    assert "not mine" not in replies[-1], "reminders from other chats must not leak"

    await command_handlers.handle_remindme_cancel(
        _msg(f"#remindme cancel {rid}"), ["#remindme", "cancel", str(rid)]
    )
    assert "✅" in replies[-1]
    assert reminder.get_pending_reminders("c@lid") == []


async def test_cancel_cannot_reach_another_chat(db, replies):
    rid = reminder.store_reminder(
        "other@lid", "theirs", datetime.now(TZ) + timedelta(hours=1)
    )
    await command_handlers.handle_remindme_cancel(
        _msg(f"#remindme cancel {rid}"), ["#remindme", "cancel", str(rid)]
    )
    assert "❌" in replies[-1]
    assert len(reminder.get_pending_reminders("other@lid")) == 1


async def test_briefing_pause_and_resume(replies, monkeypatch):
    toggled = []
    monkeypatch.setattr(
        command_handlers,
        "toggle_briefing",
        lambda bid, enabled: toggled.append((bid, enabled)) or True,
    )

    await command_handlers.handle_briefing_toggle(
        _msg("#briefing pause 3"), ["#briefing", "pause", "3"], enable=False
    )
    await command_handlers.handle_briefing_toggle(
        _msg("#briefing resume 3"), ["#briefing", "resume", "3"], enable=True
    )
    assert toggled == [(3, False), (3, True)]


async def test_briefing_run_executes_now(replies, monkeypatch):
    monkeypatch.setattr(
        command_handlers,
        "list_briefings",
        lambda: [{"id": 7, "name": "Morning", "prompt": "do it", "enabled": True}],
    )

    import briefing_executor

    async def fake_execute(prompt, chat_jid, name):
        return f"ran:{prompt}"

    monkeypatch.setattr(briefing_executor, "execute_briefing_prompt", fake_execute)

    await command_handlers.handle_briefing_run(
        _msg("#briefing run 7"), ["#briefing", "run", "7"]
    )
    assert "ran:do it" in replies[-1]


async def test_briefing_run_reports_missing_id(replies, monkeypatch):
    monkeypatch.setattr(command_handlers, "list_briefings", lambda: [])
    await command_handlers.handle_briefing_run(
        _msg("#briefing run 99"), ["#briefing", "run", "99"]
    )
    assert "not found" in replies[-1]


# ── Snooze (item 14) ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content,expected",
    [
        ("snooze 10m", timedelta(minutes=10)),
        ("snooze 10", timedelta(minutes=10)),
        ("Snooze 2h", timedelta(hours=2)),
        ("snooze 1 day", timedelta(days=1)),
        ("  snooze 15 minutes ", timedelta(minutes=15)),
        ("snooze", None),
        ("snooze the alarm please", None),
        ("let's snooze 10m", None),
    ],
)
async def test_snooze_parsing(content, expected):
    assert parse_snooze(content) == expected


async def test_snooze_reschedules_the_quoted_reminder(db, prefs, replies, monkeypatch):
    monkeypatch.setattr(message_handler, "ALLOWED_SENDERS", ["15550000000"])

    message = _msg(
        "snooze 30m",
        quoted_message_content="⏰ *Reminder*\n\ntake the bins out",
    )
    handled = await message_handler._handle_snooze(message, timedelta(minutes=30))

    assert handled
    pending = reminder.get_pending_reminders("c@lid")
    assert len(pending) == 1
    assert pending[0][1] == "take the bins out"
    assert "😴" in replies[-1]


async def test_snooze_ignores_replies_to_other_messages(db, replies):
    message = _msg("snooze 30m", quoted_message_content="what time is the meeting?")
    assert not await message_handler._handle_snooze(message, timedelta(minutes=30))
    assert reminder.get_pending_reminders("c@lid") == []


# ── History tools (item 16) ──────────────────────────────────────────────────


async def test_history_tools_are_privileged_only():
    from history_tools import make_history_tools

    tools = make_history_tools("c@lid", "15550000000")
    names = {t.name for t in tools}
    assert names == {
        "search_chat_history",
        "recent_chat_messages",
        "messages_from_person",
        "find_chats",
    }


async def test_history_formatting_uses_the_user_timezone():
    from history_tools import _format
    from whatsapp import Message

    msg = Message(
        timestamp=datetime(2026, 6, 1, 12, 0, tzinfo=ZoneInfo("UTC")),
        sender="Sam",
        content="see you at the airport",
        is_from_me=False,
        chat_jid="c@lid",
        id="M1",
        chat_name="Trip",
    )

    london = _format([msg], ZoneInfo("Europe/London"))
    tokyo = _format([msg], ZoneInfo("Asia/Tokyo"))

    assert "13:00" in london  # BST
    assert "21:00" in tokyo
    assert "Sam: see you at the airport" in london
    assert "(Trip)" in london


async def test_history_formatting_truncates_and_labels_media():
    from history_tools import SNIPPET_CHARS, _format
    from whatsapp import Message

    long_msg = Message(
        timestamp=datetime(2026, 6, 1, 12, 0, tzinfo=ZoneInfo("UTC")),
        sender="Sam",
        content="x" * (SNIPPET_CHARS + 200),
        is_from_me=False,
        chat_jid="c@lid",
        id="M1",
    )
    media_msg = Message(
        timestamp=datetime(2026, 6, 1, 12, 0, tzinfo=ZoneInfo("UTC")),
        sender="Sam",
        content="",
        is_from_me=True,
        chat_jid="c@lid",
        id="M2",
        media_type="image",
    )

    out = _format([long_msg, media_msg], ZoneInfo("UTC"))
    assert "…" in out
    assert len(out.splitlines()[0]) < SNIPPET_CHARS + 100
    assert "Me: [image]" in out


async def test_history_formatting_handles_no_results():
    from history_tools import _format

    assert _format([], TZ) == "No matching messages."
