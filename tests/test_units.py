"""Pure-unit coverage for hooks, schedule parsing, models, and failover."""

import asyncio

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

import briefing
import hooks
from fallback_model import FallbackModel, FallbackRouter
from models import ReceivedMessage, ReminderParsed

pytestmark = pytest.mark.asyncio


# ── Hook matching ────────────────────────────────────────────────────────────


@pytest.fixture
def hooks_on(monkeypatch):
    monkeypatch.setattr(hooks, "IS_HOOK_ENABLED", True)
    monkeypatch.setattr(hooks, "HOOKS", ["claude", "claude-session"])


@pytest.mark.parametrize(
    "content,expected",
    [
        ("#claude write me a poem", ("claude", "write me a poem")),
        ("@claude write me a poem", ("claude", "write me a poem")),
        ("#CLAUDE shout", ("claude", "shout")),
        ("#claude", ("claude", "")),
        ("#claude-session hi", ("claude-session", "hi")),
        # Must not match a longer word that merely starts with the hook name.
        ("#claudette hello", None),
        ("please ask #claude about it", None),
        ("", None),
    ],
)
async def test_match_hook(hooks_on, content, expected):
    assert hooks.match_hook(content) == expected


async def test_hooks_disabled_matches_nothing(monkeypatch):
    monkeypatch.setattr(hooks, "IS_HOOK_ENABLED", False)
    assert hooks.match_hook("#claude hi") is None


@pytest.mark.parametrize(
    "content,expected",
    [
        ("#claude #start", ("claude", "start")),
        ("#claude #stop", ("claude", "stop")),
        ("@claude #START", ("claude", "start")),
        ("#claude start", None),  # needs the # prefix
        ("#claude #start now", None),
    ],
)
async def test_match_hook_session_command(hooks_on, content, expected):
    assert hooks.match_hook_session_command(content) == expected


async def test_hook_sessions_are_per_chat(hooks_on):
    hooks.start_hook_session("a@lid", "claude")
    assert hooks.get_hook_session("a@lid") == "claude"
    assert hooks.get_hook_session("b@lid") is None

    assert hooks.stop_hook_session("a@lid") == "claude"
    assert hooks.get_hook_session("a@lid") is None
    assert hooks.stop_hook_session("a@lid") is None


# ── Schedule parsing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "schedule,cron",
    [
        ("9am everyday", "0 9 * * *"),
        ("every day at 9am", "0 9 * * *"),
        ("every morning", "0 9 * * *"),
        ("9pm everyday", "0 21 * * *"),
        ("12am daily", "0 0 * * *"),
        ("12pm daily", "0 12 * * *"),
        ("8:30am monday", "30 8 * * 1"),
        ("5pm friday", "0 17 * * 5"),
    ],
)
async def test_parse_schedule_to_cron(schedule, cron):
    assert briefing.parse_schedule_to_cron(schedule) == cron


async def test_get_next_run_is_in_the_future():
    from datetime import datetime

    from config import TZ

    base = datetime(2026, 6, 1, 10, 0, tzinfo=TZ)
    nxt = briefing.get_next_run_from_cron("0 9 * * *", base)

    assert nxt > base
    assert nxt.hour == 9
    assert nxt.tzinfo is not None


async def test_get_next_run_follows_the_base_timezone():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    london = ZoneInfo("Europe/London")
    base = datetime(2026, 6, 1, 10, 0, tzinfo=london)
    nxt = briefing.get_next_run_from_cron("0 9 * * *", base)

    assert nxt.hour == 9
    assert nxt.utcoffset() == base.utcoffset()


# ── Models ───────────────────────────────────────────────────────────────────


async def test_received_message_defaults_every_field():
    message = ReceivedMessage.from_dict({})
    assert message.chat_jid == ""
    assert message.is_from_me is False
    assert message.file_length == 0
    assert message.quoted_message_id == ""


async def test_received_message_reads_known_fields():
    message = ReceivedMessage.from_dict(
        {
            "chat_jid": "c@lid",
            "content": "hi",
            "is_from_me": True,
            "file_length": 42,
            "quoted_message_id": "Q1",
            "unexpected": "ignored",
        }
    )
    assert message.chat_jid == "c@lid"
    assert message.is_from_me is True
    assert message.file_length == 42
    assert message.quoted_message_id == "Q1"
    assert not hasattr(message, "unexpected")


async def test_reminder_parsed_requires_both_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReminderParsed(reminder_message="x")


# ── Failover state machine ───────────────────────────────────────────────────


class _FakeModel:
    """Primary/backup stand-in that can be made to fail on demand."""

    def __init__(self, name, error=None, delay=0.0):
        self.model = name
        self.error = error
        self.delay = delay
        self.calls = 0

    async def get_response(self, *args, **kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return f"{self.model}-response"


def _connection_error():
    return APIConnectionError(request=None)


def _status_error(code):
    import httpx

    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    response = httpx.Response(status_code=code, request=request)
    return APIStatusError("boom", response=response, body=None)


def _fallback(primary, backup, sticky=300, timeout=5):
    return FallbackModel(
        primary=primary,
        backup=backup,
        router=FallbackRouter(sticky_seconds=sticky),
        primary_timeout_seconds=timeout,
    )


async def test_primary_is_used_when_healthy():
    primary, backup = _FakeModel("primary"), _FakeModel("backup")
    assert await _fallback(primary, backup).get_response() == "primary-response"
    assert backup.calls == 0


async def test_connection_failure_routes_to_backup():
    primary = _FakeModel("primary", error=_connection_error())
    backup = _FakeModel("backup")
    assert await _fallback(primary, backup).get_response() == "backup-response"
    assert backup.calls == 1


async def test_server_errors_route_to_backup_but_client_errors_do_not():
    backup = _FakeModel("backup")
    server_side = _FakeModel("primary", error=_status_error(503))
    assert await _fallback(server_side, backup).get_response() == "backup-response"

    backup2 = _FakeModel("backup")
    client_side = _FakeModel("primary", error=_status_error(404))
    with pytest.raises(APIStatusError):
        await _fallback(client_side, backup2).get_response()
    assert backup2.calls == 0, "a 404 would fail identically on the backup"


async def test_a_stalled_primary_times_out_to_backup():
    primary = _FakeModel("primary", delay=5)
    backup = _FakeModel("backup")
    model = _fallback(primary, backup, timeout=0.05)
    assert await model.get_response() == "backup-response"


async def test_backup_is_sticky_until_the_probe_window():
    primary = _FakeModel("primary", error=_connection_error())
    backup = _FakeModel("backup")
    model = _fallback(primary, backup, sticky=60)

    await model.get_response()  # fails over
    primary.error = None  # primary recovers, but we shouldn't notice yet
    await model.get_response()

    assert primary.calls == 1, "should have stayed on backup"
    assert backup.calls == 2


async def test_primary_is_reclaimed_after_the_sticky_window():
    primary = _FakeModel("primary", error=_connection_error())
    backup = _FakeModel("backup")
    model = _fallback(primary, backup, sticky=0)  # probe on every call

    await model.get_response()
    primary.error = None
    assert await model.get_response() == "primary-response"
    # And it stays on primary now.
    assert await model.get_response() == "primary-response"


async def test_failed_probe_keeps_us_on_backup():
    primary = _FakeModel("primary", error=_connection_error())
    backup = _FakeModel("backup")
    model = _fallback(primary, backup, sticky=0)

    assert await model.get_response() == "backup-response"
    assert await model.get_response() == "backup-response"
    assert primary.calls == 2  # probed each time, failed each time


async def test_no_backup_configured_propagates_the_error():
    primary = _FakeModel("primary", error=_connection_error())
    with pytest.raises(APIConnectionError):
        await _fallback(primary, None).get_response()


async def test_timeout_error_is_classified_as_primary_failure():
    from fallback_model import _is_primary_failure

    assert _is_primary_failure(APITimeoutError(request=None))
    assert _is_primary_failure(TimeoutError())
    assert _is_primary_failure(_status_error(500))
    assert not _is_primary_failure(_status_error(400))
    assert not _is_primary_failure(ValueError("nope"))
