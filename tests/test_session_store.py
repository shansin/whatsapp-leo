"""Tests for durable, bounded conversation sessions."""

import pytest

import session_store
from session_store import TrimmedSQLiteSession, get_session

pytestmark = pytest.mark.asyncio


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


def _turn(n):
    return [_user(f"q{n}"), _assistant(f"a{n}")]


async def test_history_survives_a_new_session_object(tmp_path, monkeypatch):
    """History used to die with the process and with cache eviction."""
    db = tmp_path / "sessions.db"
    monkeypatch.setattr(session_store, "SESSIONS_DB_PATH", str(db))
    monkeypatch.setattr(session_store, "_sessions", {})

    await get_session("chat@lid").add_items(_turn(1))

    # Simulate a restart: drop every cached object, reopen from disk.
    monkeypatch.setattr(session_store, "_sessions", {})
    items = await get_session("chat@lid").get_items()

    assert [i["content"] for i in items] == ["q1", "a1"]


async def test_same_chat_returns_the_same_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setattr(session_store, "_sessions", {})
    assert get_session("a@lid") is get_session("a@lid")
    assert get_session("a@lid") is not get_session("b@lid")


async def test_trim_keeps_the_newest_items(tmp_path):
    session = TrimmedSQLiteSession("chat", db_path=str(tmp_path / "s.db"))
    for n in range(5):
        await session.add_items(_turn(n))

    removed = await session.trim(4)

    items = await session.get_items()
    assert removed == 6
    assert [i["content"] for i in items] == ["q3", "a3", "q4", "a4"]


async def test_trim_is_a_noop_under_the_limit(tmp_path):
    session = TrimmedSQLiteSession("chat", db_path=str(tmp_path / "s.db"))
    await session.add_items(_turn(1))
    assert await session.trim(40) == 0
    assert len(await session.get_items()) == 2


async def test_trim_disabled_by_zero(tmp_path):
    session = TrimmedSQLiteSession("chat", db_path=str(tmp_path / "s.db"))
    for n in range(5):
        await session.add_items(_turn(n))
    assert await session.trim(0) == 0
    assert len(await session.get_items()) == 10


async def test_trim_never_orphans_a_tool_result(tmp_path):
    """Cutting between a tool call and its output would break the next request."""
    session = TrimmedSQLiteSession("chat", db_path=str(tmp_path / "s.db"))
    await session.add_items(
        [
            _user("q0"),
            {"type": "function_call", "name": "search", "call_id": "1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "1", "output": "result"},
            _assistant("a0"),
            _user("q1"),
            _assistant("a1"),
        ]
    )

    # A naive cut of 2 would land on the function_call, stranding its output.
    await session.trim(4)

    items = await session.get_items()
    assert items[0] == _user("q1"), "history must start on a user turn"


async def test_trim_keeps_everything_when_no_boundary_exists(tmp_path):
    session = TrimmedSQLiteSession("chat", db_path=str(tmp_path / "s.db"))
    await session.add_items([_assistant(f"a{n}") for n in range(6)])

    assert await session.trim(2) == 0, "better to keep too much than corrupt history"
    assert len(await session.get_items()) == 6
