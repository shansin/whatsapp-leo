"""Tests for the shared SQLite store and polling-scheduler base."""

import asyncio
import sqlite3

import pytest

from sqlite_store import PollingScheduler, SqliteStore

pytestmark = pytest.mark.asyncio


def _migrate(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.commit()


async def test_connection_is_reused_and_migrated_once(tmp_path):
    migrations = []
    store = SqliteStore(
        lambda: str(tmp_path / "s.db"),
        lambda conn: migrations.append(1) or _migrate(conn),
    )

    first, second = store.connect(), store.connect()
    assert first is second
    assert migrations == [1], "migration must run once per connection"
    assert first.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


async def test_reset_reopens_and_remigrates(tmp_path):
    migrations = []
    store = SqliteStore(
        lambda: str(tmp_path / "s.db"),
        lambda conn: migrations.append(1) or _migrate(conn),
    )

    first = store.connect()
    store.reset()
    second = store.connect()

    assert first is not second
    assert migrations == [1, 1]


async def test_path_is_read_lazily(tmp_path):
    """Tests repoint DB_PATH after import, so the path can't be captured early."""
    path = {"value": str(tmp_path / "a.db")}
    store = SqliteStore(lambda: path["value"], _migrate)
    store.connect()

    path["value"] = str(tmp_path / "b.db")
    store.reset()
    store.connect()

    assert (tmp_path / "a.db").exists()
    assert (tmp_path / "b.db").exists()


class _Scheduler(PollingScheduler):
    name = "test scheduler"

    def __init__(self, store, error=None):
        super().__init__(store, interval=0.01)
        self.polls = 0
        self.error = error

    async def poll(self):
        self.polls += 1
        if self.error:
            raise self.error


async def _run_briefly(scheduler, until):
    task = asyncio.create_task(scheduler.run())
    for _ in range(200):
        if until():
            break
        await asyncio.sleep(0.01)
    task.cancel()


async def test_scheduler_polls_repeatedly(tmp_path):
    scheduler = _Scheduler(SqliteStore(lambda: str(tmp_path / "s.db"), _migrate))
    await _run_briefly(scheduler, lambda: scheduler.polls >= 3)
    assert scheduler.polls >= 3


async def test_a_failing_poll_does_not_kill_the_loop(tmp_path):
    scheduler = _Scheduler(
        SqliteStore(lambda: str(tmp_path / "s.db"), _migrate),
        error=ValueError("boom"),
    )
    await _run_briefly(scheduler, lambda: scheduler.polls >= 3)
    assert scheduler.polls >= 3, "loop must survive an exception in poll()"


async def test_db_errors_reset_the_connection(tmp_path):
    store = SqliteStore(lambda: str(tmp_path / "s.db"), _migrate)
    scheduler = _Scheduler(store, error=sqlite3.OperationalError("database is locked"))
    first = store.connect()

    await _run_briefly(scheduler, lambda: scheduler.polls >= 2)

    assert store.connect() is not first, "a locked DB must trigger a reconnect"
