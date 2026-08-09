"""Shared SQLite connection handling and polling-scheduler base.

The one-shot reminder, recurring reminder and briefing modules each grew their
own copy of: a module-global connection, WAL pragmas, a "have I migrated yet"
flag, and a `while True` poll loop with the same reconnect-on-OperationalError
handling. Three copies of the same code is three places for a fix to be
forgotten.
"""

import asyncio
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable

from logging_setup import logger


class SqliteStore:
    """A lazily-opened SQLite connection that migrates itself once."""

    def __init__(self, path_getter: Callable[[], str], migrate: Callable):
        # The path is read through a callable so tests can repoint it, and so
        # the file isn't touched at import time.
        self._path_getter = path_getter
        self._migrate = migrate
        self._conn: sqlite3.Connection | None = None
        self._migrated = False

    def connect(self) -> sqlite3.Connection:
        """Return the connection, opening and migrating it if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._path_getter(), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._migrated = False
        if not self._migrated:
            self._migrate(self._conn)
            self._migrated = True
        return self._conn

    def reset(self) -> None:
        """Drop the connection so the next call reopens it."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
        self._conn = None
        self._migrated = False


class PollingScheduler(ABC):
    """Background loop that checks a store for due work at a fixed interval.

    Subclasses implement ``poll()``. Everything a poll does — the query and the
    send — must stay off the event loop; ``asyncio.to_thread`` is the tool.
    """

    #: Human-readable name used in log lines.
    name = "scheduler"

    def __init__(self, store: SqliteStore, interval: float):
        self._store = store
        self._interval = interval

    @property
    def store(self) -> SqliteStore:
        return self._store

    def conn(self) -> sqlite3.Connection:
        return self._store.connect()

    @abstractmethod
    async def poll(self) -> None:
        """Do one round of work. Exceptions are logged, never fatal."""

    async def run(self) -> None:
        logger.info(f"{self.name} started (every {self._interval}s)")
        while True:
            try:
                await self.poll()
            except sqlite3.OperationalError as e:
                # Locked or corrupted; drop the connection and reopen next tick.
                logger.warning(f"{self.name}: DB error, reconnecting ({e})")
                self._store.reset()
            except Exception:
                logger.exception(f"{self.name}: error in poll loop")
            await asyncio.sleep(self._interval)
