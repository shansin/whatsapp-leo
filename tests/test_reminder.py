"""Tests for reminder persistence and the scheduler's event-loop behaviour."""

import asyncio
import threading
from datetime import datetime, timedelta

import pytest

import reminder
from config import TZ

pytestmark = pytest.mark.asyncio


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the reminder module at a throwaway database."""
    monkeypatch.setattr(reminder, "DB_PATH", str(tmp_path / "reminders.db"))
    reminder._store.reset()
    reminder._recurring_store.reset()
    yield
    reminder._store.reset()
    reminder._recurring_store.reset()


async def test_validate_rejects_past_times():
    with pytest.raises(ValueError):
        reminder.validate_reminder_time(datetime.now(TZ) - timedelta(minutes=1))
    reminder.validate_reminder_time(datetime.now(TZ) + timedelta(minutes=1))


async def test_only_due_unfired_reminders_are_returned(db):
    past = reminder.store_reminder(
        "chat@lid", "due", datetime.now(TZ) - timedelta(minutes=5)
    )
    reminder.store_reminder(
        "chat@lid", "later", datetime.now(TZ) + timedelta(hours=1)
    )

    due = reminder.get_due_reminders()
    assert [row[0] for row in due] == [past]

    reminder.mark_fired(past)
    assert reminder.get_due_reminders() == []


async def test_scheduler_fires_and_marks_reminders(db, monkeypatch):
    monkeypatch.setattr(reminder, "POLL_INTERVAL", 0.01)
    sent = []

    def send(chat_jid, text, reply_to=None, reply_to_sender=None):
        sent.append((chat_jid, text, reply_to))
        return True, "ok"

    rid = reminder.store_reminder(
        "chat@lid",
        "drink water",
        datetime.now(TZ) - timedelta(minutes=1),
        message_id="MSG1",
        sender_jid="1@s.whatsapp.net",
    )

    task = asyncio.create_task(reminder.ReminderScheduler(send_fn=send).run())
    for _ in range(200):
        if sent:
            break
        await asyncio.sleep(0.01)
    task.cancel()

    assert sent, "scheduler never fired the due reminder"
    chat_jid, text, reply_to = sent[0]
    assert chat_jid == "chat@lid"
    assert "drink water" in text
    assert reply_to == "MSG1"
    assert reminder.get_due_reminders() == [], f"reminder {rid} was not marked fired"


async def test_scheduler_work_runs_off_the_event_loop(db, monkeypatch):
    """send_fn is a socket call with a 30s timeout.

    Called inline it froze all message processing for its duration, so it must
    run on a worker thread, not the loop thread.
    """
    monkeypatch.setattr(reminder, "POLL_INTERVAL", 0.01)
    loop_thread = threading.get_ident()
    send_threads: list[int] = []

    def send(chat_jid, text, reply_to=None, reply_to_sender=None):
        send_threads.append(threading.get_ident())
        return True, "ok"

    reminder.store_reminder("chat@lid", "hi", datetime.now(TZ) - timedelta(minutes=1))

    task = asyncio.create_task(reminder.ReminderScheduler(send_fn=send).run())
    for _ in range(200):
        if send_threads:
            break
        await asyncio.sleep(0.01)
    task.cancel()

    assert send_threads, "scheduler never fired the reminder"
    assert send_threads[0] != loop_thread, "send blocked the event loop"
