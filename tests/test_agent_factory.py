"""Tests for agent caching, per-chat locking, and reminder parsing."""

import asyncio
import pytest
from agents.models.interface import Model

import agent_factory
from agent_factory import AgentFactory, parse_remindme_with_agent
from fallback_model import FallbackModel, FallbackRouter
from models import ReminderParsed

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def isolated_sessions(tmp_path, monkeypatch):
    import session_store

    monkeypatch.setattr(session_store, "SESSIONS_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setattr(session_store, "_sessions", {})


class FakeModel(Model):
    """Minimal Model the Agent constructor will accept."""

    def __init__(self, name="test-model"):
        self.model = name

    async def get_response(self, *args, **kwargs):  # pragma: no cover - never run
        raise NotImplementedError

    def stream_response(self, *args, **kwargs):  # pragma: no cover - never run
        raise NotImplementedError


def _model(name="test-model"):
    return FakeModel(name)


async def test_fallback_model_exposes_a_stable_name():
    """Cache keys used repr() (with an object address) before this property."""
    model = FallbackModel(
        primary=_model("qwen3.5:35b"),
        backup=None,
        router=FallbackRouter(sticky_seconds=1),
        primary_timeout_seconds=1,
    )
    assert model.model == "qwen3.5:35b"
    assert model.model == FallbackModel(
        primary=_model("qwen3.5:35b"),
        backup=None,
        router=FallbackRouter(sticky_seconds=1),
        primary_timeout_seconds=1,
    ).model


async def test_agent_is_cached_per_chat_and_model():
    factory = AgentFactory()
    text, vision = _model("text"), _model("vision")

    a1, s1 = await factory.get_agent("chat@lid", [], text, "instr")
    a2, s2 = await factory.get_agent("chat@lid", [], text, "instr")
    a3, s3 = await factory.get_agent("chat@lid", [], vision, "instr")

    assert a1 is a2
    assert a3 is not a1
    assert list(factory._agents) == [("chat@lid", "text"), ("chat@lid", "vision")]
    # One history per chat, shared by the text and vision agents.
    assert s1 is s2 is s3


async def test_session_outlives_agent_eviction(monkeypatch):
    monkeypatch.setattr(agent_factory, "MAX_AGENTS", 1)
    factory = AgentFactory()

    _, session_before = await factory.get_agent("a@lid", [], _model(), "i")
    await factory.get_agent("b@lid", [], _model(), "i")  # evicts a@lid
    _, session_after = await factory.get_agent("a@lid", [], _model(), "i")

    assert ("a@lid", "test-model") in factory._agents  # recreated after eviction
    assert session_before is session_after, "history must not die with the agent"


async def test_lock_is_per_chat():
    factory = AgentFactory()
    assert factory.lock_for("a@lid") is factory.lock_for("a@lid")
    assert factory.lock_for("a@lid") is not factory.lock_for("b@lid")


async def test_reminder_parser_does_not_mutate_the_shared_agent(monkeypatch):
    """Two concurrent #remindme messages must not share instructions."""
    seen_agents = []
    seen_instructions = []

    class FakeResult:
        final_output = ReminderParsed(
            reminder_message="call dentist", remind_at="2030-01-01T10:00:00"
        )

    class FakeRunner:
        @staticmethod
        async def run(agent, content):
            seen_agents.append(agent)
            seen_instructions.append(agent.instructions)
            await asyncio.sleep(0.01)  # force the two calls to overlap
            return FakeResult()

    monkeypatch.setattr(agent_factory, "Runner", FakeRunner)

    await asyncio.gather(
        parse_remindme_with_agent("#remindme in 5 minutes call dentist"),
        parse_remindme_with_agent("#remindme tomorrow at 9 buy milk"),
    )

    assert seen_agents[0] is not seen_agents[1], "each call needs its own agent"
    assert agent_factory._reminder_parser_agent not in seen_agents
    assert agent_factory._reminder_parser_agent.instructions == ""
    assert all("current date and time is" in i for i in seen_instructions)


async def test_reminder_parser_returns_parsed_values(monkeypatch):
    class FakeResult:
        final_output = ReminderParsed(
            reminder_message="stretch", remind_at="2030-06-01T08:30:00"
        )

    class FakeRunner:
        @staticmethod
        async def run(agent, content):
            return FakeResult()

    monkeypatch.setattr(agent_factory, "Runner", FakeRunner)

    remind_at, text = await parse_remindme_with_agent("#remindme ...")
    assert text == "stretch"
    assert remind_at.year == 2030 and remind_at.hour == 8
    assert remind_at.tzinfo is not None, "naive times must be pinned to the local TZ"


async def test_reminder_parser_rejects_garbage(monkeypatch):
    class FakeResult:
        final_output = ReminderParsed(reminder_message="x", remind_at="not a time")

    class FakeRunner:
        @staticmethod
        async def run(agent, content):
            return FakeResult()

    monkeypatch.setattr(agent_factory, "Runner", FakeRunner)

    with pytest.raises(ValueError):
        await parse_remindme_with_agent("#remindme whenever")
