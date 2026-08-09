"""Tests for #model — listing and live-switching the Ollama model."""

import importlib

import pytest

import command_handlers
import config
import message_handler
import model_override
import ollama_models
from message_handler import match_command

pytestmark = pytest.mark.asyncio

INSTALLED = ["gemma3:27b", "llama3.1:latest", "qwen3:14b"]


# ── Command matching ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content,expected",
    [
        ("#model", "#model"),
        ("#model list", "#model"),
        ("#MODEL set qwen3:14b", "#model"),
        ("which #model are you running?", None),
        ("#models", None),
    ],
)
async def test_model_command_matching(content, expected):
    assert match_command(content) == expected


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sent(monkeypatch) -> list:
    """Capture outbound replies instead of sending them."""
    calls: list = []
    import reply

    def fake_send(*args, **kwargs):
        calls.append(args)
        return True, "ok"

    monkeypatch.setattr(message_handler, "whatsapp_send_message", fake_send)
    monkeypatch.setattr(reply, "whatsapp_send_message", fake_send)
    return calls


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Point the override file at tmp_path and restore the live model after."""
    monkeypatch.setattr(
        model_override, "OVERRIDE_PATH", str(tmp_path / "model_override.json")
    )
    monkeypatch.setattr(
        ollama_models, "list_models", _stub_list(INSTALLED)
    )
    before_text, before_vision = config.MODEL_NAME, config.VISION_MODEL_NAME
    before_model = config._cached_model
    yield
    config.set_text_model(before_text or "restore")
    config.set_vision_model(before_vision)
    config.MODEL_NAME = before_text
    assert config._cached_model is before_model


def _stub_list(models):
    async def _list(base_url=None):
        return list(models)

    return _list


async def _run(content, phone="15551234567"):
    await message_handler.process_message(
        {
            "chat_jid": "c@lid",
            "phone_number": phone,
            "sender_jid": f"{phone}@s.whatsapp.net",
            "id": "MSG1",
            "content": content,
            "is_from_me": False,
        }
    )


@pytest.fixture
def owner(monkeypatch):
    """Make the test sender the instance owner."""
    monkeypatch.setattr(message_handler, "ALLOWED_SENDERS", ["15551234567"])


# ── Access control ───────────────────────────────────────────────────────────


async def test_model_command_is_owner_only(sent, monkeypatch):
    monkeypatch.setattr(message_handler, "ALLOWED_SENDERS", ["19998887777"])
    await _run("#model")
    assert len(sent) == 1
    assert "🔒" in sent[0][1]


# ── Listing ──────────────────────────────────────────────────────────────────


async def test_list_marks_the_active_models(sent, owner, isolated):
    config.set_text_model("qwen3:14b")
    config.set_vision_model("gemma3:27b")

    await _run("#model")

    text = sent[0][1]
    for name in INSTALLED:
        assert name in text
    assert "`qwen3:14b` ✅" in text
    assert "`gemma3:27b` 👁" in text


async def test_list_reports_an_unreachable_ollama(sent, owner, isolated, monkeypatch):
    async def boom(base_url=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ollama_models, "list_models", boom)
    await _run("#model list")
    assert "Couldn't reach Ollama" in sent[0][1]


# ── Switching ────────────────────────────────────────────────────────────────


async def test_unknown_model_is_refused_and_changes_nothing(sent, owner, isolated):
    config.set_text_model("qwen3:14b")

    await _run("#model set llmaa3")

    assert config.MODEL_NAME == "qwen3:14b"
    assert model_override.load() == {}
    text = sent[0][1]
    assert "No installed model matches" in text
    assert "llama3.1:latest" in text  # shows what is available


async def test_switch_mutates_the_shared_model_object(sent, owner, isolated):
    """The regression that matters.

    message_handler, briefing_executor and agent_factory all import
    _cached_model by value, so rebinding it would silently leave briefings and
    reminder parsing on the old model.
    """
    config.set_text_model("qwen3:14b")
    shared = config._cached_model

    await _run("#model set llama3.1:latest")

    assert config.MODEL_NAME == "llama3.1:latest"
    assert config._cached_model is shared
    assert shared.model == "llama3.1:latest"
    assert "`qwen3:14b` → `llama3.1:latest`" in sent[0][1]


async def test_bare_name_is_shorthand_for_set(sent, owner, isolated):
    await _run("#model qwen3:14b")
    assert config.MODEL_NAME == "qwen3:14b"


async def test_partial_name_resolves_to_the_installed_tag(sent, owner, isolated):
    await _run("#model set llama3.1")
    assert config.MODEL_NAME == "llama3.1:latest"


async def test_vision_switch_leaves_the_text_model_alone(sent, owner, isolated):
    config.set_text_model("qwen3:14b")

    await _run("#model vision gemma3:27b")

    assert config.VISION_MODEL_NAME == "gemma3:27b"
    assert config.MODEL_NAME == "qwen3:14b"
    assert config._cached_vision_model.model == "gemma3:27b"
    assert model_override.load() == {"vision": "gemma3:27b"}


async def test_switch_clears_the_agent_cache(sent, owner, isolated):
    from agent_factory import agent_factory

    agent_factory._agents[("c@lid", "qwen3:14b")] = (object(), 0.0)
    await _run("#model set qwen3:14b")
    assert agent_factory._agents == {}


async def test_missing_name_gets_usage(sent, owner, isolated):
    before = config.MODEL_NAME
    await _run("#model set")
    assert "Usage: #model set <name>" in sent[0][1]
    assert config.MODEL_NAME == before


# ── Persistence ──────────────────────────────────────────────────────────────


async def test_switch_persists_and_is_reapplied_on_import(sent, owner, isolated):
    await _run("#model set qwen3:14b")
    assert model_override.load() == {"text": "qwen3:14b"}

    # A restart re-runs config's apply block against the stored file.
    config.set_text_model("gemma3:27b")
    override = model_override.load()
    if override.get("text"):
        config.set_text_model(override["text"])
    assert config.MODEL_NAME == "qwen3:14b"


async def test_reset_drops_the_override(sent, owner, isolated, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "llama3.1:latest")
    monkeypatch.setenv("VISION_MODEL_NAME", "gemma3:27b")
    await _run("#model set qwen3:14b")

    await _run("#model reset")

    assert model_override.load() == {}
    assert config.MODEL_NAME == "llama3.1:latest"
    assert config.VISION_MODEL_NAME == "gemma3:27b"


async def test_unreadable_override_file_is_not_fatal(monkeypatch, tmp_path):
    path = tmp_path / "model_override.json"
    path.write_text("{not json")
    monkeypatch.setattr(model_override, "OVERRIDE_PATH", str(path))
    assert model_override.load() == {}


# ── resolve() ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "typed,expected",
    [
        ("qwen3:14b", "qwen3:14b"),
        ("llama3.1", "llama3.1:latest"),  # implicit :latest
        ("gemma3", "gemma3:27b"),  # unique prefix
        ("nope", None),
        ("", None),
    ],
)
async def test_resolve(typed, expected):
    assert ollama_models.resolve(typed, INSTALLED) == expected


async def test_ambiguous_prefix_is_refused():
    """Silently picking one of several tags would be worse than asking."""
    assert ollama_models.resolve("qwen3", ["qwen3:14b", "qwen3:32b"]) is None


async def test_ambiguous_name_lists_only_the_candidates(
    sent, owner, isolated, monkeypatch
):
    """`gemma3` when several tags share the prefix is a question, not an error."""
    monkeypatch.setattr(
        ollama_models, "list_models", _stub_list(["gemma3:12b", "gemma3:27b", "qwen3:14b"])
    )

    await _run("#model set gemma3")

    text = sent[0][1]
    assert "ambiguous" in text
    assert "gemma3:12b" in text and "gemma3:27b" in text
    assert "qwen3:14b" not in text


@pytest.mark.parametrize(
    "base,expected",
    [
        ("http://host:11434/v1", "http://host:11434/api/tags"),
        ("http://host:11434", "http://host:11434/api/tags"),
        ("http://host:11434/v1/", "http://host:11434/api/tags"),
    ],
)
async def test_tags_url_strips_the_openai_compat_suffix(base, expected):
    assert ollama_models.tags_url(base) == expected


async def test_command_handlers_module_imports_cleanly():
    importlib.reload(ollama_models)
    assert hasattr(command_handlers, "handle_model_command")
