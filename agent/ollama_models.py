"""Discovery of the models an Ollama instance actually has installed.

Used by the `#model` command and the Gradio test UI. Ollama's native model list
lives at `/api/tags` on the bare root, while OLLAMA_BASE_URL points at the
OpenAI-compatible `/v1` path, so the suffix has to come off first.
"""

from __future__ import annotations

import httpx

from config import OLLAMA_BASE_URL

DEFAULT_BASE_URL = "http://localhost:11434"
_TIMEOUT = 5.0


def tags_url(base_url: str | None = None) -> str:
    """Build the /api/tags URL from an OpenAI-compat base URL."""
    url = (base_url or OLLAMA_BASE_URL or DEFAULT_BASE_URL).rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return f"{url.rstrip('/')}/api/tags"


async def list_models(base_url: str | None = None) -> list[str]:
    """Model names installed on the Ollama instance, sorted.

    Raises on failure — callers report the error rather than silently acting on
    an empty or guessed list.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(tags_url(base_url))
        resp.raise_for_status()
        models = resp.json().get("models", [])
    return sorted(m["name"] for m in models if m.get("name"))


def list_models_sync(base_url: str | None = None) -> list[str]:
    """Blocking variant for the Gradio UI, which builds its dropdown at startup."""
    resp = httpx.get(tags_url(base_url), timeout=_TIMEOUT)
    resp.raise_for_status()
    return sorted(m["name"] for m in resp.json().get("models", []) if m.get("name"))


def candidates(name: str, available: list[str]) -> list[str]:
    """Installed tags a partial name could mean, e.g. `gemma4` → the three sizes."""
    name = name.strip()
    return [m for m in available if m.startswith(f"{name}:")] if name else []


def resolve(name: str, available: list[str]) -> str | None:
    """Match a user-typed model name against the installed list.

    Ollama tags carry an explicit `:tag` suffix, so `#model set llama3.1` has to
    find `llama3.1:latest`. Returns None when there is no match, or when a
    prefix is ambiguous — silently picking one of several would be worse than
    asking the user to be specific.
    """
    name = name.strip()
    if not name:
        return None
    if name in available:
        return name
    if f"{name}:latest" in available:
        return f"{name}:latest"
    matches = candidates(name, available)
    return matches[0] if len(matches) == 1 else None
