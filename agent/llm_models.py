"""Discovery of the models an LLM server actually serves.

Used by the `#model` command and the Gradio test UI. Reads the OpenAI-compatible
`/v1/models` list, which SGLang, vLLM and Ollama all serve, so it works with
whichever engine LLM_BASE_URL points at.
"""

from __future__ import annotations

import httpx

from config import LLM_BASE_URL

DEFAULT_BASE_URL = "http://localhost:30000/v1"
_TIMEOUT = 5.0


def models_url(base_url: str | None = None) -> str:
    """Build the /v1/models URL, with or without the /v1 suffix on the base."""
    url = (base_url or LLM_BASE_URL or DEFAULT_BASE_URL).rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return f"{url}/models"


def _names(payload: dict) -> list[str]:
    return sorted(m["id"] for m in payload.get("data", []) if m.get("id"))


async def list_models(base_url: str | None = None) -> list[str]:
    """Model names the server serves, sorted.

    Raises on failure — callers report the error rather than silently acting on
    an empty or guessed list.
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(models_url(base_url))
        resp.raise_for_status()
        return _names(resp.json())


def list_models_sync(base_url: str | None = None) -> list[str]:
    """Blocking variant for the Gradio UI, which builds its dropdown at startup."""
    resp = httpx.get(models_url(base_url), timeout=_TIMEOUT)
    resp.raise_for_status()
    return _names(resp.json())


def candidates(name: str, available: list[str]) -> list[str]:
    """Served models a partial name could mean, e.g. `gemma4` → the three sizes."""
    name = name.strip()
    return [m for m in available if m.startswith(f"{name}:")] if name else []


def resolve(name: str, available: list[str]) -> str | None:
    """Match a user-typed model name against the served list.

    Ollama tags carry an explicit `:tag` suffix, so `#model set llama3.1` has to
    find `llama3.1:latest`; SGLang serves bare names like `qwen3.8`. Returns
    None when there is no match, or when a prefix is ambiguous — silently
    picking one of several would be worse than asking the user to be specific.
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
