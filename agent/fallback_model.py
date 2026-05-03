"""Failover wrapper that routes model calls between a primary and backup Ollama.

Backup is treated as a degraded experience: we leave it as soon as the primary
recovers. State is shared across all FallbackModel instances that point at the
same FallbackRouter, so a primary outage on the text path immediately reroutes
the vision path too.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError
from agents.models.interface import Model

from logging_setup import logger


# Exceptions that indicate the primary endpoint itself is unhealthy.
# 4xx errors (bad request, model-not-found, auth) are NOT in this list — those
# would fail identically against the backup, so we let them propagate.
_PRIMARY_FAILURE_EXC: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    asyncio.TimeoutError,
)


def _is_primary_failure(exc: BaseException) -> bool:
    if isinstance(exc, _PRIMARY_FAILURE_EXC):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code is not None and exc.status_code >= 500
    return False


class FallbackRouter:
    """Tracks whether we're currently on backup and gates probe attempts."""

    def __init__(self, sticky_seconds: float):
        self._sticky_seconds = sticky_seconds
        self._on_backup = False
        self._backup_started_at = 0.0
        self._lock = asyncio.Lock()

    async def should_use_backup(self) -> bool:
        async with self._lock:
            return self._on_backup

    async def due_for_probe(self) -> bool:
        async with self._lock:
            if not self._on_backup:
                return False
            return (time.monotonic() - self._backup_started_at) >= self._sticky_seconds

    async def mark_primary_failed(self) -> None:
        async with self._lock:
            if not self._on_backup:
                logger.warning("Primary Ollama failed; switching to backup")
            self._on_backup = True
            self._backup_started_at = time.monotonic()

    async def mark_primary_healthy(self) -> None:
        async with self._lock:
            if self._on_backup:
                logger.warning("Primary Ollama recovered; leaving backup")
            self._on_backup = False
            self._backup_started_at = 0.0

    async def extend_backup_window(self) -> None:
        async with self._lock:
            logger.warning("Primary still failing on probe; staying on backup")
            self._backup_started_at = time.monotonic()


class FallbackModel(Model):
    """Routes get_response/stream_response between primary and backup models."""

    def __init__(
        self,
        primary: Model,
        backup: Model | None,
        router: FallbackRouter,
        primary_timeout_seconds: float,
    ):
        self._primary = primary
        self._backup = backup
        self._router = router
        self._timeout = primary_timeout_seconds

    async def get_response(self, *args: Any, **kwargs: Any):
        if self._backup is None:
            return await self._primary.get_response(*args, **kwargs)

        if await self._router.should_use_backup():
            if await self._router.due_for_probe():
                try:
                    result = await asyncio.wait_for(
                        self._primary.get_response(*args, **kwargs),
                        timeout=self._timeout,
                    )
                    await self._router.mark_primary_healthy()
                    return result
                except Exception as e:
                    if _is_primary_failure(e):
                        await self._router.extend_backup_window()
                    else:
                        raise
            return await self._backup.get_response(*args, **kwargs)

        try:
            return await asyncio.wait_for(
                self._primary.get_response(*args, **kwargs),
                timeout=self._timeout,
            )
        except Exception as e:
            if not _is_primary_failure(e):
                raise
            logger.warning(f"Primary Ollama call failed ({e!r}); retrying on backup")
            await self._router.mark_primary_failed()
            return await self._backup.get_response(*args, **kwargs)

    async def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        # No timeout on streams — once chunks start flowing we trust them.
        # We only fall back if the stream raises a connection/5xx failure
        # before yielding anything.
        if self._backup is None:
            async for chunk in self._primary.stream_response(*args, **kwargs):
                yield chunk
            return

        use_backup = await self._router.should_use_backup()
        if use_backup and await self._router.due_for_probe():
            try:
                async for chunk in self._primary.stream_response(*args, **kwargs):
                    yield chunk
                await self._router.mark_primary_healthy()
                return
            except Exception as e:
                if _is_primary_failure(e):
                    await self._router.extend_backup_window()
                else:
                    raise
            use_backup = True

        if use_backup:
            async for chunk in self._backup.stream_response(*args, **kwargs):
                yield chunk
            return

        try:
            async for chunk in self._primary.stream_response(*args, **kwargs):
                yield chunk
            return
        except Exception as e:
            if not _is_primary_failure(e):
                raise
            logger.warning(f"Primary Ollama stream failed ({e!r}); retrying on backup")
            await self._router.mark_primary_failed()
            async for chunk in self._backup.stream_response(*args, **kwargs):
                yield chunk
