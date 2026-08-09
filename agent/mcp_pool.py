"""Long-lived pool of MCP servers.

Entering ``mcp_stack()`` per message costs seconds: ``npx`` package resolution,
Node startup, ``uvx`` environment resolution (potentially a network round-trip)
and a Playwright browser profile spin-up — all repeated for every single
message.  This module starts each configured server **once** and keeps it
connected for the process lifetime, handing out the appropriate subset
(privileged vs basic) per message.

Each server is owned by a supervisor task that connects it, waits for a restart
request, then cleans it up.  Connect and cleanup therefore always happen in the
same task, which anyio's cancel scopes require — this is why the servers cannot
simply be parked in a module-level ``AsyncExitStack``.

A server whose transport dies (or that never came up) is excluded from
``servers()`` and respawned in the background with exponential backoff, so a
crashed MCP no longer requires an agent restart.
"""

import asyncio

import anyio
from agents.mcp import MCPServer, MCPServerStdio
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, CallToolResult, TextContent

import write_guard
from config import ALWAYS_SERVERS, MCP_REGISTRY, PRIVILEGED_SERVERS
from logging_setup import logger

# Errors that mean "the pipe to the subprocess is gone" — the only class of
# failure worth tearing the server down for.  Tool-level errors come back as a
# CallToolResult with isError set, not as exceptions.
_TRANSPORT_ERRORS = (
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
    ConnectionError,
    BrokenPipeError,
)

# Backoff bounds for respawning a server that failed to connect.
_RESTART_BACKOFF_MIN = 2.0
_RESTART_BACKOFF_MAX = 120.0

# How long a request already in flight waits for a mid-run restart to finish.
_RESTART_GRACE_SECONDS = 30.0


def _is_transport_error(exc: BaseException) -> bool:
    """True if ``exc`` indicates the stdio transport is dead.

    A subprocess that dies mid-request surfaces as an McpError carrying the
    CONNECTION_CLOSED code rather than as a raw anyio stream error, so both
    shapes have to be recognised.
    """
    if isinstance(exc, _TRANSPORT_ERRORS):
        return True
    if isinstance(exc, McpError):
        return getattr(exc.error, "code", None) == CONNECTION_CLOSED
    # anyio wraps concurrent failures in an ExceptionGroup.
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_transport_error(e) for e in exc.exceptions)
    return False


class ManagedMCPServer(MCPServer):
    """A single MCP server kept alive across messages, with auto-restart.

    Implements the ``MCPServer`` interface by delegating to an inner
    ``MCPServerStdio`` whose lifecycle is owned by ``_supervise()``.
    """

    def __init__(self, name: str, entry: dict, tool_filter=None):
        super().__init__()
        self._name = name
        self._params = entry["params"]
        self._timeout = entry["timeout"]
        self._tool_filter = tool_filter

        self._inner: MCPServerStdio | None = None
        self._generation = 0  # bumped on every connect attempt, success or not
        self._healthy = False
        self._stopping = False
        self._restart_requested = asyncio.Event()
        self._state = asyncio.Condition()
        self._task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def healthy(self) -> bool:
        return self._healthy and self._inner is not None

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """Spawn the supervisor and wait for the first connect attempt.

        Returns True if the server came up.  A failed start is not fatal: the
        supervisor keeps retrying in the background.
        """
        self._task = asyncio.create_task(self._supervise(), name=f"mcp-{self._name}")
        async with self._state:
            await self._state.wait_for(lambda: self._generation > 0 or self._stopping)
        return self.healthy

    async def stop(self) -> None:
        """Shut the server down and stop the supervisor."""
        self._stopping = True
        self._restart_requested.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=15)
            except TimeoutError:
                logger.warning(f"MCP '{self._name}' did not shut down in time; cancelling")
                self._task.cancel()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"MCP '{self._name}' shutdown error: {e}")

    async def _supervise(self) -> None:
        """Own the inner server: connect → serve → cleanup, forever."""
        backoff = _RESTART_BACKOFF_MIN
        while not self._stopping:
            inner = MCPServerStdio(
                params=self._params,
                client_session_timeout_seconds=self._timeout,
                tool_filter=self._tool_filter,
                cache_tools_list=True,
                name=self._name,
            )
            try:
                await inner.connect()
            except Exception as e:
                await self._publish(inner=None, healthy=False, bump=True)
                logger.error(f"MCP '{self._name}' failed to start: {e}")
                if self._stopping:
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RESTART_BACKOFF_MAX)
                continue

            backoff = _RESTART_BACKOFF_MIN
            await self._publish(inner=inner, healthy=True, bump=True)
            logger.info(f"MCP '{self._name}' connected")

            await self._restart_requested.wait()

            # Not a new generation yet — waiters must keep blocking until the
            # replacement is actually connected (or has failed to connect).
            await self._publish(inner=None, healthy=False, bump=False)
            try:
                await inner.cleanup()
            except Exception as e:
                logger.warning(f"MCP '{self._name}' cleanup error: {e}")
            if not self._stopping:
                logger.info(f"MCP '{self._name}' restarting")

        logger.info(f"MCP '{self._name}' supervisor stopped")

    async def _publish(
        self, inner: MCPServerStdio | None, healthy: bool, bump: bool
    ) -> None:
        """Publish new state and wake everyone waiting on it.

        ``bump`` advances the generation counter, which is what ``_restart_after``
        waits on — so only connect *outcomes* bump it, never teardown.
        """
        async with self._state:
            self._inner = inner
            self._healthy = healthy
            if healthy:
                # Any restart asked for while we were down referred to the
                # generation we just replaced; it is already satisfied.
                self._restart_requested.clear()
            if bump:
                self._generation += 1
            self._state.notify_all()

    async def _restart_after(self, generation: int) -> None:
        """Request a restart of ``generation`` and wait for its replacement.

        A no-op if another caller already cycled past that generation.
        """
        async with self._state:
            if self._generation != generation:
                return  # already replaced by someone else
            self._restart_requested.set()
        try:
            async with self._state:
                await asyncio.wait_for(
                    self._state.wait_for(lambda: self._generation > generation),
                    timeout=_RESTART_GRACE_SECONDS,
                )
        except TimeoutError:
            logger.warning(f"MCP '{self._name}' restart timed out after {generation}")

    # ── delegation ──────────────────────────────────────────────────────────

    async def _current(self) -> tuple[MCPServerStdio, int]:
        async with self._state:
            if self._inner is None or not self._healthy:
                raise RuntimeError(f"MCP server '{self._name}' is not available")
            return self._inner, self._generation

    async def _delegate(self, call):
        """Run ``call(inner)``, restarting the server once on transport death."""
        inner, generation = await self._current()
        try:
            return await call(inner)
        except Exception as e:
            if not _is_transport_error(e):
                raise
            logger.warning(f"MCP '{self._name}' transport died ({e}); restarting")
            await self._restart_after(generation)

        inner, generation = await self._current()
        try:
            return await call(inner)
        except Exception as e:
            if _is_transport_error(e):
                # The retry killed it too (e.g. the call itself is fatal to the
                # server). Cycle it anyway so the next caller finds it up.
                logger.warning(f"MCP '{self._name}' died again on retry; restarting")
                await self._restart_after(generation)
            raise

    async def connect(self) -> None:
        """No-op: the pool owns connection lifecycle."""

    async def cleanup(self) -> None:
        """No-op: the pool owns connection lifecycle. Use ``stop()``."""

    async def list_tools(self, run_context=None, agent=None):
        return await self._delegate(lambda s: s.list_tools(run_context, agent))

    async def call_tool(self, tool_name: str, arguments: dict | None):
        blocked = write_guard.refusal(self._name, tool_name)
        if blocked:
            # Returned as a tool error rather than raised, so the model can
            # recover by asking the user instead of aborting the run.
            return CallToolResult(
                content=[TextContent(type="text", text=blocked)], isError=True
            )
        return await self._delegate(lambda s: s.call_tool(tool_name, arguments))

    async def list_prompts(self):
        return await self._delegate(lambda s: s.list_prompts())

    async def get_prompt(self, name: str, arguments: dict | None = None):
        return await self._delegate(lambda s: s.get_prompt(name, arguments))

    def invalidate_tools_cache(self) -> None:
        if self._inner is not None:
            self._inner.invalidate_tools_cache()


class MCPPool:
    """Process-wide registry of long-lived MCP servers."""

    def __init__(self):
        self._servers: dict[str, ManagedMCPServer] = {}
        self._start_lock = asyncio.Lock()
        self._started = False

    async def ensure_started(self) -> None:
        """Start every gated server once. Safe to call from any code path."""
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            await self._start()
            self._started = True

    async def _start(self) -> None:
        from tools_config import make_tool_filter

        names = [
            name
            for name in PRIVILEGED_SERVERS + ALWAYS_SERVERS
            if _gate_open(name)
        ]
        self._servers = {
            name: ManagedMCPServer(name, MCP_REGISTRY[name], make_tool_filter(name))
            for name in names
        }

        results = await asyncio.gather(
            *(srv.start() for srv in self._servers.values()), return_exceptions=True
        )
        up = [n for n, ok in zip(self._servers, results, strict=True) if ok is True]
        down = [n for n in self._servers if n not in up]
        logger.info(
            f"MCP pool started: {len(up)}/{len(self._servers)} up "
            f"(up={up or '-'}, retrying={down or '-'})"
        )

    def servers(self, is_privileged: bool = False) -> list[MCPServer]:
        """Return the healthy servers for this caller, in registry order.

        Brave is ordered last so the model's recency bias keeps web-search
        tools visible even alongside many tools from other servers.
        """
        order = (PRIVILEGED_SERVERS if is_privileged else []) + ALWAYS_SERVERS
        return [
            self._servers[name]
            for name in order
            if name in self._servers and self._servers[name].healthy
        ]

    async def stop(self) -> None:
        await asyncio.gather(
            *(srv.stop() for srv in self._servers.values()), return_exceptions=True
        )
        self._servers = {}
        self._started = False


def _gate_open(name: str) -> bool:
    """Evaluate a registry entry's optional ``gate`` callable."""
    from config import WORKSPACE_MCP_PATH

    gate = MCP_REGISTRY[name].get("gate")
    if gate is None:
        return True
    if gate():
        return True
    if name == "workspace" and WORKSPACE_MCP_PATH:
        logger.warning(f"Workspace MCP not found at {WORKSPACE_MCP_PATH}")
    return False


# Process-wide singleton.
mcp_pool = MCPPool()
