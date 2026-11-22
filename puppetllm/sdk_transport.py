"""Local workarounds for the optional Agent SDK's subprocess transport.

Keep the SDK-internal dependency here. With SDK 0.2.182 its buffer guard counts
characters rather than UTF-8 bytes, and close() can leave an exited process's
stdout pipe open. Recheck these workarounds when upgrading the SDK.
"""

from __future__ import annotations

import asyncio

import anyio
from anyio.abc import ObjectReceiveStream
from claude_agent_sdk import CLIJSONDecodeError
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from .agent_sdk import await_cleanup


class _ByteLimitedTextStream(ObjectReceiveStream[str]):
    """Bound each NDJSON line before the SDK buffers or parses its contents."""

    def __init__(self, stream: ObjectReceiveStream[str], limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self._line_bytes = 0

    async def receive(self) -> str:
        chunk = await self._stream.receive()
        parts = chunk.split("\n")
        for index, part in enumerate(parts):
            self._line_bytes += len(part.encode("utf-8"))
            if self._line_bytes > self._limit:
                # Include sizes only, never any output or tool arguments.
                raise CLIJSONDecodeError(
                    "CLI JSON line exceeded the UTF-8 byte limit",
                    ValueError(f"Buffer size {self._line_bytes} exceeds limit {self._limit}"),
                )
            if index < len(parts) - 1:
                self._line_bytes = 0
        return chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class RelaySDKTransport(SubprocessCLITransport):
    async def connect(self) -> None:
        self._cleanup_task = None
        await super().connect()
        if self._stdout_stream is not None:
            self._stdout_stream = _ByteLimitedTextStream(self._stdout_stream, self._max_buffer_size)

    async def close(self) -> None:
        # SDK 0.2.182 uses detached tasks, not a TaskGroup entered in connect().
        # It is safe to run subprocess cleanup in its own asyncio task. Share
        # one task between concurrent/repeated close() calls so only one owns
        # the process fields and terminate/kill escalation.
        task = getattr(self, "_cleanup_task", None)
        if task is None:
            task = self._cleanup_task = asyncio.create_task(self._close_subprocess())
        await await_cleanup(task)

    async def _close_subprocess(self) -> None:
        process = self._process
        try:
            await super().close()
        finally:
            # Waiting for exit alone does not close a paused stdout pipe. Use
            # AnyIO's public Process cleanup while the event loop is alive.
            # Leave live-process termination/reaping to the SDK lifecycle.
            if process is not None and process.returncode is not None:
                with anyio.move_on_after(1, shield=True):
                    await process.aclose()
        if process is not None and process.returncode is None:
            # The SDK retains an unreaped child for its atexit handler. Do not
            # report successful cleanup or permit a successful outer response.
            raise RuntimeError("Agent SDK subprocess did not exit during bounded cleanup")
