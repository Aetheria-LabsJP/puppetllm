"""One-query loopback gate enforcing a hard limit on upstream Messages calls.

The CLI can recover/retry while its SDK consumer is already unwinding. Keep
this gate alive until SDK cleanup completes: only the first admitted HTTP call
may leave the process. No request body, credentials or upstream error text is
retained in diagnostics.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import math
import secrets
import socket
from typing import Any
from urllib.parse import urlsplit
import zlib

import httpx
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, StreamingResponse
import uvicorn

from .agent_sdk import await_cleanup


_HOP = frozenset({b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
                  b"te", b"trailer", b"transfer-encoding", b"upgrade", b"host", b"content-length"})
_USAGE = frozenset({"input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"})
_LIMIT = 64 * 1024


def _headers(items: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    excluded = set(_HOP)
    for name, value in items:
        if name.lower() == b"connection":
            excluded.update(token.strip().lower() for token in value.split(b","))
    return [(name, value) for name, value in items if name.lower() not in excluded]


def _numeric_usage(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    return {key: number for key, number in value.items() if key in _USAGE and (
        type(number) is int and number >= 0 or type(number) is float and math.isfinite(number) and number >= 0)}


class _Metadata:
    """Bound SSE line/event retention even for oversized private content blocks."""

    def __init__(self, callback) -> None:
        self.callback = callback
        self.line = bytearray()
        self.data = bytearray()
        self.discard = False
        self.drop_line = False

    def _event(self) -> None:
        if self.data and not self.discard:
            try:
                value = json.loads(self.data)
                if isinstance(value, dict) and value.get("type") == "message_delta":
                    delta = value.get("delta")
                    reason = delta.get("stop_reason") if isinstance(delta, dict) else None
                    self.callback(reason if isinstance(reason, str) else None,
                                  _numeric_usage(value.get("usage")))
            except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
                pass
        self.data.clear()
        self.discard = False

    def _line(self) -> None:
        line = bytes(self.line).removesuffix(b"\r")
        self.line.clear()
        if not line:
            self._event()
        elif not self.discard and line.startswith(b"data:"):
            value = line[5:]
            if value.startswith(b" "):
                value = value[1:]
            if len(self.data) + len(value) + 1 > _LIMIT:
                self.discard = True
                self.data.clear()
            else:
                if self.data:
                    self.data.extend(b"\n")
                self.data.extend(value)

    def feed(self, chunk: bytes) -> None:
        offset = 0
        while offset < len(chunk):
            ending = chunk.find(b"\n", offset)
            end = len(chunk) if ending < 0 else ending
            if not self.drop_line:
                if len(self.line) + len(self.data) + end - offset > _LIMIT:
                    self.line.clear()
                    self.data.clear()
                    self.discard = self.drop_line = True
                else:
                    self.line.extend(memoryview(chunk)[offset:end])
            if ending < 0:
                break
            if self.drop_line:
                self.drop_line = False
            else:
                self._line()
            offset = ending + 1

    def finish(self) -> None:
        if self.line and not self.drop_line:
            self._line()
        self._event()


class _GateServer(uvicorn.Server):
    # A per-query server must never replace the relay's SIGTERM/SIGINT handlers.
    @contextmanager
    def capture_signals(self):
        yield


class SDKAPIGate:
    """``async with SDKAPIGate(target) as gate`` exposes ``gate.base_url``.

    The target's base path and incoming query string are preserved. Only POST
    to the Messages API is accepted. Counters are reserved synchronously before
    reading a body or awaiting an upstream operation, including concurrent calls.
    Progress callbacks are optional synchronous methods; their failures do not
    change admission or break raw response delivery.
    """

    def __init__(self, target: str, max_requests: int = 1, progress: Any = None) -> None:
        try:
            parsed = urlsplit(target)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None
                    or parsed.query or parsed.fragment):
                raise ValueError()
            endpoint = httpx.URL(target)
            self._target_path = endpoint.raw_path.rstrip(b"/")
            self._upstream = str(endpoint.copy_with(raw_path=self._target_path + b"/v1/messages"))
        except (ValueError, TypeError, httpx.InvalidURL):
            raise ValueError("SDK API gate requires an HTTP target without credentials, query or fragment") from None
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("SDK API gate requires a positive request limit")
        self.max_requests = max_requests
        self.progress = progress
        self.started_requests = 0
        self.blocked_requests = 0
        self.stop_reason: str | None = None
        self.usage: dict[str, int | float] = {}
        self.status_codes: list[int] = []
        self.base_url: str | None = None
        self._prefix = b"/sdk-" + secrets.token_urlsafe(32).encode("ascii")
        self._path = self._prefix + self._target_path + b"/v1/messages"
        self._socket = self._client = self._server = self._server_task = self._close_task = None
        self._responses: set[httpx.Response] = set()
        self._handlers: set[asyncio.Task] = set()
        self._closing = False

    def _notify(self, name: str, *args: Any) -> None:
        try:
            callback = getattr(self.progress, name, None)
            if callback is not None:
                callback(*args)
        except Exception:
            pass  # Observability must never enable a retry or expose raw errors.

    def _metadata(self, stop_reason: str | None, usage: dict[str, int | float]) -> None:
        if stop_reason is not None:
            self.stop_reason = stop_reason
        self.usage.update(usage)
        self._notify("api_response_metadata", stop_reason, dict(usage))

    @staticmethod
    def _error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"type": "error", "error": {"type": "api_error", "message": message}},
                            status_code=status, headers={"x-should-retry": "false"})

    async def _handle(self, request: Request):
        if request.method != "POST" or request.scope.get("raw_path") != self._path:
            return self._error(404, "SDK API gate path is unavailable")
        if self._closing or self.started_requests >= self.max_requests:
            self.blocked_requests += 1
            self._notify("api_request_blocked")
            return self._error(503, "SDK API request limit reached; automatic retry is disabled")
        # No await before reserving the only billable request, even if the first
        # request is blocked in body upload, connection setup or response headers.
        self.started_requests += 1
        self._notify("api_request_started")
        task = asyncio.current_task()
        self._handlers.add(task)
        try:
            headers = _headers(list(request.headers.raw))
            headers = [(name, value) for name, value in headers if name.lower() != b"accept-encoding"]
            headers.append((b"accept-encoding", b"identity"))
            url = httpx.URL(self._upstream).copy_with(query=request.scope.get("query_string", b""))
            # Admission precedes upload buffering. Let HTTPX compute the correct
            # Content-Length instead of requiring upstream chunked-upload support.
            outgoing = self._client.build_request("POST", url, headers=headers, content=await request.body())
            response = await self._client.send(outgoing, stream=True, follow_redirects=False)
            self.status_codes.append(response.status_code)
            if 300 <= response.status_code < 400:
                await response.aclose()
                self._handlers.discard(task)
                return self._error(502, "SDK API redirects are not permitted")
            self._responses.add(response)
        except asyncio.CancelledError:
            self._handlers.discard(task)
            raise
        except Exception:
            self._handlers.discard(task)
            return self._error(502, "SDK API upstream connection failed")

        async def raw_body():
            observer = _Metadata(self._metadata)
            encoding = response.headers.get("content-encoding", "identity").strip().lower()
            decoder = (zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else
                       zlib.decompressobj() if encoding == "deflate" else None)
            observe = response.headers.get("content-type", "").split(";", 1)[0].lower() == "text/event-stream"
            try:
                async for chunk in response.aiter_raw():
                    if observe:
                        try:
                            if decoder is not None:
                                remaining = chunk
                                while remaining:
                                    observer.feed(decoder.decompress(remaining, _LIMIT))
                                    remaining = decoder.unconsumed_tail
                            elif encoding in ("identity", ""):
                                observer.feed(chunk)
                        except zlib.error:
                            observe = False
                    yield chunk
                if observe:
                    observer.finish()
            except (httpx.HTTPError, OSError):
                pass  # The incomplete original stream is not replaced with invented data.
            finally:
                try:
                    await response.aclose()
                finally:
                    self._responses.discard(response)
                    self._handlers.discard(task)

        result = StreamingResponse(raw_body(), status_code=response.status_code)
        result.raw_headers = _headers(list(response.headers.raw))
        return result

    async def __aenter__(self) -> SDKAPIGate:
        try:
            self._client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(retries=0, trust_env=False),
                timeout=httpx.Timeout(None, connect=10), follow_redirects=False, trust_env=False)
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("127.0.0.1", 0))
            self._socket.listen(128)
            self._socket.setblocking(False)
            port = self._socket.getsockname()[1]
            self.base_url = f"http://127.0.0.1:{port}" + (self._prefix + self._target_path).decode("ascii")
            app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
            app.add_api_route("/{path:path}", self._handle,
                              methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE"])
            self._server = _GateServer(uvicorn.Config(
                app, host="127.0.0.1", port=port, log_config=None, log_level="critical", access_log=False,
                lifespan="off", server_header=False, date_header=False, timeout_graceful_shutdown=1))
            self._server_task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
            async def started():
                while not self._server.started:
                    if self._server_task.done():
                        self._server_task.result()
                        raise RuntimeError("SDK API gate failed to start")
                    await asyncio.sleep(0.001)
            await asyncio.wait_for(started(), 5)
            return self
        except BaseException:
            await self.close()
            raise

    async def _close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        try:
            for task in tuple(self._handlers):
                task.cancel()
            if self._handlers:
                await asyncio.gather(*tuple(self._handlers), return_exceptions=True)
            await asyncio.gather(*(response.aclose() for response in tuple(self._responses)), return_exceptions=True)
            if self._client is not None:
                await self._client.aclose()
        finally:
            try:
                if self._server_task is not None:
                    done, _ = await asyncio.wait({self._server_task}, timeout=3)
                    if not done:
                        self._server_task.cancel()
                    await asyncio.gather(self._server_task, return_exceptions=True)
            finally:
                if self._socket is not None:
                    self._socket.close()
                self._responses.clear()
                self._handlers.clear()

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await await_cleanup(self._close_task)

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    @property
    def used(self) -> bool:
        """Whether the SDK sent at least one request through this gate."""
        return self.started_requests > 0
