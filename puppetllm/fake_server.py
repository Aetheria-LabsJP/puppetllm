"""Anthropic / Bedrock / OpenAI compatible fake server (for debug/regression).

See **README.md** for the formal spec, usage examples, and design decisions.

The Anthropic SDK honors the `ANTHROPIC_BASE_URL` environment variable, so pointing
it at localhost lets a human or another agent supply the response while still exercising
the SDK's real path (HTTP → SSE → stream event parsing). The Bedrock SDK
(`AnthropicBedrock` / boto3) and the OpenAI SDK can likewise be pointed at this server
by swapping their base_url.

Architecture (provider-independent canonical core + adapters):
- This file = canonical core: normalized snapshot management + /_control/* + cost/cache computation
- Anthropic path: `/v1/messages` (implemented in this file)
- Bedrock path  : `providers/bedrock.py` (`/model/{id}/invoke[-with-response-stream]`)
- OpenAI path   : `providers/openai.py` (`/v1/chat/completions`)
- Response content blocks / control API are provider-common (injection uses the same /_control/respond)

Implemented surface:
- POST /v1/messages                  — Anthropic compatible (SSE / single JSON)
- POST /model/{id}/invoke[...]        — Bedrock compatible (added by providers/bedrock.py)
- POST /v1/chat/completions          — OpenAI compatible (added by providers/openai.py)
- GET  /_control/pending             — pending requests (including provider)
- GET  /_control/wait_for_pending    — long-poll until the next pending arrives
- POST /_control/respond             — inject a response into a pending request
- POST /_control/auto                — simple auto-response (text only)
- POST /_control/error               — inject an HTTP error response
- GET  /_control/history             — (request, response, usage, cost, cache) history
- GET  /_control/stats               — cumulative summary of estimated cost, tokens, and cache
- GET  /_control/cache               — pseudo prompt-cache index
- POST /_control/clear               — empty pending/history/cache
- GET  /_control/health              — health check

The control endpoints are localhost only (for debugging), with no authorization.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import pricing
from .cache_sim import CacheSimulator, analyze_request


# ── Configuration (environment variables) ─────────────────────────────

# Pseudo prompt-cache TTL (seconds) and whether to honor the TTL.
def _parse_cache_ttl(v: str | None) -> float:
    # A non-numeric / non-finite / non-positive value must not crash startup or produce a
    # degenerate cache (inf → never pruned = unbounded growth; nan/negative → always-cold +
    # never pruned). Fall back to the 300s default. Mirrors _parse_cache_min's strictness.
    if v is None:
        return 300.0
    try:
        n = float(v)
    except ValueError:
        return 300.0
    return n if math.isfinite(n) and n > 0 else 300.0


_CACHE_TTL = _parse_cache_ttl(os.environ.get("PUPPETLLM_CACHE_TTL"))
_CACHE_HONOR_TTL = os.environ.get("PUPPETLLM_CACHE_HONOR_TTL", "1") != "0"
# Minimum cacheable threshold. Unset/invalid/negative = per-model (Opus 4096/Sonnet 1024 etc.). `0` disables it (cache all prefixes).
def _parse_cache_min(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n >= 0 else None  # negative values are meaningless (effectively cache-all), so fall back to per-model


_CACHE_MIN_TOKENS = _parse_cache_min(os.environ.get("PUPPETLLM_CACHE_MIN_TOKENS"))


# ── Server state ──────────────────────────────────────────────────────


class _ServerState:
    def __init__(self) -> None:
        # multi-pending (README §parallel-request support): pending_id → {request, future, started_at}.
        # Can hold multiple concurrent in-flight requests (for testing cases where the
        # caller hits the API in parallel via asyncio.gather etc.). With only a single
        # pending, the dict just holds one entry.
        self.pending: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self.turn_count: int = 0
        self.lock = asyncio.Lock()
        # The futures currently waiting in /_control/wait_for_pending.
        self.pending_arrival_waiters: list[asyncio.Future[dict[str, Any]]] = []
        # Pseudo prompt cache (prefix hash → hit/miss).
        self.cache = CacheSimulator(ttl_seconds=_CACHE_TTL, honor_ttl=_CACHE_HONOR_TTL,
                                    min_cacheable_tokens=_CACHE_MIN_TOKENS)

    def _oldest_pending(self) -> dict[str, Any] | None:
        """Return the **unresolved** pending entry with the oldest received_at (or None). Call within the lock.

        Skip done futures (resolved / cancelled) — this keeps entries that linger for
        the brief instant between resolution and record completion, or that are left
        behind by an unexpected error path, from being shown to the responder (prevents
        ghost pendings).
        """
        live = [e for e in self.pending.values() if not e["future"].done()]
        if not live:
            return None
        return min(live, key=lambda e: e["request"].get("received_at", 0))


state = _ServerState()
app = FastAPI(title="puppetllm fake-llm-api")


async def _parse_json_body(request: Request) -> tuple[dict[str, Any] | None, str | None]:
    """JSON-parse the request body. → (body, None) or (None, error message).

    When a responder sends deeply nested payloads via curl (e.g. render_chart's input),
    broken shell escaping tends to produce malformed JSON. Pass the error message back to
    the caller so it can return a clear 400 instead of an opaque 500 — the control
    endpoints use `_plain_400`, while provider paths format it with their own error
    envelope (Anthropic/Bedrock/OpenAI shape).
    """
    try:
        body = await request.json()
    except Exception as e:
        return None, f"invalid JSON body: {str(e)[:200]}"
    if not isinstance(body, dict):
        return None, "invalid JSON body: must be a JSON object"
    return body, None


def _plain_400(message: str) -> JSONResponse:
    """The legacy-format 400 for the control endpoints (/_control/*)."""
    return JSONResponse({"error": message}, status_code=400)


def _anthropic_error(status: int, etype: str, message: str,
                     headers: dict[str, str] | None = None) -> JSONResponse:
    """Anthropic's official error envelope (`{"type":"error","error":{...}}`)."""
    return JSONResponse(
        {"type": "error", "error": {"type": etype, "message": message}},
        status_code=status, headers=headers,
    )


def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:24]}"


# ── canonical: cost/cache computation ────────────────────────────────


def _compute_usage(snapshot: dict[str, Any], content_blocks: list[dict[str, Any]]) -> tuple[dict, dict]:
    """Build usage and an estimated cost from the snapshot (input) and the response content_blocks.

    Follows Anthropic usage semantics:
      input_tokens               = input not served from cache (= total input - read - creation)
      cache_creation_input_tokens = amount written to cache this time (the prefix on a miss)
      cache_read_input_tokens     = amount read from cache (the prefix on a hit)
    """
    model = snapshot.get("model")
    total_in = int(snapshot.get("input_tokens_total", 0))
    cache = snapshot.get("cache") or {}
    read = int(cache.get("cache_read_tokens", 0))
    creation = int(cache.get("cache_creation_tokens", 0))
    uncached = max(0, total_in - read - creation)
    output = pricing.estimate_output_tokens(content_blocks)
    cost = pricing.compute_cost(
        model,
        input_tokens=uncached,
        output_tokens=output,
        cache_write_tokens=creation,
        cache_read_tokens=read,
    )
    usage = {
        "input_tokens": uncached,
        "output_tokens": output,
        "cache_creation_input_tokens": creation,
        "cache_read_input_tokens": read,
    }
    return usage, cost


async def _record_and_reset(
    request_snapshot: dict[str, Any],
    response_blocks: list[dict[str, Any]] | None,
    injected_error: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    cost: dict[str, Any] | None = None,
) -> None:
    """Append one entry to history and remove that pending from the registry.

    Called from the main flow. `request_snapshot` is a copy captured by the handler;
    its `pending_id` is used as the key to remove only its own entry (without affecting
    other in-flight requests). usage / cost / cache are recorded alongside so that
    /_control/stats can aggregate them.
    """
    entry: dict[str, Any] = {
        "turn": request_snapshot.get("turn"),
        "provider": request_snapshot.get("provider"),
        "model": request_snapshot.get("model"),
        "request": request_snapshot,
        "response_blocks": response_blocks,
        "usage": usage,
        "cost": cost,
        "cache": request_snapshot.get("cache"),
        "completed_at": time.time(),
    }
    if injected_error is not None:
        entry["injected_error"] = injected_error
    pid = request_snapshot.get("pending_id")
    async with state.lock:
        # If clear ran first and the entry is already gone, don't append to history
        # (prevents a "pre-clear request" from resurrecting and mixing in after a clear).
        if pid is None or pid in state.pending:
            state.history.append(entry)
        if pid is not None:
            state.pending.pop(pid, None)


# ── canonical: request registration / awaiting response (provider-common) ──

# Whitelist of auxiliary parameters carried along in the snapshot. Lets the responder
# see the "constraints a real API would honor" (forcing tool_choice / response_format /
# stop_sequences etc.). Unlike the main body (system/messages/tools/max_tokens), the
# core does not interpret these — it just passes them through.
_EXTRA_PARAM_KEYS: tuple[str, ...] = (
    "tool_choice", "response_format", "temperature", "top_p", "top_k",
    "stop_sequences", "stop", "thinking", "parallel_tool_calls", "n",
    "metadata", "reasoning_effort", "service_tier",
)


async def register_request(
    provider: str,
    model: str | None,
    body: dict[str, Any],
    is_stream: bool,
    *,
    simulate_cache: bool = True,
) -> tuple[dict[str, Any], asyncio.Future]:
    """Build a normalized snapshot, register it as pending, and return a future to await the response.

    Input-token estimation and pseudo-cache judgment (hit/miss) are done here inside the
    same lock as turn numbering, so that ordering is deterministic even for parallel
    requests. Provider-independent.

    With simulate_cache=False, pseudo-cache observation is skipped and cache is always
    "none" (for the OpenAI path: its caching is an automatic scheme rather than
    cache_control based, so it is not simulated).
    """
    system = body.get("system")
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    # Analysis with multi-breakpoint + prefix-match support (computes segments/breakpoints/total at once).
    request_cache = analyze_request(system, tools, messages)
    input_tokens_total = request_cache.total_tokens
    now = time.time()

    async with state.lock:
        state.turn_count += 1
        turn = state.turn_count
        pending_id = uuid.uuid4().hex[:16]
        if simulate_cache:
            cache = state.cache.observe(request_cache, model, now)
        else:
            # Same shape as observe()'s "none" (stats counts only hit/miss, so none is not aggregated)
            cache = {"status": "none", "cache_read_tokens": 0, "cache_creation_tokens": 0,
                     "prefix_hash": None, "read_seg_count": 0,
                     "breakpoints": len(request_cache.breakpoints)}
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        request_snapshot = {
            "pending_id": pending_id,
            "turn": turn,
            "provider": provider,
            "model": model,
            "system": system,
            "messages": messages,
            "tools": tools,
            "max_tokens": body.get("max_tokens"),
            # Retain auxiliary parameters (tool_choice / response_format / temperature etc.) as pass-through.
            # The responder can look at these and inject a response consistent with "constraints a real API would honor".
            "params": {k: body[k] for k in _EXTRA_PARAM_KEYS if k in body},
            "stream": is_stream,
            "received_at": now,
            "input_tokens_total": input_tokens_total,
            "cache": cache,
        }
        state.pending[pending_id] = {
            "request": request_snapshot,
            "future": fut,
            "started_at": now,
        }
        # Wake watchers waiting in /_control/wait_for_pending.
        for w in state.pending_arrival_waiters:
            if not w.done():
                w.set_result(request_snapshot)
        state.pending_arrival_waiters.clear()

    return request_snapshot, fut


def _discard_pending(snapshot: dict[str, Any]) -> None:
    """Reliably remove a pending entry (for the error/cancel paths).

    A single dict.pop within the event loop is atomic, so no lock is needed — cleanup can
    happen without awaiting even while a CancelledError is propagating (the key to
    preventing ghost pendings).
    """
    pid = snapshot.get("pending_id")
    if pid is not None:
        state.pending.pop(pid, None)


async def await_resolution(snapshot: dict[str, Any], fut: asyncio.Future) -> dict[str, Any]:
    """Await the control-injected response, record it in history, and return the result as a tagged dict.

    The returned "kind":
      "cleared" → was /_control/clear'd (caller returns 503)
      "error"   → injected error ({"status", "type", "message", "code", "param"})
      "ok"      → success ({"content_blocks", "usage", "cost", "model", "message_id", "stop_reason"})
    Provider-independent. Encoding is done by each provider.

    No path (cancel / unexpected exception) leaves a pending entry behind — if it did, a
    responder in a long-poll would forever keep seeing an unresolvable pending and spin.
    """
    try:
        response_payload = await fut
    except RuntimeError as e:
        # Cancellation via clear. State has already been reset by clear.
        return {"kind": "cleared", "detail": str(e)}
    except BaseException:
        # Task cancellation from client disconnect etc.: clean up the entry, then propagate.
        _discard_pending(snapshot)
        raise

    try:
        model = snapshot.get("model")
        if isinstance(response_payload, dict) and response_payload.get("_inject_error"):
            status = int(response_payload.get("status", 500))
            etype = str(response_payload.get("type", "api_error"))
            emsg = str(response_payload.get("message", "fake_server injected error"))
            await _record_and_reset(
                snapshot, response_blocks=None,
                injected_error={"status": status, "type": etype, "message": emsg},
            )
            return {"kind": "error", "status": status, "type": etype, "message": emsg,
                    "code": response_payload.get("code"),
                    "param": response_payload.get("param")}

        content_blocks = response_payload.get("content") or []
        if not isinstance(content_blocks, list):
            content_blocks = []
        # Keep only the block types puppetllm models (text / tool_use), dropping unknown
        # ones (thinking, etc.) ONCE here — this is the single source of truth, so history,
        # usage, and the encoded response (stream & non-stream, all providers) all agree.
        # (The encoders also skip unknown blocks defensively, but this is what makes usage
        # and /_control/history reflect exactly what the caller receives.)
        content_blocks = [b for b in content_blocks
                          if isinstance(b, dict) and b.get("type") in ("text", "tool_use")]
        # Assign ids for any tool_use missing one, all in one place (so stream / non-stream /
        # all providers use the same id. Previously only stream generated one and non-stream passed through).
        for b in content_blocks:
            if b.get("type") == "tool_use" and not b.get("id"):
                b["id"] = f"toolu_{uuid.uuid4().hex[:24]}"
        usage, cost = _compute_usage(snapshot, content_blocks)
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        await _record_and_reset(snapshot, response_blocks=content_blocks, usage=usage, cost=cost)
        return {
            "kind": "ok",
            "content_blocks": content_blocks,
            "usage": usage,
            "cost": cost,
            "model": model,
            "message_id": message_id,
            # stop_reason override specified by the responder via /_control/respond (None if absent
            # = each encoder auto-determines it from whether tool_use is present).
            "stop_reason": response_payload.get("stop_reason"),
        }
    except BaseException:
        # Unexpected error such as in usage computation: a 500 is returned, but the pending is always cleaned up.
        _discard_pending(snapshot)
        raise


# ── SSE / stream event construction helpers ──────────────────────────


def _sse_event(event_name: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def stream_event_dicts(
    message_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    stop_reason: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Build the (event_name, data) sequence for the Anthropic streaming protocol.

    Returns a wire-format-independent dict sequence so the same event stream can be used
    for both SSE (Anthropic) and eventstream (Bedrock). If usage is unspecified, fake
    values are used. Specifying stop_reason overrides the auto-determination (based on
    whether tool_use is present), useful for testing branches like max_tokens.
    """
    # If usage was provided, use the real (estimated) values; otherwise the legacy fake values.
    if usage is not None:
        start_usage = {
            "input_tokens": usage.get("input_tokens", 1),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            # At message_start no output has been generated yet. The cumulative output is returned on the message_delta side.
            "output_tokens": 0,
        }
        delta_usage = {"output_tokens": usage.get("output_tokens", 0)}
    else:
        start_usage = {"input_tokens": 1, "output_tokens": 1}
        delta_usage = {"output_tokens": 100}

    out: list[tuple[str, dict[str, Any]]] = []
    out.append(("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": start_usage,
        },
    }))

    derived_stop = "end_turn"
    # The stream `index` is **a running count of emitted blocks**. Using enumerate's
    # original position would create gaps when unknown blocks are skipped, crashing the
    # real SDK's stream accumulator with an IndexError.
    idx = -1
    for block in content_blocks:
        btype = block.get("type") if isinstance(block, dict) else None
        if btype == "text":
            idx += 1
            out.append(("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "text", "text": ""},
            }))
            text = str(block.get("text", ""))
            chunk_size = 80
            chunks = (
                [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                if text else [""]
            )
            for chunk in chunks:
                out.append(("content_block_delta", {
                    "type": "content_block_delta", "index": idx,
                    "delta": {"type": "text_delta", "text": chunk},
                }))
            out.append(("content_block_stop", {
                "type": "content_block_stop", "index": idx,
            }))
        elif btype == "tool_use":
            derived_stop = "tool_use"
            idx += 1
            tool_id = str(block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
            out.append(("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": str(block.get("name", "")),
                    "input": {},
                },
            }))
            input_json = json.dumps(block.get("input", {}) or {}, ensure_ascii=False)
            out.append(("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            }))
            out.append(("content_block_stop", {
                "type": "content_block_stop", "index": idx,
            }))
        else:
            # Skip unknown blocks (anything other than text/tool_use) (don't consume an index = don't create a gap).
            continue

    out.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason or derived_stop, "stop_sequence": None},
        "usage": delta_usage,
    }))
    out.append(("message_stop", {"type": "message_stop"}))
    return out


def _build_sse_stream(
    message_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    stop_reason: str | None = None,
) -> list[bytes]:
    """Build the SSE byte sequence that satisfies the Anthropic streaming protocol.

    note: passing `usage` includes estimated token/cache values. If unspecified, fake
    values (legacy behavior). Like the real API, insert one `ping` right after
    message_start (SSE path only; the SDK ignores it).
    """
    events = stream_event_dicts(message_id, model, content_blocks, usage, stop_reason)
    out = [_sse_event(events[0][0], events[0][1]), _sse_event("ping", {"type": "ping"})]
    out.extend(_sse_event(name, data) for name, data in events[1:])
    return out


def _build_non_stream_response(
    message_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    # Keep parity with the streaming path (stream_event_dicts), which only emits
    # text/tool_use blocks: filter unknown block types here too so the same injection
    # yields the same content whether the caller used stream=True or not. puppetllm
    # only simulates text/tool_use (thinking/redacted_thinking/etc. are not modeled).
    content_blocks = [b for b in content_blocks
                      if isinstance(b, dict) and b.get("type") in ("text", "tool_use")]
    if stop_reason is None:
        stop_reason = "tool_use" if any(
            b.get("type") == "tool_use" for b in content_blocks
        ) else "end_turn"
    if usage is not None:
        usage_out = {
            "input_tokens": usage.get("input_tokens", 1),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        }
    else:
        usage_out = {"input_tokens": 1, "output_tokens": 100}
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_out,
    }


# ── Anthropic compatible endpoint ────────────────────────────────────


@app.post("/v1/messages")
async def messages(request: Request) -> Any:
    req_id = _new_request_id()
    headers = {"request-id": req_id}
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        # Like the real API, errors on the Anthropic path are always returned in the official envelope.
        return _anthropic_error(400, "invalid_request_error", errmsg, headers=headers)
    is_stream = bool(body.get("stream"))
    model = body.get("model")

    snapshot, fut = await register_request("anthropic", model, body, is_stream)
    result = await await_resolution(snapshot, fut)

    if result["kind"] == "cleared":
        return _anthropic_error(503, "api_error",
                                f"request cleared: {result['detail']}", headers=headers)
    if result["kind"] == "error":
        return _anthropic_error(result["status"], result["type"], result["message"],
                                headers=headers)

    model_out = model or "claude-sonnet-mock"
    content_blocks = result["content_blocks"]
    usage = result["usage"]
    message_id = result["message_id"]
    stop_reason = result.get("stop_reason")

    if is_stream:
        events = _build_sse_stream(message_id, model_out, content_blocks, usage, stop_reason)

        async def gen():
            for evt in events:
                yield evt
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
    return JSONResponse(
        _build_non_stream_response(message_id, model_out, content_blocks, usage, stop_reason),
        headers=headers,
    )


# ── Control endpoints ────────────────────────────────────────────────


@app.get("/_control/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "turn_count": state.turn_count}


@app.get("/_control/pending")
async def pending() -> dict[str, Any]:
    """Return all current pendings (multi-pending).

    Backward compatible: `has_pending` (bool), and if there is at least one pending, the
    oldest is also placed in `request` / `waiting_for_seconds`. Parallel-aware callers use
    the `pending` array.
    """
    now = time.time()
    async with state.lock:
        items = [
            {
                "pending_id": pid,
                "request": e["request"],
                "waiting_for_seconds": round(now - e["started_at"], 2),
            }
            for pid, e in state.pending.items()
            if not e["future"].done()  # don't show resolved/awaiting-cleanup entries (ghost prevention)
        ]
        oldest = state._oldest_pending()
    items.sort(key=lambda x: x["request"].get("received_at", 0))
    if not items:
        return {"has_pending": False, "pending": [], "count": 0}
    return {
        "has_pending": True,
        "count": len(items),
        "pending": items,
        # backward compatible (oldest pending)
        "request": oldest["request"] if oldest else None,
        "waiting_for_seconds": round(now - oldest["started_at"], 2) if oldest else None,
    }


# Long-poll safety cap (with some margin, staying within the Bash tool's 10-minute timeout)
_WAIT_TIMEOUT_MAX = 600.0
_WAIT_TIMEOUT_DEFAULT = 270.0  # < 5 minutes (to stay within the Anthropic prompt cache TTL)


@app.get("/_control/wait_for_pending")
async def wait_for_pending(timeout: float = _WAIT_TIMEOUT_DEFAULT) -> dict[str, Any]:
    """Long-polling: block until a pending appears (up to timeout seconds).

    If a pending already exists, return immediately. Otherwise, wait for the next
    /v1/messages to arrive.
    """
    timeout = max(0.5, min(float(timeout), _WAIT_TIMEOUT_MAX))

    waiter: asyncio.Future[dict[str, Any]] | None = None
    async with state.lock:
        oldest = state._oldest_pending()
        if oldest is not None:
            return {
                "has_pending": True,
                "request": oldest["request"],
                "pending_id": oldest["request"].get("pending_id"),
                "waiting_for_seconds": round(time.time() - oldest["started_at"], 2),
            }
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        state.pending_arrival_waiters.append(waiter)

    try:
        request_snapshot = await asyncio.wait_for(waiter, timeout=timeout)
    except asyncio.TimeoutError:
        return {"has_pending": False, "timeout": True}
    except BaseException:
        raise
    finally:
        async with state.lock:
            try:
                state.pending_arrival_waiters.remove(waiter)
            except ValueError:
                pass

    return {
        "has_pending": True,
        "request": request_snapshot,
        "pending_id": request_snapshot.get("pending_id"),
        "waiting_for_seconds": 0.0,
    }


async def _resolve_target_future(
    pending_id: str | None,
) -> tuple[asyncio.Future[dict[str, Any]] | None, JSONResponse | None]:
    """Resolve the target pending future for injection (multi-pending).

    - `pending_id` given: use that entry (400 if it doesn't exist)
    - unspecified: if there is exactly 1 pending, use it (backward compatible). 0 → 400, multiple → 400
    """
    async with state.lock:
        if pending_id is not None:
            entry = state.pending.get(pending_id)
            if entry is None or entry["future"].done():
                return None, JSONResponse(
                    {"error": f"no pending request with pending_id={pending_id}"}, status_code=400
                )
            return entry["future"], None
        ids = [pid for pid, e in state.pending.items() if not e["future"].done()]
        if not ids:
            return None, JSONResponse({"error": "no pending request"}, status_code=400)
        if len(ids) > 1:
            return None, JSONResponse(
                {"error": "multiple pending requests; specify pending_id", "pending_ids": ids},
                status_code=400,
            )
        return state.pending[ids[0]]["future"], None


def _safe_set_result(
    fut: asyncio.Future[dict[str, Any]], value: dict[str, Any]
) -> JSONResponse | None:
    """Inject a result into the future. Returns 409 if it's done (e.g. race with clear)."""
    try:
        fut.set_result(value)
    except asyncio.InvalidStateError:
        return JSONResponse(
            {"error": "pending request already resolved (e.g. cleared)"}, status_code=409
        )
    return None


@app.post("/_control/respond")
async def respond(request: Request) -> Any:
    """Inject Body: `{"content": [<content_block>, ...], "pending_id"?, "stop_reason"?}`.

    The content_block type is "text" | "tool_use". When `pending_id` is omitted, inject
    into the single pending if there is one (backward compatible). With multiple in-flight,
    `pending_id` is required. `stop_reason` (optional) overrides the auto-determination
    (e.g. "max_tokens" — for testing truncation branches; converted to finish_reason on
    the OpenAI path).
    """
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _plain_400(errmsg)
    content = body.get("content", [])
    # Validate the shape here and return 400 (if the encoder crashes after the future is
    # resolved, history records a success while the calling SDK gets a 500 — an inconsistency).
    if not isinstance(content, list) or not all(
        isinstance(b, dict) and isinstance(b.get("type"), str) for b in content
    ):
        return _plain_400("content must be a list of content-block objects with a string 'type'")
    stop_reason = body.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        return _plain_400("stop_reason must be a string")
    fut, err = await _resolve_target_future(body.get("pending_id"))
    if err is not None:
        return err
    err = _safe_set_result(fut, {"content": content, "stop_reason": stop_reason})
    if err is not None:
        return err
    return {"ok": True}


@app.post("/_control/auto")
async def auto(request: Request) -> Any:
    """Simple: inject `{"text": "...", "pending_id"?: "..."}` as a text-only response."""
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _plain_400(errmsg)
    fut, err = await _resolve_target_future(body.get("pending_id"))
    if err is not None:
        return err
    err = _safe_set_result(fut, {
        "content": [{"type": "text", "text": body.get("text", "(empty)")}]
    })
    if err is not None:
        return err
    return {"ok": True}


@app.post("/_control/error")
async def inject_error(request: Request) -> Any:
    """Error injection: make a pending request return an HTTP error.

    Body: {"status": 429, "type": "rate_limit_error", "message": "...",
           "code"?: "...", "param"?: "..."}
    On any of the Anthropic / Bedrock / OpenAI paths, each provider converts status/type
    into its own path's error format (code/param are used only in the OpenAI format).
    The SDK auto-retries 5xx/429/408.
    """
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _plain_400(errmsg)
    try:
        status = int(body.get("status", 500))
    except (TypeError, ValueError):
        return _plain_400("status must be an integer")
    if not (100 <= status <= 599):
        return _plain_400("status must be in [100, 599]")
    fut, err = await _resolve_target_future(body.get("pending_id"))
    if err is not None:
        return err
    err = _safe_set_result(fut, {
        "_inject_error": True,
        "status": status,
        "type": str(body.get("type", "api_error")),
        "message": str(body.get("message", "fake_server injected error")),
        "code": body.get("code"),
        "param": body.get("param"),
    })
    if err is not None:
        return err
    return {"ok": True}


@app.get("/_control/history")
async def history() -> dict[str, Any]:
    # Snapshot under the lock (a copy) for consistency with stats(); avoids handing out
    # the live list while a concurrent request appends to it.
    async with state.lock:
        return {"turn_count": state.turn_count, "history": list(state.history)}


@app.get("/_control/stats")
async def stats() -> dict[str, Any]:
    """Aggregate cumulative estimated-cost, token, and cache summaries from history.

    note: everything is an estimate (approx tokenizer). It does not match real billing (see README).
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_usd": 0.0,
        "cache_savings_usd": 0.0,
    }
    by_model: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    cache_misses = 0
    completed = 0
    errors = 0

    async with state.lock:
        hist = list(state.history)

    for e in hist:
        if e.get("injected_error") is not None:
            errors += 1
            continue
        usage = e.get("usage") or {}
        cost = e.get("cost") or {}
        cache = e.get("cache") or {}
        model = e.get("model") or "unknown"
        completed += 1

        for k in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens"):
            totals[k] += int(usage.get(k, 0))
        totals["total_usd"] += float(cost.get("total_usd", 0.0))

        read = int(usage.get("cache_read_input_tokens", 0))
        totals["cache_savings_usd"] += pricing.cache_savings_usd(model, read)

        cstatus = cache.get("status")
        if cstatus == "hit":
            cache_hits += 1
        elif cstatus == "miss":
            cache_misses += 1

        m = by_model.setdefault(model, {"requests": 0, "total_usd": 0.0,
                                        "input_tokens": 0, "output_tokens": 0})
        m["requests"] += 1
        m["total_usd"] = round(m["total_usd"] + float(cost.get("total_usd", 0.0)), 6)
        m["input_tokens"] += int(usage.get("input_tokens", 0))
        m["output_tokens"] += int(usage.get("output_tokens", 0))

    cache_total = cache_hits + cache_misses
    totals["total_usd"] = round(totals["total_usd"], 6)
    totals["cache_savings_usd"] = round(totals["cache_savings_usd"], 6)
    return {
        "is_estimate": True,
        "turn_count": state.turn_count,
        "completed_requests": completed,
        "error_requests": errors,
        "totals": totals,
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": round(cache_hits / cache_total, 4) if cache_total else 0.0,
            "index_size": len(state.cache.index),
        },
        "by_model": by_model,
    }


@app.get("/_control/cache")
async def cache_index() -> dict[str, Any]:
    """The pseudo prompt-cache's current index (by prefix hash)."""
    now = time.time()
    # entries() iterates state.cache.index. Since register_request does cache.observe →
    # index mutation within the lock, this must also take a snapshot within the lock, or
    # a concurrent scan hits "dictionary changed size during iteration".
    async with state.lock:
        entries = state.cache.entries(now)
    return {
        "ttl_seconds": state.cache.ttl_seconds,
        "honor_ttl": state.cache.honor_ttl,
        "entries": entries,
    }


@app.post("/_control/clear")
async def clear() -> dict[str, Any]:
    async with state.lock:
        # Cancel all in-flight pendings with 503 (the main handler gracefully returns 503)
        for entry in state.pending.values():
            fut = entry["future"]
            if not fut.done():
                fut.set_exception(RuntimeError("cleared by control"))
        state.pending.clear()
        state.history.clear()
        state.turn_count = 0
        state.cache.reset()
    return {"ok": True}


# ── Registering the provider adapters ────────────────────────────────
# Each provider router references the canonical helpers (register_request /
# await_resolution / stream_event_dicts / _build_non_stream_response) at call time, so
# import & include them at the very end after all helpers are defined (avoids circular imports).

from .providers import bedrock as _bedrock  # noqa: E402
from .providers import openai as _openai  # noqa: E402

app.include_router(_bedrock.build_router())
app.include_router(_openai.build_router())


# ── Stand-alone startup ──────────────────────────────────────────────


def main() -> int:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Fake Anthropic/Bedrock/OpenAI API server for debugging")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"[puppetllm] starting on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[puppetllm] Anthropic: set ANTHROPIC_BASE_URL=http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[puppetllm] Bedrock:   point AnthropicBedrock base_url to http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[puppetllm] OpenAI:    set OPENAI_BASE_URL=http://{args.host}:{args.port}/v1  (note the /v1)", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
