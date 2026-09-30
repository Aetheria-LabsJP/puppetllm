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
- POST /model/{id}/converse[-stream]  — Bedrock Converse compatible (added by providers/converse.py)
- POST /v1/chat/completions          — OpenAI compatible (added by providers/openai.py)
- /v1/messages/batches[...]           — Anthropic Message Batches compatible (added by batches.py)
- /model-invocation-job[s][...]       — Bedrock batch inference (added by providers/bedrock_batch.py)
- /{bucket}[/{key}]                   — S3 emulation for batch inference I/O (added by providers/s3.py)
- GET  /_control/pending             — pending requests (including provider)
- GET  /_control/wait_for_pending    — long-poll until the next pending arrives
- POST /_control/respond             — inject a response into a pending request
- POST /_control/auto                — simple auto-response (text only)
- POST /_control/error               — inject an HTTP error response
- GET  /_control/history             — (request, response, usage, cost, cache) history
- GET  /_control/stats               — cumulative summary of estimated cost, tokens, and cache
- GET  /_control/cache               — pseudo prompt-cache index
- POST /_control/clear               — empty pending/history/cache/batches/bedrock_jobs (not the S3 store)
- GET  /_control/health              — health check
- GET  /_control/batches, POST /_control/batch/{end,result} — batch injection (added by batches.py)
- GET  /_control/bedrock_jobs        — Bedrock batch-inference job registry (added by providers/bedrock_batch.py)

The control endpoints are localhost only (for debugging), with no authorization.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import math
import os
import sys
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from . import capabilities as _capabilities
from . import control_models as _cm
from . import harness as _harness
from . import pricing
from .cache_sim import CacheControlError, CacheSimulator, analyze_request
from .openai_wire import strip_private


class RequestValidationError(ValueError):
    """A request the real API rejects with 400 invalid_request_error (raised by
    register_request; each route encodes it in its own error envelope)."""


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
# 1-hour breakpoints: explicit override, else 12x the 5-minute TTL (so shortening
# PUPPETLLM_CACHE_TTL for tests scales both kinds).
_CACHE_TTL_1H = (_parse_cache_ttl(os.environ.get("PUPPETLLM_CACHE_TTL_1H"))
                 if os.environ.get("PUPPETLLM_CACHE_TTL_1H") else _CACHE_TTL * 12)
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
        # Message Batches (batches.py). batch_id → batch dict:
        #   {"id", "created_at", "expires_at", "ended_at", "cancel_initiated_at",
        #    "processing_status": "in_progress"|"canceling"|"ended",
        #    "entries": {custom_id: {"pending_id", "result": None | {"type", ...}}}}
        # Held here (not in batches.py) so that clear() wipes it atomically with pending,
        # and so the control endpoints below can resolve custom_id → pending_id without
        # a circular import.
        self.batches: dict[str, dict[str, Any]] = {}
        # Bedrock batch inference jobs (providers/bedrock_batch.py). job_id → job dict.
        self.bedrock_jobs: dict[str, dict[str, Any]] = {}
        # Bumped by /_control/clear: a request that started before a clear and would
        # otherwise (re)create state afterwards checks this and gives up instead.
        self.clear_generation: int = 0
        # pending_id → the clear generation it was registered in (an answer resolved
        # before a clear must not be delivered by a handler that resumes after it).
        self.pending_generation: dict[str, int] = {}
        self.turn_count: int = 0
        self.lock = asyncio.Lock()
        # The futures currently waiting in /_control/wait_for_pending.
        self.pending_arrival_waiters: list[asyncio.Future[dict[str, Any]]] = []
        # Pseudo prompt cache (prefix hash → hit/miss).
        self.cache = CacheSimulator(ttl_seconds=_CACHE_TTL, honor_ttl=_CACHE_HONOR_TTL,
                                    min_cacheable_tokens=_CACHE_MIN_TOKENS,
                                    ttl_1h_seconds=_CACHE_TTL_1H)

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
# Scenario rules, latency / rate-limit simulation, pending timeout and the fake clock.
harness = _harness.Harness()
app = FastAPI(title="puppetllm fake-llm-api", version=__version__)


def _now() -> float:
    """Wall time shifted by `/_control/clock/advance`: the clock behind the cache TTLs,
    the rate-limit window, pending deadlines and history timestamps (batch lifecycles
    and wire-level timestamps stay on wall time)."""
    return harness.now()


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
                     headers: dict[str, str] | None = None,
                     request_id: str | None = None) -> JSONResponse:
    """Anthropic's official error envelope (`{"type":"error","error":{...}}`)."""
    # Every error carries a request id, in the body AND the `request-id` header (the real
    # API does both on every route, batches included).
    headers = dict(headers or {})
    rid = request_id or headers.get("request-id") or _new_request_id()
    headers.setdefault("request-id", rid)
    return JSONResponse(
        {"type": "error", "error": {"type": etype, "message": message}, "request_id": rid},
        status_code=status, headers=headers,
    )


def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:24]}"


# The real Batches API bills all token usage at 50% of the standard price.
_BATCH_DISCOUNT = 0.5


def _apply_batch_discount(cost: dict[str, Any]) -> dict[str, Any]:
    """Apply the Message Batches 50% discount to an estimated-cost dict (pricing.compute_cost).

    Returns a new dict (does not mutate the input) with each USD field halved and a
    `batch_discount` marker so history/stats readers can tell the discount was applied.
    """
    out = dict(cost)
    token_parts = ("input_usd", "output_usd", "cache_write_usd", "cache_read_usd")
    for k in token_parts:
        out[k] = round(float(out.get(k, 0.0)) * _BATCH_DISCOUNT, 6)
    # The discount is on token prices; a server tool's flat fee is billed as-is.
    out["total_usd"] = round(sum(out[k] for k in token_parts)
                             + float(out.get("server_tool_usd", 0.0)), 6)
    out["batch_discount"] = _BATCH_DISCOUNT
    return out


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
    creation_1h = int(cache.get("cache_creation_1h_tokens", 0))
    creation_5m = int(cache.get("cache_creation_5m_tokens", creation - creation_1h))
    uncached = max(0, total_in - read - creation)
    # OpenAI `n`: the fake duplicates one generated answer into n choices, and the real API
    # bills every choice — fold it into the canonical usage so wire, history and stats agree.
    n = _choices(snapshot)
    # Server-tool results are not generated tokens; the calls themselves are — but only
    # on the routes whose wire carries them (Converse and OpenAI drop them, so the client
    # saw neither the call nor a search: no tokens, no count, no fee).
    carries = _carries_server_tools(snapshot)
    billable = _GENERATED_BLOCK_TYPES if carries else ("text", "tool_use", "thinking", "redacted_thinking")
    generated = [b for b in content_blocks if isinstance(b, dict) and b.get("type") in billable]
    output = pricing.estimate_output_tokens(generated) * n
    server_tool_use = _server_tool_requests(content_blocks) if carries else None
    geo, speed = _geo_and_speed(snapshot)
    cost = pricing.compute_cost(
        model,
        input_tokens=uncached,
        output_tokens=output,
        cache_write_tokens=creation_5m,
        cache_read_tokens=read,
        cache_write_1h_tokens=creation_1h,
        inference_geo=geo, speed=speed,
        web_search_requests=(server_tool_use or {}).get("web_search_requests", 0) * n,
    )
    usage = {
        "input_tokens": uncached,
        "output_tokens": output,
        "cache_creation_input_tokens": creation,
        "cache_read_input_tokens": read,
        # Real-API breakdown of cache writes by TTL.
        "cache_creation": {"ephemeral_5m_input_tokens": creation_5m,
                           "ephemeral_1h_input_tokens": creation_1h},
        "output_tokens_details": {"thinking_tokens": _thinking_tokens(content_blocks) * n},
        "server_tool_use": ({k: v * n for k, v in server_tool_use.items()}
                            if server_tool_use else None),
        # Anthropic Message Batches carry `batch_id`; Bedrock batch inference carries
        # `job_id`. Either way the request is batch traffic.
        "service_tier": "batch" if (snapshot.get("batch_id") or snapshot.get("job_id"))
                        else "standard",
        "inference_geo": geo or "global",
    }
    if speed == "fast":
        usage["speed"] = "fast"  # the real API reports the speed actually used
    return usage, cost


def _choices(snapshot: dict[str, Any]) -> int:
    try:
        return max(1, int(snapshot.get("choices") or 1))
    except (TypeError, ValueError):
        return 1


def _geo_and_speed(snapshot: dict[str, Any]) -> tuple[str | None, str | None]:
    params = snapshot.get("params") or {}
    geo = params.get("inference_geo")
    speed = params.get("speed")
    if snapshot.get("provider") == "openai":
        speed = None  # `speed` is not a Chat Completions parameter; never price it there
    return (geo if isinstance(geo, str) else None,
            speed if isinstance(speed, str) else None)


def _carries_server_tools(snapshot: dict[str, Any]) -> bool:
    """Whether this pending's route puts server-side tool blocks on the wire (Anthropic
    Messages, Bedrock InvokeModel and the Mantle alias; not Converse, not OpenAI)."""
    return snapshot.get("provider") in ("anthropic", "bedrock") and snapshot.get("api") != "converse"


def _server_tool_requests(content_blocks: list[dict[str, Any]]) -> dict[str, int] | None:
    """`usage.server_tool_use` for the server_tool_use blocks of an answer (None when there
    are none, as the real API reports)."""
    calls = [b for b in content_blocks if isinstance(b, dict) and b.get("type") == "server_tool_use"]
    if not calls:
        return None
    return {"web_search_requests": sum(1 for b in calls if b.get("name") == "web_search"),
            "web_fetch_requests": sum(1 for b in calls if b.get("name") == "web_fetch")}


def _thinking_tokens(content_blocks: list[dict[str, Any]]) -> int:
    blocks = [b for b in content_blocks
              if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")]
    return pricing.approx_tokens(blocks) if blocks else 0


# Usage keys that may carry an object (or null) rather than an int.
_USAGE_OBJECT_KEYS = ("cache_creation", "output_tokens_details", "server_tool_use")
_USAGE_SCALAR_KEYS = ("service_tier", "inference_geo", "speed")
# Wire-format usage: only the fields the real API emits, in a stable order.
_USAGE_WIRE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                    "cache_creation", "output_tokens", "output_tokens_details",
                    "server_tool_use", "service_tier", "inference_geo", "speed", "iterations")


def _usage_wire(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Project an internal usage dict onto the real API's usage object shape."""
    if usage is None:
        return {"input_tokens": 1, "output_tokens": 100,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    out: dict[str, Any] = {}
    for k in _USAGE_WIRE_KEYS:
        if k in usage:
            out[k] = usage[k]
    out.setdefault("input_tokens", 1)
    out.setdefault("output_tokens", 0)
    out.setdefault("cache_creation_input_tokens", 0)
    out.setdefault("cache_read_input_tokens", 0)
    out.setdefault("cache_creation", {"ephemeral_5m_input_tokens": out["cache_creation_input_tokens"],
                                      "ephemeral_1h_input_tokens": 0})
    out.setdefault("output_tokens_details", {"thinking_tokens": 0})
    out.setdefault("server_tool_use", None)
    out.setdefault("service_tier", "standard")
    return out


def _block_payload_error(blocks: list[dict[str, Any]]) -> str | None:
    """Payload constraints an SDK enforces client-side. `redacted_thinking.data` is base64
    on every wire (botocore models it as a Blob and decodes it), so a responder that sends
    plain text would crash the boto3 client with a `binascii.Error` — refuse it here, where
    the message can say why."""
    for i, b in enumerate(blocks):
        if b.get("type") == "redacted_thinking":
            data = b.get("data")
            if data is None:
                continue
            try:
                base64.b64decode(str(data), validate=True)
            except (ValueError, binascii.Error):
                return (f"content[{i}].data: redacted_thinking data must be base64 "
                        f"(the Bedrock SDKs decode it client-side)")
        elif b.get("type") in _SERVER_RESULT_BLOCK_TYPES:
            # The SDKs model these with a required tool_use_id and content.
            if not isinstance(b.get("tool_use_id"), str) or not b.get("tool_use_id"):
                return f"content[{i}].tool_use_id: {b['type']} needs the id of its server_tool_use"
            if "content" not in b:
                return f"content[{i}].content: {b['type']} needs a content member"
        elif b.get("type") in ("server_tool_use", "mcp_tool_use"):
            if not isinstance(b.get("name"), str) or not b.get("name"):
                return f"content[{i}].name: {b['type']} needs a tool name"
            if b.get("input") is not None and not isinstance(b.get("input"), dict):
                return f"content[{i}].input: must be an object"
            if b["type"] == "mcp_tool_use" and (not isinstance(b.get("server_name"), str)
                                                or not b.get("server_name")):
                return f"content[{i}].server_name: mcp_tool_use needs its MCP server name"
    return None


def dropped_block_types(blocks: Any) -> list[str]:
    """The `type`s in an injected content list that the server does not model (reported
    back to the injector, or refused under `strict_blocks`)."""
    return [str(b.get("type")) for b in blocks
            if isinstance(b, dict) and b.get("type") not in _MODELED_BLOCK_TYPES]


_ANTHROPIC_ERROR_TYPE_BY_STATUS = {
    400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
    404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
    500: "api_error", 529: "overloaded_error",
}


def _normalize_blocks(blocks: Any) -> list[dict[str, Any]]:
    """Keep only the block types puppetllm models and fill in what the real API always
    returns: an id on every tool_use, a signature and string body on thinking blocks.

    This is the single source of truth, so history, usage and every encoder (stream and
    non-stream, all providers) agree — including the partial content of a mid-stream error.
    """
    out = [b for b in blocks if isinstance(b, dict) and b.get("type") in _MODELED_BLOCK_TYPES]
    for b in out:
        if b.get("type") == "tool_use" and not b.get("id"):
            b["id"] = f"toolu_{uuid.uuid4().hex[:24]}"
        elif b.get("type") == "server_tool_use":
            if not isinstance(b.get("input"), dict):
                b["input"] = {}
            if not b.get("id"):
                b["id"] = f"srvtoolu_{uuid.uuid4().hex[:24]}"
        elif b.get("type") == "mcp_tool_use":
            if not isinstance(b.get("input"), dict):
                b["input"] = {}
            if not b.get("id"):
                b["id"] = f"mcptoolu_{uuid.uuid4().hex[:24]}"
        elif b.get("type") == "thinking":
            b["thinking"] = "" if b.get("thinking") is None else str(b["thinking"])
            if not b.get("signature"):
                b["signature"] = _fake_signature()
        elif b.get("type") == "redacted_thinking":
            b["data"] = "" if b.get("data") is None else str(b["data"])
    return out


async def _record_and_reset(
    request_snapshot: dict[str, Any],
    response_blocks: list[dict[str, Any]] | None,
    injected_error: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    cost: dict[str, Any] | None = None,
    usage_overridden: bool = False,
    batch: bool = False,
    harness_meta: dict[str, Any] | None = None,
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
        "completed_at": _now(),
    }
    if injected_error is not None:
        entry["injected_error"] = injected_error
    if usage_overridden:
        # Real token counts were supplied via /_control/respond (e.g. relayed from an
        # upstream API) instead of the approx tokenizer.
        entry["usage_overridden"] = True
    if batch:
        # Came in via the Message Batches route (cost carries the 50% discount).
        entry["batch"] = True
    if harness_meta:
        # Answered by the harness, not a responder: {"source": rule|default|unmatched|
        # rate_limit|timeout, "rule_id"?: ...}.
        entry["harness"] = harness_meta
    pid = request_snapshot.get("pending_id")
    async with state.lock:
        # If clear ran first and the entry is already gone, don't append to history
        # (prevents a "pre-clear request" from resurrecting and mixing in after a clear).
        if pid is None or pid in state.pending:
            state.history.append(entry)
        if pid is not None:
            state.pending.pop(pid, None)
            state.pending_generation.pop(pid, None)


# ── canonical: request registration / awaiting response (provider-common) ──

# Whitelist of auxiliary parameters carried along in the snapshot. Lets the responder
# see the "constraints a real API would honor" (forcing tool_choice / response_format /
# stop_sequences etc.). Unlike the main body (system/messages/tools/max_tokens), the
# core does not interpret these — it just passes them through.
_EXTRA_PARAM_KEYS: tuple[str, ...] = (
    # Anthropic Messages API (2026)
    "tool_choice", "temperature", "top_p", "top_k", "stop_sequences", "thinking",
    "output_config", "cache_control", "metadata", "service_tier", "inference_geo",
    "container", "context_management", "fallbacks", "speed", "mcp_servers",
    "anthropic_beta",
    # OpenAI Chat Completions (also carried through the OpenAI route's normalization)
    "response_format", "stop", "parallel_tool_calls", "n", "reasoning_effort",
    "max_completion_tokens", "seed", "store", "prompt_cache_key", "safety_identifier",
    "user", "logprobs", "top_logprobs", "frequency_penalty", "presence_penalty",
    "logit_bias", "prediction", "web_search_options", "verbosity", "modalities",
    "audio", "stream_options",
)


async def register_request(
    provider: str,
    model: str | None,
    body: dict[str, Any],
    is_stream: bool,
    *,
    simulate_cache: bool = True,
    extra: dict[str, Any] | None = None,
    request_headers: Any = None,
) -> tuple[dict[str, Any], asyncio.Future]:
    """Build a normalized snapshot, register it as pending, and return a future to await the response.

    Input-token estimation and pseudo-cache judgment (hit/miss) are done here inside the
    same lock as turn numbering, so that ordering is deterministic even for parallel
    requests. Provider-independent.

    With simulate_cache=False, pseudo-cache observation is skipped and cache is always
    "none" (for the OpenAI path: its caching is an automatic scheme rather than
    cache_control based, so it is not simulated).

    `extra` is merged into the snapshot before it becomes visible to waiters — used by
    the batches route to tag each pending with `batch_id` / `custom_id` so the responder
    can tell which batch entry it is answering.
    """
    system = body.get("system")
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    shape_err = _request_shape_error(system, tools, messages)
    if shape_err is not None:
        raise RequestValidationError(shape_err)
    params = {k: body[k] for k in _EXTRA_PARAM_KEYS if k in body}
    # The `anthropic-beta` request header (Anthropic route) is surfaced like Bedrock's
    # `anthropic_beta` body field so the responder can see which betas the app requested.
    beta_header = request_headers.get("anthropic-beta") if request_headers is not None else None
    if beta_header and "anthropic_beta" not in params:
        params["anthropic_beta"] = [b.strip() for b in str(beta_header).split(",") if b.strip()]
    # Analysis with multi-breakpoint + prefix-match support (computes segments/breakpoints/total at once).
    # Top-level cache_control (automatic caching) and prompt-rendered params (effort / thinking /
    # tool_choice) are folded in exactly like the real API.
    try:
        request_cache = analyze_request(system, tools, messages,
                                        top_level_cache_control=body.get("cache_control"),
                                        params=params, model=model)
    except CacheControlError as e:
        raise RequestValidationError(str(e)) from e
    input_tokens_total = request_cache.total_tokens
    now = _now()

    async with state.lock:
        state.turn_count += 1
        turn = state.turn_count
        pending_id = uuid.uuid4().hex[:16]
        is_batch_entry = bool(extra) and ("batch_id" in extra or "job_id" in extra)
        # Rate admission comes first: a throttled request must not warm the pseudo cache
        # (the real API never processed it). Batch entries are not throttled.
        throttled = None if is_batch_entry else harness.throttle(
            {"pending_id": pending_id, "provider": provider,
             "input_tokens_total": input_tokens_total}, now)
        if simulate_cache and throttled is None:
            cache = state.cache.observe(request_cache, model, now)
        else:
            # Same shape as observe()'s "none" (stats counts only hit/miss, so none is not aggregated)
            cache = {"status": "none", "cache_read_tokens": 0, "cache_creation_tokens": 0,
                     "cache_creation_5m_tokens": 0, "cache_creation_1h_tokens": 0,
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
            # Retain auxiliary parameters (tool_choice / output_config / temperature etc.) as pass-through.
            # The responder can look at these and inject a response consistent with "constraints a real API would honor".
            "params": params,
            "stream": is_stream,
            "received_at": now,
            "input_tokens_total": input_tokens_total,
            "cache": cache,
        }
        if extra:
            request_snapshot.update(extra)
        timeout = harness.config.pending_timeout_s
        state.pending_generation[pending_id] = state.clear_generation
        state.pending[pending_id] = {
            "request": request_snapshot,
            "future": fut,
            "started_at": now,
            # Fixed at registration (a later config change does not move it); None for
            # batch entries, which the real batch APIs give hours.
            "deadline": (now + timeout) if (timeout is not None and not is_batch_entry) else None,
        }
        if _apply_harness(request_snapshot, fut, throttled):
            # Answered by a rule / the unmatched policy / the rate limiter: the future is
            # already resolved, so a responder never sees this pending.
            return request_snapshot, fut
        # Wake watchers waiting in /_control/wait_for_pending.
        for w in state.pending_arrival_waiters:
            if not w.done():
                w.set_result(request_snapshot)
        state.pending_arrival_waiters.clear()

    return request_snapshot, fut


def _apply_harness(snapshot: dict[str, Any], fut: asyncio.Future,
                   throttled: dict[str, Any] | None) -> bool:
    """Resolve a fresh pending the way `/_control/respond` / `/_control/error` would when
    the harness has an answer for it. Runs inside `state.lock` right after registration.

    Order: rate limiter (`throttled`, decided before the cache was touched; a throttled
    request never reaches a rule, as on the real APIs) → first matching rule →
    `on_unmatched` policy (`pending` keeps it for a responder).

    Stored payloads are deep-copied: the encoders fill generated fields (`tool_use.id`,
    thinking signatures) into the blocks they receive, and a rule / default that answers
    repeatedly must hand out fresh values every time and stay as the user posted it."""
    if throttled is not None:
        fut.set_result({"_inject_error": True, **throttled, "content": [],
                        "_harness": {"source": "rate_limit"}})
        return True
    rule, payload = harness.decide(snapshot)
    if payload is not None:
        fut.set_result({**copy.deepcopy(payload),
                        "_harness": {"source": "rule", "rule_id": rule.id}})
        return True
    cfg = harness.config
    if cfg.on_unmatched == "default" and cfg.default_response is not None:
        fut.set_result({**copy.deepcopy(cfg.default_response), "_harness": {"source": "default"}})
        return True
    if cfg.on_unmatched == "error":
        fut.set_result({"_inject_error": True, "content": [], **copy.deepcopy(cfg.unmatched_error),
                        "_harness": {"source": "unmatched"}})
        return True
    return False


def validate_request_body(body: dict[str, Any], model: str | None) -> None:
    """Everything `register_request` would refuse, without registering: the shape checks
    and the token / cache analysis. Raises RequestValidationError. Lets a batch create
    reject a malformed entry before any entry consumed a scenario step."""
    system, messages, tools = body.get("system"), body.get("messages", []), body.get("tools", [])
    shape_err = _request_shape_error(system, tools, messages)
    if shape_err is not None:
        raise RequestValidationError(shape_err)
    params = {k: body[k] for k in _EXTRA_PARAM_KEYS if k in body}
    try:
        analyze_request(system, tools, messages, top_level_cache_control=body.get("cache_control"),
                        params=params, model=model)
    except CacheControlError as e:
        raise RequestValidationError(str(e)) from e


def _request_shape_error(system: Any, tools: Any, messages: Any) -> str | None:
    """The structural checks every request route relies on before token analysis (which
    iterates these containers): the real API refuses them with 400 too."""
    if not isinstance(messages, list):
        return "messages: must be a list"
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return f"messages.{i}: must be an object with role and content"
        if m.get("content") is not None and not isinstance(m.get("content"), (str, list)):
            return f"messages.{i}.content: must be a string or a list of content blocks"
        if isinstance(m.get("content"), list) and not all(isinstance(b, dict) for b in m["content"]):
            return f"messages.{i}.content: every block must be an object"
    if system is not None and not isinstance(system, (str, list)):
        return "system: must be a string or a list of text blocks"
    if isinstance(system, list) and not all(isinstance(b, dict) for b in system):
        return "system: every block must be an object"
    if tools is not None and (not isinstance(tools, list)
                              or not all(isinstance(t, dict) for t in tools)):
        return "tools: must be a list of tool objects"
    return None


async def _wait_payload(snapshot: dict[str, Any], fut: asyncio.Future,
                        request: Request | None) -> dict[str, Any]:
    """Await the pending's answer while watching for the two things a responder cannot
    see: the client hanging up (the pending is dropped — no ghost for the responder to
    answer) and the pending's deadline passing (the server answers with
    `config.timeout_error`). Both are polled, so a fake-clock advance expires a deadline
    on the next tick too. The deadline is the one stored on the pending entry at
    registration (`/_control/pending` shows the same value); batch entries have none."""
    entry = state.pending.get(snapshot.get("pending_id"))
    deadline = entry.get("deadline") if entry else None
    if fut.done() or (request is None and deadline is None):
        return await fut
    while not fut.done():
        await asyncio.wait({fut}, timeout=_HARNESS_POLL_S)
        if fut.done():
            break
        disconnected = request is not None and await request.is_disconnected()
        if fut.done():
            # A responder answered while the disconnect check yielded: its injection was
            # accepted, so it is delivered and recorded like any other.
            break
        if disconnected:
            _discard_pending(snapshot)
            fut.set_exception(RuntimeError("client disconnected"))
            break
        if deadline is not None and _now() >= deadline:
            fut.set_result({"_inject_error": True, "content": [],
                            **copy.deepcopy(harness.config.timeout_error),
                            "_latency": dict(_harness._NO_LATENCY),
                            "_harness": {"source": "timeout"}})
            break
    return await fut


# How often a waiting request checks for a client disconnect / an expired deadline.
_HARNESS_POLL_S = 0.25


def frame_pacer(result: dict[str, Any]) -> Any:
    """The per-frame delay of a stream: `await pace()` before each frame sleeps
    `ttfb_ms` first, then `chunk_delay_ms` between frames (both from the answer's latency
    settings; zero → just a scheduling point so other tasks run between frames)."""
    lat = result.get("_latency") or {}
    ttfb = int(lat.get("ttfb_ms") or 0) / 1000.0
    chunk = int(lat.get("chunk_delay_ms") or 0) / 1000.0
    first = True

    async def pace() -> None:
        nonlocal first
        await asyncio.sleep(ttfb if first else chunk)
        first = False

    return pace


def _discard_pending(snapshot: dict[str, Any]) -> None:
    """Reliably remove a pending entry (for the error/cancel paths).

    A single dict.pop within the event loop is atomic, so no lock is needed — cleanup can
    happen without awaiting even while a CancelledError is propagating (the key to
    preventing ghost pendings).
    """
    pid = snapshot.get("pending_id")
    if pid is not None:
        state.pending.pop(pid, None)
        state.pending_generation.pop(pid, None)


async def await_resolution(snapshot: dict[str, Any], fut: asyncio.Future,
                           *, is_batch: bool = False,
                           request: Request | None = None) -> dict[str, Any]:
    """Await the control-injected response, record it in history, and return the result as a tagged dict.

    The returned "kind":
      "cleared" → was /_control/clear'd (caller returns a retryable error: 529 overloaded_error
                  on the Anthropic route, 503 on Bedrock / OpenAI)
      "error"   → injected error ({"status", "type", "message", "code", "param"})
      "ok"      → success ({"content_blocks", "usage", "cost", "model", "message_id", "stop_reason"})
      "batch_override" → a batch control endpoint (cancel / end / result) finalized this
                  entry synchronously as canceled/expired ({"type"}); nothing was recorded
                  in history (the real API doesn't bill unprocessed batch entries either)
    Provider-independent. Encoding is done by each provider.

    is_batch=True (batches route): the estimated cost gets the 50% batch discount and the
    history entry is tagged with `batch: true`.

    `request` (the route's Request) lets the wait notice a client that hung up; the
    pending is then dropped instead of lingering for a responder to answer.

    No path (cancel / unexpected exception) leaves a pending entry behind — if it did, a
    responder in a long-poll would forever keep seeing an unresolvable pending and spin.
    """
    # The clear generation the pending was registered in: an answer resolved before a
    # clear (this task resuming after the reset) must not be delivered or recorded.
    # (A pending no longer known to the registry was cleared: -1 never matches.)
    generation = state.pending_generation.get(snapshot.get("pending_id"), -1)
    try:
        response_payload = await _wait_payload(snapshot, fut, request)
    except RuntimeError as e:
        # Cancellation via clear (state already reset there) or a client disconnect
        # (the pending was dropped by the watcher).
        return {"kind": "cleared", "detail": str(e)}
    except BaseException:
        # Task cancellation from client disconnect etc.: clean up the entry, then propagate.
        _discard_pending(snapshot)
        raise

    if isinstance(response_payload, dict) and response_payload.get("_batch_override"):
        # Batch control finalized this custom_id (canceled/expired) synchronously and
        # already popped the pending — the discard here is a no-op safety net.
        _discard_pending(snapshot)
        return {"kind": "batch_override", "type": response_payload["_batch_override"]}
    if state.clear_generation != generation:
        _discard_pending(snapshot)
        return {"kind": "cleared", "detail": "cleared by control"}
    latency = harness.latency_for(response_payload.get("_latency")
                                  if isinstance(response_payload, dict) else None)
    if latency["delay_ms"]:
        # Pre-response delay (`delay_ms`): the answer is decided, the wire waits. Slept in
        # short slices so a /_control/clear or a client hang-up ends the wait at once —
        # the request is then reported as cleared and not recorded — and a cancelled
        # handler still drops its pending.
        remaining = latency["delay_ms"] / 1000.0
        try:
            while remaining > 0 and state.clear_generation == generation:
                step = min(remaining, _HARNESS_POLL_S)
                await asyncio.sleep(step)
                remaining -= step
                if request is not None and await request.is_disconnected():
                    # The client is gone, but the answer was given (a relay may have paid
                    # for it): stop waiting and record it like a delivered one.
                    break
        except BaseException:
            _discard_pending(snapshot)
            raise
        if state.clear_generation != generation:
            _discard_pending(snapshot)
            return {"kind": "cleared", "detail": "cleared by control"}
    harness_meta = response_payload.get("_harness") if isinstance(response_payload, dict) else None

    try:
        model = snapshot.get("model")
        if isinstance(response_payload, dict) and response_payload.get("_inject_error"):
            status = int(response_payload.get("status", 500))
            etype = str(response_payload.get("type") or "api_error")
            emsg = str(response_payload.get("message", "fake_server injected error"))
            after_events = response_payload.get("after_events")
            partial = _normalize_blocks(response_payload.get("content") or [])
            err_entry = {"status": status, "type": etype, "message": emsg}
            if response_payload.get("headers"):
                err_entry["headers"] = dict(response_payload["headers"])
            after_blocks = response_payload.get("after_blocks")
            if ((after_events is not None or after_blocks is not None) and snapshot.get("stream")
                    and snapshot.get("provider") != "openai"):
                # (The OpenAI route has no mid-stream error form: it answers a plain
                # HTTP error, so nothing partial went on its wire.)
                # Mid-stream failure: record the content the responder supplied for the
                # partial stream (the wire carries the first `after_events` of its events,
                # or the first `after_blocks` complete blocks). A non-streaming request gets
                # the plain HTTP error, so its history entry must not claim a mid-stream
                # failure that never went on the wire.
                if after_events is not None:
                    err_entry["after_events"] = after_events
                else:
                    err_entry["after_blocks"] = after_blocks
                err_entry["partial_content"] = partial
            await _record_and_reset(
                snapshot, response_blocks=None, injected_error=err_entry, batch=is_batch,
                harness_meta=harness_meta,
            )
            return {"kind": "error", "status": status, "type": etype, "message": emsg,
                    "_latency": latency,
                    "code": response_payload.get("code"),
                    "param": response_payload.get("param"),
                    # Extra response headers requested by the injector (e.g. retry-after).
                    "headers": response_payload.get("headers") or {},
                    # Mid-stream injection: emit this many events of `content` first, then
                    # the error event (streaming requests only; None = plain HTTP error).
                    "after_events": response_payload.get("after_events"),
                    "after_blocks": response_payload.get("after_blocks"),
                    "original_status": response_payload.get("original_status"),
                    "content_blocks": partial,
                    "message_id": f"msg_{uuid.uuid4().hex[:24]}"}

        content_blocks = response_payload.get("content") or []
        if not isinstance(content_blocks, list):
            content_blocks = []
        # Keep only the block types puppetllm models (text / tool_use / thinking /
        # redacted_thinking), dropping unknown ones ONCE here — this is the single source of
        # truth, so history, usage, and the encoded response (stream & non-stream, all
        # providers) all agree. (The encoders also skip unknown blocks defensively, but this
        # is what makes usage and /_control/history reflect exactly what the caller receives.)
        content_blocks = _normalize_blocks(content_blocks)
        usage, cost = _compute_usage(snapshot, content_blocks)
        # Optional usage override from /_control/respond (validated there): real token
        # counts, e.g. relayed from an upstream API. Partial overrides are allowed —
        # provided keys replace the approx values; cost is recomputed from the result.
        override = response_payload.get("usage")
        usage_overridden = bool(override)
        if usage_overridden:
            override = dict(override)
            n = _choices(snapshot)
            if n > 1:
                # The override describes ONE generated answer (a relay never forwards `n`);
                # the fake hands out n copies, so bill n like the real API would. Only the
                # overridden keys are scaled — the computed values are already n-folded.
                if "output_tokens" in override:
                    override["output_tokens"] = override["output_tokens"] * n
                otd = override.get("output_tokens_details")
                if isinstance(otd, dict) and isinstance(otd.get("thinking_tokens"), int):
                    override["output_tokens_details"] = {
                        **otd, "thinking_tokens": otd["thinking_tokens"] * n}
            usage.update(override)
            # Real-API invariant: thinking_tokens <= output_tokens (an override of the total
            # alone must not leave the computed thinking estimate above it).
            otd = usage.get("output_tokens_details")
            if isinstance(otd, dict) and isinstance(otd.get("thinking_tokens"), int) \
                    and otd["thinking_tokens"] > usage["output_tokens"]:
                usage["output_tokens_details"] = {**otd, "thinking_tokens": usage["output_tokens"]}
            # The 5m/1h split follows the override when it carries one, is re-derived from
            # the sim only when the total write count was NOT overridden, and otherwise
            # defaults to "all 5m" (an overridden total with no breakdown).
            if "cache_creation" not in override and "cache_creation_input_tokens" in override:
                usage["cache_creation"] = {
                    "ephemeral_5m_input_tokens": usage["cache_creation_input_tokens"],
                    "ephemeral_1h_input_tokens": 0}
            cc = usage.get("cache_creation")
            creation_1h = int(cc.get("ephemeral_1h_input_tokens") or 0) if isinstance(cc, dict) else 0
            geo = usage.get("inference_geo")
            speed = usage.get("speed")
            if not _carries_server_tools(snapshot):
                # The wire drops server-tool blocks here, so an overridden count is not
                # something the client saw either: neither reported nor billed.
                usage["server_tool_use"] = None
            stu = usage.get("server_tool_use") if isinstance(usage.get("server_tool_use"), dict) else {}
            cost = pricing.compute_cost(
                model,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cache_write_tokens=max(0, usage["cache_creation_input_tokens"] - creation_1h),
                cache_read_tokens=usage["cache_read_input_tokens"],
                cache_write_1h_tokens=creation_1h,
                inference_geo=geo if isinstance(geo, str) else None,
                speed=speed if isinstance(speed, str) else None,
                # An overridden server_tool_use is already the real (n-scaled) count.
                web_search_requests=int(stu.get("web_search_requests") or 0),
            )
        if is_batch:
            cost = _apply_batch_discount(cost)
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        if not is_batch:
            # Batch entries are outside the rate limiter (never admitted, never charged).
            harness.note_output(int(usage.get("output_tokens") or 0), _now())
        await _record_and_reset(snapshot, response_blocks=content_blocks, usage=usage,
                                cost=cost, usage_overridden=usage_overridden, batch=is_batch,
                                harness_meta=harness_meta)
        return {
            "kind": "ok",
            "_latency": latency,
            "content_blocks": content_blocks,
            "usage": usage,
            "cost": cost,
            "model": model,
            "message_id": message_id,
            # stop_reason override specified by the responder via /_control/respond (None if absent
            # = each encoder auto-determines it from whether tool_use is present).
            "stop_reason": response_payload.get("stop_reason"),
            "stop_sequence": response_payload.get("stop_sequence"),
            "stop_details": response_payload.get("stop_details"),
        }
    except BaseException:
        # Unexpected error such as in usage computation: a 500 is returned, but the pending is always cleaned up.
        _discard_pending(snapshot)
        raise


# ── SSE / stream event construction helpers ──────────────────────────

# Content block types the server models end-to-end (history / usage / encoders agree).
# Blocks the model produces itself (billed as output) …
_GENERATED_BLOCK_TYPES = ("text", "tool_use", "thinking", "redacted_thinking",
                          "server_tool_use", "mcp_tool_use")
# … and the results of server-side tools, which the real API places in the same assistant
# turn. Carried through opaquely (a fixture, never executed): non-stream responses and
# history hold them verbatim, a stream emits each as one content_block_start / stop pair.
_SERVER_RESULT_BLOCK_TYPES = ("web_search_tool_result", "web_fetch_tool_result",
                              "code_execution_tool_result", "bash_code_execution_tool_result",
                              "text_editor_code_execution_tool_result",
                              "tool_search_tool_result", "mcp_tool_result")
_MODELED_BLOCK_TYPES = _GENERATED_BLOCK_TYPES + _SERVER_RESULT_BLOCK_TYPES


def _fake_signature() -> str:
    # Opaque, base64-looking token — the real signature is a server-side MAC over the
    # thinking text; apps must treat it as opaque and round-trip it verbatim.
    return "sig_" + uuid.uuid4().hex + uuid.uuid4().hex


def _resolve_stop(content_blocks: list[dict[str, Any]], stop_reason: str | None,
                  stop_sequence: Any = None, stop_details: Any = None,
                  params: dict[str, Any] | None = None) -> tuple[str, Any, Any]:
    """Derive (stop_reason, stop_sequence, stop_details) with the real API's defaults:
    tool_use when a tool_use block is present else end_turn; stop_sequence only when
    stop_reason == stop_sequence (defaulting to the first configured one); stop_details
    only on refusal (an object with null category/explanation when not supplied)."""
    if stop_reason is None:
        stop_reason = "tool_use" if any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in content_blocks
        ) else "end_turn"
    if stop_reason == "stop_sequence":
        if stop_sequence is None:
            seqs = (params or {}).get("stop_sequences") or (params or {}).get("stop")
            if isinstance(seqs, str):
                stop_sequence = seqs
            elif isinstance(seqs, list) and seqs:
                stop_sequence = seqs[0]
    else:
        stop_sequence = None
    if stop_reason == "refusal":
        # Documented fields are guaranteed present; extra fields a relay forwards from a real
        # refusal (recommended_model, fallback_credit_token, ...) pass through untouched.
        base = {"type": "refusal", "category": None, "explanation": None}
        stop_details = {**base, **(stop_details if isinstance(stop_details, dict) else {}),
                        "type": "refusal"}
    else:
        stop_details = None
    return stop_reason, stop_sequence, stop_details


def _sse_event(event_name: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def stream_event_dicts(
    message_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    stop_reason: str | None = None,
    stop_sequence: Any = None,
    stop_details: Any = None,
    params: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Build the (event_name, data) sequence for the Anthropic streaming protocol.

    Returns a wire-format-independent dict sequence so the same event stream can be used
    for both SSE (Anthropic) and eventstream (Bedrock). If usage is unspecified, fake
    values are used. Specifying stop_reason overrides the auto-determination (based on
    whether tool_use is present), useful for testing branches like max_tokens.
    """
    # If usage was provided, use the real (estimated) values; otherwise the legacy fake values.
    # message_start carries the input side with output_tokens = 1 (the real API reports a
    # non-zero value even for an empty response); message_delta carries the CUMULATIVE usage
    # (the real API repeats input / cache counts there too).
    final_usage = _usage_wire(usage)
    start_usage = dict(final_usage)
    start_usage["output_tokens"] = 1
    delta_usage = dict(final_usage)
    stop_reason, stop_sequence, stop_details = _resolve_stop(
        content_blocks, stop_reason, stop_sequence, stop_details, params)

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
            "stop_details": None,
            "usage": start_usage,
        },
    }))

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
        elif btype == "thinking":
            # thinking_delta* (none when the text is empty = display "omitted"), then exactly
            # one signature_delta before content_block_stop — the real event sequence.
            idx += 1
            out.append(("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }))
            text = str(block.get("thinking") or "")
            chunk_size = 80
            for i in range(0, len(text), chunk_size):
                out.append(("content_block_delta", {
                    "type": "content_block_delta", "index": idx,
                    "delta": {"type": "thinking_delta", "thinking": text[i:i + chunk_size]},
                }))
            out.append(("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "signature_delta",
                          "signature": str(block.get("signature") or "")},
            }))
            out.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
        elif btype == "redacted_thinking":
            idx += 1
            out.append(("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "redacted_thinking",
                                  "data": str(block.get("data") or "")},
            }))
            out.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
        elif btype in ("tool_use", "server_tool_use", "mcp_tool_use"):
            idx += 1
            tool_id = str(block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
            start_block: dict[str, Any] = {"type": btype, "id": tool_id,
                                           "name": str(block.get("name", "")), "input": {}}
            if btype == "mcp_tool_use":
                start_block["server_name"] = str(block.get("server_name", ""))
            out.append(("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": start_block,
            }))
            # The real stream opens with an empty partial_json and then sends the JSON in
            # several chunks; SDK accumulators concatenate them, so chunking is spec-valid.
            input_json = json.dumps(block.get("input", {}) or {}, ensure_ascii=False)
            pieces = [""] + [input_json[i:i + 40] for i in range(0, len(input_json), 40)]
            for piece in pieces:
                out.append(("content_block_delta", {
                    "type": "content_block_delta", "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": piece},
                }))
            out.append(("content_block_stop", {
                "type": "content_block_stop", "index": idx,
            }))
        elif btype in _SERVER_RESULT_BLOCK_TYPES:
            # A server tool's result arrives whole: one start carrying the block, one stop.
            idx += 1
            out.append(("content_block_start", {
                "type": "content_block_start", "index": idx, "content_block": strip_private(block),
            }))
            out.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
        else:
            # Skip unknown blocks (anything outside _MODELED_BLOCK_TYPES) (don't consume an index = don't create a gap).
            continue

    out.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence,
                  "stop_details": stop_details},
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
    stop_sequence: Any = None,
    stop_details: Any = None,
    params: dict[str, Any] | None = None,
) -> list[bytes]:
    """Build the SSE byte sequence that satisfies the Anthropic streaming protocol.

    note: passing `usage` includes estimated token/cache values. If unspecified, fake
    values (legacy behavior). Like the real API, insert one `ping` right after
    message_start (SSE path only; the SDK ignores it).
    """
    events = stream_event_dicts(message_id, model, content_blocks, usage, stop_reason,
                                stop_sequence, stop_details, params)
    out = [_sse_event(events[0][0], events[0][1]), _sse_event("ping", {"type": "ping"})]
    out.extend(_sse_event(name, data) for name, data in events[1:])
    return out


def _build_non_stream_response(
    message_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    stop_reason: str | None = None,
    stop_sequence: Any = None,
    stop_details: Any = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Keep parity with the streaming path (stream_event_dicts), which only emits the
    # modeled block types: filter unknown block types here too so the same injection
    # yields the same content whether the caller used stream=True or not.
    content_blocks = [b for b in content_blocks
                      if isinstance(b, dict) and b.get("type") in _MODELED_BLOCK_TYPES]
    stop_reason, stop_sequence, stop_details = _resolve_stop(
        content_blocks, stop_reason, stop_sequence, stop_details, params)
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        # puppetllm-private keys (e.g. `_openai_custom`) never reach an Anthropic wire.
        "content": strip_private(content_blocks),
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "stop_details": stop_details,
        "usage": _usage_wire(usage),
    }


# ── Anthropic compatible endpoint ────────────────────────────────────


@app.post("/v1/messages")
async def messages(request: Request) -> Any:
    req_id = _new_request_id()
    headers = {"request-id": req_id}
    # A Bedrock / Mantle client pointed straight at the fake's root posts here with an
    # `anthropic.`-prefixed (or ARN) model id: hand it to the Bedrock adapter so the model is
    # normalized, the pending is tagged provider=bedrock, and the receipt is logged.
    try:
        peek = await request.json()
    except Exception:
        peek = None
    if isinstance(peek, dict) and _bedrock.is_bedrock_model_id(peek.get("model")):
        return await _bedrock.handle_messages(request)
    return await handle_messages(request, provider="anthropic", headers=headers)


async def handle_messages(request: Request, *, provider: str, headers: dict[str, str],
                          model_override: str | None = None,
                          extra: dict[str, Any] | None = None,
                          on_registered: Any = None) -> Any:
    """Shared Messages-API handler (Anthropic route and Bedrock's Messages/Mantle alias).

    `model_override` replaces the body model in the snapshot / response (Bedrock ids are
    normalized to the Anthropic name), `extra` is merged into the snapshot, and
    `on_registered(snapshot)` is called once the pending exists (receipt logging).
    """
    req_id = headers["request-id"]
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        # Like the real API, errors on the Anthropic path are always returned in the official envelope.
        return _anthropic_error(400, "invalid_request_error", errmsg, headers=headers)
    is_stream = bool(body.get("stream"))
    model = model_override if model_override is not None else body.get("model")

    try:
        snapshot, fut = await register_request(provider, model, body, is_stream, extra=extra,
                                               request_headers=request.headers)
    except RequestValidationError as e:
        return _anthropic_error(400, "invalid_request_error", str(e), headers=headers)
    if on_registered is not None:
        on_registered(snapshot)
    result = await await_resolution(snapshot, fut, request=request)

    if result["kind"] == "cleared":
        # 529 overloaded_error is the documented "temporarily unavailable, retry" shape.
        return _anthropic_error(529, "overloaded_error",
                                f"request cleared: {result['detail']}", headers=headers)
    model_out = model or "claude-sonnet-mock"
    if result["kind"] == "error":
        if is_stream and is_mid_stream(result):
            # Mid-stream failure, the way the real API reports it: a 200 SSE response that
            # carries some events and then an `error` event (SDKs surface it from the
            # stream; no HTTP status is involved).
            partial = partial_stream_events(result, model_out, snapshot)
            frames = [_sse_event(name, data) for name, data in partial]
            if frames:
                # The SSE path normally puts one `ping` right after `message_start`, so a
                # truncated stream must carry it too or it would not look like a prefix of
                # the stream this same server emits. It is not counted by `after_events`,
                # which counts protocol events only.
                frames.insert(1, _sse_event("ping", {"type": "ping"}))
            frames.append(_sse_event("error", {
                "type": "error",
                "error": {"type": result["type"], "message": result["message"]},
                "request_id": req_id}))

            pace = frame_pacer(result)

            async def gen_err():
                for frame in frames:
                    await pace()
                    yield frame

            # No injected error headers here: the response itself is a 200 stream, and a
            # `retry-after` on a 200 would be nonsense.
            return StreamingResponse(gen_err(), media_type="text/event-stream", headers=headers)
        return _anthropic_error(result["status"], result["type"], result["message"],
                                headers={**headers, **_error_headers(result)})

    content_blocks = result["content_blocks"]
    usage = result["usage"]
    message_id = result["message_id"]
    stop_args = (result.get("stop_reason"), result.get("stop_sequence"),
                 result.get("stop_details"), snapshot.get("params"))

    if is_stream:
        events = _build_sse_stream(message_id, model_out, content_blocks, usage, *stop_args)

        pace = frame_pacer(result)

        async def gen():
            for evt in events:
                await pace()
                yield evt

        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
    return JSONResponse(
        _build_non_stream_response(message_id, model_out, content_blocks, usage, *stop_args),
        headers=headers,
    )


def partial_stream_events(result: dict[str, Any], model: str,
                          snapshot: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """The first `after_events` events of the stream that `content_blocks` would have
    produced (for a mid-stream error injection). `after_events: 0` yields nothing — the
    error is the first thing on the wire.

    The count is clamped so the terminal `message_delta` / `message_stop` pair is never
    emitted: a stream that failed mid-flight must not also look like it completed.

    `ping` is not one of these events (it has no equivalent on the Bedrock event stream);
    the SSE caller re-inserts it after `message_start` so the bytes on the wire still form
    a prefix of a normal stream.
    """
    snapshot = snapshot or {}
    params = snapshot.get("params")
    usage, _cost = _compute_usage({**snapshot, "model": model}, result["content_blocks"])
    events = stream_event_dicts(result["message_id"], model, result["content_blocks"], usage,
                                None, None, None, params)
    return cut_partial_stream(events, result, "content_block_stop")


def is_mid_stream(result: dict[str, Any]) -> bool:
    """Whether an injected error is a mid-stream failure (`after_events` or `after_blocks`)."""
    return result.get("after_events") is not None or result.get("after_blocks") is not None


def cut_partial_stream(events: list[tuple[str, dict[str, Any]]], result: dict[str, Any],
                       block_stop: str) -> list[tuple[str, dict[str, Any]]]:
    """The prefix of a route's event list that goes on the wire before the injected error:
    the first `after_events` events, or — `after_blocks` — the message start plus the
    first N complete blocks (`block_stop` names the route's block-end event). Never the
    terminal pair, so a failed stream never looks completed."""
    body = events[:max(0, len(events) - 2)]
    if result.get("after_blocks") is not None:
        n = int(result["after_blocks"])
        stops = [i for i, (name, _d) in enumerate(body) if name == block_stop]
        if n <= 0:
            return body[:1]
        if n > len(stops):
            return body
        return body[:stops[n - 1] + 1]
    return body[:int(result.get("after_events") or 0)]


def _error_headers(result: dict[str, Any]) -> dict[str, str]:
    """Extra response headers attached to an injected error (validated in /_control/error)."""
    return {str(k): str(v) for k, v in (result.get("headers") or {}).items()}


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
    now = _now()
    async with state.lock:
        items = [
            {
                "pending_id": pid,
                "request": e["request"],
                "waiting_for_seconds": round(now - e["started_at"], 2),
                # When `pending_timeout_s` was configured at registration: the fake-clock
                # time at which the server answers for the responder, and how long that leaves.
                "deadline": e.get("deadline"),
                "timeout_in_seconds": (round(max(0.0, e["deadline"] - now), 2)
                                       if e.get("deadline") is not None else None),
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
                "waiting_for_seconds": round(_now() - oldest["started_at"], 2),
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


def _pending_for_custom_id(
    batch_id: str | None, custom_id: str,
) -> tuple[str | None, JSONResponse | None]:
    """Resolve batch_id (optional) + custom_id → pending_id. Call within the lock.

    With batch_id omitted, the custom_id must be unresolved in exactly one batch
    (mirrors the pending_id-omitted convention: ambiguity → 400 listing candidates).
    Batches still being created ARE addressable here on purpose: injecting a
    response while registration is still running is safe (auto-end is suppressed
    by the `creating` flag until the create call finishes).
    """
    if batch_id is not None:
        batch = state.batches.get(batch_id)
        if batch is None:
            return None, _plain_400(f"unknown batch_id: {batch_id}")
        entry = batch["entries"].get(custom_id)
        if entry is None:
            return None, _plain_400(f"no custom_id={custom_id} in batch {batch_id}")
        if entry["result"] is not None:
            return None, _plain_400(f"custom_id={custom_id} already resolved in batch {batch_id}")
        return entry["pending_id"], None
    matches = [b for b in state.batches.values()
               if custom_id in b["entries"] and b["entries"][custom_id]["result"] is None]
    if not matches:
        return None, _plain_400(f"no unresolved custom_id={custom_id} in any batch")
    if len(matches) > 1:
        return None, JSONResponse(
            {"error": f"custom_id={custom_id} is unresolved in multiple batches; specify batch_id",
             "batch_ids": [b["id"] for b in matches]},
            status_code=400,
        )
    return matches[0]["entries"][custom_id]["pending_id"], None


async def _resolve_target_future(
    pending_id: str | None,
    custom_id: str | None = None,
    batch_id: str | None = None,
) -> tuple[asyncio.Future[dict[str, Any]] | None, JSONResponse | None]:
    """Resolve the target pending future for injection (multi-pending).

    - `pending_id` given: use that entry (400 if it doesn't exist)
    - `custom_id` given (batches): resolve via the batch registry (batch_id optional
      when the custom_id is unambiguous). `pending_id` wins if both are given.
    - unspecified: if there is exactly 1 pending, use it (backward compatible). 0 → 400, multiple → 400
    """
    # Type guards: a non-string id (e.g. a dict from a malformed injection payload)
    # would raise on dict lookup and turn into an opaque 500.
    if pending_id is not None and not isinstance(pending_id, str):
        return None, _plain_400("pending_id must be a string")
    if custom_id is not None and not isinstance(custom_id, str):
        return None, _plain_400("custom_id must be a string")
    if batch_id is not None and not isinstance(batch_id, str):
        return None, _plain_400("batch_id must be a string")
    async with state.lock:
        if pending_id is None and custom_id is not None:
            pending_id, err = _pending_for_custom_id(batch_id, custom_id)
            if err is not None:
                return None, err
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


_USAGE_OVERRIDE_KEYS = ("input_tokens", "output_tokens",
                        "cache_creation_input_tokens", "cache_read_input_tokens")
_USAGE_OVERRIDE_ALL = _USAGE_OVERRIDE_KEYS + _USAGE_OBJECT_KEYS + _USAGE_SCALAR_KEYS
# Sane ceiling for an override token count (~1e12). Keeps cost math finite; far above any
# real context window.
_USAGE_MAX = 10 ** 12


def _respond_payload(body: dict[str, Any], *, strict: bool | None = None,
                     ) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    """Validate a `/_control/respond` body and build the future payload (also used for
    rule steps, `config.default_response` and `{"text": ...}` shorthands). → (payload, None)
    or (None, 400)."""
    if body.get("text") is not None and not body.get("content"):
        # `/_control/auto` shorthand accepted everywhere a response body is (a generated
        # client sends it next to an empty default `content`; `text: null` is unset).
        if not isinstance(body["text"], str):
            return None, _plain_400("text must be a string")
        body = {**body, "content": [{"type": "text", "text": body["text"]}]}
    content = body.get("content", [])
    # Validate the shape here and return 400 (if the encoder crashes after the future is
    # resolved, history records a success while the calling SDK gets a 500 — an inconsistency).
    if not isinstance(content, list) or not all(
        isinstance(b, dict) and isinstance(b.get("type"), str) for b in content
    ):
        return None, _plain_400("content must be a list of content-block objects with a string 'type'")
    bad_payload = _block_payload_error(content)
    if bad_payload is not None:
        return None, _plain_400(bad_payload)
    stop_reason = body.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        return None, _plain_400("stop_reason must be a string")
    stop_sequence = body.get("stop_sequence")
    if stop_sequence is not None and not isinstance(stop_sequence, str):
        return None, _plain_400("stop_sequence must be a string")
    stop_details = body.get("stop_details")
    if stop_details is not None and not isinstance(stop_details, dict):
        return None, _plain_400("stop_details must be an object")
    usage = body.get("usage")
    if isinstance(usage, dict) and usage:
        # A generated client serializes the unset token counts as null: treat as absent
        # (an all-null object overrides nothing; an empty object is still refused).
        stripped = {k: v for k, v in usage.items()
                    if not (k in _USAGE_OVERRIDE_KEYS and v is None)}
        usage = stripped or None
    if usage is not None:
        # Upper bound guards downstream cost math: without it, huge ints overflow float()
        # (int*float in pricing) or produce inf that then poisons /_control/stats JSON.
        # Object-valued keys (cache_creation / output_tokens_details / server_tool_use) and
        # scalar tags (service_tier / inference_geo / speed) are passed through so a relay can
        # forward the upstream usage object intact.
        def _ok(k: str, v: Any) -> bool:
            if k in _USAGE_OVERRIDE_KEYS:
                return type(v) is int and 0 <= v <= _USAGE_MAX
            if k == "cache_creation":
                # must be the documented object with both TTL buckets
                return (isinstance(v, dict)
                        and set(v) == {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"}
                        and all(type(x) is int and 0 <= x <= _USAGE_MAX for x in v.values()))
            if k in _USAGE_OBJECT_KEYS:
                return v is None or (isinstance(v, dict) and v and all(
                    type(x) is int and 0 <= x <= _USAGE_MAX for x in v.values()))
            if k in _USAGE_SCALAR_KEYS:
                return isinstance(v, str) and bool(v)
            return False
        if not isinstance(usage, dict) or not usage or not all(
            _ok(k, v) for k, v in usage.items()
        ) or not any(k in _USAGE_OVERRIDE_KEYS for k in usage):
            return None, _plain_400(
                f"usage must be a non-empty object with integer values in [0, {_USAGE_MAX}] "
                "for keys among: " + ", ".join(_USAGE_OVERRIDE_KEYS)
                + " (plus optional " + ", ".join(_USAGE_OBJECT_KEYS + _USAGE_SCALAR_KEYS) + ")")
    lat_err = _harness.validate_latency(body)
    if lat_err is not None:
        return None, _plain_400(lat_err)
    dropped = dropped_block_types(content)
    if dropped and (harness.config.strict_blocks if strict is None else strict):
        return None, _plain_400(_dropped_message(dropped))
    return {"content": content, "stop_reason": stop_reason, "stop_sequence": stop_sequence,
            "stop_details": stop_details, "usage": usage, "_dropped": dropped,
            "_latency": {k: body[k] for k in _harness.LATENCY_KEYS if body.get(k) is not None}}, None


def _dropped_message(dropped: list[str]) -> str:
    return (f"content: block type(s) {sorted(set(dropped))} are not modelled by puppetllm "
            f"(allowed: {list(_MODELED_BLOCK_TYPES)}); a tool_result belongs in the next "
            "user turn, not in the answer")


def _injection_ok(payload: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """`{"ok": true}` plus `dropped` when the injection lost unknown block types."""
    out: dict[str, Any] = {"ok": True, **extra}
    if payload.get("_dropped"):
        out["dropped"] = payload["_dropped"]
    return out


@app.post("/_control/respond", openapi_extra=_cm.schema_of(_cm.RespondBody))
async def respond(request: Request) -> Any:
    """Inject Body: `{"content": [...], "pending_id"?, "stop_reason"?, "stop_sequence"?,
    "stop_details"?, "usage"?}`.

    `content` blocks: text / tool_use / thinking / redacted_thinking and the server-side
    tool blocks (server_tool_use, mcp_tool_use and the *_tool_result types); anything
    else is dropped and named in the answer's `dropped` (a 400 under `config.strict_blocks`).
    `{"text": "..."}` is shorthand for one text block; `{"responses": [...]}` injects
    several answers, each with its own target. When `pending_id` is omitted, inject
    into the single pending if there is one (backward compatible). With multiple in-flight,
    `pending_id` is required — or, for batch entries, address by `custom_id` (+ optional
    `batch_id` when the custom_id appears in several batches).
    `stop_reason` (optional) overrides the auto-determination
    (e.g. "max_tokens" — for testing truncation branches; converted to finish_reason on
    the OpenAI path). `usage` (optional) overrides the approx token counts with real
    ones — a dict with any subset of input_tokens / output_tokens /
    cache_creation_input_tokens / cache_read_input_tokens as non-negative ints
    (used by relay responders forwarding to a real API).
    `delay_ms` / `ttfb_ms` / `chunk_delay_ms` / `jitter_ms` (optional) shape the timing
    of this one answer (see `/_control/config` `latency` for the defaults).
    """
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _plain_400(errmsg)
    if body.get("responses") is not None:
        return await _respond_bulk(body)
    payload, err = _respond_payload(body)
    if err is not None:
        return err
    fut, err = await _resolve_target_future(body.get("pending_id"),
                                            body.get("custom_id"), body.get("batch_id"))
    if err is not None:
        return err
    err = _safe_set_result(fut, payload)
    if err is not None:
        return err
    return _injection_ok(payload)


async def _respond_bulk(body: dict[str, Any]) -> Any:
    """`{"responses": [<respond body with its target>, ...]}`: every item is validated and
    its target resolved before any future is set, so a bad item leaves all pendings
    untouched. Targets must be distinct."""
    items = body.get("responses")
    if not isinstance(items, list) or not items or not all(isinstance(i, dict) for i in items):
        return _plain_400("responses must be a non-empty list of respond bodies")
    # Unset defaults a generated client serializes (null targets, an empty content) are
    # fine; a real outer answer or target is not.
    if any(body.get(k) is not None for k in ("pending_id", "custom_id", "batch_id", "text")) \
            or body.get("content"):
        return _plain_400("with responses, the target and the answer go inside each item")
    prepared: list[tuple[asyncio.Future, dict[str, Any]]] = []
    seen: set[int] = set()
    for i, item in enumerate(items):
        payload, err = _respond_payload(item)
        if err is not None:
            detail = json.loads(bytes(err.body).decode()).get("error")
            return _plain_400(f"responses[{i}]: {detail}")
        fut, err = await _resolve_target_future(item.get("pending_id"), item.get("custom_id"),
                                                item.get("batch_id"))
        if err is not None:
            detail = json.loads(bytes(err.body).decode()).get("error")
            return _plain_400(f"responses[{i}]: {detail}")
        if id(fut) in seen:
            return _plain_400(f"responses[{i}]: the same pending is addressed twice")
        seen.add(id(fut))
        prepared.append((fut, payload))
    results = []
    dropped: list[str] = []
    for fut, payload in prepared:
        err = _safe_set_result(fut, payload)
        results.append({"ok": err is None})
        dropped += payload.get("_dropped") or []
    out: dict[str, Any] = {"ok": all(r["ok"] for r in results), "results": results}
    if dropped:
        out["dropped"] = dropped
    if not out["ok"]:
        # A target resolved between validation and application (a clear, a timeout): the
        # other items were applied, so the status says "partially".
        return JSONResponse(out, status_code=409)
    return out


@app.post("/_control/respond_all", openapi_extra=_cm.schema_of(_cm.RespondAllBody))
async def respond_all(request: Request) -> Any:
    """One answer (a `/_control/respond` body without a target) for every live pending —
    a parallel fan-out's requests answered in one call. 400 when nothing is pending."""
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _plain_400(errmsg)
    if any(k in body for k in ("pending_id", "custom_id", "batch_id", "responses")):
        return _plain_400("respond_all takes no target; use /_control/respond for one pending")
    payload, err = _respond_payload(body)
    if err is not None:
        return err
    async with state.lock:
        targets = [(pid, e["future"]) for pid, e in state.pending.items() if not e["future"].done()]
        if not targets:
            return JSONResponse({"error": "no pending request"}, status_code=400)
        for _pid, fut in targets:
            fut.set_result(copy.deepcopy(payload))
    return _injection_ok(payload, count=len(targets), pending_ids=[pid for pid, _f in targets])


@app.post("/_control/auto", openapi_extra=_cm.schema_of(_cm.AutoBody), deprecated=True)
async def auto(request: Request) -> Any:
    """Deprecated alias: `/_control/respond` accepts `{"text": "..."}` with the same
    targeting, so use that. Injects a text-only response into one pending.

    Batch entries can also be addressed with `custom_id` (+ optional `batch_id`).
    """
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _plain_400(errmsg)
    text = body.get("text", "(empty)")
    if not isinstance(text, str):
        return _plain_400("text must be a string")
    # Only the text, the target and the latency keys are read; any other key is ignored.
    payload, err = _respond_payload({"content": [{"type": "text", "text": text}],
                                     **{k: body[k] for k in _harness.LATENCY_KEYS if k in body}})
    if err is not None:
        return err
    fut, err = await _resolve_target_future(body.get("pending_id"),
                                            body.get("custom_id"), body.get("batch_id"))
    if err is not None:
        return err
    err = _safe_set_result(fut, payload)
    if err is not None:
        return err
    return _injection_ok(payload)


# Headers an injector may not set: framing / hop-by-hop values would corrupt the response
# the server itself writes (h11 fails mid-write and the client sees a dropped connection).
_RESERVED_INJECT_HEADERS = frozenset((
    "content-length", "content-type", "transfer-encoding", "connection", "keep-alive",
    "upgrade", "te", "trailer", "proxy-connection",
))


def _validate_inject_headers(hdrs: Any) -> str | None:
    """Return an error message if `hdrs` is not a safe string → string/number map."""
    if not isinstance(hdrs, dict):
        return "headers must be an object of string → string/number"
    for k, v in hdrs.items():
        if not isinstance(k, str) or not isinstance(v, (str, int, float)) or isinstance(v, bool):
            return "headers must be an object of string → string/number"
        if not k or not k.isascii() or not all(c.isalnum() or c in "-_" for c in k):
            return f"headers: invalid header name {k!r} (ASCII letters, digits, '-' and '_')"
        if k.lower() in _RESERVED_INJECT_HEADERS:
            return f"headers: {k!r} is a framing header and cannot be injected"
        sv = str(v)
        if any(ord(c) < 32 or ord(c) > 255 for c in sv) or "\x7f" in sv:
            return f"headers: value of {k!r} must be printable Latin-1 without control characters"
    return None


def _error_payload(body: dict[str, Any], *, strict: bool | None = None,
                   ) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    """Validate a `/_control/error` body and build the future payload (also used for rule
    steps and `config.unmatched_error` / `config.timeout_error`). → (payload, None) or
    (None, 400)."""
    try:
        status = int(body.get("status", 500))
    except (TypeError, ValueError):
        return None, _plain_400("status must be an integer")
    if not (100 <= status <= 599):
        return None, _plain_400("status must be in [100, 599]")
    hdrs = body.get("headers")
    if hdrs is not None:
        err_msg = _validate_inject_headers(hdrs)
        if err_msg is not None:
            return None, _plain_400(err_msg)
    after_events = body.get("after_events")
    if after_events is not None and not (type(after_events) is int and after_events >= 0):
        return None, _plain_400("after_events must be a non-negative integer")
    after_blocks = body.get("after_blocks")
    if after_blocks is not None and not (type(after_blocks) is int and after_blocks >= 0):
        return None, _plain_400("after_blocks must be a non-negative integer")
    if after_events is not None and after_blocks is not None:
        return None, _plain_400("after_events and after_blocks cannot both be given")
    original_status = body.get("original_status")
    if original_status is not None and not (type(original_status) is int
                                            and 100 <= original_status <= 599):
        return None, _plain_400("original_status must be an integer HTTP status (100-599)")
    partial_content = body.get("content", [])
    if not isinstance(partial_content, list) or not all(
        isinstance(b, dict) and isinstance(b.get("type"), str) for b in partial_content
    ):
        return None, _plain_400("content must be a list of content-block objects with a string 'type'")
    bad_payload = _block_payload_error(partial_content)
    if bad_payload is not None:
        return None, _plain_400(bad_payload)
    lat_err = _harness.validate_latency(body)
    if lat_err is not None:
        return None, _plain_400(lat_err)
    dropped = dropped_block_types(partial_content)
    if dropped and (harness.config.strict_blocks if strict is None else strict):
        return None, _plain_400(_dropped_message(dropped))
    return {
        "_inject_error": True,
        "_dropped": dropped,
        "status": status,
        # No `type` given: the Anthropic vocabulary for that status, so the SDK raises its
        # specific class (RateLimitError for a 429) — the Bedrock and OpenAI routes already
        # derive their own type from the status.
        "type": str(body.get("type") or _ANTHROPIC_ERROR_TYPE_BY_STATUS.get(status, "api_error")),
        "message": str(body.get("message", "fake_server injected error")),
        "code": body.get("code"),
        "param": body.get("param"),
        "headers": {k: str(v) for k, v in (hdrs or {}).items()},
        "after_events": after_events,
        # Route-independent form: fail after this many complete content blocks.
        "after_blocks": after_blocks,
        # Bedrock: the status of the UPSTREAM failure a ModelErrorException /
        # ModelStreamErrorException reports as `originalStatusCode` (a 424 wrapping a 429).
        "original_status": original_status,
        "content": partial_content,
        "_latency": {k: body[k] for k in _harness.LATENCY_KEYS if body.get(k) is not None},
    }, None


@app.post("/_control/error", openapi_extra=_cm.schema_of(_cm.ErrorBody))
async def inject_error(request: Request) -> Any:
    """Error injection: make a pending request return an HTTP error.

    Body: {"status": 429, "type": "rate_limit_error", "message": "...",
           "code"?: "...", "param"?: "...", "headers"?: {"retry-after": "3"},
           "after_events"?: 3, "content"?: [...]}
    `after_events` (streaming requests only) turns the injection into a MID-STREAM
    failure: the response starts as a normal 200 stream, emits the first N events of
    `content` (default: none), then the provider's error event (Anthropic `event: error`,
    Bedrock / Converse event-stream exception frame). Non-streaming requests ignore it
    and get the plain HTTP error.
    On any of the Anthropic / Bedrock / OpenAI paths, each provider converts status/type
    into its own path's error format (code/param are used only in the OpenAI format).
    `headers` (string → string) are attached to the error response verbatim — e.g.
    `retry-after` on a 429, or `anthropic-ratelimit-*` / `x-ratelimit-*` values — so an
    app's backoff logic can be exercised. The SDK auto-retries 5xx/429/408.
    """
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _plain_400(errmsg)
    payload, err = _error_payload(body)
    if err is not None:
        return err
    fut, err = await _resolve_target_future(body.get("pending_id"),
                                            body.get("custom_id"), body.get("batch_id"))
    if err is not None:
        return err
    err = _safe_set_result(fut, payload)
    if err is not None:
        return err
    return _injection_ok(payload)


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
        # Batch entries are billed at 50%, so the amount a cache hit saved is halved
        # too — otherwise savings would be overstated for batch traffic (the discount
        # factor is recorded on the entry's cost by _apply_batch_discount).
        savings_factor = float(cost.get("batch_discount", 1.0))
        totals["cache_savings_usd"] += pricing.cache_savings_usd(model, read) * savings_factor

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
    now = _now()
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


@app.post("/_control/clear", openapi_extra=_cm.schema_of(_cm.ClearBody))
async def clear(request: Request) -> Any:
    """Reset the server. Optional body `{"config": true}` also restores the harness
    configuration to its defaults (a test fixture's full teardown); by default the
    configuration is kept and only scenario state (rules, rate window, clock) goes."""
    reset_config = False
    raw = await request.body()
    if raw.strip():
        body, errmsg = await _parse_json_body(request)
        if errmsg is not None:
            return _plain_400(errmsg)
        reset_config = body.get("config", False)
        if not isinstance(reset_config, bool):
            return _plain_400("config must be a boolean")
    async with state.lock:
        # Cancel all in-flight pendings (the main handlers gracefully return a retryable error)
        for entry in state.pending.values():
            fut = entry["future"]
            if not fut.done():
                fut.set_exception(RuntimeError("cleared by control"))
        state.pending.clear()
        state.pending_generation.clear()
        state.history.clear()
        state.turn_count = 0
        state.cache.reset()
        # Batch registry too: the collector tasks awaiting the cancelled futures see
        # kind "cleared" and return without touching the (now gone) batch objects.
        state.batches.clear()
        state.bedrock_jobs.clear()
        state.clear_generation += 1
        # Scenario state (rules, rate-limit window, fake clock) — and the configuration
        # only when asked.
        if reset_config:
            harness.config = _harness.Config()
        harness.reset()
    return {"ok": True}



# ── Harness: configuration / rules / clock ───────────────────────────


def _compile_rule(raw: Any) -> tuple[_harness.Rule | None, JSONResponse | None]:
    """Validate one rule object and pre-build each step's future payload with the same
    validators `/_control/respond` and `/_control/error` use."""
    if not isinstance(raw, dict):
        return None, _plain_400("each rule must be an object")
    # A generated client serializes unset optional fields as null: treat as absent.
    raw = {k: v for k, v in raw.items() if v is not None}
    unknown = sorted(k for k in raw if k not in ("id", "match", "steps", "repeat"))
    if unknown:
        return None, _plain_400(f"rule: unknown key(s) {unknown}")
    rid = raw.get("id")
    if rid is None:
        rid = _harness.new_rule_id()
    elif not isinstance(rid, str) or not rid:
        return None, _plain_400("rule.id must be a non-empty string")
    match = raw.get("match", {})
    if isinstance(match, dict):
        # A generated client serializes unset optional fields as null: treat as absent.
        match = {k: v for k, v in match.items() if v is not None}
    err_msg = _harness.validate_match(match)
    if err_msg is not None:
        return None, _plain_400(f"rule {rid}: {err_msg}")
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, _plain_400(f"rule {rid}: steps must be a non-empty list")
    repeat = raw.get("repeat", False)
    if not isinstance(repeat, bool):
        return None, _plain_400(f"rule {rid}: repeat must be a boolean")
    payloads: list[dict[str, Any]] = []
    dropped_steps: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if isinstance(step, dict):
            step = {k: v for k, v in step.items() if v is not None}
        if not isinstance(step, dict) or ("respond" in step) == ("error" in step):
            return None, _plain_400(
                f"rule {rid}: step {i} must be an object with exactly one of 'respond' / 'error'")
        extra = sorted(k for k in step if k not in ("respond", "error") + _harness.LATENCY_KEYS)
        if extra:
            return None, _plain_400(f"rule {rid}: step {i}: unknown key(s) {extra}")
        lat_err = _harness.validate_latency(step, f"step {i}: ")
        if lat_err is not None:
            return None, _plain_400(f"rule {rid}: {lat_err}")
        action = step.get("respond") if "respond" in step else step.get("error")
        if not isinstance(action, dict):
            return None, _plain_400(f"rule {rid}: step {i}: the action must be an object")
        # Latency on the step applies unless the action body sets the same key itself.
        merged = {**{k: step[k] for k in _harness.LATENCY_KEYS if step.get(k) is not None}, **action}
        payload, err = (_respond_payload(merged) if "respond" in step else _error_payload(merged))
        if err is not None:
            detail = json.loads(bytes(err.body).decode()).get("error")
            return None, _plain_400(f"rule {rid}: step {i}: {detail}")
        payloads.append(payload)
        for t in payload.get("_dropped") or []:
            dropped_steps.append({"rule": rid, "step": i, "type": t})
    rule = _harness.Rule(id=rid, match=dict(match), steps=list(steps), payloads=payloads,
                         repeat=repeat)
    rule.dropped = dropped_steps
    return rule, None


def _compile_rules(body: Any) -> tuple[list[_harness.Rule] | None, JSONResponse | None]:
    """`{"rules": [...]}`, a bare list, or a single rule object."""
    if isinstance(body, dict) and "rules" in body:
        raw = body["rules"]
    elif isinstance(body, dict):
        raw = [body]
    else:
        raw = body
    if not isinstance(raw, list):
        return None, _plain_400('body must be {"rules": [...]}, a list of rules, or one rule')
    rules: list[_harness.Rule] = []
    for r in raw:
        rule, err = _compile_rule(r)
        if err is not None:
            return None, err
        rules.append(rule)
    ids = [r.id for r in rules]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        return None, _plain_400(f"duplicate rule id(s): {dup}")
    return rules, None


async def _parse_rules_body(request: Request) -> tuple[Any, JSONResponse | None]:
    try:
        body = await request.json()
    except Exception as e:
        return None, _plain_400(f"invalid JSON body: {str(e)[:200]}")
    return body, None


def _rules_view(new_rules: list[Any] | None = None) -> dict[str, Any]:
    rules = [r.as_dict() for r in harness.rules]
    out = {"rules": rules,
           "unconsumed": [r.id for r in harness.rules if not r.exhausted and not r.repeat],
           "all_consumed": all(r.exhausted or r.repeat for r in harness.rules)}
    dropped = [d for r in (new_rules or []) for d in getattr(r, "dropped", [])]
    if dropped:
        # Block types the rules just posted will silently lose (unless strict_blocks).
        out["dropped"] = dropped
    return out


@app.get("/_control/rules")
async def rules_get() -> dict[str, Any]:
    """The scenario rules with their consumption counters. `unconsumed` lists the rules
    that still hold steps nobody asked for — a test's "every expected call happened"
    assertion is `all_consumed == true`."""
    return _rules_view()


@app.put("/_control/rules", openapi_extra=_cm.schema_of(_cm.RulesBody))
async def rules_put(request: Request) -> Any:
    """Replace the rule list (counters restart)."""
    body, err = await _parse_rules_body(request)
    if err is not None:
        return err
    rules, err = _compile_rules(body)
    if err is not None:
        return err
    async with state.lock:
        harness.set_rules(rules)
    return {"ok": True, **_rules_view(rules)}


@app.post("/_control/rules", openapi_extra=_cm.schema_of(_cm.RulesBody))
async def rules_post(request: Request) -> Any:
    """Append rules (a `{"rules": [...]}` object, a list, or one rule). Rules are tried in
    order, so a rule added later only answers what the earlier ones leave."""
    body, err = await _parse_rules_body(request)
    if err is not None:
        return err
    rules, err = _compile_rules(body)
    if err is not None:
        return err
    async with state.lock:
        existing = {r.id for r in harness.rules}
        clash = sorted(r.id for r in rules if r.id in existing)
        if clash:
            return _plain_400(f"rule id(s) already present: {clash} (PUT replaces the list)")
        harness.add_rules(rules)
    return {"ok": True, **_rules_view(rules)}


@app.delete("/_control/rules")
async def rules_delete() -> dict[str, Any]:
    async with state.lock:
        harness.set_rules([])
    return {"ok": True, **_rules_view()}


@app.put("/_control/rules/{rule_id}", openapi_extra=_cm.schema_of(_cm.Rule))
async def rule_put(rule_id: str, request: Request) -> Any:
    """Replace one rule in place (its position in the order is kept). Counters restart
    unless the new rule merely extends the old one's steps under the same match, in which
    case consumption carries over. A rule that does not exist yet is appended."""
    body, err = await _parse_rules_body(request)
    if err is not None:
        return err
    if not isinstance(body, dict):
        return _plain_400("body must be one rule object")
    if body.get("id") not in (None, rule_id):
        return _plain_400("rule.id in the body must match the path")
    rule, err = _compile_rule({**body, "id": rule_id})
    if err is not None:
        return err
    async with state.lock:
        rules = list(harness.rules)
        idx = next((i for i, r in enumerate(rules) if r.id == rule_id), None)
        if idx is None:
            rules.append(rule)
        else:
            old = rules[idx]
            if old.match == rule.match and rule.steps[:len(old.steps)] == old.steps:
                # Steps were only appended (or repeat toggled): keep what was consumed so a
                # rule extended after traffic does not replay its first step. A repeating
                # rule's cursor never went past its last step, so the new steps come next.
                rule.consumed = min(old.consumed, len(old.steps))
                rule.matched = old.matched
            rules[idx] = rule
        harness.set_rules(rules)
    return {"ok": True, **_rules_view([rule])}


@app.delete("/_control/rules/{rule_id}")
async def rule_delete(rule_id: str) -> Any:
    async with state.lock:
        keep = [r for r in harness.rules if r.id != rule_id]
        if len(keep) == len(harness.rules):
            return JSONResponse({"error": f"no rule with id={rule_id}"}, status_code=404)
        harness.set_rules(keep)
    return {"ok": True, **_rules_view()}


def _config_view() -> dict[str, Any]:
    return {"config": harness.config.as_dict(), "clock_offset_seconds": harness.clock_offset,
            "rate_limit_window": harness.rate_limit_snapshot(_now())}


def _finite_number(v: Any) -> float | None:
    """`v` as a finite float, or None when it is not a plain finite JSON number (bools,
    strings, NaN/inf and integers too large for a float all return None)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def apply_config(body: dict[str, Any]) -> str | None:
    """Merge `body` into the harness configuration (only the keys present change; `null`
    restores a key's default). Returns an error message or None. Shared by
    `/_control/config`, the CLI flags and the startup environment."""
    if not isinstance(body, dict):
        return "body must be an object"
    known = ("pending_timeout_s", "on_unmatched", "default_response", "unmatched_error",
             "timeout_error", "latency", "rate_limit", "seed", "strict_blocks", "default_headers")
    unknown = sorted(k for k in body if k not in known)
    if unknown:
        return f"unknown config key(s) {unknown}; allowed: {list(known)}"
    cfg = harness.config
    defaults = _harness.Config()
    new: dict[str, Any] = {}
    # The strictness the payloads in this same body are compiled under.
    strict = cfg.strict_blocks
    if "strict_blocks" in body:
        strict = body["strict_blocks"] if isinstance(body["strict_blocks"], bool) else False
    if "pending_timeout_s" in body:
        v = body["pending_timeout_s"]
        if v is not None and _finite_number(v) is None or (v is not None and v < 0):
            return "pending_timeout_s must be a non-negative number of seconds or null"
        new["pending_timeout_s"] = float(v) if v is not None else None
    if "on_unmatched" in body:
        v = body["on_unmatched"] if body["on_unmatched"] is not None else "pending"
        if v not in _harness.UNMATCHED_MODES:
            return f"on_unmatched must be one of {list(_harness.UNMATCHED_MODES)}"
        new["on_unmatched"] = v
    if "default_response" in body:
        v = body["default_response"]
        if v is not None:
            if isinstance(v, str):
                v = {"text": v}
            if not isinstance(v, dict):
                return "default_response must be a /_control/respond body, a string, or null"
            payload, err = _respond_payload(v, strict=strict)
            if err is not None:
                return "default_response: " + json.loads(bytes(err.body).decode()).get("error", "")
            v = payload
        new["default_response"] = v
    for key in ("unmatched_error", "timeout_error"):
        if key in body:
            v = body[key]
            if v is None:
                v = getattr(defaults, key)
            elif not isinstance(v, dict):
                return f"{key} must be a /_control/error body or null"
            else:
                payload, err = _error_payload(v, strict=strict)
                if err is not None:
                    return f"{key}: " + json.loads(bytes(err.body).decode()).get("error", "")
                v = payload
            new[key] = v
    if "latency" in body:
        v = body["latency"] if body["latency"] is not None else {}
        if not isinstance(v, dict) or any(k not in _harness.LATENCY_KEYS for k in v):
            return f"latency must be an object with keys among {list(_harness.LATENCY_KEYS)}"
        lat_err = _harness.validate_latency(v, "latency.")
        if lat_err is not None:
            return lat_err
        new["latency"] = {k: int(x) for k, x in v.items() if x is not None}
    if "rate_limit" in body:
        v = body["rate_limit"]
        if v is not None:
            if not isinstance(v, dict) or not v or any(k not in ("rpm", "itpm", "otpm") for k in v):
                return "rate_limit must be an object with keys among rpm / itpm / otpm, or null"
            for k, x in v.items():
                if x is not None and (type(x) is not int or x < 1):
                    return f"rate_limit.{k} must be a positive integer"
            v = {k: x for k, x in v.items() if x is not None} or None
        new["rate_limit"] = v
    if "seed" in body:
        v = body["seed"]
        if v is not None and type(v) is not int:
            return "seed must be an integer or null"
        new["seed"] = v
    if "strict_blocks" in body:
        v = body["strict_blocks"] if body["strict_blocks"] is not None else False
        if not isinstance(v, bool):
            return "strict_blocks must be a boolean"
        new["strict_blocks"] = v
    if "default_headers" in body:
        v = body["default_headers"] if body["default_headers"] is not None else {}
        hdr_err = _validate_inject_headers(v)
        if hdr_err is not None:
            return "default_headers: " + hdr_err
        new["default_headers"] = {k: str(x) for k, x in v.items()}
    for k, v in new.items():
        setattr(cfg, k, v)
    if "seed" in new:
        harness.reseed()
    if "rate_limit" in new:
        harness._window.clear()  # a new budget starts from an empty window
    if cfg.on_unmatched == "default" and cfg.default_response is None:
        # Nothing to answer with: fall back to the interactive behaviour rather than
        # silently keeping every request pending under a misleading setting.
        cfg.default_response = {"content": [{"type": "text", "text": "(puppetllm default response)"}],
                                "stop_reason": None, "stop_sequence": None, "stop_details": None,
                                "usage": None, "_latency": {}}
    return None


@app.get("/_control/config")
async def config_get() -> dict[str, Any]:
    """The harness configuration: pending timeout, unmatched policy, default response,
    latency defaults, rate limit and RNG seed. Survives `/_control/clear`."""
    return _config_view()


@app.post("/_control/config", openapi_extra=_cm.schema_of(_cm.ConfigBody))
@app.put("/_control/config", openapi_extra=_cm.schema_of(_cm.ConfigBody))
async def config_set(request: Request) -> Any:
    """Change any subset of the configuration; keys absent from the body keep their value
    and `null` restores the default. `default_response` takes a `/_control/respond` body
    (or a plain string), `unmatched_error` / `timeout_error` a `/_control/error` body."""
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _plain_400(errmsg)
    async with state.lock:
        err_msg = apply_config(body)
    if err_msg is not None:
        return _plain_400(err_msg)
    return {"ok": True, **_config_view()}


@app.get("/_control/clock")
async def clock_get() -> dict[str, Any]:
    return {"now": _now(), "clock_offset_seconds": harness.clock_offset}


@app.post("/_control/clock/advance", openapi_extra=_cm.schema_of(_cm.ClockAdvanceBody))
async def clock_advance(request: Request) -> Any:
    """Move the server's clock forward by `seconds` — the pseudo prompt cache's TTLs and
    pending deadlines elapse accordingly, without sleeping. Reset by `/_control/clear`."""
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _plain_400(errmsg)
    v = _finite_number(body.get("seconds"))
    if v is None or v < 0 or v > 10 * 365 * 86400:
        return _plain_400("seconds must be a non-negative number (at most ten years)")
    async with state.lock:
        harness.clock_offset += v
    return {"ok": True, "now": _now(), "clock_offset_seconds": harness.clock_offset}


# ── Token counting / model catalogue ─────────────────────────────────


# Messages-API parameters that have no meaning for count_tokens.
_MESSAGES_ONLY_PARAMS = ("max_tokens", "stream", "temperature", "top_p", "top_k",
                         "stop_sequences", "metadata", "service_tier")


def count_input_tokens(body: dict[str, Any], model: str | None) -> int:
    """The estimate `/v1/messages` itself would put in `usage.input_tokens` for a cold
    request (same tokenizer, same prompt-rendered params). Raises RequestValidationError."""
    params = {k: body[k] for k in _EXTRA_PARAM_KEYS if k in body}
    shape_err = _request_shape_error(body.get("system"), body.get("tools"), body.get("messages", []))
    if shape_err is not None:
        raise RequestValidationError(shape_err)
    try:
        rc = analyze_request(body.get("system"), body.get("tools") or [], body.get("messages", []),
                             top_level_cache_control=body.get("cache_control"),
                             params=params, model=model)
    except CacheControlError as e:
        raise RequestValidationError(str(e)) from e
    return rc.total_tokens


@app.post("/v1/messages/count_tokens", openapi_extra=_cm.schema_of(_cm.CountTokensBody))
async def count_tokens(request: Request) -> Any:
    """Anthropic `count_tokens`: the request's estimated input tokens, without creating a
    pending (nothing for a responder to answer, nothing in history)."""
    headers = {"request-id": _new_request_id()}
    body, errmsg = await _parse_json_body(request)
    if errmsg is not None:
        return _anthropic_error(400, "invalid_request_error", errmsg, headers=headers)
    model = body.get("model")
    if not isinstance(model, str) or not model:
        return _anthropic_error(400, "invalid_request_error", "model: field required",
                                headers=headers)
    if not isinstance(body.get("messages"), list):
        return _anthropic_error(400, "invalid_request_error", "messages: field required",
                                headers=headers)
    extra = [k for k in _MESSAGES_ONLY_PARAMS if k in body]
    if extra:
        # Generation parameters are not count_tokens inputs; refused like the real
        # endpoint refuses fields it does not know.
        return _anthropic_error(400, "invalid_request_error",
                                f"{extra[0]}: Extra inputs are not permitted", headers=headers)
    if _bedrock.is_bedrock_model_id(model):
        model = _bedrock.normalize_model_id(model).canonical
    try:
        n = count_input_tokens(body, model)
    except RequestValidationError as e:
        return _anthropic_error(400, "invalid_request_error", str(e), headers=headers)
    return JSONResponse({"input_tokens": n}, headers=headers)


def _catalogue_dialect(request: Request) -> str:
    """`/v1/models` is served in the Anthropic shape unless the caller looks like an
    OpenAI client: a bare `Authorization: Bearer` with neither `x-api-key` nor
    `anthropic-version` (the Anthropic SDK always sends the version header)."""
    h = request.headers
    if "x-api-key" in h or "anthropic-version" in h:
        return "anthropic"
    if h.get("authorization", "").lower().startswith("bearer "):
        return "openai"
    return "anthropic"


# A fixed creation timestamp so the catalogue is stable across restarts.
_CATALOGUE_CREATED = 1_767_225_600  # 2026-01-01T00:00:00Z


def _model_entry(dialect: str, model_id: str, display_name: str | None) -> dict[str, Any]:
    if dialect == "openai":
        return {"id": model_id, "object": "model", "created": _CATALOGUE_CREATED,
                "owned_by": "puppetllm"}
    return {"type": "model", "id": model_id,
            "display_name": display_name or model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_CATALOGUE_CREATED))}


@app.get("/_control/capabilities")
async def capabilities() -> dict[str, Any]:
    """The compatibility matrix: every surface with accepted / injectable / streamed /
    relayed flags, how each content block type travels on each route, and the operations
    refused on purpose."""
    return _capabilities.matrix(__version__)


@app.get("/models", include_in_schema=False)
async def list_models_root(request: Request) -> Any:
    """`/v1`-less alias (an OpenAI client whose base_url omits `/v1`)."""
    return await list_models(request)


@app.get("/models/{model_id:path}", include_in_schema=False)
async def get_model_root(model_id: str, request: Request) -> Any:
    return await get_model(model_id, request)


@app.get("/v1/models")
async def list_models(request: Request) -> Any:
    """The model catalogue (`pricing.KNOWN_MODELS`), in the dialect of the caller. It is
    informational: `/v1/messages` accepts any model id whether listed or not. The
    Anthropic shape pages with `limit` / `after_id` / `before_id` like the real endpoint,
    so an SDK's auto-pagination terminates."""
    dialect = _catalogue_dialect(request)
    family = "openai" if dialect == "openai" else "anthropic"
    entries = [_model_entry(dialect, mid, name) for mid, name in pricing.KNOWN_MODELS[family]]
    if dialect == "openai":
        return {"object": "list", "data": entries}
    qp = request.query_params
    headers = {"request-id": _new_request_id()}
    raw_limit = qp.get("limit")
    limit = 20
    if raw_limit is not None:
        try:
            limit = int(raw_limit) if raw_limit.isascii() and raw_limit.isdigit() and len(raw_limit) <= 6 else -1
        except ValueError:
            limit = -1
        if not 1 <= limit <= 1000:
            return _anthropic_error(400, "invalid_request_error",
                                    "limit: must be an integer between 1 and 1000",
                                    headers=headers)
    after_id, before_id = qp.get("after_id"), qp.get("before_id")
    if after_id is not None and before_id is not None:
        return _anthropic_error(400, "invalid_request_error",
                                "after_id and before_id cannot both be specified", headers=headers)
    ids = [e["id"] for e in entries]
    if after_id is not None:
        if after_id not in ids:
            return _anthropic_error(400, "invalid_request_error",
                                    f"after_id: unknown model id {after_id!r}", headers=headers)
        entries = entries[ids.index(after_id) + 1:]
        page = entries[:limit]
        has_more = len(entries) > limit
    elif before_id is not None:
        if before_id not in ids:
            return _anthropic_error(400, "invalid_request_error",
                                    f"before_id: unknown model id {before_id!r}", headers=headers)
        entries = entries[:ids.index(before_id)]
        page = entries[-limit:]
        has_more = len(entries) > limit
    else:
        page = entries[:limit]
        has_more = len(entries) > limit
    return JSONResponse({"data": page, "has_more": has_more,
                         "first_id": page[0]["id"] if page else None,
                         "last_id": page[-1]["id"] if page else None}, headers=headers)


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> Any:
    """One catalogue entry. An id that is not listed is synthesized rather than refused,
    since the server answers requests for any model id."""
    dialect = _catalogue_dialect(request)
    if not model_id:
        # `/v1/models/` — an empty id is nothing to synthesize.
        return await _capabilities.refuse(request, "openai" if dialect == "openai" else "anthropic")
    family = "openai" if dialect == "openai" else "anthropic"
    names = dict(pricing.KNOWN_MODELS[family])
    return _model_entry(dialect, model_id, names.get(model_id))


# ── Registering the provider adapters ────────────────────────────────
# Each provider router references the canonical helpers (register_request /
# await_resolution / stream_event_dicts / _build_non_stream_response) at call time, so
# import & include them at the very end after all helpers are defined (avoids circular imports).

from .providers import bedrock as _bedrock  # noqa: E402
from .providers import openai as _openai  # noqa: E402
from . import batches as _batches  # noqa: E402
from .providers import s3 as _s3  # noqa: E402
from .providers import converse as _converse  # noqa: E402
from .providers import bedrock_batch as _bedrock_batch  # noqa: E402

app.include_router(_bedrock.build_router())
app.include_router(_openai.build_router())
app.include_router(_batches.build_router())
app.include_router(_converse.build_router())
app.include_router(_bedrock_batch.build_router())
# Unimplemented operations of the LLM APIs answer in their own envelope (and a wrong
# method on a real path with 405) before the S3 catch-alls can see them.
app.include_router(_capabilities.build_router())
# The S3 emulation's `/{bucket}` / `/{bucket}/{key}` catch-alls go LAST so every API path
# above keeps precedence (reserved segments are refused there as bucket names too).
app.include_router(_s3.build_router())
# Outside the router: settles request paths the router's own convertor mis-handles.
app.add_middleware(_s3.ControlCharGuard)


def _is_control_plane_path(path: str) -> bool:
    """`/_control/*`, the docs pages and the schema — matched by path segment, so an S3
    bucket that merely starts with one of these names is not mistaken for them."""
    first = path.split("/", 2)[1] if path.startswith("/") else ""
    return first in ("_control", "docs", "redoc", "openapi.json")


class DefaultHeadersMiddleware:
    """Adds `config.default_headers` to every API response — not to the control plane or
    the docs — without overriding a header the route set itself."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        path = scope.get("path", "") if scope.get("type") == "http" else ""
        extra = harness.config.default_headers if path else {}
        if not extra or _is_control_plane_path(path):
            await self.inner(scope, receive, send)
            return

        async def send_with_headers(message: Any) -> None:
            if message.get("type") == "http.response.start":
                present = {k.decode("latin-1").lower() for k, _v in message.get("headers", [])}
                added = []
                for k, v in extra.items():
                    if k.lower() in present:
                        continue
                    try:
                        added.append((k.lower().encode("ascii"), v.encode("latin-1")))
                    except UnicodeEncodeError:
                        continue  # refused by the validator; never let it break a response
                message = {**message, "headers": list(message.get("headers", [])) + added}
            await send(message)

        await self.inner(scope, receive, send_with_headers)


app.add_middleware(DefaultHeadersMiddleware)


def _openapi_with_control_schemas() -> dict[str, Any]:
    """The generated document plus the control-body models under `components.schemas`,
    which is where the typed bodies' `$ref`s point (a schema embedded in one operation
    cannot carry its own definitions in OpenAPI)."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    schema.setdefault("components", {}).setdefault("schemas", {}).update(_cm.COMPONENT_SCHEMAS)
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_control_schemas  # type: ignore[method-assign]


# ── Startup configuration ────────────────────────────────────────────


def load_config_file(path: str) -> None:
    """A JSON file `{"config": {...}, "rules": [...]}` (either key optional; a bare list
    is taken as rules). Applied at startup (`--config` / `PUPPETLLM_CONFIG`)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):
        doc = {"rules": doc}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected an object with 'config' and/or 'rules'")
    if doc.get("config") is not None:
        err_msg = apply_config(doc["config"])
        if err_msg is not None:
            raise ValueError(f"{path}: config: {err_msg}")
    if doc.get("rules") is not None:
        rules, err = _compile_rules({"rules": doc["rules"]})
        if err is not None:
            raise ValueError(f"{path}: rules: {json.loads(bytes(err.body).decode()).get('error')}")
        harness.set_rules(rules)


def load_rules_file(path: str) -> None:
    """A JSON file holding a rule list (or `{"rules": [...]}`), applied at startup
    (`--rules`)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    rules, err = _compile_rules(doc)
    if err is not None:
        raise ValueError(f"{path}: {json.loads(bytes(err.body).decode()).get('error')}")
    harness.set_rules(rules)


def parse_default_response(raw: str) -> Any:
    """`--default-response` / `PUPPETLLM_DEFAULT_RESPONSE`: a `{`-prefixed value must be a
    valid JSON `/_control/respond` body (a typo there is an error, not a text answer that
    happens to look like JSON); anything else is the text to answer with."""
    if raw.lstrip().startswith("{"):
        try:
            return json.loads(raw)
        except ValueError as e:
            raise ValueError(f"not valid JSON ({e}); a value starting with '{{' must be a "
                             "/_control/respond body") from None
    return raw


def _configure_from_env() -> None:
    """PUPPETLLM_PENDING_TIMEOUT (seconds), PUPPETLLM_DEFAULT_RESPONSE (text, or a JSON
    object = a /_control/respond body; setting it selects `on_unmatched: default`),
    PUPPETLLM_ON_UNMATCHED, PUPPETLLM_SEED, PUPPETLLM_CONFIG (file). An invalid value is
    reported on stderr and ignored rather than preventing startup."""
    env = os.environ
    body: dict[str, Any] = {}
    if env.get("PUPPETLLM_PENDING_TIMEOUT"):
        try:
            body["pending_timeout_s"] = float(env["PUPPETLLM_PENDING_TIMEOUT"])
        except ValueError:
            print("[puppetllm] PUPPETLLM_PENDING_TIMEOUT must be a number of seconds; ignored",
                  file=sys.stderr)
    if env.get("PUPPETLLM_DEFAULT_RESPONSE"):
        try:
            body["default_response"] = parse_default_response(env["PUPPETLLM_DEFAULT_RESPONSE"])
            body.setdefault("on_unmatched", "default")
        except ValueError as e:
            print(f"[puppetllm] PUPPETLLM_DEFAULT_RESPONSE ignored: {e}", file=sys.stderr)
    if env.get("PUPPETLLM_ON_UNMATCHED"):
        body["on_unmatched"] = env["PUPPETLLM_ON_UNMATCHED"]
    if env.get("PUPPETLLM_SEED"):
        try:
            body["seed"] = int(env["PUPPETLLM_SEED"])
        except ValueError:
            print("[puppetllm] PUPPETLLM_SEED must be an integer; ignored", file=sys.stderr)
    if body:
        err_msg = apply_config(body)
        if err_msg is not None:
            print(f"[puppetllm] invalid harness environment setting ignored: {err_msg}",
                  file=sys.stderr)
    if env.get("PUPPETLLM_CONFIG"):
        try:
            load_config_file(env["PUPPETLLM_CONFIG"])
        except (OSError, ValueError) as e:
            print(f"[puppetllm] PUPPETLLM_CONFIG ignored: {e}", file=sys.stderr)


_configure_from_env()


# ── Stand-alone startup ──────────────────────────────────────────────


def add_serve_arguments(parser: Any) -> None:
    """The `serve` flags (shared with the `puppetllm` console command)."""
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pending-timeout", type=float, default=None, metavar="SECONDS",
                        help="answer a pending with the configured timeout error after this "
                             "long (default: wait forever for a responder)")
    parser.add_argument("--default-response", default=None, metavar="TEXT_OR_JSON",
                        help="answer every unmatched request with this text (or a JSON "
                             "/_control/respond body) instead of keeping it pending")
    parser.add_argument("--on-unmatched", choices=_harness.UNMATCHED_MODES, default=None,
                        help="what an unmatched request gets: pending (default), default, "
                             "or error")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for latency jitter (deterministic timing)")
    parser.add_argument("--config", default=None, metavar="FILE",
                        help='JSON file {"config": {...}, "rules": [...]} loaded at startup')
    parser.add_argument("--rules", default=None, metavar="FILE",
                        help="JSON file holding a rule list loaded at startup")


def serve(args: Any) -> int:
    """Run the server in-process with the parsed `serve` arguments."""
    import uvicorn

    body: dict[str, Any] = {}
    if args.pending_timeout is not None:
        body["pending_timeout_s"] = args.pending_timeout
    if args.default_response is not None:
        try:
            body["default_response"] = parse_default_response(args.default_response)
        except ValueError as e:
            print(f"[puppetllm] error: --default-response: {e}", file=sys.stderr)
            return 2
        body.setdefault("on_unmatched", "default")
    if args.on_unmatched is not None:
        body["on_unmatched"] = args.on_unmatched
    if args.seed is not None:
        body["seed"] = args.seed
    if body:
        err_msg = apply_config(body)
        if err_msg is not None:
            print(f"[puppetllm] error: {err_msg}", file=sys.stderr)
            return 2
    for path, kind in ((args.config, "config"), (args.rules, "rules")):
        if path is None:
            continue
        try:
            (load_config_file if kind == "config" else load_rules_file)(path)
        except (OSError, ValueError) as e:
            print(f"[puppetllm] error: --{kind} {path}: {e}", file=sys.stderr)
            return 2

    print(f"[puppetllm] starting on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[puppetllm] Anthropic: set ANTHROPIC_BASE_URL=http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[puppetllm] Bedrock:   point AnthropicBedrock base_url to http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[puppetllm] OpenAI:    set OPENAI_BASE_URL=http://{args.host}:{args.port}/v1  (note the /v1)", file=sys.stderr)
    if harness.rules or harness.config.on_unmatched != "pending":
        print(f"[puppetllm] harness:   {len(harness.rules)} rule(s), "
              f"on_unmatched={harness.config.on_unmatched}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m puppetllm.fake_server",
        description="Fake Anthropic/Bedrock/OpenAI API server for debugging and testing")
    add_serve_arguments(parser)
    return serve(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
