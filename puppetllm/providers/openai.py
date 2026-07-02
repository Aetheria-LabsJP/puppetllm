"""OpenAI Chat Completions compatible adapter.

Formal spec: README.md

Pointing the base_url of the `openai` SDK (or any OpenAI-compatible client) at
`http://<host>:<port>/v1` makes `POST /v1/chat/completions` arrive on this route.
Since its path differs from the Anthropic route `/v1/messages` there is no collision,
and **no switch configuration is needed** (the provider is auto-detected by path,
just like the Bedrock route).

Differences from the Anthropic route:

- The request is in OpenAI chat form (system/developer are roles inside messages,
  tools are `{type:"function", function:{...}}`, tool execution results are
  `role:"tool"` messages, arguments are JSON **strings**) → here it is
  **normalized to canonical (Anthropic-like)** and placed on pending. The responder
  can read the same shape regardless of provider (system / messages / tools /
  tool_result block).
- The response is converted from canonical content blocks (text / tool_use) into
  `chat.completion` (non-stream) / `chat.completion.chunk` SSE (stream, terminated
  by `data: [DONE]`).
- No pseudo prompt-cache observation is performed (OpenAI's cache is an automatic
  scheme not based on cache_control, so it is out of scope for simulation). The
  pending cache is always "none".
- Error injection is returned in OpenAI form `{"error": {"message", "type", "param",
  "code"}}` + HTTP status (the SDK maps exceptions by status).

Response injection (/_control/respond etc.) is fully shared with the Anthropic /
Bedrock routes.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

# canonical core. ⚠ Beware circular import (same constraint as providers/bedrock.py):
# fake_server imports this module at its end and calls build_router(). Here we hold
# only a module reference; attributes like `fs.register_request` must always be
# resolved at call-time.
from .. import fake_server as fs

# Stream content in the same split width as the Anthropic route's text_delta.
_TEXT_CHUNK = 80


# ── request normalization (OpenAI chat → canonical) ─────────────────────────


def _content_text(content: Any) -> str:
    """Extract and concatenate text from OpenAI message content (str | parts list).

    Parts other than text (image_url etc.) are not stringified (and do not count
    toward approximate tokens).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(p.get("text", "")) for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return "" if content is None else str(content)


def normalize_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    """OpenAI chat-form body → canonical {system, messages, tools, max_tokens}.

    - role system/developer → canonical `system` (concatenated if multiple)
    - role tool → tool_result block inside a canonical user turn
    - assistant's tool_calls → tool_use block (the arguments JSON string is parsed to a dict)
    - tools' function definition → {name, description, input_schema}
    """
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("system", "developer"):
            system_parts.append(_content_text(m.get("content")))
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": str(m.get("tool_call_id") or ""),
                "content": _content_text(m.get("content")),
            }
            # Consecutive role:tool messages are merged into a single user turn
            # (in the real Anthropic form, parallel tool results become multiple
            # tool_result blocks within one message).
            prev = messages[-1] if messages else None
            if (prev is not None and prev.get("role") == "user"
                    and isinstance(prev.get("content"), list) and prev["content"]
                    and all(isinstance(x, dict) and x.get("type") == "tool_result"
                            for x in prev["content"])):
                prev["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _content_text(m.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                fn = fn if isinstance(fn, dict) else {}
                raw = fn.get("arguments")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except ValueError:
                    args = {"_raw": raw}  # don't swallow broken arguments; expose them as-is
                blocks.append({"type": "tool_use", "id": str(tc.get("id") or ""),
                               "name": str(fn.get("name") or ""), "input": args})
            messages.append({"role": "assistant", "content": blocks or text})
        else:
            # user (or unknown role): pass content through as-is, whether str or parts list.
            # OpenAI's text part `{type:"text", text}` has the same shape as canonical, so it's compatible.
            messages.append({"role": str(role or "user"), "content": m.get("content")})
    tools = []
    for t in body.get("tools") or []:
        if isinstance(t, dict) and t.get("type") == "function":
            fn = t.get("function")
            if not isinstance(fn, dict):
                continue  # skip malformed tool definitions (don't 500)
            tools.append({
                "name": fn.get("name"),
                "description": fn.get("description"),
                "input_schema": fn.get("parameters") or {},
            })
    max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_tokens")
    out = {
        "system": "\n\n".join(p for p in system_parts if p) or None,
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
    }
    # Auxiliary parameters (tool_choice / response_format / temperature etc.) are also
    # passed through onto canonical, and the core (register_request) picks them up into
    # snapshot["params"].
    for k in fs._EXTRA_PARAM_KEYS:
        if k in body:
            out[k] = body[k]
    return out


# ── response conversion (canonical blocks → OpenAI chat form) ───────────────

# canonical (Anthropic vocabulary) stop_reason → OpenAI finish_reason.
# Unknown values pass through as-is (the responder can directly specify OpenAI
# vocabulary such as "content_filter").
_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
}


def _openai_call_id(raw: Any) -> str:
    """Map a canonical tool_use id to the OpenAI `call_...` id form.

    The canonical (Anthropic-style) core assigns `toolu_...` ids to tool_use blocks
    that lack one, so by the time this runs the id is almost always present. Rewrite the
    `toolu_` prefix to `call_` (deterministic, so the same block yields the same id in
    both the non-stream and stream encoders); ids the responder set explicitly are kept
    as-is; a truly empty id falls back to a fresh `call_...`.
    """
    s = str(raw or "")
    if s.startswith("toolu_"):
        return "call_" + s[len("toolu_"):]
    return s or f"call_{uuid.uuid4().hex[:24]}"


def _to_chat_message(content_blocks: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """canonical content blocks → (assistant message, finish_reason).

    The tool_use input is converted back to a JSON **string** (arguments) to match
    the OpenAI form. finish_reason is auto-determined by the presence of tool_calls
    (handled the same as the Anthropic route's stop_reason).
    """
    texts = [str(b.get("text", "")) for b in content_blocks
             if isinstance(b, dict) and b.get("type") == "text"]
    tool_calls: list[dict[str, Any]] = []
    for b in content_blocks:
        if not (isinstance(b, dict) and b.get("type") == "tool_use"):
            continue
        tool_calls.append({
            "id": _openai_call_id(b.get("id")),
            "type": "function",
            "function": {
                "name": str(b.get("name", "")),
                "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
            },
        })
    message: dict[str, Any] = {"role": "assistant",
                               "content": "".join(texts) if texts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, ("tool_calls" if tool_calls else "stop")


def _usage_out(usage: dict[str, Any]) -> dict[str, Any]:
    """canonical usage (Anthropic vocabulary) → OpenAI usage vocabulary.

    prompt_tokens is the total input (= uncached + cache read + creation). The OpenAI
    route has no cache observation, so read/creation are effectively 0, but the
    conversion is written in the general form.
    """
    read = int(usage.get("cache_read_input_tokens", 0))
    prompt = (int(usage.get("input_tokens", 0)) + read
              + int(usage.get("cache_creation_input_tokens", 0)))
    completion = int(usage.get("output_tokens", 0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        # The real API always includes details (even at 0). Prevents AttributeError on the reader side.
        "prompt_tokens_details": {"cached_tokens": read, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0,
                                      "accepted_prediction_tokens": 0,
                                      "rejected_prediction_tokens": 0},
    }


def build_non_stream_response(
    completion_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any],
    created: int,
    stop_reason: str | None = None,
    n: int = 1,
) -> dict[str, Any]:
    message, finish = _to_chat_message(content_blocks)
    if stop_reason is not None:
        finish = _FINISH_REASON_MAP.get(stop_reason, stop_reason)
    # n>1: the real API returns n independent samples, but the fake duplicates the same
    # injected content (for index compatibility with apps that read choices[i]; it does
    # not simulate content diversity).
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": i, "message": message,
                     "logprobs": None, "finish_reason": finish}
                    for i in range(max(1, n))],
        "usage": _usage_out(usage),
    }


def stream_chunk_dicts(
    completion_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any],
    created: int,
    include_usage: bool,
    stop_reason: str | None = None,
) -> list[dict[str, Any]]:
    """Build the sequence of chunk dicts for the OpenAI streaming protocol.

    Matches the real API: the first delta is role, text is an incremental content
    delta, a tool call is "leading delta of id+name → following delta of arguments",
    and the terminating delta is empty + finish_reason. When include_usage is set,
    every chunk carries `usage: null`, and after termination a usage-only chunk (empty
    choices) is sent (per the real API spec).
    """
    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        c = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta,
                         "logprobs": None, "finish_reason": finish}],
        }
        if include_usage:
            c["usage"] = None
        return c

    out = [chunk({"role": "assistant", "content": ""})]
    finish = "stop"
    tc_index = 0
    for b in content_blocks:
        btype = b.get("type") if isinstance(b, dict) else None
        if btype == "text":
            text = str(b.get("text", ""))
            pieces = [text[i:i + _TEXT_CHUNK]
                      for i in range(0, len(text), _TEXT_CHUNK)] or [""]
            for p in pieces:
                out.append(chunk({"content": p}))
        elif btype == "tool_use":
            finish = "tool_calls"
            tool_id = _openai_call_id(b.get("id"))
            out.append(chunk({"tool_calls": [{
                "index": tc_index, "id": tool_id, "type": "function",
                "function": {"name": str(b.get("name", "")), "arguments": ""},
            }]}))
            out.append(chunk({"tool_calls": [{
                "index": tc_index,
                "function": {"arguments": json.dumps(b.get("input") or {},
                                                     ensure_ascii=False)},
            }]}))
            tc_index += 1
        # Skip unknown blocks (anything other than text/tool_use) (same as the Anthropic route).
    if stop_reason is not None:
        finish = _FINISH_REASON_MAP.get(stop_reason, stop_reason)
    out.append(chunk({}, finish=finish))
    if include_usage:
        out.append({
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [], "usage": _usage_out(usage),
        })
    return out


def _openai_error_response(status: int, etype: str, message: str,
                           code: Any = None, param: Any = None,
                           headers: dict[str, str] | None = None) -> JSONResponse:
    """OpenAI-style error response. The SDK determines the exception kind by HTTP status.

    code/param are passed through from the same-named fields of /_control/error
    (e.g. code="rate_limit_exceeded" — for testing apps that branch on code).
    """
    return JSONResponse(
        {"error": {"message": message, "type": etype, "param": param, "code": code}},
        status_code=status, headers=headers,
    )


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        headers = {"x-request-id": f"req_{uuid.uuid4().hex[:24]}"}
        body, errmsg = await fs._parse_json_body(request)
        if errmsg is not None:
            return _openai_error_response(400, "invalid_request_error", errmsg,
                                          headers=headers)
        is_stream = bool(body.get("stream"))
        model = body.get("model")
        canonical = normalize_chat_body(body)

        snapshot, fut = await fs.register_request(
            "openai", model, canonical, is_stream, simulate_cache=False)
        result = await fs.await_resolution(snapshot, fut)

        if result["kind"] == "cleared":
            return _openai_error_response(
                503, "service_unavailable", f"request cleared: {result['detail']}",
                headers=headers)
        if result["kind"] == "error":
            return _openai_error_response(
                result["status"], result["type"], result["message"],
                code=result.get("code"), param=result.get("param"), headers=headers)

        model_out = model or "gpt-mock"
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        stop_reason = result.get("stop_reason")

        if is_stream:
            so = body.get("stream_options")
            include_usage = isinstance(so, dict) and bool(so.get("include_usage"))
            # note: n>1 in streaming is unsupported (rarely used in practice, so choice 0 only).
            payloads = [
                f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode("utf-8")
                for c in stream_chunk_dicts(
                    completion_id, model_out, result["content_blocks"],
                    result["usage"], created, include_usage, stop_reason)
            ]
            payloads.append(b"data: [DONE]\n\n")

            async def gen():
                for p in payloads:
                    yield p
                    await asyncio.sleep(0)

            return StreamingResponse(gen(), media_type="text/event-stream",
                                     headers=headers)
        try:
            n = max(1, min(int(body.get("n") or 1), 16))
        except (TypeError, ValueError):
            n = 1
        return JSONResponse(build_non_stream_response(
            completion_id, model_out, result["content_blocks"],
            result["usage"], created, stop_reason, n), headers=headers)

    return router
