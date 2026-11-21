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
import hashlib
import json
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from .._responses import SafeJSONResponse as JSONResponse

from ..openai_wire import (image_url_to_canonical as _image_url_to_canonical,
                           openai_call_id as _openai_call_id,
                           tool_call_from_canonical, tool_call_to_canonical)

# canonical core. WARNING: circular import (same constraint as providers/bedrock.py):
# fake_server imports this module at its end and calls build_router(). Here we hold
# only a module reference; attributes like `fs.register_request` must always be
# resolved at call-time.
from .. import fake_server as fs

# Stream content in the same split width as the Anthropic route's text_delta.
_TEXT_CHUNK = 80


# ── request normalization (OpenAI chat → canonical) ─────────────────────────


def _content_text(content: Any) -> str:
    """Extract and concatenate text from OpenAI message content (str | parts list).

    `text` and assistant `refusal` parts are included; other parts (image_url etc.) are
    not stringified (and do not count toward approximate tokens).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(p.get("text", "")) if p.get("type") == "text" else str(p.get("refusal", ""))
            for p in content
            if isinstance(p, dict) and p.get("type") in ("text", "refusal")
        )
    return "" if content is None else str(content)


def _user_content_to_canonical(content: Any) -> Any:
    """User content: str passes through; a parts list is converted part by part
    (`text` → text, `image_url` → image; `input_audio` / `file` are kept verbatim so the
    responder can still see them)."""
    if not isinstance(content, list):
        return content
    out: list[Any] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t == "text":
            out.append({"type": "text", "text": str(p.get("text", ""))})
        elif t == "image_url":
            blk = _image_url_to_canonical(p)
            if blk is not None:
                out.append(blk)
        else:
            out.append(p)
    return out


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
            if not text and isinstance(m.get("refusal"), str) and m["refusal"]:
                blocks.append({"type": "text", "text": m["refusal"]})
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                blk = tool_call_to_canonical(tc)
                if blk is not None:
                    blocks.append(blk)
            messages.append({"role": "assistant", "content": blocks or text})
        else:
            # user (or unknown role): str passes through; a parts list is converted
            # (OpenAI's text part `{type:"text", text}` has the same shape as canonical;
            # `image_url` becomes a canonical image block).
            messages.append({"role": str(role or "user"),
                             "content": _user_content_to_canonical(m.get("content"))})
    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            fn = t.get("function")
            if not isinstance(fn, dict):
                continue  # skip malformed tool definitions (don't 500)
            tool: dict[str, Any] = {
                "name": fn.get("name"),
                "description": fn.get("description"),
                "input_schema": fn.get("parameters") or {},
            }
            if fn.get("strict") is not None:
                tool["strict"] = bool(fn["strict"])
            tools.append(tool)
        elif t.get("type") == "custom":
            # OpenAI custom (free-form) tools take a raw string. Represent them canonically
            # as a tool with a single string `input` and tag the origin so the OpenAI
            # encoder / relay can emit them as `custom` calls again.
            cu = t.get("custom")
            if not isinstance(cu, dict):
                continue
            tools.append({
                "name": cu.get("name"),
                "description": cu.get("description"),
                "input_schema": {"type": "object", "properties": {"input": {"type": "string"}},
                                 "required": ["input"]},
                "_openai_custom": True,
                **({"format": cu["format"]} if cu.get("format") is not None else {}),
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
# OpenAI's own vocabulary passes through (the responder can directly specify
# "content_filter" etc.); any other unknown value falls back to "stop" so a non-standard
# string never reaches the app's SDK as a bogus finish_reason.
_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    # A model refusal is an ordinary `stop` whose text lives in `message.refusal`
    # (OpenAI's shape); `content_filter` is reserved for omitted-by-filter and can still
    # be requested verbatim by a responder.
    "refusal": "stop",
    # Bedrock's guardrail stop: Chat Completions' name for an answer a filter cut.
    "guardrail_intervened": "content_filter",
    "pause_turn": "stop",
    "model_context_window_exceeded": "length",
}
_OPENAI_FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter", "function_call")


def finish_reason_for(stop_reason: str) -> str:
    if stop_reason in _OPENAI_FINISH_REASONS:
        return stop_reason
    return _FINISH_REASON_MAP.get(stop_reason, "stop")


def _to_chat_message(content_blocks: list[dict[str, Any]],
                     refusal: bool = False) -> tuple[dict[str, Any], str]:
    """canonical content blocks → (assistant message, finish_reason).

    The tool_use input is converted back to a JSON **string** (arguments) to match
    the OpenAI form. finish_reason is auto-determined by the presence of tool_calls
    (handled the same as the Anthropic route's stop_reason). A refusal (canonical
    `stop_reason: "refusal"`) takes OpenAI's refusal shape: the text goes to
    `message.refusal`, `content` is null, and finish_reason is `"stop"`.
    """
    texts = [str(b.get("text", "")) for b in content_blocks
             if isinstance(b, dict) and b.get("type") == "text"]
    tool_calls: list[dict[str, Any]] = []
    for b in content_blocks:
        if not (isinstance(b, dict) and b.get("type") == "tool_use"):
            continue
        tool_calls.append(tool_call_from_canonical(b))
    joined = "".join(texts) if texts else None
    message: dict[str, Any] = {"role": "assistant",
                               "content": None if refusal else joined,
                               "refusal": (joined or "") if refusal else None,
                               "annotations": []}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, ("tool_calls" if tool_calls else "stop")





def _usage_out(usage: dict[str, Any]) -> dict[str, Any]:
    """canonical usage (Anthropic vocabulary) → OpenAI usage vocabulary.

    prompt_tokens is the total input (= uncached + cache read + creation). The OpenAI
    route has no cache observation, so read/creation are effectively 0, but the
    conversion is written in the general form. `n` choices are already folded into the
    canonical usage by the core (so wire, history and stats agree).
    """
    read = int(usage.get("cache_read_input_tokens", 0))
    prompt = (int(usage.get("input_tokens", 0)) + read
              + int(usage.get("cache_creation_input_tokens", 0)))
    completion = int(usage.get("output_tokens", 0))
    otd = usage.get("output_tokens_details")
    reasoning = int(otd.get("thinking_tokens") or 0) if isinstance(otd, dict) else 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        # The real API always includes details (even at 0). Prevents AttributeError on the
        # reader side. Only fields the current SDK / API define are emitted.
        "prompt_tokens_details": {"cached_tokens": read, "audio_tokens": 0,
                                  "cache_write_tokens": int(usage.get("cache_creation_input_tokens", 0))},
        "completion_tokens_details": {"reasoning_tokens": reasoning, "audio_tokens": 0,
                                      "accepted_prediction_tokens": 0,
                                      "rejected_prediction_tokens": 0},
    }


def service_tier_out(requested: Any) -> str:
    """The tier the response reports: echo an explicit request tier, else `default`
    (the real API resolves `auto` to the tier actually used)."""
    return requested if isinstance(requested, str) and requested not in ("", "auto") else "default"


_TOKEN_RE = re.compile(r"\s*\S+|\s+")


def _fake_logprobs(text: str, top_k: int) -> list[dict[str, Any]]:
    """A deterministic `logprobs.content` for a text answer: whitespace-attached pieces as
    tokens, a logprob in [-2, 0) derived from each token, `top_k` alternatives per token
    (the token itself first, then made-up variants at lower probability)."""
    out: list[dict[str, Any]] = []
    for tok in _TOKEN_RE.findall(text):
        h = int(hashlib.sha256(tok.encode("utf-8", "surrogatepass")).hexdigest()[:8], 16)
        lp = -round((h % 2000) / 1000.0, 6)
        out.append({"token": tok, "logprob": lp, "top_logprobs": [
            {"token": tok.rstrip() + "_" * i if i else tok, "logprob": round(lp - 0.5 * i, 6)}
            for i in range(top_k)]})
    return _with_bytes(out, top_k)


def _with_bytes(entries: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Logprob entries in the wire shape (`bytes` filled in, `top_logprobs` cut to what
    the request asked for)."""
    out: list[dict[str, Any]] = []
    for e in entries:
        item: dict[str, Any] = {"token": e["token"], "logprob": e["logprob"],
                                "bytes": list(e["token"].encode("utf-8", "replace")),
                                "top_logprobs": []}
        if top_k > 0:
            tops = e.get("top_logprobs") or [{"token": e["token"], "logprob": e["logprob"]}]
            item["top_logprobs"] = [{"token": t["token"], "logprob": t["logprob"],
                                     "bytes": list(t["token"].encode("utf-8", "replace"))}
                                    for t in tops[:top_k]]
        out.append(item)
    return out


def _logprob_entries(text: str, injected: list[dict[str, Any]] | None, top_k: int) -> list[dict[str, Any]]:
    return (_with_bytes(injected, top_k) if injected else _fake_logprobs(text, top_k)) if text else []


def _choice_logprobs(text: str, injected: list[dict[str, Any]] | None,
                     requested: tuple[bool, int], refusal: bool = False) -> dict[str, Any] | None:
    """The `logprobs` field of a choice: None unless the request asked; else the injected
    fixture or deterministic values for the text — under `content`, or under `refusal`
    when the answer is a refusal (the real API's shape)."""
    wanted, top_k = requested
    if not wanted or not text:
        return None
    return _logprobs_field(_logprob_entries(text, injected, top_k), refusal)


def _logprobs_field(entries: list[dict[str, Any]], refusal: bool) -> dict[str, Any]:
    return {"content": None, "refusal": entries} if refusal else {"content": entries, "refusal": None}


def _logprobs_request_error(body: dict[str, Any]) -> str | None:
    """The real API's rules: `logprobs` is a boolean; `top_logprobs` only with
    `logprobs: true`, an integer 0..20."""
    if body.get("logprobs") is not None and not isinstance(body["logprobs"], bool):
        return "logprobs must be a boolean"
    top = body.get("top_logprobs")
    if top is None:
        return None
    if not body.get("logprobs"):
        return "top_logprobs is only allowed when logprobs is enabled"
    if isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= 20:
        return "top_logprobs must be an integer between 0 and 20"
    return None


def _choice_sets(content_blocks: list[dict[str, Any]], stop_reason: str | None, n: int,
                 choices: list[dict[str, Any]] | None, logprobs: list[dict[str, Any]] | None
                 ) -> list[tuple[list[dict[str, Any]], str | None, list[dict[str, Any]] | None]]:
    """(blocks, stop_reason, injected logprobs) per choice index: the individually
    injected choices, else the one answer repeated n times."""
    if choices:
        return [(c["content_blocks"], c.get("stop_reason") or stop_reason, c.get("logprobs") or logprobs)
                for c in choices]
    return [(content_blocks, stop_reason, logprobs)] * max(1, n)


def _text_of_blocks(blocks: list[dict[str, Any]]) -> str:
    return "".join(str(b.get("text", "")) for b in blocks
                   if isinstance(b, dict) and b.get("type") == "text")


def build_non_stream_response(
    completion_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any],
    created: int,
    stop_reason: str | None = None,
    n: int = 1,
    service_tier: Any = None,
    choices: list[dict[str, Any]] | None = None,
    logprobs: list[dict[str, Any]] | None = None,
    logprobs_requested: tuple[bool, int] = (False, 0),
) -> dict[str, Any]:
    # n>1: the real API returns n independent samples; the fake returns the individually
    # injected `choices` when the responder supplied them, else the same answer at every
    # index (for index compatibility with apps that read choices[i]).
    out_choices = []
    for i, (blocks, stop, lp) in enumerate(_choice_sets(content_blocks, stop_reason, n, choices, logprobs)):
        message, finish = _to_chat_message(blocks, refusal=(stop == "refusal"))
        if stop is not None:
            finish = finish_reason_for(stop)
        out_choices.append({"index": i, "message": message,
                            "logprobs": _choice_logprobs(_text_of_blocks(blocks), lp, logprobs_requested,
                                                         refusal=(stop == "refusal")),
                            "finish_reason": finish})
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": out_choices,
        "usage": _usage_out(usage),
        "service_tier": service_tier_out(service_tier),
        "system_fingerprint": None,
    }


def stream_chunk_dicts(
    completion_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any],
    created: int,
    include_usage: bool,
    stop_reason: str | None = None,
    n: int = 1,
    service_tier: Any = None,
    choices: list[dict[str, Any]] | None = None,
    logprobs: list[dict[str, Any]] | None = None,
    logprobs_requested: tuple[bool, int] = (False, 0),
) -> list[dict[str, Any]]:
    """Build the sequence of chunk dicts for the OpenAI streaming protocol.

    Matches the real API: the first delta is role, text is an incremental content
    delta, a tool call is "leading delta of id+name → following delta of arguments",
    and the terminating delta is empty + finish_reason. When include_usage is set,
    every chunk carries `usage: null`, and after termination a usage-only chunk (empty
    choices) is sent (per the real API spec). With n>1 each choice index streams its own
    answer (the individually injected `choices`, else the same one), interleaved by
    position. A content delta carries the logprobs of its tokens when the request asked.
    """
    tier = service_tier_out(service_tier)

    def chunk(delta: dict[str, Any], finish: str | None = None, index: int = 0,
              lp: dict[str, Any] | None = None) -> dict[str, Any]:
        c = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "service_tier": tier, "system_fingerprint": None,
            "choices": [{"index": index, "delta": delta,
                         "logprobs": lp, "finish_reason": finish}],
        }
        if include_usage:
            c["usage"] = None
        return c

    def deltas_for(blocks: list[dict[str, Any]], stop: str | None,
                   injected: list[dict[str, Any]] | None
                   ) -> tuple[list[tuple[dict[str, Any], dict[str, Any] | None]], str]:
        refusal = stop == "refusal"
        wanted, top_k = logprobs_requested
        # The whole answer's logprobs, handed out piece by piece as the text streams.
        whole_text = _text_of_blocks(blocks)
        entries = _logprob_entries(whole_text, injected, top_k) if wanted else []
        pos = 0  # next entry to hand out
        out: list[tuple[dict[str, Any], dict[str, Any] | None]] = [
            ({"role": "assistant", "content": "", "refusal": None}, None)]
        finish = "stop"
        tc_index = 0
        for b in blocks:
            btype = b.get("type") if isinstance(b, dict) else None
            if btype == "text":
                text = str(b.get("text", ""))
                pieces = [text[i:i + _TEXT_CHUNK]
                          for i in range(0, len(text), _TEXT_CHUNK)] or [""]
                for p in pieces:
                    lp = None
                    if wanted:
                        start, covered = pos, 0
                        while pos < len(entries) and covered < len(p):
                            covered += len(entries[pos]["token"])
                            pos += 1
                        lp = _logprobs_field(entries[start:pos], refusal)
                    # a refusal streams as `delta.refusal` (OpenAI's shape), never as content
                    out.append(({"refusal": p} if refusal else {"content": p}, lp))
            elif btype == "tool_use":
                finish = "tool_calls"
                call = tool_call_from_canonical(b)
                if call["type"] == "custom":
                    out.append(({"tool_calls": [{
                        "index": tc_index, "id": call["id"], "type": "custom",
                        "custom": {"name": call["custom"]["name"], "input": ""}}]}, None))
                    out.append(({"tool_calls": [{
                        "index": tc_index, "custom": {"input": call["custom"]["input"]}}]}, None))
                else:
                    out.append(({"tool_calls": [{
                        "index": tc_index, "id": call["id"], "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": ""},
                    }]}, None))
                    out.append(({"tool_calls": [{
                        "index": tc_index,
                        "function": {"arguments": call["function"]["arguments"]},
                    }]}, None))
                tc_index += 1
            # Skip other blocks (thinking etc. have no Chat Completions representation).
        if stop is not None:
            finish = finish_reason_for(stop)
        return out, finish


    per_choice = [deltas_for(blocks, stop, lp)
                  for blocks, stop, lp in _choice_sets(content_blocks, stop_reason, n, choices, logprobs)]
    out: list[dict[str, Any]] = []
    for pos in range(max(len(d) for d, _f in per_choice)):
        for i, (deltas, _finish) in enumerate(per_choice):
            if pos < len(deltas):
                d, lp = deltas[pos]
                out.append(chunk(d, index=i, lp=lp))
    for i, (_deltas, finish) in enumerate(per_choice):
        out.append(chunk({}, finish=finish, index=i))
    if include_usage:
        out.append({
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "service_tier": tier, "system_fingerprint": None,
            "choices": [], "usage": _usage_out(usage),
        })
    return out


# Anthropic-vocabulary error types → OpenAI vocabulary (used when an injector passed an
# Anthropic name or the default `api_error` to the OpenAI route).
_ERROR_TYPE_MAP = {
    "invalid_request_error": "invalid_request_error",
    "authentication_error": "authentication_error",
    "permission_error": "permission_error",
    "not_found_error": "not_found_error",
    "rate_limit_error": "rate_limit_error",
    "overloaded_error": "server_error",
    "api_error": None,  # resolve by status
    "timeout_error": "server_error",
    "billing_error": "insufficient_quota",
    "request_too_large": "invalid_request_error",
}


def error_type_for(status: int, etype: str | None) -> str:
    if etype and etype not in _ERROR_TYPE_MAP:
        return etype  # already OpenAI-style (or caller-chosen) — pass through
    mapped = _ERROR_TYPE_MAP.get(etype or "api_error")
    if mapped:
        return mapped
    if status == 429:
        return "rate_limit_error"
    if status == 503:
        return "service_unavailable_error"
    if status >= 500:
        return "server_error"
    if status in (401,):
        return "authentication_error"
    if status in (403,):
        return "permission_error"
    if status == 404:
        return "not_found_error"
    return "invalid_request_error"


def _openai_error_response(status: int, etype: str, message: str,
                           code: Any = None, param: Any = None,
                           headers: dict[str, str] | None = None) -> JSONResponse:
    """OpenAI-style error response. The SDK determines the exception kind by HTTP status.

    code/param are passed through from the same-named fields of /_control/error
    (e.g. code="rate_limit_exceeded" — for testing apps that branch on code). The error
    `type` is translated to OpenAI vocabulary when the injector used an Anthropic name.
    """
    return JSONResponse(
        {"error": {"message": message, "type": error_type_for(status, etype),
                   "param": param, "code": code}},
        status_code=status, headers=headers,
    )


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/chat/completions", include_in_schema=False)
    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        started = time.time()
        headers = {"x-request-id": f"req_{uuid.uuid4().hex[:24]}",
                   "openai-version": "2020-10-01", "openai-processing-ms": "0"}

        def _finish_headers() -> dict[str, str]:
            headers["openai-processing-ms"] = str(max(0, int((time.time() - started) * 1000)))
            return headers

        body, errmsg = await fs._parse_json_body(request)
        if errmsg is not None:
            fs.note_rejected("openai", None, headers["x-request-id"], errmsg, path=request.url.path)
            return _openai_error_response(400, "invalid_request_error", errmsg,
                                          headers=headers)
        is_stream = bool(body.get("stream"))
        model = body.get("model")
        # Container shapes the normalizer iterates; the real API refuses these with 400.
        if not isinstance(body.get("messages"), list):
            fs.note_rejected("openai", model, headers["x-request-id"], "messages: must be a list",
                             path=request.url.path)
            return _openai_error_response(400, "invalid_request_error",
                                          "messages: must be a list", headers=headers,
                                          param="messages")
        if body.get("tools") is not None and not isinstance(body.get("tools"), list):
            fs.note_rejected("openai", model, headers["x-request-id"], "tools: must be a list",
                             path=request.url.path)
            return _openai_error_response(400, "invalid_request_error",
                                          "tools: must be a list", headers=headers, param="tools")
        lp_err = _logprobs_request_error(body)
        if lp_err is not None:
            fs.note_rejected("openai", model, headers["x-request-id"], lp_err, path=request.url.path)
            return _openai_error_response(400, "invalid_request_error", lp_err, headers=headers,
                                          param="logprobs" if lp_err.startswith("logprobs") else "top_logprobs")
        canonical = normalize_chat_body(body)

        try:
            n = max(1, min(int(body.get("n") or 1), 16))
        except (TypeError, ValueError):
            n = 1
        try:
            snapshot, fut = await fs.register_request(
                "openai", model, canonical, is_stream, simulate_cache=False,
                extra={"choices": n} if n > 1 else None,
                request_headers=request.headers, request_path=request.url.path,
                request_id=headers["x-request-id"])
        except fs.RequestValidationError as e:
            return _openai_error_response(400, "invalid_request_error", str(e),
                                          headers=_finish_headers())
        result = await fs.await_resolution(snapshot, fut, request=request)

        if result["kind"] == "cleared":
            return _openai_error_response(
                503, "service_unavailable_error", f"request cleared: {result['detail']}",
                headers=_finish_headers())
        if result["kind"] == "error":
            return _openai_error_response(
                result["status"], result["type"], result["message"],
                code=result.get("code"), param=result.get("param"),
                headers={**_finish_headers(), **fs._error_headers(result)})

        model_out = model or "gpt-mock"
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        stop_reason = result.get("stop_reason")
        logprobs_requested = (bool(body.get("logprobs")), body.get("top_logprobs") or 0)
        choices, injected_logprobs = result.get("choices"), result.get("logprobs")

        if is_stream:
            so = body.get("stream_options")
            include_usage = isinstance(so, dict) and bool(so.get("include_usage"))
            payloads = [
                f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode("utf-8")
                for c in stream_chunk_dicts(
                    completion_id, model_out, result["content_blocks"],
                    result["usage"], created, include_usage, stop_reason, n,
                    body.get("service_tier"), choices, injected_logprobs, logprobs_requested)
            ]
            payloads.append(b"data: [DONE]\n\n")

            pace = fs.frame_pacer(result)

            async def gen():
                for p in payloads:
                    await pace()
                    yield p

            return StreamingResponse(gen(), media_type="text/event-stream",
                                     headers=_finish_headers())
        return JSONResponse(build_non_stream_response(
            completion_id, model_out, result["content_blocks"],
            result["usage"], created, stop_reason, n, body.get("service_tier"),
            choices, injected_logprobs, logprobs_requested),
            headers=_finish_headers())

    return router
