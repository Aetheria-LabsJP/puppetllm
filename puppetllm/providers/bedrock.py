"""Bedrock (InvokeModel / InvokeModelWithResponseStream) compatible adapter.

Formal spec: README.md

When the base_url of the `AnthropicBedrock` SDK / boto3 `bedrock-runtime` is pointed
at the fake server, requests arrive in the following form (differences from the
Anthropic route):

- model goes in the **URL path** `/model/{model_id}/invoke[-with-response-stream]` (not in the body)
- body is the messages form with `anthropic_version` (`max_tokens` / `system` / `messages` / `tools`)
- streaming is not SSE but **AWS event stream binary** (`application/vnd.amazon.eventstream`)

It reuses the canonical core (fake_server)'s register_request / await_resolution /
stream_event_dicts / _build_non_stream_response as-is, and here it is responsible only
for "extracting model from URL" and "event stream encoding". Response injection
(/_control/respond etc.) is fully shared with the Anthropic route.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import eventstream

# canonical core. ⚠ Beware circular import: fake_server imports this module at its end
# and calls build_router(). Here we hold only a module reference; attributes like
# `fs.register_request` must **always be resolved at call-time (inside the route
# handler)** — touching fs.* at import time grabs a half-executed fake_server and
# results in AttributeError. This is not a problem on the normal path (startup via
# fake_server), but note that you hit this ordering constraint if you write, e.g.,
# tests that import `providers.bedrock` standalone.
from .. import fake_server as fs

_EVENTSTREAM_MEDIA = "application/vnd.amazon.eventstream"


def _bedrock_error_response(status: int, etype: str, message: str) -> JSONResponse:
    """Bedrock-style error response (status + __type + x-amzn-ErrorType header)."""
    return JSONResponse(
        {"message": message, "__type": etype},
        status_code=status,
        headers={"x-amzn-ErrorType": etype},
    )


def build_router() -> APIRouter:
    router = APIRouter()

    # NOTE: `{model_id}` is a single path segment. A foundation model id
    # (`anthropic.claude-3-5-sonnet-20241022-v2:0`) or a cross-region inference profile
    # (`us.anthropic.claude-...`) contains no `/`, so it's fine. An application
    # inference profile ARN that contains `/` gets `%2F`→`/` decoded on the ASGI side
    # and won't match routing (404). Unsupported for debug purposes (apps use a plain
    # model id).

    @router.post("/model/{model_id}/invoke")
    async def invoke(model_id: str, request: Request) -> Any:
        req_id = str(uuid.uuid4())
        body, errmsg = await fs._parse_json_body(request)
        if errmsg is not None:
            return _bedrock_error_response(400, "ValidationException", errmsg)

        snapshot, fut = await fs.register_request("bedrock", model_id, body, is_stream=False)
        result = await fs.await_resolution(snapshot, fut)

        if result["kind"] == "cleared":
            return _bedrock_error_response(503, "ServiceUnavailableException",
                                           f"request cleared: {result['detail']}")
        if result["kind"] == "error":
            return _bedrock_error_response(
                result["status"], result["type"], result["message"]
            )

        # Bedrock invoke (non-stream) returns the model's raw response body (= Anthropic message JSON).
        # Like real Bedrock, attach requestid and token-count headers (for apps that read them; approximate values).
        usage = result["usage"]
        in_total = (usage.get("input_tokens", 0)
                    + usage.get("cache_read_input_tokens", 0)
                    + usage.get("cache_creation_input_tokens", 0))
        return JSONResponse(
            fs._build_non_stream_response(
                result["message_id"], model_id, result["content_blocks"], usage,
                result.get("stop_reason"),
            ),
            headers={
                "x-amzn-requestid": req_id,
                "X-Amzn-Bedrock-Input-Token-Count": str(in_total),
                "X-Amzn-Bedrock-Output-Token-Count": str(usage.get("output_tokens", 0)),
            },
        )

    @router.post("/model/{model_id}/invoke-with-response-stream")
    async def invoke_stream(model_id: str, request: Request) -> Any:
        req_id = str(uuid.uuid4())
        body, errmsg = await fs._parse_json_body(request)
        if errmsg is not None:
            return _bedrock_error_response(400, "ValidationException", errmsg)

        snapshot, fut = await fs.register_request("bedrock", model_id, body, is_stream=True)
        result = await fs.await_resolution(snapshot, fut)

        if result["kind"] == "cleared":
            return _bedrock_error_response(503, "ServiceUnavailableException",
                                           f"request cleared: {result['detail']}")
        if result["kind"] == "error":
            # Errors before streaming starts are returned via HTTP status (the SDK maps exceptions by status).
            return _bedrock_error_response(
                result["status"], result["type"], result["message"]
            )

        events = fs.stream_event_dicts(
            result["message_id"], model_id, result["content_blocks"], result["usage"],
            result.get("stop_reason"),
        )
        # Real Bedrock bundles invocationMetrics into the final chunk (message_stop).
        usage = result["usage"]
        latency_ms = max(0, int((time.time() - snapshot.get("received_at", time.time())) * 1000))
        for _name, data in events:
            if data.get("type") == "message_stop":
                data["amazon-bedrock-invocationMetrics"] = {
                    "inputTokenCount": (usage.get("input_tokens", 0)
                                        + usage.get("cache_read_input_tokens", 0)
                                        + usage.get("cache_creation_input_tokens", 0)),
                    "outputTokenCount": usage.get("output_tokens", 0),
                    "invocationLatency": latency_ms,
                    "firstByteLatency": latency_ms,
                }
        frames = [eventstream.encode_chunk(data) for _name, data in events]

        async def gen():
            for frame in frames:
                yield frame
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type=_EVENTSTREAM_MEDIA,
                                 headers={"x-amzn-requestid": req_id})

    return router
