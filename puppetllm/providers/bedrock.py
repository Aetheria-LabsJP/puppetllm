"""Bedrock (InvokeModel / InvokeModelWithResponseStream) compatible adapter.

Formal spec: README.md

When the base_url of the `AnthropicBedrock` SDK / boto3 `bedrock-runtime` is pointed
at the fake server, requests arrive in the following form (differences from the
Anthropic route):

- model goes in the **URL path** `/model/{model_id}/invoke[-with-response-stream]` (not in the body)
- body is the messages form with `anthropic_version` (`max_tokens` / `system` / `messages` / `tools`)
- streaming is not SSE but **AWS event stream binary** (`application/vnd.amazon.eventstream`)

It reuses the canonical core (fake_server)'s register_request / await_resolution /
stream_event_dicts / _build_non_stream_response as-is. This module is responsible for:

- extracting the model from the URL and normalizing the Bedrock model id
  (`anthropic.claude-...-v1:0`, suffix-less `anthropic.claude-opus-5`, cross-region
  `us.`/`eu.`/`apac.`/`jp.`/`au.`/`global.`/`us-gov.` prefixes, and foundation-model /
  inference-profile ARNs) into the Anthropic-side model name — used for the response
  `model` field and cost aggregation
- the Messages-API alias `POST /anthropic/v1/messages` that the `bedrock-runtime` /
  `bedrock-mantle` hosts serve (used by `AnthropicBedrockMantle`): the plain Anthropic
  handler with Bedrock model-id normalization and the `[bedrock]` receipt log
- validating `anthropic_version` (missing / wrong value → 400 ValidationException)
- Bedrock-style error bodies + `x-amzn-ErrorType`, with an HTTP status → AWS exception
  name mapping when the injected type is not already an AWS exception name
- Bedrock response headers (token counts / invocation latency) and event stream encoding
- a one-line stderr log per received request so Bedrock traffic is distinguishable

Response injection (/_control/respond etc.) is fully shared with the Anthropic route.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import eventstream

# canonical core. WARNING: circular import. fake_server imports this module at its end
# and calls build_router(). Here we hold only a module reference; attributes like
# `fs.register_request` must **always be resolved at call-time (inside the route
# handler)** — touching fs.* at import time grabs a half-executed fake_server and
# results in AttributeError. This is not a problem on the normal path (startup via
# fake_server), but note that you hit this ordering constraint if you write, e.g.,
# tests that import `providers.bedrock` standalone.
from .. import fake_server as fs

_EVENTSTREAM_MEDIA = "application/vnd.amazon.eventstream"

# The only anthropic_version Bedrock's InvokeModel accepts for Anthropic models.
ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Exception members each streaming operation's union actually defines. InvokeModel's stream
# adds `modelTimeoutException`; ConverseStream does not have it.
_INVOKE_STREAM_MEMBERS = frozenset((
    "internalServerException", "modelStreamErrorException", "validationException",
    "throttlingException", "modelTimeoutException", "serviceUnavailableException"))
_CONVERSE_STREAM_MEMBERS = _INVOKE_STREAM_MEMBERS - {"modelTimeoutException"}


def stream_exception_fields(member: str, result: dict[str, Any]) -> dict[str, Any]:
    """Members that model more than `message` (ModelStreamErrorException carries the
    upstream's original status and message)."""
    if member == "modelStreamErrorException":
        return {"originalStatusCode": int(result.get("original_status") or result.get("status", 500)),
                "originalMessage": str(result.get("message", ""))}
    return {}


_SERVICE_TIERS = ("priority", "default", "flex", "reserved")
_LATENCIES = ("standard", "optimized")


def _invoke_options(request: Request) -> tuple[dict[str, str], str | None]:
    """The InvokeModel options boto3 sends as HEADERS (`serviceTier=` →
    `X-Amzn-Bedrock-Service-Tier`, `performanceConfigLatency=` →
    `X-Amzn-Bedrock-PerformanceConfig-Latency`). Returns (options, error)."""
    out: dict[str, str] = {}
    tier = request.headers.get("x-amzn-bedrock-service-tier")
    if tier is not None:
        if tier not in _SERVICE_TIERS:
            return out, f"X-Amzn-Bedrock-Service-Tier: must be one of {list(_SERVICE_TIERS)}"
        out["service_tier"] = tier
    latency = request.headers.get("x-amzn-bedrock-performanceconfig-latency")
    if latency is not None:
        if latency not in _LATENCIES:
            return out, f"X-Amzn-Bedrock-PerformanceConfig-Latency: must be one of {list(_LATENCIES)}"
        out["latency"] = latency
    return out, None


def _option_headers(snapshot: dict[str, Any]) -> dict[str, str]:
    """Echo the tier / latency the request asked for, as the real service does (the
    response models both headers; the tier is `default` when none was requested)."""
    opts = snapshot.get("bedrock_options") or {}
    out = {"X-Amzn-Bedrock-Service-Tier": opts.get("service_tier") or "default"}
    if opts.get("latency"):
        out["X-Amzn-Bedrock-PerformanceConfig-Latency"] = opts["latency"]
    return out


def stream_exception_member(name: str, *, operation: str = "invoke",
                            status: int | None = None) -> str:
    """Event-stream exception member for an AWS exception name (`ThrottlingException` →
    `throttlingException`), restricted to the members that operation's union defines.

    Names outside it are reported through the closest documented member — a mid-stream
    timeout on ConverseStream, for instance, becomes `modelStreamErrorException`.
    """
    members = _CONVERSE_STREAM_MEMBERS if operation == "converse" else _INVOKE_STREAM_MEMBERS
    camel = name[:1].lower() + name[1:] if name else "internalServerException"
    if camel in members:
        return camel
    by_name = {"modelErrorException": "modelStreamErrorException",
               "modelTimeoutException": "modelStreamErrorException",
               "accessDeniedException": "validationException",
               "resourceNotFoundException": "validationException"}
    if camel in by_name:
        return by_name[camel]
    # Anything else the union cannot carry (`ModelNotReadyException`,
    # `ServiceQuotaExceededException`, a 529 …) keeps its STATUS CLASS: a 429-class
    # injection must surface as a throttle, not as an internal error.
    if status is not None:
        if status == 429:
            return "throttlingException"
        if status == 503:
            return "serviceUnavailableException"
        if status in (408, 504, 424):
            return "modelStreamErrorException"
        if 400 <= status < 500:
            return "validationException"
    return "internalServerException"


# HTTP status → AWS exception name (used when /_control/error is given an Anthropic-style
# `type` such as `rate_limit_error`, or no type at all). Statuses not listed fall back to
# InternalServerException (5xx) / ValidationException (4xx). A `type` that already ends
# in "Exception" is passed through untouched, so callers can name any AWS exception.
STATUS_TO_EXCEPTION: dict[int, str] = {
    400: "ValidationException",
    401: "UnrecognizedClientException",   # AWS common error for bad/missing credentials
    403: "AccessDeniedException",
    404: "ResourceNotFoundException",
    408: "ModelTimeoutException",
    413: "RequestEntityTooLargeException",
    424: "ModelErrorException",
    429: "ThrottlingException",
    500: "InternalServerException",
    503: "ServiceUnavailableException",
    504: "ModelTimeoutException",
    529: "overloaded_error",              # Bedrock passes Anthropic's 529 through verbatim
}

# `[region.]anthropic.<name>[-vN[:M]]`
#   anthropic.claude-3-5-sonnet-20241022-v2:0     → claude-3-5-sonnet-20241022
#   us.anthropic.claude-haiku-4-5-20251001-v1:0   → claude-haiku-4-5-20251001
#   apac.anthropic.claude-sonnet-4-5-20250929-v1:0 → claude-sonnet-4-5-20250929
#   global.anthropic.claude-opus-5                 → claude-opus-5   (suffix-less current ids)
# The optional region prefix is a single dot-less segment (us / eu / apac / jp / au / us-gov /
# global / ...). ARNs (`arn:aws:bedrock:<region>:<account>:foundation-model/<id>` and
# `...:inference-profile/<id>`) are reduced to their trailing id first.
_MODEL_ID_RE = re.compile(
    r"^(?:(?P<region>[a-z][a-z0-9-]*)\.)?anthropic\.(?P<name>[A-Za-z0-9.-]+?)(?:-v\d+(?::\d+)?)?$"
)
_ARN_RE = re.compile(
    r"^arn:aws(?:-[a-z-]+)?:bedrock:[a-z0-9-]*:\d*:"   # partitions: aws, aws-cn, aws-us-gov, ...
    r"(?:foundation-model|inference-profile|application-inference-profile)/(?P<id>.+)$"
)


@dataclass(frozen=True)
class BedrockModel:
    raw: str                 # model id exactly as it appeared in the URL path (decoded)
    canonical: str           # Anthropic-side model name (== raw when the id is not recognized)
    region: str | None       # cross-region inference profile prefix (`us`, `eu`, ...) or None


def normalize_model_id(model_id: str) -> BedrockModel:
    """Map a Bedrock model id / cross-region inference profile id to the Anthropic model name.

    Unrecognized ids (no `anthropic.` segment) are returned unchanged so that anything
    else still flows through the proxy; pricing then falls back to substring matching.
    """
    arn = _ARN_RE.match(model_id)
    m = _MODEL_ID_RE.match(arn.group("id") if arn else model_id)
    if m is None:
        return BedrockModel(raw=model_id, canonical=model_id, region=None)
    return BedrockModel(raw=model_id, canonical=m.group("name"), region=m.group("region"))


def exception_name_for(status: int, etype: str | None) -> str:
    """Pick the AWS exception name for an injected error (see STATUS_TO_EXCEPTION)."""
    if etype and etype.endswith("Exception"):
        return etype
    if status in STATUS_TO_EXCEPTION:
        return STATUS_TO_EXCEPTION[status]
    return "InternalServerException" if status >= 500 else "ValidationException"


def _log(msg: str) -> None:
    print(f"[bedrock] {msg}", file=sys.stderr, flush=True)


def _bedrock_error_response(status: int, etype: str | None, message: str,
                            *, request_id: str | None = None,
                            extra_headers: dict[str, str] | None = None,
                            resource_name: str | None = None,
                            original_status: int | None = None) -> JSONResponse:
    """Bedrock-style error response (status + __type + x-amzn-ErrorType header)."""
    name = exception_name_for(status, etype)
    headers = {"x-amzn-ErrorType": name}
    if request_id is not None:
        headers["x-amzn-requestid"] = request_id
    if extra_headers:
        headers.update(extra_headers)
    body: dict[str, Any] = {"message": message, "__type": name}
    # The two 424s wrap an UPSTREAM failure and are modeled differently: ModelErrorException
    # carries {originalStatusCode, resourceName}, ModelStreamErrorException
    # {originalStatusCode, originalMessage}. `original_status` (from /_control/error) is
    # that upstream status; without it the injected status is the best available value.
    if name == "ModelErrorException":
        body["originalStatusCode"] = original_status or status
        if resource_name:
            body["resourceName"] = resource_name
    elif name == "ModelStreamErrorException":
        body["originalStatusCode"] = original_status or status
        body["originalMessage"] = message
    return JSONResponse(body, status_code=status, headers=headers)


def _validate_version(body: dict[str, Any]) -> str | None:
    """Return an error message when `anthropic_version` is missing or not the accepted value."""
    if "anthropic_version" not in body:
        return ("Malformed input request: #: required key [anthropic_version] not found, "
                "please reformat your input and try again.")
    v = body["anthropic_version"]
    if v != ANTHROPIC_VERSION:
        return (f"Malformed input request: anthropic_version must be "
                f"{ANTHROPIC_VERSION!r} (got {v!r}), please reformat your input and try again.")
    return None


def _token_headers(usage: dict[str, Any], latency_ms: int) -> dict[str, str]:
    """Bedrock's InvokeModel response headers. Like Converse's `inputTokens`, the input
    count excludes cached tokens, which are reported in their own cache-read / cache-write
    headers (total = input + cache read + cache write)."""
    return {
        "X-Amzn-Bedrock-Input-Token-Count": str(usage.get("input_tokens", 0)),
        "X-Amzn-Bedrock-Output-Token-Count": str(usage.get("output_tokens", 0)),
        "X-Amzn-Bedrock-Cache-Read-Input-Token-Count": str(usage.get("cache_read_input_tokens", 0)),
        "X-Amzn-Bedrock-Cache-Write-Input-Token-Count": str(usage.get("cache_creation_input_tokens", 0)),
        "X-Amzn-Bedrock-Invocation-Latency": str(latency_ms),
    }


def _invocation_metrics(usage: dict[str, Any], latency_ms: int, first_byte_ms: int) -> dict[str, Any]:
    return {
        "inputTokenCount": usage.get("input_tokens", 0),
        "outputTokenCount": usage.get("output_tokens", 0),
        "cacheReadInputTokenCount": usage.get("cache_read_input_tokens", 0),
        "cacheWriteInputTokenCount": usage.get("cache_creation_input_tokens", 0),
        "invocationLatency": latency_ms,
        "firstByteLatency": first_byte_ms,
    }


def _latency_ms(snapshot: dict[str, Any]) -> int:
    return max(0, int((time.time() - snapshot.get("received_at", time.time())) * 1000))


async def _receive(model_id: str, request: Request, *, is_stream: bool, req_id: str
                   ) -> tuple[BedrockModel, dict[str, Any], asyncio.Future] | JSONResponse:
    """Shared front half of both routes: parse, validate, register as pending, log."""
    op = "invoke-with-response-stream" if is_stream else "invoke"
    model = normalize_model_id(model_id)
    body, errmsg = await fs._parse_json_body(request)
    # `anthropic_version` is an Anthropic-model contract; other vendors' models (Llama,
    # Titan, ...) have their own body shapes and pass through unchecked.
    if errmsg is None and model.canonical != model.raw:
        errmsg = _validate_version(body)
    options: dict[str, str] = {}
    if errmsg is None:
        options, errmsg = _invoke_options(request)
    if errmsg is not None:
        _log(f"{op} model={model.raw} rejected: {errmsg}")
        return _bedrock_error_response(400, "ValidationException", errmsg, request_id=req_id)
    if options.get("service_tier") and isinstance(body, dict):
        # Make the requested tier visible to the responder in the canonical vocabulary
        # (`priority | flex` → auto, `default | reserved` → standard_only), like Converse.
        body.setdefault("service_tier",
                        "auto" if options["service_tier"] in ("priority", "flex") else "standard_only")

    try:
        snapshot, fut = await fs.register_request(
            "bedrock", model.canonical, body, is_stream=is_stream,
            extra={"bedrock_model_id": model.raw, "bedrock_options": options},
        )
    except fs.RequestValidationError as e:
        _log(f"{op} model={model.raw} rejected: {e}")
        return _bedrock_error_response(400, "ValidationException", str(e), request_id=req_id)
    mapped = f" -> {model.canonical}" if model.canonical != model.raw else ""
    _log(f"{op} model={model.raw}{mapped} pending={snapshot['pending_id']} "
         f"turn={snapshot['turn']} request_id={req_id}")
    return model, snapshot, fut


def build_router() -> APIRouter:
    router = APIRouter()

    # NOTE: `{model_id:path}` may span several segments: a foundation model id
    # (`anthropic.claude-3-5-sonnet-20241022-v2:0`, arriving percent-encoded as `%3A` and
    # decoded on the ASGI side), a cross-region inference profile (`us.anthropic.claude-...`),
    # or an ARN (`arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic...`,
    # whose `%2F` is decoded to `/` on the ASGI side) all route here.

    @router.post("/model/{model_id:path}/invoke")
    async def invoke(model_id: str, request: Request) -> Any:
        req_id = str(uuid.uuid4())
        received = await _receive(model_id, request, is_stream=False, req_id=req_id)
        if isinstance(received, JSONResponse):
            return received
        model, snapshot, fut = received
        result = await fs.await_resolution(snapshot, fut)

        if result["kind"] == "cleared":
            return _bedrock_error_response(503, "ServiceUnavailableException",
                                           f"request cleared: {result['detail']}",
                                           request_id=req_id)
        if result["kind"] == "error":
            return _bedrock_error_response(
                result["status"], result["type"], result["message"], request_id=req_id,
                resource_name=model.raw, original_status=result.get("original_status"),
                extra_headers=fs._error_headers(result),
            )

        # Bedrock invoke (non-stream) returns the model's raw response body (= Anthropic
        # message JSON, whose `model` is the Anthropic-side name). Like real Bedrock,
        # attach requestid, token-count and latency headers (approximate values).
        usage = result["usage"]
        return JSONResponse(
            fs._build_non_stream_response(
                result["message_id"], model.canonical, result["content_blocks"], usage,
                result.get("stop_reason"), result.get("stop_sequence"),
                result.get("stop_details"), snapshot.get("params"),
            ),
            headers={"x-amzn-requestid": req_id,
                     **_option_headers(snapshot),
                     **_token_headers(usage, _latency_ms(snapshot))},
        )

    @router.post("/model/{model_id:path}/invoke-with-response-stream")
    async def invoke_stream(model_id: str, request: Request) -> Any:
        req_id = str(uuid.uuid4())
        received = await _receive(model_id, request, is_stream=True, req_id=req_id)
        if isinstance(received, JSONResponse):
            return received
        model, snapshot, fut = received
        result = await fs.await_resolution(snapshot, fut)

        if result["kind"] == "cleared":
            return _bedrock_error_response(503, "ServiceUnavailableException",
                                           f"request cleared: {result['detail']}",
                                           request_id=req_id)
        if result["kind"] == "error":
            if result.get("after_events") is not None:
                # Mid-stream failure: a 200 event stream carrying some `chunk` frames and
                # then an exception frame (boto3 raises EventStreamError; the anthropic
                # SDK surfaces it while iterating).
                partial = fs.partial_stream_events(result, model.canonical, snapshot)
                frames = [eventstream.encode_chunk(data) for _n, data in partial]
                member = stream_exception_member(
                    exception_name_for(result["status"], result["type"]),
                    status=result["status"])
                frames.append(eventstream.encode_exception(
                    member, result["message"], stream_exception_fields(member, result)))

                async def gen_err():
                    for frame in frames:
                        yield frame
                        await asyncio.sleep(0)

                return StreamingResponse(gen_err(), media_type=_EVENTSTREAM_MEDIA,
                                         headers={"x-amzn-requestid": req_id,
                                                  **_option_headers(snapshot),
                                                  "X-Amzn-Bedrock-Content-Type": "application/json"})
            # Errors before streaming starts are returned via HTTP status (the SDK maps exceptions by status).
            return _bedrock_error_response(
                result["status"], result["type"], result["message"], request_id=req_id,
                resource_name=model.raw, original_status=result.get("original_status"),
                extra_headers=fs._error_headers(result),
            )

        events = fs.stream_event_dicts(
            result["message_id"], model.canonical, result["content_blocks"], result["usage"],
            result.get("stop_reason"), result.get("stop_sequence"), result.get("stop_details"),
            snapshot.get("params"),
        )
        # Real Bedrock bundles invocationMetrics into the final chunk (message_stop).
        # firstByteLatency = time to the first frame (the resolution arriving); the
        # invocation latency is stamped when the last frame is built.
        usage = result["usage"]
        first_byte_ms = _latency_ms(snapshot)
        for _name, data in events:
            if data.get("type") == "message_stop":
                data["amazon-bedrock-invocationMetrics"] = _invocation_metrics(
                    usage, _latency_ms(snapshot), first_byte_ms)
        frames = [eventstream.encode_chunk(data) for _name, data in events]

        async def gen():
            for frame in frames:
                yield frame
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type=_EVENTSTREAM_MEDIA,
                                 headers={"x-amzn-requestid": req_id,
                                          **_option_headers(snapshot),
                                          "X-Amzn-Bedrock-Content-Type": "application/json"})

    @router.post("/anthropic/v1/messages")
    async def messages_alias(request: Request) -> Any:
        """Bedrock's Messages-API surface (`https://bedrock-runtime.<region>.amazonaws.com/anthropic`
        and `https://bedrock-mantle.<region>.api.aws/anthropic`), used by `AnthropicBedrockMantle`.
        """
        return await handle_messages(request)

    return router


def is_bedrock_model_id(model: Any) -> bool:
    """Whether a Messages-API body `model` is a Bedrock id (`[region.]anthropic.…` / ARN)."""
    return isinstance(model, str) and normalize_model_id(model).canonical != model


async def handle_messages(request: Request) -> Any:
    """Messages-API request carrying a Bedrock model id — the `AnthropicBedrockMantle`
    wire format: same as first-party `/v1/messages` (SSE streaming, Anthropic error envelope,
    `anthropic-version` header, no `anthropic_version` body field), except that the model id
    carries the `anthropic.` prefix and is normalized like the InvokeModel route.

    Reached via the `/anthropic/v1/messages` alias (the real host path) and via the plain
    `/v1/messages` route when the body model looks like a Bedrock id (a Mantle client pointed
    straight at the fake's root posts there).
    """
    req_id = fs._new_request_id()
    headers = {"request-id": req_id, "x-amzn-requestid": str(uuid.uuid4())}
    # Peek at the body for the model id; fs.handle_messages re-parses it (cached by Starlette).
    try:
        raw = await request.json()
    except Exception:
        raw = None
    raw_model = raw.get("model") if isinstance(raw, dict) else None
    model = normalize_model_id(str(raw_model)) if isinstance(raw_model, str) else None

    def _on_registered(snapshot: dict[str, Any]) -> None:
        label = model.raw if model else repr(raw_model)
        mapped = (f" -> {model.canonical}" if model and model.canonical != model.raw else "")
        _log(f"messages model={label}{mapped} "
             f"pending={snapshot['pending_id']} turn={snapshot['turn']} request_id={req_id}")

    return await fs.handle_messages(
        request, provider="bedrock", headers=headers,
        model_override=model.canonical if model else None,
        extra={"bedrock_model_id": model.raw} if model else None,
        on_registered=_on_registered,
    )
