"""What this server implements, as data: the compatibility matrix behind
`GET /_control/capabilities` and the README table, and the catch-all that answers every
path the server does not implement in the caller's own error envelope.

A "surface" is one API operation (or a family of them) the server accepts. For each one:
  accepted    the request is parsed and becomes a pending (or is answered directly)
  injectable  a responder / rule answers it through /_control/respond|error
  streamed    the streaming form is served (SSE or AWS event stream)
  relayed     puppetllm.relay forwards it to a real upstream API
The `not_implemented` list names operations of the same APIs that are refused on purpose.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# Content block types a responder may inject and how each route carries them
# (`bedrock_invoke` covers InvokeModel and the Mantle alias, which share the wire).
CONTENT_BLOCKS: dict[str, dict[str, Any]] = {
    "text": {"anthropic": "text", "bedrock_invoke": "text", "converse": "text",
             "openai": "content"},
    "tool_use": {"anthropic": "tool_use", "bedrock_invoke": "tool_use", "converse": "toolUse",
                 "openai": "tool_calls"},
    "thinking": {"anthropic": "thinking", "bedrock_invoke": "thinking",
                 "converse": "reasoningContent", "openai": "dropped"},
    "redacted_thinking": {"anthropic": "redacted_thinking", "bedrock_invoke": "redacted_thinking",
                          "converse": "reasoningContent.redactedContent", "openai": "dropped"},
    "server_tool_use": {"anthropic": "server_tool_use (input_json_delta on a stream)",
                        "bedrock_invoke": "server_tool_use", "converse": "dropped",
                        "openai": "dropped"},
    "mcp_tool_use": {"anthropic": "mcp_tool_use", "bedrock_invoke": "mcp_tool_use",
                     "converse": "dropped", "openai": "dropped"},
    "web_search_tool_result": {"anthropic": "one content_block_start with the whole block",
                               "bedrock_invoke": "same", "converse": "dropped", "openai": "dropped"},
    "web_fetch_tool_result": {"anthropic": "same", "bedrock_invoke": "same", "converse": "dropped",
                              "openai": "dropped"},
    "code_execution_tool_result": {"anthropic": "same", "bedrock_invoke": "same",
                                   "converse": "dropped", "openai": "dropped"},
    "bash_code_execution_tool_result": {"anthropic": "same", "bedrock_invoke": "same",
                                        "converse": "dropped", "openai": "dropped"},
    "text_editor_code_execution_tool_result": {"anthropic": "same", "bedrock_invoke": "same",
                                               "converse": "dropped", "openai": "dropped"},
    "tool_search_tool_result": {"anthropic": "same", "bedrock_invoke": "same",
                                "converse": "dropped", "openai": "dropped"},
    "mcp_tool_result": {"anthropic": "same", "bedrock_invoke": "same", "converse": "dropped",
                        "openai": "dropped"},
}

SURFACES: list[dict[str, Any]] = [
    {"provider": "anthropic", "operation": "Messages", "method": "POST", "path": "/v1/messages",
     "accepted": True, "injectable": True, "streamed": True, "relayed": True,
     "notes": "Bedrock-form model ids are handed to the Bedrock adapter"},
    {"provider": "anthropic", "operation": "Messages count_tokens", "method": "POST",
     "path": "/v1/messages/count_tokens", "accepted": True, "injectable": False,
     "streamed": False, "relayed": False, "notes": "answered directly, nothing recorded"},
    {"provider": "anthropic", "operation": "Message Batches", "method": "POST/GET/DELETE",
     "path": "/v1/messages/batches*", "accepted": True, "injectable": True, "streamed": False,
     "relayed": True, "notes": "each custom_id is an ordinary pending; lifecycle via /_control/batch/*"},
    {"provider": "anthropic", "operation": "Models", "method": "GET",
     "path": "/v1/models, /v1/models/{id}", "accepted": True, "injectable": False,
     "streamed": False, "relayed": False, "notes": "static catalogue; unlisted ids synthesized"},
    {"provider": "bedrock", "operation": "InvokeModel", "method": "POST",
     "path": "/model/{id}/invoke", "accepted": True, "injectable": True, "streamed": False,
     "relayed": True, "notes": "Anthropic-shaped bodies; any model id is accepted, but the "
                               "answer is always in the Anthropic Messages shape"},
    {"provider": "bedrock", "operation": "InvokeModelWithResponseStream", "method": "POST",
     "path": "/model/{id}/invoke-with-response-stream", "accepted": True, "injectable": True,
     "streamed": True, "relayed": True, "notes": "AWS event stream with invocation metrics"},
    {"provider": "bedrock", "operation": "Converse / ConverseStream", "method": "POST",
     "path": "/model/{id}/converse, /model/{id}/converse-stream", "accepted": True,
     "injectable": True, "streamed": True, "relayed": True,
     "notes": "thinking / tool blocks translated; server-tool blocks dropped and not billed"},
    {"provider": "bedrock", "operation": "CountTokens", "method": "POST",
     "path": "/model/{id}/count-tokens", "accepted": True, "injectable": False,
     "streamed": False, "relayed": False, "notes": "invokeModel (base64 blob) or converse input"},
    {"provider": "bedrock", "operation": "Messages (Mantle)", "method": "POST",
     "path": "/anthropic/v1/messages", "accepted": True, "injectable": True, "streamed": True,
     "relayed": True, "notes": "same wire as /v1/messages (server-tool blocks included)"},
    {"provider": "bedrock", "operation": "Batch inference", "method": "POST/GET",
     "path": "/model-invocation-job*", "accepted": True, "injectable": True, "streamed": False,
     "relayed": True, "notes": "records become pendings; outputs written to the bundled S3"},
    {"provider": "s3", "operation": "Buckets and objects (subset)", "method": "PUT/GET/HEAD/DELETE/POST",
     "path": "/{bucket}, /{bucket}/{key}", "accepted": True, "injectable": False,
     "streamed": False, "relayed": False,
     "notes": "path-style, unauthenticated; unmodelled operations answer 501 in an S3 envelope"},
    {"provider": "openai", "operation": "Chat Completions", "method": "POST",
     "path": "/v1/chat/completions (alias /chat/completions)", "accepted": True,
     "injectable": True, "streamed": True, "relayed": True,
     "notes": "text and tool_calls; thinking and server-tool blocks dropped and not billed; "
              "a mid-stream error injection answers a plain HTTP error"},
    {"provider": "openai", "operation": "Models", "method": "GET",
     "path": "/v1/models, /v1/models/{id} (alias /models)", "accepted": True,
     "injectable": False, "streamed": False, "relayed": False,
     "notes": "OpenAI shape when the request carries a bare Bearer token"},
]

# Operations of the same APIs that are refused on purpose (the catch-all names them).
NOT_IMPLEMENTED: list[dict[str, str]] = [
    {"provider": "anthropic", "path": "/v1/complete", "operation": "Text Completions (legacy)"},
    {"provider": "anthropic", "path": "/v1/files*", "operation": "Files API"},
    {"provider": "anthropic", "path": "/v1/skills*", "operation": "Skills API"},
    {"provider": "anthropic", "path": "/v1/agents*, /v1/sessions*", "operation": "Managed Agents"},
    {"provider": "anthropic", "path": "/v1/organizations*", "operation": "Admin API"},
    {"provider": "openai", "path": "/v1/responses*", "operation": "Responses API"},
    {"provider": "openai", "path": "/v1/embeddings", "operation": "Embeddings"},
    {"provider": "openai", "path": "/v1/completions", "operation": "Completions (legacy)"},
    {"provider": "openai", "path": "/v1/assistants*, /v1/threads*, /v1/vector_stores*, "
                                    "/v1/fine_tuning*, /v1/audio*, /v1/images*, /v1/realtime*",
     "operation": "Assistants, vector stores, fine-tuning, audio, images, Realtime"},
    {"provider": "bedrock", "path": "/guardrail*", "operation": "ApplyGuardrail"},
    {"provider": "bedrock", "path": "/async-invoke*", "operation": "StartAsyncInvoke and friends"},
    {"provider": "bedrock", "path": "/model/{id}/invoke-with-bidirectional-stream",
     "operation": "InvokeModelWithBidirectionalStream"},
    {"provider": "bedrock", "path": "/model/{id}/*", "operation": "other model operations"},
]


def matrix(version: str) -> dict[str, Any]:
    return {"version": version, "surfaces": SURFACES, "content_blocks": CONTENT_BLOCKS,
            "not_implemented": NOT_IMPLEMENTED}


# ── the catch-all for unimplemented paths ────────────────────────────


def anthropic_dialect(request: Request) -> str:
    """`anthropic` unless the request looks like an OpenAI client (a bare Bearer token
    with neither `x-api-key` nor `anthropic-version`)."""
    h = request.headers
    if "x-api-key" in h or "anthropic-version" in h:
        return "anthropic"
    if h.get("authorization", "").lower().startswith("bearer "):
        return "openai"
    return "anthropic"


def _allowed_methods(app: Any, path: str) -> set[str]:
    """Methods the app's real routes accept for `path` (empty when none matches it)."""
    from .providers.s3 import _api_routes
    allowed: set[str] = set()
    for regex, methods in _api_routes(app):
        if regex.match(path):
            allowed |= set(methods)
    return allowed


def known_operation(path: str) -> str | None:
    """The name of the deliberately unimplemented operation a path belongs to, if any
    (`{id}` in a listed path stands for one segment; entries are tried in order, so the
    generic `/model/{id}/*` comes last)."""
    import fnmatch
    import re
    for entry in NOT_IMPLEMENTED:
        for pattern in entry["path"].split(", "):
            pattern = re.sub(r"\{[^/]*\}", "*", pattern.strip())
            if pattern.endswith("*") and not pattern.endswith("/*"):
                # `/v1/files*` means the resource and everything under it, not `/v1/filesystem`.
                base = pattern[:-1]
                if path == base or fnmatch.fnmatchcase(path, base + "/*"):
                    return entry["operation"]
            elif fnmatch.fnmatchcase(path, pattern):
                return entry["operation"]
    return None


async def refuse(request: Request, provider: str) -> JSONResponse:
    """The answer for a path nothing implements: 405 + Allow when the path exists with
    other methods, else a 404 in `provider`'s envelope naming the method, the path and —
    when it is one of the operations refused on purpose — that operation."""
    from . import fake_server as fs
    from .providers import bedrock as _bedrock
    from .providers import openai as _openai

    # Drain the body first: botocore sends `Expect: 100-continue` and keeps the
    # connection without sending the body on a non-100 answer, which would poison the
    # next request on that socket (the S3 emulation does the same).
    await request.body()
    path = request.url.path
    allowed = _allowed_methods(request.app, path)
    if allowed and request.method not in allowed:
        return JSONResponse({"detail": "Method Not Allowed"}, status_code=405,
                            headers={"Allow": ", ".join(sorted(allowed))})
    what = f"{request.method} {path}"
    op = known_operation(path)
    message = (f"puppetllm does not implement {what}" + (f" ({op})" if op else "")
               + "; see GET /_control/capabilities for the supported operations")
    if provider == "bedrock":
        return _bedrock._bedrock_error_response(404, "UnknownOperationException", message)
    if provider == "openai":
        return _openai._openai_error_response(404, "invalid_request_error",
                                              f"Unknown request URL: {what}. {message}",
                                              code="unknown_url")
    return fs._anthropic_error(404, "not_found_error", message,
                               headers={"request-id": fs._new_request_id()})


def build_router() -> APIRouter:
    """Routes for every API prefix this server owns, answering what nothing else handled
    through `refuse` — never an S3 error for an LLM API path."""
    router = APIRouter()
    methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    _refuse = refuse

    @router.api_route("/v1", methods=methods, include_in_schema=False, name=FALLBACK_ROUTE_NAME)
    @router.api_route("/v1/{rest:path}", methods=methods, include_in_schema=False,
                      name=FALLBACK_ROUTE_NAME)
    async def v1_fallback(request: Request, rest: str = "") -> Any:
        return await _refuse(request, anthropic_dialect(request))

    @router.api_route("/anthropic", methods=methods, include_in_schema=False,
                      name=FALLBACK_ROUTE_NAME)
    @router.api_route("/anthropic/{rest:path}", methods=methods, include_in_schema=False,
                      name=FALLBACK_ROUTE_NAME)
    async def anthropic_fallback(request: Request, rest: str = "") -> Any:
        return await _refuse(request, "anthropic")

    for prefix in ("model", "guardrail", "async-invoke", "model-invocation-job",
                   "model-invocation-jobs"):
        @router.api_route(f"/{prefix}", methods=methods, include_in_schema=False,
                          name=FALLBACK_ROUTE_NAME)
        @router.api_route(f"/{prefix}/{{rest:path}}", methods=methods, include_in_schema=False,
                          name=FALLBACK_ROUTE_NAME)
        async def bedrock_fallback(request: Request, rest: str = "") -> Any:
            return await _refuse(request, "bedrock")

    @router.api_route("/chat", methods=methods, include_in_schema=False, name=FALLBACK_ROUTE_NAME)
    @router.api_route("/chat/{rest:path}", methods=methods, include_in_schema=False,
                      name=FALLBACK_ROUTE_NAME)
    async def openai_root_fallback(request: Request, rest: str = "") -> Any:
        return await _refuse(request, "openai")

    return router


# Routes registered by `build_router`, excluded from the "real routes" registry that the
# 405 logic consults (they would otherwise allow every method on every path).
FALLBACK_ROUTE_NAME = "puppetllm_fallback"
