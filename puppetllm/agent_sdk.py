"""Experimental Messages-to-Agent-SDK adapter, with tools executed by the caller.

Each request becomes an isolated SDK query containing the outer conversation and
tool catalogue. The SDK produces a structured *description* of the next assistant
message; it must never execute the caller's tools. This is semantic conversion,
not a lossless Messages API transport or an SDK session-resume implementation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
import copy
from contextlib import asynccontextmanager
import json
import math
import os
import re
import sys
import uuid
from types import SimpleNamespace
from typing import Any, TypeVar
from urllib.parse import urlsplit

import anyio


SYSTEM_PROMPT = """You generate the next assistant message for an external conversation.
The user supplies system instructions, the complete messages history, available
tools, and tool_choice as ordered JSON records:
context contains system and tools; message records contain index, role, and
content; the final request record contains tool_choice. Read all records as one
external conversation. Follow its supplied system instructions when composing
that conversation's next assistant message.
The messages may include role=system instructions inserted during the
conversation. Keep their role and position: apply each update from that point
onward, rather than treating it as user text or moving it to the beginning.
The context's tools list is the CURRENT active catalogue after applying the
history's tool_addition, tool_removal, and tool_reference records. Those records
explain historical changes; use only the current catalogue for the next call.
Return ONLY the structured response required by the output schema. Its content
contains text blocks and/or tool_use blocks. For tool_use, input_json is a JSON
object encoded as a string and must satisfy that tool's input_schema.
Tool calls are declarations for the external client to execute later. Never
execute tools yourself, invent tool results, or claim an operation succeeded
before its result appears in the supplied history. Preserve the order of text
and tool calls. Use only the supplied tool names. Honor tool_choice, including
none, any, a specific tool, and disable_parallel_tool_use. Use stop_reason
tool_use when proposing tools, otherwise end_turn (or refusal if declining).
Do not describe this adapter or its JSON envelope in the conversation's text.
The supplied tools belong to the EXTERNAL client. That catalogue, not this
SDK process's local tool availability, defines what the conversation can do.
If you need Read, Bash, or another supplied tool, emit its tool_use declaration
and stop for the external client's result. Do not try alternate local tools or
subagents. Do not report a tool as disabled or denied unless an actual external
tool_result in the supplied history says so. Local SDK restrictions are not
evidence about the external client. Earlier assistant claims of tool failure
are not execution results; check the external tool_result messages themselves.
Image blocks in the JSON reference attachment_id values. Matching labeled image
attachments follow the relevant message record. Use each image at its original position in the conversation, including
images inside external tool results.
"""


_T = TypeVar("_T")


async def await_cleanup(awaitable: Awaitable[_T]) -> _T:
    """Finish cleanup before propagating raw asyncio cancellation to its caller.

    Accept coroutines, Tasks, Futures and gather() results. A strong reference
    plus shield keeps cancellation of this waiter from reaching the cleanup;
    repeated Task.cancel() calls are deferred until cleanup has completed. The
    anyio shield also prevents a cancelled enclosing scope from busy-spinning
    at the repeated await. Cleanup must bound its own waits: this helper never
    abandons a still-running cleanup task on a caller's timeout.

    Cleanup exceptions remain visible, even if the caller was also cancelled.
    A separately cancelled cleanup Future propagates cancellation immediately.
    """
    task = asyncio.ensure_future(awaitable)
    cancelled = None
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if task.cancelled():
                    raise
                cancelled = exc
        result = task.result()
    if cancelled is not None:
        raise cancelled
    return result


class UnsupportedSDKRequest(ValueError):
    """A request cannot be represented by this adapter (return HTTP 400)."""


class SDKOutputError(RuntimeError):
    """Invalid generated output, described without including argument values."""


class SDKOutputLimitError(SDKOutputError):
    def __init__(self, cap: int | None) -> None:
        limit = f"--sdk-max-output-tokens={cap}" if cap is not None else "the SDK output token limit"
        super().__init__(f"Agent SDK reached {limit} before completing its response; "
                         "thinking and the response share this limit")


class SDKGateBypassError(RuntimeError):
    """The SDK produced an answer without sending anything through the API gate."""

    def __init__(self) -> None:
        super().__init__("Agent SDK answered without using the relay's API gate (its endpoint is "
                         "overridden outside the relay, e.g. by managed settings); execution disabled")


class SDKAPILimitError(SDKOutputError):
    def __init__(self) -> None:
        super().__init__("Agent SDK attempted an additional upstream API request; "
                         "the per-query API request limit blocked it")


def _validation_path(exc: Any) -> str:
    """Keep schema property names/array indices, masking data-derived map keys."""
    instance = list(exc.absolute_path)
    schema = list(exc.absolute_schema_path)
    parts = []
    position = index = 0
    while index < len(schema) and position < len(instance):
        step = schema[index]
        if step == "properties" and index + 1 < len(schema):
            name = schema[index + 1]
            if instance[position] == name:
                parts.append(name)
                position += 1
            index += 2
        elif step in ("additionalProperties", "patternProperties"):
            parts.append("<key>")
            position += 1
            index += 2 if step == "patternProperties" else 1
        elif step in ("items", "prefixItems", "additionalItems", "contains"):
            if isinstance(instance[position], int):
                parts.append(instance[position])
                position += 1
            index += 1
            if index < len(schema) and isinstance(schema[index], int):
                index += 1
        else:
            index += 1
    parts.extend(part if isinstance(part, int) else "<key>" for part in instance[position:])
    return "$" + "".join(f"[{json.dumps(part, ensure_ascii=False)}]" for part in parts)


def _output_validation_error(exc: Any, scope: str) -> SDKOutputError:
    path = _validation_path(exc)
    if exc.validator == "type":
        detail = f"expected {exc.validator_value}"
    elif exc.validator == "required" and isinstance(exc.instance, dict):
        missing = [name for name in exc.validator_value if name not in exc.instance]
        detail = "missing required fields " + ", ".join(missing)
    else:
        detail = f"schema rule {exc.validator} failed"
    # ValidationError.__str__ includes the instance (file contents, commands,
    # credentials, etc.). Report the schema location/rule, never its value.
    return SDKOutputError(f"Agent SDK {scope} invalid at {path[:160]}: {detail[:160]}")


def _normalize_tool_input(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Decode typed JSON literals throughout arguments before strict validation.

    Native tool calls can stringify arrays/objects as well as scalar values.
    Only convert to an explicitly declared type, leaving strings and ambiguous
    unions alone. Never guess a singleton array or discard unknown arguments.
    """
    def normalize(value: Any, field: Any) -> Any:
        if not isinstance(field, dict):
            return value
        kind = field.get("type")
        if isinstance(value, str) and kind in ("boolean", "number", "integer", "array", "object"):
            try:
                decoded = json.loads(value, parse_constant=_reject_constant, parse_float=_finite_float)
            except ValueError:
                decoded = None
            if (kind == "boolean" and type(decoded) is bool
                    or kind in ("number", "integer") and (type(decoded) is int
                        or type(decoded) is float and math.isfinite(decoded))
                    or kind == "array" and isinstance(decoded, list)
                    or kind == "object" and isinstance(decoded, dict)):
                value = decoded
        if isinstance(value, dict) and kind in (None, "object"):
            properties = field.get("properties") or {}
            for name in field.get("required") or []:
                definition = properties.get(name)
                if name not in value and isinstance(definition, dict) and "default" in definition:
                    value[name] = copy.deepcopy(definition["default"])
            for name in value:
                definition = properties.get(name, field.get("additionalProperties"))
                value[name] = normalize(value[name], definition)
        elif isinstance(value, list) and kind in (None, "array"):
            items = field.get("items")
            # Both modern prefixItems and older tuple-style items are schemas
            # for individual positions; homogeneous items applies to the rest.
            prefix = field.get("prefixItems") or (items if isinstance(items, list) else [])
            for index in range(len(value)):
                definition = (prefix[index] if index < len(prefix) else
                              field.get("additionalItems") if isinstance(items, list) else items)
                value[index] = normalize(value[index], definition)
        return value

    return normalize(copy.deepcopy(args), schema)


class SDKRunError(RuntimeError):
    """Actionable SDK failure without exposing raw results or conversation text."""

    def __init__(self, source: Any, *, max_turns: int) -> None:
        subtype = getattr(source, "subtype", None)
        reason = getattr(source, "terminal_reason", None)
        status = getattr(source, "api_error_status", None)
        has_status = type(status) is int and 400 <= status <= 599
        self.status = status if has_status else 502
        self.error_type = {400: "invalid_request_error", 401: "authentication_error",
                           403: "permission_error", 404: "not_found_error", 413: "request_too_large",
                           429: "rate_limit_error", 529: "overloaded_error"}.get(self.status, "api_error")
        # Whether the same input may run again: only a rejection that cost nothing.
        self.repeatable = reason in ("prompt_too_long", "blocking_limit")
        if self.repeatable:
            # Answer in the Messages API's own wording so the client that owns the
            # history can shorten it and resend. Only the token counts are kept.
            self.status, self.error_type = 400, "invalid_request_error"
            result = getattr(source, "result", None)
            counts = re.search(r"~(\d+) tokens \(limit (\d+)\)", result) if isinstance(result, str) else None
            super().__init__("prompt is too long" + (
                f": {counts[1]} tokens > {counts[2]} maximum" if counts else ""))
            return
        if subtype == "error_max_turns" or reason == "max_turns":
            detail = (f"reached --sdk-max-turns={max_turns} before producing a response; "
                      "increase --sdk-max-turns if this repeats")
        elif subtype == "error_max_structured_output_retries":
            detail = "exhausted structured-output validation retries"
        elif has_status or reason == "api_error":
            detail = f"upstream API failed (HTTP {status})" if type(status) is int else "upstream API failed"
        elif reason in ("aborted_streaming", "aborted_tools"):
            detail = f"run was interrupted ({reason})"
        elif subtype == "error_during_execution":
            detail = "failed during execution"
        else:
            detail = "ended without a successful result"
        super().__init__(f"Agent SDK {detail}")


EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _custom_tool(tool: Any) -> dict[str, Any]:
    if (not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]
            or not isinstance(tool.get("input_schema"), dict)
            or tool.get("type") not in (None, "custom")):
        raise UnsupportedSDKRequest("agent-sdk supports custom tools with input_schema only")
    if "defer_loading" in tool and type(tool["defer_loading"]) is not bool:
        raise UnsupportedSDKRequest("agent-sdk tool defer_loading must be boolean")
    return {key: copy.deepcopy(tool[key]) for key in ("name", "description", "input_schema") if key in tool}


class _ToolCatalogue:
    """Resolve historical custom-tool changes without executing any tool."""

    def __init__(self, tools: list[Any]) -> None:
        self.definitions: dict[str, dict[str, Any]] = {}
        self.active: set[str] = set()
        for tool in tools:
            definition = _custom_tool(tool)
            if definition["name"] in self.definitions:
                raise UnsupportedSDKRequest("agent-sdk requires unique tool names")
            self._define(definition, deferred=tool.get("defer_loading", False))

    def _define(self, definition: dict[str, Any], *, deferred: bool) -> None:
        name = definition["name"]
        self.definitions[name] = definition
        if deferred:
            self.active.discard(name)
        else:
            self.active.add(name)

    def _reference(self, reference: Any, field: str) -> str:
        if (not isinstance(reference, dict) or reference.get("type") != "tool_reference"
                or not isinstance(reference.get(field), str)
                or reference[field] not in self.definitions):
            raise UnsupportedSDKRequest("agent-sdk tool reference must name a declared custom tool")
        return reference[field]

    def change(self, block: dict[str, Any]) -> dict[str, Any]:
        tool = block.get("tool")
        if (block["type"] == "tool_addition" and isinstance(tool, dict)
                and tool.get("type") == "tool_definition"):
            raw = tool.get("definition")
            definition = _custom_tool(raw)
            deferred = raw.get("defer_loading", False)
            self._define(definition, deferred=deferred)
            historical = copy.deepcopy(definition)
            if deferred:
                historical["defer_loading"] = True
            return {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": historical}}
        name = self._reference(tool, "name")
        if block["type"] == "tool_addition":
            self.active.add(name)
        else:
            self.active.discard(name)
        return {"type": block["type"], "tool": {"type": "tool_reference", "name": name}}

    def search_reference(self, block: dict[str, Any]) -> dict[str, Any]:
        name = self._reference(block, "tool_name")
        self.active.add(name)
        return {"type": "tool_reference", "tool_name": name}

    def current(self) -> list[dict[str, Any]]:
        return [tool for name, tool in self.definitions.items() if name in self.active]


def _content(value: Any, *, system: bool = False, images: list[dict[str, Any]] | None = None,
             omitted_thinking: set[str] | None = None, catalogue: _ToolCatalogue | None = None,
             tool_result: bool = False) -> Any:
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise UnsupportedSDKRequest("agent-sdk requires text or content-block lists")
    result = []
    for block in value:
        if not isinstance(block, dict):
            raise UnsupportedSDKRequest("agent-sdk received an invalid content block")
        kind = block.get("type")
        if not system and omitted_thinking is not None and kind in ("thinking", "redacted_thinking"):
            # Signed reasoning belongs to its original API conversation. This
            # adapter supplies history as JSON to a fresh SDK session, so omit
            # reasoning and signatures rather than replaying or exposing them
            # as ordinary assistant text. Preserve adjacent text/tools/results.
            omitted_thinking.add(kind)
        elif kind == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif system and catalogue is not None and kind in ("tool_addition", "tool_removal"):
            result.append(catalogue.change(block))
        elif tool_result and catalogue is not None and kind == "tool_reference":
            result.append(catalogue.search_reference(block))
        elif not system and kind == "tool_use":
            result.append({k: block[k] for k in ("type", "id", "name", "input") if k in block})
        elif not system and kind == "tool_result":
            result.append({"type": kind, "tool_use_id": block.get("tool_use_id"),
                           "content": _content(block.get("content", ""), images=images,
                                               catalogue=catalogue, tool_result=True),
                           "is_error": bool(block.get("is_error", False))})
        elif not system and kind == "image" and images is not None:
            source = block.get("source")
            if (not isinstance(source, dict) or source.get("type") != "base64"
                    or source.get("media_type") not in ("image/png", "image/jpeg", "image/gif", "image/webp")
                    or not isinstance(source.get("data"), str) or not source["data"]):
                raise UnsupportedSDKRequest("agent-sdk images require a base64 PNG, JPEG, GIF, or WebP source")
            images.append({"type": "image", "source": {
                key: source[key] for key in ("type", "media_type", "data")}})
            result.append({"type": "image", "attachment_id": f"image_{len(images)}"})
        else:
            raise UnsupportedSDKRequest(
                f"agent-sdk does not support {kind!r} content blocks; "
                "supported blocks are text, base64 image, tool_use, tool_result, "
                "assistant thinking, system tool changes, and tool-result tool_reference")
    return result


def _choice(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "auto"}
    if isinstance(value, str):
        value = {"type": "any" if value == "required" else value}
    if not isinstance(value, dict):
        raise UnsupportedSDKRequest("agent-sdk received an invalid tool_choice")
    value = copy.deepcopy(value)
    if value.get("type") == "function":
        value = {"type": "tool", "name": (value.get("function") or {}).get("name")}
    if value.get("type") not in ("auto", "none", "any", "tool"):
        raise UnsupportedSDKRequest("agent-sdk received an unsupported tool_choice")
    return value


def prepare_request(req: dict[str, Any], model: str, dropped: list[str]) -> dict[str, Any]:
    catalogue = _ToolCatalogue(req.get("tools") or [])
    params = req.get("params") or {}
    choice = _choice(params.get("tool_choice"))
    if params.get("parallel_tool_calls") is False:
        choice["disable_parallel_tool_use"] = True
    # These are NOT SDK generation parameters. Report their loss, rather than
    # claiming the outer token budget, signed thinking, cache, or betas survived.
    output_config = params.get("output_config")
    effort = output_config.get("effort") if isinstance(output_config, dict) else None
    if effort not in EFFORTS:
        effort = None
    dropped.extend(k for k, v in params.items()
                   if k not in ("tool_choice", "parallel_tool_calls", "metadata") and v is not None
                   and not (k == "output_config" and effort is not None))
    if effort is not None:
        dropped.extend(f"output_config.{k}" for k in output_config if k != "effort")
    if req.get("max_tokens") is not None:
        dropped.append("max_tokens (SDK response envelope has a separate budget)")
    messages = []
    images: list[dict[str, Any]] = []
    omitted_thinking: set[str] = set()
    for index, message in enumerate(req.get("messages") or []):
        if not isinstance(message, dict):
            raise UnsupportedSDKRequest(f"agent-sdk messages[{index}] must be a message object")
        role = message.get("role")
        if role not in ("user", "assistant", "system"):
            raise UnsupportedSDKRequest(
                f"agent-sdk messages[{index}] requires a user, assistant, or system role")
        # Recent Claude Code versions send system instructions inside messages
        # (e.g. Opus 5.5's per-turn updates). Preserve their chronological scope.
        messages.append({"role": role, "content": _content(
            message.get("content"), system=role == "system", images=images,
            omitted_thinking=omitted_thinking if role == "assistant" else None, catalogue=catalogue)})
        # Per-message generation settings, including output_config.effort, are
        # not SDK options. Account for them instead of silently discarding them.
        dropped.extend(f"messages[{index}].{key}" for key, value in message.items()
                       if key not in ("role", "content") and value is not None)
    # Aggregate by block type: a resumed history can contain hundreds of these.
    # Record the loss for provenance/strict mode without logging reasoning data.
    dropped.extend(f"{kind} content blocks (omitted from SDK history)"
                   for kind in sorted(omitted_thinking))
    tools = catalogue.current()
    if choice["type"] == "tool" and choice.get("name") not in catalogue.active:
        raise UnsupportedSDKRequest("tool_choice names a tool absent from the active catalogue")
    if choice["type"] == "any" and not tools:
        raise UnsupportedSDKRequest("tool_choice=any requires at least one active tool")
    transcript = {"system": _content(req.get("system"), system=True),
                  "messages": messages, "tools": tools, "tool_choice": choice}
    body = {"model": model, "transcript": transcript, "images": images}
    if effort is not None:
        body["effort"] = effort
    return body


def response_schema(names: list[str]) -> dict[str, Any]:
    blocks = [{"type": "object", "properties": {
        "type": {"const": "text"}, "text": {"type": "string"}},
        "required": ["type", "text"], "additionalProperties": False}]
    if names:
        blocks.append({"type": "object", "properties": {
            "type": {"const": "tool_use"}, "name": {"enum": names},
            "input_json": {"type": "string", "description": "A JSON object matching the tool input_schema"}},
            "required": ["type", "name", "input_json"], "additionalProperties": False})
    return {"type": "object", "properties": {
        "content": {"type": "array", "minItems": 1, "items": {"anyOf": blocks}},
        "stop_reason": {"enum": ["end_turn", "tool_use", "refusal"]}},
        "required": ["content", "stop_reason"], "additionalProperties": False}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


class _NativeMessageBuffer:
    """Collect SDK block events until the upstream API message has finished."""

    def __init__(self, sdk: Any) -> None:
        self.sdk = sdk
        self.reset()

    def reset(self, message_id: str | None = None) -> None:
        self.message_id = message_id
        self.model = ""
        self.content: list[Any] = []
        self.usage: dict[str, Any] = {}
        self.stop_reason: str | None = None
        self.failed = False

    @property
    def pending_tools(self) -> bool:
        return any(isinstance(block, self.sdk.ToolUseBlock) and block.name != "StructuredOutput"
                   for block in self.content)

    def feed(self, message: Any) -> Any:
        if getattr(message, "parent_tool_use_id", None) is not None:
            return None
        if isinstance(message, self.sdk.StreamEvent):
            event = message.event
            kind = event.get("type")
            if kind == "message_start":
                start = event.get("message") or {}
                self.reset(start.get("id"))
                self.model = start.get("model", "")
                self.usage.update(start.get("usage") or {})
            elif kind == "message_delta":
                self.stop_reason = (event.get("delta") or {}).get("stop_reason", self.stop_reason)
                self.usage.update(event.get("usage") or {})
            elif kind == "error":
                self.failed = True
                self.content.clear()
            elif kind == "message_stop":
                completed = None
                if self.content and not self.failed:
                    completed = self.sdk.AssistantMessage(content=list(self.content), model=self.model,
                        message_id=self.message_id, stop_reason=self.stop_reason, usage=dict(self.usage))
                self.reset()
                return completed
        elif isinstance(message, self.sdk.AssistantMessage):
            if message.message_id and self.message_id and message.message_id != self.message_id:
                self.reset(message.message_id)
            if message.error:
                self.failed = True
                self.content.clear()
            elif not self.failed:
                self.message_id = message.message_id or self.message_id
                self.model = message.model
                self.content.extend(message.content)
                self.usage.update(message.usage or {})
        return None


def input_content(body: dict[str, Any], *, cache_ttl: str = "5m") -> list[dict[str, Any]]:
    """Keep unchanged history at stable content-block boundaries for cache reuse.

    Still stateless: the outer client owns history, including edits and compact.
    One explicit breakpoint stores the history prefix before the final request.
    The SDK adds its own system/turn breakpoints; stay within the API's limit.
    """
    def text_record(record: dict[str, Any]) -> dict[str, Any]:
        return {"type": "text", "text": json.dumps(record, ensure_ascii=False, separators=(",", ":"))}

    def append_image(content: list[dict[str, Any]], index: int) -> None:
        content.extend([{"type": "text", "text": f"Attachment image_{index + 1}:"},
                        copy.deepcopy(body["images"][index])])

    transcript = body["transcript"]
    content = [text_record({"record_type": "context", "system": transcript["system"],
                            "tools": transcript["tools"]})]

    def image_indices(blocks: Any):
        if not isinstance(blocks, list):
            return
        for block in blocks:
            if block["type"] == "image":
                yield int(block["attachment_id"].removeprefix("image_")) - 1
            elif block["type"] == "tool_result":
                yield from image_indices(block["content"])

    for index, message in enumerate(transcript["messages"]):
        content.append(text_record({"record_type": "message", "index": index, **message}))
        for image_index in image_indices(message["content"]):
            append_image(content, image_index)
    # Must match the SDK's own breakpoints: the API refuses 1h after 5m.
    content[-1]["cache_control"] = {"type": "ephemeral", **({"ttl": "1h"} if cache_ttl == "1h" else {})}
    # Changing tool_choice must not change the history prefix being cached.
    content.append(text_record({"record_type": "request", "tool_choice": transcript["tool_choice"]}))
    return content


class AgentSDKBackend:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        # Do not inherit the frontend's ANTHROPIC_BASE_URL and recurse into the
        # pending queue. --target is always explicitly set for the SDK child.
        def endpoint(url: str) -> tuple[str | None, int | None]:
            parsed = urlsplit(url)
            host = parsed.hostname
            if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                host = "loopback"
            return host, parsed.port or (443 if parsed.scheme == "https" else 80)
        if endpoint(cfg.target) == endpoint(cfg.puppet):
            raise ValueError("agent-sdk --target must not point back to --puppet")
        try:
            import jsonschema
            from referencing import Registry
            self.jsonschema = jsonschema
            self.registry = Registry()  # no remote schema downloads
            if not cfg.dry_run:
                import claude_agent_sdk
                self.sdk = claude_agent_sdk
        except ImportError as exc:
            raise ValueError('agent-sdk requires: pip install "puppetllm[agent-sdk]"') from exc

    def _validator(self, schema: dict[str, Any]) -> Any:
        cls = self.jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
        return cls(schema, registry=self.registry)

    def prepare(self, req: dict[str, Any], model: str, dropped: list[str]) -> dict[str, Any]:
        body = prepare_request(req, model, dropped)
        if body.get("effort") not in (None, self.effort(body)):
            dropped.append(f"output_config.effort (capped to {self.effort(body)})")
        for tool in body["transcript"]["tools"]:
            try:
                self._validator(tool["input_schema"])
            except self.jsonschema.SchemaError as exc:
                raise UnsupportedSDKRequest(f"invalid input_schema for tool {tool['name']!r}") from exc
        return body

    def decode(self, data: Any, transcript: dict[str, Any]) -> dict[str, Any]:
        catalogue = {t["name"]: t for t in transcript["tools"]}
        try:
            self._validator(response_schema(list(catalogue))).validate(data)
        except self.jsonschema.ValidationError as exc:
            raise _output_validation_error(exc, "response") from None
        blocks = []
        calls = []
        for block in data["content"]:
            if block["type"] == "text":
                if block["text"]:  # the Messages API refuses empty text blocks in a replayed history
                    blocks.append(dict(block))
            else:
                scope = f"tool {block['name']} input"
                try:
                    args = json.loads(block["input_json"], parse_constant=_reject_constant, parse_float=_finite_float)
                except ValueError:
                    raise SDKOutputError(f"Agent SDK {scope} is not valid finite JSON") from None
                if not isinstance(args, dict):
                    raise SDKOutputError(f"Agent SDK {scope} must be a JSON object")
                schema = catalogue[block["name"]]["input_schema"]
                args = _normalize_tool_input(args, schema)
                try:
                    self._validator(schema).validate(args)
                except self.jsonschema.ValidationError as exc:
                    raise _output_validation_error(exc, scope) from None
                call = {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex,
                        "name": block["name"], "input": args}
                calls.append(call)
                blocks.append(call)
        choice = transcript["tool_choice"]
        kind = choice["type"]
        if (kind == "none" and calls or kind in ("any", "tool") and not calls
                or kind == "tool" and any(c["name"] != choice["name"] for c in calls)
                or choice.get("disable_parallel_tool_use") and len(calls) > 1):
            raise SDKOutputError("Agent SDK response violates tool_choice")
        if not calls and data["stop_reason"] == "tool_use" or not blocks and data["stop_reason"] != "refusal":
            raise SDKOutputError("Agent SDK response has no usable content")
        return {"content": blocks, "stop_reason": "tool_use" if calls else data["stop_reason"]}

    def _direct_tool_request(self, message: Any, transcript: dict[str, Any]) -> dict[str, Any] | None:
        """Hand off native tool calls before the inner CLI rejects disabled tools.

        The model sometimes calls Read/Bash directly instead of describing that
        call through StructuredOutput. They are disabled *locally*, but the outer
        client offered them. Only forward names and arguments valid for that
        client's catalogue; never execute or invent a tool result here.
        """
        calls = [b for b in message.content if isinstance(b, self.sdk.ToolUseBlock)]
        if not calls:
            return None
        if any(b.name == "StructuredOutput" for b in calls):
            if any(b.name != "StructuredOutput" for b in calls):
                raise SDKOutputError("Agent SDK mixed structured output and native tool calls")
            return None
        if message.stop_reason != "tool_use":
            raise SDKOutputError("Agent SDK native tool response did not finish with tool_use")
        names = {t["name"] for t in transcript["tools"]}
        if any(b.name not in names for b in calls):
            raise SDKOutputError("Agent SDK requested a tool absent from the external catalogue")
        blocks = []
        for block in message.content:
            if isinstance(block, self.sdk.TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, self.sdk.ToolUseBlock):
                try:
                    encoded = json.dumps(block.input, ensure_ascii=False, allow_nan=False)
                except ValueError:
                    raise SDKOutputError("Agent SDK native tool input is not valid finite JSON") from None
                blocks.append({"type": "tool_use", "name": block.name,
                               "input_json": encoded})
        return self.decode({"content": blocks, "stop_reason": "tool_use"}, transcript)

    @staticmethod
    def _with_usage(payload: dict[str, Any], usage: Any) -> dict[str, Any]:
        if isinstance(usage, dict):
            counts = {k: usage[k] for k in ("input_tokens", "output_tokens",
                      "cache_read_input_tokens", "cache_creation_input_tokens") if k in usage}
            if counts:
                payload["usage"] = counts
        return payload

    def effort(self, body: dict[str, Any] | None = None) -> str | None:
        """The request's own effort, never above --sdk-effort; None is the SDK default."""
        limit = self.cfg.sdk_effort
        asked = (body or {}).get("effort")
        if limit == "auto":
            return asked
        if asked is None:
            return limit
        return min(asked, limit, key=EFFORTS.index)

    def generation_settings(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        settings = {"adapter_system_prompt": SYSTEM_PROMPT, "target": self.cfg.target,
                "max_turns": self.cfg.sdk_max_turns,
                "api_retries": self.cfg.sdk_api_retries,
                "structured_retries": self.cfg.sdk_structured_retries,
                "nonstreaming_fallback": False, "retry_watchdog": False}
        effort = self.effort(body)
        output_cap = self.cfg.sdk_max_output_tokens
        # Only settings that change generation are part of the guard key.
        if effort is not None:
            settings["effort"] = effort
        if output_cap is not None:
            settings["max_output_tokens"] = output_cap
        limit = self.cfg.sdk_max_api_requests
        if limit != 1:
            settings["max_api_requests"] = limit
        return settings

    async def generate(self, body: dict[str, Any], *, progress: Any = None) -> dict[str, Any]:
        from .sdk_egress import SDKAPIGate

        # CLI error recovery can issue more HTTP requests even with all its
        # retry knobs disabled. Enforce the budget at the actual egress path.
        async with SDKAPIGate(self.cfg.target, max_requests=self.cfg.sdk_max_api_requests,
                             progress=progress) as gate:
            try:
                answer = await self._generate(body, progress=progress, api_target=gate.base_url)
            except Exception:
                if gate.stop_reason == "max_tokens":
                    raise SDKOutputLimitError(self.cfg.sdk_max_output_tokens) from None
                if gate.stop_reason == "refusal":
                    return self._with_usage({"content": [], "stop_reason": "refusal"}, gate.usage)
                if gate.blocked_requests:
                    # The CLI tried again after a response; that last response is the outcome.
                    if gate.status_codes and gate.status_codes[-1] >= 400:
                        raise SDKRunError(SimpleNamespace(api_error_status=gate.status_codes[-1]),
                                          max_turns=self.cfg.sdk_max_turns) from None
                    raise SDKAPILimitError() from None
                raise
            if gate.stop_reason == "max_tokens":
                raise SDKOutputLimitError(self.cfg.sdk_max_output_tokens)
            if not gate.used:
                raise SDKGateBypassError()
            return answer

    async def _generate(self, body: dict[str, Any], *, progress: Any = None,
                        api_target: str) -> dict[str, Any]:
        from .sdk_transport import RelaySDKTransport

        @asynccontextmanager
        async def managed_client(client):
            try:
                connected = await client.__aenter__()
            except BaseException:
                await await_cleanup(client.__aexit__(*sys.exc_info()))
                raise
            try:
                yield connected
            except BaseException:
                if not await await_cleanup(client.__aexit__(*sys.exc_info())):
                    raise
            else:
                await await_cleanup(client.__aexit__(None, None, None))

        env = {"ANTHROPIC_BASE_URL": api_target,
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
               "CLAUDE_CODE_USE_BEDROCK": "0", "CLAUDE_CODE_USE_VERTEX": "0",
               "CLAUDE_CODE_USE_FOUNDRY": "0", "CLAUDE_CODE_USE_ANTHROPIC_AWS": "0",
               "CLAUDE_CODE_USE_MANTLE": "0",
               "CLAUDE_CODE_MAX_RETRIES": str(self.cfg.sdk_api_retries),
               "CLAUDE_CODE_RETRY_WATCHDOG": "0",
               "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
               # The outer client and the upstream API decide what fits; the inner
               # CLI's own estimate would refuse long histories before sending them.
               "CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE": "1000000000",
               "MAX_STRUCTURED_OUTPUT_RETRIES": str(self.cfg.sdk_structured_retries)}
        output_cap = self.cfg.sdk_max_output_tokens
        if output_cap is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(output_cap)
        cache_ttl = self.cfg.sdk_cache_ttl
        if cache_ttl == "1h":
            env["ENABLE_PROMPT_CACHING_1H"] = "1"
        key = os.environ.get(self.cfg.api_key_env)
        if key:
            env["ANTHROPIC_API_KEY"] = key
            env["ANTHROPIC_AUTH_TOKEN"] = ""
        transcript = body["transcript"]
        result = None
        buffer = _NativeMessageBuffer(self.sdk)
        incomplete_native = False
        max_buffer = self.cfg.sdk_max_buffer_bytes
        # New session per outer API request: the outer CLI owns the authoritative
        # history (including /compact, edits, forks and concurrent requests).
        # For some models the CLI puts the working directory ahead of the history,
        # so a per-request path would defeat the prompt cache: keep one empty one.
        cwd = os.path.join(os.path.expanduser(self.cfg.sdk_state_dir), "sdk-cwd")
        os.makedirs(cwd, mode=0o700, exist_ok=True)
        options = self.sdk.ClaudeAgentOptions(
            model=body["model"], system_prompt=SYSTEM_PROMPT, tools=[],
            effort=self.effort(body),
            setting_sources=[], mcp_servers={}, strict_mcp_config=True,
            permission_mode="dontAsk", cwd=cwd, env=env,
            settings=json.dumps({"disableAllHooks": True}),
            verbatim_prompts=True,
            include_partial_messages=True, max_buffer_size=max_buffer,
            stderr=progress.stderr if progress is not None else None,
            max_turns=self.cfg.sdk_max_turns,
            extra_args={"no-session-persistence": None, "disable-slash-commands": None},
            output_format={"type": "json_schema", "schema": response_schema(
                [t["name"] for t in transcript["tools"]])})
        # A client context can close the subprocess immediately on a tool
        # handoff. Waiting for a final ResultMessage lets the inner runtime
        # produce "tool disabled" feedback that can corrupt the answer.
        try:
            transport = RelaySDKTransport(prompt="", options=options)
            # The SDK does not close a transport whose client failed while connecting.
            try:
                async with managed_client(self.sdk.ClaudeSDKClient(options=options, transport=transport)) as client:
                    if progress is not None:
                        progress.stage("connected")
                    content = input_content(body, cache_ttl=cache_ttl)

                    async def input_messages():
                        yield {"type": "user", "message": {"role": "user", "content": content}}

                    await client.query(input_messages())
                    if progress is not None:
                        progress.stage("waiting_sdk_event")
                    async for message in client.receive_response():
                        if progress is not None:
                            progress.message(message)
                        if (isinstance(message, self.sdk.StreamEvent)
                                and message.event.get("type") == "message_delta"
                                and (message.event.get("delta") or {}).get("stop_reason") == "max_tokens"):
                            raise SDKOutputLimitError(output_cap)
                        if isinstance(message, self.sdk.StreamEvent) and message.event.get("type") == "message_start":
                            incomplete_native = False  # a fresh upstream attempt
                        completed = buffer.feed(message)
                        if completed is not None:
                            if completed.stop_reason is None and any(
                                    isinstance(block, self.sdk.ToolUseBlock) and block.name != "StructuredOutput"
                                    for block in completed.content):
                                # The CLI can synthesize message_stop while recovering
                                # from an API error. Wait for its terminal error/status.
                                # After a terminal delta the CLI can synthesize an
                                # indistinguishable stop even on error; see docs/agent-sdk.md.
                                incomplete_native = True
                                continue
                            direct = self._direct_tool_request(completed, transcript)
                            if direct is not None:
                                return self._with_usage(direct, completed.usage)
                        if isinstance(message, self.sdk.ResultMessage):
                            result = message
            finally:
                await await_cleanup(transport.close())
        except self.sdk.ResultError as exc:
            raise SDKRunError(exc, max_turns=self.cfg.sdk_max_turns) from None
        except self.sdk.CLIJSONDecodeError as exc:
            detail = str(getattr(exc, "original_error", ""))
            if re.fullmatch(r"Buffer size \d+ exceeds limit \d+", detail):
                raise SDKOutputError(
                    f"Agent SDK response exceeded --sdk-max-buffer-bytes={max_buffer}; "
                    "increase the limit or request a smaller response") from None
            raise SDKOutputError("Agent SDK returned malformed CLI JSON") from None
        if result is None or result.is_error or result.subtype != "success":
            raise SDKRunError(result, max_turns=self.cfg.sdk_max_turns)
        if incomplete_native or buffer.pending_tools:
            raise SDKOutputError("Agent SDK native tool response ended before message_stop")
        # Never fall back to assistant text: it may be a status update or JSON
        # instructions, not the message intended for the outer conversation.
        payload = self.decode(result.structured_output, transcript)
        return self._with_usage(payload, result.usage)
