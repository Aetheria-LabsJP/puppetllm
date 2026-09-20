"""Bedrock Converse / ConverseStream compatible adapter.

`POST /model/{modelId}/converse` and `POST /model/{modelId}/converse-stream` accept the
Converse request schema (`messages[].content[]` unions, `system[]`, `inferenceConfig`,
`toolConfig`, `additionalModelRequestFields`, …), translate it to the canonical
(Anthropic Messages-style) form the responder sees, and encode the injected canonical
blocks back as a Converse response (`output.message`, `stopReason`, `usage`, `metrics`)
or a ConverseStream event stream (`messageStart` … `messageStop`, `metadata`).

Mapping (both directions):
  {text}                                  <-> {"type": "text"}
  {image: {format, source: {bytes | s3Location}}} <-> {"type": "image", "source": {"type": "base64" | "url", ...}}
  {document: {name, format, source}}      <-> {"type": "document", ...}
  {toolUse: {toolUseId, name, input}}     <-> {"type": "tool_use", "id", "name", "input"}
  {toolResult: {toolUseId, content, status}} <-> {"type": "tool_result", "tool_use_id", "content", "is_error"}
  {reasoningContent: {reasoningText: {text, signature}}} <-> {"type": "thinking", "thinking", "signature"}
  {reasoningContent: {redactedContent}}   <-> {"type": "redacted_thinking", "data"}
  {cachePoint: {type: "default", ttl?}}   ->  `cache_control` on the PRECEDING block / tool
  inferenceConfig.{maxTokens, temperature, topP, stopSequences} -> max_tokens / temperature / top_p / stop_sequences
  toolConfig.tools[].toolSpec             ->  {name, description, input_schema, strict?}
  toolConfig.toolChoice {auto|any|tool}   ->  tool_choice {type: auto|any|tool}
  additionalModelRequestFields            ->  merged into the canonical body (thinking, output_config, top_k, anthropic_beta, …);
                                              keys Converse models itself (messages, system, tools, max_tokens, …) are a ValidationException
  outputConfig.effort / textFormat        ->  output_config.effort / output_config.format {type: json_schema, schema}

Response `stopReason` uses the Converse vocabulary (`refusal` -> `content_filtered`,
`pause_turn` -> `end_turn`); `usage.inputTokens` excludes cached tokens and
`cacheDetails` lists the 1h / 5m cache writes. Errors use the Bedrock JSON shape
(`message` + `__type`, `x-amzn-ErrorType`) with the same status → exception mapping as
the InvokeModel route.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
import re
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import bedrock as _bedrock
from . import eventstream
from .. import fake_server as fs
from ..openai_wire import strip_private

_EVENTSTREAM_MEDIA = "application/vnd.amazon.eventstream"
_TEXT_CHUNK = 80
_JSON_CHUNK = 40

_IMAGE_MEDIA = {"png": "image/png", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
_DOC_MEDIA = {"pdf": "application/pdf", "txt": "text/plain", "md": "text/markdown",
              "csv": "text/csv", "html": "text/html", "doc": "application/msword",
              "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
              "xls": "application/vnd.ms-excel",
              "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}

# canonical stop_reason -> Converse stopReason
_STOP_REASON_MAP = {
    "end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence", "refusal": "content_filtered", "pause_turn": "end_turn",
    "model_context_window_exceeded": "model_context_window_exceeded",
}
_CONVERSE_STOP_REASONS = ("end_turn", "tool_use", "max_tokens", "stop_sequence",
                          "guardrail_intervened", "content_filtered", "malformed_model_output",
                          "malformed_tool_use", "model_context_window_exceeded")


class ConverseValidationError(ValueError):
    pass


# ── request: Converse -> canonical ────────────────────────────────────


# ContentBlock / SystemContentBlock / union members the adapter knows about. AWS defines
# these as UNIONS: exactly one member per object, anything else is a ValidationException.
_CONTENT_MEMBERS = ("text", "image", "document", "video", "audio", "toolUse", "toolResult",
                    "guardContent", "cachePoint", "reasoningContent", "citationsContent",
                    "searchResult", "toolAddition", "toolRemoval")
_SYSTEM_MEMBERS = ("text", "guardContent", "cachePoint")
_IMAGE_FORMATS = ("png", "jpeg", "gif", "webp")
_DOC_FORMATS = ("pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md")
_VIDEO_FORMATS = ("mkv", "mov", "mp4", "webm", "flv", "mpeg", "mpg", "wmv", "three_gp")
_SERVICE_TIERS = ("priority", "default", "flex", "reserved")
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_TOOL_USE_ID_RE = re.compile(r"^[a-zA-Z0-9_.:-]{1,64}$")
_MAX_REQUEST_METADATA = 16
# ConverseRequest members (botocore model). `modelId` travels in the path, so one in the
# body is an unknown member like any other — it would otherwise be silently ignored.
_REQUEST_MEMBERS = frozenset((
    "messages", "system", "inferenceConfig", "toolConfig", "guardrailConfig",
    "additionalModelRequestFields", "promptVariables", "additionalModelResponseFieldPaths",
    "requestMetadata", "performanceConfig", "serviceTier", "outputConfig"))
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_REQUEST_METADATA_KEY_RE = re.compile(r"^[a-zA-Z0-9\s:_@$#=/+,\-.]{1,256}$")
_REQUEST_METADATA_VALUE_RE = re.compile(r"^[a-zA-Z0-9\s:_@$#=/+,\-.]{0,256}$")
# Request parameters that have a Converse field of their own. `additionalModelRequestFields`
# is for what Converse cannot express (`thinking`, `top_k`, `anthropic_beta`, …).
_CONVERSE_OWNED_FIELDS = frozenset((
    "messages", "system", "tools", "tool_choice", "max_tokens", "temperature", "top_p",
    "stop_sequences", "stream", "model", "anthropic_version", "service_tier", "metadata"))


def _one_of(obj: Any, members: tuple[str, ...], where: str) -> str:
    """Return the single union member present, or raise. AWS rejects zero and multiple."""
    if not isinstance(obj, dict):
        raise ConverseValidationError(f"{where}: must be an object")
    present = [m for m in members if m in obj]
    if len(present) != 1:
        raise ConverseValidationError(
            f"{where}: exactly one of {list(members)} must be set "
            f"(got {sorted(k for k in obj if k in members) or 'none'})")
    unknown = sorted(k for k in obj if k not in members)
    if unknown:
        raise ConverseValidationError(f"{where}: unknown member(s) {unknown}")
    return present[0]


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ConverseValidationError(message)


def _cache_control(cp: Any, where: str) -> dict[str, Any]:
    """Validate a cachePoint and convert it to the canonical `cache_control` marker."""
    _require(isinstance(cp, dict), f"{where}.cachePoint: must be an object")
    _require(cp.get("type") == "default", f"{where}.cachePoint.type: must be 'default'")
    ttl = cp.get("ttl", "5m")
    _require(ttl in ("5m", "1h"), f"{where}.cachePoint.ttl: must be '5m' or '1h'")
    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl == "1h":
        cc["ttl"] = "1h"
    return cc


class _CachePointSink:
    """Attaches `cachePoint` markers to the block they follow.

    A cache point caches everything up to and including the previous block — which may live
    in an earlier message (or in `system` / `tools`), since checkpoints are evaluated over
    the flattened tools → system → messages sequence. Two markers in a row would address the
    same block, so the second is rejected instead of silently replacing the first.
    """

    def __init__(self) -> None:
        self.last: dict[str, Any] | None = None

    def block(self, blk: dict[str, Any]) -> dict[str, Any]:
        self.last = blk
        return blk

    def mark(self, cp: Any, where: str) -> None:
        _require(self.last is not None, f"{where}: cachePoint must follow a content block")
        _require("cache_control" not in self.last,
                 f"{where}: two cachePoints cannot address the same block")
        self.last["cache_control"] = _cache_control(cp, where)


def _system_to_canonical(system: Any, sink: _CachePointSink) -> list[dict[str, Any]] | None:
    if system is None:
        return None
    _require(isinstance(system, list), "system: must be an array of system content blocks")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(system):
        member = _one_of(item, _SYSTEM_MEMBERS, f"system[{i}]")
        if member == "text":
            _require(isinstance(item["text"], str) and item["text"],
                     f"system[{i}].text: must be a non-empty string")
            out.append(sink.block({"type": "text", "text": item["text"]}))
        elif member == "guardContent":
            out.append(sink.block(_guard_content(item["guardContent"], f"system[{i}]")))
        else:
            sink.mark(item["cachePoint"], f"system[{i}]")
    return out


def _guard_content(gc: Any, where: str) -> dict[str, Any]:
    """`guardContent` is a union of `text` (a GuardrailConverseTextBlock) / `image`."""
    member = _one_of(gc, ("text", "image"), f"{where}.guardContent")
    if member == "image":
        img = gc["image"]
        _require(isinstance(img, dict) and isinstance(img.get("source"), dict),
                 f"{where}.guardContent.image.source: is required")
        fmt = img.get("format")
        # GuardrailConverseImageBlock: `png | jpeg` only, and a `bytes` source only.
        _require(fmt in ("png", "jpeg"),
                 f"{where}.guardContent.image.format: must be png or jpeg")
        return {"type": "image",
                "source": _source_to_canonical(img["source"], f"{where}.guardContent.image",
                                               _IMAGE_MEDIA[fmt], ("bytes",))}
    block = gc["text"]
    _require(isinstance(block, dict) and isinstance(block.get("text"), str),
             f"{where}.guardContent.text.text: a string is required")
    return {"type": "text", "text": block["text"]}


def _tool_result_content(content: Any, where: str) -> Any:
    _require(isinstance(content, list), f"{where}.content: an array is required")
    out: list[dict[str, Any]] = []
    for i, c in enumerate(content):
        member = _one_of(c, ("json", "text", "image", "document", "video", "searchResult"),
                         f"{where}.content[{i}]")
        if member == "text":
            _require(isinstance(c["text"], str), f"{where}.content[{i}].text: must be a string")
            out.append({"type": "text", "text": c["text"]})
        elif member == "json":
            out.append({"type": "text", "text": json.dumps(c["json"], ensure_ascii=False)})
        elif member == "image":
            out.append(_image_to_canonical(c["image"], f"{where}.content[{i}]"))
        elif member == "document":
            out.append(_document_to_canonical(c["document"], f"{where}.content[{i}]"))
        elif member == "video":
            out.append(_video_to_canonical(c["video"], f"{where}.content[{i}].video"))
        else:
            _require(isinstance(c["searchResult"], dict),
                     f"{where}.content[{i}].searchResult: must be an object")
            out.append(dict(c))  # kept verbatim for the responder
    return out


def _source_to_canonical(src: Any, where: str, media: str,
                        members: tuple[str, ...]) -> dict[str, Any]:
    """One `*Source` union. The members differ per block type — `ImageSource` and
    `VideoSource` are `bytes | s3Location`, only `DocumentSource` also takes
    `text | content` — so the caller passes the set that actually applies."""
    member = _one_of(src, members, f"{where}.source")
    if member == "bytes":
        _require(isinstance(src["bytes"], str) and src["bytes"],
                 f"{where}.source.bytes: base64 data is required")
        return {"type": "base64", "media_type": media, "data": src["bytes"]}
    if member == "s3Location":
        loc = src["s3Location"]
        _require(isinstance(loc, dict) and isinstance(loc.get("uri"), str) and loc["uri"],
                 f"{where}.source.s3Location.uri: an s3:// URI is required")
        return {"type": "url", "url": loc["uri"]}
    if member == "text":
        _require(isinstance(src["text"], str), f"{where}.source.text: must be a string")
        return {"type": "text", "media_type": "text/plain", "data": src["text"]}
    return {"type": "content", "content": src["content"]}


def _image_to_canonical(img: Any, where: str = "image") -> dict[str, Any]:
    _require(isinstance(img, dict), f"{where}: must be an object")
    fmt = img.get("format")
    _require(fmt in _IMAGE_FORMATS, f"{where}.format: must be one of {list(_IMAGE_FORMATS)}")
    _require(isinstance(img.get("source"), dict), f"{where}.source: is required")
    return {"type": "image",
            "source": _source_to_canonical(img["source"], where, _IMAGE_MEDIA[fmt],
                                           ("bytes", "s3Location"))}


def _document_to_canonical(doc: Any, where: str = "document") -> dict[str, Any]:
    _require(isinstance(doc, dict), f"{where}: must be an object")
    _require(isinstance(doc.get("name"), str) and doc["name"], f"{where}.name: is required")
    fmt = doc.get("format")
    _require(fmt is None or fmt in _DOC_FORMATS,
             f"{where}.format: must be one of {list(_DOC_FORMATS)}")
    _require(isinstance(doc.get("source"), dict), f"{where}.source: is required")
    media = _DOC_MEDIA.get(fmt or "txt", "application/octet-stream")
    return {"type": "document", "title": doc["name"],
            "source": _source_to_canonical(doc["source"], where, media,
                                           ("bytes", "s3Location", "text", "content"))}


def _video_to_canonical(vid: Any, where: str = "video") -> dict[str, Any]:
    _require(isinstance(vid, dict), f"{where}: must be an object")
    _require(vid.get("format") in _VIDEO_FORMATS,
             f"{where}.format: must be one of {list(_VIDEO_FORMATS)}")
    _require(isinstance(vid.get("source"), dict), f"{where}.source: is required")
    src = _source_to_canonical(vid["source"], where, f"video/{vid['format']}",
                               ("bytes", "s3Location"))
    # No canonical Anthropic equivalent; validated, then handed to the responder verbatim.
    return {"type": "video", "format": vid["format"], "source": src}


def _block_to_canonical(b: Any, where: str) -> dict[str, Any] | None:
    """One Converse ContentBlock -> canonical block (None for a cachePoint, handled above)."""
    member = _one_of(b, _CONTENT_MEMBERS, where)
    if member == "text":
        _require(isinstance(b["text"], str), f"{where}.text: must be a string")
        return {"type": "text", "text": b["text"]}
    if member == "image":
        return _image_to_canonical(b["image"], f"{where}.image")
    if member == "document":
        return _document_to_canonical(b["document"], f"{where}.document")
    if member == "toolUse":
        tu = b["toolUse"]
        _require(isinstance(tu, dict), f"{where}.toolUse: must be an object")
        _require(isinstance(tu.get("toolUseId"), str) and _TOOL_USE_ID_RE.match(tu["toolUseId"]),
                 f"{where}.toolUse.toolUseId: must match [a-zA-Z0-9_.:-]{{1,64}}")
        _require(isinstance(tu.get("name"), str) and _TOOL_NAME_RE.match(tu["name"]),
                 f"{where}.toolUse.name: must match [a-zA-Z0-9_-]{{1,64}}")
        _require("input" in tu, f"{where}.toolUse.input: is required")
        return {"type": "tool_use", "id": tu["toolUseId"], "name": tu["name"],
                "input": tu["input"]}
    if member == "toolResult":
        tr = b["toolResult"]
        _require(isinstance(tr, dict), f"{where}.toolResult: must be an object")
        _require(isinstance(tr.get("toolUseId"), str) and _TOOL_USE_ID_RE.match(tr["toolUseId"]),
                 f"{where}.toolResult.toolUseId: must match [a-zA-Z0-9_.:-]{{1,64}}")
        status = tr.get("status")
        _require(status in (None, "success", "error"),
                 f"{where}.toolResult.status: must be 'success' or 'error'")
        blk: dict[str, Any] = {"type": "tool_result", "tool_use_id": tr["toolUseId"],
                               "content": _tool_result_content(tr.get("content"),
                                                               f"{where}.toolResult")}
        if status == "error":
            blk["is_error"] = True
        return blk
    if member == "reasoningContent":
        rc = b["reasoningContent"]
        which = _one_of(rc, ("reasoningText", "redactedContent"), f"{where}.reasoningContent")
        if which == "reasoningText":
            rt = rc["reasoningText"]
            _require(isinstance(rt, dict) and isinstance(rt.get("text"), str),
                     f"{where}.reasoningContent.reasoningText.text: is required")
            out = {"type": "thinking", "thinking": rt["text"]}
            if rt.get("signature") is not None:
                _require(isinstance(rt["signature"], str),
                         f"{where}.reasoningContent.reasoningText.signature: must be a string")
                out["signature"] = rt["signature"]
            return out
        _require(isinstance(rc["redactedContent"], str),
                 f"{where}.reasoningContent.redactedContent: must be base64 data")
        return {"type": "redacted_thinking", "data": rc["redactedContent"]}
    if member == "guardContent":
        return _guard_content(b["guardContent"], where)
    if member == "cachePoint":
        return None  # handled by the caller
    if member == "video":
        return _video_to_canonical(b["video"], f"{where}.video")
    if member == "citationsContent":
        cc = b["citationsContent"]
        _require(isinstance(cc, dict), f"{where}.citationsContent: must be an object")
        for field in ("content", "citations"):
            _require(field not in cc or isinstance(cc[field], list),
                     f"{where}.citationsContent.{field}: must be an array")
    else:
        # audio / searchResult: this adapter validates that the member is an object and
        # hands it to the responder verbatim — it does not model their inner shape.
        _require(isinstance(b[member], dict), f"{where}.{member}: must be an object")
    return dict(b)


def _messages_to_canonical(messages: Any, sink: _CachePointSink) -> list[dict[str, Any]]:
    _require(isinstance(messages, list) and messages,
             "messages: must be a non-empty array")
    out: list[dict[str, Any]] = []
    for mi, m in enumerate(messages):
        _require(isinstance(m, dict), f"messages[{mi}]: must be an object")
        _require(m.get("role") in ("user", "assistant", "system"),
                 f"messages[{mi}].role: must be user, assistant or system")
        content = m.get("content")
        _require(isinstance(content, list),
                 f"messages[{mi}].content: must be an array of content blocks")
        blocks: list[dict[str, Any]] = []
        for bi, b in enumerate(content):
            where = f"messages[{mi}].content[{bi}]"
            if isinstance(b, dict) and "cachePoint" in b:
                _one_of(b, _CONTENT_MEMBERS, where)  # exactly one member, nothing unknown
                sink.mark(b["cachePoint"], where)
                continue
            blk = _block_to_canonical(b, where)
            if blk is not None:
                blocks.append(sink.block(blk))
        out.append({"role": m["role"], "content": blocks})
    return out


def _tools_to_canonical(tool_config: Any, sink: _CachePointSink) -> tuple[list[dict[str, Any]], Any]:
    if tool_config is None:
        return [], None
    _require(isinstance(tool_config, dict), "toolConfig: must be an object")
    tools_in = tool_config.get("tools")
    _require(isinstance(tools_in, list) and tools_in,
             "toolConfig.tools: must be a non-empty array")
    tools: list[dict[str, Any]] = []
    for i, t in enumerate(tools_in):
        member = _one_of(t, ("toolSpec", "cachePoint", "systemTool"), f"toolConfig.tools[{i}]")
        if member == "toolSpec":
            spec = t["toolSpec"]
            _require(isinstance(spec, dict), f"toolConfig.tools[{i}].toolSpec: must be an object")
            _require(isinstance(spec.get("name"), str) and _TOOL_NAME_RE.match(spec["name"]),
                     f"toolConfig.tools[{i}].toolSpec.name: must match [a-zA-Z0-9_-]{{1,64}}")
            schema = spec.get("inputSchema")
            schema = schema.get("json") if isinstance(schema, dict) else None
            _require(isinstance(schema, dict),
                     f"toolConfig.tools[{i}].toolSpec.inputSchema.json: must be an object")
            _require(schema.get("type", "object") == "object",
                     f"toolConfig.tools[{i}].toolSpec.inputSchema.json: the top-level "
                     f"schema type must be object")
            tool: dict[str, Any] = {"name": spec["name"], "input_schema": schema}
            if spec.get("description") is not None:
                _require(isinstance(spec["description"], str) and spec["description"],
                         f"toolConfig.tools[{i}].toolSpec.description: must be a non-empty string")
                tool["description"] = spec["description"]
            if spec.get("strict") is not None:
                _require(isinstance(spec["strict"], bool),
                         f"toolConfig.tools[{i}].toolSpec.strict: must be a boolean")
                tool["strict"] = spec["strict"]
            tools.append(sink.block(tool))
        elif member == "cachePoint":
            sink.mark(t["cachePoint"], f"toolConfig.tools[{i}]")
        else:
            st = t["systemTool"]
            _require(isinstance(st, dict) and isinstance(st.get("name"), str) and st["name"],
                     f"toolConfig.tools[{i}].systemTool.name: is required")
            tools.append(sink.block(dict(t)))  # systemTool: verbatim
    tc_in = tool_config.get("toolChoice")
    tool_choice: Any = None
    if tc_in is not None:
        which = _one_of(tc_in, ("auto", "any", "tool"), "toolConfig.toolChoice")
        _require(isinstance(tc_in[which], dict),
                 f"toolConfig.toolChoice.{which}: must be an object")
        if which == "tool":
            spec = tc_in["tool"]
            _require(isinstance(spec, dict) and isinstance(spec.get("name"), str) and spec["name"],
                     "toolConfig.toolChoice.tool.name: is required")
            _require(_TOOL_NAME_RE.match(spec["name"]),
                     "toolConfig.toolChoice.tool.name: must match [a-zA-Z0-9_-]{1,64}")
            tool_choice = {"type": "tool", "name": spec["name"]}
        else:
            tool_choice = {"type": which}
    return tools, tool_choice


def _inference_config(inf: Any) -> dict[str, Any]:
    if inf is None:
        return {}
    _require(isinstance(inf, dict), "inferenceConfig: must be an object")
    out: dict[str, Any] = {}
    if inf.get("maxTokens") is not None:
        v = inf["maxTokens"]
        _require(type(v) is int and v >= 1, "inferenceConfig.maxTokens: must be an integer >= 1")
        out["max_tokens"] = v
    for src, dst in (("temperature", "temperature"), ("topP", "top_p")):
        if inf.get(src) is not None:
            v = inf[src]
            _require(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1,
                     f"inferenceConfig.{src}: must be a number in [0, 1]")
            out[dst] = v
    if inf.get("stopSequences") is not None:
        v = inf["stopSequences"]
        _require(isinstance(v, list) and all(isinstance(x, str) and x for x in v),
                 "inferenceConfig.stopSequences: must be an array of non-empty strings")
        if v:
            out["stop_sequences"] = list(v)
    return out


_PROMPT_ARN_RE = re.compile(r"^arn:aws(?:-[a-z-]+)?:bedrock:[a-z0-9-]+:[0-9]{12}:prompt/")


def is_prompt_arn(model_id: str) -> bool:
    """A prompt-management ARN in the model-id position (`arn:…:prompt/ID[:version]`)."""
    return bool(_PROMPT_ARN_RE.match(model_id or ""))


def to_canonical(body: dict[str, Any], *, prompt_arn: bool = False) -> dict[str, Any]:
    """Converse request body -> canonical (Anthropic Messages-style) body.

    Raises ConverseValidationError for shapes the real API rejects with ValidationException.
    Cache points are resolved over the flattened tools -> system -> messages sequence, the
    order in which the real API evaluates checkpoints.
    """
    _require(isinstance(body, dict), "request body must be a JSON object")
    unknown = sorted(k for k in body if k not in _REQUEST_MEMBERS)
    _require(not unknown, f"unknown request member(s) {unknown}")
    sink = _CachePointSink()
    tools, tool_choice = _tools_to_canonical(body.get("toolConfig"), sink)
    system = _system_to_canonical(body.get("system"), sink)
    if prompt_arn and body.get("messages") is None and isinstance(body.get("promptVariables"), dict):
        # A prompt-management ARN as `modelId`: the stored prompt supplies the messages and
        # the request carries only `promptVariables`, so `messages` is legitimately absent
        # (the model requires `modelId` alone). The responder sees an empty list plus
        # `converse.promptVariables`. With an ordinary model id `messages` stays required —
        # a request that sends prompt-management fields there is a production mistake.
        messages: list[dict[str, Any]] = []
    else:
        messages = _messages_to_canonical(body.get("messages"), sink)
    out: dict[str, Any] = {"messages": messages}
    if system:
        out["system"] = system
    if tools:
        out["tools"] = tools
    if tool_choice is not None:
        out["tool_choice"] = tool_choice
    out.update(_inference_config(body.get("inferenceConfig")))
    extra = body.get("additionalModelRequestFields")
    if extra is not None:
        _require(isinstance(extra, dict), "additionalModelRequestFields: must be an object")
        # Only parameters the Converse schema has no field for belong here. A key Converse
        # already models would either bypass its validation (`system` as a bare string,
        # `tools` without toolSpec checks) or silently lose to the Converse field; both
        # hide a request production would not accept.
        clash = sorted(k for k in extra if k in _CONVERSE_OWNED_FIELDS)
        _require(not clash,
                 f"additionalModelRequestFields: {', '.join(clash)} — set through the "
                 f"Converse fields (messages / system / inferenceConfig / toolConfig / "
                 f"serviceTier / outputConfig) instead")
        _require(not ("output_config" in extra and not isinstance(extra["output_config"], dict)),
                 "additionalModelRequestFields.output_config: must be an object")
        for k, v in extra.items():
            out.setdefault(k, v)
    out_cfg = body.get("outputConfig")
    if out_cfg is not None:
        _require(isinstance(out_cfg, dict), "outputConfig: must be an object")
        unknown = sorted(k for k in out_cfg if k not in ("effort", "textFormat"))
        _require(not unknown, f"outputConfig: unknown member(s) {unknown}")
        effort = out_cfg.get("effort")
        _require(effort is None or effort in _EFFORTS,
                 f"outputConfig.effort: must be one of {list(_EFFORTS)}")
        fmt = out_cfg.get("textFormat")
        if fmt is not None:
            # OutputFormat: `type` (json_schema) and `structure` (a union, currently only
            # `jsonSchema`) are both required.
            _require(isinstance(fmt, dict), "outputConfig.textFormat: must be an object")
            _require(fmt.get("type") == "json_schema",
                     "outputConfig.textFormat.type: must be json_schema")
            which = _one_of(fmt.get("structure"), ("jsonSchema",),
                            "outputConfig.textFormat.structure")
            jsd = fmt["structure"][which]
            _require(isinstance(jsd, dict),
                     "outputConfig.textFormat.structure.jsonSchema: must be an object")
            unknown = sorted(k for k in jsd if k not in ("schema", "name", "description"))
            _require(not unknown,
                     f"outputConfig.textFormat.structure.jsonSchema: unknown member(s) {unknown}")
            # JsonSchemaDefinition.schema is a JSON *string* (required); name/description
            # are optional strings. The canonical body wants the decoded schema object.
            _require(isinstance(jsd.get("schema"), str) and jsd["schema"].strip(),
                     "outputConfig.textFormat.structure.jsonSchema.schema: a JSON string is required")
            try:
                decoded = json.loads(jsd["schema"])
            except ValueError as e:
                raise ConverseValidationError(
                    f"outputConfig.textFormat.structure.jsonSchema.schema: not valid JSON ({e})")
            _require(isinstance(decoded, dict),
                     "outputConfig.textFormat.structure.jsonSchema.schema: must encode a JSON object")
            for opt in ("name", "description"):
                _require(jsd.get(opt) is None or isinstance(jsd[opt], str),
                         f"outputConfig.textFormat.structure.jsonSchema.{opt}: must be a string")
        if effort is not None or fmt is not None:
            # The native fields map to the same place the responder already reads
            # (`output_config.effort` / `output_config.format`), taking precedence over a
            # spelling that also arrived through additionalModelRequestFields.
            cfg = out.setdefault("output_config", {})
            if effort is not None:
                cfg["effort"] = effort
            if fmt is not None:
                jsd = fmt["structure"]["jsonSchema"]
                cfg["format"] = {"type": "json_schema", "schema": json.loads(jsd["schema"])}
                if jsd.get("name"):
                    cfg["format"]["name"] = jsd["name"]
    tier = body.get("serviceTier")
    if tier is not None:
        _require(isinstance(tier, dict) and tier.get("type") in _SERVICE_TIERS,
                 f"serviceTier.type: must be one of {list(_SERVICE_TIERS)}")
        # Converse `priority | default | flex | reserved` -> Anthropic `auto | standard_only`,
        # the same mapping the relay uses in the other direction (`default` <-> standard_only).
        out["service_tier"] = "auto" if tier["type"] in ("priority", "flex") else "standard_only"
    meta = body.get("requestMetadata")
    if meta is not None:
        _require(isinstance(meta, dict) and 1 <= len(meta) <= _MAX_REQUEST_METADATA
                 and all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()),
                 f"requestMetadata: 1-{_MAX_REQUEST_METADATA} string entries")
        for k, v in meta.items():
            _require(bool(_REQUEST_METADATA_KEY_RE.match(k)),
                     f"requestMetadata: key {k[:40]!r} must be 1-256 characters of "
                     f"[a-zA-Z0-9\\s:_@$#=/+,-.]")
            _require(bool(_REQUEST_METADATA_VALUE_RE.match(v)),
                     f"requestMetadata[{k[:40]!r}]: value must be at most 256 characters of "
                     f"[a-zA-Z0-9\\s:_@$#=/+,-.]")
    perf = body.get("performanceConfig")
    if perf is not None:
        _require(isinstance(perf, dict), "performanceConfig: must be an object")
        # `latency` is optional in the model (`{}` means the default, `standard`).
        _require(perf.get("latency") in (None, "standard", "optimized"),
                 "performanceConfig.latency: must be 'standard' or 'optimized'")
    return out


def converse_extras(body: dict[str, Any]) -> dict[str, Any]:
    """Converse-only request fields kept on the snapshot for the responder / encoders."""
    keys = ("additionalModelResponseFieldPaths", "guardrailConfig", "performanceConfig",
            "promptVariables", "requestMetadata", "serviceTier", "outputConfig")
    return {k: body[k] for k in keys if k in body}


# ── response: canonical -> Converse ───────────────────────────────────


def block_from_canonical(b: dict[str, Any]) -> dict[str, Any] | None:
    """canonical block -> Converse ContentBlock (None when it has no representation).

    Only the block types an assistant turn can actually contain are handled: a responder's
    blocks are normalized to text / tool_use / thinking / redacted_thinking before they
    reach any encoder, exactly as the real API's output message is limited to
    `text` / `toolUse` / `reasoningContent`.
    """
    t = b.get("type")
    if t == "text":
        return {"text": str(b.get("text", ""))}
    if t == "tool_use":
        return {"toolUse": {"toolUseId": str(b.get("id") or ""), "name": str(b.get("name", "")),
                            "input": b.get("input") if b.get("input") is not None else {}}}
    if t == "thinking":
        rt: dict[str, Any] = {"text": str(b.get("thinking") or "")}
        if b.get("signature"):
            rt["signature"] = str(b["signature"])
        return {"reasoningContent": {"reasoningText": rt}}
    if t == "redacted_thinking":
        return {"reasoningContent": {"redactedContent": str(b.get("data") or "")}}
    return None


def stop_reason_out(stop_reason: str | None) -> str:
    if stop_reason in _CONVERSE_STOP_REASONS:
        return stop_reason
    return _STOP_REASON_MAP.get(stop_reason or "end_turn", "end_turn")


def usage_out(usage: dict[str, Any]) -> dict[str, Any]:
    inp = int(usage.get("input_tokens", 0))
    out_t = int(usage.get("output_tokens", 0))
    read = int(usage.get("cache_read_input_tokens", 0))
    write = int(usage.get("cache_creation_input_tokens", 0))
    u: dict[str, Any] = {"inputTokens": inp, "outputTokens": out_t,
                         "totalTokens": inp + read + write + out_t}
    if read or write:
        u["cacheReadInputTokens"] = read
        u["cacheWriteInputTokens"] = write
    cc = usage.get("cache_creation")
    details = []
    if isinstance(cc, dict):
        for ttl, key in (("1h", "ephemeral_1h_input_tokens"), ("5m", "ephemeral_5m_input_tokens")):
            n = int(cc.get(key) or 0)
            if n:
                details.append({"inputTokens": n, "ttl": ttl})
    if write:
        u["cacheDetails"] = details if details else [{"inputTokens": write, "ttl": "5m"}]
    return u


_MAX_RESPONSE_FIELD_PATHS = 10


def _json_pointer(doc: Any, pointer: str) -> tuple[bool, Any]:
    """Resolve one RFC 6901 pointer. Returns (found, value); a malformed pointer raises.

    The real API "rejects an empty JSON Pointer or incorrectly structured JSON Pointer with
    a 400 error code. If the JSON Pointer is valid, but the requested field is not in the
    model response, it is ignored."
    """
    if not pointer.startswith("/"):
        raise ConverseValidationError(
            f"additionalModelResponseFieldPaths: invalid JSON pointer {pointer!r}")
    # Validate the whole pointer's syntax first: a malformed escape late in the path is a
    # 400 even when an earlier member is missing (the lookup would stop before reaching it).
    tokens = []
    for raw in pointer[1:].split("/"):
        # RFC 6901: `~` may only introduce `~0` (a literal ~) or `~1` (a literal /)
        i = 0
        while True:
            i = raw.find("~", i)
            if i < 0:
                break
            if i + 1 >= len(raw) or raw[i + 1] not in "01":
                raise ConverseValidationError(
                    f"additionalModelResponseFieldPaths: invalid escape in {pointer!r}")
            i += 2
        tokens.append(raw.replace("~1", "/").replace("~0", "~"))
    cur = doc
    for token in tokens:
        if isinstance(cur, dict) and token in cur:
            cur = cur[token]
        elif isinstance(cur, list) and token.isdigit() and int(token) < len(cur):
            cur = cur[int(token)]
        else:
            return False, None
    return True, cur


def additional_fields(native: dict[str, Any], paths: Any) -> dict[str, Any] | None:
    """Resolve `additionalModelResponseFieldPaths` (JSON pointers into the native
    Messages response); missing fields are ignored, malformed pointers are a 400.

    Results are keyed by the pointer's last token (matching the documented
    `["/stop_sequence"] -> {"stop_sequence": …}` example) and fall back to the full pointer
    when two paths would otherwise collide.
    """
    if paths is None:
        return None
    if not isinstance(paths, list):
        raise ConverseValidationError("additionalModelResponseFieldPaths: must be an array")
    if len(paths) > _MAX_RESPONSE_FIELD_PATHS:
        raise ConverseValidationError(
            f"additionalModelResponseFieldPaths: at most {_MAX_RESPONSE_FIELD_PATHS} paths")
    out: dict[str, Any] = {}
    for p in paths:
        if not isinstance(p, str) or not p or len(p) > 256:
            raise ConverseValidationError(
                "additionalModelResponseFieldPaths: entries must be non-empty JSON pointers "
                "of at most 256 characters")
        found, value = _json_pointer(native, p)
        if not found:
            continue
        key = (p.rstrip("/").split("/")[-1] or p).replace("~1", "/").replace("~0", "~")
        if key in out and out[key] != value:
            key = p
        out[key] = value
    return out or None


def build_response(result: dict[str, Any], model: str, snapshot: dict[str, Any],
                   latency_ms: int) -> dict[str, Any]:
    blocks = strip_private(result["content_blocks"])
    stop_reason, _seq, _details = fs._resolve_stop(blocks, result.get("stop_reason"),
                                                   result.get("stop_sequence"),
                                                   result.get("stop_details"),
                                                   snapshot.get("params"))
    content = [c for c in (block_from_canonical(b) for b in blocks) if c is not None]
    resp: dict[str, Any] = {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason_out(stop_reason),
        "usage": usage_out(result["usage"]),
        "metrics": {"latencyMs": latency_ms},
    }
    extras = snapshot.get("converse") or {}
    native = fs._build_non_stream_response(result["message_id"], model, blocks, result["usage"],
                                           result.get("stop_reason"), result.get("stop_sequence"),
                                           result.get("stop_details"), snapshot.get("params"))
    add = additional_fields(native, extras.get("additionalModelResponseFieldPaths"))
    if add is not None:
        resp["additionalModelResponseFields"] = add
    if isinstance(extras.get("performanceConfig"), dict):
        resp["performanceConfig"] = {"latency": extras["performanceConfig"].get("latency") or "standard"}
    if isinstance(extras.get("serviceTier"), dict) and extras["serviceTier"].get("type"):
        resp["serviceTier"] = {"type": extras["serviceTier"]["type"]}
    return resp


def stream_events(content_blocks: list[dict[str, Any]], stop_reason: str,
                  usage: dict[str, Any] | None, latency_ms: int,
                  extras: dict[str, Any] | None = None,
                  additional: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """(event name, payload) sequence of a ConverseStream response. `usage=None` omits the
    trailing metadata (used for partial streams that end in an exception)."""
    out: list[tuple[str, dict[str, Any]]] = [("messageStart", {"role": "assistant"})]
    idx = -1
    for b in strip_private(content_blocks):
        t = b.get("type")
        if t == "text":
            idx += 1
            text = str(b.get("text", ""))
            pieces = [text[i:i + _TEXT_CHUNK] for i in range(0, len(text), _TEXT_CHUNK)] or [""]
            for p in pieces:
                out.append(("contentBlockDelta", {"contentBlockIndex": idx, "delta": {"text": p}}))
            out.append(("contentBlockStop", {"contentBlockIndex": idx}))
        elif t == "tool_use":
            idx += 1
            out.append(("contentBlockStart", {"contentBlockIndex": idx, "start": {"toolUse": {
                "toolUseId": str(b.get("id") or ""), "name": str(b.get("name", ""))}}}))
            js = json.dumps(b.get("input") if b.get("input") is not None else {}, ensure_ascii=False)
            for i in range(0, len(js), _JSON_CHUNK):
                out.append(("contentBlockDelta", {"contentBlockIndex": idx,
                                                  "delta": {"toolUse": {"input": js[i:i + _JSON_CHUNK]}}}))
            out.append(("contentBlockStop", {"contentBlockIndex": idx}))
        elif t == "thinking":
            idx += 1
            text = str(b.get("thinking") or "")
            for i in range(0, len(text), _TEXT_CHUNK):
                out.append(("contentBlockDelta", {"contentBlockIndex": idx,
                                                  "delta": {"reasoningContent": {"text": text[i:i + _TEXT_CHUNK]}}}))
            if b.get("signature"):
                out.append(("contentBlockDelta", {"contentBlockIndex": idx,
                                                  "delta": {"reasoningContent": {"signature": str(b["signature"])}}}))
            out.append(("contentBlockStop", {"contentBlockIndex": idx}))
        elif t == "redacted_thinking":
            idx += 1
            out.append(("contentBlockDelta", {"contentBlockIndex": idx,
                                              "delta": {"reasoningContent": {"redactedContent": str(b.get("data") or "")}}}))
            out.append(("contentBlockStop", {"contentBlockIndex": idx}))
    if usage is None:
        return out
    stop_event: dict[str, Any] = {"stopReason": stop_reason_out(stop_reason)}
    if additional:
        stop_event["additionalModelResponseFields"] = additional
    out.append(("messageStop", stop_event))
    meta: dict[str, Any] = {"usage": usage_out(usage), "metrics": {"latencyMs": latency_ms}}
    extras = extras or {}
    if isinstance(extras.get("performanceConfig"), dict):
        meta["performanceConfig"] = {"latency": extras["performanceConfig"].get("latency") or "standard"}
    if isinstance(extras.get("serviceTier"), dict) and extras["serviceTier"].get("type"):
        meta["serviceTier"] = {"type": extras["serviceTier"]["type"]}
    out.append(("metadata", meta))
    return out


# ── routes ────────────────────────────────────────────────────────────


def _error(status: int, etype: str | None, message: str, req_id: str,
           extra_headers: dict[str, str] | None = None, *,
           resource_name: str | None = None,
           original_status: int | None = None) -> JSONResponse:
    return _bedrock._bedrock_error_response(status, etype, message, request_id=req_id,
                                            resource_name=resource_name,
                                            original_status=original_status,
                                            extra_headers=extra_headers)


async def _receive(model_id: str, request: Request, *, is_stream: bool, req_id: str,
                   ) -> tuple[Any, dict[str, Any], asyncio.Future] | JSONResponse:
    op = "converse-stream" if is_stream else "converse"
    model = _bedrock.normalize_model_id(model_id)
    body, errmsg = await fs._parse_json_body(request)
    if errmsg is not None:
        _bedrock._log(f"{op} model={model.raw} rejected: {errmsg}")
        return _error(400, "ValidationException", errmsg, req_id)
    try:
        canonical = to_canonical(body, prompt_arn=is_prompt_arn(model_id))
        extras = converse_extras(body)
        additional_fields({}, extras.get("additionalModelResponseFieldPaths"))  # pointer syntax check
        snapshot, fut = await fs.register_request(
            "bedrock", model.canonical, canonical, is_stream=is_stream,
            extra={"bedrock_model_id": model.raw, "api": "converse", "converse": extras},
        )
    except (ConverseValidationError, fs.RequestValidationError) as e:
        _bedrock._log(f"{op} model={model.raw} rejected: {e}")
        return _error(400, "ValidationException", str(e), req_id)
    mapped = f" -> {model.canonical}" if model.canonical != model.raw else ""
    _bedrock._log(f"{op} model={model.raw}{mapped} pending={snapshot['pending_id']} "
                  f"turn={snapshot['turn']} request_id={req_id}")
    return model, snapshot, fut


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/model/{model_id:path}/converse")
    async def converse(model_id: str, request: Request) -> Any:
        req_id = str(uuid.uuid4())
        received = await _receive(model_id, request, is_stream=False, req_id=req_id)
        if isinstance(received, JSONResponse):
            return received
        model, snapshot, fut = received
        result = await fs.await_resolution(snapshot, fut)
        if result["kind"] == "cleared":
            return _error(503, "ServiceUnavailableException",
                          f"request cleared: {result['detail']}", req_id)
        if result["kind"] == "error":
            return _error(result["status"], result["type"], result["message"], req_id,
                          fs._error_headers(result), resource_name=model.raw,
                          original_status=result.get("original_status"))
        try:
            resp = build_response(result, model.canonical, snapshot, _bedrock._latency_ms(snapshot))
        except ConverseValidationError as e:
            return _error(400, "ValidationException", str(e), req_id)
        return JSONResponse(resp, headers={"x-amzn-requestid": req_id})

    @router.post("/model/{model_id:path}/converse-stream")
    async def converse_stream(model_id: str, request: Request) -> Any:
        req_id = str(uuid.uuid4())
        received = await _receive(model_id, request, is_stream=True, req_id=req_id)
        if isinstance(received, JSONResponse):
            return received
        model, snapshot, fut = received
        result = await fs.await_resolution(snapshot, fut)
        headers = {"x-amzn-requestid": req_id}
        if result["kind"] == "cleared":
            return _error(503, "ServiceUnavailableException",
                          f"request cleared: {result['detail']}", req_id)
        if result["kind"] == "error":
            if result.get("after_events") is not None:
                partial = stream_events(result["content_blocks"], "end_turn", None, 0)
                n = int(result.get("after_events") or 0)
                frames = [eventstream.encode_event(name, data) for name, data in partial[:n]]
                member = _bedrock.stream_exception_member(
                    _bedrock.exception_name_for(result["status"], result["type"]),
                    operation="converse", status=result["status"])
                frames.append(eventstream.encode_exception(
                    member, result["message"], _bedrock.stream_exception_fields(member, result)))
            else:
                return _error(result["status"], result["type"], result["message"], req_id,
                              fs._error_headers(result), resource_name=model.raw,
                              original_status=result.get("original_status"))
        else:
            blocks = result["content_blocks"]
            stop_reason, _s, _d = fs._resolve_stop(blocks, result.get("stop_reason"),
                                                   result.get("stop_sequence"),
                                                   result.get("stop_details"), snapshot.get("params"))
            extras = snapshot.get("converse") or {}
            native = fs._build_non_stream_response(
                result["message_id"], model.canonical, strip_private(blocks), result["usage"],
                result.get("stop_reason"), result.get("stop_sequence"),
                result.get("stop_details"), snapshot.get("params"))
            try:
                additional = additional_fields(native, extras.get("additionalModelResponseFieldPaths"))
            except ConverseValidationError as e:
                return _error(400, "ValidationException", str(e), req_id)
            events = stream_events(blocks, stop_reason, result["usage"],
                                   _bedrock._latency_ms(snapshot), extras, additional)
            frames = [eventstream.encode_event(name, data) for name, data in events]

        async def gen():
            for frame in frames:
                yield frame
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type=_EVENTSTREAM_MEDIA, headers=headers)

    return router
