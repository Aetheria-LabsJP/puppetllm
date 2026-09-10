"""Relay responder: auto-forward pending requests to a real LLM API.

This turns puppetllm into a cross-provider bridge: an app keeps speaking its own
SDK (Anthropic / OpenAI / Bedrock — all normalized to the canonical form by the
server), while this relay translates each pending request to a real upstream API
(any OpenAI-compatible endpoint such as OpenAI / xAI Grok / Groq / Ollama /
OpenRouter, or the native Anthropic API), calls it, and injects the upstream
response back as canonical content blocks — including the **real** stop_reason
and token usage (via the /_control/respond `stop_reason` / `usage` fields).

The relay is just another responder client of the control plane. It does not
change the server. By default, while running, it claims EVERY pending it can see
(the control plane's pending list is global) and forwards it upstream — so it does
not share a live queue with a human / AI-agent responder (responder/CLAUDE.md,
responder/AGENTS.md); stop the relay and you can take over by hand at any time.
Pass --only "<glob>,..." to claim only the pendings whose inbound model matches,
leaving the rest for another responder — that is the one way relay and a human /
AI-agent responder run concurrently (partitioned by model, not the same request).

Usage:
  python -m puppetllm.relay --target https://api.x.ai/v1 \\
      --api-key-env XAI_API_KEY --model grok-3
  python -m puppetllm.relay --kind anthropic --model claude-sonnet-4-5

Notes:
- The upstream call is non-streaming; the app still receives SSE (the server
  pseudo-streams the injected blocks), but first-token latency equals the full
  upstream response time.
- Upstream API errors are relayed via /_control/error (status/type/message and,
  for OpenAI-style errors, code/param), so the app's SDK raises the same
  exception class it would against the real API.
- This mode calls a real API and therefore incurs real cost.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import os
import sys
from collections import OrderedDict
from typing import Any
from urllib.parse import urlsplit

import httpx

from . import openai_wire as _oai

# Upper bound on the quarantine set (see Relay.quarantined). Far above any realistic
# count of concurrently-failing pendings; keeps a long-running relay from leaking.
_QUARANTINE_MAX = 4096

OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

# canonical stop_reason <- OpenAI finish_reason
_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",  # legacy OpenAI single-function form
    "content_filter": "refusal",
}
# An unrecognized finish_reason is mapped here rather than passed through verbatim, so a
# non-standard upstream value never reaches the app's SDK as a bogus stop_reason.
_FINISH_FALLBACK = "end_turn"


def _log(msg: str) -> None:
    print(f"[relay] {msg}", file=sys.stderr, flush=True)


def _as_int(v: Any) -> int:
    """Best-effort int coercion for upstream-supplied token counts (0 on garbage)."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ── model mapping ────────────────────────────────────────────────────


def parse_model_map(spec: str | None) -> list[tuple[str, str]]:
    """Parse "pat=model,pat2=model2" into ordered (glob-pattern, target-model) pairs."""
    out: list[tuple[str, str]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--model-map entry has no '=': {part!r}")
        pat, target = part.split("=", 1)
        out.append((pat.strip(), target.strip()))
    return out


def map_model(inbound: Any, force: str | None, mapping: list[tuple[str, str]]) -> str:
    """--model wins; else the first matching --model-map glob; else pass through."""
    if force:
        return force
    m = str(inbound or "")
    for pat, target in mapping:
        if fnmatch.fnmatch(m, pat):
            return target
    return m


# Model-id prefixes that clearly belong to the "other" vendor. Used only to warn on a likely
# passthrough misconfiguration (e.g. forwarding "claude-*" to an OpenAI endpoint with no
# --model / --model-map); never affects routing.
_ANTHROPIC_HINTS = ("claude", "anthropic.")
_OPENAI_HINTS = ("gpt", "o1", "o1-", "o3", "o3-", "o4", "grok", "gemini", "llama",
                 "mistral", "qwen", "deepseek")


def looks_foreign(model: str, kind: str) -> bool:
    """True if `model` looks like it names the vendor opposite to `kind` (heuristic).

    Only meaningful for an unmapped passthrough: an Anthropic-named model going to an OpenAI
    endpoint, or vice versa, is almost always a missing --model / --model-map.
    """
    m = (model or "").lower()
    if not m:
        return False
    if kind == "anthropic":
        return any(m.startswith(h) for h in _OPENAI_HINTS)
    return any(m.startswith(h) for h in _ANTHROPIC_HINTS)


# ── canonical -> OpenAI-compatible request ───────────────────────────


_image_to_openai = _oai.image_to_openai


def _effort_to_anthropic(effort: Any) -> str | None:
    """OpenAI reasoning_effort → Anthropic output_config.effort."""
    if not isinstance(effort, str):
        return None
    return {"none": "low", "minimal": "low"}.get(effort, effort) if effort in (
        "none", "minimal", "low", "medium", "high", "xhigh", "max") else None


def _effort_to_openai(effort: Any) -> str | None:
    """Anthropic output_config.effort → OpenAI reasoning_effort (`max` is Anthropic-only)."""
    if not isinstance(effort, str):
        return None
    return {"max": "xhigh"}.get(effort, effort) if effort in (
        "low", "medium", "high", "xhigh", "max") else None


# service_tier vocabularies differ: Anthropic `auto` | `standard_only`, OpenAI
# `auto` | `default` | `flex` | `priority` (+ `scale`). Map instead of forwarding verbatim.
_TIER_TO_OPENAI = {"auto": "auto", "standard_only": "default",
                   "default": "default", "flex": "flex", "priority": "priority", "scale": "scale"}
_TIER_TO_ANTHROPIC = {"auto": "auto", "standard_only": "standard_only",
                      "default": "standard_only", "flex": "auto", "priority": "auto", "scale": "auto"}


_warned_dropped: set[str] = set()


def _warn_dropped(what: str) -> None:
    """Log once per parameter that cannot be translated for the upstream kind."""
    if what not in _warned_dropped:
        _warned_dropped.add(what)
        _log(f"WARNING: {what} cannot be translated for this upstream and is dropped")


def _text_of(content: Any) -> str:
    """Join the text of a canonical str-or-blocks content value (cache_control etc. dropped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return "" if content is None else str(content)


def _tool_result_text(content: Any) -> str:
    """Stringify a canonical tool_result content (str, or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(b.get("text", "")) if isinstance(b, dict) and b.get("type") == "text"
            else json.dumps(_oai.strip_private(b), ensure_ascii=False)
            for b in content
        )
    return json.dumps(_oai.strip_private(content), ensure_ascii=False)


def _tool_choice_to_openai(tc: Any) -> Any:
    """Map a canonical/Anthropic tool_choice to the OpenAI form (passthrough if already OpenAI-style)."""
    if isinstance(tc, str):
        return tc  # "auto" / "required" / "none" (OpenAI-style, e.g. inbound via the OpenAI route)
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "auto":
            return "auto"
        if t in ("any", "required"):
            return "required"
        if t == "none":
            return "none"
        if t == "tool":  # Anthropic forced-tool form
            return {"type": "function", "function": {"name": tc.get("name")}}
        if t == "function" or "function" in tc:  # already OpenAI-style
            return tc
    return None


def resolve_max_tokens_param(cfg: Any) -> str:
    """Which field carries the token limit on the OpenAI route.

    `max_tokens` is deprecated on the official API (rejected by o-series / reasoning
    models) in favor of `max_completion_tokens`, while many OpenAI-compatible backends
    still only know `max_tokens`. Default ("auto"): `max_completion_tokens` when the
    target host is api.openai.com, `max_tokens` otherwise; an explicit value wins.
    """
    explicit = getattr(cfg, "max_tokens_param", None)
    if explicit in ("max_tokens", "max_completion_tokens"):
        return explicit
    target = str(getattr(cfg, "target", "") or "")
    try:
        host = (urlsplit(target).hostname or "").lower()
    except ValueError:
        host = ""
    return "max_completion_tokens" if host == "api.openai.com" else "max_tokens"


def to_openai_request(req: dict[str, Any], model: str,
                      cfg: Any = None) -> dict[str, Any]:
    """Canonical snapshot -> OpenAI-compatible chat.completions request body."""
    max_tokens_param = resolve_max_tokens_param(cfg)
    messages: list[dict[str, Any]] = []
    system = req.get("system")
    if system:
        messages.append({"role": "system", "content": _text_of(system)})

    for m in req.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "assistant":
            if isinstance(content, list):
                texts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        texts.append(str(b.get("text", "")))
                    elif b.get("type") == "tool_use":
                        tool_calls.append(_oai.tool_call_from_canonical(b, keep_id=True))
                joined = "".join(texts)
                # Drop a fully-empty assistant turn (e.g. it held only thinking blocks,
                # which the server strips): {"content": None} with no tool_calls is
                # rejected by some strict OpenAI-compatible backends.
                if joined or tool_calls:
                    msg: dict[str, Any] = {"role": "assistant",
                                           "content": joined or None}
                    if tool_calls:
                        msg["tool_calls"] = tool_calls
                    messages.append(msg)
            else:
                messages.append({"role": "assistant", "content": _text_of(content)})
        else:  # user (or unknown -> treat as user)
            if isinstance(content, list):
                parts: list[dict[str, Any]] = []
                has_image = False
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_result":
                        # tool messages must directly follow the assistant tool_calls turn.
                        # tool_results are emitted here in-loop while any sibling text is
                        # buffered and appended after — so tool messages always precede the
                        # user text of the same turn regardless of intra-turn block order.
                        messages.append({
                            "role": "tool",
                            "tool_call_id": str(b.get("tool_use_id") or ""),
                            "content": _tool_result_text(b.get("content")),
                        })
                    elif b.get("type") == "text":
                        parts.append({"type": "text", "text": str(b.get("text", ""))})
                    elif b.get("type") == "image":
                        part = _image_to_openai(b)
                        if part is not None:
                            parts.append(part)
                            has_image = True
                        else:
                            parts.append({"type": "text",
                                          "text": "[image omitted by relay]"})
                    elif b.get("type") in ("image_url", "input_audio", "file"):
                        parts.append(b)  # already OpenAI-shaped (OpenAI-inbound passthrough)
                        has_image = True
                    elif b.get("type") in ("input_image", "document"):
                        # not translated; keep the turn non-empty so the message
                        # sequence stays valid rather than silently vanishing.
                        parts.append({"type": "text",
                                      "text": "[non-text content omitted by relay]"})
                if parts:
                    if has_image:
                        messages.append({"role": "user", "content": parts})
                    else:
                        messages.append({"role": "user",
                                         "content": "".join(p["text"] for p in parts)})
            else:
                messages.append({"role": "user", "content": _text_of(content)})

    body: dict[str, Any] = {"model": model, "messages": messages}
    if req.get("max_tokens") is not None:
        body[max_tokens_param] = req["max_tokens"]
    tools = req.get("tools") or []
    if tools:
        out_tools: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            if t.get("_openai_custom"):
                custom: dict[str, Any] = {"name": t.get("name")}
                if t.get("description"):
                    custom["description"] = t["description"]
                if t.get("format") is not None:
                    custom["format"] = t["format"]
                out_tools.append({"type": "custom", "custom": custom})
                continue
            fn: dict[str, Any] = {
                "name": t.get("name"),
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object"},
            }
            if t.get("strict") is not None:
                fn["strict"] = bool(t["strict"])
            out_tools.append({"type": "function", "function": fn})
        body["tools"] = out_tools
    params = req.get("params") or {}
    for k in _OPENAI_PASSTHROUGH_PARAMS:
        if k in params:
            body[k] = params[k]
    # Anthropic-inbound effort → OpenAI reasoning_effort (`max` clamps to `xhigh`).
    oc = params.get("output_config")
    if "reasoning_effort" not in body and isinstance(oc, dict) and oc.get("effort"):
        eff = _effort_to_openai(oc["effort"])
        if eff is not None:
            body["reasoning_effort"] = eff
        else:
            _warn_dropped(f"output_config.effort={oc['effort']!r}")
    tier = params.get("service_tier")
    if isinstance(tier, str):
        if tier in _TIER_TO_OPENAI:
            body["service_tier"] = _TIER_TO_OPENAI[tier]
        else:
            _warn_dropped(f"service_tier={tier!r}")
    if params.get("speed") == "fast":
        _warn_dropped("speed=fast (Anthropic fast mode has no Chat Completions equivalent)")
    if "stop_sequences" in params:
        body["stop"] = params["stop_sequences"]
    elif "stop" in params:
        body["stop"] = params["stop"]
    tc = _tool_choice_to_openai(params.get("tool_choice"))
    if tc is not None:
        body["tool_choice"] = tc
    return body


# Chat Completions parameters forwarded verbatim from the snapshot when present.
# (`n` is deliberately NOT forwarded — the relay uses one choice, so extra samples would
# only be billed; `service_tier` is mapped, not forwarded.)
_OPENAI_PASSTHROUGH_PARAMS = (
    "temperature", "top_p", "response_format", "parallel_tool_calls", "reasoning_effort",
    "seed", "frequency_penalty", "presence_penalty", "logit_bias", "logprobs",
    "top_logprobs", "verbosity", "prompt_cache_key", "safety_identifier", "user",
    "metadata", "store", "prediction", "web_search_options", "modalities", "audio",
)


def from_openai_response(resp: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-compatible chat.completion -> /_control/respond payload fields."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    blocks: list[dict[str, Any]] = []
    # content is normally a plain string, but some OpenAI-compatible backends (vLLM / routers)
    # return a list of content parts — join their text rather than str()-ing the whole list,
    # which would deliver literal "[{'type': 'text', ...}]" to the app.
    text = _text_of(message.get("content"))
    if text:
        blocks.append({"type": "text", "text": text})
    refusal = message.get("refusal")
    refused = isinstance(refusal, str) and bool(refusal)
    if refused:
        # A refusal comes back with content null + refusal text; surface it as the
        # message text and as stop_reason "refusal" so the app can branch on it.
        blocks.append({"type": "text", "text": refusal})
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        blk = _oai.tool_call_to_canonical(tc)
        if blk is not None:
            blocks.append(blk)

    finish = choice.get("finish_reason")
    stop_reason = _FINISH_TO_STOP.get(finish, _FINISH_FALLBACK) if finish else None
    if refused:
        stop_reason = "refusal"
    # If the message carries tool calls, the turn IS a tool_use turn regardless of what
    # finish_reason says. Some OpenAI-compatible backends (Ollama / llama.cpp / vLLM /
    # routers) emit tool_calls while still reporting finish_reason "stop"/"length"; passing
    # that through as end_turn/max_tokens would stall an Anthropic-SDK agent's tool loop
    # (it keys on stop_reason == "tool_use"). Mirror real Anthropic + the server's own
    # auto-derivation and force tool_use whenever a tool_use block is present.
    if any(b.get("type") == "tool_use" for b in blocks):
        stop_reason = "tool_use"

    u = resp.get("usage")
    usage = None
    # Only override puppetllm's approx when the upstream actually reported tokens; some
    # OpenAI-compatible backends omit usage, and forcing zeros would be worse than the estimate.
    if isinstance(u, dict) and (u.get("prompt_tokens") or u.get("completion_tokens")):
        details = u.get("prompt_tokens_details")
        # Guard non-dict garbage: a malformed upstream sending e.g. a list here must not turn
        # a successful, billed response into a 502 (AttributeError on .get). Treat as absent.
        cached = max(0, _as_int(details.get("cached_tokens"))
                     if isinstance(details, dict) else 0)
        written = max(0, _as_int(details.get("cache_write_tokens"))
                      if isinstance(details, dict) else 0)
        usage = {
            # canonical (Anthropic) vocabulary: input_tokens excludes cached reads/writes.
            # Clamp every value non-negative — a buggy/hostile upstream reporting negatives
            # would otherwise be rejected by the server and hang/loop the request.
            "input_tokens": max(0, _as_int(u.get("prompt_tokens")) - cached - written),
            "output_tokens": max(0, _as_int(u.get("completion_tokens"))),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": written,
        }
    return {"content": blocks, "stop_reason": stop_reason, "usage": usage}


# ── canonical -> Anthropic request (near-verbatim) ───────────────────


def _tool_choice_to_anthropic(tc: Any) -> Any:
    """Map an OpenAI/canonical tool_choice to the Anthropic form (inverse of the OpenAI
    mapping). Returns None if there's nothing to send."""
    if isinstance(tc, str):
        return {"auto": {"type": "auto"}, "required": {"type": "any"},
                "any": {"type": "any"}, "none": {"type": "none"}}.get(tc)
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "function":  # OpenAI forced-tool form
            return {"type": "tool", "name": (tc.get("function") or {}).get("name")}
        if t in ("auto", "any", "none", "tool"):  # already Anthropic-style
            return tc
    return None


def to_anthropic_request(req: dict[str, Any], model: str,
                         cfg: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        # /v1/messages requires max_tokens; OpenAI-route inbound may not have one.
        "max_tokens": req.get("max_tokens") or 4096,
        # puppetllm-private keys (`_openai_custom`, `_openai_detail`) never go on the wire:
        # the real Messages API rejects unknown fields.
        "messages": _oai.strip_private(req.get("messages") or []),
    }
    if req.get("system"):
        body["system"] = _oai.strip_private(req["system"])
    if req.get("tools"):
        tools = []
        for t in req["tools"]:
            if not isinstance(t, dict):
                continue
            if t.get("_openai_custom"):
                # OpenAI free-form tools have no Anthropic equivalent beyond their canonical
                # single-string schema; drop the OpenAI-only `format`.
                t = {k: v for k, v in t.items() if k != "format"}
            tools.append(_oai.strip_private(t))
        body["tools"] = tools
    params = req.get("params") or {}
    # thinking / output_config are forwarded: the server keeps thinking blocks (with the
    # upstream's real signatures) in injected responses, so a preserved-thinking tool loop
    # round-trips. (Mixing responders mid-conversation — fake signatures from a human
    # responder, then relay — would still 400 upstream.)
    for k in ("temperature", "top_p", "top_k", "stop_sequences", "thinking", "output_config",
              "cache_control", "inference_geo", "container", "context_management", "speed"):
        if k in params:
            body[k] = params[k]
    if not isinstance(body.get("output_config"), dict):
        body.pop("output_config", None)  # a malformed value must not crash the relay
    tier = params.get("service_tier")
    if isinstance(tier, str):
        if tier in _TIER_TO_ANTHROPIC:
            body["service_tier"] = _TIER_TO_ANTHROPIC[tier]
        else:
            _warn_dropped(f"service_tier={tier!r}")
    # OpenAI-inbound structured output / effort → the Anthropic equivalents.
    rf = params.get("response_format")
    if isinstance(rf, dict) and "output_config" not in body:
        if rf.get("type") == "json_schema" and isinstance(rf.get("json_schema"), dict) \
                and isinstance(rf["json_schema"].get("schema"), dict):
            body["output_config"] = {"format": {"type": "json_schema",
                                                "schema": rf["json_schema"]["schema"]}}
        elif rf.get("type") in ("json_object", "json_schema"):
            _warn_dropped(f"response_format.type={rf.get('type')} (without a schema)")
    effort = _effort_to_anthropic(params.get("reasoning_effort"))
    if effort is not None and "effort" not in (body.get("output_config") or {}):
        body.setdefault("output_config", {})
        body["output_config"] = {**body["output_config"], "effort": effort}
    elif params.get("reasoning_effort") is not None and effort is None:
        _warn_dropped(f"reasoning_effort={params.get('reasoning_effort')!r}")
    if body.get("output_config") == {}:
        del body["output_config"]
    # metadata: Anthropic only accepts {"user_id": str}. An OpenAI-inbound app may attach an
    # arbitrary metadata dict, which the real /v1/messages would 400 on — so forward only the
    # user_id subset and drop the rest.
    md = params.get("metadata")
    if isinstance(md, dict) and md.get("user_id") is not None:
        body["metadata"] = {"user_id": md["user_id"]}
    # OpenAI-inbound apps put stop sequences under "stop" (str or list).
    if "stop_sequences" not in body and "stop" in params:
        stop = params["stop"]
        body["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    tc = _tool_choice_to_anthropic(params.get("tool_choice"))
    if tc is not None:
        body["tool_choice"] = tc
    return body


def from_anthropic_response(resp: dict[str, Any]) -> dict[str, Any]:
    u = resp.get("usage")
    usage = None
    if isinstance(u, dict) and (u.get("input_tokens") or u.get("output_tokens")):
        usage = {k: max(0, _as_int(u.get(k))) for k in (
            "input_tokens", "output_tokens",
            "cache_creation_input_tokens", "cache_read_input_tokens")}
        # Forward the richer usage objects intact (cache TTL breakdown, thinking tokens,
        # server tool use, service tier, inference geo) — the server passes them through.
        for k in ("cache_creation", "output_tokens_details", "server_tool_use"):
            v = u.get(k)
            if isinstance(v, dict) and all(type(x) is int and x >= 0 for x in v.values()):
                usage[k] = v
        for k in ("service_tier", "inference_geo", "speed"):
            if isinstance(u.get(k), str):
                usage[k] = u[k]
    out: dict[str, Any] = {
        # thinking / redacted_thinking blocks are kept by the server (with the upstream's
        # signatures); unknown block types are filtered on inject
        "content": resp.get("content") or [],
        "stop_reason": resp.get("stop_reason"),
        "usage": usage,
    }
    if isinstance(resp.get("stop_sequence"), str):
        out["stop_sequence"] = resp["stop_sequence"]
    if isinstance(resp.get("stop_details"), dict):
        out["stop_details"] = resp["stop_details"]
    return out


# ── upstream call + error relaying ───────────────────────────────────


def _upstream_error_fields(status: int, body: Any) -> dict[str, Any]:
    """Extract error {status,type,message,code,param} from an upstream error body.

    status is clamped to the injectable [400, 599] range — a non-standard upstream status
    (e.g. 999) must not make the server reject our /_control/error injection (which would
    leave the pending unresolved and hot-loop the upstream).
    """
    status = min(max(int(status), 400), 599)
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):  # OpenAI / Anthropic envelope
        return {"status": status,
                "type": str(err.get("type") or "api_error"),
                "message": str(err.get("message") or "upstream error"),
                "code": err.get("code"), "param": err.get("param")}
    return {"status": status, "type": "api_error",
            "message": json.dumps(body, ensure_ascii=False)[:400]
            if isinstance(body, (dict, list)) else str(body)[:400]}


class Relay:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.model_map = parse_model_map(cfg.model_map)
        # Optional inbound-model claim filter: relay only claims pendings whose inbound model
        # matches one of these globs (others are left for a human / AI-agent responder).
        self.only = [g.strip() for g in (getattr(cfg, "only", None) or "").split(",")
                     if g.strip()]
        self.max_concurrency = max(0, int(getattr(cfg, "max_concurrency", 0) or 0))
        self.inflight: set[str] = set()
        # pids whose inject the server rejected (400). Bounded: quarantine only has to
        # bridge the gap until the server drops the (now error-resolved) pending, so old
        # entries are safe to evict once the set grows past a generous cap.
        self.quarantined: OrderedDict[str, None] = OrderedDict()
        self._warned_empty_model = False
        self._warned_foreign = False
        self._tasks: set[asyncio.Task] = set()  # strong refs (create_task is weakly held)
        self.handled = 0
        api_key = os.environ.get(cfg.api_key_env, "")
        if not api_key:
            _log(f"WARNING: env {cfg.api_key_env} is empty — upstream calls will "
                 f"likely fail with 401")
        if cfg.kind == "anthropic":
            headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
            self.upstream_url = cfg.target.rstrip("/") + "/v1/messages"
        else:
            headers = {"Authorization": f"Bearer {api_key}"}
            self.upstream_url = cfg.target.rstrip("/") + "/chat/completions"
        self.http = httpx.AsyncClient(timeout=cfg.timeout, headers=headers)
        self.ctl = httpx.AsyncClient(timeout=cfg.poll_timeout + 30,
                                     base_url=cfg.puppet)

    async def close(self) -> None:
        await self.http.aclose()
        await self.ctl.aclose()

    def _quarantine(self, pid: str) -> None:
        """Mark a pid as un-reclaimable, evicting the oldest entry past the cap."""
        self.quarantined[pid] = None
        while len(self.quarantined) > _QUARANTINE_MAX:
            self.quarantined.popitem(last=False)

    async def _post_ctl(self, path: str, body: dict[str, Any],
                        *, retries: int = 3) -> httpx.Response | None:
        """POST to the control plane with bounded backoff on TRANSPORT errors. Returns the
        response, or None if every attempt failed at the transport level. Retrying the POST
        (rather than the whole handler) is deliberate: we already hold the upstream response,
        so re-delivering it must never re-invoke — and re-bill — the upstream."""
        for attempt in range(retries):
            try:
                return await self.ctl.post(path, json=body)
            except httpx.HTTPError as e:
                if attempt + 1 < retries:
                    await asyncio.sleep(0.2 * (attempt + 1))
                else:
                    _log(f"inject {path} transport error after {retries} attempts: {e!r}")
        return None

    async def _inject(self, pid: str, path: str, payload: dict[str, Any]) -> bool:
        """POST a resolution to the control plane; always resolves the pending exactly once.

        Returns True once the pending is settled (200; 409/gone = someone else resolved it;
        or, on a payload rejection / undeliverable resolution, after a 502 fallback). We NEVER
        drop back to the run loop with the pending still claimable, because that would let it
        be re-forwarded to the paid upstream in a hot loop. Two failure modes are handled the
        same way — quarantine the pid so it can't be re-claimed, then best-effort a plain 502:
          - 400: the server rejected OUR payload (malformed usage/content/status).
          - transport failure: the resolution could not be delivered at all (control plane
            unreachable), even after retries."""
        r = await self._post_ctl(path, {"pending_id": pid, **payload})
        if r is not None:
            if r.status_code == 200:
                return True
            if r.status_code == 409 or (r.status_code == 400 and "no pending" in r.text):
                return True  # already resolved / gone — nothing more to do
            _log(f"{pid}: inject {path} rejected {r.status_code} {r.text[:120]} — "
                 f"quarantining + falling back to 502")
        else:
            _log(f"{pid}: inject {path} undeliverable (control plane unreachable) — "
                 f"quarantining + falling back to 502")
        # Either the server rejected our payload (400) or we could not deliver it at all.
        # In both cases: never re-forward to the upstream. Quarantine + fall back to a 502.
        self._quarantine(pid)
        if path != "/_control/error":
            await self._post_ctl("/_control/error", {
                "pending_id": pid, "status": 502, "type": "api_error",
                "message": "relay: could not deliver upstream response "
                           "(control plane rejected the payload or was unreachable)"})
        return True

    def _request_headers(self, req: dict[str, Any]) -> dict[str, str]:
        """Per-request upstream headers. Towards Anthropic, the betas the app asked for
        (`anthropic-beta`, captured as `params.anthropic_beta`) are forwarded — fast mode,
        compaction, context editing etc. are rejected upstream without them."""
        if self.cfg.kind != "anthropic":
            return {}
        betas = (req.get("params") or {}).get("anthropic_beta")
        if isinstance(betas, str):
            betas = [betas]
        if not isinstance(betas, list):
            return {}
        names = [str(b).strip() for b in betas if str(b).strip()]
        return {"anthropic-beta": ",".join(names)} if names else {}

    def _claims(self, item: dict[str, Any]) -> bool:
        """Whether the relay should claim this pending. With --only set, claim only when the
        inbound model matches one of the globs (leaving the rest for another responder);
        otherwise claim everything."""
        if not self.only:
            return True
        model = str((item.get("request") or {}).get("model") or "")
        return any(fnmatch.fnmatch(model, g) for g in self.only)

    async def handle(self, pid: str, pending: dict[str, Any]) -> None:
        try:
            # Everything that can raise lives inside the try, so the pending is ALWAYS
            # resolved (never a permanent ghost) — including request-shape access / mapping.
            req = pending["request"]
            inbound_model = req.get("model")
            model = map_model(inbound_model, self.cfg.model, self.model_map)
            if not model and not self._warned_empty_model:
                # Empty target model (no inbound model and no --model/--model-map). The
                # upstream will 400; warn once so the cause isn't mistaken for a relay bug.
                self._warned_empty_model = True
                _log("WARNING: resolved upstream model is empty — pass --model or "
                     "--model-map (the upstream will likely reject an empty model)")
            elif (model == str(inbound_model or "") and not self._warned_foreign
                    and looks_foreign(model, self.cfg.kind)):
                # Unmapped passthrough of a model that names the OTHER vendor (e.g. a
                # "claude-*" id going to an OpenAI endpoint). Almost always a missing
                # --model / --model-map; the upstream will reject it. Warn once.
                self._warned_foreign = True
                _log(f"WARNING: forwarding model {model!r} verbatim to a {self.cfg.kind} "
                     f"endpoint — it names a different vendor and will likely be rejected; "
                     f"pass --model or --model-map to route it")
            if self.cfg.kind == "anthropic":
                body = to_anthropic_request(req, model, self.cfg)
            else:
                body = to_openai_request(req, model, self.cfg)
            r = await self.http.post(self.upstream_url, json=body,
                                     headers=self._request_headers(req))
            try:
                data = r.json()
            except ValueError:
                data = r.text
            if r.status_code >= 400:
                fields = _upstream_error_fields(r.status_code, data)
                _log(f"{pid}: upstream {r.status_code} ({fields['type']}) — relaying")
                await self._inject(pid, "/_control/error", fields)
                return
            valid = (isinstance(data, dict) and (
                isinstance(data.get("choices"), list) and data["choices"]
                if self.cfg.kind != "anthropic"
                else isinstance(data.get("content"), list)))
            if not valid:
                # 2xx with a non-JSON body or an unexpected response shape (no choices /
                # no content) — treat as an upstream fault so the pending is always
                # resolved (never leave the app's request hanging) and the misbehaving
                # upstream surfaces as an error rather than a silent empty message.
                _log(f"{pid}: upstream 2xx with unexpected response shape — relaying as 502")
                await self._inject(pid, "/_control/error", {
                    "status": 502, "type": "api_error",
                    "message": f"relay: unexpected upstream response shape: "
                               f"{str(data)[:200]}"})
                return
            payload = (from_anthropic_response(data) if self.cfg.kind == "anthropic"
                       else from_openai_response(data))
            n_tools = sum(1 for b in payload["content"]
                          if isinstance(b, dict) and b.get("type") == "tool_use")
            out_tokens = payload["usage"]["output_tokens"] if payload["usage"] else "?"
            _log(f"{pid}: {req.get('provider')}:{req.get('model')} -> "
                 f"{self.cfg.kind}:{model} ok (stop={payload['stop_reason']}, "
                 f"tools={n_tools}, out_tokens={out_tokens})")
            await self._inject(pid, "/_control/respond", payload)
        except httpx.HTTPError as e:
            # Network-level failure to reach the upstream: surface as a 502 to the app.
            _log(f"{pid}: upstream unreachable: {e!r}")
            await self._inject(pid, "/_control/error", {
                "status": 502, "type": "api_error",
                "message": f"relay: upstream unreachable: {e!r}"[:400]})
        except Exception as e:
            # Any other failure (unexpected shape, mapping error, ...) must still resolve the
            # pending — otherwise the app's request hangs forever. Surface as a 502.
            _log(f"{pid}: relay error: {e!r}")
            try:
                await self._inject(pid, "/_control/error", {
                    "status": 502, "type": "api_error",
                    "message": f"relay: internal error: {e!r}"[:400]})
            except Exception:
                pass
        finally:
            self.inflight.discard(pid)
            self.handled += 1

    async def run(self) -> None:
        _log(f"puppet={self.cfg.puppet} -> {self.cfg.kind} upstream={self.upstream_url} "
             f"model={self.cfg.model or '(map/passthrough)'}")
        try:
            while not self.cfg.max_requests or self.handled < self.cfg.max_requests:
                try:
                    if not self.inflight:
                        # Idle: long-poll for the next arrival.
                        await self.ctl.get("/_control/wait_for_pending",
                                           params={"timeout": self.cfg.poll_timeout})
                    r = await self.ctl.get("/_control/pending")
                    pendings = r.json().get("pending") or []
                except (httpx.HTTPError, ValueError) as e:
                    # A transient control-plane blip must not crash the relay (which would
                    # strand every in-flight upstream call). Back off and retry.
                    _log(f"control plane error: {e!r}; retrying in 2s")
                    await asyncio.sleep(2)
                    continue
                claimed = 0
                skipped = 0  # visible pendings this pass we deliberately did not claim
                for item in pendings:
                    # Honor --max-requests as a hard cap: already-handled + in-flight must
                    # not exceed it, even when several pendings arrive in one poll.
                    if self.cfg.max_requests and \
                            self.handled + len(self.inflight) >= self.cfg.max_requests:
                        break
                    # --max-concurrency: don't fan out more simultaneous upstream calls than
                    # this. Remaining pendings stay visible and are picked up as slots free.
                    if self.max_concurrency and len(self.inflight) >= self.max_concurrency:
                        skipped += 1
                        break
                    pid = item.get("pending_id")
                    if not pid or pid in self.inflight or pid in self.quarantined:
                        continue
                    if not self._claims(item):
                        # Left for another responder (human / AI agent). Not an error.
                        skipped += 1
                        continue
                    self.inflight.add(pid)
                    claimed += 1
                    t = asyncio.create_task(self.handle(pid, item))
                    self._tasks.add(t)
                    t.add_done_callback(self._tasks.discard)
                if self.inflight or skipped or (pendings and not claimed):
                    # Either in-flight upstream calls keep entries pending (short-poll for
                    # follow-up turns), or we saw pendings but claimed none this pass (all
                    # quarantined / capped / filtered). Sleep either way: wait_for_pending
                    # returns instantly while any pending is visible, so skipping the sleep
                    # here would busy-spin the control plane at 100% CPU.
                    await asyncio.sleep(0.15)
            # Drain: let in-flight handlers finish before exiting (--max-requests mode).
            while self.inflight:
                await asyncio.sleep(0.05)
        finally:
            await self.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m puppetllm.relay",
        description="Relay responder: forward puppetllm pendings to a real LLM API "
                    "(cross-provider bridge)")
    p.add_argument("--puppet", default=os.environ.get("PUPPET_URL",
                   "http://127.0.0.1:8765"),
                   help="puppetllm control-plane URL (default: $PUPPET_URL or "
                        "http://127.0.0.1:8765)")
    p.add_argument("--kind", choices=("openai", "anthropic"), default="openai",
                   help="upstream API style: any OpenAI-compatible endpoint "
                        "(OpenAI/Grok/Groq/Ollama/...) or the native Anthropic API")
    p.add_argument("--target", default=None,
                   help="upstream base URL (default: official endpoint of --kind; "
                        "e.g. https://api.x.ai/v1 for Grok)")
    p.add_argument("--api-key-env", default=None,
                   help="env var holding the upstream API key (default: "
                        "OPENAI_API_KEY / ANTHROPIC_API_KEY per --kind)")
    p.add_argument("--model", default=None,
                   help="force this upstream model for every request")
    p.add_argument("--model-map", default=None,
                   help='glob mapping "pat=model,pat2=model2", e.g. '
                        '"claude-*=grok-3,gpt-*=grok-3" (first match wins; '
                        '--model takes precedence)')
    p.add_argument("--max-tokens-param",
                   choices=("auto", "max_tokens", "max_completion_tokens"), default="auto",
                   help="which field to send the token limit as on the OpenAI route "
                        "(default auto: max_completion_tokens when the target host is "
                        "api.openai.com — max_tokens is deprecated there and rejected by "
                        "o-series / reasoning models — and max_tokens for other "
                        "OpenAI-compatible backends)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="upstream request timeout in seconds (default 120)")
    p.add_argument("--poll-timeout", type=float, default=55.0,
                   help="control-plane long-poll timeout in seconds (default 55)")
    p.add_argument("--max-requests", type=int, default=0,
                   help="exit after handling N requests (0 = run forever)")
    p.add_argument("--max-concurrency", type=int, default=0,
                   help="cap simultaneous in-flight upstream calls (0 = unlimited); "
                        "protects against a burst fanning out and tripping upstream rate limits")
    p.add_argument("--only", default=None,
                   help='claim only pendings whose inbound model matches one of these globs '
                        '(comma-separated), leaving the rest for a human / AI-agent responder — '
                        'e.g. "gpt-*" to relay OpenAI-model requests while you answer the others '
                        'by hand')
    return p


def resolve_defaults(cfg: argparse.Namespace) -> argparse.Namespace:
    if cfg.target is None:
        cfg.target = ANTHROPIC_DEFAULT_URL if cfg.kind == "anthropic" else OPENAI_DEFAULT_URL
    if cfg.api_key_env is None:
        cfg.api_key_env = ("ANTHROPIC_API_KEY" if cfg.kind == "anthropic"
                           else "OPENAI_API_KEY")
    return cfg


def main(argv: list[str] | None = None) -> int:
    cfg = resolve_defaults(build_parser().parse_args(argv))
    try:
        asyncio.run(Relay(cfg).run())
    except KeyboardInterrupt:
        _log("interrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
