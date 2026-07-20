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

import httpx

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
            else json.dumps(b, ensure_ascii=False)
            for b in content
        )
    return json.dumps(content, ensure_ascii=False)


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


def to_openai_request(req: dict[str, Any], model: str,
                      cfg: Any = None) -> dict[str, Any]:
    """Canonical snapshot -> OpenAI-compatible chat.completions request body."""
    max_tokens_param = getattr(cfg, "max_tokens_param", "max_tokens") or "max_tokens"
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
                        tool_calls.append({
                            "id": str(b.get("id") or ""),
                            "type": "function",
                            "function": {
                                "name": str(b.get("name", "")),
                                "arguments": json.dumps(b.get("input") or {},
                                                        ensure_ascii=False),
                            },
                        })
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
                texts = []
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
                        texts.append(str(b.get("text", "")))
                    elif b.get("type") in ("image", "input_image", "document"):
                        # multimodal blocks aren't translated; keep the turn non-empty so
                        # the message sequence stays valid rather than silently vanishing.
                        texts.append("[non-text content omitted by relay]")
                if texts:
                    messages.append({"role": "user", "content": "".join(texts)})
            else:
                messages.append({"role": "user", "content": _text_of(content)})

    body: dict[str, Any] = {"model": model, "messages": messages}
    if req.get("max_tokens") is not None:
        body[max_tokens_param] = req["max_tokens"]
    tools = req.get("tools") or []
    if tools:
        body["tools"] = [{
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object"},
            },
        } for t in tools if isinstance(t, dict)]
    params = req.get("params") or {}
    for k in ("temperature", "top_p", "response_format", "parallel_tool_calls",
              "reasoning_effort"):
        if k in params:
            body[k] = params[k]
    if "stop_sequences" in params:
        body["stop"] = params["stop_sequences"]
    elif "stop" in params:
        body["stop"] = params["stop"]
    tc = _tool_choice_to_openai(params.get("tool_choice"))
    if tc is not None:
        body["tool_choice"] = tc
    return body


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
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except ValueError:
            args = {"_raw": raw}
        blocks.append({"type": "tool_use", "id": str(tc.get("id") or ""),
                       "name": str(fn.get("name") or ""), "input": args})

    finish = choice.get("finish_reason")
    stop_reason = _FINISH_TO_STOP.get(finish, _FINISH_FALLBACK) if finish else None
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
        usage = {
            # canonical (Anthropic) vocabulary: input_tokens excludes cached reads.
            # Clamp every value non-negative — a buggy/hostile upstream reporting negatives
            # would otherwise be rejected by the server and hang/loop the request.
            "input_tokens": max(0, _as_int(u.get("prompt_tokens")) - cached),
            "output_tokens": max(0, _as_int(u.get("completion_tokens"))),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
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
        "messages": req.get("messages") or [],
    }
    if req.get("system"):
        body["system"] = req["system"]
    if req.get("tools"):
        body["tools"] = req["tools"]
    params = req.get("params") or {}
    # NOTE: "thinking" is intentionally NOT forwarded — the server strips thinking blocks
    # from injected responses, so a preserved-thinking tool loop would 400 on the real API.
    for k in ("temperature", "top_p", "top_k", "stop_sequences"):
        if k in params:
            body[k] = params[k]
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
    return {
        # unknown block types (thinking etc.) are filtered by the server on inject
        "content": resp.get("content") or [],
        "stop_reason": resp.get("stop_reason"),
        "usage": usage,
    }


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
            r = await self.http.post(self.upstream_url, json=body)
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
    p.add_argument("--max-tokens-param", choices=("max_tokens", "max_completion_tokens"),
                   default="max_tokens",
                   help="which field to send the token limit as on the OpenAI route "
                        "(default max_tokens; use max_completion_tokens for OpenAI "
                        "reasoning / official gpt-5 endpoints that reject max_tokens)")
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
