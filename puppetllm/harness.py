"""Test-harness behaviours layered on the pending model (no provider knowledge here).

Everything in this module is driven from `fake_server` at two points:

- right after a request became a pending (`register_request`): the rate limiter may
  throttle it, a scenario rule may answer it, or the unmatched policy may answer it — all
  by resolving the pending's future with exactly the payload `/_control/respond` or
  `/_control/error` would have injected, so history, usage and every encoder see no
  difference between a scripted answer and a human one;
- while a route waits for that future (`await_resolution`): the pending timeout, the
  client-disconnect watch and the pre-response delay.

State:
  `Config`   — server-level settings (`/_control/config`), survive `/_control/clear`.
  `Harness`  — rules, rate-limit windows, the fake clock offset and the seeded RNG; the
               rules, windows and clock are reset by `/_control/clear` (a test's teardown).

Rule shape (validated in `fake_server` with the same validators the control endpoints
use, so an action body is exactly a `/_control/respond` or `/_control/error` body):

    {"id": "weather",                       # optional; generated when absent
     "match": {"provider": "anthropic",     # every key optional; all present keys must match
               "model": "claude-*",         # fnmatch glob against the snapshot's model id
               "tools": ["get_weather"],    # every listed tool must be offered by the request
               "has_tool_result": false,    # the last user turn carries tool_result blocks
               "last_user_text": "weather", # regex over the last user turn's text (tool_result text included)
               "turn": 3, "stream": true},
     "steps": [                             # consumed in order, one per matching request
        {"respond": {"content": [...]}, "delay_ms": 200},
        {"error": {"status": 429, "message": "slow down"}},
        {"respond": {"text": "final"}}],
     "repeat": false}                        # true: the last step answers forever

An exhausted rule (all steps consumed, no repeat) stops matching, so the next rule — or
the unmatched policy — takes over. Rules are tried in order; the first live match wins.

Latency fields (`delay_ms`, `ttfb_ms`, `chunk_delay_ms`, `jitter_ms`) may sit on any step,
on a `/_control/respond` / `/_control/error` body, or in `config.latency` as defaults.
"""

from __future__ import annotations

import fnmatch
import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

LATENCY_KEYS = ("delay_ms", "ttfb_ms", "chunk_delay_ms", "jitter_ms")
UNMATCHED_MODES = ("pending", "default", "error")
_MATCH_KEYS = ("provider", "model", "tools", "has_tool_result", "last_user_text", "turn",
               "stream")


@dataclass
class Config:
    # Seconds a pending may wait for an answer before the server answers for it (None:
    # forever — the interactive default).
    pending_timeout_s: float | None = None
    # What a request gets when no rule matches: keep it pending (human-in-the-loop),
    # answer with `default_response`, or refuse with `unmatched_error`.
    on_unmatched: str = "pending"
    default_response: dict[str, Any] | None = None
    unmatched_error: dict[str, Any] = field(default_factory=lambda: _error_default(
        500, "puppetllm: no rule matched this request"))
    # The error a pending gets when `pending_timeout_s` elapses.
    timeout_error: dict[str, Any] = field(default_factory=lambda: _error_default(
        504, "puppetllm: no response was injected in time"))
    # Defaults applied to every answer (a step / injection may override each key).
    latency: dict[str, int] = field(default_factory=dict)
    # {"rpm": N, "itpm": N, "otpm": N} over a sliding 60 s window; None = unlimited.
    rate_limit: dict[str, int] | None = None
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"pending_timeout_s": self.pending_timeout_s, "on_unmatched": self.on_unmatched,
                "default_response": _public(self.default_response),
                "unmatched_error": _public(self.unmatched_error),
                "timeout_error": _public(self.timeout_error), "latency": dict(self.latency),
                "rate_limit": self.rate_limit, "seed": self.seed}


def _error_default(status: int, message: str) -> dict[str, Any]:
    """A default error payload in the exact shape `/_control/error` compiles bodies to, so
    that `GET /_control/config` shows the same keys before and after a round trip."""
    return {"_inject_error": True, "status": status, "type": "api_error", "message": message,
            "code": None, "param": None, "headers": {}, "after_events": None,
            "original_status": None, "content": [], "_latency": {}}


def _public(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """A compiled payload without its private marker keys (`_inject_error`, `_latency`…),
    for display in `/_control/config`."""
    if payload is None:
        return None
    out = {k: v for k, v in payload.items() if not k.startswith("_")}
    lat = payload.get("_latency")
    if lat:
        out.update(lat)
    return out


@dataclass
class Rule:
    id: str
    match: dict[str, Any]
    steps: list[dict[str, Any]]
    # steps[i] compiled into the future payload `/_control/respond` / `/_control/error`
    # would have injected (built by fake_server with the endpoints' own validators).
    payloads: list[dict[str, Any]] = field(default_factory=list)
    repeat: bool = False
    consumed: int = 0
    matched: int = 0

    @property
    def exhausted(self) -> bool:
        return not self.repeat and self.consumed >= len(self.steps)

    def next_payload(self) -> dict[str, Any] | None:
        if self.exhausted:
            return None
        idx = min(self.consumed, len(self.steps) - 1)
        self.consumed += 1
        self.matched += 1
        return self.payloads[idx]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "match": self.match, "steps": self.steps, "repeat": self.repeat,
                "matched": self.matched, "consumed": min(self.consumed, len(self.steps)),
                "remaining": None if self.repeat else max(0, len(self.steps) - self.consumed),
                "exhausted": self.exhausted}


def _last_user_turn(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    for m in reversed(snapshot.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return m
    return None


def last_user_text(snapshot: dict[str, Any]) -> str:
    """The text of the last user turn: its string content, or its text blocks and the
    text carried by its tool_result blocks, joined by newlines."""
    m = _last_user_turn(snapshot)
    if m is None:
        return ""
    content = m.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            parts.append(str(b.get("text", "")))
        elif b.get("type") == "tool_result":
            inner = b.get("content")
            if isinstance(inner, str):
                parts.append(inner)
            elif isinstance(inner, list):
                parts.extend(str(x.get("text", "")) for x in inner
                             if isinstance(x, dict) and x.get("type") == "text")
    return "\n".join(parts)


def has_tool_result(snapshot: dict[str, Any]) -> bool:
    m = _last_user_turn(snapshot)
    content = (m or {}).get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def tool_names(snapshot: dict[str, Any]) -> set[str]:
    return {t["name"] for t in (snapshot.get("tools") or [])
            if isinstance(t, dict) and isinstance(t.get("name"), str)}


def rule_matches(match: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    """Every key present in `match` must hold; an empty match matches everything."""
    if "provider" in match and snapshot.get("provider") != match["provider"]:
        return False
    if "model" in match and not fnmatch.fnmatchcase(str(snapshot.get("model") or ""), match["model"]):
        return False
    if "tools" in match and not set(match["tools"]) <= tool_names(snapshot):
        return False
    if "has_tool_result" in match and has_tool_result(snapshot) != bool(match["has_tool_result"]):
        return False
    if "last_user_text" in match and not re.search(match["last_user_text"], last_user_text(snapshot)):
        return False
    if "turn" in match and snapshot.get("turn") != match["turn"]:
        return False
    if "stream" in match and bool(snapshot.get("stream")) != bool(match["stream"]):
        return False
    return True


def validate_match(match: Any) -> str | None:
    """Shape check for a rule's `match` (returns an error message or None)."""
    if not isinstance(match, dict):
        return "match must be an object"
    unknown = sorted(k for k in match if k not in _MATCH_KEYS)
    if unknown:
        return f"match: unknown key(s) {unknown}; allowed: {list(_MATCH_KEYS)}"
    if "provider" in match and match["provider"] not in ("anthropic", "bedrock", "openai"):
        return "match.provider must be anthropic, bedrock or openai"
    if "model" in match and not isinstance(match["model"], str):
        return "match.model must be a glob string"
    if "tools" in match and not (isinstance(match["tools"], list)
                                 and all(isinstance(t, str) and t for t in match["tools"])):
        return "match.tools must be a list of tool names"
    if "last_user_text" in match:
        if not isinstance(match["last_user_text"], str):
            return "match.last_user_text must be a regex string"
        try:
            re.compile(match["last_user_text"])
        except re.error as e:
            return f"match.last_user_text: invalid regex ({e})"
    if "turn" in match and (type(match["turn"]) is not int or match["turn"] < 1):
        return "match.turn must be a positive integer"
    for key in ("has_tool_result", "stream"):
        if key in match and not isinstance(match[key], bool):
            return f"match.{key} must be a boolean"
    return None


def validate_latency(obj: dict[str, Any], where: str = "") -> str | None:
    for key in LATENCY_KEYS:
        v = obj.get(key)
        if v is not None and (type(v) is not int or v < 0 or v > 600_000):
            return f"{where}{key} must be an integer number of milliseconds in [0, 600000]"
    return None


class Harness:
    def __init__(self) -> None:
        self.config = Config()
        self.rules: list[Rule] = []
        self.clock_offset: float = 0.0
        self._rng = random.Random()
        # [timestamp, input_tokens, output_tokens, counts_as_request] per record in the
        # sliding window: an admission (input charged at arrival) or an output-only record
        # (output charged when the answer is produced).
        self._window: list[list[float]] = []

    # ── lifecycle ──

    def reset(self) -> None:
        """What `/_control/clear` resets: scenario state, not server configuration."""
        self.rules.clear()
        self.clock_offset = 0.0
        self._window.clear()
        if self.config.seed is not None:
            self._rng.seed(self.config.seed)

    def reseed(self) -> None:
        self._rng = random.Random(self.config.seed)

    def now(self) -> float:
        """The fake clock: wall time plus whatever `/_control/clock/advance` added."""
        return time.time() + self.clock_offset

    # ── rules ──

    def set_rules(self, rules: list[Rule]) -> None:
        self.rules = list(rules)

    def add_rules(self, rules: list[Rule]) -> None:
        self.rules.extend(rules)

    def decide(self, snapshot: dict[str, Any]) -> tuple[Rule | None, dict[str, Any] | None]:
        """The first live rule whose match holds answers the request: (rule, payload) —
        or (None, None) when none matches."""
        for rule in self.rules:
            if rule.exhausted or not rule_matches(rule.match, snapshot):
                continue
            payload = rule.next_payload()
            if payload is not None:
                return rule, payload
        return None, None

    # ── latency ──

    def latency_for(self, *sources: dict[str, Any] | None) -> dict[str, int]:
        """Merge latency fields: later sources override earlier ones; config defaults
        underneath; `jitter_ms` becomes a seeded random addition to `delay_ms`."""
        merged: dict[str, int] = {k: v for k, v in self.config.latency.items() if k in LATENCY_KEYS}
        for src in sources:
            if not src:
                continue
            for k in LATENCY_KEYS:
                if src.get(k) is not None:
                    merged[k] = int(src[k])
        jitter = int(merged.pop("jitter_ms", 0) or 0)
        out = {k: int(merged.get(k, 0) or 0) for k in ("delay_ms", "ttfb_ms", "chunk_delay_ms")}
        if jitter:
            out["delay_ms"] += self._rng.randint(0, jitter)
        return out

    # ── rate limit ──

    def _prune_window(self, now: float) -> None:
        cutoff = now - 60.0
        while self._window and self._window[0][0] < cutoff:
            self._window.pop(0)

    def _usage(self) -> tuple[int, int, int]:
        """(requests, input tokens, output tokens) currently in the window."""
        return (int(sum(w[3] for w in self._window)),
                int(sum(w[1] for w in self._window)),
                int(sum(w[2] for w in self._window)))

    def _seconds_until(self, now: float, need: Any) -> float:
        """Seconds until enough of the oldest records have left the window for `need(
        requests_left, input_left, output_left)` to hold: 0 when it already does, the
        full minute when even an empty window would not (a request larger than the whole
        budget)."""
        req, inp, out = self._usage()
        if need(req, inp, out):
            return 0.0
        for rec in self._window:
            req -= int(rec[3])
            inp -= int(rec[1])
            out -= int(rec[2])
            if need(req, inp, out):
                return max(0.0, rec[0] + 60.0 - now)
        return 60.0

    def throttle(self, snapshot: dict[str, Any], now: float) -> dict[str, Any] | None:
        """Return the 429 payload when admitting this request would exceed the configured
        per-minute budget, else record it in the window and return None. A refused
        request consumes no budget (as with the real limiters), so `retry-after` — the
        time until enough admitted records leave the window — is achievable for any
        request that fits the budget at all. Input tokens are the request's total
        (cached prefixes included: admission runs before the cache is consulted)."""
        rl = self.config.rate_limit
        if not rl:
            return None
        self._prune_window(now)
        input_tokens = int(snapshot.get("input_tokens_total") or 0)
        requests, itpm, otpm = self._usage()
        rpm_l, itpm_l, otpm_l = rl.get("rpm"), rl.get("itpm"), rl.get("otpm")
        over: list[tuple[str, int, int]] = []  # (dimension, limit, current)
        if rpm_l is not None and requests + 1 > rpm_l:
            over.append(("requests", rpm_l, requests))
        if itpm_l is not None and itpm + input_tokens > itpm_l:
            over.append(("input-tokens", itpm_l, itpm))
        if otpm_l is not None and otpm > otpm_l:
            over.append(("output-tokens", otpm_l, otpm))
        if not over:
            self._window.append([now, float(input_tokens), 0.0, 1.0])
            return None
        wait = self._seconds_until(now, lambda r, i, o: (
            (rpm_l is None or r + 1 <= rpm_l) and (itpm_l is None or i + input_tokens <= itpm_l)
            and (otpm_l is None or o <= otpm_l)))
        retry_after = max(1, math.ceil(wait))
        oldest_leaves = max(0.0, self._window[0][0] + 60.0 - now) if self._window else 0.0
        over_dims = {dim for dim, _lim, _cur in over}
        dims = {
            "requests": (rpm_l, requests, lambda r, i, o: r + 1 <= (rpm_l or 0)),
            "input-tokens": (itpm_l, itpm, lambda r, i, o: i + input_tokens <= (itpm_l or 0)),
            "output-tokens": (otpm_l, otpm, lambda r, i, o: o <= (otpm_l or 0)),
        }
        quotas = {}
        for dim, (lim, cur, need) in dims.items():
            if lim is None:
                continue
            # An exhausted dimension resets when it can admit this request again; the
            # others when their oldest consumption leaves the window.
            reset_in = (max(1, math.ceil(self._seconds_until(now, need))) if dim in over_dims
                        else math.ceil(oldest_leaves))
            quotas[dim] = {"limit": lim, "remaining": max(0, lim - cur), "reset_in": reset_in,
                           "over": dim in over_dims}
        return {
            "status": 429, "type": "rate_limit_error",
            # `code` is what the OpenAI envelope carries for this case; the Anthropic and
            # Bedrock encoders ignore it.
            "code": "rate_limit_exceeded",
            "message": "puppetllm rate limit: " + ", ".join(
                f"{dim} {cur}/{lim} per minute" for dim, lim, cur in over),
            "headers": {"retry-after": str(retry_after),
                        **_ratelimit_headers(snapshot.get("provider"), quotas, now)},
            # A limiter's answer is immediate: no configured latency applies to it.
            "_latency": _NO_LATENCY,
        }

    def note_output(self, output_tokens: int, now: float) -> None:
        """Charge a finished request's output tokens to the window (for `otpm`) at the
        time they were produced — an output-only record, not a request."""
        if self.config.rate_limit and output_tokens > 0:
            self._prune_window(now)
            self._window.append([now, 0.0, float(output_tokens), 0.0])

    def rate_limit_snapshot(self, now: float) -> dict[str, Any] | None:
        rl = self.config.rate_limit
        if not rl:
            return None
        self._prune_window(now)
        requests, itpm, otpm = self._usage()
        return {**rl, "window_requests": requests, "window_input_tokens": itpm,
                "window_output_tokens": otpm}


# Latency settings that switch every delay off (immediate answers such as a limiter's 429
# or a timeout error, which no configured default should slow down).
_NO_LATENCY: dict[str, int] = {"delay_ms": 0, "ttfb_ms": 0, "chunk_delay_ms": 0, "jitter_ms": 0}


def _ratelimit_headers(provider: Any, quotas: dict[str, dict[str, Any]],
                       now: float) -> dict[str, str]:
    """The quota headers each vendor attaches, per configured dimension, so backoff code
    that reads them can be exercised (Bedrock exposes none). `quotas` maps
    requests / input-tokens / output-tokens → {limit, remaining, reset_in, over}."""
    out: dict[str, str] = {}

    def iso(seconds: int) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + seconds))

    token_dims = [quotas[d] for d in ("input-tokens", "output-tokens") if d in quotas]
    if provider == "anthropic":
        for dim, q in quotas.items():
            out[f"anthropic-ratelimit-{dim}-limit"] = str(q["limit"])
            out[f"anthropic-ratelimit-{dim}-remaining"] = str(q["remaining"])
            out[f"anthropic-ratelimit-{dim}-reset"] = iso(q["reset_in"])
        if token_dims:
            # The combined token quota the real API reports next to the split ones.
            out["anthropic-ratelimit-tokens-limit"] = str(sum(q["limit"] for q in token_dims))
            out["anthropic-ratelimit-tokens-remaining"] = str(sum(q["remaining"] for q in token_dims))
            out["anthropic-ratelimit-tokens-reset"] = iso(max(q["reset_in"] for q in token_dims))
    elif provider == "openai":
        if "requests" in quotas:
            q = quotas["requests"]
            out["x-ratelimit-limit-requests"] = str(q["limit"])
            out["x-ratelimit-remaining-requests"] = str(q["remaining"])
            out["x-ratelimit-reset-requests"] = f'{q["reset_in"]}s'
        if token_dims:
            # One token quota on this wire: the one that refused the request, else input.
            q = next((q for q in token_dims if q["over"]), token_dims[0])
            out["x-ratelimit-limit-tokens"] = str(q["limit"])
            out["x-ratelimit-remaining-tokens"] = str(q["remaining"])
            out["x-ratelimit-reset-tokens"] = f'{q["reset_in"]}s'
    return out


def new_rule_id() -> str:
    return f"rule_{uuid.uuid4().hex[:8]}"
