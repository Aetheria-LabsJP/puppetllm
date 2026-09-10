"""Pseudo prompt cache (for debugging).

Formal spec: README.md

Purpose: observe via hashing "whether the app is sending requests structured so that prompt
caching can take effect". It holds no real cache (responses always go through control).

**Reproduces the behavior of real Anthropic** (correcting the first version's single-breakpoint,
exact-hash model):
- A single request can place **multiple breakpoints** (`cache_control: {type:"ephemeral"}`,
  up to 4). Each breakpoint **writes** "the prefix up to that point" as one cache entry.
  A **top-level** `cache_control` (automatic caching) places one breakpoint on the last
  cacheable block, exactly like the real API.
- **read is prefix match**: if the longest previously written prefix matches the start of the
  current request, that portion is read (0.1x). Even at a position where the current request
  does not declare a breakpoint, it can be read if it was previously written (e.g. even if BP2
  is advanced to the end of each turn, the previous turn's prefix becomes a prefix match in the
  current turn and hits incrementally).
- **cache_control markers are not part of the cache key**: real behavior uses key = content
  tokens and ignores cache_control as directive metadata. Therefore the prefix hash/token are
  computed with **cache_control removed** (so even if the marker position moves each turn, it
  hits on content match).
- **Request parameters that the real API renders into the prompt** (effort, thinking config,
  tool_choice) are folded into the key of every prefix that reaches into `messages` — changing
  them invalidates the messages-level cache but not the tools/system prefixes, as documented.
- **TTL per breakpoint**: `cache_control.ttl` of `"5m"` (default) or `"1h"`; an entry lives
  for the TTL of the breakpoint that wrote it, refreshed on read. Cache creation is reported
  split by TTL (`ephemeral_5m` / `ephemeral_1h`) so usage and cost can mirror the real API.
- **20-block lookback**: a breakpoint looks back at most 20 positions for a prior entry, where a
  run of consecutive `tool_use` blocks (and a run of consecutive `tool_result` blocks) counts
  as a single position.

render order = tools → system → messages (the evaluation order for prefix-match).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from . import pricing

DEFAULT_TTL_SECONDS = 300.0        # Anthropic 5min ephemeral cache
DEFAULT_TTL_1H_SECONDS = 3600.0    # `cache_control: {"type": "ephemeral", "ttl": "1h"}`

# Constants for real-behavior fidelity:
_LOOKBACK_BLOCKS = 20  # a breakpoint looks back at most 20 positions for a prior entry
_MAX_BREAKPOINTS = 4   # cache_control per request is at most 4 (real-API limit)
# Minimum cacheable prefix per model (tokens). Real behavior silently does not cache a prefix
# below this (cache_creation=0). Per Anthropic's prompt-caching docs:
#   Fable 5 / 5.1, Mythos 5 / 5.1, Opus 5 = 512; Opus 4.8 = 1024; Opus 4.7 = 2048;
#   Opus 4.6 / 4.5 = 4096; Opus 4.1 / 4 = 1024; Sonnet 5 / 4.6 / 4.5 / 4 = 1024;
#   Haiku 4.5 = 4096; Haiku 3.5 = 2048.
# Matched by substring on the lowercased model id, most specific first (generation-aware —
# unlike pricing families, the minimum differs between Opus generations).
# NOTE: this sim's tokens are approximate (≈4 chars/token, underestimating Japanese), so
# decisions near the threshold are inaccurate. The main goal is to prevent over-counting
# clearly small prefixes.
_MIN_CACHEABLE_ORDER: tuple[tuple[str, int], ...] = (
    ("fable", 512), ("mythos", 512),
    ("opus-5", 512), ("opus-4-8", 1024), ("opus-4-7", 2048),
    ("opus-4-6", 4096), ("opus-4-5", 4096),
    ("opus-4-1", 1024), ("opus-4-20", 1024), ("opus-4-0", 1024),
    ("sonnet", 1024),
    ("haiku-4-5", 4096), ("3-5-haiku", 2048),
    ("opus", 4096), ("haiku", 4096),  # conservative fallbacks for unknown generations
)
_DEFAULT_MIN_CACHEABLE = 1024

# Block types that can never carry a cache breakpoint themselves.
_UNCACHEABLE_TYPES = ("thinking", "redacted_thinking")
_VALID_TTLS = ("5m", "1h")


class CacheControlError(ValueError):
    """A `cache_control` layout the real API rejects with 400 invalid_request_error."""


def _min_cacheable_for(model: Any) -> int:
    m = str(model or "").lower()
    for key, n in _MIN_CACHEABLE_ORDER:
        if key in m:
            return n
    return _DEFAULT_MIN_CACHEABLE


def _cache_control_of(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    cc = block.get("cache_control")
    if cc is None:
        return None
    _check_cache_control(cc)
    return cc


def _check_cache_control(cc: Any) -> None:
    """Validate one cache_control value the way the real API does."""
    if not isinstance(cc, dict):
        raise CacheControlError("cache_control: must be an object")
    if cc.get("type") != "ephemeral":
        raise CacheControlError("cache_control.type: must be 'ephemeral'")
    ttl = cc.get("ttl", "5m")
    if ttl not in _VALID_TTLS:
        raise CacheControlError(f"cache_control.ttl: must be one of {list(_VALID_TTLS)}")


def _has_cache_control(block: Any) -> bool:
    return isinstance(block, dict) and block.get("cache_control") is not None


def _ttl_of(cc: dict[str, Any] | None) -> str:
    return "1h" if isinstance(cc, dict) and cc.get("ttl") == "1h" else "5m"


def _is_cacheable_block(block: Any) -> bool:
    """Whether a block may hold a breakpoint (thinking blocks and empty text blocks cannot)."""
    if not isinstance(block, dict):
        return False
    t = block.get("type")
    if t in _UNCACHEABLE_TYPES:
        return False
    if t == "text" and not str(block.get("text") or ""):
        return False
    return True


def _segments(system: Any, tools: Any, messages: Any) -> list[dict[str, Any]]:
    """Lay out blocks in a single sequence in the order tools → system → messages
    (the prefix evaluation order).

    Handles system as either str / list, and message.content as either str / list.
    """
    segs: list[dict[str, Any]] = []

    for tool in tools or []:
        segs.append({"_kind": "tool", "role": None, "block": tool})

    if isinstance(system, list):
        for b in system:
            segs.append({"_kind": "system", "role": None, "block": b})
    elif isinstance(system, str) and system:
        segs.append({"_kind": "system", "role": None, "block": {"type": "text", "text": system}})

    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        role = msg.get("role") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for b in content:
                segs.append({"_kind": "message", "role": role, "block": b})
        elif isinstance(content, str):
            segs.append({"_kind": "message", "role": role, "block": {"type": "text", "text": content}})
    return segs


def _strip_cc(seg: dict[str, Any]) -> dict[str, Any]:
    """Return a content-only view of the segment with cache_control removed (for hash/token).

    Matches real behavior, which does not include cache_control in the cache key. This way, even
    if the BP position moves each turn, a prefix of the same content becomes the same byte
    sequence = the same hash and prefix-matches.
    """
    b = seg.get("block")
    if isinstance(b, dict) and b.get("cache_control") is not None:
        b = {k: v for k, v in b.items() if k != "cache_control"}
    return {"_kind": seg.get("_kind"), "role": seg.get("role"), "block": b}


def _hash_prefix(segs: list[dict[str, Any]], n: int, *salts: str | None) -> str:
    stripped = [_strip_cc(s) for s in segs[:n]]
    h = hashlib.sha256(json.dumps(stripped, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for salt in salts:
        if salt:
            h.update(b"\x00")
            h.update(salt.encode("utf-8"))
    return h.hexdigest()


def _tokens_prefix(segs: list[dict[str, Any]], n: int) -> int:
    if n <= 0:
        return 0  # approx_tokens([]) would report 1 for the empty JSON list
    return pricing.approx_tokens([_strip_cc(s)["block"] for s in segs[:n]])


def _positions(segs: list[dict[str, Any]]) -> list[int]:
    """Lookback position of each segment: consecutive tool_use blocks collapse into one
    position, and so do consecutive tool_result blocks (real-API lookback rule)."""
    pos: list[int] = []
    cur = -1
    prev_type: Any = object()
    for s in segs:
        b = s.get("block")
        t = b.get("type") if isinstance(b, dict) else None
        if not (t in ("tool_use", "tool_result") and t == prev_type):
            cur += 1
        pos.append(cur)
        prev_type = t
    return pos


# Models on which thinking is ON by default (omitting `thinking` runs adaptive): an explicit
# `{"type": "adaptive"}` equals omission there, while `{"type": "disabled"}` is a change.
# Every other current model defaults to no thinking.
_THINKING_ON_BY_DEFAULT = ("fable", "mythos", "opus-5", "sonnet-5")


def thinking_default_for(model: Any) -> dict[str, Any]:
    m = str(model or "").lower()
    if any(k in m for k in _THINKING_ON_BY_DEFAULT):
        return {"type": "adaptive"}
    return {"type": "disabled"}


def messages_salt(params: dict[str, Any] | None, model: Any = None) -> str | None:
    """Derive the key salt for messages-level prefixes from request parameters that the real
    API renders into the prompt. Modeled as messages-only (the official invalidation table
    marks tools / system as *model-specific* for thinking and effort; puppetllm does not
    model which models render the configuration ahead of them).

    Explicit defaults equal omission (the real API renders the resolved value): `effort:
    "high"`, `tool_choice: auto` (no other options), and the model's default thinking mode
    (`adaptive` on Opus 5 / Sonnet 5 / Fable / Mythos, `disabled` elsewhere; `display` is
    ignored) produce no salt.
    """
    if not params:
        return None
    parts: dict[str, Any] = {}
    oc = params.get("output_config")
    effort = oc.get("effort") if isinstance(oc, dict) else None
    if effort is None:
        effort = params.get("reasoning_effort")  # OpenAI-inbound spelling
    if effort and effort != "high":
        parts["effort"] = effort
    th = params.get("thinking")
    if isinstance(th, dict):
        # `display` is a visibility setting, not rendered into the prompt — only the mode
        # and budget can invalidate the cache.
        th_norm = {k: th.get(k) for k in ("type", "budget_tokens") if th.get(k) is not None}
        if th_norm != thinking_default_for(model):
            parts["thinking"] = th_norm
    tc = params.get("tool_choice")
    if tc is not None:
        tc_norm = {"type": "auto"} if tc == "auto" else tc
        if isinstance(tc_norm, dict) and tc_norm.get("disable_parallel_tool_use") is False:
            tc_norm = {k: v for k, v in tc_norm.items() if k != "disable_parallel_tool_use"}
        if tc_norm != {"type": "auto"}:
            parts["tool_choice"] = tc_norm
    if not parts:
        return None
    return json.dumps(parts, sort_keys=True, ensure_ascii=False)


def system_salt(params: dict[str, Any] | None) -> str | None:
    """Salt for prefixes that reach into `system` (and therefore messages): parameters the
    real API renders ahead of the system prompt. Currently `speed` (fast mode) — switching
    speed invalidates the system and messages caches but not the tools prefix."""
    if not params:
        return None
    speed = params.get("speed")
    if isinstance(speed, str) and speed not in ("", "standard"):
        return json.dumps({"speed": speed})
    return None


@dataclass
class RequestCache:
    """Cache analysis result for one request (segments + breakpoint positions + total tokens).

    prefix_hash / prefix_tokens are memoized per seg_count — observe() calls them for each
    candidate seg_count, so naively JSON-serializing every time would spend O(candidates ×
    payload) of synchronous CPU inside state.lock on a long conversation with a grown index.
    """
    segs: list[dict[str, Any]]
    breakpoints: list[int]   # segment indices that have cache_control (ascending). Each +1 is the prefix seg_count.
    total_tokens: int
    ttls: dict[int, str] = field(default_factory=dict)   # breakpoint segment index → "5m" | "1h"
    messages_start: int = 0  # index of the first `messages` segment (prefixes beyond it get the salt)
    system_start: int = 0    # index of the first `system` segment (prefixes beyond it get the system salt)
    salt: str | None = None
    sys_salt: str | None = None
    top_level: bool = False  # a breakpoint was synthesized from top-level cache_control
    _hash_memo: dict[int, str] = field(default_factory=dict, repr=False)
    _tokens_memo: dict[int, int] = field(default_factory=dict, repr=False)
    _positions_memo: list[int] | None = field(default=None, repr=False)

    def prefix_hash(self, seg_count: int) -> str:
        h = self._hash_memo.get(seg_count)
        if h is None:
            sys_salt = self.sys_salt if seg_count > self.system_start else None
            salt = self.salt if seg_count > self.messages_start else None
            h = self._hash_memo[seg_count] = _hash_prefix(self.segs, seg_count, sys_salt, salt)
        return h

    def prefix_tokens(self, seg_count: int) -> int:
        t = self._tokens_memo.get(seg_count)
        if t is None:
            t = self._tokens_memo[seg_count] = _tokens_prefix(self.segs, seg_count)
        return t

    def positions(self) -> list[int]:
        if self._positions_memo is None:
            self._positions_memo = _positions(self.segs)
        return self._positions_memo

    def ttl_at(self, bp: int) -> str:
        return self.ttls.get(bp, "5m")


def analyze_request(system: Any = None, tools: Any = None, messages: Any = None,
                    *, top_level_cache_control: Any = None,
                    params: dict[str, Any] | None = None,
                    model: Any = None) -> RequestCache:
    """Return a RequestCache from a single segments decomposition (multi-breakpoint aware).

    `top_level_cache_control` (the request's top-level `cache_control`) synthesizes a
    breakpoint on the last cacheable block, like the real API's automatic caching. `params`
    (tool_choice / thinking / output_config / speed) feeds the key salts.

    Raises CacheControlError for layouts the real API rejects with 400: malformed markers or
    unknown TTLs, more than 4 explicit breakpoints, a marker on a thinking block, a 1h
    breakpoint after a 5m one ("entries with longer TTL must appear before shorter TTLs"),
    top-level caching when 4 explicit breakpoints already exist, and a top-level TTL that
    disagrees with an explicit breakpoint on the last block.
    """
    segs = _segments(system, tools, messages)
    system_start = sum(1 for s in segs if s["_kind"] == "tool")
    messages_start = sum(1 for s in segs if s["_kind"] != "message")
    bps: list[int] = []
    ttls: dict[int, str] = {}
    explicit_markers = 0
    for i, s in enumerate(segs):
        cc = _cache_control_of(s["block"])
        if cc is None:
            continue
        if isinstance(s["block"], dict) and s["block"].get("type") in _UNCACHEABLE_TYPES:
            raise CacheControlError("cache_control: thinking blocks cannot carry a cache breakpoint")
        explicit_markers += 1  # every marker counts toward the limit, cacheable or not
        if _is_cacheable_block(s["block"]):
            bps.append(i)
            ttls[i] = _ttl_of(cc)
    if explicit_markers > _MAX_BREAKPOINTS:
        raise CacheControlError(
            f"cache_control: at most {_MAX_BREAKPOINTS} explicit breakpoints per request")
    top_level = False
    if top_level_cache_control is not None:
        _check_cache_control(top_level_cache_control)
        for i in range(len(segs) - 1, -1, -1):
            if _is_cacheable_block(segs[i]["block"]):
                if i in ttls:
                    if ttls[i] != _ttl_of(top_level_cache_control):
                        raise CacheControlError(
                            "cache_control: the top-level ttl must match the explicit "
                            "breakpoint on the last cacheable block")
                else:
                    if len(bps) >= _MAX_BREAKPOINTS:
                        raise CacheControlError(
                            f"cache_control: top-level caching needs a free breakpoint slot "
                            f"(at most {_MAX_BREAKPOINTS} per request)")
                    bps.append(i)
                    bps.sort()
                    ttls[i] = _ttl_of(top_level_cache_control)
                    top_level = True
                break
    # TTL ordering applies to the synthesized breakpoint too: "entries with longer TTL must
    # appear before shorter TTLs".
    seen_5m = False
    for bp in bps:
        if ttls[bp] == "5m":
            seen_5m = True
        elif seen_5m:
            raise CacheControlError(
                "cache_control: breakpoints with a 1h ttl must appear before 5m breakpoints")
    return RequestCache(segs=segs, breakpoints=bps, total_tokens=_tokens_prefix(segs, len(segs)),
                        ttls=ttls, messages_start=messages_start, system_start=system_start,
                        salt=messages_salt(params, model), sys_salt=system_salt(params),
                        top_level=top_level)


# ── Backward compatibility: for legacy callers / unit tests that want a single prefix ──────────────


@dataclass
class CachePrefix:
    hash: str
    tokens: int
    breakpoints: int
    segments: int


def extract_cache_prefix(
    system: Any = None, tools: Any = None, messages: Any = None
) -> CachePrefix | None:
    """Extract the cacheable prefix (up to the deepest breakpoint). None if there is no cache_control."""
    rc = analyze_request(system, tools, messages)
    if not rc.breakpoints:
        return None
    deepest = rc.breakpoints[-1] + 1
    return CachePrefix(
        hash=rc.prefix_hash(deepest), tokens=rc.prefix_tokens(deepest),
        breakpoints=len(rc.breakpoints), segments=deepest,
    )


def _none_result(rc: RequestCache) -> dict[str, Any]:
    return {"status": "none", "cache_read_tokens": 0, "cache_creation_tokens": 0,
            "cache_creation_5m_tokens": 0, "cache_creation_1h_tokens": 0,
            "prefix_hash": None, "read_seg_count": 0, "breakpoints": len(rc.breakpoints)}


class CacheSimulator:
    """Holds an index of prefix hash → entry, and decides hit/miss via multi-breakpoint + prefix match.

    The current time is passed in from the caller as `now` (for testability).
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS, honor_ttl: bool = True,
                 min_cacheable_tokens: int | None = None,
                 ttl_1h_seconds: float = DEFAULT_TTL_1H_SECONDS):
        self.ttl_seconds = ttl_seconds          # "5m" breakpoints
        self.ttl_1h_seconds = ttl_1h_seconds    # "1h" breakpoints
        self.honor_ttl = honor_ttl
        # None = per-model (_min_cacheable_for). Passing an explicit value uses it for all models (e.g. 0 for tests).
        self.min_cacheable_tokens = min_cacheable_tokens
        # hash -> {seg_count, tokens, created_at, last_seen, hits, model, ttl, ttl_seconds}
        self.index: dict[str, dict[str, Any]] = {}

    def _ttl_seconds_for(self, ttl: str) -> float:
        return self.ttl_1h_seconds if ttl == "1h" else self.ttl_seconds

    def _alive(self, entry: dict[str, Any], now: float) -> bool:
        return (not self.honor_ttl) or (now - entry["created_at"]) <= entry.get("ttl_seconds", self.ttl_seconds)

    @staticmethod
    def _key(model: str | None, content_hash: str) -> str:
        # The real cache is per-model (switching models invalidates it). Folding model into the
        # key means the same content under a different model is a different entry = prevents a
        # false hit from another model / a model overwrite.
        return f"{model}\x00{content_hash}"

    def _min_tokens(self, model: str | None) -> int:
        return self.min_cacheable_tokens if self.min_cacheable_tokens is not None else _min_cacheable_for(model)

    def _prune(self, now: float) -> None:
        """Lazily GC dead entries older than 2x their TTL (unreadable; prevents unbounded growth)."""
        if not self.honor_ttl:
            return
        dead = [h for h, e in self.index.items()
                if now - e["created_at"] > 2 * e.get("ttl_seconds", self.ttl_seconds)]
        for h in dead:
            del self.index[h]

    def observe(self, rc: RequestCache, model: str | None, now: float) -> dict[str, Any]:
        """Observe one request and return the cache decision (multi-breakpoint + prefix match +
        minimum threshold + 20-block lookback).

        Returns: {status, cache_read_tokens, cache_creation_tokens, cache_creation_5m_tokens,
                  cache_creation_1h_tokens, prefix_hash, read_seg_count, breakpoints}
        status: "hit" | "miss" | "none" (no cache_control, or all breakpoints below the minimum threshold).
        """
        self._prune(now)
        min_tok = self._min_tokens(model)
        # Valid breakpoints = those whose prefix is at or above the minimum threshold. At most 4
        # (real-API limit; on overflow the deepest ones are kept).
        eff_bps = [bp for bp in rc.breakpoints if rc.prefix_tokens(bp + 1) >= min_tok][-_MAX_BREAKPOINTS:]
        if not eff_bps:
            # cache_control is present but all are below threshold → real behavior is non-cached (prevents over-report).
            return _none_result(rc)

        n = len(rc.segs)
        pos = rc.positions()
        # READ: existing entry seg_counts in descending order. Adopt the longest one satisfying
        # (a) a valid breakpoint within 20 positions (lookback) (b) hash prefix match (c) alive.
        read_tokens = 0
        read_hash: str | None = None
        read_entry: dict[str, Any] | None = None
        read_seg = 0
        seg_counts = sorted(
            {e["seg_count"] for e in self.index.values() if 0 < e["seg_count"] <= n},
            reverse=True,
        )
        for sc in seg_counts:
            # An entry is readable only from a breakpoint at or after its end (`sc <= bp + 1`)
            # and within 20 lookback positions of it — the position check alone would let a
            # breakpoint early in a collapsed tool run "read" content that comes after it.
            if not any(sc <= bp + 1 and pos[bp] - pos[sc - 1] <= _LOOKBACK_BLOCKS for bp in eff_bps):
                continue  # not reachable within 20 positions from any breakpoint → real behavior cannot find it
            h = self._key(model, rc.prefix_hash(sc))
            e = self.index.get(h)
            if e is not None and e["seg_count"] == sc and self._alive(e, now):
                read_tokens, read_hash, read_entry, read_seg = rc.prefix_tokens(sc), h, e, sc
                break  # descending order, so the first match is the longest

        # WRITE target = deepest valid breakpoint. creation = (deepest - read), split by the TTL
        # of the breakpoint that writes each stretch (real billing: 1h writes cost 2x, 5m 1.25x).
        deepest_sc = eff_bps[-1] + 1
        deepest_tokens = rc.prefix_tokens(deepest_sc)
        creation_tokens = max(0, deepest_tokens - read_tokens)
        creation_by_ttl = {"5m": 0, "1h": 0}
        prev_sc = read_seg
        for bp in eff_bps:
            sc = bp + 1
            if sc <= prev_sc:
                continue
            creation_by_ttl[rc.ttl_at(bp)] += max(0, rc.prefix_tokens(sc) - rc.prefix_tokens(prev_sc))
            prev_sc = sc

        if read_entry is not None:  # on read, extend TTL + increment hit count
            read_entry["created_at"] = now
            read_entry["last_seen"] = now
            read_entry["hits"] += 1
            read_entry["model"] = model

        # WRITE: register/update valid breakpoint prefixes in the index (extend TTL, per-model key).
        for bp in eff_bps:
            sc = bp + 1
            ttl = rc.ttl_at(bp)
            h = self._key(model, rc.prefix_hash(sc))
            ex = self.index.get(h)
            if ex is None:
                self.index[h] = {
                    "seg_count": sc, "tokens": rc.prefix_tokens(sc),
                    "created_at": now, "last_seen": now, "hits": 0, "model": model,
                    "ttl": ttl, "ttl_seconds": self._ttl_seconds_for(ttl),
                }
            else:
                ex["created_at"] = now
                ex["last_seen"] = now
                ex["model"] = model
                ex["ttl"] = ttl
                ex["ttl_seconds"] = self._ttl_seconds_for(ttl)

        return {
            "status": "hit" if read_tokens > 0 else "miss",
            "cache_read_tokens": read_tokens,
            "cache_creation_tokens": creation_tokens,
            "cache_creation_5m_tokens": creation_by_ttl["5m"],
            "cache_creation_1h_tokens": creation_by_ttl["1h"],
            "prefix_hash": read_hash or self._key(model, rc.prefix_hash(deepest_sc)),
            "read_seg_count": read_seg,
            "breakpoints": len(rc.breakpoints),
        }

    def entries(self, now: float) -> list[dict[str, Any]]:
        out = []
        for h, e in self.index.items():
            age = now - e["created_at"]
            out.append({
                "prefix_hash": h,
                "seg_count": e["seg_count"],
                "tokens": e["tokens"],
                "hits": e["hits"],
                "model": e.get("model"),
                "ttl": e.get("ttl", "5m"),
                "age_seconds": round(age, 2),
                "alive": self._alive(e, now),
            })
        out.sort(key=lambda x: (x["seg_count"], x["prefix_hash"]))
        return out

    def reset(self) -> None:
        self.index.clear()
