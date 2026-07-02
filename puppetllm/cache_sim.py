"""Pseudo prompt cache (for debugging).

Formal spec: README.md

Purpose: observe via hashing "whether the app is sending requests structured so that prompt
caching can take effect". It holds no real cache (responses always go through control).

**Reproduces the behavior of real Anthropic** (correcting the first version's single-breakpoint,
exact-hash model):
- A single request can place **multiple breakpoints** (`cache_control: {type:"ephemeral"}`,
  up to 4). Each breakpoint **writes** "the prefix up to that point" as one cache entry.
- **read is prefix match**: if the longest previously written prefix matches the start of the
  current request, that portion is read (0.1x). Even at a position where the current request
  does not declare a breakpoint, it can be read if it was previously written (e.g. even if BP2
  is advanced to the end of each turn, the previous turn's prefix becomes a prefix match in the
  current turn and hits incrementally).
- **cache_control markers are not part of the cache key**: real behavior uses key = content
  tokens and ignores cache_control as directive metadata. Therefore the prefix hash/token are
  computed with **cache_control removed** (so even if the marker position moves each turn, it
  hits on content match).

render order = tools → system → messages (the evaluation order for prefix-match). TTL is 5
minutes (extended on read).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from . import pricing

DEFAULT_TTL_SECONDS = 300.0  # Anthropic 5min ephemeral cache

# Constants for real-behavior fidelity:
_LOOKBACK_BLOCKS = 20  # a breakpoint looks back at most 20 content blocks for a prior entry
_MAX_BREAKPOINTS = 4   # cache_control per request is at most 4 (real API returns 400 if exceeded)
# Minimum cacheable prefix per model family (tokens). Real behavior silently does not cache a
# prefix below this (cache_creation=0). Verified against Anthropic's prompt-caching docs
# (2026-07). NOTE 1: the real minimum varies BY GENERATION within a family and resolve_family
# cannot tell generations apart, so these are representative recent values:
#   Opus:   4.5/4.6 = 4096, 4.7 = 2048, 4.8 = 1024   -> 4096 (conservative)
#   Sonnet: 4.5 / 4.6 / 5 all = 1024                 -> 1024
#   Haiku:  4.5 = 4096  (the retired 3.5 was 2048)   -> 4096
# (Fable 5 / Mythos 5 are 512 but aren't matched by these family keys; they fall to the
# default below.) NOTE 2: this sim's tokens are approximate (≈4 chars/token, underestimating
# Japanese), so decisions near the threshold are inaccurate. The main goal is to prevent
# over-counting clearly small prefixes.
_MIN_CACHEABLE = {"opus": 4096, "sonnet": 1024, "haiku": 4096}
_DEFAULT_MIN_CACHEABLE = 1024


def _min_cacheable_for(model: Any) -> int:
    return _MIN_CACHEABLE.get(pricing.resolve_family(model), _DEFAULT_MIN_CACHEABLE)


def _has_cache_control(block: Any) -> bool:
    return isinstance(block, dict) and block.get("cache_control") is not None


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


def _hash_prefix(segs: list[dict[str, Any]], n: int) -> str:
    stripped = [_strip_cc(s) for s in segs[:n]]
    return hashlib.sha256(
        json.dumps(stripped, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _tokens_prefix(segs: list[dict[str, Any]], n: int) -> int:
    return pricing.approx_tokens([_strip_cc(s)["block"] for s in segs[:n]])


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
    _hash_memo: dict[int, str] = field(default_factory=dict, repr=False)
    _tokens_memo: dict[int, int] = field(default_factory=dict, repr=False)

    def prefix_hash(self, seg_count: int) -> str:
        h = self._hash_memo.get(seg_count)
        if h is None:
            h = self._hash_memo[seg_count] = _hash_prefix(self.segs, seg_count)
        return h

    def prefix_tokens(self, seg_count: int) -> int:
        t = self._tokens_memo.get(seg_count)
        if t is None:
            t = self._tokens_memo[seg_count] = _tokens_prefix(self.segs, seg_count)
        return t


def analyze_request(system: Any = None, tools: Any = None, messages: Any = None) -> RequestCache:
    """Return a RequestCache from a single segments decomposition (multi-breakpoint aware)."""
    segs = _segments(system, tools, messages)
    bps = [i for i, s in enumerate(segs) if _has_cache_control(s["block"])]
    return RequestCache(segs=segs, breakpoints=bps, total_tokens=_tokens_prefix(segs, len(segs)))


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


class CacheSimulator:
    """Holds an index of prefix hash → entry, and decides hit/miss via multi-breakpoint + prefix match.

    The current time is passed in from the caller as `now` (for testability).
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS, honor_ttl: bool = True,
                 min_cacheable_tokens: int | None = None):
        self.ttl_seconds = ttl_seconds
        self.honor_ttl = honor_ttl
        # None = per-model (_min_cacheable_for). Passing an explicit value uses it for all models (e.g. 0 for tests).
        self.min_cacheable_tokens = min_cacheable_tokens
        # hash -> {seg_count, tokens, created_at, last_seen, hits, misses, model}
        self.index: dict[str, dict[str, Any]] = {}

    def _alive(self, entry: dict[str, Any], now: float) -> bool:
        return (not self.honor_ttl) or (now - entry["created_at"]) <= self.ttl_seconds

    @staticmethod
    def _key(model: str | None, content_hash: str) -> str:
        # The real cache is per-model (switching models invalidates it). Folding model into the
        # key means the same content under a different model is a different entry = prevents a
        # false hit from another model / a model overwrite.
        return f"{model}\x00{content_hash}"

    def _min_tokens(self, model: str | None) -> int:
        return self.min_cacheable_tokens if self.min_cacheable_tokens is not None else _min_cacheable_for(model)

    def _prune(self, now: float) -> None:
        """Lazily GC dead entries older than 2x the TTL (unreadable; prevents unbounded growth)."""
        if not self.honor_ttl:
            return
        cutoff = 2 * self.ttl_seconds
        for h in [h for h, e in self.index.items() if now - e["created_at"] > cutoff]:
            del self.index[h]

    def observe(self, rc: RequestCache, model: str | None, now: float) -> dict[str, Any]:
        """Observe one request and return the cache decision (multi-breakpoint + prefix match +
        minimum threshold + 20-block lookback).

        Returns: {status, cache_read_tokens, cache_creation_tokens, prefix_hash, read_seg_count, breakpoints}
        status: "hit" | "miss" | "none" (no cache_control, or all breakpoints below the minimum threshold).
        """
        self._prune(now)
        min_tok = self._min_tokens(model)
        # Valid breakpoints = those whose prefix is at or above the minimum threshold. At most 4
        # (real-API limit; on overflow the deepest ones are kept).
        eff_bps = [bp for bp in rc.breakpoints if rc.prefix_tokens(bp + 1) >= min_tok][-_MAX_BREAKPOINTS:]
        if not eff_bps:
            # cache_control is present but all are below threshold → real behavior is non-cached (prevents over-report).
            return {"status": "none", "cache_read_tokens": 0, "cache_creation_tokens": 0,
                    "prefix_hash": None, "read_seg_count": 0, "breakpoints": len(rc.breakpoints)}

        n = len(rc.segs)
        # READ: existing entry seg_counts in descending order. Adopt the longest one satisfying
        # (a) a valid breakpoint within 20 blocks (lookback) (b) hash prefix match (c) alive.
        read_tokens = 0
        read_hash: str | None = None
        read_entry: dict[str, Any] | None = None
        read_seg = 0
        seg_counts = sorted(
            {e["seg_count"] for e in self.index.values() if 0 < e["seg_count"] <= n},
            reverse=True,
        )
        for sc in seg_counts:
            if not any(0 <= (bp + 1) - sc <= _LOOKBACK_BLOCKS for bp in eff_bps):
                continue  # not reachable within 20 blocks from any breakpoint → real behavior cannot find it
            h = self._key(model, rc.prefix_hash(sc))
            e = self.index.get(h)
            if e is not None and e["seg_count"] == sc and self._alive(e, now):
                read_tokens, read_hash, read_entry, read_seg = rc.prefix_tokens(sc), h, e, sc
                break  # descending order, so the first match is the longest

        # WRITE target = deepest valid breakpoint. creation = (deepest - read).
        deepest_sc = eff_bps[-1] + 1
        deepest_tokens = rc.prefix_tokens(deepest_sc)
        creation_tokens = max(0, deepest_tokens - read_tokens)

        if read_entry is not None:  # on read, extend TTL + increment hit count
            read_entry["created_at"] = now
            read_entry["last_seen"] = now
            read_entry["hits"] += 1
            read_entry["model"] = model

        # WRITE: register/update valid breakpoint prefixes in the index (extend TTL, per-model key).
        for bp in eff_bps:
            sc = bp + 1
            h = self._key(model, rc.prefix_hash(sc))
            ex = self.index.get(h)
            if ex is None:
                self.index[h] = {
                    "seg_count": sc, "tokens": rc.prefix_tokens(sc),
                    "created_at": now, "last_seen": now, "hits": 0, "model": model,
                }
            else:
                ex["created_at"] = now
                ex["last_seen"] = now
                ex["model"] = model

        return {
            "status": "hit" if read_tokens > 0 else "miss",
            "cache_read_tokens": read_tokens,
            "cache_creation_tokens": creation_tokens,
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
                "age_seconds": round(age, 2),
                "alive": (not self.honor_ttl) or age <= self.ttl_seconds,
            })
        out.sort(key=lambda x: (x["seg_count"], x["prefix_hash"]))
        return out

    def reset(self) -> None:
        self.index.clear()
