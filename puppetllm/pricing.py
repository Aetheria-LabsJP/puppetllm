"""Cost estimation (for debugging, approximate).

Formal spec: README.md

fake_server does not compute real token counts (that would require a real SDK path).
This module produces a "rough cost estimate" from a **heuristic approximation**
(≈4 chars/token) and a **published pricing table**. It favors "getting the order of
magnitude and the trend right" over precision. Note that it does not match actual billing.

Prices are in USD per 1M tokens (per Mtok). Anthropic / OpenAI public prices are
transcribed by hand from the official pricing pages.
On a price change, only this table needs updating.

- Anthropic: 5-minute cache write is 1.25x input, 1-hour cache write is 2x input, and
  cache read is 0.1x input (0.025x on Claude Fable 5.1 / Mythos 5.1).
- OpenAI: prompt caching is automatic (no cache_control needed) with no write surcharge →
  cache_write = input. cache_read uses the official "cached input" price; "pro" models have
  no cached-input discount → cache_read = input.
  Note: the proxy's OpenAI path does not perform pseudo-cache observation (always
  status "none"), so the cache columns are effectively unused, but we keep accurate values.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPrice:
    """per Mtok (USD)."""
    input: float
    output: float
    cache_write: float     # 5-minute cache write (Anthropic: input * 1.25 / OpenAI: no surcharge = input)
    cache_read: float      # cache read (Anthropic: input * 0.10 / OpenAI: official cached input price)
    cache_write_1h: float  # 1-hour cache write (Anthropic: input * 2 / OpenAI: n/a = input)


def _claude(inp: float, out: float, *, read: float | None = None) -> ModelPrice:
    return ModelPrice(input=inp, output=out, cache_write=round(inp * 1.25, 6),
                      cache_read=round(inp * 0.10, 6) if read is None else read,
                      cache_write_1h=round(inp * 2.0, 6))


def _openai(inp: float, out: float, *, cached: float | None = None) -> ModelPrice:
    return ModelPrice(input=inp, output=out, cache_write=inp,
                      cache_read=inp if cached is None else cached, cache_write_1h=inp)


# Per-model-family pricing. Keys are matched against the model id by substring (see
# resolve_family / _FAMILY_ORDER: more specific keys are tried first).
# The family name appears in Anthropic / Bedrock / OpenAI model ids alike, so this works commonly.
_FAMILY_PRICES: dict[str, ModelPrice] = {
    # ── Anthropic (Claude) ──────────────────────────────────────────
    # Fable / Mythos tier: $10/$50. Fable 5.1 / Mythos 5.1 cache reads are 0.025x ($0.25);
    # Fable 5 / Mythos 5 cache reads are the usual 0.1x ($1.00).
    "fable-5-1":  _claude(10.0, 50.0, read=0.25),
    "mythos-5-1": _claude(10.0, 50.0, read=0.25),
    "fable":      _claude(10.0, 50.0),
    "mythos":     _claude(10.0, 50.0),
    # Opus 4.1 / Opus 4 (retired on the Claude API, still served on Bedrock / Vertex): $15/$75.
    "opus-4-1":   _claude(15.0, 75.0),
    "opus-4-20":  _claude(15.0, 75.0),   # claude-opus-4-20250514
    "opus-4-0":   _claude(15.0, 75.0),   # claude-opus-4-0 alias
    # Opus 4.5 / 4.6 / 4.7 / 4.8 / 5: $5/$25.
    "opus":       _claude(5.0, 25.0),
    # Sonnet 5: $2/$10 (the introductory price became permanent on 2026-09-01).
    "sonnet-5":   _claude(2.0, 10.0),
    # Sonnet 4 / 4.5 / 4.6: $3/$15.
    "sonnet":     _claude(3.0, 15.0),
    # Haiku 3.5 (retired on the Claude API): $0.80/$4. Haiku 4.5: $1/$5.
    "3-5-haiku":  _claude(0.80, 4.0),
    "haiku":      _claude(1.0, 5.0),

    # ── OpenAI: current (official pricing page as of 2026-09) ──────────────────
    "gpt-6":         _openai(10.0, 50.0, cached=1.00),     # gpt-6-astra (flagship)
    "gpt-5.6-sol":   _openai(4.0, 20.0, cached=0.40),
    "gpt-5.6-terra": _openai(2.0, 12.0, cached=0.20),
    "gpt-5.6-luna":  _openai(0.20, 1.20, cached=0.02),
    "gpt-5.6":       _openai(4.0, 20.0, cached=0.40),      # alias of gpt-5.6-sol
    "gpt-5.5-pro":   _openai(30.0, 180.0),
    "gpt-5.4-pro":   _openai(30.0, 180.0),
    "gpt-5.5":       _openai(5.0, 30.0, cached=0.50),
    "gpt-5.4-mini":  _openai(0.75, 4.50, cached=0.075),
    "gpt-5.4-nano":  _openai(0.20, 1.25, cached=0.02),
    "gpt-5.4":       _openai(2.50, 15.0, cached=0.25),
    "gpt-5.2-pro":   _openai(21.0, 168.0),
    "gpt-5.2":       _openai(1.75, 14.0, cached=0.175),
    "gpt-5-pro":     _openai(15.0, 120.0),
    "chat-latest":   _openai(5.0, 30.0, cached=0.50),
    # codex models (gpt-5.3-codex etc.). Matched across generations by the "codex" substring.
    "codex":         _openai(1.75, 14.0, cached=0.175),

    # ── OpenAI: previous generations (still listed / still served) ─────────
    # gpt-5 also covers gpt-5.1 (same $1.25/$10).
    "gpt-5-mini":   _openai(0.25, 2.0, cached=0.025),
    "gpt-5-nano":   _openai(0.05, 0.40, cached=0.005),
    "gpt-5":        _openai(1.25, 10.0, cached=0.125),
    "gpt-4.1-mini": _openai(0.40, 1.60, cached=0.10),
    "gpt-4.1-nano": _openai(0.10, 0.40, cached=0.025),
    "gpt-4.1":      _openai(2.0, 8.0, cached=0.50),
    "gpt-4o-mini":  _openai(0.15, 0.60, cached=0.075),
    "gpt-4o":       _openai(2.50, 10.0, cached=1.25),
    "o4-mini":      _openai(1.10, 4.40, cached=0.275),
    "o3-pro":       _openai(20.0, 80.0),
    "o3-mini":      _openai(1.10, 4.40, cached=0.55),
    "o3":           _openai(2.0, 8.0, cached=0.50),
    "o1":           _openai(15.0, 60.0, cached=7.50),
}

# Model ids the `/v1/models` listings advertise (id -> display name). A catalogue, not a
# gate: any id an app sends is accepted, and `GET /v1/models/{id}` synthesizes an entry
# for an id that is not listed here.
KNOWN_MODELS: dict[str, list[tuple[str, str]]] = {
    "anthropic": [
        ("claude-fable-5-1", "Claude Fable 5.1"),
        ("claude-opus-5", "Claude Opus 5"),
        ("claude-sonnet-5", "Claude Sonnet 5"),
        ("claude-opus-4-8", "Claude Opus 4.8"),
        ("claude-opus-4-7", "Claude Opus 4.7"),
        ("claude-opus-4-6", "Claude Opus 4.6"),
        ("claude-opus-4-5", "Claude Opus 4.5"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
        ("claude-sonnet-4-5", "Claude Sonnet 4.5"),
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ],
    "openai": [
        ("gpt-6-astra", "GPT-6 Astra"),
        ("gpt-5.6-sol", "GPT-5.6 Sol"),
        ("gpt-5.6-terra", "GPT-5.6 Terra"),
        ("gpt-5.6-luna", "GPT-5.6 Luna"),
        ("gpt-5.5", "GPT-5.5"),
        ("gpt-5.4", "GPT-5.4"),
        ("gpt-5.4-mini", "GPT-5.4 mini"),
        ("gpt-5.4-nano", "GPT-5.4 nano"),
        ("gpt-5.3-codex", "GPT-5.3 Codex"),
        ("gpt-4.1", "GPT-4.1"),
        ("gpt-4o", "GPT-4o"),
        ("o3", "o3"),
        ("o4-mini", "o4-mini"),
    ],
}

# Fallback for unknown models (equivalent to sonnet). The is_estimate flag conveys "unknown".
# Even if unknown, a model containing "gpt" falls back to the current mid-tier gpt-5.4
# (sonnet-equivalent) (see resolve_family).
_DEFAULT_FAMILY = "sonnet"
_DEFAULT_GPT_FAMILY = "gpt-5.4"

# `inference_geo: "us"` (US-only inference) is billed at 1.1x on all token categories.
US_INFERENCE_MULTIPLIER = 1.1
# Fast mode (`speed: "fast"`, Opus 5 / 4.8 on the Claude API) is billed at 2x ($10 / $50).
FAST_MODE_MULTIPLIER = 2.0

# Heuristic: 1 token ≈ 4 characters. Japanese is ≈ 1-2 tokens per character, so this leans
# toward underestimation, but that is acceptable for "ballpark" use (see docstring).
_CHARS_PER_TOKEN = 4.0


# Family resolution priority (explicit). Held as a fixed tuple to avoid implicit dependence
# on dict iteration order. Uses substring matching (not startswith): Anthropic uses
# `claude-opus-4-...`, Bedrock uses `anthropic.claude-opus-...`, cross-region uses
# `us.anthropic.claude-opus-...`, so the family name is not at the start but carries a prefix,
# which startswith would miss.
# "More specific keys come first": generation-specific Claude keys (fable-5-1, opus-4-1,
# sonnet-5, 3-5-haiku) precede the generic family words; on the OpenAI side gpt-5.4-mini
# contains gpt-5.4 / gpt-5 too by substring, so order is mini/nano/pro → generation → plain.
# "codex" comes before the generation keys (gpt-5.4-codex etc. also resolve to codex-series
# pricing = cross-generation). The o-series keys ("o1"/"o3"/...) are short and prone to false
# matches, so they go last AND are matched on token boundaries (see _family_matches).
_FAMILY_ORDER: tuple[str, ...] = (
    "fable-5-1", "mythos-5-1", "fable", "mythos",
    "opus-4-1", "opus-4-20", "opus-4-0", "opus",
    "sonnet-5", "sonnet",
    "3-5-haiku", "haiku",
    "codex",
    "gpt-6",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6",
    "gpt-5.5-pro", "gpt-5.4-pro", "gpt-5.4-mini", "gpt-5.4-nano",
    "gpt-5.5", "gpt-5.4",
    "gpt-5.2-pro", "gpt-5.2", "gpt-5-pro",
    "gpt-5-mini", "gpt-5-nano", "gpt-5",
    "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1",
    "gpt-4o-mini", "gpt-4o",
    "chat-latest",
    "o4-mini", "o3-pro", "o3-mini", "o3", "o1",
)

_O_SERIES = re.compile(r"^o\d")


# The built-in Claude families (the entries of _FAMILY_ORDER ahead of the OpenAI ones).
_ANTHROPIC_FAMILIES: frozenset[str] = frozenset(_FAMILY_ORDER[:_FAMILY_ORDER.index("codex")])

def _family_matches(fam: str, m: str) -> bool:
    if _O_SERIES.match(fam):
        # "o1" / "o3" must not match inside e.g. "gpt-4o1..." — require a non-alphanumeric
        # (or string) boundary on both sides. "o3-mini" still matches "o3" (order handles it).
        return re.search(r"(?<![a-z0-9])" + re.escape(fam) + r"(?![a-z0-9])", m) is not None
    return fam in m


# Custom prices (`PUPPETLLM_PRICING_FILE` / `serve --pricing-file`): (model glob, price),
# consulted before the built-in families. The glob doubles as the family name in costs.
_CUSTOM_PRICES: list[tuple[str, ModelPrice]] = []
_PRICE_KEYS = ("input", "output", "cache_write", "cache_read", "cache_write_1h")
# Keeps every cost finite: price × tokens × the derived 2x cache-write rate × the 2.2x
# worst-case multipliers.
MAX_PRICE_PER_MTOK = 1_000_000.0


def load_pricing_file(path: str) -> int:
    """`{"<model glob>": {"input": $/MTok, "output": $/MTok, "cache_write"?, "cache_read"?,
    "cache_write_1h"?}, ...}`; unspecified cache rates follow the Anthropic ratios. Rates
    are finite JSON numbers from 0 to 1,000,000 $/MTok, `cache_read` at most `input`,
    no other keys. Globs are tried in file order (first match wins) against the whole
    model id, case-insensitively. An empty object clears the custom table. Returns the
    number of entries loaded."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected an object of model glob -> prices")
    loaded: list[tuple[str, ModelPrice]] = []
    for glob, spec in doc.items():
        if not isinstance(glob, str) or not glob or not isinstance(spec, dict):
            raise ValueError(f"{path}: {glob!r}: expected an object with input / output")
        if "input" not in spec or "output" not in spec:
            raise ValueError(f"{path}: {glob!r}: input and output ($ per MTok) are required")
        unknown = sorted(k for k in spec if k not in _PRICE_KEYS)
        if unknown:
            raise ValueError(f"{path}: {glob!r}: unknown key(s) {unknown}; allowed: {list(_PRICE_KEYS)}")

        def rate(key: str, default: float | None) -> float:
            v = spec.get(key)
            if v is None:
                if default is None:
                    raise ValueError(f"{path}: {glob!r}: {key} must be a number ($ per MTok)")
                return default + 0.0  # derived from a validated rate: in range by construction
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{path}: {glob!r}: {key} must be a number ($ per MTok)")
            try:
                x = float(v)
            except (ValueError, OverflowError):
                raise ValueError(f"{path}: {glob!r}: {key} must be a finite, non-negative number") from None
            if not math.isfinite(x) or not 0 <= x <= MAX_PRICE_PER_MTOK:
                raise ValueError(f"{path}: {glob!r}: {key} must be a number from 0 to "
                                 f"{MAX_PRICE_PER_MTOK:g} ($ per MTok)")
            return x + 0.0  # -0.0 → 0.0
        inp, out = rate("input", None), rate("output", None)
        price = ModelPrice(
            input=inp, output=out,
            cache_write=rate("cache_write", round(inp * 1.25, 6)),
            cache_read=rate("cache_read", round(inp * 0.1, 6)),
            cache_write_1h=rate("cache_write_1h", round(inp * 2.0, 6)))
        if price.cache_read > price.input:
            raise ValueError(f"{path}: {glob!r}: cache_read must not exceed input "
                             f"(a cache read cannot cost more than the tokens it saves)")
        loaded.append((glob, price))
    _CUSTOM_PRICES[:] = loaded
    return len(loaded)


def custom_family(model: Any) -> str | None:
    """The custom glob (from the pricing file) that prices this id, if any."""
    m = str(model or "")
    for glob, _price in _CUSTOM_PRICES:
        if fnmatch.fnmatchcase(m, glob) or fnmatch.fnmatchcase(m.lower(), glob.lower()):
            return glob
    return None


def resolve_family(model: Any) -> str:
    """Resolve the family name from a model id. Falls back to the default if unknown
    (gpt-5.4 if it contains gpt).

    model is assumed to be a str, but callers (register/usage computation) may pass through
    values from a malformed request, so non-str inputs are coerced with str() to avoid crashing.
    """
    custom = custom_family(model)
    if custom is not None:
        return custom
    m = str(model or "").lower()
    for fam in _FAMILY_ORDER:
        if _family_matches(fam, m):
            return fam
    if "gpt" in m:
        return _DEFAULT_GPT_FAMILY
    return _DEFAULT_FAMILY


def price_for(model: str | None) -> ModelPrice:
    """The price table for a model id: the first custom glob that matches it, else its
    built-in family's."""
    m = str(model or "")
    for glob, price in _CUSTOM_PRICES:
        if fnmatch.fnmatchcase(m, glob) or fnmatch.fnmatchcase(m.lower(), glob.lower()):
            return price
    return _FAMILY_PRICES[resolve_family(model)]


def is_fallback_family(model: Any) -> bool:
    """True when nothing prices this id specifically: no custom glob and no built-in family
    matched, so the default table (Sonnet, or gpt-5.4 for `gpt-*`) is used."""
    if custom_family(model) is not None:
        return False
    m = str(model or "").lower()
    return not any(_family_matches(fam, m) for fam in _FAMILY_ORDER)


def anthropic_billing(model: Any, inbound_anthropic: bool = False) -> bool:
    """Whether the fast-mode and US-inference multipliers apply to this id: a built-in
    Claude family, or an id that names Claude but matches no family; never a custom
    table, never an OpenAI family. An unrelated id priced by the default table counts
    only when `inbound_anthropic` says it arrived on an Anthropic-shaped route (a made-up
    id against the fake), not when a relay prices by an upstream nothing matches."""
    if custom_family(model) is not None:
        return False
    m = str(model or "").lower()
    if any(_family_matches(fam, m) for fam in _ANTHROPIC_FAMILIES):
        return True
    if not is_fallback_family(model):
        return False
    return inbound_anthropic or "claude" in m or "anthropic" in m


def approx_tokens(value: Any) -> int:
    """Approximate the token count of any JSON-like structure.

    Estimated as total stringified length / 4 (≈4 chars/token). dict/list are JSON-serialized.
    None/empty is 0. Not a real tokenizer, so this is a "ballpark".
    """
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    if not text:
        return 0
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


def estimate_output_tokens(content_blocks: Any) -> int:
    """Approximate token count of response content blocks."""
    return approx_tokens(content_blocks)


# Anthropic web search: $10 per 1,000 searches.
WEB_SEARCH_USD_PER_REQUEST = 0.01


def compute_cost(
    model: str | None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    inference_geo: str | None = None,
    speed: str | None = None,
    inbound_anthropic: bool = False,
    web_search_requests: int = 0,
) -> dict[str, Any]:
    """Return the estimated cost in USD. Each breakdown + total + resolved family + is_estimate.

    input_tokens is expected to be **the portion that did not hit the cache** (non-cached input).
    cache_write (5-minute TTL) / cache_write_1h (1-hour TTL) / cache_read are each billed at
    their own rate (matching Anthropic's usage semantics). `cache_write_usd` is the sum of
    both write kinds. `inference_geo="us"` (US-only inference, data residency) applies the
    official 1.1x multiplier to every token category; `speed="fast"` (fast mode) applies 2x.
    """
    fam = resolve_family(model)
    p = price_for(model)
    # Fast mode and US-only inference are Anthropic billing modifiers: they apply to the
    # Claude tables only, never to an OpenAI family, a custom table or the default table
    # an unrelated id falls back to (a relay pricing by its upstream must not inherit the
    # inbound request's `speed`).
    anthropic = anthropic_billing(model, inbound_anthropic)
    geo_mult = US_INFERENCE_MULTIPLIER if (anthropic and inference_geo == "us") else 1.0
    speed_mult = FAST_MODE_MULTIPLIER if (anthropic and speed == "fast") else 1.0
    per_mtok = 1_000_000.0 / (geo_mult * speed_mult)
    input_cost = input_tokens * p.input / per_mtok
    output_cost = output_tokens * p.output / per_mtok
    cache_write_cost = (cache_write_tokens * p.cache_write
                        + cache_write_1h_tokens * p.cache_write_1h) / per_mtok
    cache_read_cost = cache_read_tokens * p.cache_read / per_mtok
    # Server-side web search is billed per request on top of the tokens; web fetch is not.
    server_tool_cost = web_search_requests * WEB_SEARCH_USD_PER_REQUEST
    total = input_cost + output_cost + cache_write_cost + cache_read_cost + server_tool_cost
    return {
        "model_family": fam,
        "is_estimate": True,  # always an estimate (not a real tokenizer)
        "currency": "USD",
        "geo_multiplier": geo_mult,
        "speed_multiplier": speed_mult,
        "input_usd": round(input_cost, 6),
        "output_usd": round(output_cost, 6),
        "cache_write_usd": round(cache_write_cost, 6),
        "cache_read_usd": round(cache_read_cost, 6),
        "server_tool_usd": round(server_tool_cost, 6),
        "total_usd": round(total, 6),
    }


def cache_savings_usd(model: str | None, cache_read_tokens: int) -> float:
    """Approximate the amount saved by a cache read (= the difference vs. paying for the same
    tokens as normal input).

    The write-time surcharge (1.25x / 2x) is incurred separately, but here we only compute the
    rough savings per read. Used by stats to accumulate hit read tokens and show "roughly this
    much saved".
    """
    p = price_for(model)
    return round(cache_read_tokens * (p.input - p.cache_read) / 1_000_000.0, 6)
