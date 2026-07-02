"""Cost estimation (for debugging, approximate).

Formal spec: README.md

fake_server does not compute real token counts (that would require a real SDK path).
This module produces a "rough cost estimate" from a **heuristic approximation**
(≈4 chars/token) and a **published pricing table**. It favors "getting the order of
magnitude and the trend right" over precision. Note that it does not match actual billing.

Prices are in USD per 1M tokens (per Mtok). Anthropic / OpenAI public prices are
transcribed by hand. On a price change, only this table needs updating.

- Anthropic: cache_write is 1.25x input, cache_read is 0.1x input — values following
  the 5-minute cache rule (stated here as a memo).
- OpenAI: prompt caching is automatic (no cache_control needed) with no write surcharge →
  cache_write = input. cache_read uses the official "cached input" price.
  Note: the proxy's OpenAI path does not perform pseudo-cache observation (always
  status "none"), so the cache columns are effectively unused, but we keep accurate values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPrice:
    """per Mtok (USD)."""
    input: float
    output: float
    cache_write: float  # cache write (Anthropic: input * 1.25 / OpenAI: no surcharge = input)
    cache_read: float   # cache read (Anthropic: input * 0.10 / OpenAI: official cached input price)


# Per-model-family pricing. Keys are matched against the model id by substring.
# The family name appears in Anthropic / Bedrock / OpenAI model ids alike, so this works commonly.
_FAMILY_PRICES: dict[str, ModelPrice] = {
    # ── Anthropic (Claude) ──────────────────────────────────────────
    # Claude Opus (4.x = 4.6/4.7/4.8). $5/$25 (note: old Claude 3 Opus / Opus 4.0-4.1 were
    # $15/$75. Since the model id does not distinguish generations, we use the latest 4.x price).
    "opus":   ModelPrice(input=5.0,  output=25.0, cache_write=6.25,  cache_read=0.50),
    # Claude Sonnet (4.x)
    "sonnet": ModelPrice(input=3.0,  output=15.0, cache_write=3.75,  cache_read=0.30),
    # Claude Haiku (4.5). Note the 3.5 generation was cheaper at $0.80/$4 (since the model id
    # does not distinguish generations, we use the latest 4.5 price).
    "haiku":  ModelPrice(input=1.0,  output=5.0,  cache_write=1.25,  cache_read=0.10),

    # ── OpenAI: current (official pricing page as of 2026-07) ──────────────────
    # pro models have no cached-input discount (official notation "—") → cache_read = input.
    "gpt-5.5-pro":  ModelPrice(input=30.0, output=180.0, cache_write=30.0, cache_read=30.0),
    "gpt-5.4-pro":  ModelPrice(input=30.0, output=180.0, cache_write=30.0, cache_read=30.0),
    "gpt-5.5":      ModelPrice(input=5.0,  output=30.0,  cache_write=5.0,  cache_read=0.50),
    "gpt-5.4-mini": ModelPrice(input=0.75, output=4.50,  cache_write=0.75, cache_read=0.075),
    "gpt-5.4-nano": ModelPrice(input=0.20, output=1.25,  cache_write=0.20, cache_read=0.02),
    "gpt-5.4":      ModelPrice(input=2.50, output=15.0,  cache_write=2.50, cache_read=0.25),
    # codex models (gpt-5.3-codex etc.). Matched across generations by the "codex" substring.
    "codex":        ModelPrice(input=1.75, output=14.0,  cache_write=1.75, cache_read=0.175),

    # ── OpenAI: legacy (removed from the official page; former published values) ─────────
    # gpt-5 lumps the plain / 5.1 / 5.2 generations together and estimates at the old $1.25/$10.
    "gpt-5-mini":   ModelPrice(input=0.25, output=2.0,   cache_write=0.25, cache_read=0.025),
    "gpt-5-nano":   ModelPrice(input=0.05, output=0.40,  cache_write=0.05, cache_read=0.005),
    "gpt-5":        ModelPrice(input=1.25, output=10.0,  cache_write=1.25, cache_read=0.125),
    "gpt-4.1-mini": ModelPrice(input=0.40, output=1.60,  cache_write=0.40, cache_read=0.10),
    "gpt-4.1-nano": ModelPrice(input=0.10, output=0.40,  cache_write=0.10, cache_read=0.025),
    "gpt-4.1":      ModelPrice(input=2.0,  output=8.0,   cache_write=2.0,  cache_read=0.50),
    "gpt-4o-mini":  ModelPrice(input=0.15, output=0.60,  cache_write=0.15, cache_read=0.075),
    "gpt-4o":       ModelPrice(input=2.50, output=10.0,  cache_write=2.50, cache_read=1.25),
    "o4-mini":      ModelPrice(input=1.10, output=4.40,  cache_write=1.10, cache_read=0.275),
    "o3-mini":      ModelPrice(input=1.10, output=4.40,  cache_write=1.10, cache_read=0.55),
    "o3":           ModelPrice(input=2.0,  output=8.0,   cache_write=2.0,  cache_read=0.50),
}

# Fallback for unknown models (equivalent to sonnet). The is_estimate flag conveys "unknown".
# Even if unknown, a model containing "gpt" falls back to the current mid-tier gpt-5.4
# (sonnet-equivalent) (see resolve_family).
_DEFAULT_FAMILY = "sonnet"
_DEFAULT_GPT_FAMILY = "gpt-5.4"

# Heuristic: 1 token ≈ 4 characters. Japanese is ≈ 1-2 tokens per character, so this leans
# toward underestimation, but that is acceptable for "ballpark" use (see docstring).
_CHARS_PER_TOKEN = 4.0


# Family resolution priority (explicit). Held as a fixed tuple to avoid implicit dependence
# on dict iteration order. Uses substring matching (not startswith): Anthropic uses
# `claude-opus-4-...`, Bedrock uses `anthropic.claude-opus-...`, cross-region uses
# `us.anthropic.claude-opus-...`, so the family name is not at the start but carries a prefix,
# which startswith would miss.
# On the OpenAI side, "more specific keys come first" (e.g. gpt-5.4-mini contains gpt-5.4 /
# gpt-5 too by substring, so order is mini/nano/pro → generation → plain). "codex" comes
# before the generation keys (gpt-5.4-codex etc. also resolve to codex-series pricing =
# cross-generation). "o3"/"o4-mini" are short keys prone to false matches, so they go last.
_FAMILY_ORDER: tuple[str, ...] = (
    "opus", "sonnet", "haiku",
    "codex",
    "gpt-5.5-pro", "gpt-5.4-pro", "gpt-5.4-mini", "gpt-5.4-nano",
    "gpt-5.5", "gpt-5.4",
    "gpt-5-mini", "gpt-5-nano", "gpt-5",
    "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1",
    "gpt-4o-mini", "gpt-4o",
    "o4-mini", "o3-mini", "o3",
)


def resolve_family(model: Any) -> str:
    """Resolve the family name from a model id. Falls back to the default if unknown
    (gpt-5.4 if it contains gpt).

    model is assumed to be a str, but callers (register/usage computation) may pass through
    values from a malformed request, so non-str inputs are coerced with str() to avoid crashing.
    """
    m = str(model or "").lower()
    for fam in _FAMILY_ORDER:
        if fam in m:
            return fam
    if "gpt" in m:
        return _DEFAULT_GPT_FAMILY
    return _DEFAULT_FAMILY


def price_for(model: str | None) -> ModelPrice:
    return _FAMILY_PRICES[resolve_family(model)]


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


def compute_cost(
    model: str | None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> dict[str, Any]:
    """Return the estimated cost in USD. Each breakdown + total + resolved family + is_estimate.

    input_tokens is expected to be **the portion that did not hit the cache** (non-cached input).
    cache_write/read are each billed at their own rate (matching Anthropic's usage semantics).
    """
    fam = resolve_family(model)
    p = _FAMILY_PRICES[fam]
    per_mtok = 1_000_000.0
    input_cost = input_tokens * p.input / per_mtok
    output_cost = output_tokens * p.output / per_mtok
    cache_write_cost = cache_write_tokens * p.cache_write / per_mtok
    cache_read_cost = cache_read_tokens * p.cache_read / per_mtok
    total = input_cost + output_cost + cache_write_cost + cache_read_cost
    return {
        "model_family": fam,
        "is_estimate": True,  # always an estimate (not a real tokenizer)
        "currency": "USD",
        "input_usd": round(input_cost, 6),
        "output_usd": round(output_cost, 6),
        "cache_write_usd": round(cache_write_cost, 6),
        "cache_read_usd": round(cache_read_cost, 6),
        "total_usd": round(total, 6),
    }


def cache_savings_usd(model: str | None, cache_read_tokens: int) -> float:
    """Approximate the amount saved by a cache read (= the difference vs. paying for the same
    tokens as normal input).

    The write-time surcharge (1.25x) is incurred separately, but here we only compute the rough
    savings per read. Used by stats to accumulate hit read tokens and show "roughly this much saved".
    """
    p = price_for(model)
    return round(cache_read_tokens * (p.input - p.cache_read) / 1_000_000.0, 6)
