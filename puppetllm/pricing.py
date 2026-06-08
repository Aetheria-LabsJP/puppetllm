"""コスト目安の算出 (debug 用、approx)。

正式仕様: README.md

fake_server は実 token 数を計算しない (実 SDK 経路が必要)。本モジュールは
**ヒューリスティック概算** (≈4 chars/token) と **公開料金表** から「大体のコスト目安」を出す。
精度より「桁・傾向が分かる」ことを優先する。実課金とは一致しないので注意。

料金は USD / 100万トークン (per Mtok)。Anthropic 公開価格 (2025 時点) を手で写経。
価格改定時は本テーブルだけ直せばよい。cache_write は input の 1.25x、cache_read は
input の 0.1x という Anthropic の 5 分 cache 規則に沿った値 (覚え書きとして明示)。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPrice:
    """per Mtok (USD)。"""
    input: float
    output: float
    cache_write: float  # 5min cache 書き込み (= input * 1.25)
    cache_read: float   # cache 読み出し     (= input * 0.10)


# モデルファミリ別料金。キーは model id の部分一致 (opus/sonnet/haiku) で引く。
# Anthropic / Bedrock どちらの model id でもファミリ名が含まれるので共通で使える。
_FAMILY_PRICES: dict[str, ModelPrice] = {
    # Claude Opus (4.x = 4.6/4.7/4.8)。$5/$25 (旧 Claude 3 Opus / Opus 4.0-4.1 は
    # $15/$75 だったので注意。model id では世代判別しないため最新 4.x 価格を採用)。
    "opus":   ModelPrice(input=5.0,  output=25.0, cache_write=6.25,  cache_read=0.50),
    # Claude Sonnet (4.x)
    "sonnet": ModelPrice(input=3.0,  output=15.0, cache_write=3.75,  cache_read=0.30),
    # Claude Haiku (4.5)。3.5 世代は $0.80/$4 と安かった点に注意 (model id では世代を
    # 判別しないため最新 4.5 価格を採用)。
    "haiku":  ModelPrice(input=1.0,  output=5.0,  cache_write=1.25,  cache_read=0.10),
}

# 未知 model 時のフォールバック (sonnet 相当)。is_estimate フラグで「不明」を伝える。
_DEFAULT_FAMILY = "sonnet"

# ヒューリスティック: 1 token ≈ 4 文字。日本語は 1 文字 ≈ 1-2 token なので過小評価寄り
# だが「目安」用途では許容する (docstring 参照)。
_CHARS_PER_TOKEN = 4.0


# ファミリ判定の優先順位 (明示)。dict 反復順への暗黙依存を避けるため固定タプルで持つ。
# substring 一致を使う (startswith ではない): Anthropic は `claude-opus-4-...`、Bedrock は
# `anthropic.claude-opus-...`、cross-region は `us.anthropic.claude-opus-...` のように
# ファミリ名が先頭に来ず接頭辞が付くため、startswith だと取りこぼす。
_FAMILY_ORDER: tuple[str, ...] = ("opus", "sonnet", "haiku")


def resolve_family(model: str | None) -> str:
    """model id からファミリ名 (opus/sonnet/haiku) を解決。不明なら default。"""
    m = (model or "").lower()
    for fam in _FAMILY_ORDER:
        if fam in m:
            return fam
    return _DEFAULT_FAMILY


def price_for(model: str | None) -> ModelPrice:
    return _FAMILY_PRICES[resolve_family(model)]


def approx_tokens(value: Any) -> int:
    """任意の JSON 的構造のトークン数を概算する。

    文字列化した総長 / 4 で概算 (≈4 chars/token)。dict/list は JSON 直列化。
    None/空は 0。実 tokenizer ではないので「目安」。
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
    """レスポンス content blocks の概算トークン。"""
    return approx_tokens(content_blocks)


def compute_cost(
    model: str | None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> dict[str, Any]:
    """概算コストを USD で返す。各内訳 + 合計 + 使用ファミリ + is_estimate。

    input_tokens は **cache に乗らなかった分** (非キャッシュ入力) を渡す前提。
    cache_write/read はそれぞれ別単価で課金される (Anthropic の usage 意味論に合わせる)。
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
        "is_estimate": True,  # 常に概算 (実 tokenizer ではない)
        "currency": "USD",
        "input_usd": round(input_cost, 6),
        "output_usd": round(output_cost, 6),
        "cache_write_usd": round(cache_write_cost, 6),
        "cache_read_usd": round(cache_read_cost, 6),
        "total_usd": round(total, 6),
    }


def cache_savings_usd(model: str | None, cache_read_tokens: int) -> float:
    """cache read で浮いた額の概算 (= 同トークンを通常 input で払った場合との差)。

    write 時の割増 (1.25x) は別途発生するが、ここでは read 1 回あたりの粗い節約額のみ。
    stats で hit の read トークンを積算して「だいたいこれだけ得した」を示す用途。
    """
    p = price_for(model)
    return round(cache_read_tokens * (p.input - p.cache_read) / 1_000_000.0, 6)
