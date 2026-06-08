"""擬似プロンプトキャッシュ (debug 用)。

正式仕様: README.md

目的: 「アプリが prompt caching を効かせられる構造で投げているか」をハッシュで観測する。
実キャッシュは持たない (応答は常に control 経由)。

**Anthropic 実機の挙動を再現** (初版の単一 breakpoint・exact-hash モデルを是正):
- 1 リクエストに **複数 breakpoint** (`cache_control: {type:"ephemeral"}`、最大 4) を置ける。
  各 breakpoint は「そこまでの prefix」を 1 キャッシュエントリとして **write** する。
- **read は前方一致 (prefix match)**: 過去に write された最長の prefix が現リクエストの
  先頭に一致すれば、その分を read (0.1x)。現リクエストが breakpoint を宣言していない位置でも、
  過去に write 済みなら read できる (例: BP2 を毎 turn 末尾に前進させても、前 turn の prefix が
  今 turn の前方一致になり incremental に hit する)。
- **cache_control マーカーはキャッシュキーに含めない**: 実機はキー = content tokens で、
  cache_control は指示メタデータとして無視する。よって prefix の hash/token は
  **cache_control を除去**して計算する (マーカー位置が turn 毎に動いても content 一致で hit)。

render 順序 = tools → system → messages (prefix-match の評価順)。TTL は 5 分 (read で延長)。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from . import pricing

DEFAULT_TTL_SECONDS = 300.0  # Anthropic 5min ephemeral cache

# 実機 fidelity 用の定数:
_LOOKBACK_BLOCKS = 20  # breakpoint は ≤20 content block しか遡って prior entry を探さない
_MAX_BREAKPOINTS = 4   # 1 リクエストの cache_control は最大 4 (実機は超過で 400)
# モデル別の最小キャッシュ prefix (tokens)。実機はこれ未満の prefix を無言で非キャッシュ
# (cache_creation=0)。NOTE: 当 sim の token は approx (≈4 chars/token、日本語過小) なので、
# 閾値近傍の判定は不正確 (日本語 prefix は実トークンが大きく実機はキャッシュするが approx だと
# 閾値割れで非キャッシュと出ることがある)。明らかに小さい prefix の過大計上を防ぐのが主目的。
_MIN_CACHEABLE = {"opus": 4096, "sonnet": 1024, "haiku": 2048}
_DEFAULT_MIN_CACHEABLE = 1024


def _min_cacheable_for(model: Any) -> int:
    return _MIN_CACHEABLE.get(pricing.resolve_family(model), _DEFAULT_MIN_CACHEABLE)


def _has_cache_control(block: Any) -> bool:
    return isinstance(block, dict) and block.get("cache_control") is not None


def _segments(system: Any, tools: Any, messages: Any) -> list[dict[str, Any]]:
    """tools → system → messages の順にブロックを 1 列に並べる (prefix 評価順)。

    system は str / list、message.content は str / list の両方を吸収する。
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
    """セグメントから cache_control を除いた content-only ビューを返す (hash/token 用)。

    実機がキャッシュキーに cache_control を含めないのに合わせる。これにより BP の位置が
    turn 毎に動いても、同一 content の prefix は同一バイト列 = 同一 hash になり前方一致する。
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
    """1 リクエストのキャッシュ解析結果 (segments + breakpoint 位置 + 総トークン)。"""
    segs: list[dict[str, Any]]
    breakpoints: list[int]   # cache_control を持つ segment index (昇順)。各 +1 が prefix seg_count。
    total_tokens: int

    def prefix_hash(self, seg_count: int) -> str:
        return _hash_prefix(self.segs, seg_count)

    def prefix_tokens(self, seg_count: int) -> int:
        return _tokens_prefix(self.segs, seg_count)


def analyze_request(system: Any = None, tools: Any = None, messages: Any = None) -> RequestCache:
    """1 回の segments 分解で RequestCache を返す (multi-breakpoint 対応)。"""
    segs = _segments(system, tools, messages)
    bps = [i for i, s in enumerate(segs) if _has_cache_control(s["block"])]
    return RequestCache(segs=segs, breakpoints=bps, total_tokens=_tokens_prefix(segs, len(segs)))


# ── 後方互換: 単一 prefix が欲しい旧 caller / 単体テスト向け ──────────────


@dataclass
class CachePrefix:
    hash: str
    tokens: int
    breakpoints: int
    segments: int


def extract_cache_prefix(
    system: Any = None, tools: Any = None, messages: Any = None
) -> CachePrefix | None:
    """cacheable prefix を抽出 (最深 breakpoint まで)。cache_control が無ければ None。"""
    rc = analyze_request(system, tools, messages)
    if not rc.breakpoints:
        return None
    deepest = rc.breakpoints[-1] + 1
    return CachePrefix(
        hash=rc.prefix_hash(deepest), tokens=rc.prefix_tokens(deepest),
        breakpoints=len(rc.breakpoints), segments=deepest,
    )


class CacheSimulator:
    """prefix hash → エントリの index を持ち、multi-breakpoint + 前方一致で hit/miss を判定。

    時刻は呼び出し側から `now` を渡す (テスト容易性)。
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS, honor_ttl: bool = True,
                 min_cacheable_tokens: int | None = None):
        self.ttl_seconds = ttl_seconds
        self.honor_ttl = honor_ttl
        # None = モデル別 (_min_cacheable_for)。明示値を渡すと全モデルでそれを使う (テスト用に 0 等)。
        self.min_cacheable_tokens = min_cacheable_tokens
        # hash -> {seg_count, tokens, created_at, last_seen, hits, misses, model}
        self.index: dict[str, dict[str, Any]] = {}

    def _alive(self, entry: dict[str, Any], now: float) -> bool:
        return (not self.honor_ttl) or (now - entry["created_at"]) <= self.ttl_seconds

    @staticmethod
    def _key(model: str | None, content_hash: str) -> str:
        # 実機のキャッシュは model 単位 (model 切替で無効化)。キーに model を畳み込み、
        # 同一 content でも別 model なら別エントリ = 別 model からの誤 hit / model 上書きを防ぐ。
        return f"{model}\x00{content_hash}"

    def _min_tokens(self, model: str | None) -> int:
        return self.min_cacheable_tokens if self.min_cacheable_tokens is not None else _min_cacheable_for(model)

    def _prune(self, now: float) -> None:
        """TTL の 2 倍を超えた dead entry を遅延 GC (read 不能・無制限増加防止)。"""
        if not self.honor_ttl:
            return
        cutoff = 2 * self.ttl_seconds
        for h in [h for h, e in self.index.items() if now - e["created_at"] > cutoff]:
            del self.index[h]

    def observe(self, rc: RequestCache, model: str | None, now: float) -> dict[str, Any]:
        """1 リクエストを観測し cache 判定を返す (multi-breakpoint + 前方一致 + 最小閾値 + 20-block lookback)。

        返り値: {status, cache_read_tokens, cache_creation_tokens, prefix_hash, read_seg_count, breakpoints}
        status: "hit" | "miss" | "none" (cache_control 無し or 全 breakpoint が最小閾値未満)。
        """
        self._prune(now)
        min_tok = self._min_tokens(model)
        # 有効 breakpoint = prefix が最小閾値以上のもの。最大 4 (実機制限、超過分は deepest 側を採用)。
        eff_bps = [bp for bp in rc.breakpoints if rc.prefix_tokens(bp + 1) >= min_tok][-_MAX_BREAKPOINTS:]
        if not eff_bps:
            # cache_control はあるが全て閾値未満 → 実機は非キャッシュ (over-report 防止)。
            return {"status": "none", "cache_read_tokens": 0, "cache_creation_tokens": 0,
                    "prefix_hash": None, "read_seg_count": 0, "breakpoints": len(rc.breakpoints)}

        n = len(rc.segs)
        # READ: 既存エントリ seg_count を降順。(a) 20-block 以内に有効 breakpoint がある
        # (lookback) (b) hash 前方一致 (c) alive、の最長を採用。
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
                continue  # どの breakpoint からも 20-block 以内に届かない → 実機は見つけられない
            h = self._key(model, rc.prefix_hash(sc))
            e = self.index.get(h)
            if e is not None and e["seg_count"] == sc and self._alive(e, now):
                read_tokens, read_hash, read_entry, read_seg = rc.prefix_tokens(sc), h, e, sc
                break  # 降順なので最初の一致が最長

        # WRITE 先 = 最深 有効 breakpoint。creation = (最深 - read)。
        deepest_sc = eff_bps[-1] + 1
        deepest_tokens = rc.prefix_tokens(deepest_sc)
        creation_tokens = max(0, deepest_tokens - read_tokens)

        if read_entry is not None:  # read で TTL 延長 + hit カウント
            read_entry["created_at"] = now
            read_entry["last_seen"] = now
            read_entry["hits"] += 1
            read_entry["model"] = model

        # WRITE: 有効 breakpoint prefix を index に登録/更新 (TTL 延長、model 単位キー)。
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
