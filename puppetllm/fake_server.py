"""Anthropic / Bedrock 互換 fake server (debug/regression 用)。

正式仕様・使用例・設計判断は **README.md** を参照。

Anthropic SDK は環境変数 `ANTHROPIC_BASE_URL` を尊重するので、これを localhost に向ければ
SDK の真の経路 (HTTP → SSE → stream イベントパース) をすべて通したまま、人間 or 別エージェントが
レスポンスを供給できる。Bedrock SDK (`AnthropicBedrock` / boto3) も base_url 差し替えで
本 server に向けられる。

アーキテクチャ (provider 非依存の canonical core + アダプタ):
- 本ファイル = canonical core: 正規化 snapshot 管理 + /_control/* + cost/cache 計算
- Anthropic 経路: `/v1/messages` (本ファイル内に実装)
- Bedrock 経路  : `providers/bedrock.py` (`/model/{id}/invoke[-with-response-stream]`)
- 応答 content blocks / 制御 API は provider 共通 (注入は同じ /_control/respond)

実装範囲:
- POST /v1/messages                  — Anthropic 互換 (SSE / 単一 JSON)
- POST /model/{id}/invoke[...]        — Bedrock 互換 (providers/bedrock.py が追加)
- GET  /_control/pending             — 保留中リクエスト (provider 込み)
- GET  /_control/wait_for_pending    — long-poll で次 pending を待つ
- POST /_control/respond             — 保留中リクエストに応答を注入
- POST /_control/auto                — 簡易自動応答 (text のみ)
- POST /_control/error               — HTTP error response 注入
- GET  /_control/history             — (request, response, usage, cost) 履歴
- GET  /_control/stats               — コスト目安・トークン・キャッシュの累計サマリ
- GET  /_control/cache               — 擬似プロンプトキャッシュ index
- POST /_control/clear               — pending/history/cache を空に
- GET  /_control/health              — ヘルスチェック

Control 系は localhost のみ (デバッグ用)、認可なし。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import pricing
from .cache_sim import CacheSimulator, analyze_request


# ── 設定 (環境変数) ───────────────────────────────────────────────────

# 擬似プロンプトキャッシュの TTL (秒) と TTL を honor するか。
_CACHE_TTL = float(os.environ.get("PUPPETLLM_CACHE_TTL", "300"))
_CACHE_HONOR_TTL = os.environ.get("PUPPETLLM_CACHE_HONOR_TTL", "1") != "0"
# 最小キャッシュ閾値。未設定/不正/負値 = モデル別 (Opus 4096/Sonnet 1024 等)。`0` で無効 (全 prefix キャッシュ)。
def _parse_cache_min(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n >= 0 else None  # 負値は無意味 (実質全キャッシュ) なのでモデル別に戻す


_CACHE_MIN_TOKENS = _parse_cache_min(os.environ.get("PUPPETLLM_CACHE_MIN_TOKENS"))


# ── サーバ状態 ────────────────────────────────────────────────────────


class _ServerState:
    def __init__(self) -> None:
        # multi-pending (README §並列リクエスト対応): pending_id → {request, future, started_at}。
        # 複数の同時 in-flight リクエストを保持できる (Phase 2 の並列 specialist 起動を
        # proxy 経由で検証可能にするため)。単一 pending しか無い場合も dict に 1 件入るだけ。
        self.pending: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self.turn_count: int = 0
        self.lock = asyncio.Lock()
        # /_control/wait_for_pending で待機中の future 群。
        self.pending_arrival_waiters: list[asyncio.Future[dict[str, Any]]] = []
        # 擬似プロンプトキャッシュ (prefix hash → hit/miss)。
        self.cache = CacheSimulator(ttl_seconds=_CACHE_TTL, honor_ttl=_CACHE_HONOR_TTL,
                                    min_cacheable_tokens=_CACHE_MIN_TOKENS)

    def _oldest_pending(self) -> dict[str, Any] | None:
        """received_at が最古の pending entry を返す (なければ None)。lock 内で呼ぶこと。"""
        if not self.pending:
            return None
        return min(self.pending.values(), key=lambda e: e["request"].get("received_at", 0))


state = _ServerState()
app = FastAPI(title="puppetllm fake-anthropic")


async def _parse_json_body(request: Request) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    """request body を JSON parse。失敗時は 500 でなく明確な 400 を返す。

    responder が curl で複雑ネスト (render_chart の input 等) を送る際の
    shell エスケープ崩れで malformed JSON になりがち。その場合に opaque な 500 では
    なく `{"error": "invalid JSON body", ...}` の 400 を返し原因を分かりやすくする。
    """
    try:
        body = await request.json()
    except Exception as e:
        return None, JSONResponse(
            {"error": "invalid JSON body", "detail": str(e)[:200]}, status_code=400
        )
    if not isinstance(body, dict):
        return None, JSONResponse(
            {"error": "JSON body must be an object"}, status_code=400
        )
    return body, None


# ── canonical: コスト/キャッシュ計算 ─────────────────────────────────


def _compute_usage(snapshot: dict[str, Any], content_blocks: list[dict[str, Any]]) -> tuple[dict, dict]:
    """snapshot (入力) と応答 content_blocks から usage と概算 cost を作る。

    Anthropic usage 意味論に合わせる:
      input_tokens               = cache に乗らなかった入力 (= 総入力 - read - creation)
      cache_creation_input_tokens = 今回 cache に書いた分 (miss 時の prefix)
      cache_read_input_tokens     = cache から読んだ分 (hit 時の prefix)
    """
    model = snapshot.get("model")
    total_in = int(snapshot.get("input_tokens_total", 0))
    cache = snapshot.get("cache") or {}
    read = int(cache.get("cache_read_tokens", 0))
    creation = int(cache.get("cache_creation_tokens", 0))
    uncached = max(0, total_in - read - creation)
    output = pricing.estimate_output_tokens(content_blocks)
    cost = pricing.compute_cost(
        model,
        input_tokens=uncached,
        output_tokens=output,
        cache_write_tokens=creation,
        cache_read_tokens=read,
    )
    usage = {
        "input_tokens": uncached,
        "output_tokens": output,
        "cache_creation_input_tokens": creation,
        "cache_read_input_tokens": read,
    }
    return usage, cost


async def _record_and_reset(
    request_snapshot: dict[str, Any],
    response_blocks: list[dict[str, Any]] | None,
    injected_error: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    cost: dict[str, Any] | None = None,
) -> None:
    """history に 1 件追記して当該 pending を registry から除去。

    主処理から呼ばれる。`request_snapshot` は handler 側で確保した copy で、
    その `pending_id` をキーに自分の entry だけを除去する (他の in-flight に影響しない)。
    usage / cost / cache を併記して /_control/stats が集計できるようにする。
    """
    entry: dict[str, Any] = {
        "turn": request_snapshot.get("turn"),
        "provider": request_snapshot.get("provider"),
        "model": request_snapshot.get("model"),
        "request": request_snapshot,
        "response_blocks": response_blocks,
        "usage": usage,
        "cost": cost,
        "cache": request_snapshot.get("cache"),
        "completed_at": time.time(),
    }
    if injected_error is not None:
        entry["injected_error"] = injected_error
    pid = request_snapshot.get("pending_id")
    async with state.lock:
        state.history.append(entry)
        # 自分の pending_id の entry だけ除去。clear が先に走って消えていれば no-op。
        if pid is not None:
            state.pending.pop(pid, None)


# ── canonical: リクエスト登録 / 応答待ち (provider 共通) ──────────────


async def register_request(
    provider: str,
    model: str | None,
    body: dict[str, Any],
    is_stream: bool,
) -> tuple[dict[str, Any], asyncio.Future]:
    """正規化 snapshot を作り pending に登録、応答待ち future を返す。

    入力トークン概算と擬似キャッシュ判定 (hit/miss) はここで turn 採番と同じ lock 内で行い、
    並列リクエストでも順序が決まるようにする。provider に依らず共通。
    """
    system = body.get("system")
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    # multi-breakpoint + 前方一致対応の解析 (segments/breakpoints/total を一括算出)。
    request_cache = analyze_request(system, tools, messages)
    input_tokens_total = request_cache.total_tokens
    now = time.time()

    async with state.lock:
        state.turn_count += 1
        turn = state.turn_count
        pending_id = uuid.uuid4().hex[:16]
        cache = state.cache.observe(request_cache, model, now)
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        request_snapshot = {
            "pending_id": pending_id,
            "turn": turn,
            "provider": provider,
            "model": model,
            "system": system,
            "messages": messages,
            "tools": tools,
            "max_tokens": body.get("max_tokens"),
            "stream": is_stream,
            "received_at": now,
            "input_tokens_total": input_tokens_total,
            "cache": cache,
        }
        state.pending[pending_id] = {
            "request": request_snapshot,
            "future": fut,
            "started_at": now,
        }
        # /_control/wait_for_pending で待機中の watcher を起こす。
        for w in state.pending_arrival_waiters:
            if not w.done():
                w.set_result(request_snapshot)
        state.pending_arrival_waiters.clear()

    return request_snapshot, fut


async def await_resolution(snapshot: dict[str, Any], fut: asyncio.Future) -> dict[str, Any]:
    """control 経由の応答を待ち、history に記録して結果を tagged dict で返す。

    返り値の "kind":
      "cleared" → /_control/clear された (caller は 503)
      "error"   → エラー注入 ({"status", "type", "message"})
      "ok"      → 正常 ({"content_blocks", "usage", "cost", "model", "message_id"})
    provider 非依存。encode は各 provider 側で行う。
    """
    try:
        response_payload = await fut
    except RuntimeError as e:
        # clear 経由のキャンセル。状態は clear がリセット済。
        return {"kind": "cleared", "detail": str(e)}

    model = snapshot.get("model")
    if isinstance(response_payload, dict) and response_payload.get("_inject_error"):
        status = int(response_payload.get("status", 500))
        etype = str(response_payload.get("type", "api_error"))
        emsg = str(response_payload.get("message", "fake_server injected error"))
        await _record_and_reset(
            snapshot, response_blocks=None,
            injected_error={"status": status, "type": etype, "message": emsg},
        )
        return {"kind": "error", "status": status, "type": etype, "message": emsg}

    content_blocks = response_payload.get("content") or []
    if not isinstance(content_blocks, list):
        content_blocks = []
    usage, cost = _compute_usage(snapshot, content_blocks)
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    await _record_and_reset(snapshot, response_blocks=content_blocks, usage=usage, cost=cost)
    return {
        "kind": "ok",
        "content_blocks": content_blocks,
        "usage": usage,
        "cost": cost,
        "model": model,
        "message_id": message_id,
    }


# ── SSE / stream イベント構築 helpers ────────────────────────────────


def _sse_event(event_name: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def stream_event_dicts(
    message_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Anthropic streaming protocol の (event_name, data) 列を構築する。

    SSE (Anthropic) でも eventstream (Bedrock) でも同じイベント列を使えるよう、
    ワイヤ形式に依存しない dict 列で返す。usage 未指定なら fake 値。
    """
    # usage が来ていれば実 (概算) 値、無ければ従来の fake 値。
    if usage is not None:
        start_usage = {
            "input_tokens": usage.get("input_tokens", 1),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            # message_start 時点では出力は未生成。累計 output は message_delta 側で返す。
            "output_tokens": 0,
        }
        delta_usage = {"output_tokens": usage.get("output_tokens", 0)}
    else:
        start_usage = {"input_tokens": 1, "output_tokens": 1}
        delta_usage = {"output_tokens": 100}

    out: list[tuple[str, dict[str, Any]]] = []
    out.append(("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": start_usage,
        },
    }))

    stop_reason = "end_turn"
    for idx, block in enumerate(content_blocks):
        btype = block.get("type")
        if btype == "text":
            out.append(("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "text", "text": ""},
            }))
            text = str(block.get("text", ""))
            chunk_size = 80
            chunks = (
                [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                if text else [""]
            )
            for chunk in chunks:
                out.append(("content_block_delta", {
                    "type": "content_block_delta", "index": idx,
                    "delta": {"type": "text_delta", "text": chunk},
                }))
            out.append(("content_block_stop", {
                "type": "content_block_stop", "index": idx,
            }))
        elif btype == "tool_use":
            stop_reason = "tool_use"
            tool_id = str(block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
            out.append(("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": str(block.get("name", "")),
                    "input": {},
                },
            }))
            input_json = json.dumps(block.get("input", {}) or {}, ensure_ascii=False)
            out.append(("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            }))
            out.append(("content_block_stop", {
                "type": "content_block_stop", "index": idx,
            }))
        else:
            # 未知ブロック (text/tool_use 以外) は skip。
            continue

    out.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": delta_usage,
    }))
    out.append(("message_stop", {"type": "message_stop"}))
    return out


def _build_sse_stream(
    message_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
) -> list[bytes]:
    """Anthropic streaming protocol を満たす SSE バイト列を構築。

    note: `usage` を渡すと概算トークン/キャッシュ値が乗る。未指定なら fake 値 (旧挙動)。
    """
    return [_sse_event(name, data) for name, data in stream_event_dicts(message_id, model, content_blocks, usage)]


def _build_non_stream_response(
    message_id: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stop_reason = "tool_use" if any(b.get("type") == "tool_use" for b in content_blocks) else "end_turn"
    if usage is not None:
        usage_out = {
            "input_tokens": usage.get("input_tokens", 1),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        }
    else:
        usage_out = {"input_tokens": 1, "output_tokens": 100}
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_out,
    }


# ── Anthropic 互換エンドポイント ─────────────────────────────────────


@app.post("/v1/messages")
async def messages(request: Request) -> Any:
    body, err = await _parse_json_body(request)
    if err is not None:
        return err
    is_stream = bool(body.get("stream"))
    model = body.get("model")

    snapshot, fut = await register_request("anthropic", model, body, is_stream)
    result = await await_resolution(snapshot, fut)

    if result["kind"] == "cleared":
        return JSONResponse({"error": f"request cleared: {result['detail']}"}, status_code=503)
    if result["kind"] == "error":
        return JSONResponse(
            {"type": "error", "error": {"type": result["type"], "message": result["message"]}},
            status_code=result["status"],
        )

    model_out = model or "claude-sonnet-mock"
    content_blocks = result["content_blocks"]
    usage = result["usage"]
    message_id = result["message_id"]

    if is_stream:
        events = _build_sse_stream(message_id, model_out, content_blocks, usage)

        async def gen():
            for evt in events:
                yield evt
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(_build_non_stream_response(message_id, model_out, content_blocks, usage))


# ── 制御エンドポイント ───────────────────────────────────────────────


@app.get("/_control/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "turn_count": state.turn_count}


@app.get("/_control/pending")
async def pending() -> dict[str, Any]:
    """現在の全 pending を返す (multi-pending)。

    後方互換: `has_pending` (bool) と、pending が 1 件以上なら最古を `request` /
    `waiting_for_seconds` にも入れる。並列対応 caller は `pending` 配列を使う。
    """
    now = time.time()
    async with state.lock:
        items = [
            {
                "pending_id": pid,
                "request": e["request"],
                "waiting_for_seconds": round(now - e["started_at"], 2),
            }
            for pid, e in state.pending.items()
        ]
        oldest = state._oldest_pending()
    items.sort(key=lambda x: x["request"].get("received_at", 0))
    if not items:
        return {"has_pending": False, "pending": [], "count": 0}
    return {
        "has_pending": True,
        "count": len(items),
        "pending": items,
        # 後方互換 (最古 pending)
        "request": oldest["request"] if oldest else None,
        "waiting_for_seconds": round(now - oldest["started_at"], 2) if oldest else None,
    }


# Long-poll の安全上限 (Bash tool の 10 分タイムアウトを跨がない範囲で多少余裕)
_WAIT_TIMEOUT_MAX = 600.0
_WAIT_TIMEOUT_DEFAULT = 270.0  # < 5 分 (Anthropic prompt cache TTL 内に収まるよう)


@app.get("/_control/wait_for_pending")
async def wait_for_pending(timeout: float = _WAIT_TIMEOUT_DEFAULT) -> dict[str, Any]:
    """Long-polling: pending が出るまで block (最大 timeout 秒)。

    既に pending が存在すれば即返却。なければ次の /v1/messages 到着を待つ。
    """
    timeout = max(0.5, min(float(timeout), _WAIT_TIMEOUT_MAX))

    waiter: asyncio.Future[dict[str, Any]] | None = None
    async with state.lock:
        oldest = state._oldest_pending()
        if oldest is not None:
            return {
                "has_pending": True,
                "request": oldest["request"],
                "pending_id": oldest["request"].get("pending_id"),
                "waiting_for_seconds": round(time.time() - oldest["started_at"], 2),
            }
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        state.pending_arrival_waiters.append(waiter)

    try:
        request_snapshot = await asyncio.wait_for(waiter, timeout=timeout)
    except asyncio.TimeoutError:
        return {"has_pending": False, "timeout": True}
    except BaseException:
        raise
    finally:
        async with state.lock:
            try:
                state.pending_arrival_waiters.remove(waiter)
            except ValueError:
                pass

    return {
        "has_pending": True,
        "request": request_snapshot,
        "pending_id": request_snapshot.get("pending_id"),
        "waiting_for_seconds": 0.0,
    }


async def _resolve_target_future(
    pending_id: str | None,
) -> tuple[asyncio.Future[dict[str, Any]] | None, JSONResponse | None]:
    """注入先の pending future を解決する (multi-pending)。

    - `pending_id` 指定: その entry を使う (無ければ 400)
    - 未指定: pending がちょうど 1 件ならそれを使う (後方互換)。0 件→400、複数→400
    """
    async with state.lock:
        if pending_id is not None:
            entry = state.pending.get(pending_id)
            if entry is None or entry["future"].done():
                return None, JSONResponse(
                    {"error": f"no pending request with pending_id={pending_id}"}, status_code=400
                )
            return entry["future"], None
        ids = [pid for pid, e in state.pending.items() if not e["future"].done()]
        if not ids:
            return None, JSONResponse({"error": "no pending request"}, status_code=400)
        if len(ids) > 1:
            return None, JSONResponse(
                {"error": "multiple pending requests; specify pending_id", "pending_ids": ids},
                status_code=400,
            )
        return state.pending[ids[0]]["future"], None


def _safe_set_result(
    fut: asyncio.Future[dict[str, Any]], value: dict[str, Any]
) -> JSONResponse | None:
    """future に結果を注入。done なら 409 (clear と競合等)。"""
    try:
        fut.set_result(value)
    except asyncio.InvalidStateError:
        return JSONResponse(
            {"error": "pending request already resolved (e.g. cleared)"}, status_code=409
        )
    return None


@app.post("/_control/respond")
async def respond(request: Request) -> Any:
    """Body: `{"content": [<content_block>, ...], "pending_id"?: "..."}` を注入。

    content_block の type は "text" | "tool_use"。`pending_id` 省略時は pending が
    1 件ならそれに注入 (後方互換)。複数 in-flight なら `pending_id` 必須。
    """
    body, err = await _parse_json_body(request)
    if err is not None:
        return err
    fut, err = await _resolve_target_future(body.get("pending_id"))
    if err is not None:
        return err
    err = _safe_set_result(fut, {"content": body.get("content", [])})
    if err is not None:
        return err
    return {"ok": True}


@app.post("/_control/auto")
async def auto(request: Request) -> Any:
    """簡易: `{"text": "...", "pending_id"?: "..."}` を text-only 応答として注入。"""
    body, err = await _parse_json_body(request)
    if err is not None:
        return err
    fut, err = await _resolve_target_future(body.get("pending_id"))
    if err is not None:
        return err
    err = _safe_set_result(fut, {
        "content": [{"type": "text", "text": body.get("text", "(empty)")}]
    })
    if err is not None:
        return err
    return {"ok": True}


@app.post("/_control/error")
async def inject_error(request: Request) -> Any:
    """エラー注入: pending リクエストに HTTP エラーを返させる。

    Body: {"status": 429, "type": "rate_limit_error", "message": "..."}
    Anthropic / Bedrock どちらの経路でも、各 provider が status/type を自経路の
    エラー形式に直して返す。SDK は 5xx/429/408 を自動 retry する。
    """
    body, err = await _parse_json_body(request)
    if err is not None:
        return err
    try:
        status = int(body.get("status", 500))
    except (TypeError, ValueError):
        return JSONResponse({"error": "status must be an integer"}, status_code=400)
    if not (100 <= status <= 599):
        return JSONResponse({"error": "status must be in [100, 599]"}, status_code=400)
    fut, err = await _resolve_target_future(body.get("pending_id"))
    if err is not None:
        return err
    err = _safe_set_result(fut, {
        "_inject_error": True,
        "status": status,
        "type": str(body.get("type", "api_error")),
        "message": str(body.get("message", "fake_server injected error")),
    })
    if err is not None:
        return err
    return {"ok": True}


@app.get("/_control/history")
async def history() -> dict[str, Any]:
    return {"turn_count": state.turn_count, "history": state.history}


@app.get("/_control/stats")
async def stats() -> dict[str, Any]:
    """history から累計のコスト目安・トークン・キャッシュサマリを集計する。

    note: 全て概算 (approx tokenizer)。実課金とは一致しない (README 参照)。
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_usd": 0.0,
        "cache_savings_usd": 0.0,
    }
    by_model: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    cache_misses = 0
    completed = 0
    errors = 0

    async with state.lock:
        hist = list(state.history)

    for e in hist:
        if e.get("injected_error") is not None:
            errors += 1
            continue
        usage = e.get("usage") or {}
        cost = e.get("cost") or {}
        cache = e.get("cache") or {}
        model = e.get("model") or "unknown"
        completed += 1

        for k in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens"):
            totals[k] += int(usage.get(k, 0))
        totals["total_usd"] += float(cost.get("total_usd", 0.0))

        read = int(usage.get("cache_read_input_tokens", 0))
        totals["cache_savings_usd"] += pricing.cache_savings_usd(model, read)

        cstatus = cache.get("status")
        if cstatus == "hit":
            cache_hits += 1
        elif cstatus == "miss":
            cache_misses += 1

        m = by_model.setdefault(model, {"requests": 0, "total_usd": 0.0,
                                        "input_tokens": 0, "output_tokens": 0})
        m["requests"] += 1
        m["total_usd"] = round(m["total_usd"] + float(cost.get("total_usd", 0.0)), 6)
        m["input_tokens"] += int(usage.get("input_tokens", 0))
        m["output_tokens"] += int(usage.get("output_tokens", 0))

    cache_total = cache_hits + cache_misses
    totals["total_usd"] = round(totals["total_usd"], 6)
    totals["cache_savings_usd"] = round(totals["cache_savings_usd"], 6)
    return {
        "is_estimate": True,
        "turn_count": state.turn_count,
        "completed_requests": completed,
        "error_requests": errors,
        "totals": totals,
        "cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": round(cache_hits / cache_total, 4) if cache_total else 0.0,
            "index_size": len(state.cache.index),
        },
        "by_model": by_model,
    }


@app.get("/_control/cache")
async def cache_index() -> dict[str, Any]:
    """擬似プロンプトキャッシュの現在の index (prefix hash 別)。"""
    now = time.time()
    # entries() は state.cache.index を反復する。register_request は lock 内で
    # cache.observe → index を mutate するため、ここも lock 内で snapshot を取らないと
    # 並行スキャン中に "dictionary changed size during iteration" になる。
    async with state.lock:
        entries = state.cache.entries(now)
    return {
        "ttl_seconds": state.cache.ttl_seconds,
        "honor_ttl": state.cache.honor_ttl,
        "entries": entries,
    }


@app.post("/_control/clear")
async def clear() -> dict[str, Any]:
    async with state.lock:
        # 全 in-flight pending を 503 でキャンセル (main handler は graceful に 503 を返す)
        for entry in state.pending.values():
            fut = entry["future"]
            if not fut.done():
                fut.set_exception(RuntimeError("cleared by control"))
        state.pending.clear()
        state.history.clear()
        state.turn_count = 0
        state.cache.reset()
    return {"ok": True}


# ── provider アダプタの登録 ──────────────────────────────────────────
# bedrock router は canonical helper (register_request / await_resolution /
# stream_event_dicts / _build_non_stream_response) を call-time 参照するため、
# 全 helper 定義後の末尾で import & include する (循環 import 回避)。

from .providers import bedrock as _bedrock  # noqa: E402

app.include_router(_bedrock.build_router())


# ── stand-alone 起動 ─────────────────────────────────────────────────


def main() -> int:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Fake Anthropic/Bedrock API server for debugging")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"[puppetllm] starting on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[puppetllm] Anthropic: set ANTHROPIC_BASE_URL=http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[puppetllm] Bedrock:   point AnthropicBedrock base_url to http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
