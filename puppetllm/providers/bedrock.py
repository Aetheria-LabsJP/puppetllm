"""Bedrock (InvokeModel / InvokeModelWithResponseStream) 互換アダプタ。

正式仕様: README.md

`AnthropicBedrock` SDK / boto3 `bedrock-runtime` の base_url を fake server に向けると、
リクエストは以下の形で来る (Anthropic 経路との差分):

- model は **URL パス** `/model/{model_id}/invoke[-with-response-stream]` に入る (body に無い)
- body は `anthropic_version` 付きの messages 形式 (`max_tokens` / `system` / `messages` / `tools`)
- streaming は SSE ではなく **AWS event stream バイナリ** (`application/vnd.amazon.eventstream`)

canonical core (fake_server) の register_request / await_resolution / stream_event_dicts /
_build_non_stream_response をそのまま流用し、ここでは「URL から model 抽出」と
「event stream エンコード」だけを担当する。応答注入 (/_control/respond 等) は Anthropic 経路と完全共通。
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import eventstream

# canonical core。⚠ 循環 import 注意: fake_server は末尾でこの module を import して
# build_router() を呼ぶ。ここでは module 参照だけ持ち、`fs.register_request` 等の属性は
# **必ず call-time (route handler 内) に解決**すること — import 時に fs.* へ触れると
# 半分しか実行されていない fake_server を掴んで AttributeError になる。通常動線
# (fake_server 経由起動) では問題ないが、`providers.bedrock` を単体 import するテスト等を
# 書く場合はこの順序制約を踏むので注意。
from .. import fake_server as fs

_EVENTSTREAM_MEDIA = "application/vnd.amazon.eventstream"


def _bedrock_error_response(status: int, etype: str, message: str) -> JSONResponse:
    """Bedrock 風のエラー応答 (status + __type + x-amzn-ErrorType header)。"""
    return JSONResponse(
        {"message": message, "__type": etype},
        status_code=status,
        headers={"x-amzn-ErrorType": etype},
    )


def build_router() -> APIRouter:
    router = APIRouter()

    # NOTE: `{model_id}` は単一パスセグメント。foundation model id
    # (`anthropic.claude-3-5-sonnet-20241022-v2:0`) やクロスリージョン推論プロファイル
    # (`us.anthropic.claude-...`) は `/` を含まないので OK。`/` を含む application
    # inference profile の ARN は ASGI 側で `%2F`→`/` 復号され routing にマッチしない
    # (404)。debug 用途では未対応 (アプリは plain model id を使う)。

    @router.post("/model/{model_id}/invoke")
    async def invoke(model_id: str, request: Request) -> Any:
        body, err = await fs._parse_json_body(request)
        if err is not None:
            return err

        snapshot, fut = await fs.register_request("bedrock", model_id, body, is_stream=False)
        result = await fs.await_resolution(snapshot, fut)

        if result["kind"] == "cleared":
            return _bedrock_error_response(503, "ServiceUnavailableException",
                                           f"request cleared: {result['detail']}")
        if result["kind"] == "error":
            return _bedrock_error_response(
                result["status"], result["type"], result["message"]
            )

        # Bedrock invoke (non-stream) は model の素の応答 body (= Anthropic message JSON) を返す。
        return JSONResponse(fs._build_non_stream_response(
            result["message_id"], model_id, result["content_blocks"], result["usage"]
        ))

    @router.post("/model/{model_id}/invoke-with-response-stream")
    async def invoke_stream(model_id: str, request: Request) -> Any:
        body, err = await fs._parse_json_body(request)
        if err is not None:
            return err

        snapshot, fut = await fs.register_request("bedrock", model_id, body, is_stream=True)
        result = await fs.await_resolution(snapshot, fut)

        if result["kind"] == "cleared":
            return _bedrock_error_response(503, "ServiceUnavailableException",
                                           f"request cleared: {result['detail']}")
        if result["kind"] == "error":
            # streaming 開始前のエラーは HTTP status で返す (SDK は status で例外マッピング)。
            return _bedrock_error_response(
                result["status"], result["type"], result["message"]
            )

        events = fs.stream_event_dicts(
            result["message_id"], model_id, result["content_blocks"], result["usage"]
        )
        frames = [eventstream.encode_chunk(data) for _name, data in events]

        async def gen():
            for frame in frames:
                yield frame
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type=_EVENTSTREAM_MEDIA)

    return router
