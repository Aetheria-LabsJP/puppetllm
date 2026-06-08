"""AWS event stream (vnd.amazon.eventstream) バイナリエンコーダ (pure)。

Bedrock の `InvokeModelWithResponseStream` は SSE ではなく Amazon の event stream
バイナリフレーミングで応答する。boto3 / anthropic[bedrock] SDK はこれをデコードする。
fake server が Bedrock 経路を喋るには、Anthropic の各ストリームイベント (message_start /
content_block_delta / ...) を 1 つの `chunk` メッセージに包んで送る必要がある。

フレーム構造 (全て big-endian):
  [total_len u32][headers_len u32][prelude_crc u32]  ← prelude (12 bytes)
  [headers ...]                                       ← headers_len bytes
  [payload ...]                                       ← total_len - headers_len - 16
  [message_crc u32]                                   ← prelude+headers+payload の CRC32

header (1 つ):
  [name_len u8][name][value_type u8][value...]
  value_type=7 (string) のとき: [value_len u16][value]

chunk メッセージ:
  headers: :event-type=chunk, :content-type=application/json, :message-type=event
  payload: {"bytes": base64(<anthropic event json>)}

参考: AWS event stream encoding 仕様 (CRC32 は標準多項式、zlib.crc32 と同一)。
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

_HEADER_TYPE_STRING = 7


def _encode_header(name: str, value: str) -> bytes:
    name_b = name.encode("utf-8")
    value_b = value.encode("utf-8")
    return (
        struct.pack("B", len(name_b))
        + name_b
        + struct.pack("B", _HEADER_TYPE_STRING)
        + struct.pack(">H", len(value_b))
        + value_b
    )


def encode_message(headers: dict[str, str], payload: bytes) -> bytes:
    """1 つの event stream メッセージをバイナリ化する。"""
    headers_b = b"".join(_encode_header(k, v) for k, v in headers.items())
    headers_len = len(headers_b)
    total_len = 12 + headers_len + len(payload) + 4  # prelude(12) + headers + payload + msg_crc(4)

    prelude = struct.pack(">I", total_len) + struct.pack(">I", headers_len)
    prelude_crc = zlib.crc32(prelude) & 0xFFFFFFFF
    prelude_with_crc = prelude + struct.pack(">I", prelude_crc)

    message_wo_crc = prelude_with_crc + headers_b + payload
    message_crc = zlib.crc32(message_wo_crc) & 0xFFFFFFFF
    return message_wo_crc + struct.pack(">I", message_crc)


def encode_chunk(event_data: dict[str, Any]) -> bytes:
    """Anthropic のストリームイベント dict を Bedrock `chunk` フレームに包む。"""
    inner = json.dumps(event_data, ensure_ascii=False).encode("utf-8")
    payload = json.dumps(
        {"bytes": base64.b64encode(inner).decode("ascii")}
    ).encode("utf-8")
    headers = {
        ":event-type": "chunk",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    return encode_message(headers, payload)


def encode_exception(exception_type: str, message: str) -> bytes:
    """エラーを event stream の exception メッセージとして包む (mid-stream エラー用)。

    NOTE: 現状未使用。bedrock.py のエラー注入は stream 開始**前**に解決するため HTTP status で
    返している (SDK は status で例外マッピング)。将来 stream 途中でのエラー注入を実装する際の
    ユーティリティとして用意してある (例: throttling を N チャンク送出後に発生させる検証)。
    """
    payload = json.dumps({"message": message}).encode("utf-8")
    headers = {
        ":exception-type": exception_type,
        ":content-type": "application/json",
        ":message-type": "exception",
    }
    return encode_message(headers, payload)


def decode_messages(data: bytes) -> list[dict[str, Any]]:
    """テスト/検証用: encode した bytes 列をパースして event dict を取り出す。

    chunk メッセージのみ対象 (payload.bytes を base64 decode → JSON)。
    CRC は検証する (壊れていれば ValueError)。
    """
    out: list[dict[str, Any]] = []
    off = 0
    n = len(data)
    while off < n:
        if off + 12 > n:
            raise ValueError("truncated prelude")
        total_len, headers_len = struct.unpack(">II", data[off:off + 8])
        prelude_crc = struct.unpack(">I", data[off + 8:off + 12])[0]
        if (zlib.crc32(data[off:off + 8]) & 0xFFFFFFFF) != prelude_crc:
            raise ValueError("prelude CRC mismatch")
        if off + total_len > n:
            raise ValueError("truncated message")
        msg = data[off:off + total_len]
        body_crc = struct.unpack(">I", msg[-4:])[0]
        if (zlib.crc32(msg[:-4]) & 0xFFFFFFFF) != body_crc:
            raise ValueError("message CRC mismatch")
        payload = msg[12 + headers_len:-4]
        try:
            wrapper = json.loads(payload)
            inner = base64.b64decode(wrapper["bytes"])
            out.append(json.loads(inner))
        except (KeyError, ValueError):
            # chunk 以外 (exception 等) は素の payload を入れておく
            out.append({"_raw": payload.decode("utf-8", errors="replace")})
        off += total_len
    return out
