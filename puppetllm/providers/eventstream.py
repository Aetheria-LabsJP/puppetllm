"""AWS event stream (vnd.amazon.eventstream) binary encoder (pure).

Bedrock's `InvokeModelWithResponseStream` responds not with SSE but with Amazon's
event stream binary framing. The boto3 / anthropic[bedrock] SDK decodes it. For the
fake server to speak the Bedrock route, it must wrap each Anthropic stream event
(message_start / content_block_delta / ...) into a single `chunk` message and send it.

Frame structure (all big-endian):
  [total_len u32][headers_len u32][prelude_crc u32]  ← prelude (12 bytes)
  [headers ...]                                       ← headers_len bytes
  [payload ...]                                       ← total_len - headers_len - 16
  [message_crc u32]                                   ← CRC32 of prelude+headers+payload

header (one):
  [name_len u8][name][value_type u8][value...]
  when value_type=7 (string): [value_len u16][value]

chunk message:
  headers: :event-type=chunk, :content-type=application/json, :message-type=event
  payload: {"bytes": base64(<anthropic event json>)}

Reference: AWS event stream encoding spec (CRC32 uses the standard polynomial,
identical to zlib.crc32).
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
    """Serialize a single event stream message to binary."""
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
    """Wrap an Anthropic stream event dict into a Bedrock `chunk` frame."""
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
    """Wrap an error as an event stream exception message (for mid-stream errors).

    NOTE: Currently unused. bedrock.py resolves error injection **before** the stream
    starts, so it returns via HTTP status (the SDK maps exceptions by status). This is
    provided as a utility for when mid-stream error injection is implemented in the
    future (e.g. verifying throttling triggered after N chunks are emitted).
    """
    payload = json.dumps({"message": message}).encode("utf-8")
    headers = {
        ":exception-type": exception_type,
        ":content-type": "application/json",
        ":message-type": "exception",
    }
    return encode_message(headers, payload)


def decode_messages(data: bytes) -> list[dict[str, Any]]:
    """For testing/verification: parse an encoded byte sequence and extract event dicts.

    Only chunk messages are targeted (payload.bytes base64 decode → JSON).
    CRC is verified (ValueError if corrupted).
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
            # For non-chunk messages (exception etc.), store the raw payload
            out.append({"_raw": payload.decode("utf-8", errors="replace")})
        off += total_len
    return out
