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


def encode_event(event_type: str, payload: dict[str, Any]) -> bytes:
    """Wrap a raw-JSON event (ConverseStream style: `messageStart`, `contentBlockDelta`, …)
    into an event stream frame. Unlike InvokeModel's `chunk`, the payload is the event's
    JSON body itself, not a base64 `bytes` wrapper."""
    headers = {
        ":event-type": event_type,
        ":content-type": "application/json",
        ":message-type": "event",
    }
    return encode_message(headers, json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def encode_exception(exception_type: str, message: str,
                     extra: dict[str, Any] | None = None) -> bytes:
    """Wrap an error as an event stream exception message (mid-stream errors).

    `exception_type` is the stream's member name in lowerCamel (`throttlingException`,
    `validationException`, `modelStreamErrorException`, `internalServerException`,
    `serviceUnavailableException`, `modelTimeoutException`); botocore resolves it through
    the `:exception-type` header and raises `EventStreamError`.
    """
    body: dict[str, Any] = {"message": message}
    if extra:
        body.update(extra)
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        ":exception-type": exception_type,
        ":content-type": "application/json",
        ":message-type": "exception",
    }
    return encode_message(headers, payload)


def _decode_headers(raw: bytes) -> dict[str, str]:
    out: dict[str, str] = {}
    off = 0
    while off < len(raw):
        name_len = raw[off]
        name = raw[off + 1:off + 1 + name_len].decode("utf-8")
        off += 1 + name_len
        vtype = raw[off]
        off += 1
        if vtype != _HEADER_TYPE_STRING:
            raise ValueError(f"unsupported header value type {vtype}")
        vlen = struct.unpack(">H", raw[off:off + 2])[0]
        out[name] = raw[off + 2:off + 2 + vlen].decode("utf-8")
        off += 2 + vlen
    return out


def decode_frames(data: bytes) -> list[tuple[dict[str, str], bytes]]:
    """Parse an encoded byte sequence into (headers, payload) frames. CRCs are verified
    (ValueError if corrupted)."""
    out: list[tuple[dict[str, str], bytes]] = []
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
        out.append((_decode_headers(msg[12:12 + headers_len]), msg[12 + headers_len:-4]))
        off += total_len
    return out


def decode_messages(data: bytes) -> list[dict[str, Any]]:
    """For testing/verification: parse an encoded byte sequence and extract event dicts.

    InvokeModel `chunk` frames are unwrapped (payload.bytes base64 → JSON); raw-JSON
    event frames (ConverseStream) are returned as their JSON with `_event` set to the
    `:event-type`; exception frames as `{"_exception": <type>, "message": ...}`.
    """
    out: list[dict[str, Any]] = []
    for headers, payload in decode_frames(data):
        if headers.get(":message-type") == "exception":
            body = json.loads(payload) if payload else {}
            out.append({"_exception": headers.get(":exception-type"), **body})
            continue
        etype = headers.get(":event-type")
        if etype == "chunk":
            try:
                wrapper = json.loads(payload)
                out.append(json.loads(base64.b64decode(wrapper["bytes"])))
                continue
            except (KeyError, ValueError):
                out.append({"_raw": payload.decode("utf-8", errors="replace")})
                continue
        try:
            body = json.loads(payload)
        except ValueError:
            body = {"_raw": payload.decode("utf-8", errors="replace")}
        if isinstance(body, dict):
            body = {"_event": etype, **body}
        out.append(body)
    return out
