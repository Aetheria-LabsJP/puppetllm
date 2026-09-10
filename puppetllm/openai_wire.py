"""Pure conversions between OpenAI Chat Completions wire shapes and canonical (Anthropic-style)
blocks. Dependency-free (no FastAPI) so the relay can import it without loading the server.

Canonical objects may carry **private, underscore-prefixed keys** (`_openai_custom`,
`_openai_detail`) that let the OpenAI encoder / relay reproduce OpenAI-only details. They are
puppetllm-internal: `strip_private()` removes them before anything is put on an Anthropic /
Bedrock wire (the real Messages API rejects unknown keys).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

_DATA_URL_RE = re.compile(r"^data:(?P<media>[^;,]+);base64,(?P<data>.*)$", re.DOTALL)


# The only private keys puppetllm ever adds. Nothing else is touched: application data such
# as a tool input `{"_id": ...}`, a schema property `_id`, or the `_raw` wrapper for broken
# tool arguments is legitimate JSON that must reach the wire intact.
PRIVATE_KEYS = frozenset({"_openai_custom", "_openai_detail"})


def strip_private(obj: Any) -> Any:
    """Copy `obj` dropping puppetllm's private keys at block / tool level.

    Recurses through lists and through `content` (message → blocks → nested tool_result
    content) only — never into `input`, `input_schema`, `source` or other application data.
    """
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    if isinstance(obj, dict):
        return {k: (strip_private(v) if k == "content" else v)
                for k, v in obj.items() if k not in PRIVATE_KEYS}
    return obj


def openai_call_id(raw: Any) -> str:
    """Map a canonical tool_use id to the OpenAI `call_...` id form.

    The canonical (Anthropic-style) core assigns `toolu_...` ids to tool_use blocks
    that lack one; rewrite that prefix to `call_` (deterministic, so the same block
    yields the same id in both the non-stream and stream encoders); ids the responder
    set explicitly are kept as-is; a truly empty id falls back to a fresh `call_...`.
    """
    s = str(raw or "")
    if s.startswith("toolu_"):
        return "call_" + s[len("toolu_"):]
    return s or f"call_{uuid.uuid4().hex[:24]}"


def tool_call_to_canonical(tc: dict[str, Any]) -> dict[str, Any] | None:
    """One OpenAI tool call (function or custom) → canonical tool_use block."""
    if tc.get("type") == "custom" or ("custom" in tc and "function" not in tc):
        cu = tc.get("custom")
        cu = cu if isinstance(cu, dict) else {}
        return {"type": "tool_use", "id": str(tc.get("id") or ""),
                "name": str(cu.get("name") or ""), "input": {"input": cu.get("input")},
                "_openai_custom": True}
    fn = tc.get("function")
    fn = fn if isinstance(fn, dict) else {}
    raw = fn.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        args = {"_raw": raw}  # don't swallow broken arguments; expose them as-is
    return {"type": "tool_use", "id": str(tc.get("id") or ""),
            "name": str(fn.get("name") or ""), "input": args}


def tool_call_from_canonical(b: dict[str, Any], *, keep_id: bool = False) -> dict[str, Any]:
    """canonical tool_use block → OpenAI tool call (function, or custom when tagged).

    `keep_id=True` keeps the block's own id verbatim (relay: the app's ids must round-trip).
    """
    call_id = str(b.get("id") or "") if keep_id and b.get("id") else openai_call_id(b.get("id"))
    if b.get("_openai_custom"):
        inp = b.get("input")
        raw = inp.get("input") if isinstance(inp, dict) else inp
        return {"id": call_id, "type": "custom",
                "custom": {"name": str(b.get("name", "")),
                           "input": raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)}}
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": str(b.get("name", "")),
            "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
        },
    }


def image_url_to_canonical(part: dict[str, Any]) -> dict[str, Any] | None:
    """OpenAI `image_url` part → canonical (Anthropic-style) `image` block.

    data: URLs become base64 sources, http(s) URLs become url sources — the two shapes
    the Messages API accepts, so a relay to a real Anthropic endpoint can forward them.
    The OpenAI-only `detail` is kept as the private `_openai_detail` key.
    """
    iu = part.get("image_url")
    url = iu.get("url") if isinstance(iu, dict) else iu
    if not isinstance(url, str) or not url:
        return None
    m = _DATA_URL_RE.match(url)
    if m:
        blk: dict[str, Any] = {"type": "image", "source": {
            "type": "base64", "media_type": m.group("media"), "data": m.group("data")}}
    else:
        blk = {"type": "image", "source": {"type": "url", "url": url}}
    if isinstance(iu, dict) and iu.get("detail") is not None:
        blk["_openai_detail"] = iu["detail"]
    return blk


def image_to_openai(b: dict[str, Any]) -> dict[str, Any] | None:
    """canonical image block → OpenAI `image_url` part (base64 → data URL, url → url)."""
    src = b.get("source")
    if not isinstance(src, dict):
        return None
    if src.get("type") == "base64" and src.get("data"):
        iu: dict[str, Any] = {"url": f"data:{src.get('media_type') or 'image/png'};base64,{src['data']}"}
    elif src.get("type") == "url" and src.get("url"):
        iu = {"url": src["url"]}
    else:
        return None
    if b.get("_openai_detail") is not None:
        iu["detail"] = b["_openai_detail"]
    return {"type": "image_url", "image_url": iu}
