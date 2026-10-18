"""The JSON response class every route uses.

A lone surrogate in client text (`"\\ud800"`) is legal JSON but not legal UTF-8: Starlette's
`JSONResponse` would fail to encode it, turning a request the server accepted into a 500.
This one renders unescaped when it can and falls back to `\\uXXXX` escapes — the client
decodes either to the same string.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import JSONResponse as _JSONResponse


class SafeJSONResponse(_JSONResponse):
    def render(self, content: Any) -> bytes:
        text = json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None,
                          separators=(",", ":"), default=str)
        try:
            return text.encode("utf-8")
        except UnicodeEncodeError:
            return json.dumps(content, ensure_ascii=True, allow_nan=False, indent=None,
                              separators=(",", ":"), default=str).encode("ascii")
