"""The model catalogue behind `/v1/models`.

A catalogue, not a gate: any model id an app sends is accepted, and
`GET /v1/models/{id}` synthesizes an entry for an id that is not listed here. Limits,
capabilities and lifecycle states follow the providers' published model tables as of
2026-11; they describe the real models and do not change how this server answers a
request. Retired models are listed without capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# One fixed creation timestamp for every entry, so the catalogue is stable across
# restarts. It is a placeholder, not the model's release date.
CREATED = 1_767_225_600
CREATED_AT = "2026-01-01T00:00:00Z"

_M = 1_000_000


@dataclass(frozen=True)
class ClaudeModel:
    id: str
    display_name: str
    line: str
    max_input_tokens: int = _M
    max_tokens: int = 128_000
    # Accepted `thinking.type` values.
    adaptive: bool = True
    enabled: bool = False          # manual `budget_tokens`
    disabled: bool = True
    between_tools: bool = False
    # Highest effort level: None (no `effort`), "high", "max" (no xhigh) or "xhigh" (all).
    effort: str | None = "xhigh"
    compaction: bool = True
    programmatic_tools: bool = True
    aliases: tuple[str, ...] = ()
    # The published state, not derived from the dates: a deprecated model whose
    # retirement date has passed stays deprecated until the provider retires it.
    lifecycle: str = "active"
    deprecated_at: str | None = None
    retires_at: str | None = None


def _always_thinking(model_id: str, name: str, line: str) -> ClaudeModel:
    return ClaudeModel(model_id, name, line, disabled=False)


def _retired(model_id: str, name: str, line: str, max_tokens: int, deprecated: str,
             retires: str, *aliases: str) -> ClaudeModel:
    return ClaudeModel(model_id, name, line, 200_000, max_tokens, aliases=aliases,
                       lifecycle="retired", deprecated_at=f"{deprecated}T00:00:00Z",
                       retires_at=f"{retires}T00:00:00Z")


# Most recently released first, as the real listing orders them.
CLAUDE_MODELS: tuple[ClaudeModel, ...] = (
    _always_thinking("claude-fable-5-2", "Claude Fable 5.2", "fable"),
    _always_thinking("claude-mythos-5-2", "Claude Mythos 5.2", "mythos"),
    ClaudeModel("claude-haiku-5-5", "Claude Haiku 5.5", "haiku"),
    ClaudeModel("claude-sonnet-5-5", "Claude Sonnet 5.5", "sonnet",
                disabled=False, between_tools=True),
    _always_thinking("claude-opus-5-5", "Claude Opus 5.5", "opus"),
    _always_thinking("claude-fable-5-1", "Claude Fable 5.1", "fable"),
    _always_thinking("claude-mythos-5-1", "Claude Mythos 5.1", "mythos"),
    ClaudeModel("claude-opus-5", "Claude Opus 5", "opus"),
    ClaudeModel("claude-sonnet-5", "Claude Sonnet 5", "sonnet"),
    _always_thinking("claude-fable-5", "Claude Fable 5", "fable"),
    _always_thinking("claude-mythos-5", "Claude Mythos 5", "mythos"),
    ClaudeModel("claude-opus-4-8", "Claude Opus 4.8", "opus"),
    ClaudeModel("claude-opus-4-7", "Claude Opus 4.7", "opus"),
    ClaudeModel("claude-mythos-preview", "Claude Mythos Preview", "mythos", enabled=True,
                disabled=False, effort="max", lifecycle="deprecated",
                deprecated_at="2026-06-09T00:00:00Z"),
    ClaudeModel("claude-sonnet-4-6", "Claude Sonnet 4.6", "sonnet", enabled=True, effort="max"),
    ClaudeModel("claude-opus-4-6", "Claude Opus 4.6", "opus", enabled=True, effort="max"),
    ClaudeModel("claude-opus-4-5-20251101", "Claude Opus 4.5", "opus", 200_000, 64_000,
                adaptive=False, enabled=True, effort="high", compaction=False,
                aliases=("claude-opus-4-5",)),
    ClaudeModel("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "haiku", 200_000, 64_000,
                adaptive=False, enabled=True, effort=None, compaction=False,
                programmatic_tools=False, aliases=("claude-haiku-4-5",)),
    ClaudeModel("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "sonnet", 200_000, 64_000,
                adaptive=False, enabled=True, effort=None, compaction=False,
                aliases=("claude-sonnet-4-5",), lifecycle="deprecated",
                deprecated_at="2026-09-30T00:00:00Z", retires_at="2026-11-30T00:00:00Z"),
    _retired("claude-opus-4-1-20250805", "Claude Opus 4.1", "opus", 32_000,
             "2026-06-05", "2026-08-05", "claude-opus-4-1"),
    _retired("claude-opus-4-20250514", "Claude Opus 4", "opus", 32_000,
             "2026-04-14", "2026-06-15", "claude-opus-4-0"),
    _retired("claude-sonnet-4-20250514", "Claude Sonnet 4", "sonnet", 64_000,
             "2026-04-14", "2026-06-15", "claude-sonnet-4-0"),
    _retired("claude-3-7-sonnet-20250219", "Claude Sonnet 3.7", "sonnet", 64_000,
             "2025-10-28", "2026-02-19", "claude-3-7-sonnet-latest"),
    _retired("claude-3-5-haiku-20241022", "Claude Haiku 3.5", "haiku", 8_192,
             "2025-12-19", "2026-02-19", "claude-3-5-haiku-latest"),
    _retired("claude-3-haiku-20240307", "Claude Haiku 3", "haiku", 4_096,
             "2026-02-19", "2026-04-20"),
)

OPENAI_MODELS: tuple[str, ...] = (
    "gpt-6.2-astra", "gpt-6.2-sol", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-astra",
    "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.3-codex",
    "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o",
    "o3",
)

LIFECYCLES = ("active", "deprecated", "retired")
_EFFORT_LEVELS = ("low", "medium", "high", "max", "xhigh")
_EFFORT_UP_TO = {None: (), "high": ("low", "medium", "high"),
                 "max": ("low", "medium", "high", "max"), "xhigh": _EFFORT_LEVELS}


def claude_model(model_id: str) -> ClaudeModel | None:
    """The catalogue entry for an id or one of its aliases."""
    for model in CLAUDE_MODELS:
        if model_id == model.id or model_id in model.aliases:
            return model
    return None


def _supported(flag: bool) -> dict[str, bool]:
    return {"supported": flag}


def _capabilities(m: ClaudeModel) -> dict[str, Any]:
    levels = _EFFORT_UP_TO[m.effort]
    return {
        "batch": _supported(True),
        "citations": _supported(True),
        "code_execution": _supported(m.programmatic_tools),
        "context_management": {
            "clear_thinking_20251015": _supported(True),
            "clear_tool_uses_20250919": _supported(True),
            "compact_20260112": _supported(m.compaction),
            "supported": True,
        },
        "effort": {**{level: _supported(level in levels) for level in _EFFORT_LEVELS},
                   "supported": bool(levels)},
        "image_input": _supported(True),
        "pdf_input": _supported(True),
        "server_tools": {"code_execution": _supported(True), "supported": True,
                         "web_search": _supported(True)},
        "structured_outputs": _supported(True),
        "thinking": {"supported": True, "types": {
            "adaptive": _supported(m.adaptive),
            "between_tools": _supported(m.between_tools),
            "disabled": _supported(m.disabled),
            "enabled": _supported(m.enabled),
        }},
    }


def anthropic_entry(model_id: str) -> dict[str, Any]:
    """The Models API object for an id or alias; an unlisted id gets an entry without
    limits, line or capabilities."""
    m = claude_model(model_id)
    if m is None:
        return {"type": "model", "id": model_id, "display_name": model_id,
                "created_at": CREATED_AT, "lifecycle": "active", "deprecated_at": None,
                "retires_at": None, "line": None, "max_input_tokens": None,
                "max_tokens": None, "capabilities": None}
    return {"type": "model", "id": m.id, "display_name": m.display_name,
            "created_at": CREATED_AT, "lifecycle": m.lifecycle,
            "deprecated_at": m.deprecated_at, "retires_at": m.retires_at, "line": m.line,
            "max_input_tokens": m.max_input_tokens, "max_tokens": m.max_tokens,
            "capabilities": None if m.lifecycle == "retired" else _capabilities(m)}


def openai_entry(model_id: str) -> dict[str, Any]:
    return {"id": model_id, "object": "model", "created": CREATED, "owned_by": "puppetllm"}
