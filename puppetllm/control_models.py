"""Typed request bodies of the `/_control/*` endpoints.

The handlers themselves keep their hand-written validation (and their `{"error": ...}`
400s, which existing harnesses depend on); these models are the published contract —
they fill the request-body schemas in `/openapi.json`, so generated clients and `/docs`
show real fields. Nested models are registered under `components.schemas`
(`COMPONENT_SCHEMAS`, merged into the document by `fake_server`).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContentBlock(BaseModel):
    """A canonical content block (`text` / `tool_use` / `thinking` / `redacted_thinking`)."""
    model_config = ConfigDict(extra="allow")
    type: str


class Usage(BaseModel):
    """A usage override: at least one key (null token counts are treated as unset)."""
    model_config = ConfigDict(extra="allow")
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _not_empty(self) -> "Usage":
        if not self.model_dump(exclude_none=True):
            raise ValueError("usage must carry at least one key")
        return self


class Latency(BaseModel):
    """Milliseconds. `delay_ms` before the answer; `ttfb_ms` before the first stream
    frame; `chunk_delay_ms` between frames; `jitter_ms` a seeded random addition."""
    delay_ms: int | None = Field(default=None, ge=0, le=600_000)
    ttfb_ms: int | None = Field(default=None, ge=0, le=600_000)
    chunk_delay_ms: int | None = Field(default=None, ge=0, le=600_000)
    jitter_ms: int | None = Field(default=None, ge=0, le=600_000)


class Target(BaseModel):
    """Which pending an injection addresses (omit all when exactly one is pending)."""
    pending_id: str | None = None
    custom_id: str | None = None
    batch_id: str | None = None


class RespondBody(Target, Latency):
    text: str | None = Field(default=None, description="shorthand for a single text block")
    content: list[ContentBlock] = Field(default_factory=list)
    stop_reason: str | None = None
    stop_sequence: str | None = None
    stop_details: dict[str, Any] | None = None
    usage: Usage | None = None


class AutoBody(Target, Latency):
    text: str = "(empty)"


class ErrorBody(Target, Latency):
    status: int = Field(default=500, ge=100, le=599)
    type: str | None = None
    message: str = "fake_server injected error"
    code: str | None = None
    param: str | None = None
    headers: dict[str, str | int | float] | None = None
    after_events: int | None = Field(default=None, ge=0)
    original_status: int | None = Field(default=None, ge=100, le=599)
    content: list[ContentBlock] = Field(default_factory=list)


class RuleMatch(BaseModel):
    provider: Literal["anthropic", "bedrock", "openai"] | None = None
    model: str | None = Field(default=None, description="fnmatch glob against the model id")
    tools: list[str] | None = Field(default=None, description="every listed tool must be offered")
    has_tool_result: bool | None = None
    last_user_text: str | None = Field(default=None, description="regex over the last user turn")
    turn: int | None = Field(default=None, ge=1)
    stream: bool | None = None


class RespondStep(Latency):
    model_config = ConfigDict(extra="forbid")
    respond: dict[str, Any] = Field(description="a /_control/respond body (or {\"text\": ...})")


class ErrorStep(Latency):
    model_config = ConfigDict(extra="forbid")
    error: dict[str, Any] = Field(description="a /_control/error body")


class Rule(BaseModel):
    id: str | None = None
    match: RuleMatch = Field(default_factory=RuleMatch)
    steps: list[RespondStep | ErrorStep] = Field(min_length=1)
    repeat: bool = False


class RulesBody(BaseModel):
    rules: list[Rule]


class RateLimit(BaseModel):
    """Per-minute budgets over a sliding window; at least one must be set."""
    rpm: int | None = Field(default=None, ge=1)
    itpm: int | None = Field(default=None, ge=1)
    otpm: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _at_least_one(self) -> "RateLimit":
        if self.rpm is None and self.itpm is None and self.otpm is None:
            raise ValueError("rate_limit needs at least one of rpm / itpm / otpm")
        return self


class ConfigBody(BaseModel):
    """Every field optional; only the fields present are changed (`null` resets one)."""
    pending_timeout_s: float | None = Field(default=None, ge=0)
    on_unmatched: Literal["pending", "default", "error"] | None = None
    default_response: dict[str, Any] | str | None = Field(
        default=None, description="a /_control/respond body, or the text to answer with")
    unmatched_error: dict[str, Any] | None = None
    timeout_error: dict[str, Any] | None = None
    latency: Latency | None = None
    rate_limit: RateLimit | None = None
    seed: int | None = None


class ClockAdvanceBody(BaseModel):
    seconds: float = Field(ge=0, le=10 * 365 * 86400)


class ClearBody(BaseModel):
    """`POST /_control/clear` — an empty body keeps the configuration."""
    config: bool = Field(default=False, description="also restore the harness configuration")


class BatchEndBody(BaseModel):
    """`POST /_control/batch/end` — force a batch to `ended`."""
    batch_id: str | None = Field(default=None, description="omit when exactly one batch is live")
    unresolved: Literal["expired", "canceled"] = "expired"


class BatchResultBody(BaseModel):
    """`POST /_control/batch/result` — a lifecycle result for one custom_id."""
    custom_id: str = Field(min_length=1)
    type: Literal["expired", "canceled"]
    batch_id: str | None = None


class CountTokensBody(BaseModel):
    """`POST /v1/messages/count_tokens` — the Messages request minus `max_tokens`."""
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[dict[str, Any]]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None


# Nested model schemas referenced by the request bodies, keyed by model name; the server
# merges them into the OpenAPI document's `components.schemas`.
COMPONENT_SCHEMAS: dict[str, Any] = {}


def schema_of(model: type[BaseModel]) -> dict[str, Any]:
    """`openapi_extra` fragment declaring `model` as the JSON request body. Nested
    definitions are hoisted into `COMPONENT_SCHEMAS` and referenced as
    `#/components/schemas/<Name>` so the references resolve from the document root."""
    schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
    COMPONENT_SCHEMAS.update(schema.pop("$defs", {}))
    return {"requestBody": {"required": True, "content": {"application/json": {"schema": schema}}}}
