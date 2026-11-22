**English** | [日本語](ja/quickstart.md) · [← README](../README.md)

# Quickstart

## The three actors

Think of it as **three actors**:

```
  +---- app / SDK -----+         +---- puppetllm ----+        +-- responder ---+
  | messages.create()  | ------> | POST /v1/messages | -----> | inject reply   |
  | blocks for reply   | <------ | held as pending   | <----- | /_control/...  |
  +--------------------+  reply  +-------------------+        +----------------+
         (1) app                    (2) fake server             (3) human / AI
```

(2) **holds (pending)** the request (1) sends; when (3) pushes a response via `/_control/*`, (1)'s `create()` returns with that response. The real API is never called.

## 1. Start the proxy

```bash
# A) Docker (recommended)
docker compose up -d
curl localhost:8765/_control/health        # → {"ok":true,"turn_count":0}

# B) Directly (Python 3.10+) — runs in foreground with a startup banner
pip install -r requirements.txt
python3 -m puppetllm --host 127.0.0.1 --port 8765
#   (or `pip install .` and use the `puppetllm` command: `puppetllm serve --port 8765`,
#    `puppetllm relay ...`, `puppetllm wait` (block until the server is healthy),
#    `puppetllm --version`; `pip install ".[test]"` adds the anthropic / openai / boto3 SDKs)
#   [puppetllm] starting on http://127.0.0.1:8765
#   [puppetllm] Anthropic: set ANTHROPIC_BASE_URL=http://127.0.0.1:8765
#   [puppetllm] Bedrock:   point AnthropicBedrock base_url to http://127.0.0.1:8765
#   [puppetllm] OpenAI:    set OPENAI_BASE_URL=http://127.0.0.1:8765/v1  (note the /v1)

# C) uvicorn directly (when you want options like --reload)
python3 -m uvicorn puppetllm.fake_server:app --host 127.0.0.1 --port 8765
```

`--host` defaults to `127.0.0.1` (localhost only). Use `0.0.0.0` only when accessing over LAN/VPN (see [Security](control-api.md#security)).

Trying the cache observation with toy prompts? They never reach a real model's minimum
cacheable prefix (512–4096 tokens), so `cache.status` stays `none`: start with
`PUPPETLLM_CACHE_MIN_TOKENS=0` to cache every prefix (see [Environment variables](control-api.md#environment-variables)).

`serve` also takes the harness settings of [Testing guide](testing.md): `--pending-timeout SECONDS`, `--default-response TEXT_OR_JSON`, `--on-unmatched pending|default|error`, `--seed N`, `--config FILE`, `--rules FILE` (the same settings are available as environment variables for Docker).

## 2. Point your app / SDK at the proxy

**Change nothing in your code** — just swap `base_url`.

**Anthropic SDK:**

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8765", api_key="sk-mock-anything")

# blocks until a response is injected
msg = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(msg.content)          # → the injected content blocks
print(msg.usage)            # → approximate input/output tokens + cache
```

The API key can be a dummy (the proxy does not validate it). Instead of `base_url`, setting the env var `ANTHROPIC_BASE_URL=http://localhost:8765` works identically (intercept without touching code). `stream=True` SSE works as-is too.

A pending waits as long as the responder takes, so raise the SDK's `timeout` (the Anthropic SDK defaults to 10 minutes, the OpenAI SDK too, but a 30 s `httpx` timeout in your own code will fire first). When you inject errors, remember that the SDKs retry 429 / 5xx twice by default with backoff — pass `max_retries=0` to see every injected error exactly once, or keep the retries to exercise them (each retry is a new pending).

**Bedrock (`AnthropicBedrock` and boto3, including Converse):** see [Bedrock](bedrock.md).

**OpenAI SDK (`openai`):**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8765/v1", api_key="sk-mock-anything")
msg = client.chat.completions.create(
    model="gpt-6.2-sol", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

Note the base_url **includes `/v1`** (the SDK appends `/chat/completions`); the env var `OPENAI_BASE_URL=http://localhost:8765/v1` works identically. Streaming (`stream=True`) and tool calls work as-is. OpenAI-format requests are **normalized to the canonical Anthropic-style form** (system / messages / tools, tool results as `tool_result` blocks) before being held, so the responder reads the same shape regardless of provider — and injects the same canonical blocks; the server converts them back to the `chat.completion` format. The pseudo prompt-cache is **not** simulated on this route (OpenAI's caching is automatic, not `cache_control`-based): cache status is always `"none"`.

OpenAI extras: with `n > 1` the responder can answer each choice individually, and a request with `logprobs: true` gets deterministic per-token log probabilities unless the responder injects its own (`choices` / `logprobs` on [`/_control/respond`](control-api.md)).

## 3. Supply the response (responder)

From another terminal / session, inject a response into the pending request.

```bash
# See what is pending
curl -s localhost:8765/_control/pending | jq
# → {"has_pending":true,"count":1,"pending":[
#      {"pending_id":"a1b2...","request":{"model":"...","system":...,"messages":[...],"tools":[...]},
#       "waiting_for_seconds":1.2}], ...}

# (a) quick: inject text only
curl -s -X POST localhost:8765/_control/auto \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello from the puppet!"}'

# (b) inject arbitrary content blocks including tool_use
curl -s -X POST localhost:8765/_control/respond \
  -H 'Content-Type: application/json' \
  -d '{"content": [
        {"type": "text", "text": "Let me check the weather."},
        {"type": "tool_use", "id": "tu_1", "name": "get_weather",
         "input": {"city": "Tokyo"}}
      ]}'
```

Besides `text` / `tool_use`, the injected content may contain `thinking` (`{"type":"thinking","thinking":"…","signature"?}` — an opaque signature is generated when omitted) and `redacted_thinking` (`{"data":"…"}`) blocks, which are kept in the response, streamed as `thinking_delta` / `signature_delta`, counted in `usage.output_tokens_details.thinking_tokens`, and expected back verbatim on the next turn — exactly the shape current models return by default. Server-side tool blocks are accepted as fixtures too (never executed): `server_tool_use` / `mcp_tool_use` (an id is generated when omitted; streamed like `tool_use` with `input_json_delta`) and the result blocks `web_search_tool_result` / `web_fetch_tool_result` / `code_execution_tool_result` / `bash_code_execution_tool_result` / `text_editor_code_execution_tool_result` / `tool_search_tool_result` / `mcp_tool_result` (carried verbatim; streamed as one `content_block_start` / `content_block_stop` pair; not counted as output tokens). `usage.server_tool_use` counts the `web_search` / `web_fetch` calls and the cost adds the official web-search price (`cost.server_tool_usd`). A `server_tool_use` does not end the turn (`stop_reason` stays `end_turn`). The blocks travel on the Anthropic route, Bedrock InvokeModel and the Mantle alias; the Converse and OpenAI routes drop them and, since the client never saw a search, do not count or bill them (history still holds the fixture). Any other block type is dropped and reported back in the injection's answer (`{"ok": true, "dropped": ["tool_result"]}`), or refused with 400 under `config.strict_blocks` — a `tool_result` is the app's part of the exchange and belongs in its next request. `stop_reason` accepts the documented vocabulary (`end_turn` / `max_tokens` / `stop_sequence` / `tool_use` / `pause_turn` / `refusal` / `model_context_window_exceeded`, plus `guardrail_intervened` for Bedrock and `content_filter` for the OpenAI route); `"refusal"` yields a `stop_details` object (`{"type":"refusal","category":null,"explanation":null}` unless you pass `stop_details`), `"stop_sequence"` fills `stop_sequence` from the request's `stop_sequences` unless you pass one.

Returning a `tool_use` makes the app run the real tool → the result is appended as `tool_result` to the next `messages.create()`, which becomes pending again. Repeating this reproduces an entire multi-turn / tool-execution loop.

**Responder loop (long-poll wait pattern):**

```bash
# Wait up to 270s for the next pending. Respond when one arrives; on timeout, loop again.
while true; do
  r=$(curl -s "localhost:8765/_control/wait_for_pending?timeout=270")
  echo "$r" | jq -e '.has_pending' >/dev/null || continue   # timeout → wait again
  pid=$(echo "$r" | jq -r '.pending_id')
  # ... read the request's system/messages/tools and build a response ...
  curl -s -X POST localhost:8765/_control/respond \
    -H 'Content-Type: application/json' \
    -d "{\"pending_id\":\"$pid\",\"content\":[{\"type\":\"text\",\"text\":\"...\"}]}"
done
```

The responder can be any of three things — they all use the same control plane and can be swapped freely (even mid-session):

1. **A human** with curl (as above).
2. **An AI agent playing the LLM** (Claude Code / Codex reading the request and improvising a faithful response) — the instruction docs are [`responder/CLAUDE.md`](../responder/CLAUDE.md) (for Claude Code) / [`responder/AGENTS.md`](../responder/AGENTS.md) (for Codex CLI and other agents following the `AGENTS.md` convention). Both cover the core principle of staying neutral, multi-pending handling, the injection format, pitfalls, and JSON-escape traps (content is nearly identical; only the runtime assumptions differ).
3. **The bundled relay** forwarding to a real API ([relay mode](relay.md)).

## 4. Inject error responses to test handling

For branch testing, you can make a pending request return any HTTP error (converted to the provider's native error format on all three routes — Anthropic / Bedrock / OpenAI). Optional `code` / `param` fields are passed through on the OpenAI route (e.g. `"code": "rate_limit_exceeded"`):

On the Bedrock route the body becomes `{"message": "...", "__type": "<AwsException>"}` with an `x-amzn-ErrorType` header (the two 424s wrap an upstream failure and carry its status as `originalStatusCode` — pass `original_status` to `/_control/error` to make it differ from the HTTP status, e.g. a 424 wrapping a 429; `ModelErrorException` adds `resourceName`, `ModelStreamErrorException` adds `originalMessage`, as botocore models them). When no `type` is given, the Anthropic route derives it from the status too (429 → `rate_limit_error`, 529 → `overloaded_error`, 400 → `invalid_request_error`, …), so the `anthropic` SDK raises its specific class. If `type` is already an AWS exception name (ends in `Exception`) it is used as-is; otherwise it is derived from `status`: 400 → `ValidationException`, 401 → `UnrecognizedClientException`, 403 → `AccessDeniedException`, 404 → `ResourceNotFoundException`, 408/504 → `ModelTimeoutException`, 413 → `RequestEntityTooLargeException`, 424 → `ModelErrorException`, 429 → `ThrottlingException`, 500 → `InternalServerException`, 503 → `ServiceUnavailableException`, 529 → `overloaded_error` (kept as-is; whether the live service wraps an upstream 529 instead is unverified) — other 4xx → `ValidationException`, other 5xx → `InternalServerException`. Pass an AWS name explicitly for the rest (`ServiceQuotaExceededException` 400, `ModelNotReadyException` 429, `ModelStreamErrorException` 424). So the same `{"status": 429, "type": "rate_limit_error"}` injection yields a `ThrottlingException` for a Bedrock client and a `rate_limit_error` for an Anthropic client.

For a **streaming** request you can also fail mid-stream, the way the real APIs do — add
`after_events` (and optionally the `content` to emit first). The response then starts as a
normal 200 stream, emits that many events, and ends with the provider's error event:
`event: error` on the Anthropic route, an event-stream **exception frame** on the Bedrock
routes. What the SDKs raise: the first-party `anthropic` client raises its usual
`APIStatusError` from the SSE `error` event; boto3 raises `botocore.exceptions.EventStreamError`
whose `Error.Code` is the frame's member name (`throttlingException`, …); `AnthropicBedrock`
raises a bare `ValueError` from its stream decoder — not an `anthropic.APIError` — so catch
accordingly. The frame's member is restricted to the operation's union
(`internalServer` / `modelStreamError` / `validation` / `throttling` / `serviceUnavailable`,
plus `modelTimeout` on InvokeModel only): a name outside it keeps its status class (a
429-class `ModelNotReadyException` becomes `throttlingException`; 408/504 stay
`modelTimeoutException` on InvokeModel and become `modelStreamErrorException` with
`originalStatusCode` on ConverseStream, whose union has no timeout member), never a silent
internal error. `after_blocks` is the route-independent form: fail after that many complete content blocks (the message start plus N block start/stop groups, whatever events the route uses to carry them; `0` fails right after the stream opened). Both apply to the Anthropic, Bedrock InvokeModel / Mantle and Converse streams; the OpenAI route has no mid-stream error form and answers a plain HTTP error (nothing partial is recorded for it). `after_events` counts each route's own events, so the same number
delivers different content per route (the InvokeModel stream and SSE have a
`content_block_start`, ConverseStream starts a text block with its first delta):

```bash
curl -s -X POST localhost:8765/_control/error \
  -d '{"status": 429, "type": "ThrottlingException", "message": "slow down",
       "after_events": 3, "content": [{"type": "text", "text": "partial answer"}]}'
```

The count is clamped so the terminal events are never emitted — a stream that failed
mid-flight never also looks like it completed. It counts protocol events only: the SSE
route still sends its usual `ping` after `message_start`, and that `ping` is not counted.
The history entry carries
`injected_error.partial_content` (the content the responder supplied for the partial
stream) alongside `after_events` — only for a streaming request, since only there did it go
on the wire. A `redacted_thinking` block's `data` must be base64 in a `respond` / `error`
`content` (the Bedrock SDKs decode it client-side; plain text would crash the caller with a
`binascii.Error`), so `/_control/*` refuses it with a 400 that says so. Non-streaming requests ignore `after_events` and get the plain
HTTP error (the OpenAI route always does).

```bash
# 429 → the SDK retries automatically
curl -s -X POST localhost:8765/_control/error \
  -d '{"status": 429, "type": "rate_limit_error", "message": "throttled"}'

# 401 → not retried (verify the auth-error branch)
curl -s -X POST localhost:8765/_control/error \
  -d '{"status": 401, "type": "authentication_error", "message": "bad key"}'
```

`status` must be an integer in 100–599. Out-of-range / non-numeric values return `400` and leave the pending untouched (the caller doesn't hang and you can retry the injection).

## Parallel requests (multi-pending)

The server can hold multiple concurrent requests. Each pending has a unique `pending_id`; inject into each individually by specifying `pending_id` on `/_control/respond` (also `auto` / `error`).

- Omitting `pending_id` is allowed only when there is **exactly one** pending. Zero → `400`; multiple → `400` (the response includes `pending_ids` so you can pick one).
- Injecting into a pending that no longer exists (already resolved, or wiped by `clear`) returns `400` (`no pending request`); only a near-simultaneous double-injection race returns `409` (`already resolved`).

How to build injection payloads (especially avoiding escape accidents with non-ASCII + nested JSON) is covered in detail in [`responder/CLAUDE.md`](../responder/CLAUDE.md) / [`responder/AGENTS.md`](../responder/AGENTS.md).
