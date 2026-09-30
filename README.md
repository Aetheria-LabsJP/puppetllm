**English** | [日本語](README.ja.md)

# puppetllm — LLM API debug proxy (fake Anthropic / Bedrock / OpenAI server)

A **fake server** compatible with the Anthropic Messages API / Bedrock (InvokeModel, Converse, batch inference) / OpenAI Chat Completions. Just point `ANTHROPIC_BASE_URL` (or the `AnthropicBedrock` / `OpenAI` `base_url`) at this server and it intercepts LLM calls **without changing a single line of your app / SDK code**, letting a human or another agent supply the responses (human-in-the-loop / AI-in-the-loop).

Use cases:

- **Zero-cost debugging**: reproduce and inspect agent / orchestration behavior without hitting the real API.
- **Deterministic testing**: inject arbitrary responses (text / tool_use, errors) to reproduce branches.
- **Scripted test harness ([§8](#8-test-harness-scenario-rules-timeouts-latency-rate-limits-fake-clock))**: match requests to canned answers with rules, script sequences such as 429 → tool_use → final text, simulate latency and rate limits, and assert what your app sent — from CI with no responder present (`puppetllm.testing`, a pytest fixture).
- **Cross-provider bridge ([relay mode](#relay-mode-cross-provider-bridge))**: run an app written for one SDK against a *different* real provider (e.g. an Anthropic-SDK agent on Grok / GPT, or an OpenAI-SDK app on Claude) — without changing a line of app code.
- **Cost estimates**: aggregate approximate tokens / pricing per request (`/_control/stats`).
- **Pseudo prompt-cache observation**: verify by hash whether your app structures requests so `cache_control` actually takes effect (`/_control/cache`).

> All figures are based on an approximate tokenizer, so they **do not match real billing**. Use them for trend analysis and structural verification.

---

## Architecture

A provider-agnostic canonical core + adapters:

- `puppetllm/fake_server.py` — canonical core (normalized snapshot management + `/_control/*` + cost/cache computation). The Anthropic route `POST /v1/messages` is built in.
- `puppetllm/providers/bedrock.py` — Bedrock route `POST /model/{id}/invoke[-with-response-stream]` (model-id normalization, `anthropic_version` validation, AWS-style errors / headers; AWS event stream framing lives in `providers/eventstream.py`).
- `puppetllm/providers/converse.py` — Bedrock Converse route `POST /model/{id}/converse[-stream]` (the Converse JSON schema translated to and from the canonical form).
- `puppetllm/providers/bedrock_batch.py` + `puppetllm/providers/s3.py` — Bedrock batch inference (`/model-invocation-job*`) over a bundled directory-backed S3 emulation.
- `puppetllm/providers/openai.py` — OpenAI route `POST /v1/chat/completions` (requests are normalized to the canonical Anthropic-style form; responses are converted back to `chat.completion` JSON / SSE chunks).
- `puppetllm/batches.py` — Anthropic Message Batches route `/v1/messages/batches*` (each custom_id is held as an ordinary pending; batch lifecycle is injectable via `/_control/batch/*`).
- `puppetllm/cache_sim.py` — pseudo prompt cache (multi-breakpoint + top-level automatic `cache_control` + prefix match + generation-aware minimum threshold + 5m / 1h TTLs + 20-block lookback + effort / thinking / tool_choice invalidation).
- `puppetllm/pricing.py` — approximate tokens + price table (Claude generations incl. Fable / Mythos / Sonnet 5 / Opus 4.1, and GPT / o-series families, per the official pricing pages) and the `/v1/models` catalogue.
- `puppetllm/harness.py` — test-harness layer: scenario rules, unmatched policy, pending timeout, latency / jitter, rate-limit window, fake clock. `puppetllm/control_models.py` holds the typed `/_control/*` request bodies published in `/openapi.json`.
- `puppetllm/testing.py` + `puppetllm/pytest_plugin.py` — the test-side client (`Puppet`, `serve()`) and the `puppet` pytest fixture.

Providers are auto-selected by URL path — no mode switch or configuration. Response content blocks / control API are common across providers (injection is always the same `/_control/respond`).

---

## Usage

Think of it as **three actors**:

```
  +---- app / SDK -----+         +---- puppetllm ----+        +-- responder ---+
  | messages.create()  | ------> | POST /v1/messages | -----> | inject reply   |
  | blocks for reply   | <------ | held as pending   | <----- | /_control/...  |
  +--------------------+  reply  +-------------------+        +----------------+
         (1) app                    (2) fake server             (3) human / AI
```

(2) **holds (pending)** the request (1) sends; when (3) pushes a response via `/_control/*`, (1)'s `create()` returns with that response. The real API is never called.

### 1. Start the proxy

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

`--host` defaults to `127.0.0.1` (localhost only). Use `0.0.0.0` only when accessing over LAN/VPN (see [Security](#security)).

Trying the cache observation with toy prompts? They never reach a real model's minimum
cacheable prefix (512–4096 tokens), so `cache.status` stays `none`: start with
`PUPPETLLM_CACHE_MIN_TOKENS=0` to cache every prefix (see [Environment variables](#environment-variables)).

`serve` also takes the harness settings of [§8](#8-test-harness-scenario-rules-timeouts-latency-rate-limits-fake-clock): `--pending-timeout SECONDS`, `--default-response TEXT_OR_JSON`, `--on-unmatched pending|default|error`, `--seed N`, `--config FILE`, `--rules FILE` (the same settings are available as environment variables for Docker).

### 2. Point your app / SDK at the proxy

**Change nothing in your code** — just swap `base_url`.

**Anthropic SDK:**

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8765", api_key="sk-mock-anything")

# blocks until a response is injected
msg = client.messages.create(
    model="claude-sonnet-4-5", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(msg.content)          # → the injected content blocks
print(msg.usage)            # → approximate input/output tokens + cache
```

The API key can be a dummy (the proxy does not validate it). Instead of `base_url`, setting the env var `ANTHROPIC_BASE_URL=http://localhost:8765` works identically (intercept without touching code). `stream=True` SSE works as-is too.

A pending waits as long as the responder takes, so raise the SDK's `timeout` (the Anthropic SDK defaults to 10 minutes, the OpenAI SDK too, but a 30 s `httpx` timeout in your own code will fire first). When you inject errors, remember that the SDKs retry 429 / 5xx twice by default with backoff — pass `max_retries=0` to see every injected error exactly once, or keep the retries to exercise them (each retry is a new pending).

**Bedrock SDK (`AnthropicBedrock`):**

```python
from anthropic import AnthropicBedrock
client = AnthropicBedrock(base_url="http://localhost:8765", aws_region="us-east-1",
                          aws_access_key="dummy", aws_secret_key="dummy")
msg = client.messages.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

Requires the bedrock extra for SigV4 signing: `pip install 'anthropic[bedrock]'`. The AWS credentials can be any dummy values (the proxy doesn't validate the signature), but the SDK needs *something* to sign with. The model goes into the URL path (`/model/{id}/invoke`) and streaming comes back as an AWS event stream — the server absorbs both. **Injecting responses is exactly the same as the Anthropic route** (use the same `/_control/respond` below).

Bedrock-route specifics (all decided from the URL path, no mode switch):

- **Model-id normalization**: `anthropic.claude-haiku-4-5-20251001-v1:0`, suffix-less current ids (`anthropic.claude-opus-5`), cross-region inference profiles (`us.` / `eu.` / `apac.` / `jp.` / `au.` / `global.` / `us-gov.` … prefixes) and foundation-model / inference-profile ARNs (any partition: `aws`, `aws-cn`, `aws-us-gov`) are mapped to the Anthropic-side name (`claude-haiku-4-5-20251001`), which is what the pending snapshot's `model`, the response `model` field and `/_control/stats` `by_model` use — so Bedrock and direct-Anthropic traffic for the same model aggregate into one row, and relay's `--model-map` / `--only` globs match `claude-*`. The raw id is kept on the snapshot / history entry as `bedrock_model_id`. Ids without an `anthropic.` segment pass through unchanged.
- **`anthropic_version` is validated** for Anthropic model ids: missing or anything other than `bedrock-2023-05-31` is rejected up front with `400 ValidationException` (no pending is created) — catches hand-rolled clients early. Other vendors' ids (`meta.llama…`, `amazon.titan…`) have their own body shapes and pass through unchecked. Malformed `cache_control` layouts (see § 5) are rejected the same way.
- **SigV4 is not verified**: `Authorization` / `X-Amz-Date` / `X-Amz-Security-Token` are ignored.
- **Response headers**: non-stream responses carry `X-Amzn-Bedrock-Input-Token-Count` (non-cached input only, like Converse's `inputTokens`) / `X-Amzn-Bedrock-Output-Token-Count` / `X-Amzn-Bedrock-Cache-Read-Input-Token-Count` / `X-Amzn-Bedrock-Cache-Write-Input-Token-Count` / `X-Amzn-Bedrock-Invocation-Latency` (+ `x-amzn-requestid`); streams carry `amazon-bedrock-invocationMetrics` (incl. `cacheReadInputTokenCount` / `cacheWriteInputTokenCount`) on the final chunk and `X-Amzn-Bedrock-Content-Type`, as real Bedrock does. The options boto3 sends as **request headers** — `serviceTier=` (`X-Amzn-Bedrock-Service-Tier`, `priority | default | flex | reserved`) and `performanceConfigLatency=` (`X-Amzn-Bedrock-PerformanceConfig-Latency`) — are validated, echoed back in the same response headers on both routes (`default` when no tier was asked for), and the tier is made visible to the responder as canonical `service_tier` (`priority` / `flex` → `auto`, `default` / `reserved` → `standard_only`, the relay's vocabulary).
- **Messages-API alias for `AnthropicBedrockMantle`**: `POST /anthropic/v1/messages` (the path served by the `bedrock-runtime` / `bedrock-mantle` hosts) is the plain Anthropic handler with Bedrock model-id normalization, and the plain `/v1/messages` route hands any request whose `model` is a Bedrock id (`[region.]anthropic.…` / ARN) to the same handler — so `AnthropicBedrockMantle(base_url="http://localhost:8765")` works whether it posts to the root or to `/anthropic` (SSE streaming, Anthropic error envelope, `anthropic-version` header, no `anthropic_version` body field). Batches / count_tokens are not served on this alias, matching Bedrock.
- **Receipt log**: every Bedrock request (and every rejection) is logged to stderr as one `[bedrock] invoke model=<raw> -> <canonical> pending=<id> …` line, so Bedrock traffic is distinguishable from the other routes at a glance.

**boto3 (`bedrock-runtime`), including Converse:**

```python
import boto3
from botocore.config import Config

rt = boto3.client("bedrock-runtime", region_name="us-east-1",
                  aws_access_key_id="dummy", aws_secret_access_key="dummy",
                  endpoint_url="http://localhost:8765")

rt.converse(modelId="anthropic.claude-opus-5",
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
            inferenceConfig={"maxTokens": 1024})

for event in rt.converse_stream(modelId="anthropic.claude-opus-5",
                                messages=[{"role": "user", "content": [{"text": "hello"}]}])["stream"]:
    print(event)          # messageStart / contentBlockDelta / … / metadata
```

`POST /model/{id}/converse` and `/converse-stream` speak the Converse JSON schema and are
normalized to the **same canonical pending** as every other route, so a responder answers
them with the usual `/_control/respond` content blocks and never sees the difference:

| Converse | canonical |
|---|---|
| `{"text": …}` | `{"type": "text", …}` |
| `{"image": {"format", "source": {"bytes" \| "s3Location"}}}` / `{"document": …}` | `{"type": "image" \| "document", "source": {…}}` |
| `{"toolUse": {"toolUseId", "name", "input"}}` | `{"type": "tool_use", "id", "name", "input"}` |
| `{"toolResult": {"toolUseId", "content", "status"}}` | `{"type": "tool_result", "tool_use_id", "content", "is_error"}` |
| `{"reasoningContent": {"reasoningText": {"text", "signature"}}}` / `{"redactedContent"}` | `{"type": "thinking", …}` / `{"type": "redacted_thinking", "data"}` |
| `{"cachePoint": {"type": "default", "ttl"?}}` | `cache_control` on the **preceding** block / tool (a real breakpoint for the pseudo cache) |
| `inferenceConfig.{maxTokens,temperature,topP,stopSequences}` | `max_tokens` / `temperature` / `top_p` / `stop_sequences` |
| `toolConfig.tools[].toolSpec` / `toolChoice {auto\|any\|tool}` | `tools[]` / `tool_choice` |
| `additionalModelRequestFields` | merged into the canonical body (`thinking`, `top_k`, `anthropic_beta`, …); a key Converse models itself (`system`, `tools`, `max_tokens`, `temperature`, …) is a `ValidationException`, never a back door around the schema |
| `outputConfig.effort` / `outputConfig.textFormat` | `output_config.effort` (`low` … `xhigh` / `max`) / `output_config.format` (`{type: json_schema, schema, name?}` — `textFormat.structure.jsonSchema.schema` is the JSON **string** the API defines, decoded here into the schema object); the native fields win over the same keys sent through `additionalModelRequestFields`, and a non-object `output_config` there is a `ValidationException` |
| `promptVariables` (prompt-management ARN as the model id) | kept for the responder under `converse.promptVariables`; `messages` may then be absent, as on the real API — the stored prompt supplies them, so the responder sees `messages: []`; with an ordinary model id `messages` stays required |
| `serviceTier.type` | `service_tier` (`priority` / `flex` → `auto`, `default` / `reserved` → `standard_only`) |

Responses are encoded back from the block types an assistant turn can actually hold
(`text` / `toolUse` / `reasoningContent` — a responder's blocks are normalized to
text / tool_use / thinking / redacted_thinking before any encoder sees them):
`output.message`, `stopReason` in the Converse vocabulary (a canonical `refusal` becomes `content_filtered`, `pause_turn` becomes
`end_turn`), `usage` with `inputTokens` excluding cached tokens plus
`cacheReadInputTokens` / `cacheWriteInputTokens` / `cacheDetails`, and `metrics.latencyMs`.
`additionalModelResponseFieldPaths` (JSON pointers, ≤ 10) are resolved against the native
Messages-API response and also ride on `messageStop` in the stream. Requests are validated against the
shapes of the schema — not its every constraint: each content block, source,
`reasoningContent`, `toolChoice` and `system` entry is a **union** (exactly one member — a block carrying both
`text` and `cachePoint` is a `ValidationException`, not a silently dropped text; so is an
unknown member on a union object or an unknown top-level request member), and each
union takes only its own members (an `ImageSource` / `VideoSource` is `bytes | s3Location`;
only a `DocumentSource` also takes `text | content`). Required fields and enums
(`image.format`, `video.format`, `document.name`, `toolUse.input`, `toolResult.content`,
`cachePoint.type` / `ttl`, `inferenceConfig` ranges, `serviceTier.type`, `requestMetadata`
1–16 entries, a `guardContent.image` being `png` / `jpeg` from `bytes`) are enforced, and
two `cachePoint`s cannot address the same block. Not checked (a request production would
reject can still pass here): decoded media size and validity, cross-block placement rules,
the inner shape of `audio` / `searchResult` (only "must be an object"), user / assistant
alternation (and `role: "system"` is accepted, as the botocore model allows), and the
`guardrailConfig` / `promptVariables` / prompt-ARN conditionals. The stream emits `messageStart` → per block
`contentBlockStart` (tool use) / `contentBlockDelta` / `contentBlockStop` → `messageStop`
→ `metadata`, as raw-JSON event-stream frames (not the `chunk` + base64 wrapper the
InvokeModel route uses).

**OpenAI SDK (`openai`):**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8765/v1", api_key="sk-mock-anything")
msg = client.chat.completions.create(
    model="gpt-5.4", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

Note the base_url **includes `/v1`** (the SDK appends `/chat/completions`); the env var `OPENAI_BASE_URL=http://localhost:8765/v1` works identically. Streaming (`stream=True`) and tool calls work as-is. OpenAI-format requests are **normalized to the canonical Anthropic-style form** (system / messages / tools, tool results as `tool_result` blocks) before being held, so the responder reads the same shape regardless of provider — and injects the same canonical blocks; the server converts them back to the `chat.completion` format. The pseudo prompt-cache is **not** simulated on this route (OpenAI's caching is automatic, not `cache_control`-based): cache status is always `"none"`.

### 3. Supply the response (responder)

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

Besides `text` / `tool_use`, the injected content may contain `thinking` (`{"type":"thinking","thinking":"…","signature"?}` — an opaque signature is generated when omitted) and `redacted_thinking` (`{"data":"…"}`) blocks, which are kept in the response, streamed as `thinking_delta` / `signature_delta`, counted in `usage.output_tokens_details.thinking_tokens`, and expected back verbatim on the next turn — exactly the shape current models return by default. Server-side tool blocks are accepted as fixtures too (never executed): `server_tool_use` / `mcp_tool_use` (an id is generated when omitted; streamed like `tool_use` with `input_json_delta`) and the result blocks `web_search_tool_result` / `web_fetch_tool_result` / `code_execution_tool_result` / `bash_code_execution_tool_result` / `text_editor_code_execution_tool_result` / `tool_search_tool_result` / `mcp_tool_result` (carried verbatim; streamed as one `content_block_start` / `content_block_stop` pair; not counted as output tokens). `usage.server_tool_use` counts the `web_search` / `web_fetch` calls and the cost adds the official web-search price (`cost.server_tool_usd`). A `server_tool_use` does not end the turn (`stop_reason` stays `end_turn`). The blocks travel on the Anthropic route, Bedrock InvokeModel and the Mantle alias; the Converse and OpenAI routes drop them and, since the client never saw a search, do not count or bill them (history still holds the fixture). Any other block type is dropped and reported back in the injection's answer (`{"ok": true, "dropped": ["tool_result"]}`), or refused with 400 under `config.strict_blocks` — a `tool_result` is the app's part of the exchange and belongs in its next request. `stop_reason` accepts the documented vocabulary (`end_turn` / `max_tokens` / `stop_sequence` / `tool_use` / `pause_turn` / `refusal` / `model_context_window_exceeded`); `"refusal"` yields a `stop_details` object (`{"type":"refusal","category":null,"explanation":null}` unless you pass `stop_details`), `"stop_sequence"` fills `stop_sequence` from the request's `stop_sequences` unless you pass one.

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
2. **An AI agent playing the LLM** (Claude Code / Codex reading the request and improvising a faithful response) — the instruction docs are [`responder/CLAUDE.md`](responder/CLAUDE.md) (for Claude Code) / [`responder/AGENTS.md`](responder/AGENTS.md) (for Codex CLI and other agents following the `AGENTS.md` convention). Both cover the core principle of staying neutral, multi-pending handling, the injection format, pitfalls, and JSON-escape traps (content is nearly identical; only the runtime assumptions differ).
3. **The bundled relay** forwarding to a real API ([relay mode](#relay-mode-cross-provider-bridge) below).

### 4. Inject error responses to test handling

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

### 5. Observe cost / tokens / cache

```bash
# Cumulative summary (all approximate)
curl -s localhost:8765/_control/stats | jq
# → {"is_estimate":true,"completed_requests":3,"error_requests":0,
#     "totals":{"input_tokens":..,"output_tokens":..,
#               "cache_read_input_tokens":..,"total_usd":..,"cache_savings_usd":..},
#     "cache":{"hits":2,"misses":1,"hit_rate":0.6667,"index_size":2},
#     "by_model":{"claude-sonnet-4-5":{"requests":3,"total_usd":..}}}

# Pseudo prompt-cache index (hit/miss per prefix hash)
curl -s localhost:8765/_control/cache | jq

# Per-request (request, response, usage, cost, cache) history
curl -s localhost:8765/_control/history | jq '.history[-1]'

# Cleanup between tests (wipe pending / history / cache)
curl -s -X POST localhost:8765/_control/clear
```

`cache_savings_usd` is "the approximate amount you would have saved for real thanks to cache hits." Use it to verify your app structures `cache_control` correctly. (Anthropic / Bedrock routes only — the OpenAI route always reports cache status `"none"` and never touches the hit/miss counters.)

What the pseudo cache reproduces (per the official prompt-caching docs):

- **Placement**: block-level `cache_control` (≤ 4 breakpoints) and the **top-level** `cache_control` (automatic caching: one breakpoint on the last cacheable block — `thinking` and empty text blocks are skipped). Prefix order `tools → system → messages`; markers are not part of the key.
- **Minimum cacheable prefix is generation-aware**: Fable 5 / 5.1, Mythos, Opus 5 = 512 tokens; Opus 4.8 = 1024; Opus 4.7 = 2048; Opus 4.6 / 4.5 = 4096; Opus 4.1 / 4 = 1024; every Sonnet = 1024; Haiku 4.5 = 4096; Haiku 3.5 = 2048. Below that the request is observed as `"none"`, like the real API (no error).
- **TTL**: `{"type":"ephemeral"}` = 5 minutes, `{"type":"ephemeral","ttl":"1h"}` = 1 hour (`PUPPETLLM_CACHE_TTL` / `PUPPETLLM_CACHE_TTL_1H` to shorten both for tests), refreshed on read. Writes are reported split by TTL in `usage.cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` and priced at 1.25x / 2x input.
- **Invalidation**: changing `output_config.effort`, `thinking`, or `tool_choice` between turns invalidates the *messages*-level prefixes; switching `speed` (fast mode) invalidates system + messages but not tools. This is a deliberate simplification of the official invalidation table, which marks the tools / system caches as *model-specific* for thinking and effort (some models render that configuration ahead of the system prompt); puppetllm models them as messages-only. Explicit defaults equal omission: `effort: high`, `tool_choice: auto` (also with `disable_parallel_tool_use: false`), and the model's default thinking mode (`adaptive` on Opus 5 / Sonnet 5 / Fable / Mythos, `disabled` on earlier models; `display: omitted`). A model switch invalidates everything.
- **Validation**: layouts the real API rejects get a `400 invalid_request_error` here too (no pending is created): a malformed marker or unknown `ttl`, more than 4 explicit breakpoints, a marker on a `thinking` block, a 1h breakpoint after a 5m one, top-level `cache_control` when 4 explicit breakpoints already exist, and a top-level `ttl` that disagrees with an explicit marker on the last block.
- **Fast mode**: `speed: "fast"` is priced at the official 2x (Anthropic / Bedrock routes only), reported as `usage.speed: "fast"`, and relayed to an Anthropic upstream together with the app's `anthropic-beta` header (dropped with a warning towards OpenAI-compatible upstreams). Model / platform eligibility is not enforced.
- **Lookback**: each breakpoint looks back ≤ 20 positions, where a run of consecutive `tool_use` (or `tool_result`) blocks counts as one position.
- **Usage shape**: responses carry the full 2026 usage object (`cache_creation`, `output_tokens_details.thinking_tokens`, `server_tool_use`, `service_tier` (`standard` / `batch`), `inference_geo`, `speed` when fast mode was requested), and `message_delta.usage` repeats the cumulative input / cache counts as the real stream does.

Pricing follows the official per-generation table (e.g. Opus 5 $5/$25, Sonnet 5 $2/$10, Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5, Fable 5.1 $10/$50 with $0.25 cache reads, Opus 4.1 $15/$75). Unknown Claude ids fall back to Sonnet 4.x pricing; unknown `gpt-*` ids to gpt-5.4.

### 6. Message Batches API

The Anthropic Message Batches surface (`/v1/messages/batches*`) is also served, so `client.messages.batches.create()` / `retrieve()` / `results()` / `cancel()` / `list()` / `delete()` work as-is:

```python
batch = client.messages.batches.create(requests=[
    {"custom_id": "r1", "params": {"model": "claude-sonnet-4-5", "max_tokens": 64,
                                   "messages": [{"role": "user", "content": "hi"}]}},
    {"custom_id": "r2", "params": {...}},
])
# poll until "ended", then iterate client.messages.batches.results(batch.id)
```

Each `custom_id` becomes an **ordinary pending** whose snapshot additionally carries `batch_id` / `custom_id`. The responder injects with the same `/_control/respond` / `auto` / `error` — addressed by `pending_id`, or by `custom_id` (+ `batch_id` if the same custom_id is unresolved in several batches):

```bash
# succeeded for r1, errored for r2 (custom_id addressing)
curl -s -X POST localhost:8765/_control/respond \
  -d '{"custom_id": "r1", "content": [{"type": "text", "text": "batch reply"}]}'
curl -s -X POST localhost:8765/_control/error \
  -d '{"custom_id": "r2", "status": 500, "type": "api_error", "message": "boom"}'
```

The batch becomes `ended` automatically once every custom_id has a result. The control plane can also drive the lifecycle:

```bash
curl -s localhost:8765/_control/batches            # registry: status/counts/unresolved custom_ids

# inject a result type respond/error cannot express (canceled | expired) for one custom_id
curl -s -X POST localhost:8765/_control/batch/result \
  -d '{"custom_id": "r2", "type": "expired"}'

# force "ended" NOW; unresolved custom_ids become expired (or canceled)
curl -s -X POST localhost:8765/_control/batch/end \
  -d '{"batch_id": "msgbatch_...", "unresolved": "expired"}'
```

Deliberate divergences from the real API (determinism over fidelity):

- **No wall-clock expiration** — `expires_at` (created + 24h) is reported, but entries expire only via `/_control/batch/end` / `/_control/batch/result`.
- **Cancel is usually immediate** — `POST .../cancel` resolves all unresolved custom_ids as `canceled` and normally returns the batch already `ended`, skipping the real API's asynchronous `canceling` phase. An injection already in flight at that instant still completes as succeeded/errored (as on the real API, where already-processing requests may finish after a cancel); while any remain the batch reports `canceling`, then settles to `ended`.
- **Costs get the real 50% batch discount** — history entries carry `"batch": true` and `cost.batch_discount = 0.5`; `/_control/stats` aggregates the discounted figures. `canceled` / `expired` entries are not recorded in history (not billed, like the real API).
- Per-request `params` are only shallow-validated (params being an object; `stream: true`, `speed` (fast mode) and `max_tokens: 0` are rejected at create time as on the real API, while an item carrying `fallbacks` is accepted and comes back as an `errored` result without ever becoming a pending — also as on the real API); the envelope is checked as strictly as the real API (`custom_id` matching `^[a-zA-Z0-9_-]{1,64}$` and unique, ≤ 100,000 requests, `limit` in `[1, 1000]` / cursors on list) so an app that production would reject is rejected here too. Params that later fail processing (e.g. a non-list `messages`) roll the whole create back with a 400 — no batch, no pendings, no history left behind.
- `results_url` is built from the incoming request's Host. Behind a reverse proxy, run uvicorn with `--proxy-headers` (and a matching `FORWARDED_ALLOW_IPS`) so it reflects the external URL.

### 7. Bedrock batch inference (with a bundled S3)

The control-plane batch API is emulated too, over a directory-backed **S3 emulation** so a
plain `boto3` S3 client can stage the input and read the results:

```python
import boto3
from botocore.config import Config

s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="d", aws_secret_access_key="d",
                  endpoint_url="http://localhost:8765",
                  config=Config(s3={"addressing_style": "path"}))     # path-style is required
bedrock = boto3.client("bedrock", region_name="us-east-1", aws_access_key_id="d",
                       aws_secret_access_key="d", endpoint_url="http://localhost:8765")

s3.create_bucket(Bucket="batch-in"); s3.create_bucket(Bucket="batch-out")
s3.put_object(Bucket="batch-in", Key="jobs/input.jsonl", Body=b'''{"recordId": "r1", "modelInput": {...}}\n''')

job = bedrock.create_model_invocation_job(
    jobName="myjob", modelId="anthropic.claude-opus-5",
    roleArn="arn:aws:iam::123456789012:role/BatchRole",
    inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://batch-in/jobs/input.jsonl"}},
    outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://batch-out/results/"}})
```

Every `{"recordId", "modelInput"}` line becomes an ordinary **pending** (provider
`bedrock`, snapshot tagged `job_arn` / `job_id` / `record_id`), so the responder answers
them with the same `/_control/respond` / `auto` / `error`. Once every record has an outcome
the job writes `<output prefix>/<jobId>/<input file>.out` and `manifest.json.out`, and
ends `Completed`. The output file name keeps the key's path relative to the input prefix,
so `a/data.jsonl` and `b/data.jsonl` do not collide. Each output line is

```json
{"recordId": "r1", "modelInput": { … as submitted … },
 "modelOutput": { … }}                                  // or, on failure:
{"recordId": "r2", "modelInput": { … },
 "error": {"errorCode": 400, "errorMessage": "…"}}
```

`recordId` is the only correlation key (results are not ordered), and the manifest is

```json
{"totalRecordCount": 3, "processedRecordCount": 3, "successRecordCount": 2,
 "errorRecordCount": 1, "inputTokenCount": 120, "outputTokenCount": 48}
```

`modelInput` is an InvokeModel body (default) or a Converse body with
`modelInvocationType: "Converse"`; `modelOutput` is the matching response shape.
An `InvokeModel` body is model-specific, so those jobs need an Anthropic `modelId`;
`Converse` is a model-independent schema, so a `Converse` job takes any model. What the
bundled store cannot honour is refused rather than echoed: `s3EncryptionKeyId` (outputs are
plain files) is a `ValidationException`, and `s3BucketOwner` must be `123456789012`, the
one account the store's buckets and the job ARNs belong to.
`get_model_invocation_job` / `list_model_invocation_jobs` / `stop_model_invocation_job`
work as usual (`GET /_control/bedrock_jobs` shows the registry with unresolved record ids).

`Stop` finalizes synchronously — the status is already terminal when the call returns —
except while the input is still being read or an output write is already in flight, when it
returns `200` with the job in `Stopping` and the terminal status follows; the job ends
`Stopped`, the terminal status the real service reports for `StopModelInvocationJob` — an injection already in flight still
completes, but records that were never processed stay in `totalRecordCount` only, are counted
in neither `processedRecordCount` nor `errorRecordCount`, and are written to no output line
(they are listed under `cancelled` in `/_control/bedrock_jobs`). A stop is accepted while the
job is still `Submitted` or registering (whatever was not yet registered is never processed —
and if that job's input then fails validation, in the read or in the registration, it ends
`Failed` with the reason rather than vanishing from under the caller who stopped it), is
idempotent once the job is `Stopping` / `Stopped`
(botocore retries a Stop whose response it lost), a `Completed` / `Failed` job answers
`ConflictException`, and a stop accepted while the outputs are being written still decides
the terminal status. If the
outputs cannot be written (output bucket deleted mid-job, disk error) the job ends **`Failed`**
with the reason in `message` — whatever was written before the failure stays — so a poller
must treat `Completed` / `Stopped` / `Failed` as the terminal set. A record whose `modelInput`
is invalid becomes an `error` record instead of failing the job, like the real service; a
malformed JSONL line, a non-string or duplicate `recordId`, a missing input / output bucket or
an input with no records rolls the whole create back with a `ValidationException`. Features AWS documents as unavailable in batch come back as
`errorCode 400` records: tool calling, structured output, and prompt caching
(`cache_control` / `cachePoint` anywhere in the record).

Other intentional differences (determinism over fidelity): no `Validating` / `Scheduled`
phases (a job is `InProgress` as soon as its records are pending), no minimum record count
beyond "at least one", and no clock-based expiry. Input JSONL is read and held in memory, so
a job's practical size is bounded by RAM rather than by AWS's 1 GB / 50,000-record limits.
`/_control/clear` drops the job registry but deliberately leaves the S3 store alone, so
outputs a cleared job had already written stay readable; a create still in flight at that
moment — reading its input, registering, waiting on an idempotent twin, or finalizing
inline — answers `400 ConflictException` (deliberately not a retryable 5xx — botocore would
otherwise re-create the job the human had just cleared). The output layout is checked at
create time (`<prefix>/<jobId>/<file>.out` must fit the store's 255-byte segment limit and
neither the prefix nor any ancestor of it may be an existing object), so a job never ends
`Failed` on that after every record was processed. A `recordId` must be a non-empty string
when present (an empty one is not silently replaced by a generated id).

The S3 emulation covers exactly what that flow needs — `PUT`/`HEAD` bucket, `PUT` / `GET` /
`HEAD` / `DELETE` object, `GET /` (list buckets), `DELETE` bucket (`409 BucketNotEmpty`
unless empty), ranged `GET` (`Range` → `206` + `Content-Range`, `416 InvalidRange` past the
end), conditional requests (`If-Match` / `If-None-Match` → `304` on a read, `412 PreconditionFailed` on a write or a
delete, and `404 NoSuchKey` for an `If-Match` write or delete on a key that does not exist,
as the real service answers — atomically, so sixteen concurrent `If-None-Match: *` writers
get exactly one winner; the `If-*-Since` forms on reads only), conditional deletes
(`delete_object(IfMatch=…)`, `*` for "only if it exists"; a weak `W/` tag never authorizes a
write or a delete; the size / last-modified forms are directory-bucket features and answer
`501`), `ExpectedBucketOwner` enforcement (every bucket belongs to `123456789012`; a wrong
value in either the query or the `x-amz-` header is `403 AccessDenied`), request-checksum
verification (`400 BadDigest`, `IncompleteBody` when `x-amz-decoded-content-length`
disagrees, `InvalidRequest` for an `aws-chunked` body without its terminating chunk) and
both listings:
`GET ?list-type=2` and the V1 `GET` with `marker`, each with `prefix`, `delimiter`
(`CommonPrefixes`), `max-keys` (clamped to 1000 rather than refused) and `encoding-type=url`
(path-style, no auth, `aws-chunked` bodies decoded).

Two of those are not optional in practice: boto3's `download_file` splits anything over
`multipart_threshold` (8 MB) into concurrent ranged GETs, and botocore asks for
`encoding-type=url` on every listing — it URL-decodes the keys it reads back only when the
response echoes `EncodingType`, so the two have to be switched together.

**Everything else is refused, not approximated.** `CreateMultipartUpload`, `CopyObject`,
object and bucket tagging / ACL / versioning / policy, `ListObjectVersions`, `DeleteObjects`,
the other sub-resources, and the `put_object` options this store cannot store
(`Tagging`, `Metadata`, SSE, ACL / grants, object lock) answer `501 NotImplemented` in an S3
envelope; `ContentType` and the other plain entity headers are accepted but not stored
(objects read back as `binary/octet-stream`). This matters more than it sounds:
`copy_object`, `put_object_tagging` and `put_object_acl` all arrive as a `PUT` on the
object's own path, so a store that ignored the sub-resource would write their body — or
their empty body — over the object and report success. Presigned-URL query auth is ignored
like header auth. For multipart upload specifically, keep staged inputs under that 8 MB
threshold or lower it in `boto3.s3.transfer.TransferConfig`. Because the body is read
**before** any refusal is sent, a refusal never leaves the keep-alive connection out of
sync: botocore's `Expect: 100-continue` PUTs get their `100 Continue` before any answer —
and that holds for a path the router itself cannot represent (a control character in a
key, which is refused before routing). A non-`STANDARD` `StorageClass` is refused too
(everything here is STANDARD), and the request checksums botocore attaches
(`Content-MD5`, `x-amz-checksum-crc32` / `sha1` / `sha256`, header or `aws-chunked`
trailer) are verified — `400 BadDigest` on a mismatch — so a body corrupted in transit is
never stored quietly (`crc32c` / `crc64nvme` are accepted unverified).

Objects live under `PUPPETLLM_S3_ROOT` (default: a per-process temp directory) and can never
be written outside it: bucket names are validated everywhere (including inside `s3://` URIs),
keys with `.` / `..` segments are refused (and a single key segment is limited to 255 bytes,
what the backing filesystem accepts — a flat 1024-byte key is legal on S3 but not here), a
write needs an existing bucket, writes land through a rename from a temp directory outside
the bucket tree so a concurrent reader never sees a half-written object (one store-wide
lock also covers the batch emulation's worker thread, so its output writes cannot
interleave inside an HTTP handler's stat → precondition → write either; the handlers take
that lock off the event loop, so a thread holding it never freezes the other routes), and a malformed
`aws-chunked` body is a `400 InvalidRequest`. Control characters in a key are refused (this
store renders keys straight into the listing XML, where they would be illegal), and a key is
never silently rewritten — `PUT /bucket//a` is a `400` rather than a quiet write to `a`.
Bucket names that collide with an API path (`model`, `v1`, `anthropic`, `_control`,
`model-invocation-job[s]`, `docs`, `redoc`, `openapi.json`) are refused, and a request to one
of those paths with the wrong method gets the API's own `405` + `Allow`, never an S3 error
envelope.

---

### 8. Test harness: scenario rules, timeouts, latency, rate limits, fake clock

Everything above needs a responder. The harness answers requests **by itself** from a
script, so the same server runs in CI with nothing attached — and history / stats /
cache work exactly as with a human responder.

**Rules** match a request and answer it; each rule holds an ordered list of steps that
are consumed one per matching request (the last step repeats when `repeat: true`; an
exhausted rule stops matching and lets the next rule — or the unmatched policy — take
over). A step is a `/_control/respond` body (`{"text": ...}` shorthand allowed) or a
`/_control/error` body, plus optional latency keys:

```bash
curl -s -X PUT localhost:8765/_control/rules -H 'content-type: application/json' -d '{
  "rules": [
    {"id": "throttle-once", "match": {"provider": "anthropic"},
     "steps": [{"error": {"status": 429, "headers": {"retry-after": "1"}}}]},
    {"id": "weather", "match": {"tools": ["get_weather"], "has_tool_result": false},
     "steps": [{"respond": {"content": [{"type": "tool_use", "id": "toolu_1",
                                        "name": "get_weather", "input": {"city": "Tokyo"}}]}}]},
    {"id": "final", "match": {"has_tool_result": true, "last_user_text": "sunny"},
     "steps": [{"respond": {"text": "It is sunny in Tokyo."}, "ttfb_ms": 200}]}
  ]}'
curl -s localhost:8765/_control/rules | jq '.all_consumed, .unconsumed'
```

Match keys (all optional, all must hold): `provider` (`anthropic` / `bedrock` / `openai`),
`model` (glob), `tools` (every listed tool name must be offered), `has_tool_result` (the
last user turn carries `tool_result` blocks), `last_user_text` (regex over the last user
turn's text, `tool_result` text included), `turn`, `stream`. Rules are tried in order;
the first live match wins. `GET /_control/rules` reports per-rule `matched` /
`remaining` counters, `unconsumed` (rules that still hold steps nobody asked for) and
`all_consumed` — a test's "every scripted call happened" assertion. `POST` appends,
`PUT` replaces the list, `PUT /_control/rules/{id}` replaces one rule in place,
`DELETE /_control/rules[/{id}]` removes. History entries answered by the harness carry
`harness: {"source": "rule", "rule_id": ...}` (`default` / `unmatched` / `rate_limit` /
`timeout` for the other sources). Block types a step would lose are listed under
`dropped` in the rules response (or refused with 400 when `config.strict_blocks` is set).

**Unmatched policy and pending timeout** (`/_control/config`, kept across
`/_control/clear`): `on_unmatched` is `pending` (the interactive default: wait for a
responder), `default` (answer with `default_response`, a `/_control/respond` body or a
plain string) or `error` (answer with `unmatched_error`, a `/_control/error` body; default
500 `api_error`). `pending_timeout_s` makes the server answer a pending nobody has answered
with `timeout_error` (default 504 `api_error`; batch entries are exempt). The deadline is
fixed when the request arrives (a later config change applies to new requests only) and is
shown in `/_control/pending` (`deadline`, `timeout_in_seconds`). A client that hangs up
while pending has its pending dropped, so a responder never sees a request nobody is
waiting for; an answer that was already injected (a relay may have paid for it) is still
recorded even if the client hangs up during its `delay_ms`.

```bash
curl -s -X POST localhost:8765/_control/config -d '{"pending_timeout_s": 30,
  "on_unmatched": "default", "default_response": "canned answer",
  "latency": {"delay_ms": 100, "jitter_ms": 50}, "rate_limit": {"rpm": 60}, "seed": 1}'
```

**Latency**: `delay_ms` (before the response), `ttfb_ms` (before the first stream frame),
`chunk_delay_ms` (between frames) and `jitter_ms` (a random 0..N ms added to `delay_ms`,
drawn from a generator seeded by `seed` — reproducible) can be set on any injection
(`/_control/respond` / `auto` / `error`), on any rule step, and as defaults in
`config.latency`. A `/_control/clear` during the `delay_ms` wait wins (the request returns
the "cleared" error and is not recorded); a stream that has already started keeps its
`ttfb_ms` / `chunk_delay_ms` pacing to the end.

**Rate limit**: `config.rate_limit` = `{"rpm", "itpm", "otpm"}` (any subset) over a
sliding 60-second window. A request beyond the budget gets a 429 with `retry-after` and
the vendor's quota headers (`anthropic-ratelimit-*` on the Anthropic route,
`x-ratelimit-*` on the OpenAI route, a `ThrottlingException` on Bedrock) before any rule
is consulted. A refused request consumes no budget (though it takes a turn number and
lands in history as an error), so waiting `retry-after` seconds succeeds for any request
that fits the budget at all and meets no competing traffic; the quota headers carry the
remaining capacity and reset time of each configured dimension (Anthropic also gets the
combined `anthropic-ratelimit-tokens-*`). Input tokens are counted before the cache is
consulted, cached prefixes included. Batch entries are neither throttled nor charged.
`GET /_control/config` shows the current window.

**Default headers**: `config.default_headers` (`{"anthropic-ratelimit-requests-remaining": "42", ...}`)
are added to every API response — 200s, streams and the S3 emulation included, never the control plane or the docs —
unless the route set the same header itself (a limiter's 429 keeps its own values), so an
app that reads the vendor's quota headers off successful responses can be exercised.

**Fake clock**: `POST /_control/clock/advance {"seconds": N}` moves the server's clock
forward — pseudo prompt-cache TTLs, rate-limit windows and pending deadlines elapse
accordingly, without sleeping (`GET /_control/clock` shows the offset; `/_control/clear`
resets it).

**Token counting and model catalogue** (no pending is created, nothing is recorded):

| Method | Path | Description |
|---|---|---|
| POST | `/v1/messages/count_tokens` | Anthropic `count_tokens`: `{"input_tokens": N}`, the same estimate `/v1/messages` bills for the request as a whole (`usage.input_tokens` + `cache_creation_input_tokens` + `cache_read_input_tokens`) |
| POST | `/model/{modelId}/count-tokens` | Bedrock `CountTokens` (`{"input": {"invokeModel": {"body": <blob>}}}` — the Messages request, base64 on the wire as boto3 sends it, raw JSON accepted too — or `{"input": {"converse": {...}}}` → `{"inputTokens": N}`) |
| GET  | `/v1/models`, `/v1/models/{id}` | The catalogue (`pricing.KNOWN_MODELS`), in the Anthropic shape (`x-api-key` or `anthropic-version` present, or no auth header; pages with `limit` / `after_id` / `before_id`) or the OpenAI shape (bare `Authorization: Bearer`). Informational: any model id is accepted by the request routes, and an unlisted id is synthesized rather than refused |

**From Python tests** — `puppetllm.testing` wraps the control plane; `serve()` runs the
server in-process on a free port:

```python
import anthropic
from puppetllm.testing import serve

with serve() as puppet:                       # or Puppet("http://127.0.0.1:8765") for a running server
    puppet.expect(tools=["get_weather"], has_tool_result=False).respond(
        content=[{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Tokyo"}}])
    puppet.expect(has_tool_result=True).error(429, headers={"retry-after": "0"}).respond(text="sunny")
    client = anthropic.Anthropic(base_url=puppet.url, api_key="test", max_retries=0)
    ...run the code under test...
    puppet.assert_consumed()                  # every scripted step was used
    puppet.assert_sent(model="claude-*", contains="Tokyo", tool="get_weather")
    puppet.assert_no_pending()
```

`Puppet` also exposes the interactive surface (`wait_pending`, `respond`, `error`,
`pending`, `history`, `stats`, `cache`, `config`, `advance_clock`, `clear`;
`clear(config=True)` also restores the defaults, `reset(baseline)` re-applies a saved
`baseline()` — configuration plus rules). With `pip install .` the `puppet` pytest fixture
(one server per session; before and after every test the state is cleared and the
configuration and rules are put back to what the server started with, so `PUPPETLLM_*`
settings and a `--config` file's rules stay in force while one test's `puppet.config(...)`
or `puppet.expect(...)` never leaks into the next; `PUPPETLLM_URL` points it at an
external server instead) is registered
automatically; `pip install ".[test]"` adds the SDKs. A startup script can come from a
file — `puppetllm serve --config scenario.json` with
`{"config": {...}, "rules": [...]}` — or from environment variables (see
[Environment variables](#environment-variables)) for Docker.

---

## Relay mode (cross-provider bridge)

`python -m puppetllm.relay` is a bundled **automatic responder** that forwards every pending request to a **real** upstream API and injects the response back — turning puppetllm into a transparent cross-provider bridge. Your app keeps speaking its own SDK; the actual model behind it becomes swappable:

```bash
# An Anthropic-SDK app running on xAI Grok:
python -m puppetllm.relay --target https://api.x.ai/v1 \
    --api-key-env XAI_API_KEY --model grok-3

# An OpenAI-SDK app running on real Claude:
python -m puppetllm.relay --kind anthropic --model claude-sonnet-4-5

# Per-model routing instead of a single forced model:
python -m puppetllm.relay --model-map "claude-*=grok-3,gpt-*=grok-3-mini"

# Relay only OpenAI-model requests; answer the rest by hand (concurrent, partitioned by model):
python -m puppetllm.relay --only "gpt-*,o3-*" --model grok-3
```

- `--kind openai` (default) speaks to **any OpenAI-compatible endpoint** — OpenAI, xAI Grok, Groq, Ollama, OpenRouter, … just point `--target` at its base URL. `--kind anthropic` speaks to the native Anthropic API.
- Requests are translated from the canonical form (system / messages / tools incl. `strict` and OpenAI `custom` tools / tool_choice / stop / temperature / seed / verbosity / prompt_cache_key / metadata …; image blocks ↔ `image_url` parts in both directions); responses come back as canonical blocks — including `thinking` blocks with the upstream's real signatures, and an OpenAI `refusal` as text with `stop_reason: "refusal"` — with the **real `stop_reason` / `stop_details` and real token usage** (via the `stop_reason` / `stop_details` / `usage` fields of `/_control/respond`, the usage carrying `cache_creation` / `output_tokens_details` / `service_tier` objects intact), so `/_control/stats` aggregates real numbers (history entries get `"usage_overridden": true`). When the upstream omits usage, puppetllm's approximation is kept instead.
- Towards a real Anthropic upstream, OpenAI-inbound `response_format: json_schema` becomes `output_config.format` and `reasoning_effort` becomes `output_config.effort` (`none` / `minimal` → `low`); `thinking` / `output_config` / `cache_control` / `inference_geo` / `speed` from Anthropic-inbound apps are forwarded as-is, and the app's `anthropic-beta` header (captured as `params.anthropic_beta`) is re-sent to an Anthropic upstream so beta features (fast mode, compaction, …) keep working through the relay. Vendor vocabularies are mapped rather than forwarded: `service_tier` (`standard_only` ↔ `default`, `flex` / `priority` → `auto`), Anthropic `effort: max` → OpenAI `reasoning_effort: xhigh`; `n` is never forwarded (the relay uses one choice, extra samples would only be billed). puppetllm's private canonical keys (`_openai_custom`, `_openai_detail`) never reach a wire. Untranslatable parameters are dropped with a one-time warning.
- `--max-tokens-param` defaults to `auto`: `max_completion_tokens` when the target URL's hostname is exactly `api.openai.com` (where `max_tokens` is deprecated and rejected by reasoning models), `max_tokens` for other OpenAI-compatible backends. Force either value explicitly if a backend disagrees.
- Upstream API errors are relayed with their status/type/message (and `code`/`param`), so your app's SDK raises the same exception class it would against the real provider.
- The relay is *just another responder*. **By default it claims every pending it sees**, so it does not share a live queue with a human / AI-agent responder — the swap is sequential: stop the relay and take over by hand. To run them **concurrently**, use `--only "<glob>,…"` so the relay claims only the matching inbound models and leaves the rest for you (or another agent).
- `--max-concurrency N` caps simultaneous in-flight upstream calls (default: unlimited), so a burst of pendings doesn't fan out and trip upstream rate limits.

Caveats: the upstream call is non-streaming, so a streaming app sees correct SSE but with first-token latency equal to the full upstream response time; audio / file parts and `document` blocks are not translated (images are); costs are **real** in this mode; `/_control/stats` prices tokens by the *inbound* model id, which may differ from the upstream model's actual pricing.

---

## Control API (localhost only, no auth)

| Method | Path | Description |
|---|---|---|
| GET  | `/_control/health` | Health check (`{"ok","turn_count"}`) |
| GET  | `/_control/pending` | List of pending requests (`pending[]` + provider; oldest also under `request`; `deadline` / `timeout_in_seconds` when a pending timeout is configured) |
| GET  | `/_control/wait_for_pending?timeout=N` | Long-poll for the next pending (default 270s / max 600s; `{"timeout":true}` if none) |
| POST | `/_control/respond` | Inject a response (`{"content":[...], "pending_id"?, "stop_reason"?, "stop_sequence"?, "stop_details"?, "usage"?}` — or `{"text": "..."}` for one text block — into a pending request; `{"responses": [<body with its own target>, ...]}` injects several at once, all validated and resolved before any is applied (a target that vanished in between makes the answer a 409 with per-item `results`). `content` blocks: `text` / `tool_use` / `thinking` / `redacted_thinking` and the server-side tool blocks (§3); other types are dropped and reported (`dropped`) or refused under `strict_blocks`. `stop_reason` overrides the auto-derived value (e.g. `"max_tokens"` to exercise truncation branches; mapped to `finish_reason: "length"` on the OpenAI route); `"refusal"` produces `stop_details` (pass `stop_details` to set `category` / `explanation`; extra fields such as `recommended_model` pass through) and, on the OpenAI route, takes OpenAI's refusal shape (`message.refusal` / `delta.refusal`, `content: null`, `finish_reason: "stop"`); `"stop_sequence"` fills `stop_sequence`. `usage` overrides the approx token counts with real ones (any non-empty subset of `input_tokens` / `output_tokens` / `cache_creation_input_tokens` / `cache_read_input_tokens`, ints in `[0, 1e12]`, plus optional `cache_creation` / `output_tokens_details` / `server_tool_use` objects and `service_tier` / `inference_geo` / `speed` strings — used by relay mode) |
| POST | `/_control/respond_all` | The same `respond` body (no target) for every live pending, batch entries included — answers a parallel fan-out in one call (`{"ok", "count", "pending_ids"}`; 400 when nothing is pending) |
| POST | `/_control/auto` | Deprecated alias of `/_control/respond` with `{"text": "..."}` (same targeting); still works |
| GET  | `/_control/capabilities` | The compatibility matrix (see [Compatibility matrix](#compatibility-matrix)): surfaces with accepted / injectable / streamed / relayed flags, how each content block type travels on each route, and the operations refused on purpose |
| POST | `/_control/config` (also PUT) / GET | Harness configuration (§8): `pending_timeout_s`, `on_unmatched`, `default_response`, `unmatched_error`, `timeout_error`, `latency`, `rate_limit`, `seed`, `strict_blocks`, `default_headers`. Partial updates; `null` restores a default; kept across `clear` |
| GET / PUT / POST / DELETE | `/_control/rules` | Scenario rules (§8): list with counters (`unconsumed`, `all_consumed`), replace, append, remove all. `PUT` / `DELETE /_control/rules/{id}` act on one rule (a `PUT` that only appends steps under the same match keeps the rule's consumption counters) |
| GET | `/_control/clock` | The fake clock (`now`, `clock_offset_seconds`) |
| POST | `/_control/clock/advance` | `{"seconds": N}` — advance the fake clock (cache TTLs, rate-limit window, pending deadlines) |
| POST | `/_control/error` | Inject an HTTP error response (`{"status","type","message", "code"?, "param"?, "headers"?, "pending_id"?, "after_events"? / "after_blocks"?, "content"?}` — the last two make a streaming request fail mid-stream, see §4). `headers` (string → string/number) are attached to the error response verbatim — e.g. `{"retry-after": 3}` on a 429, or `anthropic-ratelimit-*` / `x-ratelimit-*` values — to exercise an app's backoff logic (framing headers such as `content-length` / `transfer-encoding`, control characters and non-Latin-1 values are rejected with 400). Every Anthropic-route error body (batches included) carries `request_id`, matching the `request-id` header; the OpenAI route maps Anthropic error `type`s to its own vocabulary (`api_error` → `server_error` by status, etc.) |
| GET  | `/_control/history` | (request, response, usage, cost, cache) history |
| GET  | `/_control/stats` | Cumulative summary of cost estimates, tokens, cache |
| GET  | `/_control/cache` | Pseudo prompt-cache index |
| POST | `/_control/clear` | Empty pending / history / cache / batches / Bedrock batch jobs / rules / rate-limit window / clock offset — the configuration too with body `{"config": true}` (in-flight requests are released with a retryable error: 529 `overloaded_error` on the Anthropic route, 503 on Bedrock / OpenAI; a Bedrock batch job still being created gets a non-retryable 400 `ConflictException`; the S3 store is left alone) |
| GET  | `/_control/batches` | Batch registry (status, request_counts, unresolved custom_ids) |
| POST | `/_control/batch/result` | Inject `canceled` / `expired` for one custom_id (`{"custom_id","type","batch_id"?}`) |
| POST | `/_control/batch/end` | Force a batch to `ended`; unresolved custom_ids become `expired` (default) or `canceled` |
| GET  | `/_control/bedrock_jobs` | Bedrock batch-inference job registry (status, record counts, unresolved `recordId`s) |

On `respond` / `auto` / `error`, batch entries can be addressed with `custom_id` (+ optional `batch_id`) instead of `pending_id`, and the latency keys `delay_ms` / `ttfb_ms` / `chunk_delay_ms` / `jitter_ms` (§8) shape that one answer's timing.

The request bodies of the `/_control/*` endpoints that take one are typed (`puppetllm/control_models.py`) and published in `/openapi.json` under `components.schemas` (`/docs` renders them), so a client can be generated from the schema. `{"text": "..."}` is accepted by `/_control/respond` (and rule steps) as shorthand for one text block, also next to an empty `content`; unset optional fields sent as `null` are treated as absent, except that `/_control/config` takes `null` as "restore the default" (serialize it with unset fields omitted).

Behavior changes relative to earlier versions (apps or harnesses asserting the old values need updating): a request cleared mid-flight now gets `529 overloaded_error` on the Anthropic / Bedrock-Messages routes (was `503 api_error`); OpenAI-route error `type`s follow OpenAI's vocabulary (`server_error`, `service_unavailable_error`, … — was `api_error` / `service_unavailable`); a canonical `refusal` maps to OpenAI's `message.refusal` + `finish_reason: "stop"` (was `finish_reason: "content_filter"` — pass `"content_filter"` as the `stop_reason` to get the filter shape); Bedrock pendings / responses carry the normalized Anthropic model name; `thinking` blocks are kept instead of dropped; the OpenAI usage object bills all `n` choices (in the response, history and stats alike; such pendings carry a `choices` field in the snapshot) and echoes an explicit `service_tier`; costs apply the official 1.1x multiplier when the request carries `inference_geo: "us"`; `/_control/auto` requires `text` to be a string (other keys are still ignored) and, like every request route, malformed `messages` / `system` / `tools` containers are refused with 400 instead of failing later; `/_control/respond` / `auto` / `error` answer `{"ok": true, "dropped": [...]}` instead of a bare `{"ok": true}` when an injected block type was dropped; server-side tool blocks are kept instead of dropped; an unimplemented path under `/v1`, `/anthropic`, `/model`, `/guardrail`, `/async-invoke` or `/chat` answers 404 in the provider's envelope instead of an S3 error, so `chat`, `models`, `guardrail` and `async-invoke` are no longer usable as S3 bucket names; a wrong method on a real path answers 405 whose `Allow` lists only the methods actually served (`HEAD` is not).

### Parallel requests (multi-pending)

The server can hold multiple concurrent requests. Each pending has a unique `pending_id`; inject into each individually by specifying `pending_id` on `/_control/respond` (also `auto` / `error`).

- Omitting `pending_id` is allowed only when there is **exactly one** pending. Zero → `400`; multiple → `400` (the response includes `pending_ids` so you can pick one).
- Injecting into a pending that no longer exists (already resolved, or wiped by `clear`) returns `400` (`no pending request`); only a near-simultaneous double-injection race returns `409` (`already resolved`).

How to build injection payloads (especially avoiding escape accidents with non-ASCII + nested JSON) is covered in detail in [`responder/CLAUDE.md`](responder/CLAUDE.md) / [`responder/AGENTS.md`](responder/AGENTS.md).

---

## Compatibility matrix

What each route accepts, whether a responder / rule can answer it (injectable), whether the
streaming form is served, and whether the relay forwards it. `GET /_control/capabilities`
returns the same data (plus how every content block type travels on each route).

| Provider | Operation | Path | Accepted | Injectable | Streamed | Relayed |
|---|---|---|---|---|---|---|
| Anthropic | Messages | `POST /v1/messages` | yes | yes | yes (SSE) | yes |
| Anthropic | count_tokens | `POST /v1/messages/count_tokens` | yes | answered directly | – | – |
| Anthropic | Message Batches | `/v1/messages/batches*` | yes | yes (per custom_id) | – | yes |
| Anthropic | Models | `GET /v1/models[/{id}]` | yes | catalogue | – | – |
| Bedrock | InvokeModel | `POST /model/{id}/invoke` | yes | yes | – | yes |
| Bedrock | InvokeModelWithResponseStream | `POST /model/{id}/invoke-with-response-stream` | yes | yes | yes (event stream) | yes |
| Bedrock | Converse / ConverseStream | `POST /model/{id}/converse[-stream]` | yes | yes | yes | yes |
| Bedrock | CountTokens | `POST /model/{id}/count-tokens` | yes | answered directly | – | – |
| Bedrock | Messages (Mantle) | `POST /anthropic/v1/messages` | yes | yes | yes | yes |
| Bedrock | Batch inference | `/model-invocation-job*` | yes | yes (per record) | – | yes |
| S3 | Buckets / objects (subset) | `/{bucket}[/{key}]` | yes | – | – | – |
| OpenAI | Chat Completions | `POST /v1/chat/completions` (alias `/chat/completions`) | yes | yes | yes (SSE) | yes |
| OpenAI | Models | `GET /v1/models[/{id}]` (alias `/models`) | yes | catalogue | – | – |

Refused on purpose, in the caller's own error envelope (a 404 naming the method, the path and, for these, the operation): Anthropic Text Completions, Files, Skills, Managed Agents, Admin; OpenAI Responses, Embeddings, Completions, Assistants / vector stores / fine-tuning / audio / images / Realtime; Bedrock ApplyGuardrail, async invoke, bidirectional streams and every other `/model/{id}/*` operation. A wrong method on a real path answers 405 with `Allow`. Content blocks: `text` / `tool_use` travel on every route; `thinking` / `redacted_thinking` on the Anthropic, Bedrock and Converse routes (dropped by OpenAI); the server-side tool blocks on the Anthropic, Bedrock InvokeModel and Mantle routes (dropped, and not billed, by Converse and OpenAI).

---

## Security

- `/_control/*` has **no auth**. Anyone can read history and inject responses or errors. **Do not expose it to the public internet.**
- The default listen address is `127.0.0.1` (localhost only). Access from another host only **within a trusted network** such as LAN / VPN / Tailscale.
- If you expose it with `--host 0.0.0.0` (Docker listens on `0.0.0.0` by default, but compose restricts publishing to `127.0.0.1:8765`), always check your firewall / network policy.
- **Running the image directly with `docker run`**: the container listens on `0.0.0.0` (required for port mapping), so bind the published port to localhost — `docker run -p 127.0.0.1:8765:8765 puppetllm` — **not** `-p 8765:8765`, which would expose the unauthenticated control plane on every host interface. The provided `docker compose` already does this for you.
- The bundled **S3 emulation** is part of that control plane: any client that can reach the
  port can create buckets and read, write and delete objects without credentials. It is
  confined to `PUPPETLLM_S3_ROOT` (a per-process temp directory unless you set it), and
  nothing outside that directory is reachable — but point `PUPPETLLM_S3_ROOT` at a scratch
  directory, not at anything you care about.
- This is strictly a local debugging tool. It is not meant to sit in front of production.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PUPPETLLM_CACHE_TTL` | `300` | Pseudo-cache TTL for 5-minute breakpoints (seconds) |
| `PUPPETLLM_CACHE_TTL_1H` | 12 × `PUPPETLLM_CACHE_TTL` | TTL for `ttl: "1h"` breakpoints (seconds); defaults to 3600 and scales with the 5m TTL when unset |
| `PUPPETLLM_S3_ROOT` | a per-process temp dir | Where the S3 emulation stores objects (batch inference I/O). Set it to keep them across restarts, or to see them from the host under Docker (add the variable and a matching volume to your compose override; the shipped `docker-compose.yml` defines neither) |
| `PUPPETLLM_CACHE_HONOR_TTL` | `1` | `0` ignores the TTL (entries live forever) |
| `PUPPETLLM_CACHE_MIN_TOKENS` | (per-model) | Override the minimum cache threshold. `0` disables it (cache every prefix). Unset = the generation-aware table (Opus 5 / Fable 512, Opus 4.8 1024, Opus 4.7 2048, Opus 4.6 / 4.5 4096, Sonnet 1024, Haiku 4.5 4096, …) |
| `PUPPETLLM_PENDING_TIMEOUT` | (none) | Seconds before an unanswered pending gets the timeout error (§8) |
| `PUPPETLLM_DEFAULT_RESPONSE` | (none) | Text, or a JSON `/_control/respond` body, that answers unmatched requests; setting it selects `on_unmatched: default` |
| `PUPPETLLM_ON_UNMATCHED` | `pending` | `pending` / `default` / `error` |
| `PUPPETLLM_SEED` | (none) | Seed for latency jitter |
| `PUPPETLLM_CONFIG` | (none) | JSON file `{"config": {...}, "rules": [...]}` loaded at startup |
| `PUPPETLLM_URL` | (none) | Makes the `puppet` pytest fixture use a running server instead of starting one |

---

## Tests

```bash
# Docker (the test profile also starts the `proxy` service on 127.0.0.1:8765 via
# depends_on, so that port must be free)
docker compose --profile test run --rm proxy-test

# Or directly
pip install -r requirements.txt
python3 -m unittest puppetllm.tests.test_fake_server puppetllm.tests.test_proxy_extensions \
    puppetllm.tests.test_batches puppetllm.tests.test_conformance \
    puppetllm.tests.test_bedrock_extras puppetllm.tests.test_harness \
    puppetllm.tests.test_compat puppetllm.tests.test_boto3_interop -v
```

`puppetllm/tests/test_fake_server.py` is an executable specification of the expected behavior; `test_harness.py` covers §8 (rules, policies, timeouts, latency, rate limits, the clock, `count_tokens` / `models`, the CLI and `puppetllm.testing` against an in-process uvicorn).

`test_boto3_interop.py` is the only module that drives **real** boto3 / botocore (`>= 1.43`,
the first service model with every field it exercises) against a running uvicorn instance — every other test encodes and decodes the Bedrock event stream
with puppetllm's own codec, so a matching encoder/decoder mistake would pass unnoticed
there. It needs `boto3` (in `requirements.txt`); the compose test profile sets
`PUPPETLLM_REQUIRE_SDK_TESTS=1` so a run that cannot import it fails instead of quietly
reporting `OK` with every SDK test skipped.

---

## Layout

```
puppetllm/
├── puppetllm/              # package itself
│   ├── fake_server.py      # canonical core + Anthropic /v1/messages + /_control/* + count_tokens / models
│   ├── harness.py          # scenario rules, unmatched policy, latency, rate limit, fake clock
│   ├── control_models.py   # typed /_control/* request bodies (published in /openapi.json)
│   ├── capabilities.py     # compatibility matrix (/_control/capabilities) + the catch-all for unimplemented paths
│   ├── testing.py          # test-side client (Puppet, serve()) — pytest_plugin.py adds the `puppet` fixture
│   ├── batches.py          # Anthropic Message Batches route + batch control endpoints
│   ├── cache_sim.py        # pseudo prompt cache
│   ├── pricing.py          # approximate tokens + pricing
│   ├── relay.py            # relay responder (cross-provider bridge to a real API)
│   ├── openai_wire.py      # pure OpenAI ↔ canonical conversions shared by the adapter and the relay
│   ├── providers/          # Bedrock (invoke / converse / batch) + OpenAI adapters, S3, AWS event stream
│   │                       #   bedrock.py, converse.py, bedrock_batch.py, s3.py, openai.py, eventstream.py
│   └── tests/              # unit tests
├── responder/              # instruction docs for the responder (the agent that "plays the LLM")
│   ├── CLAUDE.md           #   for Claude Code
│   └── AGENTS.md           #   for Codex CLI and other agents following the AGENTS.md convention
├── LICENSE
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml          # `pip install .` → `puppetllm` command + pytest plugin
└── requirements.txt
```

---

## License

[MIT License](LICENSE) — Copyright (c) 2026 Aetheria Labs
