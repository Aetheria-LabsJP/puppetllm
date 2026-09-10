**English** | [日本語](README.ja.md)

# puppetllm — LLM API debug proxy (fake Anthropic / Bedrock / OpenAI server)

A **fake server** compatible with the Anthropic Messages API / Bedrock / OpenAI Chat Completions. Just point `ANTHROPIC_BASE_URL` (or the `AnthropicBedrock` / `OpenAI` `base_url`) at this server and it intercepts LLM calls **without changing a single line of your app / SDK code**, letting a human or another agent supply the responses (human-in-the-loop / AI-in-the-loop).

Use cases:

- **Zero-cost debugging**: reproduce and inspect agent / orchestration behavior without hitting the real API.
- **Deterministic testing**: inject arbitrary responses (text / tool_use, errors) to reproduce branches.
- **Cross-provider bridge ([relay mode](#relay-mode-cross-provider-bridge))**: run an app written for one SDK against a *different* real provider (e.g. an Anthropic-SDK agent on Grok / GPT, or an OpenAI-SDK app on Claude) — without changing a line of app code.
- **Cost estimates**: aggregate approximate tokens / pricing per request (`/_control/stats`).
- **Pseudo prompt-cache observation**: verify by hash whether your app structures requests so `cache_control` actually takes effect (`/_control/cache`).

> All figures are based on an approximate tokenizer, so they **do not match real billing**. Use them for trend analysis and structural verification.

---

## Architecture

A provider-agnostic canonical core + adapters:

- `puppetllm/fake_server.py` — canonical core (normalized snapshot management + `/_control/*` + cost/cache computation). The Anthropic route `POST /v1/messages` is built in.
- `puppetllm/providers/bedrock.py` — Bedrock route `POST /model/{id}/invoke[-with-response-stream]` (model-id normalization, `anthropic_version` validation, AWS-style errors / headers; AWS event stream framing lives in `providers/eventstream.py`).
- `puppetllm/providers/openai.py` — OpenAI route `POST /v1/chat/completions` (requests are normalized to the canonical Anthropic-style form; responses are converted back to `chat.completion` JSON / SSE chunks).
- `puppetllm/batches.py` — Anthropic Message Batches route `/v1/messages/batches*` (each custom_id is held as an ordinary pending; batch lifecycle is injectable via `/_control/batch/*`).
- `puppetllm/cache_sim.py` — pseudo prompt cache (multi-breakpoint + top-level automatic `cache_control` + prefix match + generation-aware minimum threshold + 5m / 1h TTLs + 20-block lookback + effort / thinking / tool_choice invalidation).
- `puppetllm/pricing.py` — approximate tokens + price table (Claude generations incl. Fable / Mythos / Sonnet 5 / Opus 4.1, and GPT / o-series families, per the official pricing pages).

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

# B) Directly (Python 3.12+) — runs in foreground with a startup banner
pip install -r requirements.txt
python3 -m puppetllm --host 127.0.0.1 --port 8765
#   [puppetllm] starting on http://127.0.0.1:8765
#   [puppetllm] Anthropic: set ANTHROPIC_BASE_URL=http://127.0.0.1:8765
#   [puppetllm] Bedrock:   point AnthropicBedrock base_url to http://127.0.0.1:8765
#   [puppetllm] OpenAI:    set OPENAI_BASE_URL=http://127.0.0.1:8765/v1  (note the /v1)

# C) uvicorn directly (when you want options like --reload)
python3 -m uvicorn puppetllm.fake_server:app --host 127.0.0.1 --port 8765
```

`--host` defaults to `127.0.0.1` (localhost only). Use `0.0.0.0` only when accessing over LAN/VPN (see [Security](#security)).

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
- **Response headers**: non-stream responses carry `X-Amzn-Bedrock-Input-Token-Count` (non-cached input only, like Converse's `inputTokens`) / `X-Amzn-Bedrock-Output-Token-Count` / `X-Amzn-Bedrock-Cache-Read-Input-Token-Count` / `X-Amzn-Bedrock-Cache-Write-Input-Token-Count` / `X-Amzn-Bedrock-Invocation-Latency` (+ `x-amzn-requestid`, `X-Amzn-Bedrock-Service-Tier`); streams carry `amazon-bedrock-invocationMetrics` (incl. `cacheReadInputTokenCount` / `cacheWriteInputTokenCount`) on the final chunk and `X-Amzn-Bedrock-Content-Type`, as real Bedrock does.
- **Messages-API alias for `AnthropicBedrockMantle`**: `POST /anthropic/v1/messages` (the path served by the `bedrock-runtime` / `bedrock-mantle` hosts) is the plain Anthropic handler with Bedrock model-id normalization, and the plain `/v1/messages` route hands any request whose `model` is a Bedrock id (`[region.]anthropic.…` / ARN) to the same handler — so `AnthropicBedrockMantle(base_url="http://localhost:8765")` works whether it posts to the root or to `/anthropic` (SSE streaming, Anthropic error envelope, `anthropic-version` header, no `anthropic_version` body field). Batches / count_tokens are not served on this alias, matching Bedrock.
- **Receipt log**: every Bedrock request (and every rejection) is logged to stderr as one `[bedrock] invoke model=<raw> -> <canonical> pending=<id> …` line, so Bedrock traffic is distinguishable from the other routes at a glance.

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

Besides `text` / `tool_use`, the injected content may contain `thinking` (`{"type":"thinking","thinking":"…","signature"?}` — an opaque signature is generated when omitted) and `redacted_thinking` (`{"data":"…"}`) blocks, which are kept in the response, streamed as `thinking_delta` / `signature_delta`, counted in `usage.output_tokens_details.thinking_tokens`, and expected back verbatim on the next turn — exactly the shape current models return by default. `stop_reason` accepts the documented vocabulary (`end_turn` / `max_tokens` / `stop_sequence` / `tool_use` / `pause_turn` / `refusal` / `model_context_window_exceeded`); `"refusal"` yields a `stop_details` object (`{"type":"refusal","category":null,"explanation":null}` unless you pass `stop_details`), `"stop_sequence"` fills `stop_sequence` from the request's `stop_sequences` unless you pass one.

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

On the Bedrock route the body becomes `{"message": "...", "__type": "<AwsException>"}` with an `x-amzn-ErrorType` header. If `type` is already an AWS exception name (ends in `Exception`) it is used as-is; otherwise it is derived from `status`: 400 → `ValidationException`, 401 → `UnrecognizedClientException`, 403 → `AccessDeniedException`, 404 → `ResourceNotFoundException`, 408/504 → `ModelTimeoutException`, 413 → `RequestEntityTooLargeException`, 424 → `ModelErrorException`, 429 → `ThrottlingException`, 500 → `InternalServerException`, 503 → `ServiceUnavailableException`, 529 → `overloaded_error` (Bedrock passes Anthropic's 529 through) — other 4xx → `ValidationException`, other 5xx → `InternalServerException`. Pass an AWS name explicitly for the rest (`ServiceQuotaExceededException` 400, `ModelNotReadyException` 429, `ModelStreamErrorException` 424). So the same `{"status": 429, "type": "rate_limit_error"}` injection yields a `ThrottlingException` for a Bedrock client and a `rate_limit_error` for an Anthropic client.

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
| GET  | `/_control/pending` | List of pending requests (`pending[]` + provider; oldest also under `request`) |
| GET  | `/_control/wait_for_pending?timeout=N` | Long-poll for the next pending (default 270s / max 600s; `{"timeout":true}` if none) |
| POST | `/_control/respond` | Inject a response (`{"content":[...], "pending_id"?, "stop_reason"?, "stop_sequence"?, "stop_details"?, "usage"?}`) into a pending request. `content` blocks: `text` / `tool_use` / `thinking` / `redacted_thinking`. `stop_reason` overrides the auto-derived value (e.g. `"max_tokens"` to exercise truncation branches; mapped to `finish_reason: "length"` on the OpenAI route); `"refusal"` produces `stop_details` (pass `stop_details` to set `category` / `explanation`; extra fields such as `recommended_model` pass through) and, on the OpenAI route, takes OpenAI's refusal shape (`message.refusal` / `delta.refusal`, `content: null`, `finish_reason: "stop"`); `"stop_sequence"` fills `stop_sequence`. `usage` overrides the approx token counts with real ones (any non-empty subset of `input_tokens` / `output_tokens` / `cache_creation_input_tokens` / `cache_read_input_tokens`, ints in `[0, 1e12]`, plus optional `cache_creation` / `output_tokens_details` / `server_tool_use` objects and `service_tier` / `inference_geo` / `speed` strings — used by relay mode) |
| POST | `/_control/auto` | Simple auto-response (`{"text":"...", "pending_id"?}`, text only) |
| POST | `/_control/error` | Inject an HTTP error response (`{"status","type","message", "code"?, "param"?, "headers"?, "pending_id"?}`). `headers` (string → string/number) are attached to the error response verbatim — e.g. `{"retry-after": 3}` on a 429, or `anthropic-ratelimit-*` / `x-ratelimit-*` values — to exercise an app's backoff logic (framing headers such as `content-length` / `transfer-encoding`, control characters and non-Latin-1 values are rejected with 400). Every Anthropic-route error body (batches included) carries `request_id`, matching the `request-id` header; the OpenAI route maps Anthropic error `type`s to its own vocabulary (`api_error` → `server_error` by status, etc.) |
| GET  | `/_control/history` | (request, response, usage, cost, cache) history |
| GET  | `/_control/stats` | Cumulative summary of cost estimates, tokens, cache |
| GET  | `/_control/cache` | Pseudo prompt-cache index |
| POST | `/_control/clear` | Empty pending / history / cache / batches (in-flight requests are released with a retryable error: 529 `overloaded_error` on the Anthropic route, 503 on Bedrock / OpenAI) |
| GET  | `/_control/batches` | Batch registry (status, request_counts, unresolved custom_ids) |
| POST | `/_control/batch/result` | Inject `canceled` / `expired` for one custom_id (`{"custom_id","type","batch_id"?}`) |
| POST | `/_control/batch/end` | Force a batch to `ended`; unresolved custom_ids become `expired` (default) or `canceled` |

On `respond` / `auto` / `error`, batch entries can be addressed with `custom_id` (+ optional `batch_id`) instead of `pending_id`.

Behavior changes relative to earlier versions (apps or harnesses asserting the old values need updating): a request cleared mid-flight now gets `529 overloaded_error` on the Anthropic / Bedrock-Messages routes (was `503 api_error`); OpenAI-route error `type`s follow OpenAI's vocabulary (`server_error`, `service_unavailable_error`, … — was `api_error` / `service_unavailable`); a canonical `refusal` maps to OpenAI's `message.refusal` + `finish_reason: "stop"` (was `finish_reason: "content_filter"` — pass `"content_filter"` as the `stop_reason` to get the filter shape); Bedrock pendings / responses carry the normalized Anthropic model name; `thinking` blocks are kept instead of dropped; the OpenAI usage object bills all `n` choices (in the response, history and stats alike; such pendings carry a `choices` field in the snapshot) and echoes an explicit `service_tier`; costs apply the official 1.1x multiplier when the request carries `inference_geo: "us"`.

### Parallel requests (multi-pending)

The server can hold multiple concurrent requests. Each pending has a unique `pending_id`; inject into each individually by specifying `pending_id` on `/_control/respond` (also `auto` / `error`).

- Omitting `pending_id` is allowed only when there is **exactly one** pending. Zero → `400`; multiple → `400` (the response includes `pending_ids` so you can pick one).
- Injecting into a pending that no longer exists (already resolved, or wiped by `clear`) returns `400` (`no pending request`); only a near-simultaneous double-injection race returns `409` (`already resolved`).

How to build injection payloads (especially avoiding escape accidents with non-ASCII + nested JSON) is covered in detail in [`responder/CLAUDE.md`](responder/CLAUDE.md) / [`responder/AGENTS.md`](responder/AGENTS.md).

---

## Security

- `/_control/*` has **no auth**. Anyone can read history and inject responses or errors. **Do not expose it to the public internet.**
- The default listen address is `127.0.0.1` (localhost only). Access from another host only **within a trusted network** such as LAN / VPN / Tailscale.
- If you expose it with `--host 0.0.0.0` (Docker listens on `0.0.0.0` by default, but compose restricts publishing to `127.0.0.1:8765`), always check your firewall / network policy.
- **Running the image directly with `docker run`**: the container listens on `0.0.0.0` (required for port mapping), so bind the published port to localhost — `docker run -p 127.0.0.1:8765:8765 puppetllm` — **not** `-p 8765:8765`, which would expose the unauthenticated control plane on every host interface. The provided `docker compose` already does this for you.
- This is strictly a local debugging tool. It is not meant to sit in front of production.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PUPPETLLM_CACHE_TTL` | `300` | Pseudo-cache TTL for 5-minute breakpoints (seconds) |
| `PUPPETLLM_CACHE_TTL_1H` | 12 × `PUPPETLLM_CACHE_TTL` | TTL for `ttl: "1h"` breakpoints (seconds); defaults to 3600 and scales with the 5m TTL when unset |
| `PUPPETLLM_CACHE_HONOR_TTL` | `1` | `0` ignores the TTL (entries live forever) |
| `PUPPETLLM_CACHE_MIN_TOKENS` | (per-model) | Override the minimum cache threshold. `0` disables it (cache every prefix). Unset = the generation-aware table (Opus 5 / Fable 512, Opus 4.8 1024, Opus 4.7 2048, Opus 4.6 / 4.5 4096, Sonnet 1024, Haiku 4.5 4096, …) |

---

## Tests

```bash
# Docker
docker compose --profile test run --rm proxy-test

# Or directly
pip install -r requirements.txt
python3 -m unittest puppetllm.tests.test_fake_server puppetllm.tests.test_proxy_extensions puppetllm.tests.test_batches puppetllm.tests.test_conformance -v
```

`puppetllm/tests/test_fake_server.py` is an executable specification of the expected behavior.

---

## Layout

```
puppetllm/
├── puppetllm/              # package itself
│   ├── fake_server.py      # canonical core + Anthropic /v1/messages + /_control/*
│   ├── batches.py          # Anthropic Message Batches route + batch control endpoints
│   ├── cache_sim.py        # pseudo prompt cache
│   ├── pricing.py          # approximate tokens + pricing
│   ├── relay.py            # relay responder (cross-provider bridge to a real API)
│   ├── openai_wire.py      # pure OpenAI ↔ canonical conversions shared by the adapter and the relay
│   ├── providers/          # Bedrock / OpenAI adapters + AWS event stream
│   └── tests/              # unit tests
├── responder/              # instruction docs for the responder (the agent that "plays the LLM")
│   ├── CLAUDE.md           #   for Claude Code
│   └── AGENTS.md           #   for Codex CLI and other agents following the AGENTS.md convention
├── LICENSE
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## License

[MIT License](LICENSE) — Copyright (c) 2026 Aetheria Labs
