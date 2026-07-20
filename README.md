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
- `puppetllm/providers/bedrock.py` — Bedrock route `POST /model/{id}/invoke[-with-response-stream]` (AWS event stream framing lives in `providers/eventstream.py`).
- `puppetllm/providers/openai.py` — OpenAI route `POST /v1/chat/completions` (requests are normalized to the canonical Anthropic-style form; responses are converted back to `chat.completion` JSON / SSE chunks).
- `puppetllm/cache_sim.py` — pseudo prompt cache (multi-breakpoint + prefix match + per-model minimum threshold + 20-block lookback).
- `puppetllm/pricing.py` — approximate tokens + price table (Claude and GPT families).

Providers are auto-selected by URL path — no mode switch or configuration. Response content blocks / control API are common across providers (injection is always the same `/_control/respond`).

---

## Usage

Think of it as **three actors**:

```
  ┌──── app / SDK ─────┐         ┌──── puppetllm ────┐        ┌── responder ───┐
  │ messages.create()  │ ──────▶ │ POST /v1/messages │ ─────▶ │ inject reply   │
  │ blocks for reply   │ ◀────── │ held as pending   │ ◀───── │ /_control/...  │
  └────────────────────┘  reply  └───────────────────┘        └────────────────┘
          ① app                      ② fake server               ③ human / AI
```

② **holds (pending)** the request ① sends; when ③ pushes a response via `/_control/*`, ①'s `create()` returns with that response. The real API is never called.

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
- Requests are translated from the canonical form (system / messages / tools / tool_choice / stop / temperature …); responses come back as canonical blocks with the **real `stop_reason` and real token usage** (via the `stop_reason` / `usage` fields of `/_control/respond`), so `/_control/stats` aggregates real numbers (history entries get `"usage_overridden": true`). When the upstream omits usage, puppetllm's approximation is kept instead.
- For OpenAI reasoning / official gpt-5 endpoints that reject `max_tokens`, pass `--max-tokens-param max_completion_tokens`.
- Upstream API errors are relayed with their status/type/message (and `code`/`param`), so your app's SDK raises the same exception class it would against the real provider.
- The relay is *just another responder*. **By default it claims every pending it sees**, so it does not share a live queue with a human / AI-agent responder — the swap is sequential: stop the relay and take over by hand. To run them **concurrently**, use `--only "<glob>,…"` so the relay claims only the matching inbound models and leaves the rest for you (or another agent).
- `--max-concurrency N` caps simultaneous in-flight upstream calls (default: unlimited), so a burst of pendings doesn't fan out and trip upstream rate limits.

Caveats: the upstream call is non-streaming, so a streaming app sees correct SSE but with first-token latency equal to the full upstream response time; multimodal (image) blocks are not translated; costs are **real** in this mode; `/_control/stats` prices tokens by the *inbound* model id, which may differ from the upstream model's actual pricing.

---

## Control API (localhost only, no auth)

| Method | Path | Description |
|---|---|---|
| GET  | `/_control/health` | Health check (`{"ok","turn_count"}`) |
| GET  | `/_control/pending` | List of pending requests (`pending[]` + provider; oldest also under `request`) |
| GET  | `/_control/wait_for_pending?timeout=N` | Long-poll for the next pending (default 270s / max 600s; `{"timeout":true}` if none) |
| POST | `/_control/respond` | Inject a response (`{"content":[...], "pending_id"?, "stop_reason"?, "usage"?}`) into a pending request. `stop_reason` overrides the auto-derived value (e.g. `"max_tokens"` to exercise truncation branches; mapped to `finish_reason: "length"` on the OpenAI route). `usage` overrides the approx token counts with real ones (any non-empty subset of `input_tokens` / `output_tokens` / `cache_creation_input_tokens` / `cache_read_input_tokens`, ints in `[0, 1e12]` — used by relay mode) |
| POST | `/_control/auto` | Simple auto-response (`{"text":"...", "pending_id"?}`, text only) |
| POST | `/_control/error` | Inject an HTTP error response (`{"status","type","message", "code"?, "param"?, "pending_id"?}`) |
| GET  | `/_control/history` | (request, response, usage, cost, cache) history |
| GET  | `/_control/stats` | Cumulative summary of cost estimates, tokens, cache |
| GET  | `/_control/cache` | Pseudo prompt-cache index |
| POST | `/_control/clear` | Empty pending / history / cache (in-flight requests released with 503) |

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
| `PUPPETLLM_CACHE_TTL` | `300` | Pseudo-cache TTL (seconds) |
| `PUPPETLLM_CACHE_HONOR_TTL` | `1` | `0` ignores the TTL (entries live forever) |
| `PUPPETLLM_CACHE_MIN_TOKENS` | (per-model) | Override the minimum cache threshold. `0` disables it (cache every prefix). Unset = Opus 4096 / Sonnet 1024 / Haiku 4096 |

---

## Tests

```bash
# Docker
docker compose --profile test run --rm proxy-test

# Or directly
pip install -r requirements.txt
python3 -m unittest puppetllm.tests.test_fake_server puppetllm.tests.test_proxy_extensions -v
```

`puppetllm/tests/test_fake_server.py` is an executable specification of the expected behavior.

---

## Layout

```
puppetllm/
├── puppetllm/              # package itself
│   ├── fake_server.py      # canonical core + Anthropic /v1/messages + /_control/*
│   ├── cache_sim.py        # pseudo prompt cache
│   ├── pricing.py          # approximate tokens + pricing
│   ├── relay.py            # relay responder (cross-provider bridge to a real API)
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
