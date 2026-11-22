**English** | [日本語](ja/control-api.md) · [← README](../README.md)

# Control API reference

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET  | `/_control/health` | Health check (`{"ok","turn_count"}`) |
| GET  | `/_control/pending` | List of pending requests (`pending[]` + provider; oldest also under `request`; `deadline` / `timeout_in_seconds` when a pending timeout is configured). `?summary=1` replaces each `request` with a short view (turn, model, `message_count`, `tools`, `last_user_text`, `has_tool_result`, ids), `?fields=model,turn` keeps only those snapshot keys |
| GET  | `/_control/wait_for_pending?timeout=N` | Long-poll for the next pending (default 270s / max 600s; `{"timeout":true}` if none) |
| POST | `/_control/respond` | Inject a response (`{"content":[...], "pending_id"?, "stop_reason"?, "stop_sequence"?, "stop_details"?, "usage"?, "provenance"?}` — or `{"text": "..."}` for one text block — into a pending request; `{"responses": [<body with its own target>, ...]}` injects several at once, all validated and resolved before any is applied (a target that vanished in between makes the answer a 409 with per-item `results`). `content` blocks: `text` / `tool_use` / `thinking` / `redacted_thinking` and the server-side tool blocks ([Supply the response](quickstart.md#3-supply-the-response-responder)); other types are dropped and reported (`dropped`) or refused under `strict_blocks`. `stop_reason` overrides the auto-derived value (e.g. `"max_tokens"` to exercise truncation branches; mapped to `finish_reason: "length"` on the OpenAI route); `"refusal"` produces `stop_details` (pass `stop_details` to set `category` / `explanation`; extra fields such as `recommended_model` pass through) and, on the OpenAI route, takes OpenAI's refusal shape (`message.refusal` / `delta.refusal`, `content: null`, `finish_reason: "stop"`); `"stop_sequence"` fills `stop_sequence`. `usage` overrides the approx token counts with real ones (any non-empty subset of `input_tokens` / `output_tokens` / `cache_creation_input_tokens` / `cache_read_input_tokens`, ints in `[0, 1e12]`, plus optional `cache_creation` / `output_tokens_details` / `server_tool_use` objects and `service_tier` / `inference_geo` / `speed` strings — used by relay mode). `provenance` (`{upstream_provider, upstream_model, usage_source, pricing_model, translation.dropped_fields}`, [Observe cost / tokens / cache](#observe-cost--tokens--cache)) is recorded with the entry; its `pricing_model` picks the price table. OpenAI extras: `choices` (instead of `content` / `text`: one `text` / `content` / `stop_reason` / `logprobs` body per choice of an `n > 1` request, cycled or cut to `n`; usage sums the answers; history keeps them under `choices`) and `logprobs` (`[{"token", "logprob", "top_logprobs"?}]` for the text answer, replacing the deterministic values the route makes up when the request asked for `logprobs`). Bedrock extras: `provider_response` (see [Bedrock](bedrock.md#guardrails-traces-and-prompt-management-fixtures)) |
| POST | `/_control/respond_all` | The same `respond` body (no target) for every live pending, batch entries included — answers a parallel fan-out in one call (`{"ok", "count", "pending_ids"}`; 400 when nothing is pending) |
| POST | `/_control/auto` | Deprecated alias of `/_control/respond` with `{"text": "..."}` (same targeting); still works |
| GET  | `/_control/capabilities` | The compatibility matrix (see [Compatibility matrix](design.md#compatibility-matrix)): surfaces with accepted / injectable / streamed / relayed flags, how each content block type travels on each route, and the operations refused on purpose |
| POST | `/_control/config` (also PUT) / GET | Harness configuration ([Testing guide](testing.md)): `pending_timeout_s`, `on_unmatched`, `default_response`, `unmatched_error`, `timeout_error`, `latency`, `rate_limit`, `seed`, `strict_blocks`, `default_headers`. Partial updates; `null` restores a default; kept across `clear` |
| GET / PUT / POST / DELETE | `/_control/prompts` | Prompt-management fixtures (`{"prompts": {"arn:aws:bedrock:...:prompt/ID": {"system"?, "messages"?, "tools"?}}}`): PUT replaces, POST adds, DELETE clears; `--config` files take a `prompts` key (see [Bedrock](bedrock.md#guardrails-traces-and-prompt-management-fixtures)) |
| GET / PUT / POST / DELETE | `/_control/rules` | Scenario rules ([Testing guide](testing.md)): list with counters (`unconsumed`, `all_consumed`), replace, append, remove all. `PUT` / `DELETE /_control/rules/{id}` act on one rule (a `PUT` that only appends steps under the same match keeps the rule's consumption counters) |
| GET | `/_control/clock` | The fake clock (`now`, `clock_offset_seconds`) |
| POST | `/_control/clock/advance` | `{"seconds": N}` — advance the fake clock (cache TTLs, rate-limit window, pending deadlines) |
| POST | `/_control/error` | Inject an HTTP error response (`{"status","type","message", "code"?, "param"?, "headers"?, "pending_id"?, "after_events"? / "after_blocks"?, "content"?, "provenance"?}` — the last two make a streaming request fail mid-stream, see [Inject error responses](quickstart.md#4-inject-error-responses-to-test-handling)). `headers` (string → string/number) are attached to the error response verbatim — e.g. `{"retry-after": 3}` on a 429, or `anthropic-ratelimit-*` / `x-ratelimit-*` values — to exercise an app's backoff logic (framing headers such as `content-length` / `transfer-encoding`, control characters and non-Latin-1 values are rejected with 400). Every Anthropic-route error body (batches included) carries `request_id`, matching the `request-id` header; the OpenAI route maps Anthropic error `type`s to its own vocabulary (`api_error` → `server_error` by status, etc.). `provenance` is recorded as for `/_control/respond` |
| GET  | `/_control/history` | (request, response, usage, cost, cache) history, in completion order. Query: `cursor` (entries recorded after that `seq`), `since_turn` (entries with a greater turn), `provider`, `model` (glob), `limit` (1–1000; the answer carries `has_more` and `next_cursor` — the `seq` to continue from, or your cursor back when nothing new was recorded — safe for paging and tailing while requests complete out of turn order), `format=jsonl` (one entry per line; `has_more` / `next_cursor` move to the `x-puppetllm-has-more` / `x-puppetllm-next-cursor` response headers). Entries carry `seq`, `request_id`, `path`, `latency_ms`, `usage_source`, `pricing_model`, `provenance` ([Observe cost / tokens / cache](#observe-cost--tokens--cache)) |
| GET  | `/_control/history/{turn}` | One entry (404 when unknown, 400 for a non-integer) |
| GET  | `/_control/metrics` | Prometheus text: cumulative counters (kept across `clear`) and live gauges ([Observe cost / tokens / cache](#observe-cost--tokens--cache)) |
| GET  | `/_control/stats` | Cumulative summary of cost estimates, tokens, cache |
| GET  | `/_control/cache` | Pseudo prompt-cache index |
| POST | `/_control/clear` | Empty pending / history / cache / batches / Bedrock batch jobs / rules / prompt fixtures / rate-limit window / clock offset — the configuration too with body `{"config": true}` (in-flight requests are released with a retryable error: 529 `overloaded_error` on the Anthropic route, 503 on Bedrock / OpenAI; a Bedrock batch job still being created gets a non-retryable 400 `ConflictException`; the S3 store is left alone) |
| GET  | `/_control/batches` | Batch registry (status, request_counts, unresolved custom_ids) |
| POST | `/_control/batch/result` | Inject `canceled` / `expired` for one custom_id (`{"custom_id","type","batch_id"?}`) |
| POST | `/_control/batch/end` | Force a batch to `ended`; unresolved custom_ids become `expired` (default) or `canceled` |
| GET  | `/_control/bedrock_jobs` | Bedrock batch-inference job registry (status, record counts, unresolved `recordId`s) |

On `respond` / `auto` / `error`, batch entries can be addressed with `custom_id` (+ optional `batch_id`) instead of `pending_id`, and the latency keys `delay_ms` / `ttfb_ms` / `chunk_delay_ms` / `jitter_ms` ([Testing guide](testing.md)) shape that one answer's timing.

The request bodies of the `/_control/*` endpoints that take one are typed (`puppetllm/control_models.py`) and published in `/openapi.json` under `components.schemas` (`/docs` renders them), so a client can be generated from the schema. `{"text": "..."}` is accepted by `/_control/respond` (and rule steps) as shorthand for one text block, also next to an empty `content`; unset optional fields sent as `null` are treated as absent, except that `/_control/config` takes `null` as "restore the default" (serialize it with unset fields omitted).

Behavior changes relative to earlier versions (apps or harnesses asserting the old values need updating): a `stop_reason` of `guardrail_intervened` ends Anthropic-shaped messages with `end_turn` and OpenAI ones with `finish_reason: "content_filter"` (it was passed through / `stop`); a Chat Completions request with `logprobs: true` gets log probabilities (was `null`), and a non-boolean `logprobs` or a `top_logprobs` without it or outside 0–20 is a 400; the fast-mode and US-inference cost multipliers apply to Claude ids only (never to an OpenAI family, a custom price table or an upstream model a relay prices by; a made-up id on an Anthropic-shaped route still gets them); a request cleared mid-flight now gets `529 overloaded_error` on the Anthropic / Bedrock-Messages routes (was `503 api_error`); OpenAI-route error `type`s follow OpenAI's vocabulary (`server_error`, `service_unavailable_error`, … — was `api_error` / `service_unavailable`); a canonical `refusal` maps to OpenAI's `message.refusal` + `finish_reason: "stop"` (was `finish_reason: "content_filter"` — pass `"content_filter"` as the `stop_reason` to get the filter shape); Bedrock pendings / responses carry the normalized Anthropic model name; `thinking` blocks are kept instead of dropped; the OpenAI usage object bills all `n` choices (in the response, history and stats alike; such pendings carry a `choices` field in the snapshot) and echoes an explicit `service_tier`; costs apply the official 1.1x multiplier when the request carries `inference_geo: "us"`; `/_control/auto` requires `text` to be a string (other keys are still ignored) and, like every request route, malformed `messages` / `system` / `tools` containers are refused with 400 instead of failing later; `/_control/respond` / `auto` / `error` answer `{"ok": true, "dropped": [...]}` instead of a bare `{"ok": true}` when an injected block type was dropped; server-side tool blocks are kept instead of dropped; an unimplemented path under `/v1`, `/anthropic`, `/model`, `/guardrail`, `/async-invoke` or `/chat` answers 404 in the provider's envelope instead of an S3 error, so `chat`, `models`, `guardrail` and `async-invoke` are no longer usable as S3 bucket names; a wrong method on a real path answers 405 whose `Allow` lists only the methods actually served (`HEAD` is not); costs carry `long_context` and a prompt past a model's long-context threshold is priced on the higher card (it was one flat card per model); prices follow the current tables, which moves ids that used to land on a generic row (Fable / Mythos 5.2 $8/$40, Opus 5.5 $4/$20, Haiku 5.5 $0.10/$0.50, Sonnet 5.5 cache reads $0.10; `gpt-6-sol` / `gpt-6-luna` have their own rows) and their cache minimums (Sonnet 5.5 and Haiku 5.5 512, Mythos Preview 2048); an unknown `gpt-*` id is priced as gpt-6.2-sol (was gpt-5.4, or $10/$50 for any id containing `gpt-6`); Grok ids are priced on xAI rows (grok-4.5 / 4.3 / 4.20 / grok-build, any other Grok id as grok-4.7; all were Sonnet); GPT-6.x / GPT-5.6 cache writes carry their surcharge; `cache_savings_usd` scales with the fast-mode and US-inference multipliers; `/v1/models` entries in the Anthropic shape carry the full Models API fields, Claude Opus 4.5 and Sonnet 4.5 are listed under their dated ids as Haiku 4.5 is (`/v1/models/{id}` resolves the undated id and answers with the dated one; a cursor must be a listed id), retired models are left out unless `lifecycle=retired` is asked for, and the listed ids changed with the current model line-up.

## Observe cost / tokens / cache

```bash
# Cumulative summary (all approximate)
curl -s localhost:8765/_control/stats | jq
# → {"is_estimate":true,"completed_requests":3,"error_requests":0,
#     "totals":{"input_tokens":..,"output_tokens":..,
#               "cache_read_input_tokens":..,"total_usd":..,"cache_savings_usd":..},
#     "cache":{"hits":2,"misses":1,"hit_rate":0.6667,"index_size":2},
#     "by_model":{"claude-sonnet-5-5":{"requests":3,"total_usd":..}}}

# Pseudo prompt-cache index (hit/miss per prefix hash)
curl -s localhost:8765/_control/cache | jq

# Per-request (request, response, usage, cost, cache) history
curl -s localhost:8765/_control/history | jq '.history[-1]'
# Filter and page it: turns after 40, one provider, a model glob, 20 at a time
curl -s 'localhost:8765/_control/history?since_turn=40&provider=anthropic&model=claude-*&limit=20' | jq
curl -s 'localhost:8765/_control/history?limit=20&cursor=60' | jq '.next_cursor, .has_more'
curl -s 'localhost:8765/_control/history?format=jsonl' > history.jsonl   # one entry per line
curl -s localhost:8765/_control/history/12 | jq                            # one turn
# Cheap polling: the short view of each pending (no conversation bodies)
curl -s 'localhost:8765/_control/pending?summary=1' | jq '.pending[].request'
# Prometheus counters and gauges
curl -s localhost:8765/_control/metrics

# Cleanup between tests (wipe pending / history / cache)
curl -s -X POST localhost:8765/_control/clear
```

`cache_savings_usd` is "the approximate amount you would have saved for real thanks to cache hits." Use it to verify your app structures `cache_control` correctly. (Anthropic / Bedrock routes only — the OpenAI route always reports cache status `"none"` and never touches the hit/miss counters.)

What the pseudo cache reproduces (per the official prompt-caching docs):

- **Placement**: block-level `cache_control` (≤ 4 breakpoints) and the **top-level** `cache_control` (automatic caching: one breakpoint on the last cacheable block — `thinking` and empty text blocks are skipped). Prefix order `tools → system → messages`; markers are not part of the key.
- **Minimum cacheable prefix is generation-aware**: Fable / Mythos 5 – 5.2, Opus 5 / 5.5, Sonnet 5.5, Haiku 5.5 = 512 tokens; Opus 4.8 = 1024; Opus 4.7 and Mythos Preview = 2048; Opus 4.6 / 4.5 = 4096; Opus 4.1 / 4 = 1024; Sonnet 5 and earlier = 1024; Haiku 4.5 = 4096; Haiku 3.5 = 2048. Below that the request is observed as `"none"`, like the real API (no error).
- **TTL**: `{"type":"ephemeral"}` = 5 minutes, `{"type":"ephemeral","ttl":"1h"}` = 1 hour (`PUPPETLLM_CACHE_TTL` / `PUPPETLLM_CACHE_TTL_1H` to shorten both for tests), refreshed on read. Writes are reported split by TTL in `usage.cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` and priced at 1.25x / 2x input.
- **Invalidation**: changing `output_config.effort`, `thinking`, or `tool_choice` between turns invalidates the *messages*-level prefixes; switching `speed` (fast mode) invalidates system + messages but not tools. This is a deliberate simplification of the official invalidation table, which marks the tools / system caches as *model-specific* for thinking and effort (some models render that configuration ahead of the system prompt); puppetllm models them as messages-only. Explicit defaults equal omission: `effort: high`, `tool_choice: auto` (also with `disable_parallel_tool_use: false`), and the model's default thinking mode (`adaptive` on Opus 5 / Sonnet 5 / Fable / Mythos, `disabled` on other models; `display: omitted`). A model switch invalidates everything.
- **Validation**: layouts the real API rejects get a `400 invalid_request_error` here too (no pending is created): a malformed marker or unknown `ttl`, more than 4 explicit breakpoints, a marker on a `thinking` block, a 1h breakpoint after a 5m one, top-level `cache_control` when 4 explicit breakpoints already exist, and a top-level `ttl` that disagrees with an explicit marker on the last block.
- **Fast mode**: `speed: "fast"` is priced at the official 2x (Anthropic / Bedrock routes only), reported as `usage.speed: "fast"`, and relayed to an Anthropic upstream together with the app's `anthropic-beta` header (dropped with a warning towards OpenAI-compatible upstreams). Model / platform eligibility is not enforced.
- **Lookback**: each breakpoint looks back ≤ 20 positions, where a run of consecutive `tool_use` (or `tool_result`) blocks counts as one position.
- **Usage shape**: responses carry the full 2026 usage object (`cache_creation`, `output_tokens_details.thinking_tokens`, `server_tool_use`, `service_tier` (`standard` / `batch`), `inference_geo`, `speed` when fast mode was requested), and `message_delta.usage` repeats the cumulative input / cache counts as the real stream does.

Pricing follows the official per-generation tables of Anthropic, OpenAI and xAI (e.g. Fable 5.2 $8/$40 with $0.20 cache reads, Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50, Opus 5 $5/$25, Sonnet 4.6 $3/$15; gpt-6.2-sol $2/$10; grok-4.7 $2/$6). Models with long-context pricing are billed on their higher card for the whole request once the prompt (every input token: uncached, cache writes and cache reads) passes the threshold — Haiku 5.5 over 100,000 tokens, GPT-6.x, GPT-5.6, GPT-5.5 and GPT-5.4 (with their pro tiers; not GPT-5.4 mini / nano, nor codex ids, which use one flat codex row) over 272K, Grok from 200K — and the cost then carries `long_context: true`; `cache_savings_usd` in the stats uses the same card. OpenAI's page gives the 272K threshold without saying that the whole request moves to the long card; that is assumed here. GPT-6.x and GPT-5.6 charge for cache writes (1.25x input with 30-minute retention, 2.5x with 12-hour); the relay calls Chat Completions, where writes are kept for 30 minutes, so the cache writes an upstream reports are priced at that rate. Unknown Claude ids fall back to Sonnet 4.x pricing, unknown `gpt-*` ids (an unlisted GPT-6 id included) to gpt-6.2-sol, and a Grok id without its own row to the grok-4.7 row.

Every history entry carries the correlation and provenance of the exchange: `seq` (the
record sequence, in completion order; the history cursor), `request_id` (the id the
response header carried: `request-id`, `x-amzn-requestid` or `x-request-id`; null for
batch entries, which carry `batch_id` / `custom_id` or `job_id` / `record_id` instead),
`path`, `latency_ms` (arrival to answer on the server's clock), `usage_source` (`estimate`
from the approx tokenizer, `override` when the responder supplied real counts, null on an
error), `pricing_model` (which price table the cost used) and, when a relay answered,
`provenance` (`upstream_provider`, `upstream_model`, `translation.dropped_fields`).
The snapshot keeps an allow-list of request headers — `user-agent`, `anthropic-version`,
the OpenAI organization / project ids, W3C / B3 / vendor trace ids (`traceparent`,
`x-cloud-trace-context`, `x-datadog-trace-id`, `sentry-trace`, …; not `tracestate` or
`baggage`, which carry key=value lists),
`x-request-id` and the other common request ids, botocore's `amz-sdk-invocation-id`,
`x-stainless-*` — and nothing else, so a credential, a session or a client address never
reaches history. `cache` explains a non-hit: `reason`
is `no_cache_control`, `below_minimum` (with `minimum_tokens` and
`eligible_prefix_tokens`, the size of the deepest marked prefix), `first_seen`,
`prefix_changed` (the request differs from the compared one from `first_changed_path`
on — a changed static prefix such as `system[0]` or `tools[1]`, or an edited turn such as
`messages[3].content[1]`; `matched_through` is the number of leading segments that were
unchanged — while a conversation that shares exactly its tools / system with the model's
previous request is `first_seen`), `parameters_changed` (same content, a prompt-rendered parameter such as
`params.speed` or `params.thinking` differs), `ttl_expired`, `lookback`, or `disabled` /
`throttled` on requests the pseudo cache did not look at. A request that sends a
`diagnostics` object (the cache-diagnosis beta's request shape) gets the same explanation
back on the Message as `diagnostics.cache_miss_reason` (plus `cache_status`,
`matched_through`, `first_changed_path`, `minimum_tokens`, `eligible_prefix_tokens` —
puppetllm's own fields, not the real beta's), in the non-stream body and in
`message_start`, on the Anthropic route and the Bedrock Mantle alias. The comparison is
against the message named by `diagnostics.previous_message_id` when the server answered
it, else against the model's previous request.

`PUPPETLLM_HISTORY_FILE` (or `serve --history-file`) appends every entry to a JSONL file as
it is recorded (a synchronous append to a local file, in history order; the path is
checked at start-up — `--history-file` refuses an unusable path, the environment variable
only warns — and a write error is reported once and never fails the request; client text
that is legal JSON but not UTF-8, such as a lone surrogate, is escaped there and in every
JSON response rather than failing them) — a durable log that `/_control/clear` does not
touch; like the metrics it is append-only, so the
early answers of a batch create that was rolled back stay in it. `--log-format json` (or
`PUPPETLLM_LOG_FORMAT=json`) turns stderr into one JSON object per lifecycle event —
`received`, `matched` (rule / rate limit / default policy / unmatched error / timeout),
`resolved` (outcome, latency, cost), `rejected` (a request route's 400 before any pending existed — not `count_tokens` or batch-envelope checks),
`canceled` (clear, disconnect, a cancelled handler, a batch canceled / expired / stopped
or rolled back, an answer that could not be encoded) — correlated by
`pending_id`, `request_id` and the batch ids; the default `text` format keeps only the
adapters' receipt lines. `GET /_control/metrics` is a Prometheus
text endpoint: cumulative counters (`puppetllm_requests_total{provider,outcome}`,
`puppetllm_tokens_total{provider,kind}`, `puppetllm_cost_usd_total{provider}`,
`puppetllm_cache_lookups_total{status}`, `puppetllm_harness_answers_total{source}`,
`puppetllm_rejected_total{provider}`, `puppetllm_canceled_total{reason}`,
`puppetllm_latency_seconds_total{provider}` — latency, like `latency_ms`, follows the
server's clock, so `/_control/clock/advance` while a request waits counts) that survive
`/_control/clear`, and live gauges
(`puppetllm_pending`, `puppetllm_turns`, `puppetllm_rules_unconsumed`,
`puppetllm_clock_offset_seconds`).

Prices come from the built-in family table; `PUPPETLLM_PRICING_FILE` (or `serve
--pricing-file`) adds your own — `{"<model glob>": {"input": 0.5, "output": 2.0,
"cache_write"?, "cache_read"?, "cache_write_1h"?}}` in dollars per million tokens,
consulted before the built-in families (the glob becomes the `model_family` in costs).
Globs are tried in file order against the whole id (`*claude-*` to catch Bedrock /
Vertex ids), case-insensitively; rates are JSON numbers up to 1,000,000 with `cache_read`
at most `input`, no other keys; an empty object clears the table. A custom entry is one
flat card: it cannot carry long-context prices, so a glob that matches a model with a
built-in long card replaces that card too. The fast-mode and
US-inference multipliers apply to Claude ids only (a built-in Claude table, an unknown id
that names Claude, or any made-up id arriving on an Anthropic-shaped route) — never to a
custom table, an OpenAI or Grok family or an upstream model a relay prices by that nothing
matches. A `pricing_model` nothing
matches is priced by the default family (Sonnet, or gpt-6.2-sol for `gpt-*`), as an unknown
inbound id is.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PUPPETLLM_CACHE_TTL` | `300` | Pseudo-cache TTL for 5-minute breakpoints (seconds) |
| `PUPPETLLM_CACHE_TTL_1H` | 12 × `PUPPETLLM_CACHE_TTL` | TTL for `ttl: "1h"` breakpoints (seconds); defaults to 3600 and scales with the 5m TTL when unset |
| `PUPPETLLM_S3_ROOT` | a per-process temp dir | Where the S3 emulation stores objects (batch inference I/O). Set it to keep them across restarts, or to see them from the host under Docker (add the variable and a matching volume to your compose override; the shipped `docker-compose.yml` defines neither) |
| `PUPPETLLM_CACHE_HONOR_TTL` | `1` | `0` ignores the TTL (entries live forever) |
| `PUPPETLLM_CACHE_MIN_TOKENS` | (per-model) | Override the minimum cache threshold. `0` disables it (cache every prefix). Unset = the generation-aware table (Fable / Opus 5 and 5.5 / Sonnet 5.5 / Haiku 5.5 512, Opus 4.8 1024, Opus 4.7 2048, Opus 4.6 / 4.5 4096, other Sonnet 1024, Haiku 4.5 4096, …) |
| `PUPPETLLM_PENDING_TIMEOUT` | (none) | Seconds before an unanswered pending gets the timeout error ([Testing guide](testing.md)) |
| `PUPPETLLM_DEFAULT_RESPONSE` | (none) | Text, or a JSON `/_control/respond` body, that answers unmatched requests; setting it selects `on_unmatched: default` |
| `PUPPETLLM_ON_UNMATCHED` | `pending` | `pending` / `default` / `error` |
| `PUPPETLLM_SEED` | (none) | Seed for latency jitter |
| `PUPPETLLM_CONFIG` | (none) | JSON file `{"config": {...}, "rules": [...], "prompts"?: {...}}` loaded at startup |
| `PUPPETLLM_URL` | (none) | Makes the `puppet` pytest fixture use a running server instead of starting one |
| `PUPPETLLM_HISTORY_FILE` | (none) | Append every history entry to this JSONL file ([Observe cost / tokens / cache](#observe-cost--tokens--cache)) |
| `PUPPETLLM_LOG_FORMAT` | `text` | `json` for one JSON event per line on stderr ([Observe cost / tokens / cache](#observe-cost--tokens--cache)) |
| `PUPPETLLM_PRICING_FILE` | (none) | JSON price table consulted before the built-in families ([Observe cost / tokens / cache](#observe-cost--tokens--cache)) |

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
