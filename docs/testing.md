**English** | [日本語](ja/testing.md) · [← README](../README.md)

# Testing guide

## Scenario rules, timeouts, latency, rate limits, fake clock

The flows in the [quickstart](quickstart.md) need a responder. The harness answers requests **by itself** from a
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
| GET  | `/v1/models`, `/v1/models/{id}` | The catalogue (`puppetllm/catalog.py`), in the Anthropic shape (`x-api-key` or `anthropic-version` present, or no auth header; each model with `max_input_tokens`, `max_tokens`, `line`, `lifecycle` / `deprecated_at` / `retires_at` and `capabilities`; pages with `limit` / `after_id` / `before_id`; `lifecycle` (repeatable, also spelled `lifecycle[]`) filters by `active` / `deprecated` / `retired`, retired models being left out by default) or the OpenAI shape (bare `Authorization: Bearer`). The `lifecycle` field is the state the provider published, fixed in the catalogue (a past `retires_at` does not retire a model); retired models are listed without `capabilities`; `created_at` / `created` is one fixed placeholder, not a release date. Informational: any model id is accepted by the request routes, and an unlisted id is synthesized (as `active`, without limits or capabilities) rather than refused |

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
`pending` (`summary=` / `fields=`), `history` (filters, `history_page` for the cursor
envelope, `history_entry`), `metrics`, `stats`, `cache`, `config`, `advance_clock`, `clear`;
`clear(config=True)` also restores the defaults, `reset(baseline)` re-applies a saved
`baseline()` — configuration plus rules). With `pip install .` the `puppet` pytest fixture
(one server per session; before and after every test the state is cleared and the
configuration and rules are put back to what the server started with, so `PUPPETLLM_*`
settings and a `--config` file's rules stay in force while one test's `puppet.config(...)`
or `puppet.expect(...)` never leaks into the next; `PUPPETLLM_URL` points it at an
external server instead) is registered
automatically; `pip install ".[test]"` adds the SDKs. A startup script can come from a
file — `puppetllm serve --config scenario.json` with
`{"config": {...}, "rules": [...], "prompts"?: {...}}` — or from environment variables (see
[Environment variables](control-api.md#environment-variables)) for Docker.

`Expectation.respond` also takes `choices` (one answer per OpenAI choice), `logprobs` and `provider_response` (a Bedrock `trace`), and `Puppet.prompts(templates, replace=False)` registers prompt-management fixtures (part of `baseline()` / `reset()`); see the [Control API reference](control-api.md).
