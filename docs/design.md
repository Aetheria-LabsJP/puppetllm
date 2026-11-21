**English** | [日本語](ja/design.md) · [← README](../README.md)

# Design notes

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
    puppetllm.tests.test_compat puppetllm.tests.test_observability \
    puppetllm.tests.test_extras \
    puppetllm.tests.test_boto3_interop -v
```

`puppetllm/tests/test_fake_server.py` is an executable specification of the expected behavior; `test_harness.py` covers [Testing guide](testing.md) (rules, policies, timeouts, latency, rate limits, the clock, `count_tokens` / `models`, the CLI and `puppetllm.testing` against an in-process uvicorn).

`test_boto3_interop.py` is the only module that drives **real** boto3 / botocore (`>= 1.43`,
the first service model with every field it exercises) against a running uvicorn instance — every other test encodes and decodes the Bedrock event stream
with puppetllm's own codec, so a matching encoder/decoder mistake would pass unnoticed
there. It needs `boto3` (in `requirements.txt`); the compose test profile sets
`PUPPETLLM_REQUIRE_SDK_TESTS=1` so a run that cannot import it fails instead of quietly
reporting `OK` with every SDK test skipped.

## Layout

```
puppetllm/
├── puppetllm/              # package itself
│   ├── fake_server.py      # canonical core + Anthropic /v1/messages + /_control/* + count_tokens / models
│   ├── harness.py          # scenario rules, unmatched policy, latency, rate limit, fake clock
│   ├── control_models.py   # typed /_control/* request bodies (published in /openapi.json)
│   ├── capabilities.py     # compatibility matrix (/_control/capabilities) + the catch-all for unimplemented paths
│   ├── __main__.py         # the `puppetllm serve|relay|wait` command line
│   ├── testing.py          # test-side client (Puppet, serve())
│   ├── pytest_plugin.py    # the `puppet` pytest fixture
│   ├── observability.py    # JSON log events + Prometheus metrics (/_control/metrics)
│   ├── _responses.py       # the JSON response class every route uses (surrogate-safe)
│   ├── batches.py          # Anthropic Message Batches route + batch control endpoints
│   ├── cache_sim.py        # pseudo prompt cache
│   ├── pricing.py          # approximate tokens + pricing
│   ├── relay.py            # relay responder (cross-provider bridge to a real API)
│   ├── openai_wire.py      # pure OpenAI ↔ canonical conversions shared by the adapter and the relay
│   ├── providers/          # Bedrock (invoke / converse / batch) + OpenAI adapters, S3, AWS event stream
│   │                       #   bedrock.py, converse.py, bedrock_batch.py, s3.py, openai.py, eventstream.py
│   └── tests/              # unit tests
├── docs/                   # the guides (docs/ja/ in Japanese): quickstart, testing, control API, bedrock, batches, relay, design
├── examples/               # tool_loop.py (a scripted tool loop), test_retry.py (the pytest fixture)
├── responder/              # instruction docs for the responder (the agent that "plays the LLM")
│   ├── CLAUDE.md           #   for Claude Code
│   └── AGENTS.md           #   for Codex CLI and other agents following the AGENTS.md convention
├── LICENSE
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml          # `pip install .` → `puppetllm` command + pytest plugin
└── requirements.txt
```
