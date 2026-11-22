**English** | [日本語](ja/agent-sdk.md) · [← README](../README.md)

# Agent SDK conversion mode (experimental)

`relay --kind agent-sdk` converts incoming `system / messages / tools` into a
Claude Agent SDK query. It validates the SDK's structured output and returns
the next `text / tool_use` blocks as a Messages API response. Your existing
Claude Code remains responsible for tool execution, approvals, and history.

```text
Claude Code → puppetllm /v1/messages → Agent SDK query() → Claude
Claude Code ← text / tool_use + SSE ← validated structured output
```

## Docker Compose (no host virtual environment needed)

Set the real upstream `ANTHROPIC_API_KEY` in the launch shell, or copy
[`.env.example`](../.env.example) to `.env` and fill it in. When switching from
manually started server/relay processes, stop them first to free port 8765.

```bash
# From this repository: start the server, relay and watchdog in the background
./start-agent-sdk.sh

# Status and relay logs
docker compose --profile agent-sdk ps
docker compose logs -f agent-sdk-relay
```

`./start-agent-sdk.sh --model claude-sonnet-4-6 --effort medium` selects another model and
effort for that start (`--help` lists the forms).

The script builds the images, preserves an existing server, and starts the
worker and status watchdog. Defaults: Opus 5.5, effort `high`, concurrency 4,
timeout 600 seconds. Override them with `PUPPETLLM_AGENT_SDK_MODEL`,
`PUPPETLLM_AGENT_SDK_CONCURRENCY`, `PUPPETLLM_AGENT_SDK_TIMEOUT`, and
`PUPPETLLM_AGENT_SDK_TARGET` in `.env`; `.env.example` lists the remaining
`PUPPETLLM_AGENT_SDK_*` variables (turn, retry and idle-warning settings).
Set `PUPPETLLM_AGENT_SDK_EFFORT` and `PUPPETLLM_AGENT_SDK_MAX_OUTPUT_TOKENS` to control
reasoning effort and the total output cap. For example:

```bash
PUPPETLLM_AGENT_SDK_MAX_OUTPUT_TOKENS=32768 ./start-agent-sdk.sh --effort low
```

Lower effort trades reasoning depth for speed; too small a cap can prevent a complete response.
An API `max_tokens` stop returns an output-limit error and does not hand off
partial tool declarations. `/status` also exposes `observed_stop_reason`.
Explicit `--model` and `--effort` arguments take
precedence over their environment and `.env` settings. They apply to that startup
without rewriting `.env`; save persistent defaults in `.env`.
Plain `docker compose up -d` continues to start only the fake server.

Run Claude Code on the host from your working directory. No project bind mount
or continuously open server/relay terminals are needed.

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8765 \
ANTHROPIC_API_KEY=puppet-local \
claude --model claude-opus-5-5
```

After code changes, run `./start-agent-sdk.sh` again with the same flags: it rebuilds and
replaces the relay and watchdog and keeps the running server.

Stop the services with `docker compose --profile agent-sdk down`. Server
history is in memory and resets when the server container is recreated.

## Status and recovery after a timeout

```bash
curl -sS http://127.0.0.1:8766/health
curl -sS http://127.0.0.1:8766/ready
curl -sS http://127.0.0.1:8766/status

# Pause SDK execution while keeping the server and watchdog available
docker compose stop agent-sdk-relay
```

`/health` checks the watchdog process. `/ready` returns 200 for a fresh worker
heartbeat that allows new work, or 503 when stopped, stale, or unable to persist
state. `/status` reports SDK event and generation times/counts, SDK-reported usage
and estimated cost.
Unobservable internal retry totals remain `null`. `api_attempt_count` counts
Messages requests dispatched by the egress gate; `api_requests_blocked` counts
requests rejected before dispatch. Counts, token estimates and cost are SDK-side
figures, not a billing statement. Progress records
exclude conversation text, thinking, signatures, tool inputs, and credentials.

Opus 5.5 can report estimated thinking tokens without visible thinking text.
In that case, check `stage: thinking`, `estimated_thinking_tokens`, and
`last_thinking_progress_at`; `generated_characters: 0` alone does not mean there
is no model progress.

Timeouts, SDK failures, cancellation and uncertain response delivery persistently
block the same effective input. Resending with a new request ID does not start
another SDK execution, even if the caller ignores `x-should-retry: false`.
The `agent-sdk-state` volume stores fingerprints and fixed failure codes; normal
restarts do not clear them. Different inputs are handled independently. A blocked
request gets a non-retryable 503 whose message contains its key, even while `/ready`
returns 200 for other work.

After fixing the cause, stop the worker and explicitly clear the affected key:

```bash
docker compose stop agent-sdk-relay
docker compose --profile agent-sdk run --rm --no-deps agent-sdk-relay \
  python3 -m puppetllm.sdk_state blocks
docker compose --profile agent-sdk run --rm --no-deps agent-sdk-relay \
  python3 -m puppetllm.sdk_state clear --key YOUR_64_CHARACTER_KEY
docker compose start agent-sdk-relay
```

Replace `YOUR_64_CHARACTER_KEY` with the listed fingerprint. After a forced
process exit, first run `recover --all` while stopped to mark abandoned active
entries failed, then clear the chosen key. Maintenance is refused while a worker
holds the state lock. Normal shutdown should not use `down -v`, which also
deletes the protection volume. The watchdog returns non-retryable 503 responses
for pending work while the worker is unavailable. Use this mode exclusively,
without another responder sharing its queue.

## Manual launch (development/debugging)

```bash
# Once, from this repository in a Python virtual environment
pip install -e '.[agent-sdk]'

# Terminal 1
puppetllm serve --port 8765

# Terminal 2: configure the SDK's credentials in this shell first
# For example, set your real ANTHROPIC_API_KEY here
puppetllm relay --kind agent-sdk --model sonnet \
    --max-concurrency 4 --timeout 300 --price-by upstream

# Terminal 3: run the existing Claude Code from your working directory
ANTHROPIC_BASE_URL=http://127.0.0.1:8765 ANTHROPIC_API_KEY=puppet-local claude
```

The SDK uses the relay process's authentication. Do not set the frontend's dummy
key in that process. `--api-key-env MY_KEY` selects another API key variable.
State (the failure guard, progress and the SDK working directory) lives in
`--sdk-state-dir` (default `~/.local/state/puppetllm/agent-sdk`, or
`PUPPETLLM_SDK_STATE_DIR`). Outside Docker the maintenance commands are
`python -m puppetllm.sdk_state blocks|clear|recover`, and
`python -m puppetllm.sdk_state watch` serves the status endpoints on
`127.0.0.1:8766` (`--host` to change).
The SDK subprocess explicitly uses `https://api.anthropic.com`, or `--target` if
provided, rather than inheriting the frontend's `ANTHROPIC_BASE_URL`. Machine-wide
Claude Code managed settings that set the endpoint override this; the relay then
refuses the answer with 503 and stops accepting requests, since the per-query
request limit could not be enforced. A target
pointing back to the same puppetllm endpoint is rejected. The target must use
the Anthropic API format.
Subagents and auto-mode permission checks use the same relay, so concurrency 1
can make them time out while queued. The example uses 4. When an outer client
disconnects, the relay cancels its upstream query and releases the slot.

- Each outer API request gets a fresh SDK session with the complete outer
  history. The input puts the system instructions
  and tool catalogue first, followed by per-message JSON records in separate
  content blocks. An explicit cache breakpoint at the end of the history lets
  subsequent requests reuse unchanged prefixes.
  `--sdk-cache-ttl 1h` (Compose: `PUPPETLLM_AGENT_SDK_CACHE_TTL`; default `5m`)
  keeps cache entries for an hour: writing them costs more than the five-minute
  kind, so it pays off only when requests are often more than five minutes apart.
  Images follow their message
  record. Initial requests and changed context/tools/history still need to
  process the changed suffix; conversation text and tool history are not
  summarized or omitted. Edits and `/compact` use the outer client's history.
  Historical assistant `thinking` / `redacted_thinking` blocks, including their
  signatures and opaque data, are omitted without converting reasoning into
  ordinary conversation text. Adjacent text, tool calls and results are retained.
  Omissions are recorded once per block type in `provenance.translation.dropped_fields`;
  strict translation rejects them. Relay logs
  elapsed time and upstream cache read/write tokens to help measure reuse.
  Built-in tools, MCP, user settings,
  hooks, and session persistence are disabled. `Read` and `Bash` are returned
  as data for the caller to execute, never executed by the inner agent.
  Composed input is sent verbatim, without expanding `@path` mentions or slash
  commands embedded in the transcript.
  If the SDK emits a native `Read`, `Bash`, or another caller-provided tool call
  instead of a structured envelope, the adapter validates and hands it off
  after collecting the text and all tool blocks through the SDK's `message_stop`
  and checking the stop reason and arguments. It closes the inner session to
  keep local tool-denial feedback out of the answer. Errors reported before the
  terminal delta and incomplete responses are failures.
  **Upstream completion has an unresolved limitation.** With SDK 0.2.182 / bundled
  CLI 2.1.325, an upstream error after `message_delta(stop_reason=tool_use)` but
  before `message_stop` causes the CLI to emit a stop notification identical to
  normal completion. The adapter can hand off tool declarations in this case.
- Supports text, base64 images, custom tool definitions, `tool_use`, and text/image
  `tool_result`. Images retain their position in the JSON transcript and are
  attached using [SDK image input](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode)
  (PNG, JPEG, GIF, WebP). Mid-conversation `role: "system"` text instructions retain their
  role and position. Per-message settings such as `output_config.effort` are
  recorded as dropped rather than silently ignored.
  Custom-tool `tool_addition` / `tool_removal` records (definitions or name
  references) and tool-search result `tool_reference` blocks are supported.
  Deferred loading and historical changes are applied in order; declarations
  are validated against the currently active schemas. Changes to that catalogue
  change the input prefix and can cause a history-cache miss for that request.
  Tool names, input JSON Schemas, and `tool_choice` are validated;
  JSON strings are decoded only for explicitly typed boolean, number, integer,
  array, or object fields (e.g. `Bash.timeout: "120000"` or a stringified
  `AskUserQuestion.questions` array). Properties and array items are normalized
  recursively; required fields receive a default only if the schema declares one.
  String/union fields retain their values. Bounds, integer constraints, array
  limits, and the rest of the schema are still validated.
  Other invalid arguments are rejected with their field/schema rule (not their
  value or dynamic map keys) in the error and an `x-should-retry: false` header.
  Non-finite numbers, including exponent overflow, are rejected before delivery.
  Tool call IDs are generated by the adapter. URL images, documents,
  and server-side tools are explicitly rejected with 400. Resuming an old session
  containing these blocks and full Claude Code feature compatibility are outside
  this mode's current scope.
- **This is semantic conversion, not lossless forwarding or equivalent model
  behavior.** Outer `max_tokens`, thinking, sampling, cache, beta settings, etc.
  are not preserved; warnings and `provenance.translation.dropped_fields` record
  them. `--strict-translation` rejects such requests before calling the SDK,
  including ordinary Messages requests containing `max_tokens`.
  Use `--sdk-effort` and `--sdk-max-output-tokens` to set the inner reasoning effort
  and total thinking-plus-envelope output cap explicitly. Effort defaults to
  `high` and the cap to `auto`; explicit effort `auto` uses the SDK/model default.
  A request's own top-level `output_config.effort` is honoured up to that value:
  `--sdk-effort` is the ceiling, so a client asking for less spends less, and with
  `auto` the request's value is used as is. Other outer generation settings are not copied.
  Explicitly changed generation settings produce a different execution fingerprint.
- `--sdk-max-turns` (default 1) bounds inner agent execution. A per-query egress gate enforces
  `--sdk-max-api-requests` (default 1; Compose: `PUPPETLLM_AGENT_SDK_MAX_API_REQUESTS`)
  on actual upstream Messages HTTP requests. The CLI can try output-limit
  recovery even with retries disabled; excess requests are rejected before
  dispatch. HTTP retries default to `--sdk-api-retries 0`.
  `--sdk-structured-retries 1` preserves delivery of a completed
  native external-tool declaration; setting it to 0 can truncate valid calls.
  Raising the turn limit can introduce additional inner API requests.
  `--sdk-idle-warning` (default 120 seconds) warns when no generation event is
  observed; it does not cancel.
  `--sdk-max-buffer-bytes` (default 4194304) bounds each SDK
  stdout JSON line in UTF-8 bytes; limit errors identify this setting so it can
  be raised for larger outputs. A custom transport supplements the SDK's default
  handling with byte counting and pipe cleanup for exited subprocesses. It
  depends on SDK internals, so the package extra and the Docker image pin SDK
  0.2.182; run the integration tests when upgrading the SDK.
  `--timeout` cancels the SDK call and returns 504; failed
  SDK runs and invalid output return 502, except that an SDK-reported upstream
  HTTP error retains its status. History length is left to the upstream API;
  an over-long request returns 400 `invalid_request_error` with
  `prompt is too long: N tokens > M maximum`, so the client that owns the history
  can compact it and resend; this rejection leaves no block behind. When the CLI
  tries a further request after an upstream error or refusal and the request limit
  stops it, the first response decides the answer (its HTTP status, or
  `stop_reason: refusal`). Errors identify turn limits, structured-output
  retry limits, or the upstream HTTP status without including raw conversation
  or credential text. SSE is emitted after the full answer
  is ready. Usage comes from the SDK and includes the conversion prompt and
  structured output overhead. Real model use incurs real cost.
- `--dry-run` logs the converted input without invoking the SDK. `--only`,
  `--model-map`, and `--max-requests` work as in the other relay modes.

Run the tests in Docker without a host Python environment:

```bash
docker compose --profile agent-sdk build agent-sdk-relay
docker run --rm --network none -e PUPPETLLM_TEST_REAL_AGENT_SDK=1 \
  puppetllm:agent-sdk \
  python3 -m unittest puppetllm.tests.test_agent_sdk puppetllm.tests.test_sdk_egress puppetllm.tests.test_sdk_guard \
    puppetllm.tests.test_sdk_runtime puppetllm.tests.test_sdk_runtime_integration \
    puppetllm.tests.test_sdk_state puppetllm.tests.test_sdk_transport -q
```

For host development, install `.[test,agent-sdk]`, then run
`python -m pytest puppetllm/tests/test_agent_sdk.py puppetllm/tests/test_sdk_*.py -q`. Default tests do not
call a model.
Set `PUPPETLLM_TEST_REAL_AGENT_SDK=1` to also run the real SDK and bundled CLI
against local fake responses, including an outer CLI file-read roundtrip. These
integration tests also perform no paid inference.
