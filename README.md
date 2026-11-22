**English** | [日本語](README.ja.md)

# puppetllm — LLM API debug proxy (fake Anthropic / Bedrock / OpenAI server)

A **fake server** compatible with the Anthropic Messages API, Bedrock (InvokeModel, Converse, batch inference with a bundled S3) and OpenAI Chat Completions. Point `ANTHROPIC_BASE_URL` (or the `AnthropicBedrock` / `OpenAI` `base_url`) at it and your app's LLM calls are **held** until a human, another agent, a scripted rule or a relay to a real provider supplies the answer — without changing a line of app code, and without the real API.

- **Zero-cost debugging** — reproduce and inspect an agent's behaviour; a human or an AI agent plays the model.
- **Deterministic tests** — rules answer requests (text / tool_use / errors / 429 sequences), simulate latency and rate limits, and a pytest fixture asserts what the app sent.
- **Cross-provider bridge** — relay an Anthropic-SDK app to GPT / Grok, or an OpenAI-SDK app to Claude.
- **Observability** — every exchange in a history API with tokens, estimated cost, cache hit / miss explanations, JSON logs and Prometheus metrics.

> Token counts and prices come from an approximate tokenizer and the public price tables. They **do not match real billing**; use them for trends and structure.

## Quickstart: two terminals

```bash
# terminal 1 — the fake server
docker compose up -d                       # or: pip install . && puppetllm serve
curl -s localhost:8765/_control/health     # → {"ok":true,"turn_count":0}
```

```python
# terminal 1 — your app, unchanged except base_url (blocks until an answer is injected)
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8765", api_key="sk-mock")
msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=256,
                             messages=[{"role": "user", "content": "What is the weather in Tokyo?"}])
print(msg.content[0].text, msg.usage)
```

```bash
# terminal 2 — the responder: see the held request, answer it
curl -s 'localhost:8765/_control/pending?summary=1' | jq '.pending[0].request'
curl -s localhost:8765/_control/respond -d '{"text": "Sunny, 23 °C."}'
curl -s localhost:8765/_control/history/1 | jq '{usage, cost}'
```

The same works for `AnthropicBedrock`, boto3 (`invoke_model`, `converse`, batch inference) and the OpenAI SDK (`base_url=http://localhost:8765/v1`), streaming included. To answer without a human, script the server:

```python
from puppetllm.testing import serve

with serve() as puppet:                      # in-process server on a free port
    puppet.expect(tools=["get_weather"]).respond(
        content=[{"type": "tool_use", "name": "get_weather", "input": {"city": "Tokyo"}}])
    puppet.expect(has_tool_result=True).respond(text="It is 23 °C and clear in Tokyo.")
    ...                                       # run your agent loop against puppet.url
    puppet.assert_consumed()
    puppet.assert_sent(count=2, tool="get_weather")
```

`examples/tool_loop.py` is that loop end to end; `examples/test_retry.py` shows the `puppet` pytest fixture driving the SDK's 429 retry.

## Documentation

| Guide | What it covers |
|---|---|
| [Quickstart](docs/quickstart.md) | Starting the server, pointing each SDK at it, answering by hand, injecting errors, parallel requests |
| [Testing guide](docs/testing.md) | Scenario rules, unmatched policies, timeouts, latency and rate-limit simulation, the fake clock, `count_tokens` / `models`, `puppetllm.testing` and the pytest fixture |
| [Control API reference](docs/control-api.md) | Every `/_control/*` endpoint, history / metrics / logs, cache diagnostics, pricing files, environment variables, security, behaviour changes |
| [Bedrock](docs/bedrock.md) | InvokeModel, Converse, guardrail and prompt-management fixtures, batch inference with the bundled S3 emulation |
| [Message Batches API](docs/batches.md) | The Anthropic batch lifecycle and how to drive it |
| [Relay mode](docs/relay.md) | Forwarding pendings to a real Anthropic / OpenAI-compatible upstream across SDKs |
| [Agent SDK conversion mode](docs/agent-sdk.md) | Experimental: a relay that answers pendings through the Claude Agent SDK, with its Docker services, status watchdog and safeguards |
| [Design notes](docs/design.md) | Architecture, the compatibility matrix, tests, layout |

The responder's own instructions (for Claude Code / Codex playing the model) live in [`responder/`](responder/).

## Install

```bash
docker compose up -d                 # image puppetllm:latest, localhost:8765
pip install .                        # `puppetllm serve|relay|wait`, the pytest plugin
pip install ".[test]"                # + anthropic / openai / boto3 for the interop tests
docker compose --profile test run --rm proxy-test   # the test suite
```

Python 3.10+. The server binds `127.0.0.1` by default; the control plane has no authentication (see [Security](docs/control-api.md#security)).

## License

MIT — Copyright (c) 2026 Aetheria Labs; see [LICENSE](LICENSE).
