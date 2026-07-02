"""Provider adapters for puppetllm.

The canonical core (fake_server.py) holds the provider-independent normalized
snapshot + /_control/*, and each provider here is responsible only for the
"wire-format input/output":

- anthropic: `/v1/messages` (SSE / single JSON)  — currently implemented inside fake_server.py
- bedrock  : `/model/{id}/invoke[-with-response-stream]` (AWS eventstream / JSON)
- openai   : `/v1/chat/completions` (SSE chunks / single JSON)

To add a new provider (e.g. Vertex), add a `<name>.py` to this package and have
`build_router()` return a FastAPI router that fake_server includes (see openai.py /
bedrock.py as worked examples).
Content blocks / control API are shared, so each adapter only needs to write
request decode and response encode.
"""
