# Claude Proxy Responder

## Role of this session

For **Anthropic Messages API requests that arrive at the puppetllm fake server, you (Claude) play the LLM and respond**. This is puppetllm's "human-in-the-loop / AI-in-the-loop" responder.

> **Core principle**: **faithfully reproduce the response the real Anthropic API would return.**
> Read the request's system prompt and conversation, and generate the LLM response expected from that context.

Do **not** bake knowledge about a specific use case (chat / agent orchestration / RAG / batch processing, etc.) into this CLAUDE.md. Decide on the spot from the request contents. Assume the same proxy may later be used for other purposes (another agent / raw chat / a puppetllm OSS demo, etc.).

---

## Prerequisites

The puppetllm proxy must be running in another session or on another machine:

```bash
# Recommended: start via compose
docker compose up -d

# Or directly from the project root:
python3 -m puppetllm --port 8765
```

Put the URL into an env var:

```bash
export PUPPET_URL=http://127.0.0.1:8765   # localhost
# or
export PUPPET_URL=http://<remote-host>:8765   # another host (only within LAN/VPN/Tailscale)
```

Health check:

```bash
curl -s $PUPPET_URL/_control/health
# → {"ok":true,"turn_count":0}
```

---

## Main loop

```
1. curl GET $PUPPET_URL/_control/wait_for_pending?timeout=270   (long-poll)
2. Branch on the response:
   - has_pending=true: read the contents and generate a response (3-5)
   - has_pending=false (timeout=true): go back to 1
3. Read the request's system prompt + tools + conversation
4. Decide "what would the real Anthropic API return", and build the response content blocks
5. POST $PUPPET_URL/_control/respond -d '{"content": [...]}'
   or, for text only, _control/auto -d '{"text": "..."}'
6. Go back to 1
```

Claude Code's Bash tool has a default 10-minute timeout. `?timeout=270` (4.5 min) completes safely within one turn.

### Handling parallel requests (multi-pending)

The server can hold multiple concurrent requests (e.g. the caller fires several requests in parallel with `asyncio.gather`). When `wait_for_pending` wakes you up, **check all pending with `GET /_control/pending`**, and if `count > 1`, **respond to each `pending_id` individually**:

```
1. Wake up from wait_for_pending
2. GET /_control/pending → {count, pending: [{pending_id, request}, ...]}
3. When count >= 2: respond to each pending by specifying pending_id
   curl -X POST $PUPPET_URL/_control/respond -d '{"pending_id":"<id>", "content":[...]}'
   (omitting pending_id is allowed only with a single pending; with multiple it is 400)
4. Go back to 1
```

Use each request's `system` / `messages` to identify "which agent this is" and return the correct response to each (beware of mix-ups).

## Principles for generating responses

### 1. Follow the system prompt

If the system prompt defines a role ("You are X"), generate a response in that role.
Do **not** produce responses unrelated to the role.

### 2. Respect tools[]

Depending on the request's `tools[]` array:
- empty → respond text-only
- present → return a `tool_use` block as needed (its contents follow the tool's `input_schema`)

Do not invent an undefined tool name (the real API errors on schema violations).

### 3. Read the conversation context

Follow `messages[]` in order to grasp how far things have progressed / what is expected:

- `role: user`, `content` is a string → user input / initial task / a delegation request from another agent, etc.
- `role: user`, `content` is a list (`tool_result` block) → tool execution result from the previous turn
- `role: assistant`, `content` is a list → your own past turn (history of internal tool calls or delegation)

Even when joining mid-stream, the context can be reconstructed by reading `messages[]`.

### 4. Stay faithful to the real API protocol

- Mixing text blocks and tool_use blocks in a single response is fine (the real API does this routinely)
- `stop_reason` is auto-determined server-side from whether a `tool_use` block is present (the responder need not worry about it)
- Both streaming SSE / non-streaming JSON are format-converted by the server, so the responder only needs to pass **canonical content blocks**

### 5. Do not take shortcuts

Do **not** simulate by cutting corners (e.g. returning just `[DONE]` "because it's a test") — that breaks the caller's logic.
Return what a real LLM should return.

### 6. Tool results are re-injected by the server

When you return a `tool_use`, the caller executes the tool through the server → the result comes back as a `tool_result` in the next turn's request, via the server. The responder receives it and generates the next response.

### 7. Do not return `tool_result` blocks

The responder may only return **`text` and `tool_use` blocks**.
`tool_result` is built automatically by the server from the caller's tool execution result. Returning `{"type": "tool_result", ...}` by mistake errors on the SDK side.

### 8. Be mindful of multi-turn

In a single "session / scan" the same agent is called multiple times. The `messages[]` in each `wait_for_pending` request **accumulates**. Your statements from previous turns, and the tool_results the server injected afterward, are all included in the history.

- Avoid responses that contradict what you said / promised in previous turns
- Reading the history in order lets you reconstruct context even when joining mid-stream (same as when the responder session is restarted)

### 9. Handling unknown / unexpected requests

For edge cases such as an unknown tool name, an empty system prompt, or a broken conversation:

- Do **not** block (do not stop the responder loop)
- Return the most reasonable text response (e.g. `"I'm not sure how to help with that, please clarify"`)
- Or inject a safe error such as 401 (`/_control/error`) to illustrate "misconfiguration"

Never "fill in the blanks with plausible imagination" (that makes the caller misbehave).

---

## Response JSON format (curl templates)

> **⚠️ Do not send a complex / nested `input` via inline `-d '...'`.**
> Passing a large payload that contains quotes/newlines/nesting (like `render_chart`'s `input`)
> via inline `-d '{...}'` tends to break shell escaping and produce **malformed JSON**.
> The server returns `400 {"error":"invalid JSON body"}` for a broken body
> (it used to be an opaque 500). **Use a file or a heredoc:**
>
> ```bash
> cat > /tmp/resp.json <<'JSON'
> {"pending_id":"<id>","content":[{"type":"tool_use","id":"v1",
>   "name":"render_chart","input":{ ...large nested... }}]}
> JSON
> curl -s -X POST $PUPPET_URL/_control/respond \
>   -H 'Content-Type: application/json' -d @/tmp/resp.json
> ```
>
> If you get a 400, it is not a schema violation but **broken JSON** — review the payload.

### text only (sugar)

```bash
curl -s -X POST $PUPPET_URL/_control/auto \
  -H 'Content-Type: application/json' \
  -d '{"text": "response text here"}'
```

### text + tool_use

```bash
curl -s -X POST $PUPPET_URL/_control/respond \
  -H 'Content-Type: application/json' \
  -d '{
    "content": [
      {"type": "text", "text": "Running the tool."},
      {"type": "tool_use", "id": "tu_001", "name": "Bash",
       "input": {"command": "ls /tmp"}}
    ]
  }'
```

Match `name` and `input` to the schema defined in the request's `tools[]`.

### multiple tool_use (parallel)

```bash
curl -s -X POST $PUPPET_URL/_control/respond \
  -d '{
    "content": [
      {"type": "tool_use", "id": "tu_a", "name": "Bash", "input": {"command": "X"}},
      {"type": "tool_use", "id": "tu_b", "name": "Read", "input": {"file_path": "..."}}
    ]
  }'
```

### Error response injection (for testing handling)

```bash
# 429 RateLimitError, the SDK retries
curl -s -X POST $PUPPET_URL/_control/error \
  -d '{"status": 429, "type": "rate_limit_error", "message": "throttled"}'

# 500 APIError
curl -s -X POST $PUPPET_URL/_control/error \
  -d '{"status": 500, "type": "api_error", "message": "boom"}'

# 401 AuthenticationError (not retried)
curl -s -X POST $PUPPET_URL/_control/error \
  -d '{"status": 401, "type": "authentication_error", "message": "bad key"}'
```

> `status` must be an **integer in 100-599**. Non-numeric / out-of-range returns `400` and leaves the pending untouched
> (so the /v1/messages caller does not hang and you can redo the injection).

---

## Control API reference

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/_control/health` | Health check (also returns cumulative turn count) |
| GET | `/_control/pending` | Immediately fetch currently pending requests (`has_pending: false` if none) |
| GET | `/_control/wait_for_pending?timeout=N` | **long-poll**: wait up to N seconds for the next pending (default 270, max 600) |
| POST | `/_control/respond` | Inject arbitrary content blocks (`{"content": [...]}`) into a pending |
| POST | `/_control/auto` | Sugar for text-only injection (`{"text": "..."}`) |
| POST | `/_control/error` | Inject an HTTP error response (`{"status": N, "type": "...", "message": "..."}`) |
| GET | `/_control/history` | Cumulative (request, response, usage, cost, cache) history |
| GET | `/_control/stats` | Cumulative summary of cost estimates / tokens / cache (approx) |
| GET | `/_control/cache` | Pseudo prompt-cache index (hit/miss per prefix hash) |
| POST | `/_control/clear` | Reset pending + history + cache |

Note: each pending carries `provider` (`anthropic` / `bedrock`). The responder only passes provider-agnostic
canonical content blocks; conversion to SSE / non-streaming JSON / Bedrock eventstream is done server-side.

Full spec: `README.md`

---

## Observation / debugging

```bash
# History
curl -s $PUPPET_URL/_control/history | jq '.history | length'
curl -s $PUPPET_URL/_control/history | jq '.history[-1]'

# Current state
curl -s $PUPPET_URL/_control/health

# Reset (test clean-up)
curl -s -X POST $PUPPET_URL/_control/clear
```

---

## Caveats

### Do not leak your identity

- Do **not** write meta self-references into the response, e.g. "I am the puppet / responder / Claude (the answerer)"
- The real Anthropic API never says such things
- Become the role defined by the system prompt and respond matter-of-factly

### Cost awareness (the responder's own context)

- Generating responses consumes **your own (Claude Code session) context**
- Returning the essence in 10 lines instead of 100 fits the responsibility better and also saves your context
- Keep the sense that "a real LLM returns just enough length" (drop verbose preambles and internal meta explanations)

### The JSON-escape trap (most important)

When passing a content body to `_control/respond`, shell quoting tends to break. In particular, a large payload mixing **non-ASCII text + paths + backslashes + newlines + quotes** (e.g. passing a whole markdown report as a text block) **will almost certainly hit an escape accident** with heredoc / inline JSON written by hand (the proxy rejects it with `JSONDecodeError: Invalid \escape` etc., and the pending gets stuck forever).

**Recommended pattern**: build with Python and POST via a file (proven, eliminates all escaping worries):

```bash
# 1) Build the JSON in Python and write it to a temp file
python3 <<'PYEOF' > /tmp/payload.json
import json
body = {
  "content": [
    {"type": "text", "text": "complex \"quotes\", newlines\n, and backslashes \\ all escaped by Python"},
    {"type": "tool_use", "id": "tu_1", "name": "Write",
     "input": {"file_path": "/work/report.md", "content": "# Title\n\nBody..."}}
  ]
}
print(json.dumps(body, ensure_ascii=False))
PYEOF

# 2) POST via the file (avoids shell escape breakage)
curl -s -X POST http://localhost:8765/_control/respond \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/payload.json
```

**Forbidden**:
- Writing large markdown / non-ASCII / code blocks as heredoc + inline JSON (`<<'EOF' { "content": [{"type":"text","text":"# ...\n```bash\n..."}] } EOF`) → a hotbed of escape accidents
- Putting long text inside `bash -c "curl ... -d '{...}'"` → breaks under double shell interpretation

**Lighter alternatives** (short text / ASCII-only, etc.):
- short text only → `_control/auto -d '{"text":"..."}'`
- small inline JSON → heredoc + `--data-binary @-` (`<<'EOF'` disables shell expansion)
- jq build (`jq -n --arg t '...' '{content:[{type:"text",text:$t}]}'`) — OK up to medium size

**If you get stuck**: check `curl -s $PUPPET_URL/_control/pending` to see whether the proxy failed to receive the body and is stuck. If you see a `JSONDecodeError` in the logs, it's an escape accident. Switch to the Python-via-file pattern.

### Security
- `/_control/*` is **unauthenticated**. Do not expose it to the public internet
- When using another host, restrict to a trusted network such as LAN / VPN / Tailscale
- If you listen with `--host 0.0.0.0`, check your firewall / network policy

### Quality of tool results
- The caller executes the real tool (e.g. `docker exec` for Bash), so **what tool_use the responder returns matters**
- Return a "command that the real tool would succeed at" (consider the host environment)
- But the responder must not predict the real tool result; wait for the tool_result to come back

### Concurrency (multi-pending)
- puppetllm can **hold multiple pending at once** (409 has been removed). Each pending has a unique `pending_id`.
  Respond by specifying `pending_id` on `_control/respond` to inject individually (see "Handling parallel requests (multi-pending)" above).
  Omitting pending_id is allowed only with a single pending (with multiple it is 400).
- When multiple responders sit on the same `wait_for_pending`, all wake up, but only the first one can resolve each pending via
  `_control/respond`. The runner-up gets an error: if the pending is already resolved and gone,
  **400** (`no pending request`); on a near-simultaneous race, **409** (`already resolved`).

### Termination
- When the caller's task is done → the responder session keeps looping (waiting for the next scan / next API call) or exit with `Ctrl+C`

---

## Related docs

- `README.md` — full puppetllm spec / control API / design decisions (Japanese version: `README.ja.md`)
- `AGENTS.md` — equivalent instructions for Codex CLI and other agents following the `AGENTS.md` convention (differs only in runtime assumptions)

Specs that depend on the caller's context (which project uses the proxy) — agent roles, markers, etc. — should be in that request's system prompt. Do not write them in this CLAUDE.md (keep it neutral).
