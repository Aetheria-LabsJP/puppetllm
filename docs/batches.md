**English** | [日本語](ja/batches.md) · [← README](../README.md)

# Message Batches API

## Lifecycle

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
