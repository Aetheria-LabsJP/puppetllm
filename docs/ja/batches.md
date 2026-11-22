[English](../batches.md) | **日本語** · [← README](../../README.ja.md)

# Message Batches API

## ライフサイクル

Anthropic Message Batches サーフェス（`/v1/messages/batches*`）も提供しているので、`client.messages.batches.create()` / `retrieve()` / `results()` / `cancel()` / `list()` / `delete()` がそのまま動く:

```python
batch = client.messages.batches.create(requests=[
    {"custom_id": "r1", "params": {"model": "claude-sonnet-5-5", "max_tokens": 64,
                                   "messages": [{"role": "user", "content": "hi"}]}},
    {"custom_id": "r2", "params": {...}},
])
# "ended" までポーリングし、client.messages.batches.results(batch.id) をイテレート
```

各 `custom_id` は**通常の pending** になる（snapshot に `batch_id` / `custom_id` が追加で載る）。responder は同じ `/_control/respond` / `auto` / `error` で注入する — `pending_id` 指定のほか、`custom_id` 指定（同じ custom_id が複数バッチで未解決なら `batch_id` も併記）でも届く:

```bash
# r1 を成功、r2 をエラーに（custom_id 指定）
curl -s -X POST localhost:8765/_control/respond \
  -d '{"custom_id": "r1", "content": [{"type": "text", "text": "batch reply"}]}'
curl -s -X POST localhost:8765/_control/error \
  -d '{"custom_id": "r2", "status": 500, "type": "api_error", "message": "boom"}'
```

全 custom_id に結果が揃うとバッチは自動的に `ended` へ遷移する。ライフサイクルは制御 API からも操作できる:

```bash
curl -s localhost:8765/_control/batches            # レジストリ: 状態 / counts / 未解決 custom_id

# respond/error では表現できない result type（canceled | expired）を個別注入
curl -s -X POST localhost:8765/_control/batch/result \
  -d '{"custom_id": "r2", "type": "expired"}'

# 今すぐ強制 "ended"。未解決の custom_id は expired（または canceled）になる
curl -s -X POST localhost:8765/_control/batch/end \
  -d '{"batch_id": "msgbatch_...", "unresolved": "expired"}'
```

実 API との意図的な差分（忠実さより決定性）:

- **時計による自動 expire はしない** — `expires_at`（作成 + 24h）は返すが、期限切れは `/_control/batch/end` / `/_control/batch/result` からの注入でのみ発生する。
- **cancel は基本的に即時** — `POST .../cancel` は未解決の custom_id を全て `canceled` にし、通常はその場で `ended` のバッチを返す（実 API の非同期 `canceling` フェーズを省略）。その瞬間すでに注入が進行中（in-flight）だったエントリは、破棄されずに succeeded/errored として完了する（実 API でも処理中リクエストは cancel 後に完了しうる）。それが残っている間は `canceling` を返し、着地後に `ended` へ遷移する。
- **コストには実 API 同様の 50% バッチ割引を適用** — history エントリに `"batch": true` と `cost.batch_discount = 0.5` が付き、`/_control/stats` は割引後の値を集計する。`canceled` / `expired` のエントリは history に記録しない（実 API 同様、課金対象外）。
- 各リクエストの `params` は浅い検証のみ（params がオブジェクトであること。`stream: true`・`speed`（fast mode）・`max_tokens: 0` は実 API 同様に作成時点で拒否。`fallbacks` を含む項目は受理したうえでその項目だけ `errored` 結果になり、pending にはならない — これも実 API と同じ）。ただし外側の形式は実 API と同じ厳しさで検証する（`custom_id` は `^[a-zA-Z0-9_-]{1,64}$` かつ一意、リクエストは 100,000 件まで、list の `limit` は `[1, 1000]`、カーソルも検証）— 本番なら弾かれるアプリがここでは通ってしまう、という事態を防ぐため。浅い検証を通過しても処理段階で失敗する params（例: `messages` が list でない）は、作成全体をロールバックして 400 を返す — バッチも pending も history も残らない。
- `results_url` は受信リクエストの Host から組み立てる。リバースプロキシ越しで使う場合は uvicorn を `--proxy-headers`（+ 適切な `FORWARDED_ALLOW_IPS`）付きで起動すること。
