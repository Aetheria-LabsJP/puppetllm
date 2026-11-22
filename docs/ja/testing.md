[English](../testing.md) | **日本語** · [← README](../../README.ja.md)

# テストガイド

## シナリオルール・タイムアウト・レイテンシ・レートリミット・擬似クロック

[クイックスタート](quickstart.md)の流れには responder が必要だった。ハーネスは台本からリクエストに**自動で**応答するので、何も接続しない CI でも同じサーバが動く — history / stats / cache は人間の responder のときと同じに動く。

**ルール**はリクエストにマッチして応答する。各ルールは順序付きの steps を持ち、マッチしたリクエスト 1 件につき 1 step を消費する（`repeat: true` なら最後の step を繰り返す。使い切ったルールはマッチしなくなり、次のルール — または未マッチ時ポリシー — に譲る）。step は `/_control/respond` のボディ（`{"text": ...}` の短縮形も可）か `/_control/error` のボディで、任意でレイテンシのキーを添える:

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

マッチのキー（すべて任意、指定したものはすべて成立が必要）: `provider`（`anthropic` / `bedrock` / `openai`）、`model`（glob）、`tools`（列挙したツール名がすべてリクエストに含まれる）、`has_tool_result`（直近の user ターンに `tool_result` ブロックがある）、`last_user_text`（直近の user ターンのテキストに対する正規表現。`tool_result` のテキストも含む）、`turn`、`stream`。ルールは順に試され、最初に成立した生きているルールが答える。`GET /_control/rules` はルールごとの `matched` / `remaining`、`unconsumed`（誰も要求しなかった step が残っているルール）、`all_consumed` — 「台本の呼び出しがすべて起きた」というテストの assert — を返す。`POST` は追加、`PUT` はリスト置換、`PUT /_control/rules/{id}` は 1 件を同じ位置で置換、`DELETE /_control/rules[/{id}]` は削除。ハーネスが答えた history エントリには `harness: {"source": "rule", "rule_id": ...}` が付く（他の出どころは `default` / `unmatched` / `rate_limit` / `timeout`）。step が失うことになるブロック型は rules の応答の `dropped` に並ぶ（`config.strict_blocks` なら 400 で拒否）。

**未マッチ時ポリシーと pending タイムアウト**（`/_control/config`。`/_control/clear` を跨いで保持される）: `on_unmatched` は `pending`（対話用の既定。responder を待つ）、`default`（`default_response` — `/_control/respond` のボディか文字列 — で答える）、`error`（`unmatched_error` — `/_control/error` のボディ。既定は 500 `api_error` — で答える）。`pending_timeout_s` を設定すると、誰も答えなかった pending にサーバが `timeout_error`（既定 504 `api_error`。バッチのエントリは対象外）で答える。期限はリクエスト到着時に確定し（後から設定を変えても新しいリクエストにだけ効く）、`/_control/pending` に出る（`deadline`、`timeout_in_seconds`）。pending 中に切断したクライアントの pending は捨てられるので、誰も待っていないリクエストを responder が見ることはない。注入済みの応答（relay が実課金している可能性がある）は、`delay_ms` の途中でクライアントが切断しても記録される。

```bash
curl -s -X POST localhost:8765/_control/config -d '{"pending_timeout_s": 30,
  "on_unmatched": "default", "default_response": "canned answer",
  "latency": {"delay_ms": 100, "jitter_ms": 50}, "rate_limit": {"rpm": 60}, "seed": 1}'
```

**レイテンシ**: `delay_ms`（応答前）、`ttfb_ms`（ストリームの最初のフレーム前）、`chunk_delay_ms`（フレーム間）、`jitter_ms`（`delay_ms` に加える 0..N ms の乱数。`seed` で初期化した生成器から引くので再現可能）を、どの注入（`/_control/respond` / `auto` / `error`）にも、どのルール step にも、`config.latency` の既定値としても指定できる。`delay_ms` の待機中の `/_control/clear` が優先される（リクエストは「cleared」のエラーで返り、記録されない）。始まってしまったストリームは `ttfb_ms` / `chunk_delay_ms` のペースで最後まで流れる。

**レートリミット**: `config.rate_limit` = `{"rpm", "itpm", "otpm"}`（任意のサブセット）を 60 秒のスライディング窓で数える。予算を超えたリクエストは、ルールを見る前に `retry-after` とベンダごとのクォータヘッダ（Anthropic 経路は `anthropic-ratelimit-*`、OpenAI 経路は `x-ratelimit-*`、Bedrock は `ThrottlingException`）付きの 429 になる。拒否したリクエストは予算を消費しない（ただし turn 番号は消費し、history にはエラーとして残る）ので、予算に収まるリクエストなら、他のトラフィックがない限り `retry-after` 秒待てば通る。クォータヘッダには設定した次元ごとの残量と回復時刻が入る（Anthropic には合算の `anthropic-ratelimit-tokens-*` も付く）。入力トークンはキャッシュ判定の前に、キャッシュ済み prefix も含めて数える。バッチのエントリは絞らず、計上もしない。現在の窓は `GET /_control/config` に出る。

**既定ヘッダ**: `config.default_headers`（`{"anthropic-ratelimit-requests-remaining": "42", ...}`）は、すべての API 応答（200 もストリームも S3 エミュレーションも。制御 API とドキュメントは除く）に付く。経路自身が同じヘッダを設定していればそちらが優先（リミッタの 429 は自分の値を保つ）。成功応答からベンダのクォータヘッダを読むアプリの検証用。

**擬似クロック**: `POST /_control/clock/advance {"seconds": N}` でサーバの時計を進める — 擬似プロンプトキャッシュの TTL、レートリミット窓、pending の期限が sleep なしで経過する（`GET /_control/clock` でオフセットを確認。`/_control/clear` で戻る）。

**トークン数とモデルカタログ**（pending は作られず、何も記録されない）:

| Method | Path | 説明 |
|---|---|---|
| POST | `/v1/messages/count_tokens` | Anthropic の `count_tokens`: `{"input_tokens": N}`。`/v1/messages` がそのリクエスト全体に課金する概算（`usage.input_tokens` + `cache_creation_input_tokens` + `cache_read_input_tokens`）と同じ値 |
| POST | `/model/{modelId}/count-tokens` | Bedrock の `CountTokens`（`{"input": {"invokeModel": {"body": <blob>}}}` — Messages リクエスト。boto3 が送るとおりワイヤ上は base64。生の JSON も受け付ける — または `{"input": {"converse": {...}}}` → `{"inputTokens": N}`） |
| GET  | `/v1/models`、`/v1/models/{id}` | カタログ（`puppetllm/catalog.py`）。Anthropic 形式（`x-api-key` か `anthropic-version` がある、または認可ヘッダなし。各モデルに `max_input_tokens`、`max_tokens`、`line`、`lifecycle` / `deprecated_at` / `retires_at`、`capabilities` を付ける。`limit` / `after_id` / `before_id` でページング。`lifecycle`（複数指定可。`lifecycle[]` とも書ける）は `active` / `deprecated` / `retired` で絞り込み、既定では廃止済みのモデルを含めない）か OpenAI 形式（素の `Authorization: Bearer`）。`lifecycle` フィールドはプロバイダが公表した状態で、カタログに固定で持つ（`retires_at` を過ぎても廃止済みにはならない）。廃止済みのモデルは `capabilities` なしで載る。`created_at` / `created` は全モデル共通の固定値で、公開日ではない。情報提供用: リクエスト経路はどんなモデル ID でも受け付け、未掲載の ID は拒否せず（`active`、上限と capabilities なしで）合成して返す |

**Python のテストから** — `puppetllm.testing` が制御 API を包む。`serve()` は空きポートでサーバをインプロセス起動する:

```python
import anthropic
from puppetllm.testing import serve

with serve() as puppet:                       # 起動済みサーバには Puppet("http://127.0.0.1:8765")
    puppet.expect(tools=["get_weather"], has_tool_result=False).respond(
        content=[{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Tokyo"}}])
    puppet.expect(has_tool_result=True).error(429, headers={"retry-after": "0"}).respond(text="sunny")
    client = anthropic.Anthropic(base_url=puppet.url, api_key="test", max_retries=0)
    ...テスト対象を実行...
    puppet.assert_consumed()                  # 台本の step がすべて使われた
    puppet.assert_sent(model="claude-*", contains="Tokyo", tool="get_weather")
    puppet.assert_no_pending()
```

`Puppet` は対話用の操作（`wait_pending`、`respond`、`error`、`pending`（`summary=` / `fields=`）、`history`（フィルタ。カーソル付きの envelope は `history_page`、1 件は `history_entry`）、`metrics`、`stats`、`cache`、`config`、`advance_clock`、`clear`。`clear(config=True)` は設定も既定に戻し、`reset(baseline)` は保存した `baseline()` — 設定とルール — を再適用する）も持つ。`pip install .` すると pytest の `puppet` フィクスチャ（セッションに 1 サーバ。テストの前後で状態を消し、設定とルールはサーバ起動時のものに戻す — `PUPPETLLM_*` の設定や `--config` のルールは生きたまま、あるテストの `puppet.config(...)` や `puppet.expect(...)` が次のテストに漏れない。`PUPPETLLM_URL` で外部サーバに向けられる）が自動登録される。`pip install ".[test]"` で SDK も入る。起動時の台本はファイル — `puppetllm serve --config scenario.json`（`{"config": {...}, "rules": [...], "prompts"?: {...}}`）— でも、Docker 向けには環境変数（[環境変数](control-api.md#環境変数)参照）でも渡せる。

`Expectation.respond` は `choices`（OpenAI の choice ごとの応答）、`logprobs`、`provider_response`（Bedrock の `trace`）も受け取り、`Puppet.prompts(templates, replace=False)` はプロンプト管理のフィクスチャを登録する（`baseline()` / `reset()` の対象）。[制御 API リファレンス](control-api.md) を参照。
