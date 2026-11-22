[English](../control-api.md) | **日本語** · [← README](../../README.ja.md)

# 制御 API リファレンス

## エンドポイント

| Method | Path | 説明 |
|---|---|---|
| GET  | `/_control/health` | ヘルスチェック（`{"ok","turn_count"}`） |
| GET  | `/_control/pending` | 保留中リクエスト一覧（`pending[]` + provider 込み、最古は `request` にも。pending タイムアウト設定時は `deadline` / `timeout_in_seconds` も）。`?summary=1` で各 `request` を短い要約（turn、model、`message_count`、`tools`、`last_user_text`、`has_tool_result`、各種 id）に、`?fields=model,turn` で指定した snapshot キーだけに絞る |
| GET  | `/_control/wait_for_pending?timeout=N` | 次の pending を long-poll で待つ（既定 270s / 上限 600s。なければ `{"timeout":true}`） |
| POST | `/_control/respond` | 保留中リクエストに応答（`{"content":[...], "pending_id"?, "stop_reason"?, "stop_sequence"?, "stop_details"?, "usage"?, "provenance"?}`、または text ブロック 1 つの短縮形 `{"text": "..."}`）を注入。`{"responses": [<対象付きのボディ>, ...]}` で複数を一括注入（全件を検証・解決してから適用。その間に消えた対象があれば項目ごとの `results` 付きの 409）。`content` のブロックは `text` / `tool_use` / `thinking` / `redacted_thinking` とサーバー側ツールのブロック（[応答を供給する](quickstart.md#3-応答を供給するresponder)）。他の型は捨てて報告（`dropped`）、`strict_blocks` なら拒否。`stop_reason` で自動判定を上書き（例 `"max_tokens"` — 打ち切り分岐のテスト用。OpenAI 経路では `finish_reason: "length"` に変換）。`"refusal"` なら `stop_details` を生成（`stop_details` で `category` / `explanation` を指定可。`recommended_model` などの追加フィールドは素通し）し、OpenAI 経路では OpenAI の refusal 形式（`message.refusal` / `delta.refusal`、`content: null`、`finish_reason: "stop"`）になる。`"stop_sequence"` なら `stop_sequence` を埋める。`usage` で概算トークンを実数値に上書き（`input_tokens` / `output_tokens` / `cache_creation_input_tokens` / `cache_read_input_tokens` の非空サブセット、`[0, 1e12]` の int。加えて `cache_creation` / `output_tokens_details` / `server_tool_use` のオブジェクトと `service_tier` / `inference_geo` / `speed` の文字列も任意 — relay モードが使用。`provenance`（`{upstream_provider, upstream_model, usage_source, pricing_model, translation.dropped_fields}`、[コスト / トークン / キャッシュを観測する](#コスト--トークン--キャッシュを観測する)）はエントリに記録され、その `pricing_model` が料金表を選ぶ。OpenAI 向けの追加: `choices`（`content` / `text` の代わりに、`n > 1` のリクエストの choice ごとに `text` / `content` / `stop_reason` / `logprobs` の本体を並べる。`n` に合わせて循環または切り詰める。usage は各応答の合計、history は `choices` に全部を残す）と `logprobs`（テキスト応答の `[{"token", "logprob", "top_logprobs"?}]`。リクエストが `logprobs` を求めたときに経路が作る決定的な値を置き換える）。Bedrock 向けの追加: `provider_response`（[Bedrock](bedrock.md#guardrail-と-trace-とプロンプト管理のフィクスチャ) 参照）） |
| POST | `/_control/respond_all` | 同じ `respond` ボディ（対象指定なし）を生きている pending すべて（バッチのエントリ含む）に注入 — 並列ファンアウトを 1 回で答える（`{"ok", "count", "pending_ids"}`。pending がなければ 400） |
| POST | `/_control/auto` | `{"text": "..."}` を受ける `/_control/respond` の非推奨エイリアス（対象指定は同じ）。引き続き動く |
| GET  | `/_control/capabilities` | 互換性マトリクス（[互換性マトリクス](design.md#互換性マトリクス)参照）: 各 surface の accepted / injectable / streamed / relayed、content ブロック型ごとの経路での扱い、意図的に未実装の操作 |
| POST | `/_control/config`（PUT も）/ GET | ハーネス設定（[テストガイド](testing.md)）: `pending_timeout_s`、`on_unmatched`、`default_response`、`unmatched_error`、`timeout_error`、`latency`、`rate_limit`、`seed`、`strict_blocks`、`default_headers`。部分更新。`null` で既定値に戻る。`clear` を跨いで保持 |
| GET / PUT / POST / DELETE | `/_control/prompts` | プロンプト管理のフィクスチャ（`{"prompts": {"arn:aws:bedrock:...:prompt/ID": {"system"?, "messages"?, "tools"?}}}`）: PUT で置換、POST で追加、DELETE で全消去。`--config` ファイルの `prompts` キーでも渡せる（[Bedrock](bedrock.md#guardrail-と-trace-とプロンプト管理のフィクスチャ) 参照） |
| GET / PUT / POST / DELETE | `/_control/rules` | シナリオルール（[テストガイド](testing.md)）: カウンタ付き一覧（`unconsumed`、`all_consumed`）、置換、追加、全削除。`PUT` / `DELETE /_control/rules/{id}` は 1 件を操作（同じ match のまま step を末尾に足すだけの `PUT` は消費カウンタを引き継ぐ） |
| GET | `/_control/clock` | 擬似クロック（`now`、`clock_offset_seconds`） |
| POST | `/_control/clock/advance` | `{"seconds": N}` — 擬似クロックを進める（キャッシュ TTL、レートリミット窓、pending の期限） |
| POST | `/_control/error` | HTTP エラー応答を注入（`{"status","type","message", "code"?, "param"?, "headers"?, "pending_id"?, "after_events"? / "after_blocks"?, "content"?, "provenance"?}` — 後ろ 2 つはストリーミングを途中で失敗させる。[エラー応答を注入する](quickstart.md#4-エラー応答を注入してハンドリングを試す) 参照）。`headers`（文字列 → 文字列/数値）はエラー応答にそのまま付与される — 429 の `{"retry-after": 3}` や `anthropic-ratelimit-*` / `x-ratelimit-*` など、アプリのバックオフ処理の検証用（`content-length` / `transfer-encoding` などのフレーミング系ヘッダ、制御文字、非 Latin-1 の値は 400 で拒否）。Anthropic 経路のエラーボディには（Batches も含め）`request-id` ヘッダと一致する `request_id` が入り、OpenAI 経路は Anthropic 語彙の `type` を自分の語彙に変換する（`api_error` → status に応じて `server_error` 等）。`provenance` は respond と同じく記録される |
| GET  | `/_control/history` | (request, response, usage, cost, cache) 履歴（完了順）。クエリ: `cursor`（その `seq` より後に記録されたもの）、`since_turn`（その turn より後）、`provider`、`model`（glob）、`limit`（1〜1000。応答に `has_more` と `next_cursor` — 続きを取る `seq`、新しい記録がなければ渡した cursor がそのまま返る。並列リクエストが turn 順と違う順で完了してもページングと tail で取りこぼさない）、`format=jsonl`（1 行 1 エントリ。`has_more` / `next_cursor` は応答ヘッダ `x-puppetllm-has-more` / `x-puppetllm-next-cursor` に移る）。エントリには `seq`、`request_id`、`path`、`latency_ms`、`usage_source`、`pricing_model`、`provenance`（[コスト / トークン / キャッシュを観測する](#コスト--トークン--キャッシュを観測する)） |
| GET  | `/_control/history/{turn}` | 1 エントリ（未知なら 404、整数でなければ 400） |
| GET  | `/_control/metrics` | Prometheus テキスト: 累積カウンタ（`clear` を跨いで保持）と現在値のゲージ（[コスト / トークン / キャッシュを観測する](#コスト--トークン--キャッシュを観測する)） |
| GET  | `/_control/stats` | コスト目安・トークン・キャッシュの累計サマリ |
| GET  | `/_control/cache` | 擬似プロンプトキャッシュ index |
| POST | `/_control/clear` | pending / history / cache / batches / Bedrock バッチジョブ / ルール / プロンプトのフィクスチャ / レートリミット窓 / クロックのオフセットを空に — ボディ `{"config": true}` なら設定も既定に戻す（in-flight はリトライ可能なエラーで解放: Anthropic 経路は 529 `overloaded_error`、Bedrock / OpenAI は 503、作成中の Bedrock バッチジョブはリトライ不能な 400 `ConflictException`。S3 ストアには触れない） |
| GET  | `/_control/batches` | バッチレジストリ（状態・request_counts・未解決 custom_id） |
| POST | `/_control/batch/result` | 1 つの custom_id に `canceled` / `expired` を注入（`{"custom_id","type","batch_id"?}`） |
| POST | `/_control/batch/end` | バッチを強制 `ended` に。未解決 custom_id は `expired`（既定）または `canceled` になる |
| GET  | `/_control/bedrock_jobs` | Bedrock バッチ推論のジョブレジストリ（状態・レコード数・未解決 `recordId`） |

`respond` / `auto` / `error` では、バッチのエントリを `pending_id` の代わりに `custom_id`（+ 任意で `batch_id`）で指定でき、レイテンシのキー `delay_ms` / `ttfb_ms` / `chunk_delay_ms` / `jitter_ms`（[テストガイド](testing.md)）でその 1 回の応答のタイミングを決められる。

ボディを取る `/_control/*` エンドポイントのリクエストボディは型付き（`puppetllm/control_models.py`）で、`/openapi.json` の `components.schemas` に載る（`/docs` でも見える）ので、スキーマからクライアントを生成できる。`/_control/respond`（とルールの step）は `{"text": "..."}` を text ブロック 1 つの短縮形として受け付け、空の `content` が並んでいても有効。未設定の任意フィールドを `null` で送っても未指定として扱う。ただし `/_control/config` だけは `null` を「既定値に戻す」と解釈するので、未設定フィールドは省いて送ること。

以前のバージョンから変わった観測可能な挙動（旧値を前提にしたアプリやテストハーネスは更新が必要）: `stop_reason` の `guardrail_intervened` は、Anthropic 形式のメッセージでは `end_turn`、OpenAI では `finish_reason: "content_filter"` で終わる（以前はそのまま / `stop`）。`logprobs: true` の Chat Completions リクエストには対数確率が返り（以前は `null`）、bool でない `logprobs` や、それなしの・0〜20 の範囲外の `top_logprobs` は 400。fast mode と US 推論のコスト倍率は Claude の ID にだけ掛かる（OpenAI のファミリ、独自の料金表、relay が上流基準で計算するモデルには掛からない。Anthropic 形式の経路に来た架空の ID には掛かる）。処理中に clear されたリクエストは Anthropic / Bedrock-Messages 経路で `529 overloaded_error`（旧 `503 api_error`）。OpenAI 経路のエラー `type` は OpenAI 語彙（`server_error`、`service_unavailable_error` 等。旧 `api_error` / `service_unavailable`）。canonical の `refusal` は OpenAI の `message.refusal` + `finish_reason: "stop"` になる（旧 `finish_reason: "content_filter"`。フィルタ形式が欲しければ `stop_reason` に `"content_filter"` を渡す）。Bedrock の pending / 応答は正規化した Anthropic モデル名を持つ。`thinking` ブロックは捨てずに保持する。OpenAI の usage は `n` 個の choice すべてを計上し（応答・history・stats で一致。該当する pending の snapshot には `choices` フィールドが載る）、明示された `service_tier` を返す。`inference_geo: "us"` のリクエストには公式の 1.1 倍を課金に適用する。`/_control/auto` は `text` が文字列であることを要求する（他のキーは従来どおり無視）。また全リクエスト経路で、形の崩れた `messages` / `system` / `tools` は後段で失敗する代わりに 400 で拒否する。`/_control/respond` / `auto` / `error` は、注入したブロック型を捨てた場合に素の `{"ok": true}` ではなく `{"ok": true, "dropped": [...]}` を返す。サーバー側ツールのブロックは捨てずに保持する。`/v1`、`/anthropic`、`/model`、`/guardrail`、`/async-invoke`、`/chat` 配下の未実装パスは S3 のエラーではなくプロバイダ形式の 404 を返す。そのため `chat`、`models`、`guardrail`、`async-invoke` は S3 のバケット名として使えなくなった。実在パスへの誤ったメソッドは 405 で、`Allow` には実際に提供するメソッドだけを載せる（`HEAD` は含まない）。コストに `long_context` が付き、モデルのロングコンテキスト料金の閾値を超えたプロンプトは高い料金表で計算する（以前は料金表がモデルごとに 1 種類だった）。料金は現行の表に従い、以前は汎用の行に当たっていた ID の値が変わる（Fable / Mythos 5.2 $8/$40、Opus 5.5 $4/$20、Haiku 5.5 $0.10/$0.50、Sonnet 5.5 の cache read $0.10。`gpt-6-sol` / `gpt-6-luna` は専用の行）。最小キャッシュ長も変わる（Sonnet 5.5 と Haiku 5.5 は 512、Mythos Preview は 2048）。未知の `gpt-*` は gpt-6.2-sol 価格で計算する（以前は gpt-5.4。`gpt-6` を含む ID はすべて $10/$50 だった）。Grok の ID は xAI の行で計算する（grok-4.5 / 4.3 / 4.20 / grok-build は専用の行、それ以外の Grok は grok-4.7 価格。以前はすべて Sonnet）。GPT-6.x / GPT-5.6 のキャッシュ書き込みには割増が掛かる。`cache_savings_usd` には fast mode と US 推論の倍率が掛かる。Anthropic 形式の `/v1/models` は Models API の全フィールドを返し、Claude Opus 4.5 と Sonnet 4.5 も Haiku 4.5 と同じく日付つきの ID で載る（`/v1/models/{id}` は日付なしの ID も引けて、日付つきの ID で返す。カーソルには掲載されている ID を使う）。廃止済みのモデルは `lifecycle=retired` を指定しない限り一覧に出ず、掲載する ID は現行のモデル構成に合わせて変わった。

## コスト / トークン / キャッシュを観測する

```bash
# 累計サマリ (全て概算)
curl -s localhost:8765/_control/stats | jq
# → {"is_estimate":true,"completed_requests":3,"error_requests":0,
#     "totals":{"input_tokens":..,"output_tokens":..,
#               "cache_read_input_tokens":..,"total_usd":..,"cache_savings_usd":..},
#     "cache":{"hits":2,"misses":1,"hit_rate":0.6667,"index_size":2},
#     "by_model":{"claude-sonnet-5-5":{"requests":3,"total_usd":..}}}

# 擬似プロンプトキャッシュの index (prefix hash 別 hit/miss)
curl -s localhost:8765/_control/cache | jq

# 1 リクエストずつの (request, response, usage, cost, cache) 履歴
curl -s localhost:8765/_control/history | jq '.history[-1]'
# 絞り込みとページング: turn 40 以降、プロバイダ指定、モデルの glob、20 件ずつ
curl -s 'localhost:8765/_control/history?since_turn=40&provider=anthropic&model=claude-*&limit=20' | jq
curl -s 'localhost:8765/_control/history?limit=20&cursor=60' | jq '.next_cursor, .has_more'
curl -s 'localhost:8765/_control/history?format=jsonl' > history.jsonl   # 1 行 1 エントリ
curl -s localhost:8765/_control/history/12 | jq                            # 1 ターン分
# 軽いポーリング: pending ごとの短い要約（会話本文なし）
curl -s 'localhost:8765/_control/pending?summary=1' | jq '.pending[].request'
# Prometheus のカウンタとゲージ
curl -s localhost:8765/_control/metrics

# テスト間のクリーンアップ (pending / history / cache を全消去)
curl -s -X POST localhost:8765/_control/clear
```

`cache_savings_usd` は「キャッシュが効いた分、本物なら浮いたであろう概算額」。アプリが `cache_control` を正しい構造で投げられているかの検証に使う。（Anthropic / Bedrock 経路のみ — OpenAI 経路は常に cache status `"none"` で hit/miss カウンタも汚さない。）

擬似キャッシュが再現する挙動（公式の prompt-caching ドキュメントに準拠）:

- **配置**: ブロック単位の `cache_control`（最大 4 つ）と、**トップレベル** `cache_control`（自動キャッシュ: 最後のキャッシュ可能ブロックに 1 つ置く。`thinking` と空テキストは飛ばす）。前方一致の順序は `tools → system → messages`、マーカー自体はキーに含まない。
- **最小キャッシュ長は世代別**: Fable / Mythos 5〜5.2、Opus 5 / 5.5、Sonnet 5.5、Haiku 5.5 = 512 トークン、Opus 4.8 = 1024、Opus 4.7 と Mythos Preview = 2048、Opus 4.6 / 4.5 = 4096、Opus 4.1 / 4 = 1024、Sonnet 5 以前 = 1024、Haiku 4.5 = 4096、Haiku 3.5 = 2048。未満なら実 API 同様にエラーなしで `"none"` と観測する。
- **TTL**: `{"type":"ephemeral"}` は 5 分、`{"type":"ephemeral","ttl":"1h"}` は 1 時間（テスト用に短縮するなら `PUPPETLLM_CACHE_TTL` / `PUPPETLLM_CACHE_TTL_1H`。read で延長）。書き込みは TTL 別に `usage.cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` に分けて報告し、単価は入力の 1.25x / 2x。
- **無効化**: `output_config.effort`、`thinking`、`tool_choice` をターン間で変えると *messages* 側のプレフィックスが無効になる。`speed`（fast mode）の切り替えは system + messages を無効化し tools は生きる。これは公式の無効化表の意図的な簡略化で、公式表は thinking / effort について tools / system キャッシュを「モデル依存」（設定を system プロンプトより前に描画するモデルがある）としているが、puppetllm は messages のみとして扱う。明示の既定値は省略と同じ扱い: `effort: high`、`tool_choice: auto`（`disable_parallel_tool_use: false` 付きも同様）、モデルの既定 thinking モード（Opus 5 / Sonnet 5 / Fable / Mythos は `adaptive`、それ以外は `disabled`。`display: omitted`）。モデル変更は全無効。
- **検証**: 実 API が拒否する配置はここでも `400 invalid_request_error`（pending は作られない）: 不正なマーカーや未知の `ttl`、明示ブレークポイント 5 個以上、`thinking` ブロック上のマーカー、5m ブレークポイントの後の 1h ブレークポイント、明示 4 個があるところへのトップレベル `cache_control`、最後のブロックの明示マーカーと食い違うトップレベル `ttl`。
- **fast mode**: `speed: "fast"` は公式の 2 倍で課金し（Anthropic / Bedrock 経路のみ）、`usage.speed: "fast"` を返し、Anthropic 上流へはアプリの `anthropic-beta` ヘッダごと転送する（OpenAI 互換上流へは警告を出して落とす）。モデル / プラットフォームの対応可否は検証しない。
- **lookback**: 各ブレークポイントは最大 20 ポジション遡る。連続する `tool_use`（または `tool_result`）ブロックの並びは 1 ポジションと数える。
- **usage の形**: 2026 年の usage オブジェクト（`cache_creation`、`output_tokens_details.thinking_tokens`、`server_tool_use`、`service_tier`（`standard` / `batch`）、`inference_geo`、fast mode 要求時は `speed`）を返し、ストリームの `message_delta.usage` には実 API と同様に累積の input / cache 各値も載る。

料金は Anthropic / OpenAI / xAI の公式の世代別表に従う（例: Fable 5.2 $8/$40・cache read $0.20、Opus 5.5 $4/$20、Sonnet 5.5 $2/$10、Haiku 5.5 $0.10/$0.50、Opus 5 $5/$25、Sonnet 4.6 $3/$15、gpt-6.2-sol $2/$10、grok-4.7 $2/$6）。ロングコンテキスト料金のあるモデルは、プロンプト（全入力トークン: キャッシュされない分、キャッシュ書き込み、キャッシュ読み出し）が閾値を超えるとリクエスト全体を高い料金表で計算し、コストに `long_context: true` を付ける — Haiku 5.5 は 100,000 トークン超、GPT-6.x / GPT-5.6 / GPT-5.5 / GPT-5.4（pro を含む。GPT-5.4 mini / nano と、単一の codex 行を使う codex の ID は対象外）は 272K 超、Grok は 200K 以上。stats の `cache_savings_usd` も同じ料金表で計算する。OpenAI のページは 272K の閾値を示すだけで、リクエスト全体がロングコンテキスト料金になるとは書いていない。ここではそう仮定している。GPT-6.x と GPT-5.6 はキャッシュ書き込みに料金が掛かる（30 分保持は入力の 1.25 倍、12 時間保持は 2.5 倍）。relay が呼ぶ Chat Completions では書き込みが 30 分保持なので、上流が報告したキャッシュ書き込みはその料金で計算する。未知の Claude ID は Sonnet 4.x 価格、未知の `gpt-*`（一覧にない GPT-6 の ID を含む）は gpt-6.2-sol 価格、専用の行がない Grok の ID は grok-4.7 の行にフォールバックする。

history の各エントリには、そのやり取りの相関情報と来歴が入る: `seq`（完了順の記録番号。history のカーソル）、`request_id`（応答ヘッダに載せた id。`request-id` / `x-amzn-requestid` / `x-request-id`。batch のエントリでは null で、代わりに `batch_id` / `custom_id` や `job_id` / `record_id` を持つ）、`path`、`latency_ms`（到着から応答までをサーバの時計で）、`usage_source`（概算トークナイザなら `estimate`、responder が実数を渡したなら `override`、エラーなら null）、`pricing_model`（コスト計算に使った料金表のモデル）、relay が答えた場合は `provenance`（`upstream_provider`、`upstream_model`、`translation.dropped_fields`）。snapshot にはリクエストヘッダの allow-list — `user-agent`、`anthropic-version`、OpenAI の organization / project id、W3C / B3 / ベンダの trace id（`traceparent`、`x-cloud-trace-context`、`x-datadog-trace-id`、`sentry-trace` など。key=value の列を運ぶ `tracestate` と `baggage` は除く）、`x-request-id` などの一般的な request id、botocore の `amz-sdk-invocation-id`、`x-stainless-*` — だけが残り、それ以外は残らない（クレデンシャル、セッション、クライアントのアドレスは history に入らない）。`cache` は hit でなかった理由を説明する: `reason` は `no_cache_control`、`below_minimum`（`minimum_tokens` と、最も深いマーカーまでのサイズ `eligible_prefix_tokens` 付き）、`first_seen`、`prefix_changed`（比較対象と `first_changed_path` から先が違う — `system[0]` や `tools[1]` のような静的プレフィックスの変更、または `messages[3].content[1]` のようなターンの編集。`matched_through` = 先頭から変わっていなかったセグメント数。同じモデルの直前のリクエストと tools / system がそのまま共通でそこから先が別の会話は `first_seen`）、`parameters_changed`（内容は同じで、プロンプトに描画されるパラメータ `params.speed` や `params.thinking` などが違う）、`ttl_expired`、`lookback`、疑似キャッシュが見なかったリクエストなら `disabled` / `throttled`。リクエストが `diagnostics` オブジェクト（cache-diagnosis beta のリクエストの形）を送っていれば、同じ説明を Message の `diagnostics.cache_miss_reason`（加えて `cache_status`、`matched_through`、`first_changed_path`、`minimum_tokens`、`eligible_prefix_tokens`。本物の beta ではなく puppetllm 独自のフィールド）として非ストリームの本文と `message_start` に返す（Anthropic 経路と Bedrock の Mantle エイリアス）。比較対象は、`diagnostics.previous_message_id` がこのサーバの答えた message なら その message のリクエスト、そうでなければ同じモデルの直前のリクエスト。

`PUPPETLLM_HISTORY_FILE`（または `serve --history-file`）は記録のたびにエントリを JSONL ファイルへ追記する（ローカルファイルへの同期 append で history と同じ順。パスは起動時に確認 — `--history-file` は使えないパスを拒否し、環境変数は警告のみ — し、書き込みエラーは 1 回だけ報告してリクエストは失敗させない。JSON としては正しいが UTF-8 にならないクライアントのテキスト — lone surrogate など — はここでも全ての JSON 応答でもエスケープして、失敗させない）— `/_control/clear` に消されない永続ログ。メトリクスと同様に追記のみなので、ロールバックされた batch create の先行した応答も残る。`--log-format json`（または `PUPPETLLM_LOG_FORMAT=json`）にすると stderr がライフサイクルイベント 1 件 1 行の JSON になる — `received`、`matched`（ルール / レートリミット / 既定ポリシー / unmatched エラー / タイムアウト）、`resolved`（結果、レイテンシ、コスト）、`rejected`（リクエスト経路で pending になる前に返した 400。`count_tokens` と batch のエンベロープ検査は数えない）、`canceled`（clear、切断、ハンドラのキャンセル、batch の cancel / expire / stop やロールバック、応答をエンコードできなかった内部エラー）— `pending_id`、`request_id`、batch の id で相関できる。既定の `text` はアダプタの受付行だけを出す。`GET /_control/metrics` は Prometheus のテキスト形式: `/_control/clear` を跨いで積み上がるカウンタ（`puppetllm_requests_total{provider,outcome}`、`puppetllm_tokens_total{provider,kind}`、`puppetllm_cost_usd_total{provider}`、`puppetllm_cache_lookups_total{status}`、`puppetllm_harness_answers_total{source}`、`puppetllm_rejected_total{provider}`、`puppetllm_canceled_total{reason}`、`puppetllm_latency_seconds_total{provider}` — レイテンシは `latency_ms` と同じくサーバの時計基準なので、待機中の `/_control/clock/advance` も加算される）と、現在値のゲージ（`puppetllm_pending`、`puppetllm_turns`、`puppetllm_rules_unconsumed`、`puppetllm_clock_offset_seconds`）。

料金は内蔵のファミリ表から引く。`PUPPETLLM_PRICING_FILE`（または `serve --pricing-file`）で独自の表を足せる — `{"<モデルの glob>": {"input": 0.5, "output": 2.0, "cache_write"?, "cache_read"?, "cache_write_1h"?}}`（100 万トークンあたりのドル）。内蔵ファミリより先に照合され、glob がコストの `model_family` になる。glob はファイルの順に、ID 全体に対して大文字小文字を無視して照合する（Bedrock / Vertex の ID は `*claude-*` で拾う）。料金は 1,000,000 までの JSON の数値で、`cache_read` は `input` 以下、他のキーは不可。空のオブジェクトは表を消す。独自の表は単一の料金表で、ロングコンテキスト料金は持てない。内蔵のロングコンテキスト料金があるモデルに一致する glob は、その料金ごと置き換える。fast mode と US 推論の倍率は Claude の ID（内蔵の Claude 表、Claude を名乗る未知の ID、Anthropic 形式の経路に来た架空の ID）にだけ掛かり、独自の表、OpenAI や Grok のファミリ、relay が上流基準で計算する一致なしのモデルには掛からない。どの表にも一致しない `pricing_model` は、未知の受信側 ID と同じく既定ファミリ（Sonnet、`gpt-*` なら gpt-6.2-sol）で計算される。

## 環境変数

| 変数 | 既定 | 説明 |
|---|---|---|
| `PUPPETLLM_CACHE_TTL` | `300` | 5 分ブレークポイントの擬似キャッシュ TTL（秒） |
| `PUPPETLLM_CACHE_TTL_1H` | `PUPPETLLM_CACHE_TTL` × 12 | `ttl: "1h"` ブレークポイントの TTL（秒）。未設定なら 3600 で、5m 側の TTL に連動する |
| `PUPPETLLM_S3_ROOT` | プロセスごとの一時ディレクトリ | S3 エミュレーションがオブジェクトを置く場所（バッチ推論の入出力）。再起動をまたいで残したい場合や、Docker でホストから見たい場合に指定する（その場合は compose の override に環境変数と対応するボリュームを足すこと。同梱の `docker-compose.yml` はどちらも定義していない） |
| `PUPPETLLM_CACHE_HONOR_TTL` | `1` | `0` で TTL を無視（常に生存） |
| `PUPPETLLM_CACHE_MIN_TOKENS` | （モデル別） | 最小キャッシュ閾値の上書き。`0` で無効（全 prefix キャッシュ）。未設定は世代別テーブル（Fable / Opus 5・5.5 / Sonnet 5.5 / Haiku 5.5 512、Opus 4.8 1024、Opus 4.7 2048、Opus 4.6 / 4.5 4096、その他の Sonnet 1024、Haiku 4.5 4096 …） |
| `PUPPETLLM_PENDING_TIMEOUT` | （なし） | 応答のない pending にタイムアウトエラーを返すまでの秒数（[テストガイド](testing.md)） |
| `PUPPETLLM_DEFAULT_RESPONSE` | （なし） | 未マッチのリクエストに返すテキスト、または JSON の `/_control/respond` ボディ。設定すると `on_unmatched: default` になる |
| `PUPPETLLM_ON_UNMATCHED` | `pending` | `pending` / `default` / `error` |
| `PUPPETLLM_SEED` | （なし） | レイテンシのジッタの seed |
| `PUPPETLLM_CONFIG` | （なし） | 起動時に読む JSON ファイル `{"config": {...}, "rules": [...], "prompts"?: {...}}` |
| `PUPPETLLM_URL` | （なし） | pytest の `puppet` フィクスチャを、起動する代わりに稼働中のサーバへ向ける |
| `PUPPETLLM_HISTORY_FILE` | （なし） | history の各エントリをこの JSONL ファイルに追記（[コスト / トークン / キャッシュを観測する](#コスト--トークン--キャッシュを観測する)） |
| `PUPPETLLM_LOG_FORMAT` | `text` | `json` で stderr を 1 行 1 イベントの JSON に（[コスト / トークン / キャッシュを観測する](#コスト--トークン--キャッシュを観測する)） |
| `PUPPETLLM_PRICING_FILE` | （なし） | 内蔵ファミリより先に照合する JSON 料金表（[コスト / トークン / キャッシュを観測する](#コスト--トークン--キャッシュを観測する)） |

## セキュリティ

- `/_control/*` は **認可なし**。誰でも履歴を読め、応答やエラーを注入できる。**public internet に晒さない**。
- 既定の listen は `127.0.0.1`（localhost のみ）。別ホストから使うのは LAN / VPN / Tailscale 等の **trusted network 内に限る**。
- `--host 0.0.0.0`（Docker は既定で `0.0.0.0` listen だが compose は `127.0.0.1:8765` に publish 制限）で公開する場合は firewall / network policy を必ず確認する。
- **イメージを `docker run` で直接起動する場合**: コンテナは `0.0.0.0` で listen する（ポート転送に必須）ので、公開ポートは localhost に束ねる — `docker run -p 127.0.0.1:8765:8765 puppetllm` — こと。`-p 8765:8765` だと無認可の制御面が全ホストインターフェースに露出する。付属の `docker compose` は既にこの形になっている。
- 同梱の **S3 エミュレーション**も同じ制御面の一部で、ポートに到達できるクライアントは
  認証なしでバケットを作り、オブジェクトを読み書き削除できる。`PUPPETLLM_S3_ROOT`
  （未設定ならプロセスごとの一時ディレクトリ）の外には決して出られないが、
  `PUPPETLLM_S3_ROOT` は使い捨てのディレクトリに向けること。
- あくまでローカルデバッグ用途。本番の前段に置くものではない。
