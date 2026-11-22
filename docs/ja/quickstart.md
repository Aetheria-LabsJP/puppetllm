[English](../quickstart.md) | **日本語** · [← README](../../README.ja.md)

# クイックスタート

## 3 つの役者

全体像は **3 つの登場人物**で考える:

```
  +--- アプリ / SDK ---+         +----- puppetllm ------+        +-- responder --+
  | messages.create()  | ------> | POST /v1/messages    | -----> | 応答を注入    |
  | (応答までブロック) | <------ | (pending として保留) | <----- | /_control/... |
  +--------------------+  応答   +----------------------+        +---------------+
        (1)アプリ                 (2)fake server (本体)        (3)供給側 (人 or AI)
```

(1) が投げたリクエストを (2) が**保留 (pending)** し、(3) が `/_control/*` で応答を流し込むと、(1) の `create()` がその応答で返る。実 API は一切叩かない。

## 1. proxy を起動する

```bash
# A) Docker (推奨)
docker compose up -d
curl localhost:8765/_control/health        # → {"ok":true,"turn_count":0}

# B) 直接 (Python 3.10+) — 起動バナーと共に foreground 実行
pip install -r requirements.txt
python3 -m puppetllm --host 127.0.0.1 --port 8765
#   (`pip install .` すれば `puppetllm` コマンドが使える: `puppetllm serve --port 8765`、
#    `puppetllm relay ...`、`puppetllm wait`（ヘルスチェックが通るまで待つ）、`puppetllm --version`。
#    `pip install ".[test]"` で anthropic / openai / boto3 の SDK も入る)
#   [puppetllm] starting on http://127.0.0.1:8765
#   [puppetllm] Anthropic: set ANTHROPIC_BASE_URL=http://127.0.0.1:8765
#   [puppetllm] Bedrock:   point AnthropicBedrock base_url to http://127.0.0.1:8765
#   [puppetllm] OpenAI:    set OPENAI_BASE_URL=http://127.0.0.1:8765/v1  (note the /v1)

# C) uvicorn 直叩き (reload 等のオプションを使いたいとき)
python3 -m uvicorn puppetllm.fake_server:app --host 127.0.0.1 --port 8765
```

`--host` 既定は `127.0.0.1`（localhost のみ）。LAN/VPN 越しに使うときだけ `0.0.0.0` にする（[セキュリティ](control-api.md#セキュリティ)参照）。

小さなプロンプトでキャッシュ観測を試すと、実モデルの最小キャッシュ prefix（512〜4096 トークン）に届かず `cache.status` が `none` のままになる。まずは `PUPPETLLM_CACHE_MIN_TOKENS=0` で全 prefix をキャッシュ対象にする（[環境変数](control-api.md#環境変数)参照）。

`serve` は [テストガイド](testing.md) のハーネス設定も受け取る: `--pending-timeout SECONDS`、`--default-response TEXT_OR_JSON`、`--on-unmatched pending|default|error`、`--seed N`、`--config FILE`、`--rules FILE`（Docker 向けに同じ設定を環境変数でも指定できる）。

## 2. アプリ / SDK を proxy に向ける

コードは**一切変えず**、base_url を差し替えるだけ。

**Anthropic SDK:**

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8765", api_key="sk-mock-anything")

# 応答が注入されるまでブロックする
msg = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(msg.content)          # → 注入された content blocks
print(msg.usage)            # → 概算 input/output トークン + キャッシュ
```

API key はダミーで良い（proxy は検証しない）。`base_url` の代わりに環境変数 `ANTHROPIC_BASE_URL=http://localhost:8765` を立てても同じ（コードを触らずに横取りできる）。`stream=True` の SSE もそのまま動く。

pending は responder が応答するまで待ち続けるので、SDK の `timeout` は長めにする（Anthropic SDK・OpenAI SDK の既定は 10 分だが、自前コードの 30 秒の `httpx` タイムアウトはそれより先に切れる）。エラーを注入するときは、SDK が既定で 429 / 5xx をバックオフ付きで 2 回リトライすることに注意 — 注入したエラーを 1 回だけ観測したいなら `max_retries=0`、リトライ処理を試したいならそのまま（リトライごとに新しい pending になる）。

**Bedrock（`AnthropicBedrock` と boto3、Converse 含む）:** [Bedrock](bedrock.md) を参照。

**OpenAI SDK (`openai`):**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8765/v1", api_key="sk-mock-anything")
msg = client.chat.completions.create(
    model="gpt-6.2-sol", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

base_url は **`/v1` 込み**（SDK が `/chat/completions` を後置する）。環境変数 `OPENAI_BASE_URL=http://localhost:8765/v1` でも同じ。streaming (`stream=True`) と tool call もそのまま動く。OpenAI 形式のリクエストは保留前に **canonical（Anthropic 風）に正規化**される（system / messages / tools、tool 結果は `tool_result` block）ので、responder は provider に依らず同じ形を読み、同じ canonical blocks を注入すればよい — `chat.completion` 形式への逆変換は server が行う。擬似プロンプトキャッシュはこの経路では**シミュレートしない**（OpenAI のキャッシュは `cache_control` ベースでない自動方式）: cache status は常に `"none"`。

OpenAI 向けの追加: `n > 1` なら responder は choice ごとに答えられ、`logprobs: true` のリクエストには決定的なトークンごとの対数確率を返す（responder が自分の値を注入すればそれを使う）。[`/_control/respond`](control-api.md) の `choices` / `logprobs` を参照。

## 3. 応答を供給する（responder）

別ターミナル / 別セッションで、保留中リクエストに応答を注入する。

```bash
# 何が保留中か見る
curl -s localhost:8765/_control/pending | jq
# → {"has_pending":true,"count":1,"pending":[
#      {"pending_id":"a1b2...","request":{"model":"...","system":...,"messages":[...],"tools":[...]},
#       "waiting_for_seconds":1.2}], ...}

# (a) text だけ即注入する簡易版
curl -s -X POST localhost:8765/_control/auto \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello from the puppet!"}'

# (b) tool_use を含む任意の content blocks を注入する
curl -s -X POST localhost:8765/_control/respond \
  -H 'Content-Type: application/json' \
  -d '{"content": [
        {"type": "text", "text": "天気を調べます。"},
        {"type": "tool_use", "id": "tu_1", "name": "get_weather",
         "input": {"city": "Tokyo"}}
      ]}'
```

注入できる content は `text` / `tool_use` のほか、`thinking`（`{"type":"thinking","thinking":"…","signature"?}` — signature 省略時は不透明な値を生成）と `redacted_thinking`（`{"data":"…"}`）も含められる。これらは応答に保持され、ストリームでは `thinking_delta` / `signature_delta` として流れ、`usage.output_tokens_details.thinking_tokens` に計上され、次ターンではそのまま送り返されることを期待する — 現行モデルが既定で返す形そのもの。サーバー側ツールのブロックもフィクスチャとして受け付ける（実行はしない）: `server_tool_use` / `mcp_tool_use`（id 省略時は生成。ストリームでは `tool_use` と同じく `input_json_delta`）と、結果ブロック `web_search_tool_result` / `web_fetch_tool_result` / `code_execution_tool_result` / `bash_code_execution_tool_result` / `text_editor_code_execution_tool_result` / `tool_search_tool_result` / `mcp_tool_result`（そのまま保持。ストリームでは `content_block_start` / `content_block_stop` の 1 組。出力トークンには数えない）。`usage.server_tool_use` に `web_search` / `web_fetch` の呼び出し数が入り、コストには公式の web 検索料金が加算される（`cost.server_tool_usd`）。`server_tool_use` があってもターンは終わらない（`stop_reason` は `end_turn` のまま）。これらのブロックが流れるのは Anthropic 経路、Bedrock InvokeModel、Mantle エイリアス。Converse と OpenAI 経路は捨て、クライアントが検索を見ていない以上、計上も課金もしない（history にはフィクスチャとして残る）。それ以外のブロック型は捨てたうえで注入の応答に報告する（`{"ok": true, "dropped": ["tool_result"]}`）。`config.strict_blocks` なら 400 で拒否。`tool_result` はアプリ側が次のリクエストで送るものなので、応答には入れない。`stop_reason` は公式語彙（`end_turn` / `max_tokens` / `stop_sequence` / `tool_use` / `pause_turn` / `refusal` / `model_context_window_exceeded`。加えて Bedrock の `guardrail_intervened`、OpenAI 経路の `content_filter`）を受け付け、`"refusal"` なら `stop_details`（`stop_details` を渡さなければ `{"type":"refusal","category":null,"explanation":null}`）、`"stop_sequence"` ならリクエストの `stop_sequences` から `stop_sequence` を埋める（明示指定も可）。

`tool_use` を返すとアプリ側が実ツールを実行 → 結果が次の `messages.create()` に `tool_result` として積まれて再び pending になる。これを繰り返すことでマルチターン / ツール実行ループを丸ごと再現できる。

**responder ループ（long-poll で待ち受ける運用）:**

```bash
# 次の pending を最大 270 秒待つ。来たら応答、来なければ timeout で抜けて再ループ。
while true; do
  r=$(curl -s "localhost:8765/_control/wait_for_pending?timeout=270")
  echo "$r" | jq -e '.has_pending' >/dev/null || continue   # timeout → 再待機
  pid=$(echo "$r" | jq -r '.pending_id')
  # ... request の system/messages/tools を読んで応答を組み立て ...
  curl -s -X POST localhost:8765/_control/respond \
    -H 'Content-Type: application/json' \
    -d "{\"pending_id\":\"$pid\",\"content\":[{\"type\":\"text\",\"text\":\"...\"}]}"
done
```

responder は次の 3 択で、どれも同じ制御 API を使うため**自由に入れ替え可能**（セッション途中でも）:

1. **人間**（上記のように curl で）
2. **AI エージェントに「LLM のフリ」をさせる**（Claude Code / Codex がリクエストを読んで忠実に即興応答する）— 指示書は [`responder/CLAUDE.md`](../../responder/CLAUDE.md)（Claude Code 用）/ [`responder/AGENTS.md`](../../responder/AGENTS.md)（Codex CLI など `AGENTS.md` 規約の agent 用）。どちらも中立に保つための核心原則・複数 pending 対応・注入フォーマット・禁忌・JSON escape の罠を網羅（内容はほぼ同じで、ランタイム前提だけ差分）
3. **同梱の relay** で実 API に転送する（[relay モード](relay.md)）

## 4. エラー応答を注入してハンドリングを試す

分岐テスト用に、任意の HTTP エラーを pending に返させられる（Anthropic / Bedrock / OpenAI の 3 経路すべてで各 provider のエラー形式に変換される）。任意の `code` / `param` フィールドは OpenAI 経路で素通しされる（例 `"code": "rate_limit_exceeded"`）:

Bedrock 経路ではボディが `{"message": "...", "__type": "<AwsException>"}` になり、`x-amzn-ErrorType` ヘッダが付く（2 つの 424 は上流の失敗を包むもので、その status を `originalStatusCode` に載せる — HTTP status と違えたい場合（424 が 429 を包む等）は `/_control/error` に `original_status` を渡す。botocore のモデルどおり `ModelErrorException` は `resourceName`、`ModelStreamErrorException` は `originalMessage` を持つ）。`type` を省略した場合は Anthropic 経路も status から導出する（429 → `rate_limit_error`、529 → `overloaded_error`、400 → `invalid_request_error` …）ので、`anthropic` SDK は固有のクラスを投げる。`type` がすでに AWS の例外名（`Exception` で終わる）ならそのまま使い、そうでなければ `status` から導出する: 400 → `ValidationException`、401 → `UnrecognizedClientException`、403 → `AccessDeniedException`、404 → `ResourceNotFoundException`、408/504 → `ModelTimeoutException`、413 → `RequestEntityTooLargeException`、424 → `ModelErrorException`、429 → `ThrottlingException`、500 → `InternalServerException`、503 → `ServiceUnavailableException`、529 → `overloaded_error`（そのまま通す。実サービスが上流の 529 を包み直すかどうかは未検証）— その他 4xx → `ValidationException`、その他 5xx → `InternalServerException`。それ以外は AWS の例外名を明示する（`ServiceQuotaExceededException` 400、`ModelNotReadyException` 429、`ModelStreamErrorException` 424）。つまり同じ `{"status": 429, "type": "rate_limit_error"}` の注入が、Bedrock クライアントには `ThrottlingException`、Anthropic クライアントには `rate_limit_error` として届く。

**ストリーミング**リクエストなら、実 API と同じように途中で失敗させることもできる。
`after_events`（と、先に流す `content`）を足すと、応答は通常の 200 ストリームとして始まり、
その数だけイベントを流してから provider のエラーイベントで終わる: Anthropic 経路は `event: error`、
Bedrock 系経路は event stream の**例外フレーム**になる。SDK が投げるもの: 純正 `anthropic` クライアントは
SSE の `error` イベントから通常どおり `APIStatusError`、boto3 は `Error.Code` にフレームのメンバー名
（`throttlingException` など）を持つ `botocore.exceptions.EventStreamError`、`AnthropicBedrock` は
ストリームデコーダから素の `ValueError`（`anthropic.APIError` ではない）を投げるので、それに合わせて
捕捉すること。フレームのメンバーは操作ごとの union に限られる（`internalServer` / `modelStreamError` /
`validation` / `throttling` / `serviceUnavailable`、InvokeModel のみ `modelTimeout` も）。union 外の
名前はステータス階級を保つ — 429 系の `ModelNotReadyException` は `throttlingException`、408/504 は
InvokeModel では `modelTimeoutException` のまま、timeout メンバーを持たない ConverseStream では
`originalStatusCode` 付きの `modelStreamErrorException` になり、黙って internal error に落ちることはない。
`after_blocks` は経路に依存しない形: 完成したブロックがその数だけ流れたあとで失敗する（メッセージ開始 + N 個のブロックの開始〜終了。イベントの種類は経路ごと。`0` ならストリームが開いた直後に失敗）。どちらも Anthropic、Bedrock InvokeModel / Mantle、Converse のストリームに効く。OpenAI 経路には途中失敗の形がなく素の HTTP エラーになる（部分的な記録もしない）。`after_events` は経路ごとの自前のイベントを数えるので、同じ数でも経路によって届く内容は違う
（InvokeModel ストリームと SSE には `content_block_start` があり、ConverseStream はテキストブロックを
最初のデルタで始める）:

```bash
curl -s -X POST localhost:8765/_control/error \
  -d '{"status": 429, "type": "ThrottlingException", "message": "slow down",
       "after_events": 3, "content": [{"type": "text", "text": "partial answer"}]}'
```

イベント数は終端イベントを含まないようにクランプするので、途中で失敗したストリームが同時に
「正常終了した」ようにも見えることはない。数えるのはプロトコルイベントだけで、SSE 経路が
`message_start` の直後に流す `ping` はカウントに含まない。history エントリには `after_events` と併せて
`injected_error.partial_content`（部分ストリーム用に responder が渡した content）が載る — 実際にワイヤに
流れたストリーミングリクエストの場合だけ。`respond` / `error` の `content` に入れる `redacted_thinking` の
`data` は base64 でなければならない（Bedrock の SDK はクライアント側でデコードするので、素のテキストだと
呼び手が `binascii.Error` で落ちる）。`/_control/*` は理由を添えた 400 で拒否する。非ストリーミングのリクエストは `after_events` を無視して通常の HTTP エラーに
なる（OpenAI 経路は常に通常のエラー）。

```bash
# 429 → SDK が自動 retry する
curl -s -X POST localhost:8765/_control/error \
  -d '{"status": 429, "type": "rate_limit_error", "message": "throttled"}'

# 401 → retry されない (認証エラー分岐の確認)
curl -s -X POST localhost:8765/_control/error \
  -d '{"status": 401, "type": "authentication_error", "message": "bad key"}'
```

`status` は 100–599 の整数。範囲外・非数値は `400` を返し、pending は触らない（呼び出し側はハングせず注入をやり直せる）。

## 並列リクエスト（multi-pending）

server は同時複数リクエストを保持できる。各 pending は一意な `pending_id` を持ち、`/_control/respond`（`auto` / `error` も）に `pending_id` を指定して個別に注入する。

- `pending_id` 省略は pending が**ちょうど 1 件**のときのみ可。0 件 → `400`、複数 → `400`（`pending_ids` を返すので選んで指定）。
- 存在しなくなった pending（解決済み / `clear` で消えた等）への注入は `400`（`no pending request`）。ほぼ同時の二重注入レースのみ `409`（`already resolved`）。

注入ペイロードの組み立て方（特に日本語 + ネスト JSON の escape 事故回避）は [`responder/CLAUDE.md`](../../responder/CLAUDE.md) / [`responder/AGENTS.md`](../../responder/AGENTS.md) に詳しい。
