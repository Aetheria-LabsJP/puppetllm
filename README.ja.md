**日本語** | [English](README.md)

# puppetllm — LLM API debug proxy (fake Anthropic / Bedrock / OpenAI server)

Anthropic Messages API / Bedrock（InvokeModel・Converse・バッチ推論）/ OpenAI Chat Completions 互換の **fake server**。`ANTHROPIC_BASE_URL`（または `AnthropicBedrock` / `OpenAI` の base_url）をこのサーバに向けるだけで、**アプリ / SDK のコードを 1 行も変えずに** LLM 呼び出しを横取りし、人間 or 別エージェントが応答を供給できる（human-in-the-loop / AI-in-the-loop）。

用途:

- **ゼロ円デバッグ**: 実 API を叩かずにエージェント / オーケストレーションの挙動を再現・検証する
- **決定論的テスト**: 任意のレスポンス（text / tool_use、エラー）を注入して分岐を再現する
- **クロスプロバイダブリッジ（[relay モード](#relay-モードクロスプロバイダブリッジ)）**: ある SDK 向けに書かれたアプリを**別の実プロバイダ**で動かす（例: Anthropic SDK の agent を Grok / GPT で、OpenAI SDK のアプリを Claude で）— アプリのコードは 1 行も変えずに
- **コスト目安**: リクエストごとの概算トークン / 料金を集計する（`/_control/stats`）
- **擬似プロンプトキャッシュ観測**: アプリが `cache_control` を効かせられる構造で投げているかをハッシュで観測する（`/_control/cache`）

> 概算（approx tokenizer）ベースなので**実課金とは一致しない**。傾向把握・構造検証用。

---

## アーキテクチャ

provider 非依存の canonical core + アダプタ:

- `puppetllm/fake_server.py` — canonical core（正規化 snapshot 管理 + `/_control/*` + cost/cache 計算）。Anthropic 経路 `POST /v1/messages` を内蔵
- `puppetllm/providers/bedrock.py` — Bedrock 経路 `POST /model/{id}/invoke[-with-response-stream]`（モデル ID 正規化・`anthropic_version` 検証・AWS 形式のエラー / ヘッダ。AWS event stream フレーミングは `providers/eventstream.py`）
- `puppetllm/providers/converse.py` — Bedrock Converse 経路 `POST /model/{id}/converse[-stream]`（Converse の JSON スキーマと canonical の相互変換）
- `puppetllm/providers/bedrock_batch.py` + `puppetllm/providers/s3.py` — Bedrock バッチ推論（`/model-invocation-job*`）と、同梱のディレクトリ実装 S3 エミュレーション
- `puppetllm/providers/openai.py` — OpenAI 経路 `POST /v1/chat/completions`（リクエストは canonical（Anthropic 風）に正規化し、レスポンスは `chat.completion` JSON / SSE chunk に変換）
- `puppetllm/batches.py` — Anthropic Message Batches 経路 `/v1/messages/batches*`（各 custom_id を通常の pending として保持。バッチのライフサイクルは `/_control/batch/*` から注入可能）
- `puppetllm/cache_sim.py` — 擬似プロンプトキャッシュ（multi-breakpoint + トップレベル `cache_control` の自動キャッシュ + 前方一致 + 世代別最小閾値 + 5m / 1h TTL + 20-block lookback + effort / thinking / tool_choice による無効化）
- `puppetllm/pricing.py` — 概算トークン + 料金表（Claude は Fable / Mythos / Sonnet 5 / Opus 4.1 を含む世代別、GPT / o 系ファミリ。公式料金ページに準拠）

provider は **URL パスで自動判別**（モード切替・設定は不要）。応答 content blocks / 制御 API は provider 共通（注入は同じ `/_control/respond`）。

---

## 使い方

全体像は **3 つの登場人物**で考える:

```
  +--- アプリ / SDK ---+         +----- puppetllm ------+        +-- responder --+
  | messages.create()  | ------> | POST /v1/messages    | -----> | 応答を注入    |
  | (応答までブロック) | <------ | (pending として保留) | <----- | /_control/... |
  +--------------------+  応答   +----------------------+        +---------------+
        (1)アプリ                 (2)fake server (本体)        (3)供給側 (人 or AI)
```

(1) が投げたリクエストを (2) が**保留 (pending)** し、(3) が `/_control/*` で応答を流し込むと、(1) の `create()` がその応答で返る。実 API は一切叩かない。

### 1. proxy を起動する

```bash
# A) Docker (推奨)
docker compose up -d
curl localhost:8765/_control/health        # → {"ok":true,"turn_count":0}

# B) 直接 (Python 3.12+) — 起動バナーと共に foreground 実行
pip install -r requirements.txt
python3 -m puppetllm --host 127.0.0.1 --port 8765
#   [puppetllm] starting on http://127.0.0.1:8765
#   [puppetllm] Anthropic: set ANTHROPIC_BASE_URL=http://127.0.0.1:8765
#   [puppetllm] Bedrock:   point AnthropicBedrock base_url to http://127.0.0.1:8765
#   [puppetllm] OpenAI:    set OPENAI_BASE_URL=http://127.0.0.1:8765/v1  (note the /v1)

# C) uvicorn 直叩き (reload 等のオプションを使いたいとき)
python3 -m uvicorn puppetllm.fake_server:app --host 127.0.0.1 --port 8765
```

`--host` 既定は `127.0.0.1`（localhost のみ）。LAN/VPN 越しに使うときだけ `0.0.0.0` にする（[セキュリティ](#セキュリティ)参照）。

### 2. アプリ / SDK を proxy に向ける

コードは**一切変えず**、base_url を差し替えるだけ。

**Anthropic SDK:**

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8765", api_key="sk-mock-anything")

# 応答が注入されるまでブロックする
msg = client.messages.create(
    model="claude-sonnet-4-5", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(msg.content)          # → 注入された content blocks
print(msg.usage)            # → 概算 input/output トークン + キャッシュ
```

API key はダミーで良い（proxy は検証しない）。`base_url` の代わりに環境変数 `ANTHROPIC_BASE_URL=http://localhost:8765` を立てても同じ（コードを触らずに横取りできる）。`stream=True` の SSE もそのまま動く。

**Bedrock SDK (`AnthropicBedrock`):**

```python
from anthropic import AnthropicBedrock
client = AnthropicBedrock(base_url="http://localhost:8765", aws_region="us-east-1",
                          aws_access_key="dummy", aws_secret_key="dummy")
msg = client.messages.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

SigV4 署名のため bedrock extra が必要: `pip install 'anthropic[bedrock]'`。AWS クレデンシャルはダミーで良い（proxy は署名を検証しない）が、SDK が署名を作るために何かしらは必要。model は URL パス (`/model/{id}/invoke`) に入り、streaming は AWS event stream で返る — どちらも server が吸収する。**応答の注入方法は Anthropic 経路と完全に同じ**（下記 `/_control/respond` をそのまま使う）。

Bedrock 経路固有の挙動（すべて URL パスから判定。モード切替は不要）:

- **モデル ID の正規化**: `anthropic.claude-haiku-4-5-20251001-v1:0`、サフィックスなしの現行 ID（`anthropic.claude-opus-5`）、クロスリージョン推論プロファイル（`us.` / `eu.` / `apac.` / `jp.` / `au.` / `global.` / `us-gov.` … プレフィックス）、foundation-model / inference-profile の ARN（`aws` / `aws-cn` / `aws-us-gov` の全パーティション）を Anthropic 側の名前（`claude-haiku-4-5-20251001`）にマップする。pending snapshot の `model`・応答の `model` フィールド・`/_control/stats` の `by_model` はこの正規名を使うので、同じモデルの Bedrock 経由と Anthropic 直の呼び出しが 1 行に合流し、relay の `--model-map` / `--only` の glob も `claude-*` でマッチする。生の ID は snapshot / history エントリの `bedrock_model_id` に保持する。`anthropic.` セグメントを含まない ID はそのまま素通しする。
- **`anthropic_version` を検証する**（Anthropic モデル ID のみ）: 欠落、または `bedrock-2023-05-31` 以外の値は受付時点で `400 ValidationException` として弾く（pending は作られない）— 自前実装クライアントのミスを早期に検出する。他ベンダーの ID（`meta.llama…`、`amazon.titan…`）はボディ形式が異なるため検証せず素通しする。不正な `cache_control` 配置（§5 参照）も同じ形で弾く。
- **SigV4 は検証しない**: `Authorization` / `X-Amz-Date` / `X-Amz-Security-Token` は無視する。
- **レスポンスヘッダ**: 非ストリーミング応答には `X-Amzn-Bedrock-Input-Token-Count`（Converse の `inputTokens` と同様、非キャッシュ分のみ）/ `X-Amzn-Bedrock-Output-Token-Count` / `X-Amzn-Bedrock-Cache-Read-Input-Token-Count` / `X-Amzn-Bedrock-Cache-Write-Input-Token-Count` / `X-Amzn-Bedrock-Invocation-Latency`（+ `x-amzn-requestid`）を付与する。ストリーミングは実 Bedrock と同様に最終チャンクの `amazon-bedrock-invocationMetrics`（`cacheReadInputTokenCount` / `cacheWriteInputTokenCount` 込み）に載せ、`X-Amzn-Bedrock-Content-Type` を付ける。boto3 が**リクエストヘッダ**で送るオプション — `serviceTier=`（`X-Amzn-Bedrock-Service-Tier`、`priority | default | flex | reserved`）と `performanceConfigLatency=`（`X-Amzn-Bedrock-PerformanceConfig-Latency`）— は検証した上で両経路の応答ヘッダにそのまま echo し（ティア未指定なら `default`）、ティアは responder に canonical の `service_tier` として見える（`priority` / `flex` → `auto`、`default` / `reserved` → `standard_only`。relay と同じ語彙）。
- **`AnthropicBedrockMantle` 向け Messages API エイリアス**: `POST /anthropic/v1/messages`（`bedrock-runtime` / `bedrock-mantle` ホストが提供するパス）は Bedrock モデル ID 正規化付きの Anthropic ハンドラで、さらに素の `/v1/messages` 経路も `model` が Bedrock 形式（`[region.]anthropic.…` / ARN）なら同じハンドラに渡すため、`AnthropicBedrockMantle(base_url="http://localhost:8765")` はルート直下でも `/anthropic` 付きでも動く（SSE ストリーミング、Anthropic エラー形式、`anthropic-version` ヘッダ、ボディに `anthropic_version` なし）。Batches / count_tokens はこのエイリアスでは提供しない（Bedrock と同じ）。
- **受信ログ**: Bedrock 経路のリクエスト（および拒否）は 1 件ごとに stderr へ `[bedrock] invoke model=<raw> -> <canonical> pending=<id> …` の 1 行を出すので、他経路との区別が一目でつく。

**boto3 (`bedrock-runtime`) — Converse 含む:**

```python
import boto3
from botocore.config import Config

rt = boto3.client("bedrock-runtime", region_name="us-east-1",
                  aws_access_key_id="dummy", aws_secret_access_key="dummy",
                  endpoint_url="http://localhost:8765")

rt.converse(modelId="anthropic.claude-opus-5",
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
            inferenceConfig={"maxTokens": 1024})

for event in rt.converse_stream(modelId="anthropic.claude-opus-5",
                                messages=[{"role": "user", "content": [{"text": "hello"}]}])["stream"]:
    print(event)          # messageStart / contentBlockDelta / … / metadata
```

`POST /model/{id}/converse` と `/converse-stream` は Converse の JSON スキーマを話し、他経路と
**同じ canonical な pending** に正規化される。responder はいつもどおり `/_control/respond` に
content blocks を注入するだけで、経路の違いを意識しなくてよい:

| Converse | canonical |
|---|---|
| `{"text": …}` | `{"type": "text", …}` |
| `{"image": {"format", "source": {"bytes" \| "s3Location"}}}` / `{"document": …}` | `{"type": "image" \| "document", "source": {…}}` |
| `{"toolUse": {"toolUseId", "name", "input"}}` | `{"type": "tool_use", "id", "name", "input"}` |
| `{"toolResult": {"toolUseId", "content", "status"}}` | `{"type": "tool_result", "tool_use_id", "content", "is_error"}` |
| `{"reasoningContent": {"reasoningText": {"text", "signature"}}}` / `{"redactedContent"}` | `{"type": "thinking", …}` / `{"type": "redacted_thinking", "data"}` |
| `{"cachePoint": {"type": "default", "ttl"?}}` | **直前の**ブロック / ツールへの `cache_control`（擬似キャッシュの実ブレークポイントになる） |
| `inferenceConfig.{maxTokens,temperature,topP,stopSequences}` | `max_tokens` / `temperature` / `top_p` / `stop_sequences` |
| `toolConfig.tools[].toolSpec` / `toolChoice {auto\|any\|tool}` | `tools[]` / `tool_choice` |
| `additionalModelRequestFields` | canonical ボディにマージ（`thinking`、`top_k`、`anthropic_beta` など）。Converse 自身が持つキー（`system`、`tools`、`max_tokens`、`temperature` …）は `ValidationException` — スキーマの抜け道にはならない |
| `outputConfig.effort` / `outputConfig.textFormat` | `output_config.effort`（`low` … `xhigh` / `max`）/ `output_config.format`（`{type: json_schema, schema, name?}` — `textFormat.structure.jsonSchema.schema` は API 定義どおり JSON **文字列**で、ここでスキーマオブジェクトにデコードする）。native 側が `additionalModelRequestFields` の同名キーより優先され、そこに非オブジェクトの `output_config` があれば `ValidationException` |
| `promptVariables`（モデル ID がプロンプト管理の ARN） | responder 向けに `converse.promptVariables` に保持。その場合 `messages` は実 API 同様省略可 — 保存済みプロンプトが供給するので responder には `messages: []` として見える。通常のモデル ID では `messages` は必須のまま |
| `serviceTier.type` | `service_tier`（`priority` / `flex` → `auto`、`default` / `reserved` → `standard_only`） |

応答は assistant ターンが実際に持ちうるブロック種別だけを戻す（`text` / `toolUse` /
`reasoningContent`。responder のブロックはエンコーダに渡る前に text / tool_use / thinking /
redacted_thinking に正規化される）: `output.message`、Converse 語彙の `stopReason`（canonical の `refusal` は
`content_filtered`、`pause_turn` は `end_turn`）、`inputTokens` がキャッシュ分を除いた
`usage`（+ `cacheReadInputTokens` / `cacheWriteInputTokens` / `cacheDetails`）、`metrics.latencyMs`。
`additionalModelResponseFieldPaths`（JSON ポインタ、最大 10 件）はネイティブの Messages API 応答に対して
解決し、ストリームでは `messageStop` にも載せる。リクエストはスキーマの「形」を検証する（制約すべてではない）: content block・
source・`reasoningContent`・`toolChoice`・`system` の各要素は**ユニオン**（メンバーはちょうど 1 つ。`text` と
`cachePoint` を同時に持つブロックはテキストを黙って捨てずに `ValidationException`。ユニオンの未知の
メンバーや、リクエスト最上位の未知のキーも同様）で、各ユニオンは自分の
メンバーしか受け付けない（`ImageSource` / `VideoSource` は `bytes | s3Location` のみ。`text | content` を
取るのは `DocumentSource` だけ）。必須フィールドと enum
（`image.format`、`video.format`、`document.name`、`toolUse.input`、`toolResult.content`、`cachePoint.type` / `ttl`、
`inferenceConfig` の範囲、`serviceTier.type`、`requestMetadata` の 1〜16 件、`guardContent.image` が `bytes` 由来の
`png` / `jpeg` であること）も検証し、2 つの `cachePoint` が同じブロックを指すことも拒否する。検証しないもの（本番なら弾かれるリクエストが通りうる）:
デコード後のメディアサイズと妥当性、ブロック間の配置ルール、`audio` / `searchResult` の内部構造
（「オブジェクトであること」だけ）、user / assistant の交互性（botocore のモデルどおり `role: "system"`
も受け付ける）、`guardrailConfig` / `promptVariables` /
プロンプト ARN の条件付き制約。
ストリームは `messageStart` → ブロックごとに `contentBlockStart`（tool use）/ `contentBlockDelta` /
`contentBlockStop` → `messageStop` → `metadata` を、生 JSON の event stream フレームとして流す
（InvokeModel 経路の `chunk` + base64 ラッパーとは異なる）。

**OpenAI SDK (`openai`):**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8765/v1", api_key="sk-mock-anything")
msg = client.chat.completions.create(
    model="gpt-5.4", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

base_url は **`/v1` 込み**（SDK が `/chat/completions` を後置する）。環境変数 `OPENAI_BASE_URL=http://localhost:8765/v1` でも同じ。streaming (`stream=True`) と tool call もそのまま動く。OpenAI 形式のリクエストは保留前に **canonical（Anthropic 風）に正規化**される（system / messages / tools、tool 結果は `tool_result` block）ので、responder は provider に依らず同じ形を読み、同じ canonical blocks を注入すればよい — `chat.completion` 形式への逆変換は server が行う。擬似プロンプトキャッシュはこの経路では**シミュレートしない**（OpenAI のキャッシュは `cache_control` ベースでない自動方式）: cache status は常に `"none"`。

### 3. 応答を供給する（responder）

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

注入できる content は `text` / `tool_use` のほか、`thinking`（`{"type":"thinking","thinking":"…","signature"?}` — signature 省略時は不透明な値を生成）と `redacted_thinking`（`{"data":"…"}`）も含められる。これらは応答に保持され、ストリームでは `thinking_delta` / `signature_delta` として流れ、`usage.output_tokens_details.thinking_tokens` に計上され、次ターンではそのまま送り返されることを期待する — 現行モデルが既定で返す形そのもの。`stop_reason` は公式語彙（`end_turn` / `max_tokens` / `stop_sequence` / `tool_use` / `pause_turn` / `refusal` / `model_context_window_exceeded`）を受け付け、`"refusal"` なら `stop_details`（`stop_details` を渡さなければ `{"type":"refusal","category":null,"explanation":null}`）、`"stop_sequence"` ならリクエストの `stop_sequences` から `stop_sequence` を埋める（明示指定も可）。

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
2. **AI エージェントに「LLM のフリ」をさせる**（Claude Code / Codex がリクエストを読んで忠実に即興応答する）— 指示書は [`responder/CLAUDE.md`](responder/CLAUDE.md)（Claude Code 用）/ [`responder/AGENTS.md`](responder/AGENTS.md)（Codex CLI など `AGENTS.md` 規約の agent 用）。どちらも中立に保つための核心原則・複数 pending 対応・注入フォーマット・禁忌・JSON escape の罠を網羅（内容はほぼ同じで、ランタイム前提だけ差分）
3. **同梱の relay** で実 API に転送する（下記 [relay モード](#relay-モードクロスプロバイダブリッジ)）

### 4. エラー応答を注入してハンドリングを試す

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
`after_events` は経路ごとの自前のイベントを数えるので、同じ数でも経路によって届く内容は違う
（InvokeModel ストリームと SSE には `content_block_start` があり、ConverseStream はテキストブロックを
最初のデルタで始める）
（anthropic SDK は反復中に surface する）:

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

### 5. コスト / トークン / キャッシュを観測する

```bash
# 累計サマリ (全て概算)
curl -s localhost:8765/_control/stats | jq
# → {"is_estimate":true,"completed_requests":3,"error_requests":0,
#     "totals":{"input_tokens":..,"output_tokens":..,
#               "cache_read_input_tokens":..,"total_usd":..,"cache_savings_usd":..},
#     "cache":{"hits":2,"misses":1,"hit_rate":0.6667,"index_size":2},
#     "by_model":{"claude-sonnet-4-5":{"requests":3,"total_usd":..}}}

# 擬似プロンプトキャッシュの index (prefix hash 別 hit/miss)
curl -s localhost:8765/_control/cache | jq

# 1 リクエストずつの (request, response, usage, cost, cache) 履歴
curl -s localhost:8765/_control/history | jq '.history[-1]'

# テスト間のクリーンアップ (pending / history / cache を全消去)
curl -s -X POST localhost:8765/_control/clear
```

`cache_savings_usd` は「キャッシュが効いた分、本物なら浮いたであろう概算額」。アプリが `cache_control` を正しい構造で投げられているかの検証に使う。（Anthropic / Bedrock 経路のみ — OpenAI 経路は常に cache status `"none"` で hit/miss カウンタも汚さない。）

擬似キャッシュが再現する挙動（公式の prompt-caching ドキュメントに準拠）:

- **配置**: ブロック単位の `cache_control`（最大 4 つ）と、**トップレベル** `cache_control`（自動キャッシュ: 最後のキャッシュ可能ブロックに 1 つ置く。`thinking` と空テキストは飛ばす）。前方一致の順序は `tools → system → messages`、マーカー自体はキーに含まない。
- **最小キャッシュ長は世代別**: Fable 5 / 5.1、Mythos、Opus 5 = 512 トークン、Opus 4.8 = 1024、Opus 4.7 = 2048、Opus 4.6 / 4.5 = 4096、Opus 4.1 / 4 = 1024、Sonnet 全世代 = 1024、Haiku 4.5 = 4096、Haiku 3.5 = 2048。未満なら実 API 同様にエラーなしで `"none"` と観測する。
- **TTL**: `{"type":"ephemeral"}` は 5 分、`{"type":"ephemeral","ttl":"1h"}` は 1 時間（テスト用に短縮するなら `PUPPETLLM_CACHE_TTL` / `PUPPETLLM_CACHE_TTL_1H`。read で延長）。書き込みは TTL 別に `usage.cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` に分けて報告し、単価は入力の 1.25x / 2x。
- **無効化**: `output_config.effort`、`thinking`、`tool_choice` をターン間で変えると *messages* 側のプレフィックスが無効になる。`speed`（fast mode）の切り替えは system + messages を無効化し tools は生きる。これは公式の無効化表の意図的な簡略化で、公式表は thinking / effort について tools / system キャッシュを「モデル依存」（設定を system プロンプトより前に描画するモデルがある）としているが、puppetllm は messages のみとして扱う。明示の既定値は省略と同じ扱い: `effort: high`、`tool_choice: auto`（`disable_parallel_tool_use: false` 付きも同様）、モデルの既定 thinking モード（Opus 5 / Sonnet 5 / Fable / Mythos は `adaptive`、それ以前は `disabled`。`display: omitted`）。モデル変更は全無効。
- **検証**: 実 API が拒否する配置はここでも `400 invalid_request_error`（pending は作られない）: 不正なマーカーや未知の `ttl`、明示ブレークポイント 5 個以上、`thinking` ブロック上のマーカー、5m ブレークポイントの後の 1h ブレークポイント、明示 4 個があるところへのトップレベル `cache_control`、最後のブロックの明示マーカーと食い違うトップレベル `ttl`。
- **fast mode**: `speed: "fast"` は公式の 2 倍で課金し（Anthropic / Bedrock 経路のみ）、`usage.speed: "fast"` を返し、Anthropic 上流へはアプリの `anthropic-beta` ヘッダごと転送する（OpenAI 互換上流へは警告を出して落とす）。モデル / プラットフォームの対応可否は検証しない。
- **lookback**: 各ブレークポイントは最大 20 ポジション遡る。連続する `tool_use`（または `tool_result`）ブロックの並びは 1 ポジションと数える。
- **usage の形**: 2026 年の usage オブジェクト（`cache_creation`、`output_tokens_details.thinking_tokens`、`server_tool_use`、`service_tier`（`standard` / `batch`）、`inference_geo`、fast mode 要求時は `speed`）を返し、ストリームの `message_delta.usage` には実 API と同様に累積の input / cache 各値も載る。

料金は公式の世代別表に従う（例: Opus 5 $5/$25、Sonnet 5 $2/$10、Sonnet 4.6 $3/$15、Haiku 4.5 $1/$5、Fable 5.1 $10/$50・cache read $0.25、Opus 4.1 $15/$75）。未知の Claude ID は Sonnet 4.x 価格、未知の `gpt-*` は gpt-5.4 価格にフォールバックする。

### 6. Message Batches API

Anthropic Message Batches サーフェス（`/v1/messages/batches*`）も提供しているので、`client.messages.batches.create()` / `retrieve()` / `results()` / `cancel()` / `list()` / `delete()` がそのまま動く:

```python
batch = client.messages.batches.create(requests=[
    {"custom_id": "r1", "params": {"model": "claude-sonnet-4-5", "max_tokens": 64,
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

### 7. Bedrock バッチ推論（S3 エミュレーション同梱）

コントロールプレーンのバッチ API も、ディレクトリ実装の **S3 エミュレーション**の上に載せて提供する。
素の `boto3` S3 クライアントで入力を置き、結果を読める:

```python
import boto3
from botocore.config import Config

s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="d", aws_secret_access_key="d",
                  endpoint_url="http://localhost:8765",
                  config=Config(s3={"addressing_style": "path"}))     # path-style 必須
bedrock = boto3.client("bedrock", region_name="us-east-1", aws_access_key_id="d",
                       aws_secret_access_key="d", endpoint_url="http://localhost:8765")

s3.create_bucket(Bucket="batch-in"); s3.create_bucket(Bucket="batch-out")
s3.put_object(Bucket="batch-in", Key="jobs/input.jsonl", Body=b'''{"recordId": "r1", "modelInput": {...}}\n''')

job = bedrock.create_model_invocation_job(
    jobName="myjob", modelId="anthropic.claude-opus-5",
    roleArn="arn:aws:iam::123456789012:role/BatchRole",
    inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://batch-in/jobs/input.jsonl"}},
    outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://batch-out/results/"}})
```

`{"recordId", "modelInput"}` の各行が通常の **pending**（provider は `bedrock`、snapshot に
`job_arn` / `job_id` / `record_id` が付く）になるので、responder は同じ `/_control/respond` /
`auto` / `error` で答える。全レコードの結果が揃うと、ジョブは
`<出力プレフィックス>/<jobId>/<入力ファイル名>.out` と `manifest.json.out` を書き、`Completed` で終わる。
出力ファイル名は入力プレフィックスからの相対パスを保つので、`a/data.jsonl` と `b/data.jsonl` は衝突しない。
出力の各行は次の形:

```json
{"recordId": "r1", "modelInput": { … 送信したもの … },
 "modelOutput": { … }}                                  // 失敗した場合は:
{"recordId": "r2", "modelInput": { … },
 "error": {"errorCode": 400, "errorMessage": "…"}}
```

結果に順序保証はないので `recordId` が唯一の対応付けキー。manifest は次の形:

```json
{"totalRecordCount": 3, "processedRecordCount": 3, "successRecordCount": 2,
 "errorRecordCount": 1, "inputTokenCount": 120, "outputTokenCount": 48}
```

`modelInput` は InvokeModel のボディ（既定）か、`modelInvocationType: "Converse"` なら Converse の
ボディで、`modelOutput` はそれぞれ対応する応答形式になる。InvokeModel のボディはモデル固有なので
その場合は Anthropic の `modelId` が必要。`Converse` はモデル非依存のスキーマなので任意のモデルを取れる。同梱ストアが守れないものは echo せず拒否する：
`s3EncryptionKeyId`（出力は素のファイル）は `ValidationException`、`s3BucketOwner` はストアのバケットと
ジョブ ARN が属する唯一のアカウント `123456789012` でなければならない。`get_model_invocation_job` /
`list_model_invocation_jobs` / `stop_model_invocation_job` も通常どおり使える
（`GET /_control/bedrock_jobs` でレジストリと未解決の recordId を確認できる）。

`Stop` は同期的に確定する — 呼び出しが返った時点でステータスは終端になっている — ただし入力の読み込み中や
出力の書き出しが既に進行中の場合だけは、ジョブを `Stopping` のまま `200` を返し、終端ステータスは後から
付く。ジョブは実サービスが `StopModelInvocationJob` に対して返す終端ステータス `Stopped` で終わる。注入が進行中だったレコードはそのまま完了するが、未処理のまま中断されたレコードは
`totalRecordCount` にだけ残り、`processedRecordCount` にも `errorRecordCount` にも数えず、出力行も書かない
（`/_control/bedrock_jobs` の `cancelled` に並ぶ）。Stop はジョブがまだ `Submitted` または登録中の間も受理され
（未登録だった分は処理されない。そのジョブの入力がその後の検証 — 読み込み時でも登録時でも — で落ちた場合は、
止めた側の目の前から消えるのではなく理由付きの `Failed` で終わる）、`Stopping` / `Stopped` になった後は冪等（botocore は
応答を取りこぼした Stop をリトライする）、`Completed` / `Failed` のジョブには `ConflictException`、
出力の書き出し中に受理されたものも終端ステータスを決める。出力が書けなかった場合
（ジョブ実行中の出力バケット削除、ディスクエラー）はジョブが **`Failed`** で終わり `message` に理由が入る —
失敗前に書けた分は残る — ので、ポーリングする側は `Completed` / `Stopped` / `Failed` を終端の集合として
扱うこと。`modelInput` が不正な
レコードは実サービス同様ジョブを失敗させずに `error` レコードになるが、JSONL 行自体が壊れている場合、
`recordId` が文字列でない / 重複している場合、入力 / 出力バケットが存在しない場合、レコードが 1 件もない
場合は作成全体をロールバックして `ValidationException` を返す。AWS がバッチで非対応としている機能は `errorCode 400` のレコードとして
返す: ツール呼び出し、構造化出力、そしてプロンプトキャッシュ（レコード内のどこかにある
`cache_control` / `cachePoint`）。

その他の意図的な差分（忠実さより決定性）: `Validating` / `Scheduled` のフェーズはない
（レコードが pending になった時点で `InProgress`）。最小レコード数は「1 件以上」だけで、
時計による expire もない。入力 JSONL は全量をメモリに読むので、実用上のジョブサイズは AWS の
1 GB / 50,000 レコードではなく RAM で決まる。`/_control/clear` はジョブレジストリを捨てるが
S3 ストアには意図的に触れないので、クリア前に書き終えた出力はそのまま読める。その瞬間に進行中だった
create — 入力読み込み中、登録中、同一トークンの双子を待機中、インラインで確定中のいずれでも — は
`400 ConflictException` を返す（あえてリトライ可能な 5xx にしない — botocore が、人が消したばかりの
ジョブを作り直してしまうため）。出力のレイアウト（`<prefix>/<jobId>/<file>.out` がストアの
255 バイト・セグメント制限に収まること、prefix とその祖先のどれも既存オブジェクトでないこと）は作成時に
検査するので、全レコード処理後にそれで `Failed` になることはない。`recordId` を付ける場合は空でない
文字列でなければならない（空文字列を生成 ID で黙って置き換えることはしない）。

S3 エミュレーションはこのフローに必要な範囲だけ — バケットの `PUT`/`HEAD`、オブジェクトの
`PUT` / `GET` / `HEAD` / `DELETE`、`GET /`（バケット一覧）、バケットの `DELETE`（空でなければ
`409 BucketNotEmpty`）、レンジ `GET`（`Range` → `206` + `Content-Range`、範囲外は
`416 InvalidRange`）、条件付きリクエスト（`If-Match` / `If-None-Match` は読みで `304`、書きと削除で
`412 PreconditionFailed`、存在しないキーへの `If-Match` 付き書き・削除は実サービス同様 `404 NoSuchKey` —
アトミックなので `If-None-Match: *` の 16 並列書き込みは勝者がちょうど 1 つ。`If-*-Since` 形式は読みのみ）、
条件付き削除（`delete_object(IfMatch=…)`、`*` は「存在すれば」。弱い `W/` タグは書き・削除を決して許可しない。
size / last-modified 形式はディレクトリバケット向けなので `501`）、`ExpectedBucketOwner` の強制（全バケットは
`123456789012` の所有。クエリでも `x-amz-` ヘッダでも、誤った値があれば `403 AccessDenied`）、リクエストチェックサムの検証（`400 BadDigest`、`x-amz-decoded-content-length` が
食い違えば `IncompleteBody`、終端チャンクの無い `aws-chunked` は `InvalidRequest`）、そして
2 種類のリスト（`GET ?list-type=2` と
`marker` を使う V1 の `GET`）で、いずれも `prefix` / `delimiter`（`CommonPrefixes`）/
`max-keys`（1000 超は拒否せずクランプ）/ `encoding-type=url` に対応（path-style、認証なし、
`aws-chunked` ボディはデコードする）。

このうち 2 つは実用上省けない。boto3 の `download_file` は `multipart_threshold`（8 MB）を
超えるオブジェクトを並列のレンジ GET に分割するし、botocore はすべてのリストで
`encoding-type=url` を要求する — 読み戻したキーを URL デコードするのはレスポンスが
`EncodingType` を echo したときだけなので、この 2 つは必ずセットで切り替える必要がある。

**それ以外は近似せず拒否する。** `CreateMultipartUpload`、`CopyObject`、オブジェクト /
バケットの tagging・ACL・バージョニング・ポリシー、`ListObjectVersions`、`DeleteObjects`、
その他のサブリソース、そしてこのストアが保存できない `put_object` のオプション（`Tagging`、
`Metadata`、SSE、ACL / grant、object lock）は S3 のエラー封筒で `501 NotImplemented` を返す。
`ContentType` などの素のエンティティヘッダは受け付けるが保存しない（読み戻すと
`binary/octet-stream`）。これは見た目より重要で、
`copy_object`・`put_object_tagging`・`put_object_acl` はいずれもオブジェクト自身のパスへの
`PUT` として届くため、サブリソースを無視するストアはそれらのボディ（あるいは空ボディ）を
オブジェクトに上書きして「成功」を返してしまう。presigned URL のクエリ認証はヘッダ認証と同様に
無視する。マルチパートアップロードについては、
置く入力を 8 MB 未満に収めるか `boto3.s3.transfer.TransferConfig` で下げること。拒否はボディを
読んだ後に行うので、keep-alive 接続がずれることはない — botocore の `Expect: 100-continue` 付き
PUT は、どんな応答よりも先に `100 Continue` を受け取る。ルータ自身が表現できないパス（キー中の
制御文字）もルーティング前に同じ扱いで拒否する。`STANDARD` 以外の `StorageClass` も拒否し
（ここにあるものは全部 STANDARD）、botocore が付けるリクエストチェックサム（`Content-MD5`、
`x-amz-checksum-crc32` / `sha1` / `sha256`、ヘッダ形式でも `aws-chunked` のトレーラでも）を検証して
不一致は `400 BadDigest` にするので、転送中に壊れたボディが黙って保存されることはない
（`crc32c` / `crc64nvme` は未検証のまま受理）。

オブジェクトは `PUPPETLLM_S3_ROOT`（既定: プロセスごとの一時ディレクトリ）配下に置き、
その外には決して書けない: バケット名は `s3://` URI の中も含めてどこでも検証し、`.` / `..` を含むキーは拒否
（キーの 1 セグメントはバックエンドのファイルシステムが受け付ける 255 バイトまで — S3 では合法な
フラットな 1024 バイトのキーはここでは置けない）、
書き込みには既存バケットが必要で、書き込みはバケットツリーの外の一時ディレクトリからの rename 経由なので
並行リーダーが書きかけを見ることはなく（ストア全体のロックがバッチエミュレーションのワーカースレッドも
覆うので、その出力書き込みが HTTP ハンドラの stat → 前提条件 → 書き込みの間に割り込むこともない。ハンドラは
このロックをイベントループの外で取るので、スレッドがロックを持っていても他の経路が凍ることはない）、
壊れた `aws-chunked` ボディは `400 InvalidRequest` になる。キー中の制御文字は拒否する
（このストアはキーをそのままリスト XML に書き出すので、そこでは不正な文字になる）。キーを黙って
書き換えることもしない — `PUT /bucket//a` は `a` への静かな書き込みではなく `400` になる。
API のパスと衝突するバケット名（`model`、`v1`、`anthropic`、`_control`、`model-invocation-job[s]`、
`docs`、`redoc`、`openapi.json`）は拒否し、それらのパスにメソッド違いでアクセスした場合は S3 の
エラー XML ではなく API 本来の `405` + `Allow` を返す。

---

## relay モード（クロスプロバイダブリッジ）

`python -m puppetllm.relay` は、全 pending を**実在の上流 API** に転送して応答を注入して返す**自動 responder**。puppetllm が透過的なクロスプロバイダ・ブリッジになる。アプリは自分の SDK を話し続けたまま、背後の実モデルを差し替えられる:

```bash
# Anthropic SDK のアプリを xAI Grok で動かす:
python -m puppetllm.relay --target https://api.x.ai/v1 \
    --api-key-env XAI_API_KEY --model grok-3

# OpenAI SDK のアプリを実 Claude で動かす:
python -m puppetllm.relay --kind anthropic --model claude-sonnet-4-5

# モデル単位のルーティング (単一強制でなく):
python -m puppetllm.relay --model-map "claude-*=grok-3,gpt-*=grok-3-mini"

# OpenAI モデルのリクエストだけ relay し、残りは手動で答える（モデルで分割した並行運用）:
python -m puppetllm.relay --only "gpt-*,o3-*" --model grok-3
```

- `--kind openai`（既定）は **OpenAI 互換の任意エンドポイント**に対応 — OpenAI / xAI Grok / Groq / Ollama / OpenRouter など、`--target` に base URL を向けるだけ。`--kind anthropic` は本家 Anthropic API。
- リクエストは canonical（system / messages / tools（`strict` と OpenAI の `custom` ツール込み）/ tool_choice / stop / temperature / seed / verbosity / prompt_cache_key / metadata 等。画像ブロック ↔ `image_url` パートは双方向に変換）から変換され、レスポンスは canonical blocks として戻る — 上流の実 signature 付き `thinking` ブロック、OpenAI の `refusal` はテキスト + `stop_reason: "refusal"` — **実の `stop_reason` / `stop_details` と実トークン usage** 込み（`/_control/respond` の `stop_reason` / `stop_details` / `usage` フィールドを使用。usage は `cache_creation` / `output_tokens_details` / `service_tier` のオブジェクトごと透過）。`/_control/stats` は実数値を集計する（history エントリに `"usage_overridden": true`）。上流が usage を返さない場合は puppetllm の概算を維持する。
- 実 Anthropic 上流へは、OpenAI 形式で入った `response_format: json_schema` を `output_config.format` に、`reasoning_effort` を `output_config.effort`（`none` / `minimal` → `low`）に変換する。Anthropic 形式で入った `thinking` / `output_config` / `cache_control` / `inference_geo` / `speed` はそのまま転送し、アプリの `anthropic-beta` ヘッダ（`params.anthropic_beta` として捕捉）も Anthropic 上流へ再送するので、ベータ機能（fast mode、compaction など）が relay 越しでも動く。ベンダー固有の語彙は転送せず変換する: `service_tier`（`standard_only` ↔ `default`、`flex` / `priority` → `auto`）、Anthropic の `effort: max` → OpenAI の `reasoning_effort: xhigh`。`n` は転送しない（relay は 1 choice しか使わず、余分なサンプルは課金されるだけ）。puppetllm 内部の canonical キー（`_openai_custom`、`_openai_detail`）はどの wire にも出ない。変換できないパラメータは 1 回だけ警告して落とす。
- `--max-tokens-param` の既定は `auto`: ターゲット URL のホスト名が正確に `api.openai.com` なら `max_completion_tokens`（そこでは `max_tokens` は非推奨で reasoning 系モデルに拒否される）、他の OpenAI 互換バックエンドには `max_tokens`。バックエンドが合わない場合は明示指定する。
- 上流 API のエラーは status/type/message（および `code`/`param`）ごと中継されるので、アプリの SDK は実プロバイダ相手と同じ例外クラスを送出する。
- relay は*あくまで responder の一種*。**既定では見えた pending を全て掴む**ため、人間 / AI エージェント responder とはライブなキューを共有しない（切替は逐次的: relay を止めて手動に引き継ぐ）。**並行**させたい場合は `--only "<glob>,…"` を使い、マッチする inbound モデルだけを掴ませて残りを人間（や別エージェント）に委ねる。
- `--max-concurrency N` で同時 in-flight な上流呼び出し数を上限化（既定: 無制限）。pending がバーストしても一気にファンアウトして上流のレート制限を踏まない。

注意: 上流呼び出しは非ストリーミングのため、ストリーミングアプリの SSE は正しく動くが最初のトークンまでの遅延が上流の完全応答時間になる。音声 / ファイルパートと `document` ブロックは未変換（画像は変換する）。このモードは**実課金**が発生する。`/_control/stats` の料金換算は*受信側* model id 基準なので、上流モデルの実際の価格とはずれうる。

---

## 制御 API（localhost のみ、認可なし）

| Method | Path | 説明 |
|---|---|---|
| GET  | `/_control/health` | ヘルスチェック（`{"ok","turn_count"}`） |
| GET  | `/_control/pending` | 保留中リクエスト一覧（`pending[]` + provider 込み、最古は `request` にも） |
| GET  | `/_control/wait_for_pending?timeout=N` | 次の pending を long-poll で待つ（既定 270s / 上限 600s。なければ `{"timeout":true}`） |
| POST | `/_control/respond` | 保留中リクエストに応答（`{"content":[...], "pending_id"?, "stop_reason"?, "stop_sequence"?, "stop_details"?, "usage"?}`）を注入。`content` のブロックは `text` / `tool_use` / `thinking` / `redacted_thinking`。`stop_reason` で自動判定を上書き（例 `"max_tokens"` — 打ち切り分岐のテスト用。OpenAI 経路では `finish_reason: "length"` に変換）。`"refusal"` なら `stop_details` を生成（`stop_details` で `category` / `explanation` を指定可。`recommended_model` などの追加フィールドは素通し）し、OpenAI 経路では OpenAI の refusal 形式（`message.refusal` / `delta.refusal`、`content: null`、`finish_reason: "stop"`）になる。`"stop_sequence"` なら `stop_sequence` を埋める。`usage` で概算トークンを実数値に上書き（`input_tokens` / `output_tokens` / `cache_creation_input_tokens` / `cache_read_input_tokens` の非空サブセット、`[0, 1e12]` の int。加えて `cache_creation` / `output_tokens_details` / `server_tool_use` のオブジェクトと `service_tier` / `inference_geo` / `speed` の文字列も任意 — relay モードが使用） |
| POST | `/_control/auto` | 簡易自動応答（`{"text":"...", "pending_id"?}`、text のみ） |
| POST | `/_control/error` | HTTP エラー応答を注入（`{"status","type","message", "code"?, "param"?, "headers"?, "pending_id"?, "after_events"?, "content"?}` — 後ろ 2 つはストリーミングを途中で失敗させる。§4 参照）。`headers`（文字列 → 文字列/数値）はエラー応答にそのまま付与される — 429 の `{"retry-after": 3}` や `anthropic-ratelimit-*` / `x-ratelimit-*` など、アプリのバックオフ処理の検証用（`content-length` / `transfer-encoding` などのフレーミング系ヘッダ、制御文字、非 Latin-1 の値は 400 で拒否）。Anthropic 経路のエラーボディには（Batches も含め）`request-id` ヘッダと一致する `request_id` が入り、OpenAI 経路は Anthropic 語彙の `type` を自分の語彙に変換する（`api_error` → status に応じて `server_error` 等） |
| GET  | `/_control/history` | (request, response, usage, cost, cache) 履歴 |
| GET  | `/_control/stats` | コスト目安・トークン・キャッシュの累計サマリ |
| GET  | `/_control/cache` | 擬似プロンプトキャッシュ index |
| POST | `/_control/clear` | pending / history / cache / batches / Bedrock バッチジョブを空に（in-flight はリトライ可能なエラーで解放: Anthropic 経路は 529 `overloaded_error`、Bedrock / OpenAI は 503、作成中の Bedrock バッチジョブはリトライ不能な 400 `ConflictException`。S3 ストアには触れない） |
| GET  | `/_control/batches` | バッチレジストリ（状態・request_counts・未解決 custom_id） |
| POST | `/_control/batch/result` | 1 つの custom_id に `canceled` / `expired` を注入（`{"custom_id","type","batch_id"?}`） |
| POST | `/_control/batch/end` | バッチを強制 `ended` に。未解決 custom_id は `expired`（既定）または `canceled` になる |
| GET  | `/_control/bedrock_jobs` | Bedrock バッチ推論のジョブレジストリ（状態・レコード数・未解決 `recordId`） |

`respond` / `auto` / `error` では、バッチのエントリを `pending_id` の代わりに `custom_id`（+ 任意で `batch_id`）で指定できる。

以前のバージョンから変わった観測可能な挙動（旧値を前提にしたアプリやテストハーネスは更新が必要）: 処理中に clear されたリクエストは Anthropic / Bedrock-Messages 経路で `529 overloaded_error`（旧 `503 api_error`）。OpenAI 経路のエラー `type` は OpenAI 語彙（`server_error`、`service_unavailable_error` 等。旧 `api_error` / `service_unavailable`）。canonical の `refusal` は OpenAI の `message.refusal` + `finish_reason: "stop"` になる（旧 `finish_reason: "content_filter"`。フィルタ形式が欲しければ `stop_reason` に `"content_filter"` を渡す）。Bedrock の pending / 応答は正規化した Anthropic モデル名を持つ。`thinking` ブロックは捨てずに保持する。OpenAI の usage は `n` 個の choice すべてを計上し（応答・history・stats で一致。該当する pending の snapshot には `choices` フィールドが載る）、明示された `service_tier` を返す。`inference_geo: "us"` のリクエストには公式の 1.1 倍を課金に適用する。

### 並列リクエスト（multi-pending）

server は同時複数リクエストを保持できる。各 pending は一意な `pending_id` を持ち、`/_control/respond`（`auto` / `error` も）に `pending_id` を指定して個別に注入する。

- `pending_id` 省略は pending が**ちょうど 1 件**のときのみ可。0 件 → `400`、複数 → `400`（`pending_ids` を返すので選んで指定）。
- 存在しなくなった pending（解決済み / `clear` で消えた等）への注入は `400`（`no pending request`）。ほぼ同時の二重注入レースのみ `409`（`already resolved`）。

注入ペイロードの組み立て方（特に日本語 + ネスト JSON の escape 事故回避）は [`responder/CLAUDE.md`](responder/CLAUDE.md) / [`responder/AGENTS.md`](responder/AGENTS.md) に詳しい。

---

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

---

## 環境変数

| 変数 | 既定 | 説明 |
|---|---|---|
| `PUPPETLLM_CACHE_TTL` | `300` | 5 分ブレークポイントの擬似キャッシュ TTL（秒） |
| `PUPPETLLM_CACHE_TTL_1H` | `PUPPETLLM_CACHE_TTL` × 12 | `ttl: "1h"` ブレークポイントの TTL（秒）。未設定なら 3600 で、5m 側の TTL に連動する |
| `PUPPETLLM_S3_ROOT` | プロセスごとの一時ディレクトリ | S3 エミュレーションがオブジェクトを置く場所（バッチ推論の入出力）。再起動をまたいで残したい場合や、Docker でホストから見たい場合に指定する（その場合は compose の override に環境変数と対応するボリュームを足すこと。同梱の `docker-compose.yml` はどちらも定義していない） |
| `PUPPETLLM_CACHE_HONOR_TTL` | `1` | `0` で TTL を無視（常に生存） |
| `PUPPETLLM_CACHE_MIN_TOKENS` | （モデル別） | 最小キャッシュ閾値の上書き。`0` で無効（全 prefix キャッシュ）。未設定は世代別テーブル（Opus 5 / Fable 512、Opus 4.8 1024、Opus 4.7 2048、Opus 4.6 / 4.5 4096、Sonnet 1024、Haiku 4.5 4096 …） |

---

## テスト

```bash
# Docker（test プロファイルは depends_on で `proxy` サービスも 127.0.0.1:8765 に起動するので、
# そのポートが空いている必要がある）
docker compose --profile test run --rm proxy-test

# または直接
pip install -r requirements.txt
python3 -m unittest puppetllm.tests.test_fake_server puppetllm.tests.test_proxy_extensions \
    puppetllm.tests.test_batches puppetllm.tests.test_conformance \
    puppetllm.tests.test_bedrock_extras puppetllm.tests.test_boto3_interop -v
```

`puppetllm/tests/test_fake_server.py` は期待挙動の executable specification。

`test_boto3_interop.py` だけが**実物の** boto3 / botocore（`>= 1.43`。このテストが使う全フィールドを
持つ最初のサービスモデル）を uvicorn インスタンスに向けて駆動する。他のテストは Bedrock の event stream を puppetllm 自身のコーデックで符号化・復号
しているので、エンコーダとデコーダが揃って間違っていても素通りしてしまう。実行には `boto3`
が要る（`requirements.txt` に入っている）。compose のテストプロファイルは
`PUPPETLLM_REQUIRE_SDK_TESTS=1` を立てるので、import できない状態で走らせると
SDK テストを全部スキップしたまま `OK` と黙って報告せず失敗する。

---

## レイアウト

```
puppetllm/
├── puppetllm/              # パッケージ本体
│   ├── fake_server.py      # canonical core + Anthropic /v1/messages + /_control/*
│   ├── batches.py          # Anthropic Message Batches 経路 + バッチ制御エンドポイント
│   ├── cache_sim.py        # 擬似プロンプトキャッシュ
│   ├── pricing.py          # 概算トークン + 料金
│   ├── relay.py            # relay responder (実 API へのクロスプロバイダ・ブリッジ)
│   ├── openai_wire.py      # OpenAI ↔ canonical の純粋変換（アダプタと relay が共用）
│   ├── providers/          # Bedrock（invoke / converse / batch）/ OpenAI アダプタ、S3、AWS event stream
│   │                       #   bedrock.py, converse.py, bedrock_batch.py, s3.py, openai.py, eventstream.py
│   └── tests/              # 単体テスト
├── responder/              # responder (LLM のフリをするエージェント) 向け指示書
│   ├── CLAUDE.md           #   Claude Code 用
│   └── AGENTS.md           #   Codex CLI など AGENTS.md 規約の agent 用
├── LICENSE
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ライセンス

[MIT License](LICENSE) — Copyright (c) 2026 Aetheria Labs
