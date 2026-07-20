**日本語** | [English](README.md)

# puppetllm — LLM API debug proxy (fake Anthropic / Bedrock / OpenAI server)

Anthropic Messages API / Bedrock / OpenAI Chat Completions 互換の **fake server**。`ANTHROPIC_BASE_URL`（または `AnthropicBedrock` / `OpenAI` の base_url）をこのサーバに向けるだけで、**アプリ / SDK のコードを 1 行も変えずに** LLM 呼び出しを横取りし、人間 or 別エージェントが応答を供給できる（human-in-the-loop / AI-in-the-loop）。

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
- `puppetllm/providers/bedrock.py` — Bedrock 経路 `POST /model/{id}/invoke[-with-response-stream]`（AWS event stream フレーミングは `providers/eventstream.py`）
- `puppetllm/providers/openai.py` — OpenAI 経路 `POST /v1/chat/completions`（リクエストは canonical（Anthropic 風）に正規化し、レスポンスは `chat.completion` JSON / SSE chunk に変換）
- `puppetllm/cache_sim.py` — 擬似プロンプトキャッシュ（multi-breakpoint + 前方一致 + モデル別最小閾値 + 20-block lookback）
- `puppetllm/pricing.py` — 概算トークン + 料金表（Claude / GPT 両ファミリ）

provider は **URL パスで自動判別**（モード切替・設定は不要）。応答 content blocks / 制御 API は provider 共通（注入は同じ `/_control/respond`）。

---

## 使い方

全体像は **3 つの登場人物**で考える:

```
  ┌─── アプリ / SDK ───┐         ┌───── puppetllm ──────┐        ┌── responder ──┐
  │ messages.create()  │ ──────▶ │ POST /v1/messages    │ ─────▶ │ 応答を注入    │
  │ (応答までブロック) │ ◀────── │ (pending として保留) │ ◀───── │ /_control/... │
  └────────────────────┘  応答   └──────────────────────┘        └───────────────┘
         ①アプリ                   ②fake server (本体)           ③供給側 (人 or AI)
```

①が投げたリクエストを②が**保留 (pending)** し、③が `/_control/*` で応答を流し込むと、①の `create()` がその応答で返る。実 API は一切叩かない。

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
- リクエストは canonical（system / messages / tools / tool_choice / stop / temperature 等）から変換され、レスポンスは**実の `stop_reason` と実トークン usage** 込みで canonical blocks として戻る（`/_control/respond` の `stop_reason` / `usage` フィールドを使用）。`/_control/stats` は実数値を集計する（history エントリに `"usage_overridden": true`）。上流が usage を返さない場合は puppetllm の概算を維持する。
- `max_tokens` を拒否する OpenAI reasoning / 公式 gpt-5 系エンドポイント向けには `--max-tokens-param max_completion_tokens` を指定する。
- 上流 API のエラーは status/type/message（および `code`/`param`）ごと中継されるので、アプリの SDK は実プロバイダ相手と同じ例外クラスを送出する。
- relay は*あくまで responder の一種*。**既定では見えた pending を全て掴む**ため、人間 / AI エージェント responder とはライブなキューを共有しない（切替は逐次的: relay を止めて手動に引き継ぐ）。**並行**させたい場合は `--only "<glob>,…"` を使い、マッチする inbound モデルだけを掴ませて残りを人間（や別エージェント）に委ねる。
- `--max-concurrency N` で同時 in-flight な上流呼び出し数を上限化（既定: 無制限）。pending がバーストしても一気にファンアウトして上流のレート制限を踏まない。

注意: 上流呼び出しは非ストリーミングのため、ストリーミングアプリの SSE は正しく動くが最初のトークンまでの遅延が上流の完全応答時間になる。マルチモーダル（画像）ブロックは未変換。このモードは**実課金**が発生する。`/_control/stats` の料金換算は*受信側* model id 基準なので、上流モデルの実際の価格とはずれうる。

---

## 制御 API（localhost のみ、認可なし）

| Method | Path | 説明 |
|---|---|---|
| GET  | `/_control/health` | ヘルスチェック（`{"ok","turn_count"}`） |
| GET  | `/_control/pending` | 保留中リクエスト一覧（`pending[]` + provider 込み、最古は `request` にも） |
| GET  | `/_control/wait_for_pending?timeout=N` | 次の pending を long-poll で待つ（既定 270s / 上限 600s。なければ `{"timeout":true}`） |
| POST | `/_control/respond` | 保留中リクエストに応答（`{"content":[...], "pending_id"?, "stop_reason"?, "usage"?}`）を注入。`stop_reason` で自動判定を上書き（例 `"max_tokens"` — 打ち切り分岐のテスト用。OpenAI 経路では `finish_reason: "length"` に変換）。`usage` で概算トークンを実数値に上書き（`input_tokens` / `output_tokens` / `cache_creation_input_tokens` / `cache_read_input_tokens` の非空サブセット、`[0, 1e12]` の int — relay モードが使用） |
| POST | `/_control/auto` | 簡易自動応答（`{"text":"...", "pending_id"?}`、text のみ） |
| POST | `/_control/error` | HTTP エラー応答を注入（`{"status","type","message", "code"?, "param"?, "pending_id"?}`） |
| GET  | `/_control/history` | (request, response, usage, cost, cache) 履歴 |
| GET  | `/_control/stats` | コスト目安・トークン・キャッシュの累計サマリ |
| GET  | `/_control/cache` | 擬似プロンプトキャッシュ index |
| POST | `/_control/clear` | pending / history / cache を空に（in-flight は 503 で解放） |

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
- あくまでローカルデバッグ用途。本番の前段に置くものではない。

---

## 環境変数

| 変数 | 既定 | 説明 |
|---|---|---|
| `PUPPETLLM_CACHE_TTL` | `300` | 擬似キャッシュ TTL（秒） |
| `PUPPETLLM_CACHE_HONOR_TTL` | `1` | `0` で TTL を無視（常に生存） |
| `PUPPETLLM_CACHE_MIN_TOKENS` | （モデル別） | 最小キャッシュ閾値の上書き。`0` で無効（全 prefix キャッシュ）。未設定は Opus 4096 / Sonnet 1024 / Haiku 4096 |

---

## テスト

```bash
# Docker
docker compose --profile test run --rm proxy-test

# または直接
pip install -r requirements.txt
python3 -m unittest puppetllm.tests.test_fake_server puppetllm.tests.test_proxy_extensions -v
```

`puppetllm/tests/test_fake_server.py` は期待挙動の executable specification。

---

## レイアウト

```
puppetllm/
├── puppetllm/              # パッケージ本体
│   ├── fake_server.py      # canonical core + Anthropic /v1/messages + /_control/*
│   ├── cache_sim.py        # 擬似プロンプトキャッシュ
│   ├── pricing.py          # 概算トークン + 料金
│   ├── relay.py            # relay responder (実 API へのクロスプロバイダ・ブリッジ)
│   ├── providers/          # Bedrock / OpenAI アダプタ + AWS event stream
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
