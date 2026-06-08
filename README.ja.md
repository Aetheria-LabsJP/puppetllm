<!-- Language: **日本語** | [English](README.md) -->

# puppetllm — Claude API debug proxy (fake Anthropic / Bedrock server)

Anthropic Messages API / Bedrock 互換の **fake server**。`ANTHROPIC_BASE_URL`（または `AnthropicBedrock` の base_url）をこのサーバに向けるだけで、**アプリ / SDK のコードを 1 行も変えずに** LLM 呼び出しを横取りし、人間 or 別エージェントが応答を供給できる（human-in-the-loop / AI-in-the-loop）。

用途:

- **ゼロ円デバッグ**: 実 API を叩かずにエージェント / オーケストレーションの挙動を再現・検証する
- **決定論的テスト**: 任意のレスポンス（text / tool_use、エラー）を注入して分岐を再現する
- **コスト目安**: リクエストごとの概算トークン / 料金を集計する（`/_control/stats`）
- **擬似プロンプトキャッシュ観測**: アプリが `cache_control` を効かせられる構造で投げているかをハッシュで観測する（`/_control/cache`）

> 概算（approx tokenizer）ベースなので**実課金とは一致しない**。傾向把握・構造検証用。

---

## アーキテクチャ

provider 非依存の canonical core + アダプタ:

- `puppetllm/fake_server.py` — canonical core（正規化 snapshot 管理 + `/_control/*` + cost/cache 計算）。Anthropic 経路 `POST /v1/messages` を内蔵
- `puppetllm/providers/bedrock.py` — Bedrock 経路 `POST /model/{id}/invoke[-with-response-stream]`（AWS event stream フレーミングは `providers/eventstream.py`）
- `puppetllm/cache_sim.py` — 擬似プロンプトキャッシュ（multi-breakpoint + 前方一致 + モデル別最小閾値 + 20-block lookback）
- `puppetllm/pricing.py` — 概算トークン + 料金表

応答 content blocks / 制御 API は provider 共通（注入は同じ `/_control/respond`）。

---

## 使い方

全体像は **3 つの登場人物**で考える:

```
  ┌── アプリ / SDK ──┐         ┌──── puppetllm ────┐        ┌── responder ──┐
  │ messages.create()│ ──────▶ │ POST /v1/messages │ ─────▶ │ 応答を注入     │
  │ (応答までブロック)│ ◀────── │ (pending として保留)│ ◀───── │ /_control/...  │
  └──────────────────┘  応答    └───────────────────┘        └────────────────┘
       ①アプリ                    ②fake server (本体)              ③供給側 (人 or AI)
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
    model="claude-opus-4-20250514", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
print(msg.content)          # → 注入された content blocks
print(msg.usage)            # → 概算 input/output トークン + キャッシュ
```

API key はダミーで良い（proxy は検証しない）。`base_url` の代わりに環境変数 `ANTHROPIC_BASE_URL=http://localhost:8765` を立てても同じ（コードを触らずに横取りできる）。`stream=True` の SSE もそのまま動く。

**Bedrock SDK (`AnthropicBedrock`):**

```python
from anthropic import AnthropicBedrock
client = AnthropicBedrock(base_url="http://localhost:8765", aws_region="us-east-1")
msg = client.messages.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

model は URL パス (`/model/{id}/invoke`) に入り、streaming は AWS event stream で返る — どちらも server が吸収する。**応答の注入方法は Anthropic 経路と完全に同じ**（下記 `/_control/respond` をそのまま使う）。

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

人間ではなく **AI エージェントに「LLM のフリ」をさせて忠実に応答させる**運用が強力。その際のエージェント向け指示書が [`responder/CLAUDE.md`](responder/CLAUDE.md)（Claude Code 用）/ [`responder/AGENTS.md`](responder/AGENTS.md)（Codex CLI など `AGENTS.md` 規約の agent 用）。どちらも中立に保つための核心原則・複数 pending 対応・注入フォーマット・禁忌・JSON escape の罠を網羅（内容はほぼ同じで、ランタイム前提だけ差分）。

### 4. エラー応答を注入してハンドリングを試す

分岐テスト用に、任意の HTTP エラーを pending に返させられる（Anthropic / Bedrock 両経路で各形式に変換される）:

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
#     "cache":{"hits":2,"misses":1,"hit_rate":0.67,"index_size":2},
#     "by_model":{"claude-opus-4-20250514":{"requests":3,"total_usd":..}}}

# 擬似プロンプトキャッシュの index (prefix hash 別 hit/miss)
curl -s localhost:8765/_control/cache | jq

# 1 リクエストずつの (request, response, usage, cost, cache) 履歴
curl -s localhost:8765/_control/history | jq '.history[-1]'

# テスト間のクリーンアップ (pending / history / cache を全消去)
curl -s -X POST localhost:8765/_control/clear
```

`cache_savings_usd` は「キャッシュが効いた分、本物なら浮いたであろう概算額」。アプリが `cache_control` を正しい構造で投げられているかの検証に使う。

---

## 制御 API（localhost のみ、認可なし）

| Method | Path | 説明 |
|---|---|---|
| GET  | `/_control/health` | ヘルスチェック（`{"ok","turn_count"}`） |
| GET  | `/_control/pending` | 保留中リクエスト一覧（`pending[]` + provider 込み、最古は `request` にも） |
| GET  | `/_control/wait_for_pending?timeout=N` | 次の pending を long-poll で待つ（既定 270s / 上限 600s。なければ `{"timeout":true}`） |
| POST | `/_control/respond` | 保留中リクエストに応答（`{"content":[...], "pending_id"?}`）を注入 |
| POST | `/_control/auto` | 簡易自動応答（`{"text":"...", "pending_id"?}`、text のみ） |
| POST | `/_control/error` | HTTP エラー応答を注入（`{"status","type","message", "pending_id"?}`） |
| GET  | `/_control/history` | (request, response, usage, cost, cache) 履歴 |
| GET  | `/_control/stats` | コスト目安・トークン・キャッシュの累計サマリ |
| GET  | `/_control/cache` | 擬似プロンプトキャッシュ index |
| POST | `/_control/clear` | pending / history / cache を空に（in-flight は 503 で解放） |

### 並列リクエスト（multi-pending）

server は同時複数リクエストを保持できる。各 pending は一意な `pending_id` を持ち、`/_control/respond`（`auto` / `error` も）に `pending_id` を指定して個別に注入する。

- `pending_id` 省略は pending が**ちょうど 1 件**のときのみ可。0 件 → `400`、複数 → `400`（`pending_ids` を返すので選んで指定）。
- 既に解決済み（`clear` 等）の pending へ注入すると `409`。

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
| `PUPPETLLM_CACHE_MIN_TOKENS` | （モデル別） | 最小キャッシュ閾値の上書き。`0` で無効（全 prefix キャッシュ）。未設定は Opus 4096 / Sonnet 1024 / Haiku 2048 |

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
│   ├── providers/          # Bedrock アダプタ + AWS event stream
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
