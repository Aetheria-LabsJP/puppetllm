[English](../design.md) | **日本語** · [← README](../../README.ja.md)

# 設計ノート

## アーキテクチャ

provider 非依存の canonical core + アダプタ:

- `puppetllm/fake_server.py` — canonical core（正規化 snapshot 管理 + `/_control/*` + cost/cache 計算）。Anthropic 経路 `POST /v1/messages` を内蔵
- `puppetllm/providers/bedrock.py` — Bedrock 経路 `POST /model/{id}/invoke[-with-response-stream]`（モデル ID 正規化・`anthropic_version` 検証・AWS 形式のエラー / ヘッダ。AWS event stream フレーミングは `providers/eventstream.py`）
- `puppetllm/providers/converse.py` — Bedrock Converse 経路 `POST /model/{id}/converse[-stream]`（Converse の JSON スキーマと canonical の相互変換）
- `puppetllm/providers/bedrock_batch.py` + `puppetllm/providers/s3.py` — Bedrock バッチ推論（`/model-invocation-job*`）と、同梱のディレクトリ実装 S3 エミュレーション
- `puppetllm/providers/openai.py` — OpenAI 経路 `POST /v1/chat/completions`（リクエストは canonical（Anthropic 風）に正規化し、レスポンスは `chat.completion` JSON / SSE chunk に変換）
- `puppetllm/batches.py` — Anthropic Message Batches 経路 `/v1/messages/batches*`（各 custom_id を通常の pending として保持。バッチのライフサイクルは `/_control/batch/*` から注入可能）
- `puppetllm/cache_sim.py` — 擬似プロンプトキャッシュ（multi-breakpoint + トップレベル `cache_control` の自動キャッシュ + 前方一致 + 世代別最小閾値 + 5m / 1h TTL + 20-block lookback + effort / thinking / tool_choice による無効化）
- `puppetllm/pricing.py` — 概算トークン + 料金表（Claude の世代別、GPT / o 系・Grok ファミリ、ロングコンテキスト料金つき。公式料金ページに準拠）
- `puppetllm/catalog.py` — `/v1/models` のカタログ（ID、上限、ライフサイクル、capabilities）
- `puppetllm/harness.py` — テストハーネス層: シナリオルール・未マッチ時ポリシー・pending タイムアウト・レイテンシ / ジッタ・レートリミット窓・擬似クロック。`puppetllm/control_models.py` は `/openapi.json` に載せる `/_control/*` の型付きリクエストボディ
- `puppetllm/testing.py` + `puppetllm/pytest_plugin.py` — テスト側クライアント（`Puppet`、`serve()`）と pytest の `puppet` フィクスチャ

provider は **URL パスで自動判別**（モード切替・設定は不要）。応答 content blocks / 制御 API は provider 共通（注入は同じ `/_control/respond`）。

## 互換性マトリクス

各経路が何を受け付け、responder / ルールが答えられるか（injectable）、ストリーミング形式を返すか、relay が転送するか。`GET /_control/capabilities` が同じデータ（+ content ブロック型ごとの経路での扱い）を返す。

| プロバイダ | 操作 | パス | 受付 | 注入 | ストリーム | relay |
|---|---|---|---|---|---|---|
| Anthropic | Messages | `POST /v1/messages` | ○ | ○ | ○（SSE） | ○ |
| Anthropic | count_tokens | `POST /v1/messages/count_tokens` | ○ | 直接応答 | – | – |
| Anthropic | Message Batches | `/v1/messages/batches*` | ○ | ○（custom_id ごと） | – | ○ |
| Anthropic | Models | `GET /v1/models[/{id}]` | ○ | カタログ | – | – |
| Bedrock | InvokeModel | `POST /model/{id}/invoke` | ○ | ○ | – | ○ |
| Bedrock | InvokeModelWithResponseStream | `POST /model/{id}/invoke-with-response-stream` | ○ | ○ | ○（event stream） | ○ |
| Bedrock | Converse / ConverseStream | `POST /model/{id}/converse[-stream]` | ○ | ○ | ○ | ○ |
| Bedrock | CountTokens | `POST /model/{id}/count-tokens` | ○ | 直接応答 | – | – |
| Bedrock | Messages（Mantle） | `POST /anthropic/v1/messages` | ○ | ○ | ○ | ○ |
| Bedrock | バッチ推論 | `/model-invocation-job*` | ○ | ○（レコードごと） | – | ○ |
| S3 | バケット / オブジェクト（部分集合） | `/{bucket}[/{key}]` | ○ | – | – | – |
| OpenAI | Chat Completions | `POST /v1/chat/completions`（エイリアス `/chat/completions`） | ○ | ○ | ○（SSE） | ○ |
| OpenAI | Models | `GET /v1/models[/{id}]`（エイリアス `/models`） | ○ | カタログ | – | – |

意図的に未実装（呼び出し側の形式の 404 で、メソッドとパス、以下については操作名も明示して拒否）: Anthropic の Text Completions・Files・Skills・Managed Agents・Admin、OpenAI の Responses・Embeddings・Completions・Assistants / vector stores / fine-tuning / audio / images / Realtime、Bedrock の ApplyGuardrail・async invoke・双方向ストリーム、その他すべての `/model/{id}/*` 操作。実在パスへの誤ったメソッドは `Allow` 付きの 405。content ブロック: `text` / `tool_use` は全経路、`thinking` / `redacted_thinking` は Anthropic・Bedrock・Converse 経路（OpenAI は捨てる）、サーバー側ツールのブロックは Anthropic・Bedrock InvokeModel・Mantle 経路（Converse と OpenAI は捨て、課金もしない）。

## テスト

```bash
# Docker（test プロファイルは depends_on で `proxy` サービスも 127.0.0.1:8765 に起動するので、
# そのポートが空いている必要がある）
docker compose --profile test run --rm proxy-test

# または直接
pip install -r requirements.txt
python3 -m unittest puppetllm.tests.test_fake_server puppetllm.tests.test_proxy_extensions \
    puppetllm.tests.test_batches puppetllm.tests.test_conformance \
    puppetllm.tests.test_bedrock_extras puppetllm.tests.test_harness \
    puppetllm.tests.test_compat puppetllm.tests.test_observability \
    puppetllm.tests.test_extras \
    puppetllm.tests.test_boto3_interop -v
```

`puppetllm/tests/test_fake_server.py` は期待挙動の executable specification。`test_harness.py` は [テストガイド](testing.md)（ルール、ポリシー、タイムアウト、レイテンシ、レートリミット、クロック、`count_tokens` / `models`、CLI、インプロセス uvicorn に対する `puppetllm.testing`）を検証する。

`test_boto3_interop.py` だけが**実物の** boto3 / botocore（`>= 1.43`。このテストが使う全フィールドを
持つ最初のサービスモデル）を uvicorn インスタンスに向けて駆動する。他のテストは Bedrock の event stream を puppetllm 自身のコーデックで符号化・復号
しているので、エンコーダとデコーダが揃って間違っていても素通りしてしまう。実行には `boto3`
が要る（`requirements.txt` に入っている）。compose のテストプロファイルは
`PUPPETLLM_REQUIRE_SDK_TESTS=1` を立てるので、import できない状態で走らせると
SDK テストを全部スキップしたまま `OK` と黙って報告せず失敗する。

Agent SDK relay には専用のモジュール（`test_agent_sdk.py`、`test_sdk_*.py`）があり、`puppetllm:agent-sdk` イメージで実行する。[Agent SDK 変換モード](agent-sdk.md) を参照。

## レイアウト

```
puppetllm/
├── puppetllm/              # パッケージ本体
│   ├── fake_server.py      # canonical core + Anthropic /v1/messages + /_control/* + count_tokens / models
│   ├── harness.py          # シナリオルール・未マッチ時ポリシー・レイテンシ・レートリミット・擬似クロック
│   ├── control_models.py   # /_control/* の型付きリクエストボディ（/openapi.json に載る）
│   ├── catalog.py          # /v1/models のカタログ
│   ├── capabilities.py     # 互換性マトリクス（/_control/capabilities）+ 未実装パスのキャッチオール
│   ├── __main__.py         # `puppetllm serve|relay|wait` コマンドライン
│   ├── testing.py          # テスト側クライアント（Puppet、serve()）
│   ├── pytest_plugin.py    # `puppet` pytest フィクスチャ
│   ├── observability.py    # JSON ログイベント + Prometheus メトリクス（/_control/metrics）
│   ├── _responses.py       # 全ルートが使う JSON 応答クラス（surrogate 安全）
│   ├── batches.py          # Anthropic Message Batches 経路 + バッチ制御エンドポイント
│   ├── cache_sim.py        # 擬似プロンプトキャッシュ
│   ├── pricing.py          # 概算トークン + 料金
│   ├── relay.py            # relay responder (実 API へのクロスプロバイダ・ブリッジ)
│   ├── agent_sdk.py        # 実験的な relay kind: Claude Agent SDK 経由で答える
│   │                       #   sdk_transport.py, sdk_egress.py, sdk_guard.py, sdk_runtime.py, sdk_state.py
│   ├── openai_wire.py      # OpenAI ↔ canonical の純粋変換（アダプタと relay が共用）
│   ├── providers/          # Bedrock（invoke / converse / batch）/ OpenAI アダプタ、S3、AWS event stream
│   │                       #   bedrock.py, converse.py, bedrock_batch.py, s3.py, openai.py, eventstream.py
│   └── tests/              # 単体テスト
├── docs/                   # ガイド（docs/ja/ が日本語）: quickstart、testing、control-api、bedrock、batches、relay、agent-sdk、design
├── examples/               # tool_loop.py（台本付きツールループ）、test_retry.py（pytest フィクスチャ）
├── responder/              # responder (LLM のフリをするエージェント) 向け指示書
│   ├── CLAUDE.md           #   Claude Code 用
│   └── AGENTS.md           #   Codex CLI など AGENTS.md 規約の agent 用
├── LICENSE
├── Dockerfile
├── docker-compose.yml      # `proxy`、`test` プロファイル、任意の `agent-sdk` プロファイル
├── start-agent-sdk.sh      # サーバ・Agent SDK relay・監視をビルドして起動
├── .env.example            # Agent SDK relay の設定（.env にコピーして使う）
├── pyproject.toml          # `pip install .` → `puppetllm` コマンド + pytest プラグイン
└── requirements.txt
```
