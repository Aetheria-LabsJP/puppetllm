[English](README.md) | **日本語**

# puppetllm — LLM API デバッグプロキシ（Anthropic / Bedrock / OpenAI 互換の fake サーバ）

Anthropic Messages API、Bedrock（InvokeModel、Converse、S3 エミュレーション同梱のバッチ推論）、OpenAI Chat Completions と互換の **fake サーバ**。`ANTHROPIC_BASE_URL`（または `AnthropicBedrock` / `OpenAI` の `base_url`）を向けるだけで、アプリの LLM 呼び出しを**保留**し、人間・別のエージェント・台本ルール・実プロバイダへの relay のいずれかが答えを供給する。アプリのコードは 1 行も変えず、実 API も呼ばない。

- **コストゼロのデバッグ** — エージェントの挙動を再現・観察する。人間や AI エージェントがモデル役を演じる。
- **決定的なテスト** — ルールがリクエストに答え（テキスト / tool_use / エラー / 429 の連続）、レイテンシとレートリミットを模擬し、pytest フィクスチャがアプリの送信内容を検証する。
- **クロスプロバイダブリッジ** — Anthropic SDK のアプリを GPT / Grok へ、OpenAI SDK のアプリを Claude へ relay する。
- **観測性** — 全やり取りを history API に、トークン・概算コスト・キャッシュ hit / miss の理由・JSON ログ・Prometheus メトリクスとともに。

> トークン数と料金は概算トークナイザと公開価格表によるもので、**実請求とは一致しない**。傾向と構造の確認に使うこと。

## クイックスタート: ターミナル 2 枚

```bash
# ターミナル 1 — fake サーバ
docker compose up -d                       # または: pip install . && puppetllm serve
curl -s localhost:8765/_control/health     # → {"ok":true,"turn_count":0}
```

```python
# ターミナル 1 — アプリ。base_url 以外はそのまま（答えが注入されるまでブロックする）
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8765", api_key="sk-mock")
msg = client.messages.create(model="claude-sonnet-4-5", max_tokens=256,
                             messages=[{"role": "user", "content": "東京の天気は？"}])
print(msg.content[0].text, msg.usage)
```

```bash
# ターミナル 2 — responder: 保留中のリクエストを見て、答える
curl -s 'localhost:8765/_control/pending?summary=1' | jq '.pending[0].request'
curl -s localhost:8765/_control/respond -d '{"text": "晴れ、23 ℃。"}'
curl -s localhost:8765/_control/history/1 | jq '{usage, cost}'
```

`AnthropicBedrock`、boto3（`invoke_model`、`converse`、バッチ推論）、OpenAI SDK（`base_url=http://localhost:8765/v1`）でも、ストリーミング込みで同じように動く。人間なしで答えさせるなら、サーバに台本を渡す:

```python
from puppetllm.testing import serve

with serve() as puppet:                      # 空きポートでプロセス内にサーバを起こす
    puppet.expect(tools=["get_weather"]).respond(
        content=[{"type": "tool_use", "name": "get_weather", "input": {"city": "Tokyo"}}])
    puppet.expect(has_tool_result=True).respond(text="東京は 23 ℃ で晴れです。")
    ...                                       # puppet.url に向けてエージェントループを回す
    puppet.assert_consumed()
    puppet.assert_sent(count=2, tool="get_weather")
```

`examples/tool_loop.py` がこのループの完走例、`examples/test_retry.py` は `puppet` pytest フィクスチャで SDK の 429 リトライを駆動する例。

## ドキュメント

| ガイド | 内容 |
|---|---|
| [クイックスタート](docs/ja/quickstart.md) | サーバの起動、各 SDK の向け先、手動での応答、エラー注入、並列リクエスト |
| [テストガイド](docs/ja/testing.md) | シナリオルール、未マッチ時ポリシー、タイムアウト、レイテンシ・レートリミット模擬、擬似クロック、`count_tokens` / `models`、`puppetllm.testing` と pytest フィクスチャ |
| [制御 API リファレンス](docs/ja/control-api.md) | 全 `/_control/*` エンドポイント、history / メトリクス / ログ、キャッシュ診断、料金ファイル、環境変数、セキュリティ、挙動変更 |
| [Bedrock](docs/ja/bedrock.md) | InvokeModel、Converse、guardrail とプロンプト管理のフィクスチャ、S3 エミュレーション同梱のバッチ推論 |
| [Message Batches API](docs/ja/batches.md) | Anthropic バッチのライフサイクルと操作 |
| [relay モード](docs/ja/relay.md) | SDK をまたいで pending を実 Anthropic / OpenAI 互換の上流へ転送する |
| [Agent SDK 変換モード](docs/ja/agent-sdk.md) | 実験的: Claude Agent SDK 経由で pending に答える relay。Docker サービス、状態監視、安全装置 |
| [設計ノート](docs/ja/design.md) | アーキテクチャ、互換性マトリクス、テスト、レイアウト |

responder（モデル役を演じる Claude Code / Codex）向けの指示は [`responder/`](responder/) にある。

## インストール

```bash
docker compose up -d                 # イメージ puppetllm:latest、localhost:8765
pip install .                        # `puppetllm serve|relay|wait` と pytest プラグイン
pip install ".[test]"                # + 相互運用テスト用の anthropic / openai / boto3
docker compose --profile test run --rm proxy-test   # テストスイート
```

Python 3.10 以上。サーバは既定で `127.0.0.1` に bind し、制御プレーンに認証はない（[セキュリティ](docs/ja/control-api.md#セキュリティ) を参照）。

## ライセンス

MIT — Copyright (c) 2026 Aetheria Labs。[LICENSE](LICENSE) を参照。
