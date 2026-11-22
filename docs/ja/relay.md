[English](../relay.md) | **日本語** · [← README](../../README.ja.md)

# relay モード（クロスプロバイダブリッジ）

`python -m puppetllm.relay` は、全 pending を**実在の上流 API** に転送して応答を注入して返す**自動 responder**。puppetllm が透過的なクロスプロバイダ・ブリッジになる。アプリは自分の SDK を話し続けたまま、背後の実モデルを差し替えられる:

```bash
# Anthropic SDK のアプリを xAI Grok で動かす:
python -m puppetllm.relay --target https://api.x.ai/v1 \
    --api-key-env XAI_API_KEY --model grok-4.7

# OpenAI SDK のアプリを実 Claude で動かす:
python -m puppetllm.relay --kind anthropic --model claude-sonnet-5-5

# モデル単位のルーティング (単一強制でなく):
python -m puppetllm.relay --model-map "claude-*=grok-4.7,gpt-*=grok-4.3"

# OpenAI モデルのリクエストだけ relay し、残りは手動で答える（モデルで分割した並行運用）:
python -m puppetllm.relay --only "gpt-*,o3-*" --model grok-4.7
```

- `--kind openai`（既定）は **OpenAI 互換の任意エンドポイント**に対応 — OpenAI / xAI Grok / Groq / Ollama / OpenRouter など、`--target` に base URL を向けるだけ。`--kind anthropic` は本家 Anthropic API。
- リクエストは canonical（system / messages / tools（`strict` と OpenAI の `custom` ツール込み）/ tool_choice / stop / temperature / seed / verbosity / prompt_cache_key / metadata 等。画像ブロック ↔ `image_url` パートは双方向に変換）から変換され、レスポンスは canonical blocks として戻る — 上流の実 signature 付き `thinking` ブロック、OpenAI の `refusal` はテキスト + `stop_reason: "refusal"` — **実の `stop_reason` / `stop_details` と実トークン usage** 込み（`/_control/respond` の `stop_reason` / `stop_details` / `usage` フィールドを使用。usage は `cache_creation` / `output_tokens_details` / `service_tier` のオブジェクトごと透過）。`/_control/stats` は実数値を集計する（history エントリに `"usage_overridden": true`）。上流が usage を返さない場合は puppetllm の概算を維持する。
- 実 Anthropic 上流へは、OpenAI 形式で入った `response_format: json_schema` を `output_config.format` に、`reasoning_effort` を `output_config.effort`（`none` / `minimal` → `low`）に、`parallel_tool_calls: false` を `tool_choice.disable_parallel_tool_use` に、`user` を `metadata.user_id` に変換する。Anthropic 形式で入った `thinking` / `output_config` / `cache_control` / `inference_geo` / `speed` / `mcp_servers` / `fallbacks` はそのまま転送し、アプリの `anthropic-beta` ヘッダ（`params.anthropic_beta` として捕捉）も Anthropic 上流へ再送するので、ベータ機能（fast mode、compaction など）が relay 越しでも動く。ベンダー固有の語彙は転送せず変換する: `service_tier`（`standard_only` ↔ `default`、`flex` / `priority` → `auto`）、Anthropic の `effort: max` → OpenAI の `reasoning_effort: xhigh`。`n` は転送しない（relay は 1 choice しか使わず、余分なサンプルは課金されるだけ）。puppetllm 内部の canonical キー（`_openai_custom`、`_openai_detail`）はどの wire にも出ない。変換できないパラメータは 1 回だけ警告して落とす。
- `--max-tokens-param` の既定は `auto`: ターゲット URL のホスト名が正確に `api.openai.com` なら `max_completion_tokens`（そこでは `max_tokens` は非推奨で reasoning 系モデルに拒否される）、他の OpenAI 互換バックエンドには `max_tokens`。バックエンドが合わない場合は明示指定する。
- 上流 API のエラーは status/type/message（および `code`/`param`）ごと中継されるので、アプリの SDK は実プロバイダ相手と同じ例外クラスを送出する。
- relay は*あくまで responder の一種*。**既定では見えた pending を全て掴む**ため、人間 / AI エージェント responder とはライブなキューを共有しない（切替は逐次的: relay を止めて手動に引き継ぐ）。**並行**させたい場合は `--only "<glob>,…"` を使い、マッチする inbound モデルだけを掴ませて残りを人間（や別エージェント）に委ねる。
- relay が答えた応答も、上流モデルを解決した後に relay が返した失敗も `provenance`（`upstream_provider`、解決できていれば `upstream_model`、上流が usage を返したなら `usage_source: upstream` それ以外は `estimate`、`translation.dropped_fields`）を history に残す。コントロールプレーンが relay のペイロードを拒否したときに送る 502 のフォールバックも、拒否された原因が provenance 自体でなければ provenance を保つ。`--price-by upstream` なら記録するコストを受信側 ID ではなく上流モデルの料金表で計算する（エントリの `pricing_model` がどちらかを示す。上流モデルに一致する内蔵の表がなければ relay が 1 回警告する — サーバの `--pricing-file` で足す）。`--strict-translation` は、上流向けに翻訳できないものがあるリクエストを、警告付きで落とす代わりに拒否する（アプリには 400、上流には送らない）: パラメータ（`speed`、`top_k`、`thinking`、`container`、`verbosity`、`n > 1`、未知の `service_tier` など）と、relay がプレースホルダに置き換えてしまう内容（`document` ブロック、未対応の画像ソース、OpenAI の `input_audio` / `file` パート、Anthropic のサーバサイドツール、相手の API にない形の `tool_choice`、`metadata.user_id` の上限を超える `user`）。既定を頼むだけの値（`frequency_penalty: 0`、`logprobs: false`、`logit_bias: {}`、`thinking` がそのモデルの既定モードで budget なし、`verbosity: medium` など）と、答えを変えないパラメータ（`cache_control`、`user_id` 以外の `metadata`、`anthropic_beta`、`diagnostics`、`store`、`prompt_cache_key`、`safety_identifier`、`stream_options`）は数えない。`--dry-run` は上流リクエストを翻訳してログに出す（長い文字列は切る）だけで送らない（アプリには 400）。
- Anthropic 側から来た `output_config.format`（`json_schema`）は OpenAI の `response_format` に翻訳する（スキーマと名前。`strict` は付けない — Anthropic のスキーマは OpenAI の strict の規則を満たすとは限らない）。Anthropic 向けの `response_format` → `output_config` の逆方向。
- `--max-concurrency N` で同時 in-flight な上流呼び出し数を上限化（既定: 無制限）。pending がバーストしても一気にファンアウトして上流のレート制限を踏まない。

注意: 上流呼び出しは非ストリーミングのため、ストリーミングアプリの SSE は正しく動くが最初のトークンまでの遅延が上流の完全応答時間になる。音声 / ファイルパートと `document` ブロックは未変換（画像は変換する）。このモードは**実課金**が発生する。既定では記録するコスト（したがって `/_control/stats`）は*受信側* model id 基準なので、上流モデルの実際の価格とはずれうる — `--price-by upstream` で上流モデルの表に切り替えられる。

実験的な 3 つ目の kind、`--kind agent-sdk` は、素の API 呼び出しの代わりに Claude Agent SDK 経由で pending に答える。専用のガイド: [Agent SDK 変換モード](agent-sdk.md)。
