[English](../agent-sdk.md) | **日本語** · [← README](../../README.ja.md)

# Agent SDK 変換モード（実験的）

`relay --kind agent-sdk` は、Claude Code の接続先を puppetllm に向けたまま、
受信した `system / messages / tools` を **Claude Agent SDK の入力へ組み直す**。
SDK の構造化出力から次の `text / tool_use` を生成し、Messages API 応答として返す。
ツールの実行・承認と会話履歴の管理は手元の Claude Code が担当する。

```text
Claude Code → puppetllm /v1/messages → Agent SDK query() → Claude
Claude Code ← text / tool_use + SSE ← 構造化出力の検証・変換
```

## Docker Compose で起動（ホストの venv 不要）

上流の実際の `ANTHROPIC_API_KEY` を起動シェルに設定するか、
[`.env.example`](../../.env.example) を `.env` にコピーして記入する。
手動起動中のサーバーと relay から切り替える場合は、先にそれらを停止してポート8765を空ける。

```bash
# このリポジトリで実行。サーバー・relay・監視サービスをバックグラウンド起動
./start-agent-sdk.sh

# 状態・ログの確認
docker compose --profile agent-sdk ps
docker compose logs -f agent-sdk-relay
```

`./start-agent-sdk.sh --model claude-sonnet-4-6 --effort medium` で、その起動のモデルとeffortを変えられる
（指定の形式は `--help` を参照）。

スクリプトはイメージをビルドし、既存サーバーを維持してworkerと監視サービスを起動する。
既定は Opus 5.5、effort `high`、並列数4、タイムアウト600秒。
`.env` の `PUPPETLLM_AGENT_SDK_MODEL` / `PUPPETLLM_AGENT_SDK_CONCURRENCY` /
`PUPPETLLM_AGENT_SDK_TIMEOUT` / `PUPPETLLM_AGENT_SDK_TARGET` で変更できる。
それ以外の `PUPPETLLM_AGENT_SDK_*` 変数（実行回数、再試行、無進捗警告など）は `.env.example` にある。
思考量は `PUPPETLLM_AGENT_SDK_EFFORT`、出力トークン上限は `PUPPETLLM_AGENT_SDK_MAX_OUTPUT_TOKENS` で指定できる。例：

```bash
PUPPETLLM_AGENT_SDK_MAX_OUTPUT_TOKENS=32768 ./start-agent-sdk.sh --effort low
```

低いeffortは思考を減らすため、複雑な処理の品質との調整が必要。上限が小さすぎると応答が完成せずエラーになる。
APIが `max_tokens` で終了した場合は、SDK出力上限に達したことを示すエラーを返し、
途中のツール宣言を外側へ渡さない。`/status` の `observed_stop_reason` でも終了理由を確認できる。
`--model` / `--effort` 引数は対応する環境変数や `.env` の設定より優先する。
引数はその起動だけに適用され、`.env` を書き換えない。継続的な既定値は `.env` に保存する。
通常の `docker compose up -d` はサーバーだけを起動する。

Claude Code はホストの作業対象ディレクトリから使う。
サーバー・relay 用のターミナルを開き続ける必要はなく、作業ファイルのマウントも不要。

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8765 \
ANTHROPIC_API_KEY=puppet-local \
claude --model claude-opus-5-5
```

コード修正後は、同じ引数で `./start-agent-sdk.sh` をもう一度実行する。サーバーと履歴を保ったまま、relayと監視サービスをビルドし直して入れ替える。

全体を停止する場合は `docker compose --profile agent-sdk down`。
コンテナのサーバー履歴はメモリー上にあり、サーバーの再作成でリセットされる。

## 状態確認とタイムアウト後の再開

```bash
# 監視サービスの生存、workerの受付可否、進捗の順に確認
curl -sS http://127.0.0.1:8766/health
curl -sS http://127.0.0.1:8766/ready
curl -sS http://127.0.0.1:8766/status

# サーバーと監視を維持してSDK実行だけ止める
docker compose stop agent-sdk-relay
```

`/health` は監視プロセスの生存確認。`/ready` はworkerのheartbeatが新しく受付可能なら200、
停止中・状態保存の異常・heartbeat期限切れなら503になる。`/status` は開始・最後のSDKイベント・
生成イベントの時刻と件数、SDKが報告した使用量・推定費用などを返す。
構造化出力の再試行回数など、
SDKから確認できない値は `null`。`api_attempt_count` は送信ゲートが実際に上流へ送り出したMessages要求の回数、
`api_requests_blocked` は送信前に遮断した回数。回数・推定トークン・費用はSDK側の値で、実請求とは異なる。
本文・思考・署名・ツール引数・認証情報は進捗の記録に含めない。

Opus 5.5は思考の文字列を返さず、数値の推定トークンだけ更新する場合がある。
その場合は `stage: thinking`、`estimated_thinking_tokens`、`last_thinking_progress_at` を確認する。
`generated_characters: 0` でも思考の推定値が増えていれば、文字を伴わない進捗を観測できている。

タイムアウト・SDK失敗・キャンセル・応答の配信が確認できない場合、同じ有効入力の実行を
永続的にブロックする。別のリクエストIDで再送されても、`x-should-retry: false` を無視されてもSDKを起動しない。
指紋と固定の失敗理由をDocker volume `agent-sdk-state` に保存し、通常の再起動では解除しない。
別内容の要求は独立して扱う。`/ready` が200でも、ブロックされた要求自体には再試行禁止の503（メッセージにキーを含む）を返す。

原因を直した後、workerを停止して対象キーを明示的に解除する：

```bash
docker compose stop agent-sdk-relay
docker compose --profile agent-sdk run --rm --no-deps agent-sdk-relay \
  python3 -m puppetllm.sdk_state blocks
docker compose --profile agent-sdk run --rm --no-deps agent-sdk-relay \
  python3 -m puppetllm.sdk_state clear --key YOUR_64_CHARACTER_KEY
docker compose start agent-sdk-relay
```

`YOUR_64_CHARACTER_KEY` は表示された64桁のキーに置き換える。
強制終了で実行中の記録が残った場合は、停止状態で `recover --all` を実行して失敗状態に移してから
対象を `clear --key` する。保守操作はworker稼働中には拒否する。
`down -v` は保護状態のvolumeも削除するため、通常の停止では `-v` を付けない。
watchdogはworkerが停止・受付不能の場合に、待機要求へ再試行禁止の503を返す。
このモードはキューを他のresponderと共有せずに使用する。

## 手動起動（開発・デバッグ用）

```bash
# リポジトリで一度インストール（Python 仮想環境内）
pip install -e '.[agent-sdk]'

# ターミナル1
puppetllm serve --port 8765

# ターミナル2: SDK 側の認証を設定したシェルで起動
# 例: 実際の ANTHROPIC_API_KEY をこのシェルに設定しておく
puppetllm relay --kind agent-sdk --model sonnet \
    --max-concurrency 4 --timeout 300 --price-by upstream

# ターミナル3: 普段の作業ディレクトリから、既存の Claude Code を起動
ANTHROPIC_BASE_URL=http://127.0.0.1:8765 ANTHROPIC_API_KEY=puppet-local claude
```

SDK の認証は relay プロセス側のものを使う。外側のダミーキーを relay に設定しない。
`--api-key-env MY_KEY` で SDK 用の API キー変数を指定できる。
状態（失敗時の保護、進捗、SDK の作業ディレクトリ）は `--sdk-state-dir`
（既定 `~/.local/state/puppetllm/agent-sdk`、または `PUPPETLLM_SDK_STATE_DIR`）に置く。
Docker を使わない場合の保守コマンドは `python -m puppetllm.sdk_state blocks|clear|recover`、
状態の確認は `python -m puppetllm.sdk_state watch`（`127.0.0.1:8766`、`--host` で変更）。
SDK 子プロセスの接続先は既定で `https://api.anthropic.com`、変更する場合は
`--target` を指定する。外側の `ANTHROPIC_BASE_URL` は継承せず、同じ puppetllm
への循環接続も起動時に拒否する。対象は Anthropic 形式の接続先。
マシン全体の Claude Code 管理設定が接続先を上書きしている場合はそちらが優先される。
その場合は要求数の制限を適用できないため、relay は応答を 503 で拒否し、以後の受付を止める。
サブエージェントや自動承認の安全確認も同じ relay に要求を送るため、
並列数 1 では順番待ちでタイムアウトしやすい。上の例では 4 にしている。
外側で切断された要求は、内側の SDK 呼び出しもキャンセルして処理枠を解放する。

- API リクエストごとに独立した SDK セッションを作り、外側の全履歴を渡す。
  入力は、固定のシステム指示・ツール定義を先頭に置き、
  履歴をメッセージごとの JSON レコード・コンテンツブロックに分ける。
  履歴の末尾にキャッシュ境界を置くので、会話が追記されても、以前のブロックまでの
  共通プレフィックスを再利用できる。
  `--sdk-cache-ttl 1h`（Compose では `PUPPETLLM_AGENT_SDK_CACHE_TTL`、既定 `5m`）でキャッシュの保持を1時間にできる。
  書き込みの料金は5分保持より高いので、要求の間隔が5分を超えることが多い場合にだけ得になる。画像は対応する履歴ブロックの直後に添付する。
  初回やシステム指示・ツール定義・過去の履歴が変わった場合は、その変更位置以降の処理が必要。
  会話本文とツール履歴の省略や要約は行わず、`/compact` や編集後も外側から来た履歴を使う。
  過去のassistantの `thinking` / `redacted_thinking` は、署名・秘匿データとともに省略する。
  思考本文を通常の会話文へ変換せず、隣接する本文・ツール宣言・ツール結果を保って続行する。
  省略は種類ごとに `provenance.translation.dropped_fields` に記録し、厳密変換モードでは拒否する。
  relay は処理秒数と上流のキャッシュ読み込み・書き込みトークンをログに出す。
  SDK の組み込みツール・MCP・ユーザー設定・Hooks・セッション保存を無効にする。
  組み立てた入力はそのまま渡し、本文の `@path` やスラッシュコマンドを展開させない。
  `Read` や `Bash` は出力データとして返り、内側では実行しない。
  SDK が構造化出力の代わりに `Read` / `Bash` などを直接要求した場合も、
  SDK の `message_stop` までテキストと全ツールを集め、停止理由と引数を検証して返し、
  内側の「ツール無効」という応答が混ざらないよう内側のセッションを終了する。
  終端 delta 前に報告されたエラーや不完全な応答は失敗として扱う。
  **上流の完了保証には未解決の制約がある。** SDK 0.2.182 / 同梱 CLI 2.1.325 では、
  上流が `message_delta(stop_reason=tool_use)` の後、`message_stop` の前にエラーを返すと、
  CLI が正常時と区別できない停止通知を生成する。この場合はツール宣言を外側へ渡すことがある。
- テキスト、base64 画像、カスタムツール宣言、`tool_use`、テキスト・画像の `tool_result` に対応。
  画像は JSON 内に元の位置を記録し、[SDK の画像入力](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode)
  として添付する（PNG / JPEG / GIF / WebP）。
  会話途中の `role: "system"` に含まれるテキスト指示も、元の役割・順序を保って渡す。
  カスタムツールの `tool_addition` / `tool_removal`（定義または名前参照）と、
  ツール検索結果の `tool_reference` に対応する。`defer_loading` と履歴の変更を順番に適用し、
  現在有効なツール定義で次の出力を検証する。利用可能なツール一覧が変わると入力の先頭も変わるため、
  その要求では履歴キャッシュが外れる場合がある。
  メッセージ単位の `output_config`（effort など）は保持せず、変換時に記録する。
  出力のツール名・入力 JSON Schema・`tool_choice` を検証し、各呼び出しに ID を付ける。
  スキーマで型が明示された項目の JSON 文字列を、真偽値・数値・整数・配列・オブジェクトに
  戻す（例: `Bash.timeout: "120000"` や、配列が文字列化された `AskUserQuestion.questions`）。
  オブジェクトのプロパティ・配列要素も再帰的に処理し、必須項目に明示された既定値は補う。
  文字列型や複数の型を許可する項目は勝手に変換しない。範囲・整数の制約・配列の件数などスキーマの残りは
  そのまま検証する。それ以外の不正な引数は拒否し、
  値と動的な辞書キーを伏せて項目名・検証条件をエラーに示す（`x-should-retry: false` も返す）。
  指数オーバーフローを含む非有限数も、返却前に拒否する。
  URL 画像・文書・サーバー側ツールなどは 400 で明示的に拒否する。
  それらを含む既存セッションの再開や Claude Code の全機能互換は対象外。
- **意味を変換する方式であり、透過中継と同じ応答品質・挙動は保証しない。**
  `max_tokens`、thinking 設定、サンプリング設定、キャッシュ指定、ベータ機能などは
  保持せず、警告と `provenance.translation.dropped_fields` に記録する。
  `--strict-translation` はこれらがあると SDK を呼ばず 400 にする
  （通常の Messages リクエストには `max_tokens` があるため、その場合も拒否する）。
  内側の思考量は `--sdk-effort`、思考とJSON応答を合わせた出力上限は `--sdk-max-output-tokens` で明示指定できる。
  effortの既定は `high`、出力上限の既定は `auto`。effortに `auto` を明示するとSDKの設定を使う。
  外側の要求が最上位の `output_config.effort` を指定していればそれを使うが、`--sdk-effort` を上限とする
  （より低い指定の要求はその分だけ思考が減る。`auto` のときは要求の値をそのまま使う）。
  それ以外の外側の生成設定はコピーしない。
  明示的な生成設定の変更は別の実行指紋として扱う。
- `--sdk-max-turns`（既定 1）は構造化出力を含む内側の実行上限。
  `--sdk-max-api-requests`（既定1、Composeでは `PUPPETLLM_AGENT_SDK_MAX_API_REQUESTS`）は、
  query単位の送信ゲートでMessages APIの実HTTP送信回数を制限する。CLIには再試行設定0でも
  未完了ツールを回復しようとする追加送信があるため、設定値だけに依存せず超過要求を送信前に遮断する。
  API再試行は `--sdk-api-retries 0` が既定。`--sdk-structured-retries 1` は、正常な外側ツール宣言の
  終端を受け取るためのCLI設定。0では正常なツール宣言も途中で切られる場合がある。
  turn上限を増やすと内側の追加API要求が発生し得る。
  `--sdk-idle-warning`（既定120秒）は生成イベントが見えない場合の警告で、自動中断はしない。
  `--sdk-max-buffer-bytes`（既定 4194304）は SDK の出力JSON 1行の UTF-8 バイト数の受信上限。
  上限超過は設定名が分かるエラーとして返す。より大きな出力が必要ならこの上限を変更する。
  SDK の標準処理を補う transport で、日本語などのバイト数の計測と終了済みプロセスの
  パイプの解放を行う。この補正は SDK 内部の実装に依存するため、パッケージの extra と Docker イメージは
  SDK 0.2.182 に固定している。SDK 更新時には統合テストが必要。
  `--timeout` 超過で実行をキャンセルして 504、SDK の上流 HTTP エラーはそのステータス、
  その他の SDK 失敗や不正な出力には 502 を返す。履歴の長さの判定は上流 API に任せる。長すぎる要求には 400 `invalid_request_error` と
  `prompt is too long: N tokens > M maximum` を返すので、履歴を持つ外側のクライアントが compact して再送できる。
  この拒否ではブロックを残さない。上流のエラーや refusal の後に CLI が追加の要求を試みて要求数の制限で止まった場合は、
  最初の応答（その HTTP ステータス、または `stop_reason: refusal`）を結果として返す。
  実行回数上限・構造化出力の再試行上限・
  上流 HTTP ステータスは原因として表示し、生の会話や認証情報はエラーに含めない。
  外側への SSE は応答完成後の疑似ストリーム。usage は SDK が報告する値で、
  変換用プロンプト・構造化出力の分も含む。実モデル利用時には実際の費用が発生する。
- `--dry-run` で SDK を呼ばず変換入力を確認できる。通常の relay と同様に
  `--only`、`--model-map`、`--max-requests` を利用できる。

Docker内でテストする場合（ホストのPython環境は不要）：

```bash
docker compose --profile agent-sdk build agent-sdk-relay
docker run --rm --network none -e PUPPETLLM_TEST_REAL_AGENT_SDK=1 \
  puppetllm:agent-sdk \
  python3 -m unittest puppetllm.tests.test_agent_sdk puppetllm.tests.test_sdk_egress puppetllm.tests.test_sdk_guard \
    puppetllm.tests.test_sdk_runtime puppetllm.tests.test_sdk_runtime_integration \
    puppetllm.tests.test_sdk_state puppetllm.tests.test_sdk_transport -q
```

ホストで開発する場合は `pip install -e '.[test,agent-sdk]'` の後に
`python -m pytest puppetllm/tests/test_agent_sdk.py puppetllm/tests/test_sdk_*.py -q`。
通常テストはモデルを呼ばない。
`PUPPETLLM_TEST_REAL_AGENT_SDK=1` を付けると、実際の SDK / 同梱 CLI をローカルの
模擬サーバにつなぐ統合テスト（外側の CLI がファイルを読んで結果を返す往復を含む）も実行する。これも有料推論なし。
