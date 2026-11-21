[English](../bedrock.md) | **日本語** · [← README](../../README.ja.md)

# Bedrock

## Bedrock の SDK を proxy に向ける

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

SigV4 署名のため bedrock extra が必要: `pip install 'anthropic[bedrock]'`。AWS クレデンシャルはダミーで良い（proxy は署名を検証しない）が、SDK が署名を作るために何かしらは必要。model は URL パス (`/model/{id}/invoke`) に入り、streaming は AWS event stream で返る — どちらも server が吸収する。**応答の注入方法は Anthropic 経路と完全に同じ**（同じ [`/_control/respond`](quickstart.md#3-応答を供給するresponder) をそのまま使う）。

Bedrock 経路固有の挙動（すべて URL パスから判定。モード切替は不要）:

- **モデル ID の正規化**: `anthropic.claude-haiku-4-5-20251001-v1:0`、サフィックスなしの現行 ID（`anthropic.claude-opus-5`）、クロスリージョン推論プロファイル（`us.` / `eu.` / `apac.` / `jp.` / `au.` / `global.` / `us-gov.` … プレフィックス）、foundation-model / inference-profile の ARN（`aws` / `aws-cn` / `aws-us-gov` の全パーティション）を Anthropic 側の名前（`claude-haiku-4-5-20251001`）にマップする。pending snapshot の `model`・応答の `model` フィールド・`/_control/stats` の `by_model` はこの正規名を使うので、同じモデルの Bedrock 経由と Anthropic 直の呼び出しが 1 行に合流し、relay の `--model-map` / `--only` の glob も `claude-*` でマッチする。生の ID は snapshot / history エントリの `bedrock_model_id` に保持する。`anthropic.` セグメントを含まない ID はそのまま素通しする。
- **`anthropic_version` を検証する**（Anthropic モデル ID のみ）: 欠落、または `bedrock-2023-05-31` 以外の値は受付時点で `400 ValidationException` として弾く（pending は作られない）— 自前実装クライアントのミスを早期に検出する。他ベンダーの ID（`meta.llama…`、`amazon.titan…`）はボディ形式が異なるため検証せず素通しする。不正な `cache_control` 配置（[コスト / トークン / キャッシュを観測する](control-api.md#コスト--トークン--キャッシュを観測する) 参照）も同じ形で弾く。
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
| `promptVariables`（モデル ID がプロンプト管理の ARN） | responder 向けに `converse.promptVariables` に保持。その場合 `messages` は実 API 同様省略可 — `/_control/prompts` に登録したテンプレートを変数で描画したものを responder が見る（[フィクスチャ](#guardrail-と-trace-とプロンプト管理のフィクスチャ) 参照）。未登録なら `messages: []`。通常のモデル ID では `messages` は必須のまま |
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

## guardrail と trace とプロンプト管理のフィクスチャ

Bedrock 固有の 3 つの形を、模擬はせず注入できる:

- **guardrail の介入。** `stop_reason: "guardrail_intervened"` で答えると、Converse は `stopReason` にそのまま載せる。InvokeModel は、普通に終わる Anthropic 形式の本文の横に action を載せる — 本文フィールドの `amazon-bedrock-guardrailAction: INTERVENED`（ストリームでは `message_stop` チャンク）。guardrail を設定した（`X-Amzn-Bedrock-GuardrailIdentifier`）InvokeModel リクエストがそのまま通れば `NONE`、設定がなければフィールド自体がない。Anthropic 経路と Mantle エイリアスではメッセージは `end_turn` で終わり、OpenAI 経路は `content_filter` を返す。
- **trace。** 注入に `provider_response: {"converse": {"trace": {...}}}` を付けると、そのオブジェクトをそのまま Converse の応答（`trace`。ストリームでは `metadata` イベント）に載せる。`provider_response: {"invoke": {"trace": {...}}}` は InvokeModel の本文フィールド `amazon-bedrock-trace`（ストリームでは `message_stop` チャンク）になる。history には `provider_response` として残し、他の経路は記録だけして送らない。boto3 クライアントに届くのは SDK の trace の形に沿った部分だけ。
- **プロンプト管理。** `PUT /_control/prompts {"prompts": {"<prompt ARN>": {"system"?, "messages"?, "tools"?}}}`（または `--config` ファイルの `prompts` キー、`Puppet.prompts(...)`）で、system のテキスト・メッセージのテキスト・ツールの description に `{{name}}` プレースホルダを含められる Anthropic 形式のテンプレートを登録する。モデル id が登録済み ARN（バージョン付き ARN `…:prompt/ID:3` はベース ARN の登録にフォールバックする）の Converse リクエストは、リクエストが持たない system / tools をテンプレートから受け取り、`messages` を持たなければテンプレートの messages も受け取る — ルールは実際のプロンプト文に一致でき、responder もそれを読める。リクエスト自身の `messages` があればそちらが優先。`converse.prompt_arn` / `converse.prompt_rendered` が何が起きたかを示す。未知のプレースホルダはそのまま残り、未登録の ARN ならリクエストそのまま（messages は空、`converse.promptVariables` を responder に見せる）。

## バッチ推論（S3 エミュレーション同梱）

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
