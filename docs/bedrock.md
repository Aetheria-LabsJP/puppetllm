**English** | [日本語](ja/bedrock.md) · [← README](../README.md)

# Bedrock

## Pointing the Bedrock SDKs at the proxy

**Bedrock SDK (`AnthropicBedrock`):**

```python
from anthropic import AnthropicBedrock
client = AnthropicBedrock(base_url="http://localhost:8765", aws_region="us-east-1",
                          aws_access_key="dummy", aws_secret_key="dummy")
msg = client.messages.create(
    model="anthropic.claude-sonnet-5-5", max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}],
)
```

Requires the bedrock extra for SigV4 signing: `pip install 'anthropic[bedrock]'`. The AWS credentials can be any dummy values (the proxy doesn't validate the signature), but the SDK needs *something* to sign with. The model goes into the URL path (`/model/{id}/invoke`) and streaming comes back as an AWS event stream — the server absorbs both. **Injecting responses is exactly the same as the Anthropic route** (the same [`/_control/respond`](quickstart.md#3-supply-the-response-responder)).

Bedrock-route specifics (all decided from the URL path, no mode switch):

- **Model-id normalization**: `anthropic.claude-haiku-4-5-20251001-v1:0`, suffix-less current ids (`anthropic.claude-opus-5-5`), cross-region inference profiles (`us.` / `eu.` / `apac.` / `jp.` / `au.` / `global.` / `us-gov.` … prefixes) and foundation-model / inference-profile ARNs (any partition: `aws`, `aws-cn`, `aws-us-gov`) are mapped to the Anthropic-side name (`claude-haiku-4-5-20251001`), which is what the pending snapshot's `model`, the response `model` field and `/_control/stats` `by_model` use — so Bedrock and direct-Anthropic traffic for the same model aggregate into one row, and relay's `--model-map` / `--only` globs match `claude-*`. The raw id is kept on the snapshot / history entry as `bedrock_model_id`. Ids without an `anthropic.` segment pass through unchanged.
- **`anthropic_version` is validated** for Anthropic model ids: missing or anything other than `bedrock-2023-05-31` is rejected up front with `400 ValidationException` (no pending is created) — catches hand-rolled clients early. Other vendors' ids (`meta.llama…`, `amazon.titan…`) have their own body shapes and pass through unchecked. Malformed `cache_control` layouts (see [Observe cost / tokens / cache](control-api.md#observe-cost--tokens--cache)) are rejected the same way.
- **SigV4 is not verified**: `Authorization` / `X-Amz-Date` / `X-Amz-Security-Token` are ignored.
- **Response headers**: non-stream responses carry `X-Amzn-Bedrock-Input-Token-Count` (non-cached input only, like Converse's `inputTokens`) / `X-Amzn-Bedrock-Output-Token-Count` / `X-Amzn-Bedrock-Cache-Read-Input-Token-Count` / `X-Amzn-Bedrock-Cache-Write-Input-Token-Count` / `X-Amzn-Bedrock-Invocation-Latency` (+ `x-amzn-requestid`); streams carry `amazon-bedrock-invocationMetrics` (incl. `cacheReadInputTokenCount` / `cacheWriteInputTokenCount`) on the final chunk and `X-Amzn-Bedrock-Content-Type`, as real Bedrock does. The options boto3 sends as **request headers** — `serviceTier=` (`X-Amzn-Bedrock-Service-Tier`, `priority | default | flex | reserved`) and `performanceConfigLatency=` (`X-Amzn-Bedrock-PerformanceConfig-Latency`) — are validated, echoed back in the same response headers on both routes (`default` when no tier was asked for), and the tier is made visible to the responder as canonical `service_tier` (`priority` / `flex` → `auto`, `default` / `reserved` → `standard_only`, the relay's vocabulary).
- **Messages-API alias for `AnthropicBedrockMantle`**: `POST /anthropic/v1/messages` (the path served by the `bedrock-runtime` / `bedrock-mantle` hosts) is the plain Anthropic handler with Bedrock model-id normalization, and the plain `/v1/messages` route hands any request whose `model` is a Bedrock id (`[region.]anthropic.…` / ARN) to the same handler — so `AnthropicBedrockMantle(base_url="http://localhost:8765")` works whether it posts to the root or to `/anthropic` (SSE streaming, Anthropic error envelope, `anthropic-version` header, no `anthropic_version` body field). Batches / count_tokens are not served on this alias, matching Bedrock.
- **Receipt log**: every Bedrock request (and every rejection) is logged to stderr as one `[bedrock] invoke model=<raw> -> <canonical> pending=<id> …` line, so Bedrock traffic is distinguishable from the other routes at a glance.

**boto3 (`bedrock-runtime`), including Converse:**

```python
import boto3
from botocore.config import Config

rt = boto3.client("bedrock-runtime", region_name="us-east-1",
                  aws_access_key_id="dummy", aws_secret_access_key="dummy",
                  endpoint_url="http://localhost:8765")

rt.converse(modelId="anthropic.claude-opus-5-5",
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
            inferenceConfig={"maxTokens": 1024})

for event in rt.converse_stream(modelId="anthropic.claude-opus-5-5",
                                messages=[{"role": "user", "content": [{"text": "hello"}]}])["stream"]:
    print(event)          # messageStart / contentBlockDelta / … / metadata
```

`POST /model/{id}/converse` and `/converse-stream` speak the Converse JSON schema and are
normalized to the **same canonical pending** as every other route, so a responder answers
them with the usual `/_control/respond` content blocks and never sees the difference:

| Converse | canonical |
|---|---|
| `{"text": …}` | `{"type": "text", …}` |
| `{"image": {"format", "source": {"bytes" \| "s3Location"}}}` / `{"document": …}` | `{"type": "image" \| "document", "source": {…}}` |
| `{"toolUse": {"toolUseId", "name", "input"}}` | `{"type": "tool_use", "id", "name", "input"}` |
| `{"toolResult": {"toolUseId", "content", "status"}}` | `{"type": "tool_result", "tool_use_id", "content", "is_error"}` |
| `{"reasoningContent": {"reasoningText": {"text", "signature"}}}` / `{"redactedContent"}` | `{"type": "thinking", …}` / `{"type": "redacted_thinking", "data"}` |
| `{"cachePoint": {"type": "default", "ttl"?}}` | `cache_control` on the **preceding** block / tool (a real breakpoint for the pseudo cache) |
| `inferenceConfig.{maxTokens,temperature,topP,stopSequences}` | `max_tokens` / `temperature` / `top_p` / `stop_sequences` |
| `toolConfig.tools[].toolSpec` / `toolChoice {auto\|any\|tool}` | `tools[]` / `tool_choice` |
| `additionalModelRequestFields` | merged into the canonical body (`thinking`, `top_k`, `anthropic_beta`, …); a key Converse models itself (`system`, `tools`, `max_tokens`, `temperature`, …) is a `ValidationException`, never a back door around the schema |
| `outputConfig.effort` / `outputConfig.textFormat` | `output_config.effort` (`low` … `xhigh` / `max`) / `output_config.format` (`{type: json_schema, schema, name?}` — `textFormat.structure.jsonSchema.schema` is the JSON **string** the API defines, decoded here into the schema object); the native fields win over the same keys sent through `additionalModelRequestFields`, and a non-object `output_config` there is a `ValidationException` |
| `promptVariables` (prompt-management ARN as the model id) | kept for the responder under `converse.promptVariables`; `messages` may then be absent, as on the real API — the responder sees the template registered under `/_control/prompts` rendered with the variables (see [the fixtures](#guardrails-traces-and-prompt-management-fixtures)), or `messages: []` when none is; with an ordinary model id `messages` stays required |
| `serviceTier.type` | `service_tier` (`priority` / `flex` → `auto`, `default` / `reserved` → `standard_only`) |

Responses are encoded back from the block types an assistant turn can actually hold
(`text` / `toolUse` / `reasoningContent` — a responder's blocks are normalized to
text / tool_use / thinking / redacted_thinking before any encoder sees them):
`output.message`, `stopReason` in the Converse vocabulary (a canonical `refusal` becomes `content_filtered`, `pause_turn` becomes
`end_turn`), `usage` with `inputTokens` excluding cached tokens plus
`cacheReadInputTokens` / `cacheWriteInputTokens` / `cacheDetails`, and `metrics.latencyMs`.
`additionalModelResponseFieldPaths` (JSON pointers, ≤ 10) are resolved against the native
Messages-API response and also ride on `messageStop` in the stream. Requests are validated against the
shapes of the schema — not its every constraint: each content block, source,
`reasoningContent`, `toolChoice` and `system` entry is a **union** (exactly one member — a block carrying both
`text` and `cachePoint` is a `ValidationException`, not a silently dropped text; so is an
unknown member on a union object or an unknown top-level request member), and each
union takes only its own members (an `ImageSource` / `VideoSource` is `bytes | s3Location`;
only a `DocumentSource` also takes `text | content`). Required fields and enums
(`image.format`, `video.format`, `document.name`, `toolUse.input`, `toolResult.content`,
`cachePoint.type` / `ttl`, `inferenceConfig` ranges, `serviceTier.type`, `requestMetadata`
1–16 entries, a `guardContent.image` being `png` / `jpeg` from `bytes`) are enforced, and
two `cachePoint`s cannot address the same block. Not checked (a request production would
reject can still pass here): decoded media size and validity, cross-block placement rules,
the inner shape of `audio` / `searchResult` (only "must be an object"), user / assistant
alternation (and `role: "system"` is accepted, as the botocore model allows), and the
`guardrailConfig` / `promptVariables` / prompt-ARN conditionals. The stream emits `messageStart` → per block
`contentBlockStart` (tool use) / `contentBlockDelta` / `contentBlockStop` → `messageStop`
→ `metadata`, as raw-JSON event-stream frames (not the `chunk` + base64 wrapper the
InvokeModel route uses).

## Guardrails, traces and prompt-management fixtures

Three Bedrock-only shapes are injectable without being simulated:

- **Guardrail intervention.** Answer with `stop_reason: "guardrail_intervened"`: Converse reports it as `stopReason`; InvokeModel reports the action next to an Anthropic body that ends normally — the body field `amazon-bedrock-guardrailAction: INTERVENED` (on a stream, in the `message_stop` chunk). An InvokeModel request that configured a guardrail (`X-Amzn-Bedrock-GuardrailIdentifier`) and was let through gets `NONE`; without a guardrail the field is absent. On the Anthropic route and the Mantle alias the message ends with `end_turn`; the OpenAI route reports `content_filter`.
- **Traces.** `provider_response: {"converse": {"trace": {...}}}` on the injection puts the object verbatim on a Converse response (`trace`, or the `metadata` event of a stream); `provider_response: {"invoke": {"trace": {...}}}` becomes InvokeModel's body field `amazon-bedrock-trace` (in the `message_stop` chunk of a stream). History records them as `provider_response`; other routes record them and send nothing. boto3 clients keep only what follows the SDK's trace shape.
- **Prompt management.** `PUT /_control/prompts {"prompts": {"<prompt ARN>": {"system"?, "messages"?, "tools"?}}}` (or a `prompts` key in a `--config` file, or `Puppet.prompts(...)`) registers Anthropic-shaped templates whose system text, message text and tool descriptions may hold `{{name}}` placeholders. A Converse request whose model id is a registered ARN — a versioned ARN (`…:prompt/ID:3`) falls back to the template registered for the base ARN — sees the template's system / tools where the request supplies none and, when it carries no `messages`, the template's messages, so rules match on the real prompt text and the responder reads it; a request's own `messages` win. `converse.prompt_arn` / `converse.prompt_rendered` say what happened. Unknown placeholders stay as written; an unregistered ARN keeps the bare request (empty messages, `converse.promptVariables` for the responder).

## Batch inference (with a bundled S3)

The control-plane batch API is emulated too, over a directory-backed **S3 emulation** so a
plain `boto3` S3 client can stage the input and read the results:

```python
import boto3
from botocore.config import Config

s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="d", aws_secret_access_key="d",
                  endpoint_url="http://localhost:8765",
                  config=Config(s3={"addressing_style": "path"}))     # path-style is required
bedrock = boto3.client("bedrock", region_name="us-east-1", aws_access_key_id="d",
                       aws_secret_access_key="d", endpoint_url="http://localhost:8765")

s3.create_bucket(Bucket="batch-in"); s3.create_bucket(Bucket="batch-out")
s3.put_object(Bucket="batch-in", Key="jobs/input.jsonl", Body=b'''{"recordId": "r1", "modelInput": {...}}\n''')

job = bedrock.create_model_invocation_job(
    jobName="myjob", modelId="anthropic.claude-opus-5-5",
    roleArn="arn:aws:iam::123456789012:role/BatchRole",
    inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://batch-in/jobs/input.jsonl"}},
    outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://batch-out/results/"}})
```

Every `{"recordId", "modelInput"}` line becomes an ordinary **pending** (provider
`bedrock`, snapshot tagged `job_arn` / `job_id` / `record_id`), so the responder answers
them with the same `/_control/respond` / `auto` / `error`. Once every record has an outcome
the job writes `<output prefix>/<jobId>/<input file>.out` and `manifest.json.out`, and
ends `Completed`. The output file name keeps the key's path relative to the input prefix,
so `a/data.jsonl` and `b/data.jsonl` do not collide. Each output line is

```json
{"recordId": "r1", "modelInput": { … as submitted … },
 "modelOutput": { … }}                                  // or, on failure:
{"recordId": "r2", "modelInput": { … },
 "error": {"errorCode": 400, "errorMessage": "…"}}
```

`recordId` is the only correlation key (results are not ordered), and the manifest is

```json
{"totalRecordCount": 3, "processedRecordCount": 3, "successRecordCount": 2,
 "errorRecordCount": 1, "inputTokenCount": 120, "outputTokenCount": 48}
```

`modelInput` is an InvokeModel body (default) or a Converse body with
`modelInvocationType: "Converse"`; `modelOutput` is the matching response shape.
An `InvokeModel` body is model-specific, so those jobs need an Anthropic `modelId`;
`Converse` is a model-independent schema, so a `Converse` job takes any model. What the
bundled store cannot honour is refused rather than echoed: `s3EncryptionKeyId` (outputs are
plain files) is a `ValidationException`, and `s3BucketOwner` must be `123456789012`, the
one account the store's buckets and the job ARNs belong to.
`get_model_invocation_job` / `list_model_invocation_jobs` / `stop_model_invocation_job`
work as usual (`GET /_control/bedrock_jobs` shows the registry with unresolved record ids).

`Stop` finalizes synchronously — the status is already terminal when the call returns —
except while the input is still being read or an output write is already in flight, when it
returns `200` with the job in `Stopping` and the terminal status follows; the job ends
`Stopped`, the terminal status the real service reports for `StopModelInvocationJob` — an injection already in flight still
completes, but records that were never processed stay in `totalRecordCount` only, are counted
in neither `processedRecordCount` nor `errorRecordCount`, and are written to no output line
(they are listed under `cancelled` in `/_control/bedrock_jobs`). A stop is accepted while the
job is still `Submitted` or registering (whatever was not yet registered is never processed —
and if that job's input then fails validation, in the read or in the registration, it ends
`Failed` with the reason rather than vanishing from under the caller who stopped it), is
idempotent once the job is `Stopping` / `Stopped`
(botocore retries a Stop whose response it lost), a `Completed` / `Failed` job answers
`ConflictException`, and a stop accepted while the outputs are being written still decides
the terminal status. If the
outputs cannot be written (output bucket deleted mid-job, disk error) the job ends **`Failed`**
with the reason in `message` — whatever was written before the failure stays — so a poller
must treat `Completed` / `Stopped` / `Failed` as the terminal set. A record whose `modelInput`
is invalid becomes an `error` record instead of failing the job, like the real service; a
malformed JSONL line, a non-string or duplicate `recordId`, a missing input / output bucket or
an input with no records rolls the whole create back with a `ValidationException`. Features AWS documents as unavailable in batch come back as
`errorCode 400` records: tool calling, structured output, and prompt caching
(`cache_control` / `cachePoint` anywhere in the record).

Other intentional differences (determinism over fidelity): no `Validating` / `Scheduled`
phases (a job is `InProgress` as soon as its records are pending), no minimum record count
beyond "at least one", and no clock-based expiry. Input JSONL is read and held in memory, so
a job's practical size is bounded by RAM rather than by AWS's 1 GB / 50,000-record limits.
`/_control/clear` drops the job registry but deliberately leaves the S3 store alone, so
outputs a cleared job had already written stay readable; a create still in flight at that
moment — reading its input, registering, waiting on an idempotent twin, or finalizing
inline — answers `400 ConflictException` (deliberately not a retryable 5xx — botocore would
otherwise re-create the job the human had just cleared). The output layout is checked at
create time (`<prefix>/<jobId>/<file>.out` must fit the store's 255-byte segment limit and
neither the prefix nor any ancestor of it may be an existing object), so a job never ends
`Failed` on that after every record was processed. A `recordId` must be a non-empty string
when present (an empty one is not silently replaced by a generated id).

The S3 emulation covers exactly what that flow needs — `PUT`/`HEAD` bucket, `PUT` / `GET` /
`HEAD` / `DELETE` object, `GET /` (list buckets), `DELETE` bucket (`409 BucketNotEmpty`
unless empty), ranged `GET` (`Range` → `206` + `Content-Range`, `416 InvalidRange` past the
end), conditional requests (`If-Match` / `If-None-Match` → `304` on a read, `412 PreconditionFailed` on a write or a
delete, and `404 NoSuchKey` for an `If-Match` write or delete on a key that does not exist,
as the real service answers — atomically, so sixteen concurrent `If-None-Match: *` writers
get exactly one winner; the `If-*-Since` forms on reads only), conditional deletes
(`delete_object(IfMatch=…)`, `*` for "only if it exists"; a weak `W/` tag never authorizes a
write or a delete; the size / last-modified forms are directory-bucket features and answer
`501`), `ExpectedBucketOwner` enforcement (every bucket belongs to `123456789012`; a wrong
value in either the query or the `x-amz-` header is `403 AccessDenied`), request-checksum
verification (`400 BadDigest`, `IncompleteBody` when `x-amz-decoded-content-length`
disagrees, `InvalidRequest` for an `aws-chunked` body without its terminating chunk) and
both listings:
`GET ?list-type=2` and the V1 `GET` with `marker`, each with `prefix`, `delimiter`
(`CommonPrefixes`), `max-keys` (clamped to 1000 rather than refused) and `encoding-type=url`
(path-style, no auth, `aws-chunked` bodies decoded).

Two of those are not optional in practice: boto3's `download_file` splits anything over
`multipart_threshold` (8 MB) into concurrent ranged GETs, and botocore asks for
`encoding-type=url` on every listing — it URL-decodes the keys it reads back only when the
response echoes `EncodingType`, so the two have to be switched together.

**Everything else is refused, not approximated.** `CreateMultipartUpload`, `CopyObject`,
object and bucket tagging / ACL / versioning / policy, `ListObjectVersions`, `DeleteObjects`,
the other sub-resources, and the `put_object` options this store cannot store
(`Tagging`, `Metadata`, SSE, ACL / grants, object lock) answer `501 NotImplemented` in an S3
envelope; `ContentType` and the other plain entity headers are accepted but not stored
(objects read back as `binary/octet-stream`). This matters more than it sounds:
`copy_object`, `put_object_tagging` and `put_object_acl` all arrive as a `PUT` on the
object's own path, so a store that ignored the sub-resource would write their body — or
their empty body — over the object and report success. Presigned-URL query auth is ignored
like header auth. For multipart upload specifically, keep staged inputs under that 8 MB
threshold or lower it in `boto3.s3.transfer.TransferConfig`. Because the body is read
**before** any refusal is sent, a refusal never leaves the keep-alive connection out of
sync: botocore's `Expect: 100-continue` PUTs get their `100 Continue` before any answer —
and that holds for a path the router itself cannot represent (a control character in a
key, which is refused before routing). A non-`STANDARD` `StorageClass` is refused too
(everything here is STANDARD), and the request checksums botocore attaches
(`Content-MD5`, `x-amz-checksum-crc32` / `sha1` / `sha256`, header or `aws-chunked`
trailer) are verified — `400 BadDigest` on a mismatch — so a body corrupted in transit is
never stored quietly (`crc32c` / `crc64nvme` are accepted unverified).

Objects live under `PUPPETLLM_S3_ROOT` (default: a per-process temp directory) and can never
be written outside it: bucket names are validated everywhere (including inside `s3://` URIs),
keys with `.` / `..` segments are refused (and a single key segment is limited to 255 bytes,
what the backing filesystem accepts — a flat 1024-byte key is legal on S3 but not here), a
write needs an existing bucket, writes land through a rename from a temp directory outside
the bucket tree so a concurrent reader never sees a half-written object (one store-wide
lock also covers the batch emulation's worker thread, so its output writes cannot
interleave inside an HTTP handler's stat → precondition → write either; the handlers take
that lock off the event loop, so a thread holding it never freezes the other routes), and a malformed
`aws-chunked` body is a `400 InvalidRequest`. Control characters in a key are refused (this
store renders keys straight into the listing XML, where they would be illegal), and a key is
never silently rewritten — `PUT /bucket//a` is a `400` rather than a quiet write to `a`.
Bucket names that collide with an API path (`model`, `v1`, `anthropic`, `_control`,
`model-invocation-job[s]`, `docs`, `redoc`, `openapi.json`) are refused, and a request to one
of those paths with the wrong method gets the API's own `405` + `Allow`, never an S3 error
envelope.
