"""Bedrock batch inference (control-plane `bedrock` service) against the fake.

Emulates `CreateModelInvocationJob` / `GetModelInvocationJob` / `ListModelInvocationJobs` /
`StopModelInvocationJob` on top of the bundled S3 emulation (`providers/s3.py`):

  POST /model-invocation-job                      -> {"jobArn"}
  GET  /model-invocation-job/{jobIdentifier}      -> job description (ARN or 12-char id)
  POST /model-invocation-job/{jobIdentifier}/stop -> 200, empty body
  GET  /model-invocation-jobs?statusEquals=&nameContains=&submitTimeAfter=&submitTimeBefore=
                              &sortBy=CreationTime&sortOrder=&maxResults=&nextToken=

Flow: the job's input (`inputDataConfig.s3InputDataConfig.s3Uri`, one `.jsonl` object or a
prefix holding several) is read from the S3 store; every `{"recordId", "modelInput"}` line
becomes a normal **pending** (provider `bedrock`, snapshot tagged with `job_arn` /
`record_id` / `job_id`) that a responder answers with the usual `/_control/respond` /
`auto` / `error`. Once every record has an outcome the job writes
`<output prefix>/<jobId>/<input file name>.out` (one line per record, `modelOutput` or
`error {errorCode, errorMessage}`) plus `manifest.json.out`, and ends `Completed` — or
`Stopped` when the caller stopped it, the terminal status the real service reports for
`StopModelInvocationJob`. Records aborted by `Stop` were never processed: they count in
`totalRecordCount` only, never in the processed / success / error counters, and get no
output line (`/_control/bedrock_jobs` lists them separately). `modelInput` is an
InvokeModel body (`modelInvocationType: InvokeModel`, the default — `anthropic_version`
is required as on the real API) or a Converse body (`Converse`); `modelOutput` is the
matching response body.

Intentional differences from the real service (determinism over fidelity): no
`Validating` / `Scheduled` phases (a job is `InProgress` as soon as its records are
pending), no minimum record count beyond "at least one", no clock-based expiry, and `Stop`
finalizes synchronously (entries with an injection in flight still complete, as on the real API).
Prompt caching is disabled for batch records, matching the real service.
Control: `GET /_control/bedrock_jobs` lists the registry; `/_control/clear` wipes it.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime
import traceback
import uuid
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from . import bedrock as _bedrock
from . import converse as _converse
from . import s3 as _s3
from .. import fake_server as fs

_JOB_NAME_RE = re.compile(r"^[a-zA-Z0-9]{1,63}(-*[a-zA-Z0-9+\-.]){0,63}$")
_ROLE_RE = re.compile(r"^arn:aws(-[^:]+)?:iam::([0-9]{12})?:role/.+$")
_TOKEN_RE = re.compile(r"^[a-zA-Z0-9]{1,256}(-*[a-zA-Z0-9]){0,256}$")
_JOB_ID_RE = re.compile(r"(?:^|/)(?P<id>[a-z0-9]{12})$")
_IDENTIFIER_RE = re.compile(
    r"^(?:arn:aws(?:-[a-z-]+)?:bedrock:[a-z0-9-]{1,20}:[0-9]{12}:model-invocation-job/)?"
    r"[a-z0-9]{12}$")
STATUSES = ("Submitted", "InProgress", "Completed", "Failed", "Stopping", "Stopped",
            "PartiallyCompleted", "Expired", "Validating", "Scheduled")
_REGION = "us-east-1"
_ACCOUNT = "123456789012"

_collector_tasks: set[asyncio.Task] = set()


class _JobCleared(Exception):
    """`/_control/clear` removed the job while its records were still being registered."""


def _err(status: int, etype: str, message: str) -> JSONResponse:
    return _bedrock._bedrock_error_response(status, etype, message, request_id=str(uuid.uuid4()))


def _parse_timestamp(raw: str) -> float | None:
    """Accept epoch seconds or an ISO-8601 timestamp (what boto3 sends for a datetime)."""
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def _arn(job_id: str) -> str:
    return f"arn:aws:bedrock:{_REGION}:{_ACCOUNT}:model-invocation-job/{job_id}"


def _valid_identifier(identifier: str) -> bool:
    """`jobIdentifier` is a bare 12-char id or the full model-invocation-job ARN."""
    return bool(_IDENTIFIER_RE.match(identifier or ""))


def _resolve_job(identifier: str) -> dict[str, Any] | None:
    m = _JOB_ID_RE.search(identifier or "")
    return fs.state.bedrock_jobs.get(m.group("id")) if m else None


def _record_outcome(job: dict[str, Any], entry: dict[str, Any],
                    output: dict[str, Any]) -> None:
    """Attach a record's outcome and keep the counters up to date. Call within the lock."""
    entry["output"] = output
    job["n_processed"] += 1
    if "error" in output:
        job["n_errors"] += 1


def _counts(job: dict[str, Any]) -> dict[str, int]:
    """Record counters. A record aborted by `Stop` was never processed, so — like the real
    service, which only bills what it processed — it counts in neither processed nor errored.

    Maintained incrementally: a caller polling `GetModelInvocationJob` must not walk every
    entry of a 100k-record job under the global state lock.
    """
    processed, errors = job["n_processed"], job["n_errors"]
    return {"totalRecordCount": len(job["entries"]), "processedRecordCount": processed,
            "successRecordCount": processed - errors, "errorRecordCount": errors}


def job_json(job: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "jobArn": job["arn"], "jobName": job["name"], "modelId": job["model_id"],
        "roleArn": job["role_arn"], "status": job["status"],
        "inputDataConfig": job["input_config"], "outputDataConfig": job["output_config"],
        "modelInvocationType": job["invocation_type"],
        "submitTime": job["submit_time"], "lastModifiedTime": job["last_modified"],
        "jobExpirationTime": job["submit_time"] + job["timeout_hours"] * 3600.0,
        "timeoutDurationInHours": job["timeout_hours"],
        **_counts(job),
    }
    if job.get("end_time") is not None:
        out["endTime"] = job["end_time"]
    if job.get("message"):
        out["message"] = job["message"]
    if job.get("client_token"):
        out["clientRequestToken"] = job["client_token"]
    if job.get("vpc_config") is not None:
        out["vpcConfig"] = job["vpc_config"]
    return out


# ── input / output ────────────────────────────────────────────────────


def _contains_key(node: Any, key: str, depth: int = 0) -> bool:
    """Whether `key` appears anywhere in a nested request body. The depth cap only guards
    against a recursion blow-up; it is far past anything `json.loads` will hand back for a
    real request body, so a cache marker cannot hide below it."""
    if depth > 100:
        return False
    if isinstance(node, dict):
        return key in node or any(_contains_key(v, key, depth + 1) for v in node.values())
    if isinstance(node, list):
        return any(_contains_key(v, key, depth + 1) for v in node)
    return False


def _reject_unsupported(model_input: dict[str, Any], inv_type: str) -> None:
    """Features AWS documents as unavailable in batch inference.

    "Batch inference does not support tool calling (function calling) or structured output",
    and prompt caching "is not supported with the batch inference API" — a record using them
    comes back as an error record rather than silently behaving differently from production.
    """
    if inv_type == "Converse":
        tool_key, fmt_key = "toolConfig", "outputConfig"
    else:
        tool_key, fmt_key = "tools", "output_config"
    if isinstance(model_input, dict):
        if model_input.get(tool_key):
            raise _converse.ConverseValidationError(
                f"{tool_key}: tool calling is not supported by batch inference")
        fmt = model_input.get(fmt_key)
        if isinstance(fmt, dict) and (fmt.get("format") or fmt.get("textFormat")):
            raise _converse.ConverseValidationError(
                f"{fmt_key}: structured output is not supported by batch inference")
        cache_key = "cachePoint" if inv_type == "Converse" else "cache_control"
        if _contains_key(model_input, cache_key):
            raise _converse.ConverseValidationError(
                f"{cache_key}: prompt caching is not supported by batch inference")


_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")


def _validate_configs(body: dict[str, Any], in_cfg: dict[str, Any],
                      out_cfg: dict[str, Any]) -> str | None:
    """Shapes botocore will later have to parse back out of `GetModelInvocationJob`: an
    accepted `vpcConfig: "broken"` would be echoed as a response the SDK cannot model."""
    for where, cfg in (("inputDataConfig.s3InputDataConfig", in_cfg),
                       ("outputDataConfig.s3OutputDataConfig", out_cfg)):
        owner = cfg.get("s3BucketOwner")
        if owner is not None and not (isinstance(owner, str) and _ACCOUNT_RE.match(owner)):
            return f"{where}.s3BucketOwner: must be a 12-digit account id"
        if owner is not None and owner != _ACCOUNT:
            # Every bucket in the bundled store belongs to the one account the ARNs name;
            # accepting another owner would make a confused-deputy check pass vacuously.
            return f"{where}.s3BucketOwner: the bundled S3 store is owned by {_ACCOUNT}, not {owner}"
        if cfg.get("s3EncryptionKeyId") is not None:
            # Outputs are plain files; claiming KMS encryption would be a lie a test could
            # build on. Refuse rather than approximate.
            return f"{where}.s3EncryptionKeyId: SSE-KMS is not supported by this emulation"
    vpc = body.get("vpcConfig")
    if vpc is not None:
        if not isinstance(vpc, dict):
            return "vpcConfig: must be an object"
        for field, limit in (("subnetIds", 16), ("securityGroupIds", 5)):
            ids = vpc.get(field)
            if (not isinstance(ids, list) or not (1 <= len(ids) <= limit)
                    or not all(isinstance(i, str) and i for i in ids)):
                return f"vpcConfig.{field}: 1-{limit} non-empty strings are required"
        if set(vpc) - {"subnetIds", "securityGroupIds"}:
            return "vpcConfig: only subnetIds and securityGroupIds are allowed"
    tags = body.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or len(tags) > 200:
            return "tags: must be an array of at most 200 {key, value} objects"
        for i, t in enumerate(tags):
            if (not isinstance(t, dict) or not isinstance(t.get("key"), str)
                    or not isinstance(t.get("value"), str) or not t["key"]):
                return f"tags[{i}]: must be an object with string `key` and `value`"
    return None


def _read_input(s3_uri: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """(output file name, records) per JSONL object under the input URI.

    The name keeps the key's path **relative to the requested prefix**, so `a/data.jsonl`
    and `b/data.jsonl` do not both write to `data.jsonl.out`.
    """
    bucket, key = _s3.parse_s3_uri(s3_uri)
    if not _s3.bucket_exists(bucket):
        raise ValueError(f"input bucket does not exist: {bucket}")
    if key.endswith(".jsonl"):
        data = _s3.get_object(bucket, key)
        if data is None:
            raise ValueError(f"input object not found: {s3_uri}")
        files = [(key.rsplit("/", 1)[-1], data)]
    else:
        prefix = key if (not key or key.endswith("/")) else key + "/"
        files = [(o["key"][len(prefix):], _s3.get_object(bucket, o["key"]) or b"")
                 for o in _s3.list_objects(bucket, prefix, etag=False)
                 if o["key"].endswith(".jsonl")]
        if not files:
            raise ValueError(f"no .jsonl input objects under {s3_uri}")
    out = []
    for name, data in files:
        records = []
        for lineno, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{name}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(rec, dict) or not isinstance(rec.get("modelInput"), dict):
                raise ValueError(f"{name}:{lineno}: each record needs an object `modelInput`")
            records.append(rec)
        out.append((name, records))
    if not any(records for _n, records in out):
        raise ValueError(f"no records found under {s3_uri}")
    return out


def _write_outputs(job: dict[str, Any]) -> None:
    bucket, prefix = _s3.parse_s3_uri(job["output_config"]["s3OutputDataConfig"]["s3Uri"])
    prefix = prefix if (not prefix or prefix.endswith("/")) else prefix + "/"
    base = f"{prefix}{job['id']}/"
    in_tok = out_tok = 0
    for fname, record_ids in job["files"]:
        lines = []
        for rid in record_ids:
            e = job["entries"][rid]
            if e["output"] is None or e.get("cancelled"):
                continue  # unprocessed (stopped) records are not written
            line = {"recordId": rid, "modelInput": e["model_input"], **e["output"]}
            lines.append(json.dumps(line, ensure_ascii=False))
            u = e.get("usage") or {}
            in_tok += int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)) \
                + int(u.get("cache_creation_input_tokens", 0))
            out_tok += int(u.get("output_tokens", 0))
        if not lines:
            continue  # nothing was processed for this input file; the real service writes none
        _s3.put_object(bucket, f"{base}{fname}.out", ("\n".join(lines) + "\n").encode("utf-8"))
    manifest = {**_counts(job), "inputTokenCount": in_tok, "outputTokenCount": out_tok}
    _s3.put_object(bucket, f"{base}manifest.json.out", json.dumps(manifest).encode("utf-8"))


def _ready_to_end(job: dict[str, Any]) -> bool:
    """Whether every record now has an outcome. Call within the lock; the caller then
    finalizes with `_finalize` OUTSIDE the lock, so a big job's output serialization and
    filesystem writes never stall the control plane."""
    if job.get("creating") or job.get("finalizing"):
        return False
    if job["status"] in ("Completed", "PartiallyCompleted", "Stopped", "Failed"):
        return False
    if job.get("unresolved", 0) > 0:
        return False
    job["finalizing"] = True
    return True


def _publish(job: dict[str, Any], status: str, message: str | None) -> None:
    """Publish a terminal status. Deliberately synchronous: every holder of the state lock
    runs without awaiting, so this is atomic with respect to them, and it therefore still
    works on the cancellation path (where awaiting the lock would re-raise immediately)."""
    try:
        if fs.state.bedrock_jobs.get(job["id"]) is not job:
            return  # cleared while we were writing
        # A user Stop that was accepted while the outputs were being written still decides
        # the terminal status — otherwise the job could settle back to `Completed`.
        if status != "Failed" and job.get("stop_requested"):
            status = "Stopped"
        now = time.time()
        job["status"] = status
        job["message"] = message or job.get("message")
        job["end_time"] = now
        job["last_modified"] = now
    finally:
        job["finalizing"] = False


async def _finalize(job: dict[str, Any]) -> None:
    """Write the outputs, then publish the terminal status (so `Completed` always implies
    the objects exist). Never called with the state lock held.

    The write runs in a worker thread and cannot be cancelled, so the status is published
    from the write's own completion callback rather than from this coroutine: cancelling
    the request that awaits it (client disconnect, `asyncio.wait_for`, shutdown) neither
    wedges the job nor publishes `Failed` for outputs the thread then finishes writing.
    """
    writing = asyncio.ensure_future(asyncio.to_thread(_write_outputs, job))

    def _done(fut: asyncio.Future) -> None:
        exc = None if fut.cancelled() else fut.exception()
        if exc is not None:
            # The exception is swallowed by the awaiter (the job's `Failed` status IS the
            # report), so log it here or an emulator bug in the writer becomes invisible.
            _bedrock._log(f"batch job {job['id']} output write failed: "
                          f"{type(exc).__name__}: {exc}")
            traceback.print_exception(exc, file=sys.stderr)
            _publish(job, "Failed", f"could not write output: {type(exc).__name__}: {exc}")
        else:
            _publish(job, "Stopped" if job.get("stop_requested") else "Completed", None)

    writing.add_done_callback(_done)
    # shield: a cancelled awaiter must not cancel the write (which would only detach the
    # callback from a thread that keeps running anyway).
    try:
        await asyncio.shield(writing)
    except Exception:  # noqa: BLE001 — already published as `Failed` by the callback
        # Letting it escape would turn a modeled `Failed` job into a bare 500 on the Stop
        # or Create that happened to await it (and botocore retries a 500).
        return


def _is_inflight(entry: dict[str, Any]) -> bool:
    task = entry.get("task")
    if task is None or task.done():
        return False
    pentry = fs.state.pending.get(entry["pending_id"])
    return pentry is None or pentry["future"].done()


async def _collect(job_id: str, record_id: str, snapshot: dict[str, Any],
                   fut: asyncio.Future, model: str, invocation_type: str) -> None:
    try:
        res = await fs.await_resolution(snapshot, fut, is_batch=True)
        kind = res.get("kind")
        if kind in ("cleared", "batch_override"):
            return
        if kind == "error":
            output: dict[str, Any] = {"error": {"errorCode": res["status"], "errorMessage": res["message"]}}
            usage = None
        elif invocation_type == "Converse":
            output = {"modelOutput": _converse.build_response(res, model, snapshot, _bedrock._latency_ms(snapshot))}
            usage = res["usage"]
        else:
            output = {"modelOutput": fs._build_non_stream_response(
                res["message_id"], model, res["content_blocks"], res["usage"],
                res.get("stop_reason"), res.get("stop_sequence"), res.get("stop_details"),
                snapshot.get("params"))}
            usage = res["usage"]
    except asyncio.CancelledError:
        raise
    except _converse.ConverseValidationError as e:
        output = {"error": {"errorCode": 400, "errorMessage": str(e)}}
        usage = None
    except BaseException as e:  # noqa: BLE001
        print(f"[puppetllm] batch-inference collector failed for {job_id}/{record_id}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        output = {"error": {"errorCode": 500, "errorMessage": f"internal: {type(e).__name__}: {str(e)[:200]}"}}
        usage = None
    async with fs.state.lock:
        job = fs.state.bedrock_jobs.get(job_id)
        if job is None:
            return
        entry = job["entries"].get(record_id)
        if entry is None or entry["output"] is not None or entry.get("cancelled"):
            return
        _record_outcome(job, entry, output)
        entry["usage"] = usage
        job["unresolved"] = max(0, job.get("unresolved", 0) - 1)
        job["last_modified"] = time.time()
        finalize = _ready_to_end(job)
    if finalize:
        await _finalize(job)


def _spawn(job_id: str, record_id: str, snapshot: dict[str, Any], fut: asyncio.Future,
           model: str, invocation_type: str) -> asyncio.Task:
    task = asyncio.get_running_loop().create_task(
        _collect(job_id, record_id, snapshot, fut, model, invocation_type))
    _collector_tasks.add(task)
    task.add_done_callback(_collector_tasks.discard)
    return task


def _sweep(job: dict[str, Any]) -> None:
    """Undo a job that is being rolled back: release its pendings and drop the history a
    fast responder may already have produced for it. Call within the lock."""
    for e in job["entries"].values():
        pentry = fs.state.pending.pop(e["pending_id"], None) if e.get("pending_id") else None
        if pentry is not None and not pentry["future"].done():
            pentry["future"].set_result({"_batch_override": "stopped"})
    fs.state.history[:] = [h for h in fs.state.history
                           if (h.get("request") or {}).get("job_id") != job["id"]]


# ── routes ────────────────────────────────────────────────────────────


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/model-invocation-job")
    async def create_job(request: Request) -> Any:
        body, errmsg = await fs._parse_json_body(request)
        if errmsg is not None:
            return _err(400, "ValidationException", errmsg)
        name = body.get("jobName")
        if not isinstance(name, str) or len(name) > 63 or not _JOB_NAME_RE.match(name):
            return _err(400, "ValidationException", "jobName: 1-63 alphanumeric / hyphen characters are required")
        model_id = body.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            return _err(400, "ValidationException", "modelId is required")
        role = body.get("roleArn")
        if not isinstance(role, str) or not _ROLE_RE.match(role):
            return _err(400, "ValidationException", "roleArn: an IAM role ARN is required")
        in_cfg = (body.get("inputDataConfig") or {}).get("s3InputDataConfig") if isinstance(body.get("inputDataConfig"), dict) else None
        out_cfg = (body.get("outputDataConfig") or {}).get("s3OutputDataConfig") if isinstance(body.get("outputDataConfig"), dict) else None
        if not isinstance(in_cfg, dict) or not isinstance(in_cfg.get("s3Uri"), str) or not in_cfg["s3Uri"].startswith("s3://"):
            return _err(400, "ValidationException", "inputDataConfig.s3InputDataConfig.s3Uri is required")
        if not isinstance(out_cfg, dict) or not isinstance(out_cfg.get("s3Uri"), str) or not out_cfg["s3Uri"].startswith("s3://"):
            return _err(400, "ValidationException", "outputDataConfig.s3OutputDataConfig.s3Uri is required")
        if in_cfg.get("s3InputFormat") not in (None, "JSONL"):
            return _err(400, "ValidationException", "s3InputFormat: must be JSONL")
        inv_type = body.get("modelInvocationType", "InvokeModel")
        if inv_type not in ("InvokeModel", "Converse"):
            return _err(400, "ValidationException", "modelInvocationType: must be InvokeModel or Converse")
        model_probe = _bedrock.normalize_model_id(model_id)
        if inv_type != "Converse" and model_probe.canonical == model_probe.raw:
            # `Converse` is a model-independent schema, so any model works there. An
            # `InvokeModel` body is model-specific and puppetllm only models the Anthropic
            # one: accepting a Titan or Llama job would hand back an Anthropic `modelOutput`.
            return _err(400, "ValidationException",
                        f"modelId: an InvokeModel batch job needs an Anthropic model in this "
                        f"emulation (got {model_id!r}); use modelInvocationType=Converse")
        cfg_err = _validate_configs(body, in_cfg, out_cfg)
        if cfg_err is not None:
            return _err(400, "ValidationException", cfg_err)
        timeout_h = body.get("timeoutDurationInHours", 24)
        if type(timeout_h) is not int or not (24 <= timeout_h <= 168):
            return _err(400, "ValidationException", "timeoutDurationInHours: must be an integer in [24, 168]")
        token = body.get("clientRequestToken")
        if token is not None and (not isinstance(token, str) or len(token) > 256
                                  or not _TOKEN_RE.match(token)):
            return _err(400, "ValidationException", "clientRequestToken: invalid")
        model = _bedrock.normalize_model_id(model_id)
        now = time.time()
        job_id = _new_job_id()
        job: dict[str, Any] = {
            "id": job_id, "arn": _arn(job_id), "name": name, "model_id": model_id,
            # `Submitted` until the input has been read: the job is in the registry from
            # this point on, so the name and the client token are reserved.
            "role_arn": role, "status": "Submitted", "submit_time": now, "last_modified": now,
            "end_time": None, "message": None, "client_token": token,
            "timeout_hours": timeout_h, "invocation_type": inv_type,
            "vpc_config": body.get("vpcConfig"), "tags": body.get("tags"),
            "input_config": {"s3InputDataConfig": in_cfg},
            "output_config": {"s3OutputDataConfig": out_cfg},
            "files": [], "entries": {}, "stopped_records": 0,
            "stop_requested": False,
            # Counters, not derived: `job_json` is polled and must not walk every entry
            # of a large job under the global state lock.
            "n_processed": 0, "n_errors": 0,
            # Outstanding records, kept as a counter: rescanning every entry from each
            # collector would be O(n^2) under the global lock on a large job.
            "unresolved": 0,
            # Registration in progress: suppresses auto-end while `entries` is partial (a
            # fast responder can answer the first record before the last one is registered).
            "creating": True,
        }
        # Claim the name and the client token in the SAME critical section that checks them.
        # Reading the input takes real time (a large JSONL runs in a worker thread), and a
        # client retrying inside that window must get the replay, not a second job.
        deadline = time.monotonic() + 60.0
        epoch = fs.state.clear_generation
        while True:
            async with fs.state.lock:
                if fs.state.clear_generation != epoch:
                    # /_control/clear happened while we waited for a twin: everything this
                    # request belongs to is gone, so it must not recreate it afterwards.
                    return _err(400, "ConflictException",
                                "request cleared: job cleared during creation")
                twin = next((j for j in fs.state.bedrock_jobs.values()
                             if token and j.get("client_token") == token), None)
                if twin is None:
                    if any(j["name"] == name
                           and j["status"] in ("Submitted", "InProgress", "Stopping")
                           for j in fs.state.bedrock_jobs.values()):
                        return _err(400, "ConflictException",
                                    f"a job named {name!r} is already running")
                    fs.state.bedrock_jobs[job_id] = job
                    break
                if not twin.get("creating"):
                    return JSONResponse({"jobArn": twin["arn"]})  # idempotent replay
            # The twin is still being created — reading its input, or registering records —
            # and may yet be rolled back (a missing object, a bad line, a duplicate
            # recordId). Replaying its ARN now could hand back a job that will not exist;
            # wait for its creation to settle, then replay or start afresh.
            if time.monotonic() > deadline:
                return _err(400, "ConflictException",
                            f"a job with clientRequestToken {token!r} is still being created")
            await asyncio.sleep(0.02)
        try:
            # Reading and parsing the whole JSONL is filesystem + CPU work: keep it off the
            # event loop so a large input does not stall health checks and other requests.
            files = await asyncio.to_thread(_read_input, in_cfg["s3Uri"])
            try:
                out_bucket, out_key = _s3.parse_s3_uri(out_cfg["s3Uri"])
                if not _s3.bucket_exists(out_bucket):
                    raise ValueError(f"outputDataConfig: output bucket does not exist: {out_bucket}")
                # Fail now, not after every record was processed: a 255-byte input name gets
                # a `.out` suffix the backing filesystem cannot hold, and an output prefix
                # that is an existing OBJECT cannot become a directory.
                out_prefix = out_key if (not out_key or out_key.endswith("/")) else out_key + "/"
                parts = [p for p in out_prefix.split("/") if p]
                for depth in range(1, len(parts) + 1):
                    ancestor = "/".join(parts[:depth])
                    if _s3.stat_object(out_bucket, ancestor) is not None:
                        raise ValueError(f"outputDataConfig: s3://{out_bucket}/{ancestor} is an "
                                         f"existing object; this emulation cannot write under it")
                for fname, _records in files:
                    _s3._safe_key(f"{out_prefix}{job_id}/{fname}.out")
                _s3._safe_key(f"{out_prefix}{job_id}/manifest.json.out")
            except _s3.S3Error as e:
                raise ValueError(f"outputDataConfig: {e}") from None
        except (ValueError, _s3.S3Error) as e:
            prefix = "" if str(e).startswith("outputDataConfig") else "inputDataConfig: "
            async with fs.state.lock:
                if job.get("stop_requested") and fs.state.bedrock_jobs.get(job_id) is job:
                    # Someone already holds this ARN (they stopped it). The real service
                    # would end such a job `Failed` from `Validating`; a 404 would deny it
                    # ever existed.
                    job["creating"] = False
                    job["status"] = "Failed"
                    job["message"] = f"{prefix}{e}"
                    job["end_time"] = job["last_modified"] = time.time()
                else:
                    fs.state.bedrock_jobs.pop(job_id, None)
            return _err(400, "ValidationException", f"{prefix}{e}")
        except BaseException:
            async with fs.state.lock:
                fs.state.bedrock_jobs.pop(job_id, None)
            raise
        async with fs.state.lock:
            if fs.state.bedrock_jobs.get(job_id) is not job:
                return _err(400, "ConflictException",
                            "request cleared: job cleared during creation")
            if not job.get("stop_requested"):
                job["status"] = "InProgress"  # a Stop accepted meanwhile keeps `Stopping`
        seq = 0
        try:
            for fname, records in files:
                ids: list[str] = []
                for rec in records:
                    seq += 1
                    if fs.state.bedrock_jobs.get(job_id) is not job:
                        # /_control/clear wiped the registry mid-registration: stop at once
                        # rather than registering more pendings a caller could still resolve.
                        raise _JobCleared()
                    rid_in = rec.get("recordId")
                    if rid_in is not None and (not isinstance(rid_in, str) or not rid_in):
                        raise ValueError(f"recordId must be a non-empty string (got {rid_in!r})")
                    rid = rid_in if rid_in is not None else f"record-{seq:07d}"
                    if rid in job["entries"]:
                        raise ValueError(f"duplicate recordId {rid!r}")
                    ids.append(rid)
                    model_input = rec["modelInput"]
                    entry: dict[str, Any] = {"pending_id": None, "task": None, "output": None,
                                             "usage": None, "cancelled": False,
                                             "model_input": model_input}
                    if job.get("stop_requested"):
                        # Stopped by another client while still registering (the real service
                        # accepts a Stop on a `Submitted` job): the rest is never processed.
                        entry["cancelled"] = True
                        job["entries"][rid] = entry
                        job["stopped_records"] += 1
                        continue
                    # NOT in `entries` yet: `register_request` awaits, and a Stop that ran
                    # meanwhile would find an entry with no pending and mis-count it. It is
                    # inserted once its outcome — pending or error record — is known.
                    # Per-record validation failures become `error` records, like the real
                    # service (the job itself is accepted).
                    try:
                        _reject_unsupported(model_input, inv_type)
                        if inv_type == "Converse":
                            canonical = _converse.to_canonical(model_input)
                            cv = _converse.converse_extras(model_input)
                            # same pointer-syntax check the direct route does, so a bad path
                            # is a 400 error record instead of a late internal failure
                            _converse.additional_fields(
                                {}, cv.get("additionalModelResponseFieldPaths"))
                            extra = {"api": "converse", "converse": cv}
                        else:
                            v = _bedrock._validate_version(model_input)
                            if v is not None:
                                raise _converse.ConverseValidationError(v)
                            canonical = model_input
                            extra = {}
                        snapshot, fut = await fs.register_request(
                            # AWS: "Prompt caching ... is not supported with the batch
                            # inference API", so a batch record never observes the cache.
                            "bedrock", model.canonical, canonical, is_stream=False,
                            simulate_cache=False,
                            extra={"job_arn": job["arn"], "job_id": job_id, "record_id": rid,
                                   "bedrock_model_id": model.raw, **extra})
                    except (_converse.ConverseValidationError, fs.RequestValidationError) as e:
                        _record_outcome(job, entry,
                                        {"error": {"errorCode": 400, "errorMessage": str(e)}})
                        job["entries"][rid] = entry
                        continue
                    except (TypeError, ValueError, KeyError, AttributeError, IndexError) as e:
                        # A malformed `modelInput` is client input: a validation error record,
                        # the same class the real service reports when it processes the batch.
                        _record_outcome(job, entry, {"error": {
                            "errorCode": 400,
                            "errorMessage": f"invalid modelInput: {type(e).__name__}: {str(e)[:200]}"}})
                        job["entries"][rid] = entry
                        continue
                    except Exception as e:  # noqa: BLE001 — a record must never wedge the job
                        _record_outcome(job, entry, {"error": {
                            "errorCode": 500,
                            "errorMessage": f"{type(e).__name__}: {str(e)[:200]}"}})
                        job["entries"][rid] = entry
                        continue
                    entry["pending_id"] = snapshot["pending_id"]
                    if job.get("stop_requested"):
                        # A Stop landed while this record was being registered: it already
                        # walked `entries` without seeing this one, so cancel it here.
                        pentry = fs.state.pending.pop(snapshot["pending_id"], None)
                        if pentry is not None and not pentry["future"].done():
                            pentry["future"].set_result({"_batch_override": "stopped"})
                        entry["cancelled"] = True
                        job["entries"][rid] = entry
                        job["stopped_records"] += 1
                        continue
                    job["entries"][rid] = entry
                    job["unresolved"] += 1
                    entry["task"] = _spawn(job_id, rid, snapshot, fut, model.canonical, inv_type)
                    if seq % 200 == 0:
                        await asyncio.sleep(0)  # keep the event loop responsive on a big job
                job["files"].append((fname, ids))
        except _JobCleared:
            async with fs.state.lock:
                _sweep(job)
                fs.state.bedrock_jobs.pop(job_id, None)
            return _err(400, "ConflictException",
                        "request cleared: job cleared during creation")
        except ValueError as e:
            async with fs.state.lock:
                _sweep(job)
                if job.get("stop_requested") and fs.state.bedrock_jobs.get(job_id) is job:
                    # Same rule as a read-phase failure: someone already holds this ARN.
                    job["creating"] = False
                    job["status"] = "Failed"
                    job["message"] = f"inputDataConfig: {e}"
                    job["end_time"] = job["last_modified"] = time.time()
                else:
                    fs.state.bedrock_jobs.pop(job_id, None)
            return _err(400, "ValidationException", f"inputDataConfig: {e}")
        except BaseException:
            # Anything else (including cancellation) must leave no half-built job behind.
            _sweep(job)
            fs.state.bedrock_jobs.pop(job_id, None)
            raise
        async with fs.state.lock:
            if fs.state.bedrock_jobs.get(job_id) is not job:
                # /_control/clear wiped the registry while records were still registering:
                # sweep what this loop created so nothing survives the clear.
                _sweep(job)
                return _err(400, "ConflictException",
                            "request cleared: job cleared during creation")
            job["creating"] = False
            finalize = _ready_to_end(job)  # every record may already be errored
        if finalize:
            await _finalize(job)
            async with fs.state.lock:
                if fs.state.bedrock_jobs.get(job_id) is not job:
                    return _err(400, "ConflictException",
                                "request cleared: job cleared during creation")
        _bedrock._log(f"batch job {job_id} ({name}) created: {len(job['entries'])} records, "
                      f"type={inv_type} model={model.raw}")
        return JSONResponse({"jobArn": job["arn"]})

    @router.get("/model-invocation-job/{job_identifier:path}")
    async def get_job(job_identifier: str) -> Any:
        if job_identifier.endswith("/stop"):
            return JSONResponse({"detail": "Method Not Allowed"}, status_code=405,
                                headers={"Allow": "POST"})
        if not _valid_identifier(job_identifier):
            return _err(400, "ValidationException",
                        f"jobIdentifier: must be a 12-character job id or its ARN "
                        f"(got {job_identifier!r})")
        async with fs.state.lock:
            job = _resolve_job(job_identifier)
            if job is None:
                return _err(404, "ResourceNotFoundException", f"job not found: {job_identifier}")
            return JSONResponse(job_json(job))

    @router.post("/model-invocation-job/{job_identifier:path}/stop")
    async def stop_job(job_identifier: str) -> Any:
        if not _valid_identifier(job_identifier):
            return _err(400, "ValidationException",
                        f"jobIdentifier: must be a 12-character job id or its ARN "
                        f"(got {job_identifier!r})")
        async with fs.state.lock:
            job = _resolve_job(job_identifier)
            if job is None:
                return _err(404, "ResourceNotFoundException", f"job not found: {job_identifier}")
            if job["status"] in ("Stopping", "Stopped"):
                # Idempotent, like the real service: botocore retries a Stop whose response
                # it lost, and that retry must not fail because the first one succeeded.
                return Response(status_code=200)
            if job["status"] not in ("Submitted", "InProgress"):
                return _err(400, "ConflictException", f"job is {job['status']}")
            now = time.time()
            job["status"] = "Stopping"
            # Durable: a finalization already in flight re-reads this before publishing, so
            # an accepted Stop can never settle back to `Completed`.
            job["stop_requested"] = True
            job["last_modified"] = now
            for e in job["entries"].values():
                if e["output"] is not None or e.get("cancelled") or _is_inflight(e):
                    continue
                # Unblock the pending (if the record ever got one) and mark the record as
                # never processed: it is excluded from the counters and from the output file.
                if e.get("pending_id"):
                    pentry = fs.state.pending.pop(e["pending_id"], None)
                    if pentry is not None and not pentry["future"].done():
                        pentry["future"].set_result({"_batch_override": "stopped"})
                e["cancelled"] = True
                job["stopped_records"] += 1
                job["unresolved"] = max(0, job.get("unresolved", 0) - 1)
            finalize = _ready_to_end(job)
        if finalize:
            await _finalize(job)
        return Response(status_code=200)

    @router.get("/model-invocation-jobs")
    async def list_jobs(request: Request) -> Any:
        qp = request.query_params
        status = qp.get("statusEquals")
        if status is not None and status not in STATUSES:
            return _err(400, "ValidationException", f"statusEquals: must be one of {list(STATUSES)}")
        sort_by = qp.get("sortBy")
        if sort_by is not None and sort_by != "CreationTime":
            return _err(400, "ValidationException", "sortBy: must be CreationTime")
        sort_order = qp.get("sortOrder")
        if sort_order is not None and sort_order not in ("Ascending", "Descending"):
            return _err(400, "ValidationException", "sortOrder: must be Ascending or Descending")
        name_filter = qp.get("nameContains")
        if name_filter is not None and not (len(name_filter) <= 63
                                            and _JOB_NAME_RE.match(name_filter)):
            return _err(400, "ValidationException", "nameContains: invalid")
        bounds: dict[str, float] = {}
        for param in ("submitTimeAfter", "submitTimeBefore"):
            raw = qp.get(param)
            if raw is None:
                continue
            ts = _parse_timestamp(raw)
            if ts is None:
                return _err(400, "ValidationException",
                            f"{param}: must be an ISO-8601 timestamp or epoch seconds")
            bounds[param] = ts
        raw_max = qp.get("maxResults", "1000")
        if not raw_max.isdigit():  # bare int() would take "1_0", " 5 " and "+5"
            return _err(400, "ValidationException", "maxResults: must be an integer")
        max_results = int(raw_max)
        if not (1 <= max_results <= 1000):
            return _err(400, "ValidationException", "maxResults: must be in [1, 1000]")
        name_contains = name_filter
        raw_token = qp.get("nextToken", "0")
        if not raw_token.isdigit():
            return _err(400, "ValidationException", "nextToken: invalid")
        offset = int(raw_token)
        async with fs.state.lock:
            jobs = list(fs.state.bedrock_jobs.values())
        if status:
            jobs = [j for j in jobs if j["status"] == status]
        if name_contains:
            jobs = [j for j in jobs if name_contains in j["name"]]
        if "submitTimeAfter" in bounds:
            jobs = [j for j in jobs if j["submit_time"] >= bounds["submitTimeAfter"]]
        if "submitTimeBefore" in bounds:
            jobs = [j for j in jobs if j["submit_time"] <= bounds["submitTimeBefore"]]
        jobs.sort(key=lambda j: j["submit_time"], reverse=((sort_order or "Descending") != "Ascending"))
        page = jobs[offset:offset + max_results]
        out: dict[str, Any] = {"invocationJobSummaries": [job_json(j) for j in page]}
        if offset + max_results < len(jobs):
            out["nextToken"] = str(offset + max_results)
        return JSONResponse(out)

    # ── control ──

    @router.get("/_control/bedrock_jobs")
    async def control_jobs() -> Any:
        async with fs.state.lock:
            jobs = [{**job_json(j),
                      "unresolved": [rid for rid, e in j["entries"].items()
                                     if e["output"] is None and not e.get("cancelled")],
                      "cancelled": [rid for rid, e in j["entries"].items() if e.get("cancelled")]}
                    for j in fs.state.bedrock_jobs.values()]
        return {"count": len(jobs), "jobs": jobs}

    return router
