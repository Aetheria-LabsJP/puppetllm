"""Anthropic Message Batches compatible adapter + batch control endpoints.

Formal spec: README.md

Implements the Message Batches surface (`/v1/messages/batches*`) on top of the
canonical core. Each batch sub-request (custom_id) is registered as a **normal
pending** via `fs.register_request` — the responder sees it through the existing
`/_control/pending` / `wait_for_pending` (the snapshot additionally carries
`batch_id` / `custom_id`) and injects with the existing `/_control/respond` /
`auto` / `error`, addressed either by `pending_id` or by `custom_id` (+ optional
`batch_id`). Instead of an HTTP handler awaiting the pending's future, a
background collector task per custom_id awaits it and stores the outcome into
the batch registry (`fs.state.batches`).

Lifecycle: a batch is created `in_progress` and transitions to `ended`
automatically once every custom_id has a result. The control plane can force
the transition (`/_control/batch/end`) or inject the result types that
respond/error cannot express (`canceled` / `expired`, via
`/_control/batch/result`).

Deliberate divergences from the real API (determinism over fidelity):
- **No wall-clock expiration**: `expires_at` (created + 24h) is reported but
  nothing expires by the clock — expiration only happens via control injection.
- **Cancel is usually immediate**: `POST .../cancel` resolves every unresolved
  custom_id as `canceled` and ends the batch in the same call, so SDK poll loops
  normally see `ended` right away instead of the real API's asynchronous
  `canceling` phase. An injection already in flight at that instant (accepted,
  collector not yet stored) still completes as succeeded/errored — as on the real
  API, where already-processing requests may finish after a cancel — and the batch
  reports `canceling` until those land.
- **Per-request params are not deep-validated** (only custom_id uniqueness,
  params being an object, and the no-streaming rule). Invalid params surface
  when the responder inspects the pending, not as an `errored` result.
- Costs recorded to history/stats get the real API's 50% batch discount
  (`cost.batch_discount = 0.5`, history entry `batch: true`).

Endpoints:
- POST   /v1/messages/batches                    — create (returns in_progress)
- GET    /v1/messages/batches                    — list (after_id/before_id/limit)
- GET    /v1/messages/batches/{id}               — retrieve (poll target)
- GET    /v1/messages/batches/{id}/results       — JSONL results (only after ended)
- POST   /v1/messages/batches/{id}/cancel        — cancel (usually immediate, see above)
- DELETE /v1/messages/batches/{id}               — delete (only after ended)
- GET    /_control/batches                       — batch registry (status/counts/unresolved)
- POST   /_control/batch/end                     — force-end; unresolved → expired|canceled
- POST   /_control/batch/result                  — per-custom_id canceled/expired injection
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

# canonical core. WARNING: circular import (same constraint as providers/*):
# fake_server imports this module at its end and calls build_router(). Here we hold
# only a module reference; attributes like `fs.register_request` must always be
# resolved at call-time.
from . import fake_server as fs

# Real-API display value only — nothing in this server expires by the clock.
_BATCH_EXPIRES_SECONDS = 24 * 3600.0

# Real-API limits, enforced so an app that would be rejected there is rejected here
# too (silently accepting what production refuses defeats the point of the proxy).
_CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_REQUESTS_PER_BATCH = 100_000

# Strong references to collector tasks (a bare create_task result may be GC'd mid-flight).
_collector_tasks: set[asyncio.Task] = set()


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _request_counts(batch: dict[str, Any]) -> dict[str, int]:
    counts = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
    for e in batch["entries"].values():
        r = e["result"]
        if r is None:
            counts["processing"] += 1
        else:
            # result types are produced internally, so the key is always present
            counts[r["type"]] += 1
    return counts


def _batch_json(batch: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Encode a batch in the official MessageBatch shape (SDK/pydantic compatible).

    `results_url` must be an **absolute** URL (the SDK GETs it verbatim), so it is
    derived from request.base_url rather than a config value. Caveat: behind a
    reverse proxy this reflects X-Forwarded-* only if uvicorn is run with
    --proxy-headers (and FORWARDED_ALLOW_IPS covering the proxy) — otherwise it
    points at the internal listener. Fine for the intended localhost use.
    """
    ended = batch["processing_status"] == "ended"
    return {
        "id": batch["id"],
        "type": "message_batch",
        "processing_status": batch["processing_status"],
        "request_counts": _request_counts(batch),
        "created_at": _iso(batch["created_at"]),
        "expires_at": _iso(batch["expires_at"]),
        "ended_at": _iso(batch["ended_at"]),
        "cancel_initiated_at": _iso(batch["cancel_initiated_at"]),
        "archived_at": None,
        "results_url": (
            f"{base_url}/v1/messages/batches/{batch['id']}/results" if ended else None
        ),
    }


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _maybe_end(batch: dict[str, Any], now: float) -> None:
    """Transition to ended once every custom_id has a result. Call within the lock.

    Also settles a `canceling` batch (cancel left injections in flight; this runs
    again from the collector when the last one lands).

    Suppressed while the batch is still being created: `entries` is only partially
    populated then, and an "all resolved" over a partial (or empty) dict must not
    end the batch — entries registered afterwards would be stuck with a null result.
    create_batch re-runs this after clearing the flag.
    """
    if batch.get("creating") or batch["processing_status"] == "ended":
        return
    if all(e["result"] is not None for e in batch["entries"].values()):
        batch["processing_status"] = "ended"
        batch["ended_at"] = now


def _is_inflight(entry: dict[str, Any]) -> bool:
    """Is this unresolved entry's outcome already on its way from its collector?

    True only while a **live** collector task owns the outcome: the injection was
    accepted (future resolved / pending already reaped by `_record_and_reset`) and
    the task has not finished yet. Such entries must not be finalized — their
    200-acked injection would be silently discarded.

    A finished task with no stored result means nothing will ever store one (the
    collector unwound abnormally), so that entry is finalizable — otherwise it
    would be unfinalizable forever and the batch could never end.
    """
    task = entry.get("task")
    if task is None or task.done():
        return False
    pentry = fs.state.pending.get(entry["pending_id"])
    return pentry is None or pentry["future"].done()


def _finalize_unresolved(
    batch: dict[str, Any], rtype: str, now: float,
    custom_ids: set[str] | None = None,
) -> int:
    """Synchronously resolve unresolved entries as `rtype` (canceled|expired). Within the lock.

    Sets the entry result directly (deterministic — the caller's HTTP response already
    reflects the final state) and wakes the collector with a `_batch_override` sentinel,
    which it treats as "already finalized, nothing to store". The pending is popped here
    so responders never see a ghost. Returns the number of entries finalized.

    Entries that are in flight (see `_is_inflight`) are SKIPPED so their injection
    completes as succeeded/errored — mirroring the real API, where already-processing
    requests may still complete after a cancel. The collector's own `_maybe_end`
    settles the batch once those land.
    """
    n = 0
    for cid, entry in batch["entries"].items():
        if custom_ids is not None and cid not in custom_ids:
            continue
        if entry["result"] is not None or _is_inflight(entry):
            continue
        pentry = fs.state.pending.pop(entry["pending_id"], None)
        if pentry is not None and not pentry["future"].done():
            pentry["future"].set_result({"_batch_override": rtype})
        entry["result"] = {"type": rtype}
        n += 1
    _maybe_end(batch, now)
    return n


def _inflight_count(batch: dict[str, Any]) -> int:
    return sum(1 for e in batch["entries"].values()
               if e["result"] is None and _is_inflight(e))


def _sweep_pendings(batch: dict[str, Any]) -> None:
    """Pop the batch's registered pendings; wake their collectors with a no-op sentinel.

    Used when the batch object itself goes away mid-creation (rollback, or clear
    raced the creation loop). Entries whose future is already resolved are left to
    their collector to unwind: the pending pop here suppresses its history record
    (`_record_and_reset` only appends while the pending exists) and the result store
    finds the batch gone. Call from the event loop (sync) or within the lock.
    """
    for e in batch["entries"].values():
        pentry = fs.state.pending.pop(e["pending_id"], None)
        if pentry is not None and not pentry["future"].done():
            pentry["future"].set_result({"_batch_override": "canceled"})


def _purge_history(batch_id: str) -> None:
    """Drop history entries belonging to a batch that is being un-created.

    A responder may have answered an early entry while the creation loop was still
    running; its collector records to history through the normal path. When the
    create then fails (rollback) or the registry is wiped (clear), leaving that
    record behind would bill a request for a batch that does not exist and would
    contradict the "a failed create leaves nothing behind" contract. Mutates the
    list in place so concurrent holders of `state.history` see the same object.

    `state.turn_count` and the pseudo-cache index are intentionally NOT rewound:
    both are monotonic observations of "a request was seen", shared with unrelated
    in-flight requests, and cannot be attributed back safely.
    """
    fs.state.history[:] = [
        h for h in fs.state.history
        if (h.get("request") or {}).get("batch_id") != batch_id
    ]


def _rollback_creation(batch: dict[str, Any]) -> None:
    """Remove a half-created batch, every pending it registered, and its history.

    Deliberately synchronous (same reasoning as fake_server._discard_pending): dict
    mutations inside the event loop are atomic, and this must also run while a
    CancelledError is propagating, where an `await state.lock` could itself be
    cancelled and skip the cleanup — exactly the ghost this prevents.
    """
    if fs.state.batches.get(batch["id"]) is batch:
        del fs.state.batches[batch["id"]]
    _sweep_pendings(batch)
    _purge_history(batch["id"])


def _errored_result(etype: str, message: str) -> dict[str, Any]:
    """The `errored` result shape (the official error envelope nested under `error`)."""
    return {
        "type": "errored",
        "error": {"type": "error", "error": {"type": etype, "message": message}},
    }


async def _resolve_result(snapshot: dict[str, Any],
                          fut: asyncio.Future) -> dict[str, Any] | None:
    """Await one custom_id's resolution and encode it as a batch result.

    Returns None when there is nothing to store: "cleared" (/_control/clear wiped the
    registry) or "batch_override" (a control endpoint already finalized this entry).
    """
    res = await fs.await_resolution(snapshot, fut, is_batch=True)
    kind = res.get("kind")
    if kind in ("cleared", "batch_override"):
        return None
    if kind == "error":
        return _errored_result(res["type"], res["message"])
    # "ok" — same message shape as the non-stream Anthropic route
    model_out = snapshot.get("model") or "claude-sonnet-mock"
    message = fs._build_non_stream_response(
        res["message_id"], model_out, res["content_blocks"], res["usage"],
        res.get("stop_reason"),
    )
    return {"type": "succeeded", "message": message}


async def _collect(batch_id: str, custom_id: str,
                   snapshot: dict[str, Any], fut: asyncio.Future) -> None:
    """Carry one custom_id to completion and store its result into the batch registry.

    This replaces the "HTTP handler awaits the future" role of the normal routes: the
    batch create call has long since returned, so a background task carries the pending
    to completion. History/stats recording happens inside await_resolution exactly as
    for the normal routes (with is_batch=True → 50% cost discount + batch tag).

    Any unexpected failure is converted into an `errored` result rather than killing
    the task: a collector that dies with the entry still unresolved would leave the
    batch unable to ever reach `ended` (and only a full /_control/clear would recover
    it), with the traceback visible solely through asyncio's "never retrieved" warning.
    """
    try:
        result = await _resolve_result(snapshot, fut)
    except asyncio.CancelledError:
        raise
    except BaseException as e:  # noqa: BLE001 — deliberate catch-all, see docstring
        print(f"[puppetllm] batch collector failed for "
              f"{batch_id}/{custom_id}: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        result = _errored_result(
            "api_error",
            f"puppetllm: batch entry failed internally: {type(e).__name__}: {str(e)[:200]}")
    if result is None:
        return
    async with fs.state.lock:
        batch = fs.state.batches.get(batch_id)
        if batch is None:
            return  # cleared while we were encoding
        entry = batch["entries"].get(custom_id)
        if entry is None or entry["result"] is not None:
            return  # finalized by a control endpoint in the meantime
        entry["result"] = result
        _maybe_end(batch, time.time())


def _spawn_collector(batch_id: str, custom_id: str,
                     snapshot: dict[str, Any], fut: asyncio.Future) -> asyncio.Task:
    """Start the collector and return its task (stored on the entry for _is_inflight)."""
    task = asyncio.get_running_loop().create_task(_collect(batch_id, custom_id, snapshot, fut))
    _collector_tasks.add(task)
    task.add_done_callback(_collector_tasks.discard)
    return task


def _target_batch(batch_id: str | None) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    """Resolve a control-endpoint target batch. Call within the lock.

    batch_id omitted → allowed only when exactly one non-ended batch exists
    (mirrors the pending_id-omitted convention of /_control/respond).

    Batches still being created are NOT targetable: force-ending one would run
    `all(...)` over a partially-registered entries dict, ending the batch while
    later entries are still on their way (null results + ghost pendings).
    """
    if batch_id is not None:
        if not isinstance(batch_id, str):
            return None, fs._plain_400("batch_id must be a string")
        batch = fs.state.batches.get(batch_id)
        if batch is None:
            return None, fs._plain_400(f"unknown batch_id: {batch_id}")
        if batch.get("creating"):
            return None, fs._plain_400(
                f"batch {batch_id} is still being created; retry shortly")
        return batch, None
    live = [b for b in fs.state.batches.values()
            if b["processing_status"] != "ended" and not b.get("creating")]
    if not live:
        if any(b.get("creating") for b in fs.state.batches.values()):
            return None, fs._plain_400(
                "no targetable batch (a batch is still being created; retry shortly)")
        return None, fs._plain_400("no in-progress batch")
    if len(live) > 1:
        return None, JSONResponse(
            {"error": "multiple in-progress batches; specify batch_id",
             "batch_ids": [b["id"] for b in live]},
            status_code=400,
        )
    return live[0], None


def build_router() -> APIRouter:
    router = APIRouter()

    # ── Anthropic-compatible batch endpoints ─────────────────────────

    @router.post("/v1/messages/batches")
    async def create_batch(request: Request) -> Any:
        body, errmsg = await fs._parse_json_body(request)
        if errmsg is not None:
            return fs._anthropic_error(400, "invalid_request_error", errmsg)
        requests_in = body.get("requests")
        if not isinstance(requests_in, list) or not requests_in:
            return fs._anthropic_error(400, "invalid_request_error",
                                       "requests must be a non-empty array")
        if len(requests_in) > _MAX_REQUESTS_PER_BATCH:
            return fs._anthropic_error(
                400, "invalid_request_error",
                f"requests exceeds the maximum of {_MAX_REQUESTS_PER_BATCH} per batch")
        seen: set[str] = set()
        for i, item in enumerate(requests_in):
            if not isinstance(item, dict):
                return fs._anthropic_error(400, "invalid_request_error",
                                           f"requests[{i}] must be an object")
            cid = item.get("custom_id")
            if not isinstance(cid, str) or not _CUSTOM_ID_RE.match(cid):
                return fs._anthropic_error(
                    400, "invalid_request_error",
                    f"requests[{i}].custom_id must match ^[a-zA-Z0-9_-]{{1,64}}$")
            if cid in seen:
                return fs._anthropic_error(400, "invalid_request_error",
                                           f"duplicate custom_id: {cid}")
            seen.add(cid)
            params = item.get("params")
            if not isinstance(params, dict):
                return fs._anthropic_error(400, "invalid_request_error",
                                           f"requests[{i}].params must be an object")
            if params.get("stream"):
                return fs._anthropic_error(
                    400, "invalid_request_error",
                    f"requests[{i}].params.stream: streaming is not supported in batches")

        now = time.time()
        batch_id = f"msgbatch_{uuid.uuid4().hex[:24]}"
        batch: dict[str, Any] = {
            "id": batch_id,
            "created_at": now,
            "expires_at": now + _BATCH_EXPIRES_SECONDS,
            "ended_at": None,
            "cancel_initiated_at": None,
            "processing_status": "in_progress",
            # Registration in progress: suppresses auto-end and makes the batch
            # untargetable by /_control/batch/* and cancel until creation completes
            # (a force-end over a partially-registered entries dict would strand the
            # rest as null results + ghost pendings).
            "creating": True,
            "entries": {},  # custom_id → {"pending_id", "result"} (insertion order = request order)
        }
        # Register the batch object BEFORE the pendings: a responder can inject the
        # instant a pending appears, and the collector must be able to find the batch.
        async with fs.state.lock:
            fs.state.batches[batch_id] = batch
        try:
            for i, item in enumerate(requests_in):
                cid = item["custom_id"]
                params = item["params"]
                try:
                    snapshot, fut = await fs.register_request(
                        "anthropic", params.get("model"), params, is_stream=False,
                        extra={"batch_id": batch_id, "custom_id": cid},
                    )
                except Exception as e:
                    # Shallow validation deliberately lets malformed params through, so
                    # register_request (token analysis etc.) can fail mid-loop. Roll back
                    # everything already registered — a failed create must leave no
                    # half-built batch and no ghost pendings.
                    _rollback_creation(batch)
                    return fs._anthropic_error(
                        400, "invalid_request_error",
                        f"requests[{i}].params could not be processed: "
                        f"{type(e).__name__}: {str(e)[:200]}")
                # No await between register_request returning and these assignments, so
                # a concurrent injection cannot observe the entry missing. The collector
                # task is kept on the entry so _is_inflight can tell "outcome on its way"
                # from "collector gone, nobody will ever store a result".
                entry = {"pending_id": snapshot["pending_id"], "result": None, "task": None}
                batch["entries"][cid] = entry
                entry["task"] = _spawn_collector(batch_id, cid, snapshot, fut)
        except BaseException:
            # Task cancellation (client disconnect) mid-registration: same rollback,
            # synchronous so it survives CancelledError propagation.
            _rollback_creation(batch)
            raise
        async with fs.state.lock:
            if fs.state.batches.get(batch_id) is not batch:
                # /_control/clear wiped the registry while entries were still being
                # registered. Sweep the pendings this loop created after the clear (and
                # any history they already produced) so nothing survives the clear, then
                # tell the caller — same 503 shape as a cleared /v1/messages request.
                _sweep_pendings(batch)
                _purge_history(batch_id)
                return fs._anthropic_error(503, "api_error",
                                           "request cleared: batch cleared during creation")
            batch["creating"] = False
            # A fast responder may have answered every entry while registration was
            # still running (auto-end is suppressed by the creating flag) — settle now.
            _maybe_end(batch, time.time())
            return JSONResponse(_batch_json(batch, _base_url(request)))

    @router.get("/v1/messages/batches")
    async def list_batches(request: Request) -> Any:
        # Query params are parsed by hand: FastAPI's typed params would answer a bad
        # `limit` with its own 422 `{"detail": [...]}` shape, while every other error
        # on the /v1 surface uses the Anthropic envelope.
        qp = request.query_params
        try:
            limit = int(qp.get("limit", "20"))
        except ValueError:
            return fs._anthropic_error(400, "invalid_request_error",
                                       "limit must be an integer")
        # Out-of-range values are rejected rather than clamped, and unknown cursors
        # rejected rather than ignored: silently accepting what the real API refuses
        # would let a paginating app pass here and fail in production.
        if not (1 <= limit <= 100):
            return fs._anthropic_error(400, "invalid_request_error",
                                       "limit must be in [1, 100]")
        after_id = qp.get("after_id")
        before_id = qp.get("before_id")
        async with fs.state.lock:
            ordered = list(fs.state.batches.values())[::-1]  # newest first
            ids = [b["id"] for b in ordered]
            for name, cursor in (("after_id", after_id), ("before_id", before_id)):
                if cursor is not None and cursor not in ids:
                    return fs._anthropic_error(
                        404, "not_found_error",
                        f"{name} cursor not found: {cursor}")
            start, end = 0, len(ordered)
            if after_id is not None:
                start = ids.index(after_id) + 1
            if before_id is not None:
                end = ids.index(before_id)
            window = ordered[start:end]
            # before_id pages backward (take the items adjacent to the cursor)
            page = window[-limit:] if (before_id is not None and after_id is None) else window[:limit]
            data = [_batch_json(b, _base_url(request)) for b in page]
            has_more = len(window) > len(page)
        return {
            "data": data,
            "has_more": has_more,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }

    @router.get("/v1/messages/batches/{batch_id}")
    async def retrieve_batch(batch_id: str, request: Request) -> Any:
        async with fs.state.lock:
            batch = fs.state.batches.get(batch_id)
            if batch is None:
                return fs._anthropic_error(404, "not_found_error",
                                           f"message batch not found: {batch_id}")
            return JSONResponse(_batch_json(batch, _base_url(request)))

    @router.get("/v1/messages/batches/{batch_id}/results")
    async def batch_results(batch_id: str) -> Any:
        async with fs.state.lock:
            batch = fs.state.batches.get(batch_id)
            if batch is None:
                return fs._anthropic_error(404, "not_found_error",
                                           f"message batch not found: {batch_id}")
            if batch["processing_status"] != "ended":
                return fs._anthropic_error(
                    400, "invalid_request_error",
                    f"batch {batch_id} has not ended; results are not available yet")
            # Snapshot the lines under the lock; results order = request order
            # (the real API makes no ordering promise — key by custom_id).
            lines = [
                json.dumps({"custom_id": cid, "result": e["result"]}, ensure_ascii=False) + "\n"
                for cid, e in batch["entries"].items()
            ]

        async def gen():
            for line in lines:
                yield line.encode("utf-8")

        return StreamingResponse(gen(), media_type="application/x-jsonl")

    @router.post("/v1/messages/batches/{batch_id}/cancel")
    async def cancel_batch(batch_id: str, request: Request) -> Any:
        now = time.time()
        async with fs.state.lock:
            batch = fs.state.batches.get(batch_id)
            if batch is None:
                return fs._anthropic_error(404, "not_found_error",
                                           f"message batch not found: {batch_id}")
            if batch.get("creating"):
                # Only reachable via an id picked up from list/_control while the
                # create call is still registering entries (the creator itself has no
                # id yet) — cancelling now would strand the not-yet-registered rest.
                return fs._anthropic_error(
                    400, "invalid_request_error",
                    f"batch {batch_id} is still being created; retry shortly")
            if batch["processing_status"] != "ended":
                # Resolve everything cancellable as canceled and end in the same call.
                # Injections already in flight keep going (see _finalize_unresolved), and
                # while any remain the batch reports the real API's "canceling" until the
                # collectors land — never a bare "in_progress" with cancel_initiated_at.
                if batch["cancel_initiated_at"] is None:
                    batch["cancel_initiated_at"] = now
                _finalize_unresolved(batch, "canceled", now)
                if batch["processing_status"] != "ended":
                    batch["processing_status"] = "canceling"
            return JSONResponse(_batch_json(batch, _base_url(request)))

    @router.delete("/v1/messages/batches/{batch_id}")
    async def delete_batch(batch_id: str) -> Any:
        async with fs.state.lock:
            batch = fs.state.batches.get(batch_id)
            if batch is None:
                return fs._anthropic_error(404, "not_found_error",
                                           f"message batch not found: {batch_id}")
            if batch["processing_status"] != "ended":
                return fs._anthropic_error(
                    400, "invalid_request_error",
                    "cannot delete a batch that has not ended (cancel it first)")
            del fs.state.batches[batch_id]
        return {"id": batch_id, "type": "message_batch_deleted"}

    # ── Control endpoints (batch injection) ──────────────────────────

    @router.get("/_control/batches")
    async def control_batches() -> Any:
        """Batch registry view: status, counts, and which custom_ids still need injection.

        Timestamps are epoch seconds here, matching the rest of the control plane
        (`/_control/history`'s `completed_at`, pending `received_at`) rather than the
        ISO-8601 that the public `/v1/messages/batches*` route must emit for the SDK.
        """
        async with fs.state.lock:
            out = []
            for batch in fs.state.batches.values():
                out.append({
                    "batch_id": batch["id"],
                    "processing_status": batch["processing_status"],
                    "creating": bool(batch.get("creating")),
                    "created_at": batch["created_at"],  # epoch seconds (see docstring)
                    "ended_at": batch["ended_at"],
                    "request_counts": _request_counts(batch),
                    "unresolved_custom_ids": [
                        cid for cid, e in batch["entries"].items() if e["result"] is None
                    ],
                    "entries": {
                        cid: {
                            "pending_id": e["pending_id"],
                            "resolved": e["result"] is not None,
                            "result_type": e["result"]["type"] if e["result"] else None,
                        }
                        for cid, e in batch["entries"].items()
                    },
                })
        return {"count": len(out), "batches": out}

    @router.post("/_control/batch/end")
    async def control_batch_end(request: Request) -> Any:
        """Force a batch to `ended` now.

        Body: {"batch_id"?: "...", "unresolved"?: "expired" | "canceled"} (default "expired").
        Unresolved custom_ids get the given result type; already-injected results are kept.
        batch_id can be omitted when exactly one non-ended batch exists.

        Returns `{"finalized", "ended", "in_flight"}`: an injection accepted but not yet
        stored by its collector is left to complete, so `ended` can be false — the batch
        settles a moment later when those land.
        """
        body, errmsg = await fs._parse_json_body(request)
        if errmsg is not None:
            return fs._plain_400(errmsg)
        rtype = body.get("unresolved", "expired")
        if rtype not in ("expired", "canceled"):
            return fs._plain_400('unresolved must be "expired" or "canceled"')
        now = time.time()
        async with fs.state.lock:
            batch, err = _target_batch(body.get("batch_id"))
            if err is not None:
                return err
            if batch["processing_status"] == "ended":
                return fs._plain_400(f"batch {batch['id']} already ended")
            n = _finalize_unresolved(batch, rtype, now)
            ended = batch["processing_status"] == "ended"
            in_flight = _inflight_count(batch)
        # `ended` can be false when an injection was accepted but its collector has not
        # stored yet: those complete as succeeded/errored and settle the batch right
        # after. Reported explicitly so the caller doesn't read a bare ok as "ended".
        return {"ok": True, "batch_id": batch["id"], "finalized": n,
                "unresolved_as": rtype, "ended": ended, "in_flight": in_flight}

    @router.post("/_control/batch/result")
    async def control_batch_result(request: Request) -> Any:
        """Inject a `canceled` / `expired` result for one custom_id.

        Body: {"custom_id": "...", "type": "canceled" | "expired", "batch_id"?: "..."}.
        (`succeeded` / `errored` are injected through the ordinary /_control/respond and
        /_control/error — also addressable by custom_id — so they are rejected here.)
        If this was the last unresolved entry the batch transitions to ended.
        """
        body, errmsg = await fs._parse_json_body(request)
        if errmsg is not None:
            return fs._plain_400(errmsg)
        rtype = body.get("type")
        if rtype not in ("expired", "canceled"):
            return fs._plain_400(
                'type must be "expired" or "canceled" '
                "(inject succeeded/errored via /_control/respond and /_control/error)")
        cid = body.get("custom_id")
        if not isinstance(cid, str) or not cid:
            return fs._plain_400("custom_id must be a non-empty string")
        now = time.time()
        async with fs.state.lock:
            batch_id = body.get("batch_id")
            if batch_id is not None:
                batch, err = _target_batch(batch_id)
                if err is not None:
                    return err
                if cid not in batch["entries"]:
                    return fs._plain_400(f"no custom_id={cid} in batch {batch['id']}")
            else:
                # Exclude creating batches, same as _target_batch: stamping a lifecycle
                # result mid-registration could end the batch over partial entries.
                matches = [b for b in fs.state.batches.values()
                           if not b.get("creating")
                           and cid in b["entries"] and b["entries"][cid]["result"] is None]
                if not matches:
                    return fs._plain_400(f"no unresolved custom_id={cid} in any batch")
                if len(matches) > 1:
                    return JSONResponse(
                        {"error": f"custom_id={cid} is unresolved in multiple batches; "
                                  "specify batch_id",
                         "batch_ids": [b["id"] for b in matches]},
                        status_code=400,
                    )
                batch = matches[0]
            n = _finalize_unresolved(batch, rtype, now, custom_ids={cid})
            if n == 0:
                return fs._plain_400(
                    f"custom_id={cid} already resolved in batch {batch['id']} "
                    "(or its injection is in flight)")
            status = batch["processing_status"]
        return {"ok": True, "batch_id": batch["id"], "custom_id": cid,
                "type": rtype, "processing_status": status}

    return router
