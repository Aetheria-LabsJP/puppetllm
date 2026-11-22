"""Read-only SDK readiness/watchdog and explicitly locked guard maintenance.

``status`` uses only the standard library. The watch sidecar reads status.json;
it never opens the guard database, starts an SDK client, or writes worker state.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any
from urllib.parse import urlsplit

from .sdk_runtime import API_STOP_REASONS, SDK_TERMINAL_REASONS, STREAM_DELTA_TYPES, STREAM_EVENT_TYPES, _number


_MAX_STATUS_BYTES = 1024 * 1024
_REASONS = frozenset({"running", "stopping", "state_error"})
_STAGES = frozenset({
    "checking_guard", "starting_sdk", "connected", "waiting_sdk_event", "receiving_sdk",
    "generating", "thinking", "sdk_retry", "processing_sdk_message", "validating", "injecting",
    "completed", "timeout", "blocked", "failed", "cancelled", "delivery_error",
    "error", "upstream_error", "invalid_output",
})
_USAGE_FIELDS = frozenset({
    "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
    "inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "costUSD",
})
_NUMBER_FIELDS = frozenset({
    "started_at", "last_sdk_event_at", "last_generation_at", "first_sdk_event_at", "finished_at",
    "event_count", "message_start_events", "generation_events", "generated_characters",
    "api_retry_events", "structured_output_calls_observed", "api_attempt_count",
    "structured_output_retries", "sdk_reported_num_turns", "sdk_estimated_cost_usd", "stderr_lines",
    "sdk_api_error_status",
    "thinking_token_updates", "estimated_thinking_tokens", "last_thinking_progress_at",
    "configured_max_output_tokens",
    "api_requests_blocked",
})
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_KEY = re.compile(r"[0-9a-f]{64}\Z")
_MODEL = re.compile(r"[\w.:/-]{1,100}\Z")
_UNAVAILABLE_MESSAGE = "Agent SDK relay is unavailable; automatic retry is disabled."


def _positive(value: str | float) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("must be a positive finite number") from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _numeric_mapping(value: Any, fields: frozenset[str]) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise ValueError("invalid numeric mapping")
    result = {}
    for key, item in value.items():
        if key in fields:
            if not _number(item):
                raise ValueError("invalid numeric metadata")
            result[key] = item
    return result


def _progress(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("invalid request progress")
    result = {}
    for key, pattern in (("pending_id", _ID), ("key", _KEY), ("model", _MODEL)):
        item = value.get(key)
        if not isinstance(item, str) or pattern.fullmatch(item) is None:
            raise ValueError("invalid request identifier")
        result[key] = item
    if value.get("stage") not in _STAGES or not _number(value.get("started_at")):
        raise ValueError("invalid progress stage")
    result["stage"] = value["stage"]
    for key in _NUMBER_FIELDS:
        if key in value:
            item = value[key]
            if item is not None and not _number(item):
                raise ValueError("invalid request counter")
            result[key] = item
    if "last_generation_kind" in value:
        if value["last_generation_kind"] not in {"text_delta", "thinking_delta", "input_json_delta"}:
            raise ValueError("invalid generation kind")
        result["last_generation_kind"] = value["last_generation_kind"]
    if "configured_effort" in value:
        item = value["configured_effort"]
        if item not in {"sdk_default", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("invalid configured effort")
        result["configured_effort"] = item
    for key in ("observed_stop_reason", "upstream_stop_reason"):
        if key in value:
            item = value[key]
            if item not in API_STOP_REASONS:
                raise ValueError("invalid API stop reason")
            result[key] = item
    if "last_stream_event_type" in value:
        item = value["last_stream_event_type"]
        if item not in STREAM_EVENT_TYPES:
            raise ValueError("invalid stream event type")
        result["last_stream_event_type"] = item
    for key, fields in (("stream_event_counts", STREAM_EVENT_TYPES), ("stream_delta_counts", STREAM_DELTA_TYPES)):
        if key in value:
            result[key] = _numeric_mapping(value[key], fields)
    if "last_sdk_event_kind" in value:
        item = value["last_sdk_event_kind"]
        if item not in {"StreamEvent", "SystemMessage", "AssistantMessage", "ResultMessage",
                        "UserMessage", "UIInvalidateMessage", "RateLimitEvent", "other"}:
            raise ValueError("invalid SDK event kind")
        result["last_sdk_event_kind"] = item
    if "sdk_result_subtype" in value:
        item = value["sdk_result_subtype"]
        if item not in {"success", "error_max_turns", "error_max_structured_output_retries",
                        "error_during_execution", "error_max_budget_usd", "other"}:
            raise ValueError("invalid SDK result subtype")
        result["sdk_result_subtype"] = item
    if "sdk_terminal_reason" in value:
        item = value["sdk_terminal_reason"]
        if item is not None and item not in SDK_TERMINAL_REASONS:
            raise ValueError("invalid SDK terminal reason")
        result["sdk_terminal_reason"] = item
    for key in ("sdk_reported_usage", "observed_message_usage", "upstream_reported_usage"):
        if key in value:
            item = value[key]
            result[key] = None if item is None else _numeric_mapping(item, _USAGE_FIELDS)
    if "sdk_reported_model_usage" in value:
        items = value["sdk_reported_model_usage"]
        if items is not None and (not isinstance(items, list) or len(items) > 1000):
            raise ValueError("invalid model usage")
        result["sdk_reported_model_usage"] = (
            None if items is None else [_numeric_mapping(item, _USAGE_FIELDS) for item in items])
    if "last_api_retry" in value:
        result["last_api_retry"] = _numeric_mapping(value["last_api_retry"], frozenset({
            "attempt", "max_retries", "retry_delay_ms", "error_status"}))
    return result


def _unready(reason: str) -> dict[str, Any]:
    return {"version": 1, "fresh": False, "ready": False, "worker_ready": False,
            "reason": reason, "heartbeat_at": None, "heartbeat_age_seconds": None,
            "requests": [], "guard": {"blocked": None, "in_progress": None}}


def read_status(directory: str | Path, *, stale_after: float = 30,
                now: float | None = None) -> dict[str, Any]:
    """Read a bounded, regular JSON file and expose only known safe metadata.

    A missing, malformed, oversized or unsupported snapshot is unready. No raw
    file contents, exception text or paths are included in the returned error.
    """
    stale_after = _positive(stale_after)
    now = time.time() if now is None else now
    try:
        descriptor = os.open(Path(directory).expanduser().resolve() / "status.json", os.O_RDONLY
                             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_STATUS_BYTES:
                return _unready("status_invalid")
            encoded = stream.read(_MAX_STATUS_BYTES + 1)
        if len(encoded) > _MAX_STATUS_BYTES:
            return _unready("status_invalid")
        value = json.loads(encoded, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
            raise ValueError("unsupported snapshot")
        heartbeat = value.get("heartbeat_at")
        if not _number(heartbeat) or type(value.get("ready")) is not bool or value.get("reason") not in _REASONS:
            raise ValueError("invalid readiness")
        requests = value.get("requests")
        guard = value.get("guard")
        if not isinstance(requests, list) or len(requests) > 10000 or not isinstance(guard, dict):
            raise ValueError("invalid snapshot structure")
        counts = {}
        for key in ("blocked", "in_progress"):
            if type(guard.get(key)) is not int or guard[key] < 0:
                raise ValueError("invalid guard counter")
            counts[key] = guard[key]
        progress = [_progress(item) for item in requests]
        age = now - heartbeat
        fresh = 0 <= age <= stale_after
        result = {"version": 1, "fresh": fresh, "ready": fresh and value["ready"],
                  "worker_ready": value["ready"],
                  "reason": value["reason"] if fresh else "heartbeat_stale",
                  "heartbeat_at": heartbeat, "heartbeat_age_seconds": round(age, 3),
                  "requests": progress, "guard": counts}
        if "worker_id" in value:
            if not isinstance(value["worker_id"], str) or re.fullmatch(r"[0-9a-f]{32}", value["worker_id"]) is None:
                raise ValueError("invalid worker identifier")
            result["worker_id"] = value["worker_id"]
        return result
    except FileNotFoundError:
        return _unready("status_missing")
    except OSError:
        return _unready("status_unavailable")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        return _unready("status_invalid")


async def watch_once(directory: str | Path, client: Any, *, stale_after: float = 30) -> dict[str, Any]:
    """Reject waiting requests only while the worker cannot accept SDK work."""
    snapshot = read_status(directory, stale_after=stale_after)
    result = {"ready": snapshot["ready"], "reason": snapshot["reason"],
              "pending": 0, "injected": 0, "failed": 0}
    if snapshot["ready"]:
        return result
    try:
        response = await client.get("/_control/pending", params={"fields": "pending_id"})
        response.raise_for_status()
        entries = response.json().get("pending")
        if not isinstance(entries, list):
            raise ValueError("invalid pending summary")
    except Exception:
        result["reason"] = "proxy_unavailable"
        return result

    seen = set()
    for entry in entries:
        # Never inspect, retain or display entry['request'] (which may include
        # a short text preview even in the proxy's summary response).
        pid = entry.get("pending_id") if isinstance(entry, dict) else None
        if not isinstance(pid, str) or _ID.fullmatch(pid) is None:
            result["failed"] += 1
            continue
        if pid in seen:
            continue
        seen.add(pid)
        result["pending"] += 1
        # The worker may have recovered since the summary request completed.
        # Stop immediately rather than draining the recovered worker's queue.
        if read_status(directory, stale_after=stale_after)["ready"]:
            break
        try:
            response = await client.post("/_control/error", json={
                "pending_id": pid, "status": 503, "type": "api_error",
                "message": _UNAVAILABLE_MESSAGE, "headers": {"x-should-retry": "false"}})
            response.raise_for_status()
            result["injected"] += 1
        except Exception:
            # A resolved/disconnected pending must not prevent rejection of
            # other waiting requests. Never log an HTTP body or exception.
            result["failed"] += 1
    return result


def create_app(directory: str | Path, proxy_url: str, *, stale_after: float = 30,
               interval: float = 1, transport: Any = None):
    """Build the sidecar API; optional HTTP transport enables isolated tests."""
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    stale_after, interval = _positive(stale_after), _positive(interval)
    parsed = urlsplit(proxy_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("watch requires a proxy HTTP URL without credentials")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(base_url=proxy_url.rstrip("/"), timeout=5,
                                     trust_env=False, transport=transport) as client:
            async def watch_loop():
                while True:
                    await watch_once(directory, client, stale_after=stale_after)
                    await asyncio.sleep(interval)

            task = asyncio.create_task(watch_loop())
            app.state.watchdog_task = task
            try:
                yield
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="Agent SDK readiness", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/status")
    async def status():
        return read_status(directory, stale_after=stale_after)

    @app.get("/ready")
    async def ready():
        snapshot = read_status(directory, stale_after=stale_after)
        return JSONResponse(snapshot, status_code=200 if snapshot["ready"] else 503)

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m puppetllm.sdk_state",
                                     description="SDK worker readiness and explicit guard maintenance")

    def common(target):
        target.add_argument("--state-dir", default=argparse.SUPPRESS)
        target.add_argument("--stale-after", type=_positive, default=argparse.SUPPRESS)

    common(parser)
    parser.set_defaults(state_dir=os.environ.get("PUPPETLLM_SDK_STATE_DIR") or str(Path.home() / ".local/state/puppetllm/agent-sdk"),
                        stale_after=30)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "blocks", "clear", "recover", "watch"):
        command = commands.add_parser(name)
        common(command)
        if name == "clear":
            selection = command.add_mutually_exclusive_group(required=True)
            selection.add_argument("--key")
            selection.add_argument("--all", action="store_true")
        elif name == "recover":
            command.add_argument("--all", action="store_true", required=True)
        elif name == "watch":
            command.add_argument("--proxy", default="http://127.0.0.1:8765")
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8766)
            command.add_argument("--interval", type=_positive, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.state_dir = str(Path(args.state_dir).expanduser().resolve())
    if args.command == "status":
        print(json.dumps(read_status(args.state_dir, stale_after=args.stale_after), allow_nan=False))
        return 0
    if args.command == "watch":
        import uvicorn
        try:
            app = create_app(args.state_dir, args.proxy, stale_after=args.stale_after, interval=args.interval)
            if not 1 <= args.port <= 65535:
                raise ValueError("invalid port")
        except ValueError:
            print('{"error": "invalid_watch_configuration"}', file=sys.stderr)
            return 1
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
        return 0

    from .sdk_guard import GuardError, SDKGuard
    from .sdk_runtime import StateBusyError, StateLock
    try:
        with StateLock(args.state_dir):
            path = Path(args.state_dir) / "guard.sqlite3"
            if not path.is_file():
                raise GuardError("guard unavailable")
            with SDKGuard(path) as guard:
                if args.command == "blocks":
                    entries = guard.status()
                    result = {"entries": entries,
                              "blocked": sum(item["state"] == "blocked" for item in entries),
                              "in_progress": sum(item["state"] == "active" for item in entries)}
                elif args.command == "clear":
                    result = {"cleared": guard.clear(args.key)}
                else:
                    result = {"recovered": guard.recover_active()}
    except StateBusyError:
        print('{"error": "state_busy", "message": "Stop SDK workers before maintenance."}', file=sys.stderr)
        return 1
    except (GuardError, OSError):
        print('{"error": "guard_unavailable"}', file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
