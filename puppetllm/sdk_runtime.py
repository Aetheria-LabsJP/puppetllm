"""Content-free Agent SDK progress snapshots and worker/maintenance locking."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time
import uuid
from typing import Any, Callable

STREAM_EVENT_TYPES = frozenset({"message_start", "message_delta", "message_stop",
    "content_block_start", "content_block_delta", "content_block_stop", "ping", "error", "other"})
STREAM_DELTA_TYPES = frozenset({"text_delta", "thinking_delta", "input_json_delta",
    "signature_delta", "citations_delta", "other"})
SDK_TERMINAL_REASONS = frozenset({"api_error", "max_turns", "aborted_streaming", "aborted_tools",
    "max_budget_usd", "max_structured_output_retries", "success", "completed",
    "budget_exhausted", "structured_output_retry_exhausted", "prompt_too_long", "blocking_limit"})
API_STOP_REASONS = frozenset({"end_turn", "tool_use", "max_tokens", "refusal", "stop_sequence",
    "pause_turn", "model_context_window_exceeded", "other"})


class StateBusyError(RuntimeError):
    pass


class StateLock:
    """One holder per state directory: a worker, or a maintenance command."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.file = (directory / "worker.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.file.seek(0)
                if not self.file.read(1):
                    self.file.write(b"\0")
                    self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise StateBusyError("Agent SDK state is in use; stop workers before maintenance") from None

    def close(self) -> None:
        if not self.file.closed:
            if os.name == "nt":
                import msvcrt
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _number(value: Any) -> bool:
    return type(value) is int and value >= 0 or type(value) is float and math.isfinite(value) and value >= 0


def _usage(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    keys = {"input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
            "inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "costUSD"}
    return {key: item for key, item in value.items() if key in keys and _number(item)}


class RuntimeSnapshot:
    def __init__(self, directory: str | Path, *, log: Callable[[str], None] = print,
                 idle_warning: float = 120, interval: float = 10) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log = log
        self.idle_warning = idle_warning
        self.interval = interval
        self.requests: dict[str, dict[str, Any]] = {}
        self._logged: dict[str, float] = {}
        self._thinking_sources: dict[str, dict[str, int | float]] = {}
        self.ready = True
        self.reason = "running"
        self.worker_id = uuid.uuid4().hex

    def start(self, pending_id: str, key: str, model: str, *, generation_settings: dict | None = None) -> None:
        if len(self.requests) >= 100:
            completed = [pid for pid, record in self.requests.items() if record.get("finished_at")]
            if completed:
                oldest = min(completed, key=lambda pid: self.requests[pid]["finished_at"])
                del self.requests[oldest]
                self._logged.pop(oldest, None)
                self._thinking_sources.pop(oldest, None)
        safe_model = model if re.fullmatch(r"[\w.:/-]{1,100}", model) else "custom"
        self.requests[pending_id] = {
            "pending_id": pending_id, "key": key, "model": safe_model,
            "stage": "checking_guard", "started_at": time.time(), "last_sdk_event_at": None,
            "last_generation_at": None, "first_sdk_event_at": None, "event_count": 0,
            "message_start_events": 0, "generation_events": 0, "generated_characters": 0,
            "api_retry_events": 0, "structured_output_calls_observed": 0,
            "api_requests_blocked": 0,
            "api_attempt_count": None, "structured_output_retries": None,
            "sdk_reported_num_turns": None, "sdk_estimated_cost_usd": None,
            "sdk_reported_usage": None, "sdk_reported_model_usage": None,
            "stderr_lines": 0,
            "stream_event_counts": {}, "stream_delta_counts": {},
            "thinking_token_updates": 0, "estimated_thinking_tokens": None,
            "last_thinking_progress_at": None,
        }
        settings = generation_settings or {}
        effort = settings.get("effort", "sdk_default")
        self.requests[pending_id]["configured_effort"] = effort if effort in (
            "sdk_default", "low", "medium", "high", "xhigh", "max") else "sdk_default"
        cap = settings.get("max_output_tokens")
        self.requests[pending_id]["configured_max_output_tokens"] = cap if type(cap) is int and cap > 0 else None
        self._thinking_sources[pending_id] = {"stream": 0, "system": 0}
        self.emit(pending_id, force=True)

    def for_request(self, pid: str):
        runtime = self
        class Progress:
            def stage(self, stage):
                runtime.stage(pid, stage)
            def message(self, message):
                runtime.message(pid, message)
            def stderr(self, line):
                runtime.stderr(pid, line)
            def api_request_started(self):
                record = runtime.requests[pid]
                record["api_attempt_count"] = (record["api_attempt_count"] or 0) + 1
                runtime.emit(pid)
            def api_request_blocked(self):
                runtime.requests[pid]["api_requests_blocked"] += 1
                runtime.emit(pid)
            def api_response_metadata(self, stop_reason, usage):
                record = runtime.requests[pid]
                if stop_reason is not None:
                    record["upstream_stop_reason"] = stop_reason if stop_reason in API_STOP_REASONS else "other"
                record["upstream_reported_usage"] = _usage(usage)
                runtime.emit(pid)
        return Progress()

    def stage(self, pid: str, stage: str) -> None:
        self.requests[pid]["stage"] = stage
        self.emit(pid, force=True)

    def stderr(self, pid: str, _line: str) -> None:
        # The SDK can emit private data to stderr. Count lines, never retain it.
        self.requests[pid]["stderr_lines"] += 1

    def _thinking_progress(self, pid: str, value: Any, source: str, now: float, *, increment: bool = False) -> None:
        if not _number(value) or value <= 0:
            return
        sources = self._thinking_sources[pid]
        # The stream and system messages describe the same increments. Keep
        # separate totals, including across thinking blocks, then take the max.
        sources[source] = sources[source] + value if increment else max(sources[source], value)
        total = max(sources.values())
        record = self.requests[pid]
        if total > (record["estimated_thinking_tokens"] or 0):
            record["estimated_thinking_tokens"] = total
            record["thinking_token_updates"] += 1
            record["last_thinking_progress_at"] = now
            record["stage"] = "thinking"

    def message(self, pid: str, message: Any) -> None:
        record = self.requests[pid]
        now = time.time()
        record["first_sdk_event_at"] = record["first_sdk_event_at"] or now
        record["last_sdk_event_at"] = now
        record["event_count"] += 1
        kind = type(message).__name__
        record["last_sdk_event_kind"] = kind if kind in (
            "StreamEvent", "SystemMessage", "AssistantMessage", "ResultMessage",
            "UserMessage", "UIInvalidateMessage", "RateLimitEvent") else "other"
        if kind == "StreamEvent":
            event = message.event
            event_type = event.get("type")
            safe_type = event_type if event_type in STREAM_EVENT_TYPES else "other"
            record["last_stream_event_type"] = safe_type
            counts = record["stream_event_counts"]
            counts[safe_type] = counts.get(safe_type, 0) + 1
            if event_type == "message_start":
                record["message_start_events"] += 1
                record["stage"] = "receiving_sdk"
                record["observed_message_usage"] = _usage((event.get("message") or {}).get("usage"))
            elif event_type == "content_block_delta":
                delta = event.get("delta") or {}
                delta_type = delta.get("type")
                safe_delta = delta_type if delta_type in STREAM_DELTA_TYPES else "other"
                counts = record["stream_delta_counts"]
                counts[safe_delta] = counts.get(safe_delta, 0) + 1
                if delta_type == "thinking_delta":
                    self._thinking_progress(pid, delta.get("estimated_tokens"), "stream", now, increment=True)
                field = {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "partial_json"}.get(delta_type)
                value = delta.get(field) if field else None
                if isinstance(value, str) and value:
                    record["last_generation_at"] = now
                    record["generation_events"] += 1
                    record["stage"] = "generating"
                    record["last_generation_kind"] = delta_type
                    record["generated_characters"] += len(value)
            elif event_type == "message_delta":
                record.setdefault("observed_message_usage", {}).update(_usage(event.get("usage")))
                reason = (event.get("delta") or {}).get("stop_reason")
                if reason is not None:
                    record["observed_stop_reason"] = reason if reason in API_STOP_REASONS else "other"
        elif kind == "SystemMessage" and getattr(message, "subtype", None) == "thinking_tokens":
            data = getattr(message, "data", {})
            delta = data.get("estimated_tokens_delta")
            if delta is not None:
                self._thinking_progress(pid, delta, "system", now, increment=True)
            else:
                self._thinking_progress(pid, data.get("estimated_tokens"), "system", now)
        elif kind == "SystemMessage" and getattr(message, "subtype", None) == "api_retry":
            record["api_retry_events"] += 1
            record["stage"] = "sdk_retry"
            data = getattr(message, "data", {})
            record["last_api_retry"] = {key: value for key, value in data.items()
                                        if key in ("attempt", "max_retries", "retry_delay_ms", "error_status")
                                        and _number(value)}
        elif kind == "AssistantMessage":
            record["structured_output_calls_observed"] += sum(
                type(block).__name__ == "ToolUseBlock" and getattr(block, "name", None) == "StructuredOutput"
                for block in message.content)
            record["stage"] = "processing_sdk_message"
        elif kind == "ResultMessage":
            record["stage"] = "validating"
            subtype = getattr(message, "subtype", None)
            record["sdk_result_subtype"] = subtype if subtype in (
                "success", "error_max_turns", "error_max_structured_output_retries",
                "error_during_execution", "error_max_budget_usd") else "other"
            reason = getattr(message, "terminal_reason", None)
            record["sdk_terminal_reason"] = reason if reason in SDK_TERMINAL_REASONS else None
            status = getattr(message, "api_error_status", None)
            record["sdk_api_error_status"] = status if type(status) is int and 400 <= status <= 599 else None
            turns = getattr(message, "num_turns", None)
            if _number(turns):
                record["sdk_reported_num_turns"] = turns
            cost = getattr(message, "total_cost_usd", None)
            if _number(cost):
                record["sdk_estimated_cost_usd"] = cost
            record["sdk_reported_usage"] = _usage(getattr(message, "usage", None)) or None
            models = getattr(message, "model_usage", None)
            if isinstance(models, dict):
                # Use a list without model names; only numeric usage is retained.
                record["sdk_reported_model_usage"] = [_usage(value) for value in models.values()
                                                       if isinstance(value, dict)]
        self.emit(pid)

    def finish(self, pid: str, reason: str) -> None:
        record = self.requests.get(pid)
        if record is not None and not record.get("finished_at"):
            record.update(stage=reason, finished_at=time.time())
            self.emit(pid, force=True)

    def emit(self, pid: str, *, force: bool = False) -> None:
        now = time.time()
        if force or now - self._logged.get(pid, 0) >= self.interval:
            record = self.requests[pid]
            payload = {**record, "elapsed_seconds": round(now - record["started_at"], 1)}
            if not record.get("finished_at"):
                last = max(record["last_generation_at"] or 0, record["last_thinking_progress_at"] or 0,
                           record["started_at"])
                if now - last >= self.idle_warning:
                    payload["warning"] = "no_generation_event_observed"
            self.log("sdk_progress " + json.dumps(payload, ensure_ascii=True))
            self._logged[pid] = now

    def write(self, guard_status: list[dict[str, Any]]) -> None:
        now = time.time()
        for pid, record in self.requests.items():
            if not record.get("finished_at"):
                self.emit(pid)
        payload = {"version": 1, "worker_id": self.worker_id, "heartbeat_at": now,
                   "ready": self.ready, "reason": self.reason, "requests": list(self.requests.values()),
                   "guard": {"blocked": sum(row["state"] == "blocked" for row in guard_status),
                             "in_progress": sum(row["state"] == "active" for row in guard_status)}}
        temporary = self.directory / (".status-" + uuid.uuid4().hex)
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / "status.json")
        finally:
            temporary.unlink(missing_ok=True)
