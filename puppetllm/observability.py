"""Structured logging and Prometheus metrics.

Log format is chosen once per process (`PUPPETLLM_LOG_FORMAT` or `serve --log-format`):

  text  the receipt lines the adapters always printed (`[bedrock] invoke model=...`) and
        nothing else — the interactive default, quiet enough to watch by eye;
  json  one JSON object per line on stderr for every lifecycle event, correlated by
        `request_id` / `pending_id`: received, matched, resolved, rejected, canceled, plus
        the adapters' receipt lines as `log` events.

Metrics are process-wide cumulative counters (they survive `/_control/clear`, as
Prometheus counters should) plus a few live gauges, rendered by `GET /_control/metrics`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from typing import Any

LOG_FORMATS = ("text", "json")
_format = os.environ.get("PUPPETLLM_LOG_FORMAT", "text").strip().lower() or "text"
if _format not in LOG_FORMATS:
    print(f"[puppetllm] PUPPETLLM_LOG_FORMAT must be one of {list(LOG_FORMATS)}; using text",
          file=sys.stderr)
    _format = "text"


def log_format() -> str:
    return _format


def set_log_format(fmt: str) -> None:
    global _format
    if fmt not in LOG_FORMATS:
        raise ValueError(f"log format must be one of {list(LOG_FORMATS)}")
    _format = fmt


def log_event(event: str, **fields: Any) -> None:
    """A lifecycle event. Silent in text mode (the adapters' receipt lines cover the
    interactive case); one JSON line per event in json mode."""
    if _format != "json":
        return
    record = {"ts": round(time.time(), 3), "event": event}
    record.update({k: v for k, v in fields.items() if v is not None})
    print(json.dumps(record, ensure_ascii=False, default=str), file=sys.stderr, flush=True)


def log_text(source: str, message: str) -> None:
    """An adapter's receipt line: `[source] message` in text mode, a `log` event in json."""
    if _format == "json":
        log_event("log", source=source, message=message)
    else:
        print(f"[{source}] {message}", file=sys.stderr, flush=True)


class Metrics:
    """Cumulative counters keyed by (name, labels) and a registry of gauge callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)
        self._help: dict[str, str] = {}
        self._gauges: dict[str, tuple[str, Any]] = {}

    def inc(self, name: str, value: float = 1.0, **labels: Any) -> None:
        key = tuple(sorted((k, str(v)) for k, v in labels.items()))
        with self._lock:
            series = self._counters[name]
            series[key] = series.get(key, 0.0) + value

    def describe(self, name: str, help_text: str) -> None:
        self._help[name] = help_text

    def gauge(self, name: str, help_text: str, read: Any) -> None:
        """Register a gauge computed at scrape time (`read()` → number or {labels: value})."""
        self._gauges[name] = (help_text, read)

    def render(self) -> str:
        """Prometheus text exposition (version 0.0.4)."""
        lines: list[str] = []
        with self._lock:
            names = sorted(set(self._counters) | set(n for n in self._help if n not in self._gauges))
            for name in names:
                lines.append(f"# HELP {name} {self._help.get(name, name)}")
                lines.append(f"# TYPE {name} counter")
                series = self._counters.get(name, {})
                if not series:
                    lines.append(f"{name} 0")
                for key, value in sorted(series.items()):
                    lines.append(f"{name}{_labels(dict(key))} {_num(value)}")
        for name, (help_text, read) in sorted(self._gauges.items()):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            value = read()
            if isinstance(value, dict):
                for labels, v in value.items():
                    lines.append(f"{name}{_labels(dict(labels))} {_num(v)}")
            else:
                lines.append(f"{name} {_num(value)}")
        return "\n".join(lines) + "\n"


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    items = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
    return "{" + items + "}"


def _escape(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _num(v: Any) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


metrics = Metrics()
metrics.describe("puppetllm_requests_total", "Requests recorded, by provider and outcome (ok / error)")
metrics.describe("puppetllm_rejected_total",
                 "Requests refused with 400 before becoming a pending (malformed body, shape, cache_control), by provider")
metrics.describe("puppetllm_canceled_total",
                 "Pendings that ended without an answer, by reason (clear / disconnect / cancelled / batch / error)")
metrics.describe("puppetllm_harness_answers_total", "Answers produced by the harness, by source")
metrics.describe("puppetllm_tokens_total", "Estimated tokens billed, by provider and kind")
metrics.describe("puppetllm_cost_usd_total", "Estimated cost in USD, by provider")
metrics.describe("puppetllm_cache_lookups_total", "Pseudo prompt-cache decisions, by status")
metrics.describe("puppetllm_latency_seconds_total", "Seconds between arrival and answer, by provider")
