"""Tests for the observability features: history filters / paging / JSONL and the
history file, pending summaries, request correlation fields, cache-miss explanations and
`diagnostics`, structured JSON logs, Prometheus metrics, injection provenance, custom
pricing and the relay's provenance / strict-translation / dry-run flags.

Run:
  python3 -m unittest puppetllm.tests.test_observability -v
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from typing import Any

os.environ["PUPPETLLM_CACHE_MIN_TOKENS"] = "0"


def _import_fresh():
    import importlib
    from puppetllm import fake_server as fs
    importlib.reload(fs)
    return fs


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    out, name = [], None
    for line in text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: "):]
        elif line.startswith("data: ") and name is not None:
            out.append((name, json.loads(line[len("data: "):])))
            name = None
    return out


_MSG = {"model": "claude-sonnet-5", "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}
_BEDROCK = "/model/anthropic.claude-sonnet-4-5-20250929-v1:0"
_OPENAI = {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]}
_SYSTEM = [{"type": "text", "text": "long system prompt " * 40, "cache_control": {"type": "ephemeral"}}]


_GUARDED_ENV = ("PUPPETLLM_HISTORY_FILE", "PUPPETLLM_PRICING_FILE", "PUPPETLLM_CACHE_MIN_TOKENS",
                "PUPPETLLM_LOG_FORMAT", "RELAY_TEST_KEY")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        # Whatever a test sets (environment, log format, custom prices) is restored, so a
        # failure half-way cannot leak into the next test or module.
        from puppetllm import observability as obs
        from puppetllm import pricing
        saved = {k: os.environ.get(k) for k in _GUARDED_ENV}
        fmt, prices = obs.log_format(), list(pricing._CUSTOM_PRICES)

        def restore() -> None:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            obs.set_log_format(fmt)
            _import_fresh()
            pricing._CUSTOM_PRICES[:] = prices
        self.addCleanup(restore)
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

    async def _ctl(self, c, method: str, path: str, **kw: Any) -> Any:
        r = await getattr(c, method)(path, **kw)
        self.assertEqual(r.status_code, 200, f"{method} {path}: {r.text}")
        return r.json()

    async def _pending(self, c, path: str, body: dict, **kw: Any) -> Any:
        t = asyncio.create_task(c.post(path, json=body, timeout=10, **kw))
        for _ in range(50):
            if (await c.get("/_control/pending")).json().get("has_pending"):
                return t
            await asyncio.sleep(0.05)
        self.fail("never became pending")

    async def _answered(self, c, path: str, body: dict, **kw: Any) -> Any:
        t = asyncio.create_task(c.post(path, json=body, timeout=10, **kw))
        for _ in range(200):
            if t.done():
                return t.result()
            if (await c.get("/_control/pending")).json().get("has_pending"):
                await c.post("/_control/clear")
                await asyncio.gather(t, return_exceptions=True)
                self.fail("request became pending instead of being answered by a rule")
            await asyncio.sleep(0.01)
        t.cancel()
        self.fail("request neither answered nor pending")

    async def _answered_raw(self, c, path: str, raw: bytes) -> Any:
        t = asyncio.create_task(c.post(path, content=raw, headers={"content-type": "application/json"}, timeout=10))
        for _ in range(200):
            if t.done():
                return t.result()
            await asyncio.sleep(0.01)
        t.cancel()
        self.fail("request not answered")

    async def _seed(self, c, n: int = 5) -> None:
        """n answered requests alternating providers and models."""
        await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
        for i in range(n):
            if i % 2:
                r = await self._answered(c, "/v1/chat/completions", {**_OPENAI, "model": "gpt-5.4" if i % 4 == 1 else "gpt-5.5"})
            else:
                r = await self._answered(c, "/v1/messages", {**_MSG, "model": "claude-sonnet-5" if i % 4 == 0 else "claude-opus-5"})
            self.assertEqual(r.status_code, 200, r.text)


class TestHistoryAndPending(_Base):
    def test_filters_paging_and_jsonl(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._seed(c, 5)
                full = await self._ctl(c, "get", "/_control/history")
                self.assertEqual((full["count"], full["has_more"], full["next_cursor"]), (5, False, 5))
                # nothing new: the cursor passed in comes back, so a tailer just re-sends it
                tail = await self._ctl(c, "get", "/_control/history", params={"cursor": 5})
                self.assertEqual((tail["count"], tail["next_cursor"]), (0, 5))
                self.assertEqual([e["turn"] for e in full["history"]], [1, 2, 3, 4, 5])
                page = await self._ctl(c, "get", "/_control/history", params={"limit": 2})
                self.assertEqual(([e["turn"] for e in page["history"]], page["has_more"], page["next_cursor"]), ([1, 2], True, 2))
                page = await self._ctl(c, "get", "/_control/history", params={"limit": 2, "cursor": page["next_cursor"]})
                self.assertEqual(([e["turn"] for e in page["history"]], page["has_more"]), ([3, 4], True))
                page = await self._ctl(c, "get", "/_control/history", params={"limit": 2, "cursor": 4})
                self.assertEqual(([e["turn"] for e in page["history"]], page["has_more"]), ([5], False))
                self.assertEqual([e["turn"] for e in (await self._ctl(c, "get", "/_control/history", params={"since_turn": 3}))["history"]], [4, 5])
                self.assertEqual([e["turn"] for e in (await self._ctl(c, "get", "/_control/history", params={"provider": "openai"}))["history"]], [2, 4])
                self.assertEqual([e["turn"] for e in (await self._ctl(c, "get", "/_control/history", params={"model": "claude-opus-*"}))["history"]], [3])
                r = await c.get("/_control/history", params={"format": "jsonl", "limit": 3})
                self.assertEqual(r.status_code, 200)
                self.assertTrue(r.headers["content-type"].startswith("application/x-ndjson"))
                lines = [json.loads(line) for line in r.text.splitlines()]
                self.assertEqual([e["turn"] for e in lines], [1, 2, 3])
                self.assertEqual((r.headers["x-puppetllm-has-more"], r.headers["x-puppetllm-next-cursor"]), ("1", "3"))
                self.assertEqual((await self._ctl(c, "get", "/_control/history/4"))["provider"], "openai")
                self.assertEqual([e["seq"] for e in full["history"]], [1, 2, 3, 4, 5])
                self.assertEqual((await c.get("/_control/history/99")).status_code, 404)
                self.assertEqual((await c.get("/_control/history/abc")).status_code, 400)
                self.assertEqual((await c.get("/_control/history/²")).status_code, 400)
                for params in ({"limit": 0}, {"limit": 1001}, {"limit": "x"}, {"cursor": "y"}, {"provider": "azure"}):
                    self.assertEqual((await c.get("/_control/history", params=params)).status_code, 400, params)
        _run(go())

    def test_cursor_follows_completion_order(self) -> None:
        """Entries complete out of turn order (parallel pendings); paging by `cursor` and
        tailing by `since_turn` must still deliver every one of them."""
        async def go():
            async with await self._client() as c:
                tasks = []
                for i in range(4):
                    tasks.append(asyncio.create_task(c.post("/v1/messages", json=_MSG, timeout=10)))
                    for _ in range(50):
                        if len((await c.get("/_control/pending")).json()["pending"]) == i + 1:
                            break
                        await asyncio.sleep(0.02)
                pend = (await c.get("/_control/pending")).json()["pending"]
                by_turn = {p["request"]["turn"]: p["pending_id"] for p in pend}
                for turn in (2, 4, 1, 3):
                    await self._ctl(c, "post", "/_control/respond", json={"pending_id": by_turn[turn], "text": "ok"})
                await asyncio.gather(*tasks)
                page = await self._ctl(c, "get", "/_control/history", params={"limit": 2})
                self.assertEqual(([e["turn"] for e in page["history"]], page["has_more"], page["next_cursor"]), ([2, 4], True, 2))
                page = await self._ctl(c, "get", "/_control/history", params={"limit": 2, "cursor": page["next_cursor"]})
                self.assertEqual(([e["turn"] for e in page["history"]], page["has_more"], page["next_cursor"]), ([1, 3], False, 4))
                # a poller that saw seq 2 gets turns 1 and 3 next; `since_turn` filters by turn
                tail = await self._ctl(c, "get", "/_control/history", params={"cursor": 2})
                self.assertEqual([e["turn"] for e in tail["history"]], [1, 3])
                self.assertEqual([e["turn"] for e in (await self._ctl(c, "get", "/_control/history", params={"since_turn": 2}))["history"]], [4, 3])
                # the sequence survives a clear (turns restart, the cursor does not go backwards)
                await self._ctl(c, "post", "/_control/clear")
                await self._seed(c, 1)
                e = (await self._ctl(c, "get", "/_control/history"))["history"][0]
                self.assertEqual((e["turn"], e["seq"]), (1, 5))
        _run(go())

    def test_entries_carry_correlation_fields(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG, headers={
                    "user-agent": "my-app/1.0", "x-trace-id": "trace-123", "x-api-key": "sk-secret",
                    "authorization": "Bearer tok", "x-amz-security-token": "s", "anthropic-version": "2023-06-01",
                    "cookie": "a=b", "x-forwarded-for": "10.1.2.3", "x-session-id": "sess", "x-password": "pw",
                    "x-stainless-os": "Linux", "x-b3-traceid": "b3t", "baggage": "sessionId=abc",
                    "amz-sdk-invocation-id": "inv-1", "x-stainless-api-key": "nope", "sentry-trace": "st",
                    "tracestate": "congo=1", "x-amz-date": "20260101T000000Z", "amz-sdk-request": "attempt=1; max=3"})
                snap = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(snap["path"], "/v1/messages")
                self.assertTrue(snap["request_id"].startswith("req_"))
                # an allow-list: identity and trace ids only, never a credential, a session
                # or a client address
                self.assertEqual(snap["headers"], {"user-agent": "my-app/1.0", "x-trace-id": "trace-123",
                                                   "anthropic-version": "2023-06-01", "x-stainless-os": "Linux",
                                                   "x-b3-traceid": "b3t", "amz-sdk-invocation-id": "inv-1",
                                                   "sentry-trace": "st"})
                await asyncio.sleep(0.15)
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok"})
                r = await t
                self.assertEqual(r.headers["request-id"], snap["request_id"])
                e = (await self._ctl(c, "get", "/_control/history/1"))
                self.assertEqual((e["request_id"], e["path"]), (snap["request_id"], "/v1/messages"))
                self.assertGreaterEqual(e["latency_ms"], 100)
                self.assertEqual((e["usage_source"], e["pricing_model"]), ("estimate", "claude-sonnet-5"))
                self.assertNotIn("provenance", e)
                # every route stamps its own request id and path
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                r = await self._answered(c, f"{_BEDROCK}/invoke", {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 5,
                                                                      "messages": [{"role": "user", "content": "hi"}]})
                e = (await self._ctl(c, "get", "/_control/history/2"))
                self.assertEqual((e["request_id"], e["path"]), (r.headers["x-amzn-requestid"], f"{_BEDROCK}/invoke"))
                r = await self._answered(c, f"{_BEDROCK}/converse", {"messages": [{"role": "user", "content": [{"text": "hi"}]}]})
                e = (await self._ctl(c, "get", "/_control/history/3"))
                self.assertEqual((e["request_id"], e["path"]), (r.headers["x-amzn-requestid"], f"{_BEDROCK}/converse"))
                r = await self._answered(c, "/v1/chat/completions", _OPENAI)
                e = (await self._ctl(c, "get", "/_control/history/4"))
                self.assertEqual((e["request_id"], e["path"]), (r.headers["x-request-id"], "/v1/chat/completions"))
        _run(go())

    def test_pending_summary_and_fields(self) -> None:
        async def go():
            async with await self._client() as c:
                body = {**_MSG, "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
                        "messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"},
                                     {"role": "user", "content": [{"type": "text", "text": "Tokyo weather please"}]}]}
                t = await self._pending(c, "/v1/messages", body)
                s = (await self._ctl(c, "get", "/_control/pending", params={"summary": 1}))
                item = s["pending"][0]["request"]
                self.assertEqual(item["message_count"], 3)
                self.assertEqual(item["tools"], ["get_weather"])
                self.assertEqual(item["last_user_text"], "Tokyo weather please")
                self.assertFalse(item["has_tool_result"])
                self.assertNotIn("messages", item)
                self.assertEqual(s["request"]["pending_id"], item["pending_id"])
                f = (await self._ctl(c, "get", "/_control/pending", params={"fields": "model,turn"}))
                self.assertEqual(set(f["pending"][0]["request"]), {"pending_id", "model", "turn"})
                full = (await self._ctl(c, "get", "/_control/pending"))["pending"][0]["request"]
                self.assertIn("messages", full)
                await self._ctl(c, "post", "/_control/auto", json={"text": "ok"})
                await t
        _run(go())

    def test_history_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)
        os.environ["PUPPETLLM_HISTORY_FILE"] = path
        self.mod = _import_fresh()

        async def go():
            async with await self._client() as c:
                await self._seed(c, 3)
                # a lone surrogate is legal JSON but not UTF-8: the entry is still mirrored
                # (escaped) and the request is still answered
                raw = json.dumps({**_MSG, "system": _SYSTEM, "messages": [{"role": "user", "content": "x\ud800y"}]}).encode("ascii")
                r = await self._answered_raw(c, "/v1/messages", raw)
                self.assertEqual(r.status_code, 200, r.text)
                r = await c.get("/_control/history", params={"format": "jsonl"})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(len(r.text.splitlines()), 4)
                # the JSON control views survive it too (escaped where needed) and decode back
                e = (await self._ctl(c, "get", "/_control/history/4"))
                self.assertEqual(e["request"]["messages"][0]["content"], "x\ud800y")
                self.assertEqual((await self._ctl(c, "get", "/_control/history"))["count"], 4)
                # ... including a surrogate in a parameter that feeds the cache key salt
                raw = json.dumps({**_MSG, "system": _SYSTEM, "tools": [{"name": "t\ud800", "input_schema": {"type": "object"}}],
                                  "tool_choice": {"type": "tool", "name": "t\ud800"},
                                  "messages": [{"role": "user", "content": "hi"}]}).encode("ascii")
                self.assertEqual((await self._answered_raw(c, "/v1/messages", raw)).status_code, 200)
                # a surrogate in the model id (keys of stats.by_model) and an infinite number
                raw = json.dumps({**_MSG, "model": "claude-x\ud800y", "temperature": 1e999}).encode("ascii").replace(b"Infinity", b"1e999")
                self.assertEqual((await self._answered_raw(c, "/v1/messages", raw)).status_code, 200)
                self.assertEqual((await c.get("/_control/stats")).status_code, 200)
                r = await c.get("/_control/history", params={"format": "jsonl", "since_turn": 5})
                self.assertEqual(json.loads(r.text)["request"]["params"]["temperature"], None)
                await self._ctl(c, "post", "/_control/clear")  # clearing history does not touch the file
        _run(go())
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual([e["turn"] for e in lines], [1, 2, 3, 4, 5, 6])
        self.assertIsNone(lines[5]["request"]["params"]["temperature"])
        self.assertIn("latency_ms", lines[0])
        self.assertEqual(lines[3]["request"]["messages"][0]["content"], "x\ud800y")
        # an unusable path never fails a request: warned once at import and once per error
        os.environ["PUPPETLLM_HISTORY_FILE"] = os.path.join(path + ".dir", "nope", "h.jsonl")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.mod = _import_fresh()

            async def go2():
                async with await self._client() as c:
                    await self._seed(c, 2)
            _run(go2())
        self.assertEqual(buf.getvalue().count("history file"), 1)
        self.assertIn("PUPPETLLM_HISTORY_FILE", buf.getvalue())


class TestCacheReasons(_Base):
    def test_reasons_and_diagnostics(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                diag = {"diagnostics": {"previous_message_id": None}}
                r = await self._answered(c, "/v1/messages", {**_MSG, **diag})
                cache = (await self._ctl(c, "get", "/_control/history/1"))["cache"]
                self.assertEqual((cache["status"], cache["reason"]), ("none", "no_cache_control"))
                self.assertEqual(r.json()["diagnostics"]["cache_miss_reason"], "no_cache_control")
                body = {**_MSG, "system": _SYSTEM, **diag}
                r = await self._answered(c, "/v1/messages", body)
                cache = (await self._ctl(c, "get", "/_control/history/2"))["cache"]
                self.assertEqual((cache["status"], cache["reason"], cache["matched_through"]), ("miss", "first_seen", 0))
                self.assertGreater(cache["eligible_prefix_tokens"], 0)
                self.assertEqual(cache["minimum_tokens"], 0)
                self.assertEqual(r.json()["diagnostics"]["cache_miss_reason"], "first_seen")
                r = await self._answered(c, "/v1/messages", {**body, "stream": True})
                events = _sse_events(r.text)
                self.assertEqual(events[0][1]["message"]["diagnostics"]["cache_status"], "hit")
                self.assertIsNone(events[0][1]["message"]["diagnostics"]["cache_miss_reason"])
                self.assertIsNone((await self._ctl(c, "get", "/_control/history/3"))["cache"]["reason"])
                # a changed system prompt, compared with the message the client names
                # (`previous_message_id`): the miss says where the prefix diverged
                prev_id = events[0][1]["message"]["id"]
                changed = {**body, "system": [{**_SYSTEM[0], "text": "different system prompt " * 40}],
                           "diagnostics": {"previous_message_id": prev_id}}
                r = await self._answered(c, "/v1/messages", changed)
                cache = (await self._ctl(c, "get", "/_control/history/4"))["cache"]
                self.assertEqual((cache["status"], cache["reason"], cache["first_changed_path"], cache["matched_through"]),
                                 ("miss", "prefix_changed", "system[0]", 0))
                self.assertEqual(r.json()["diagnostics"]["first_changed_path"], "system[0]")
                # the message after an intact system prefix: still a hit, the reason stays None
                await self._answered(c, "/v1/messages", {**changed, "messages": [{"role": "user", "content": "another"}]})
                self.assertEqual((await self._ctl(c, "get", "/_control/history/5"))["cache"]["status"], "hit")
                # without a nominated message the model's previous request is compared: a
                # different system prompt is a changed static prefix
                other = {**body, "system": [{**_SYSTEM[0], "text": "a third system prompt " * 40}]}
                await self._answered(c, "/v1/messages", other)
                cache = (await self._ctl(c, "get", "/_control/history/6"))["cache"]
                self.assertEqual((cache["reason"], cache["first_changed_path"]), ("prefix_changed", "system[0]"))
                # same content, a parameter the real API renders into the prompt changed
                await self._answered(c, "/v1/messages", {**other, "speed": "fast"})
                cache = (await self._ctl(c, "get", "/_control/history/7"))["cache"]
                self.assertEqual((cache["status"], cache["reason"], cache["first_changed_path"]),
                                 ("miss", "parameters_changed", "params.speed"))
                # TTL expiry explained by the fake clock: dead-but-indexed, and pruned while
                # the conversation grew by a turn
                await self._ctl(c, "post", "/_control/clock/advance", json={"seconds": 400})
                await self._answered(c, "/v1/messages", changed)
                cache = (await self._ctl(c, "get", "/_control/history/8"))["cache"]
                self.assertEqual((cache["status"], cache["reason"]), ("miss", "ttl_expired"))
                await self._ctl(c, "post", "/_control/clock/advance", json={"seconds": 700})
                grown = {**changed, "messages": [*changed["messages"], {"role": "assistant", "content": "a"},
                                                 {"role": "user", "content": "b"}]}
                grown.pop("diagnostics")
                await self._answered(c, "/v1/messages", grown)
                cache = (await self._ctl(c, "get", "/_control/history/9"))["cache"]
                self.assertEqual((cache["status"], cache["reason"], cache["matched_through"]), ("miss", "ttl_expired", 1))
                # no diagnostics unless asked for; only the Messages routes carry them
                r = await self._answered(c, "/v1/messages", {**_MSG, "system": _SYSTEM})
                self.assertNotIn("diagnostics", r.json())
                r = await self._answered(c, f"{_BEDROCK}/invoke", {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 5,
                                                                  "messages": [{"role": "user", "content": "hi"}], **diag})
                self.assertNotIn("diagnostics", r.json())
                r = await self._answered(c, "/v1/chat/completions", {**_OPENAI, **diag})
                self.assertNotIn("diagnostics", r.json())
                self.assertEqual((await self._ctl(c, "get", "/_control/history/12"))["cache"]["reason"], "disabled")
                r = await self._answered(c, "/anthropic/v1/messages", {**_MSG, "model": "anthropic.claude-sonnet-4-5-20250929-v1:0", **diag})
                self.assertIn("cache_miss_reason", r.json()["diagnostics"])
        _run(go())

    def test_below_minimum(self) -> None:
        os.environ["PUPPETLLM_CACHE_MIN_TOKENS"] = "100000"
        self.mod = _import_fresh()

        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                await self._answered(c, "/v1/messages", {**_MSG, "system": _SYSTEM})
                cache = (await self._ctl(c, "get", "/_control/history/1"))["cache"]
                self.assertEqual((cache["status"], cache["reason"], cache["minimum_tokens"]), ("none", "below_minimum", 100000))
                self.assertGreater(cache["eligible_prefix_tokens"], 0)
        _run(go())


class TestMetricsAndLogs(_Base):
    def test_metrics_exposition(self) -> None:
        """Counters are process-wide and cumulative (a Prometheus scrape target), so the
        assertions are on deltas; gauges are live values."""
        def parse(text: str) -> dict[str, float]:
            out: dict[str, float] = {}
            for line in text.splitlines():
                if line and not line.startswith("#"):
                    key, value = line.rsplit(" ", 1)
                    out[key] = float(value)
            return out

        async def go():
            async with await self._client() as c:
                r0 = await c.get("/_control/metrics")
                self.assertEqual(r0.status_code, 200)
                self.assertTrue(r0.headers["content-type"].startswith("text/plain"))
                before = parse(r0.text)
                await self._seed(c, 3)
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"rpm": 1}})
                self.assertEqual((await self._answered(c, "/v1/messages", _MSG)).status_code, 200)
                self.assertEqual((await self._answered(c, "/v1/messages", _MSG)).status_code, 429)
                self.assertEqual((await c.post("/v1/messages", json={**_MSG, "messages": [123]})).status_code, 400)
                self.assertEqual((await c.post("/v1/messages", content=b"{bad", headers={"content-type": "application/json"})).status_code, 400)
                self.assertEqual((await c.post("/v1/chat/completions", json={**_OPENAI, "messages": 1})).status_code, 400)
                self.assertEqual((await c.post(f"{_BEDROCK}/converse", content=b"{bad", headers={"content-type": "application/json"})).status_code, 400)
                self.assertEqual((await c.post(f"{_BEDROCK}/invoke", content=b"{bad", headers={"content-type": "application/json"})).status_code, 400)
                self.assertEqual((await c.post("/v1/messages/batches", json={"requests": [{"custom_id": "b", "params": {**_MSG, "messages": [123]}}]})).status_code, 400)
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": None})
                await self._ctl(c, "delete", "/_control/rules")
                # a batch canceled before its entry was answered ends that pending without an answer
                b = await self._ctl(c, "post", "/v1/messages/batches", json={"requests": [{"custom_id": "c1", "params": _MSG}]})
                await self._ctl(c, "post", f"/v1/messages/batches/{b['id']}/cancel")
                await asyncio.sleep(0.05)
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "later"}}]})
                text = (await c.get("/_control/metrics")).text
                after = parse(text)

                def delta(key: str) -> float:
                    return after.get(key, 0.0) - before.get(key, 0.0)
                self.assertEqual(delta('puppetllm_requests_total{outcome="ok",provider="anthropic"}'), 3)
                self.assertEqual(delta('puppetllm_requests_total{outcome="ok",provider="openai"}'), 1)
                self.assertEqual(delta('puppetllm_requests_total{outcome="error",provider="anthropic"}'), 1)
                self.assertEqual(delta('puppetllm_harness_answers_total{source="rate_limit"}'), 1)
                self.assertEqual(delta('puppetllm_harness_answers_total{source="rule"}'), 4)
                self.assertEqual(delta('puppetllm_rejected_total{provider="anthropic"}'), 3)
                self.assertEqual(delta('puppetllm_rejected_total{provider="openai"}'), 1)
                self.assertEqual(delta('puppetllm_rejected_total{provider="bedrock"}'), 2)
                self.assertEqual(delta('puppetllm_canceled_total{reason="batch"}'), 1)
                self.assertGreater(delta('puppetllm_tokens_total{kind="output",provider="anthropic"}'), 0)
                self.assertGreater(delta('puppetllm_cost_usd_total{provider="anthropic"}'), 0)
                self.assertGreater(delta('puppetllm_cache_lookups_total{status="none"}'), 0)
                self.assertEqual(after["puppetllm_pending"], 1)
                self.assertEqual(after["puppetllm_rules_unconsumed"], 1)
                self.assertIn("# TYPE puppetllm_requests_total counter", text)
                self.assertIn("# TYPE puppetllm_pending gauge", text)
                await self._ctl(c, "post", "/_control/clear")
                self.assertEqual((await t).status_code, 529)
                final = parse((await c.get("/_control/metrics")).text)
                # counters survive a clear; gauges follow the live state
                self.assertEqual(final['puppetllm_requests_total{outcome="ok",provider="anthropic"}'],
                                 after['puppetllm_requests_total{outcome="ok",provider="anthropic"}'])
                self.assertEqual(final.get('puppetllm_canceled_total{reason="clear"}', 0) - before.get('puppetllm_canceled_total{reason="clear"}', 0), 1)
                self.assertEqual((final["puppetllm_pending"], final["puppetllm_turns"]), (0, 0))
        _run(go())

    def test_json_log_events(self) -> None:
        from puppetllm import observability as obs
        buf = io.StringIO()
        obs.set_log_format("json")

        async def go():
            async with await self._client() as c:
                with contextlib.redirect_stderr(buf):
                    await self._ctl(c, "post", "/_control/rules", json={"id": "r1", "steps": [{"respond": {"text": "ok"}}]})
                    r = await self._answered(c, "/v1/messages", _MSG)
                    self.assertEqual(r.status_code, 200)
                    self.assertEqual((await c.post("/v1/messages", json={**_MSG, "tools": 1})).status_code, 400)
                    await self._ctl(c, "post", "/_control/config", json={"on_unmatched": "default", "default_response": "dflt"})
                    self.assertEqual((await self._answered(c, "/v1/messages", _MSG)).status_code, 200)
                    await self._ctl(c, "post", "/_control/config", json={"on_unmatched": "error"})
                    self.assertEqual((await self._answered(c, "/v1/messages", _MSG)).status_code, 500)
                    await self._ctl(c, "post", "/_control/config", json={"on_unmatched": "pending"})
                    t = await self._pending(c, "/v1/messages", _MSG)
                    await self._ctl(c, "post", "/_control/clear")
                    await t
        _run(go())
        events = [json.loads(line) for line in buf.getvalue().splitlines() if line.startswith("{")]
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[:3], ["received", "matched", "resolved"])
        self.assertIn("rejected", kinds)
        self.assertIn("canceled", kinds)
        received, matched, resolved = events[0], events[1], events[2]
        self.assertEqual(received["pending_id"], resolved["pending_id"])
        self.assertEqual(received["request_id"], resolved["request_id"])
        self.assertEqual((matched["source"], matched["rule_id"], matched["request_id"]), ("rule", "r1", received["request_id"]))
        self.assertEqual((resolved["outcome"], resolved["source"], resolved["usage_source"], resolved["seq"]), ("ok", "rule", "estimate", 1))
        self.assertIn("latency_ms", resolved)
        rejected = next(e for e in events if e["event"] == "rejected")
        self.assertEqual((rejected["provider"], rejected["path"]), ("anthropic", "/v1/messages"))
        self.assertEqual([e["source"] for e in events if e["event"] == "matched"], ["rule", "default", "unmatched"])
        self.assertEqual(next(e for e in events if e["event"] == "canceled")["reason"], "cleared by control")
        self.assertTrue(all("ts" in e for e in events))

    def test_text_mode_is_quiet_and_bedrock_lines_follow_the_format(self) -> None:
        from puppetllm import observability as obs
        obs.set_log_format("text")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            obs.log_event("received", turn=1)
            obs.log_text("bedrock", "invoke model=x")
        self.assertEqual(buf.getvalue(), "[bedrock] invoke model=x\n")
        buf = io.StringIO()
        obs.set_log_format("json")
        with contextlib.redirect_stderr(buf):
            obs.log_text("bedrock", "invoke model=x")
        line = json.loads(buf.getvalue())
        self.assertEqual((line["event"], line["source"], line["message"]), ("log", "bedrock", "invoke model=x"))
        with self.assertRaises(ValueError):
            obs.set_log_format("xml")


class TestProvenanceAndPricing(_Base):
    def test_provenance_is_recorded_and_prices_by_upstream_model(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {**_MSG, "model": "claude-haiku-4-5"})
                await self._ctl(c, "post", "/_control/respond", json={
                    "text": "from upstream", "usage": {"input_tokens": 1000, "output_tokens": 1000},
                    "provenance": {"upstream_provider": "openai", "upstream_model": "gpt-5.4",
                                   "usage_source": "upstream", "pricing_model": "gpt-5.4",
                                   "translation": {"dropped_fields": ["speed=fast"]}}})
                await t
                e = await self._ctl(c, "get", "/_control/history/1")
                self.assertEqual(e["provenance"], {"upstream_provider": "openai", "upstream_model": "gpt-5.4",
                                                   "usage_source": "upstream", "pricing_model": "gpt-5.4",
                                                   "translation": {"dropped_fields": ["speed=fast"]}})
                self.assertEqual((e["usage_source"], e["pricing_model"]), ("override", "gpt-5.4"))
                from puppetllm import pricing
                self.assertEqual(e["cost"]["model_family"], pricing.resolve_family("gpt-5.4"))
                expected = pricing.compute_cost("gpt-5.4", input_tokens=1000, output_tokens=1000)["total_usd"]
                self.assertAlmostEqual(e["cost"]["total_usd"], expected, places=9)
                # no override: the inbound model prices, provenance optional
                t = await self._pending(c, "/v1/messages", {**_MSG, "model": "claude-haiku-4-5"})
                await self._ctl(c, "post", "/_control/respond", json={"text": "x", "provenance": {"upstream_provider": "anthropic"}})
                await t
                e = await self._ctl(c, "get", "/_control/history/2")
                self.assertEqual((e["pricing_model"], e["usage_source"], e["provenance"]),
                                 ("claude-haiku-4-5", "estimate", {"upstream_provider": "anthropic"}))
                # errors carry it too (and price by it); invalid shapes are refused
                t = await self._pending(c, "/v1/messages", _MSG)
                for bad in ("x", {"bogus": 1}, {"upstream_model": 5}, {"translation": {"dropped_fields": "a"}},
                            {"translation": {"extra": 1}}, {"upstream_model": ""}):
                    self.assertEqual((await c.post("/_control/respond", json={"text": "x", "provenance": bad})).status_code, 400, bad)
                await self._ctl(c, "post", "/_control/error", json={"status": 502, "provenance": {
                    "upstream_provider": "openai", "upstream_model": "gpt-5.4", "pricing_model": "gpt-5.4",
                    "translation": {"dropped_fields": None}}})
                await t
                e = await self._ctl(c, "get", "/_control/history/3")
                self.assertEqual(e["provenance"]["upstream_model"], "gpt-5.4")
                self.assertEqual(e["provenance"]["translation"], {"dropped_fields": []})
                self.assertEqual((e["pricing_model"], e["usage_source"], e["usage"]), ("gpt-5.4", None, None))
                # cache savings and the Anthropic-only billing modifiers follow the pricing model
                t = await self._pending(c, "/v1/messages", {**_MSG, "model": "claude-haiku-4-5", "speed": "fast"})
                await self._ctl(c, "post", "/_control/respond", json={
                    "text": "x", "usage": {"input_tokens": 1000, "output_tokens": 1000, "cache_read_input_tokens": 1_000_000},
                    "provenance": {"pricing_model": "gpt-5.4"}})
                await t
                e = await self._ctl(c, "get", "/_control/history/4")
                self.assertEqual(e["cost"]["speed_multiplier"], 1.0)
                # a 1,001,000-token prompt is over gpt-5.4's 272K threshold: the long card
                # ($5 input, $0.50 cached, $22.50 output) prices the request and its savings
                self.assertTrue(e["cost"]["long_context"])
                self.assertAlmostEqual(e["cost"]["total_usd"], 0.005 + 0.5 + 0.0225, places=9)
                stats = await self._ctl(c, "get", "/_control/stats")
                self.assertAlmostEqual(stats["totals"]["cache_savings_usd"], 4.5, places=6)
                # fast mode and US inference scale the saving as they scale the prices
                t = await self._pending(c, "/v1/messages", {**_MSG, "model": "claude-fable-5-2",
                                                            "speed": "fast", "inference_geo": "us"})
                await self._ctl(c, "post", "/_control/respond", json={
                    "text": "x", "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 1_000_000}})
                await t
                stats = await self._ctl(c, "get", "/_control/stats")
                self.assertAlmostEqual(stats["totals"]["cache_savings_usd"], 4.5 + (8.0 - 0.2) * 2.2, places=6)
        _run(go())

    def test_pricing_file(self) -> None:
        from puppetllm import pricing
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"my-local-*": {"input": 0.5, "output": 2.0},
                       "claude-sonnet-5": {"input": 1.0, "output": 1.0, "cache_read": 0.05}}, f)
        try:
            self.assertEqual(pricing.load_pricing_file(f.name), 2)
            self.assertEqual(pricing.resolve_family("my-local-7b"), "my-local-*")
            self.assertEqual(pricing.price_for("my-local-7b").output, 2.0)
            self.assertEqual(pricing.price_for("my-local-7b").cache_write, 0.625)
            self.assertEqual(pricing.price_for("claude-sonnet-5").cache_read, 0.05)
            cost = pricing.compute_cost("my-local-7b", input_tokens=1_000_000, output_tokens=1_000_000)
            self.assertEqual((cost["model_family"], cost["total_usd"]), ("my-local-*", 2.5))
            self.assertNotEqual(pricing.resolve_family("claude-opus-5"), "claude-sonnet-5")
            # the environment variable loads the file at import (the table starts empty)
            pricing._CUSTOM_PRICES.clear()
            os.environ["PUPPETLLM_PRICING_FILE"] = f.name
            mod = _import_fresh()
            self.assertEqual(pricing.resolve_family("my-local-7b"), "my-local-*")

            async def go():
                import httpx
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mod.app), base_url="http://t") as c:
                    await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}]})
                    await self._answered(c, "/v1/messages", {**_MSG, "model": "my-local-7b"})
                    self.assertEqual((await self._ctl(c, "get", "/_control/history/1"))["cost"]["model_family"], "my-local-*")
            _run(go())
        finally:
            os.unlink(f.name)
            pricing._CUSTOM_PRICES.clear()
        for bad in ([], {"x": {"output": 1}}, {"x": {"input": -1, "output": 1}}, {"": {"input": 1, "output": 1}},
                    {"x": {"input": "nan", "output": 1}}, {"x": {"input": 1, "output": "Infinity"}},
                    {"x": {"input": "1", "output": 1}}, {"x": {"input": None, "output": 1}},
                    {"x": {"input": 1e308, "output": 1}}, {"x": {"input": 10 ** 400, "output": 1}},
                    {"x": {"input": 1e7, "output": 1}}, {"x": {"input": 1, "output": 1, "cache_read": 2}},
                    {"x": {"input": 1, "output": 1, "cache_wrte": 1}},
                    {"x": {"input": 1, "output": 1, "cache_read": -5}}, {"x": {"input": 1, "output": 1, "cache_read": {"y": 1}}},
                    {"x": {"input": True, "output": 1}}):
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(bad, f)
            try:
                with self.assertRaises(ValueError):
                    pricing.load_pricing_file(f.name)
            finally:
                os.unlink(f.name)
        # a glob that happens to spell a built-in family name prices only ids it matches;
        # a custom entry is flat, replacing the built-in long-context card as well
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"sonnet": {"input": 99, "output": 99},
                       "claude-haiku-5-*": {"input": 1, "output": 2}}, f)
        try:
            pricing.load_pricing_file(f.name)
            flat = pricing.compute_cost("claude-haiku-5-5", input_tokens=500_000)
            self.assertEqual((flat["long_context"], flat["input_usd"]), (False, 0.5))
            self.assertAlmostEqual(pricing.cache_savings_usd("claude-haiku-5-5", 1_000_000, long_context=True), 0.9)
            self.assertEqual(pricing.price_for("sonnet").input, 99)
            self.assertEqual(pricing.price_for("claude-sonnet-4-5").input, pricing._FAMILY_PRICES["sonnet"].input)
            self.assertEqual(pricing.compute_cost("claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=0)["total_usd"],
                             pricing._FAMILY_PRICES["sonnet"].input)
            with open(f.name, "w") as g:
                g.write("{}")
            self.assertEqual(pricing.load_pricing_file(f.name), 0)  # an empty object clears
            self.assertEqual(pricing._CUSTOM_PRICES, [])
        finally:
            os.unlink(f.name)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"big": {"input": 1_000_000, "output": 1_000_000}, "zero": {"input": -0.0, "output": 0}}, f)
        try:
            self.assertEqual(pricing.load_pricing_file(f.name), 2)
            self.assertEqual(pricing.price_for("big").cache_write_1h, 2_000_000.0)
            self.assertEqual(repr(pricing.price_for("zero").input), "0.0")
        finally:
            os.unlink(f.name)
            pricing._CUSTOM_PRICES.clear()
        self.assertTrue(pricing.is_fallback_family("llama3"))
        self.assertFalse(pricing.is_fallback_family("claude-opus-5"))
        # the Claude multipliers: built-in Claude tables and unknown Claude ids only
        fast = dict(input_tokens=1_000_000, output_tokens=0, speed="fast", inference_geo="us")
        self.assertEqual(pricing.compute_cost("llama3", **fast)["speed_multiplier"], 1.0)
        # ... unless the made-up id arrived on an Anthropic-shaped route
        self.assertEqual(pricing.compute_cost("llama3", inbound_anthropic=True, **fast)["speed_multiplier"], 2.0)
        self.assertEqual(pricing.compute_cost("claude-nova-9", **fast)["speed_multiplier"], 2.0)
        self.assertEqual(pricing.compute_cost("claude-sonnet-5", **fast)["geo_multiplier"], 1.1)
        self.assertEqual(pricing.compute_cost("gpt-5.4", inbound_anthropic=True, **fast)["speed_multiplier"], 1.0)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"sonnet": {"input": 1, "output": 1}}, f)
        try:
            pricing.load_pricing_file(f.name)
            self.assertEqual(pricing.compute_cost("sonnet", **fast)["total_usd"], 1.0)
        finally:
            os.unlink(f.name)
            pricing._CUSTOM_PRICES.clear()


def _relay_cfg(**overrides: Any) -> argparse.Namespace:
    base = dict(kind="openai", target="http://upstream.test/v1", api_key_env="RELAY_TEST_KEY",
                model=None, model_map=None, only=None, max_concurrency=0, timeout=5, poll_timeout=1,
                puppet="http://puppet.test", max_tokens_param="auto", max_requests=0,
                price_by="inbound", strict_translation=False, dry_run=False)
    base.update(overrides)
    return argparse.Namespace(**base)


class TestRelayProvenance(_Base):
    def test_reverse_format_translation_and_dropped_sink(self) -> None:
        from puppetllm.relay import to_openai_request
        req = {"messages": [{"role": "user", "content": "x"}], "params": {
            "output_config": {"format": {"type": "json_schema", "name": "answer",
                                         "schema": {"type": "object", "properties": {"a": {"type": "string"}}}},
                              "effort": "max"},
            "speed": "fast", "service_tier": "weird"}}
        dropped: list[str] = []
        body = to_openai_request(req, "gpt-5.4", None, dropped)
        # no `strict`: OpenAI's strict mode has schema requirements an Anthropic schema
        # need not meet
        self.assertEqual(body["response_format"], {"type": "json_schema", "json_schema": {
            "name": "answer", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}})
        self.assertEqual(body["reasoning_effort"], "xhigh")
        self.assertEqual(sorted(dropped), ["service_tier='weird'",
                                           "speed=fast (Anthropic fast mode has no Chat Completions equivalent)"])
        dropped = []
        to_openai_request({"messages": [], "params": {"output_config": {"format": {"type": "nope"}}}}, "m", None, dropped)
        self.assertEqual(dropped, ["output_config.format.type='nope'"])
        # silently omitted parameters and content are reported too
        dropped = []
        body = to_openai_request({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "see"}, {"type": "document", "source": {"type": "text", "data": "d"}}]}],
            "params": {"top_k": 5, "thinking": {"type": "enabled", "budget_tokens": 1024}, "cache_control": {"type": "ephemeral"}}},
            "m", None, dropped)
        self.assertNotIn("top_k", body)
        self.assertEqual(sorted(dropped), ["document block (content omitted)", "thinking (no Chat Completions equivalent)",
                                           "top_k (no Chat Completions equivalent)"])
        from puppetllm.relay import to_anthropic_request
        dropped = []
        to_anthropic_request({"messages": [], "params": {"seed": 1, "logit_bias": {"1": 2}, "temperature": 0.5,
                                                         "verbosity": "low", "n": 3}}, "m", None, dropped)
        self.assertEqual(sorted(dropped), ["logit_bias (no Messages API equivalent)", "n (the relay forwards one choice)",
                                           "seed (no Messages API equivalent)", "verbosity (no Messages API equivalent)"])
        # default-asking values lose nothing; translatable OpenAI params are translated
        dropped = []
        body = to_anthropic_request({"messages": [], "tools": [{"name": "t", "input_schema": {"type": "object"}}], "params": {
            "frequency_penalty": 0, "presence_penalty": 0.0, "logprobs": False, "logit_bias": {}, "top_logprobs": None,
            "parallel_tool_calls": False, "user": "u-1", "mcp_servers": [{"type": "url", "url": "http://x", "name": "m"}],
            "fallbacks": {"models": ["other"]}}}, "m", None, dropped)
        self.assertEqual(dropped, [])
        self.assertEqual(body["tool_choice"], {"type": "auto", "disable_parallel_tool_use": True})
        self.assertEqual(body["metadata"], {"user_id": "u-1"})
        self.assertEqual(body["mcp_servers"][0]["name"], "m")
        self.assertIn("fallbacks", body)
        dropped = []
        to_openai_request({"messages": [], "params": {"thinking": {"type": "disabled"}, "top_k": 0, "seed": 0}}, "m", None, dropped)
        self.assertEqual(dropped, [])
        dropped = []
        to_openai_request({"model": "claude-sonnet-5", "messages": [], "params": {"thinking": {"type": "adaptive"}}}, "m", None, dropped)
        self.assertEqual(dropped, [])  # what the SDKs send by default on a model that defaults to it
        dropped = []
        body = to_openai_request({"messages": [], "tools": [
            {"type": "web_search_20250305", "name": "web_search"}, {"type": "mcp_toolset", "mcp_server_name": "s"},
            {"name": "ok", "input_schema": {"type": "object"}}],
            "params": {"output_config": {"task_budget": {"type": "fixed", "tokens": 1}}, "n": 2,
                       "tool_choice": {"type": "allowed_tools", "tools": []}}}, "m", None, dropped)
        self.assertEqual([t["function"]["name"] for t in body["tools"]], ["ok"])
        self.assertEqual(body["tool_choice"], {"type": "allowed_tools", "tools": []})  # OpenAI's own form passes
        self.assertEqual(sorted(dropped), ["n (the relay forwards one choice)",
                                           "output_config.task_budget (no Chat Completions equivalent)",
                                           "tool type 'mcp_toolset' (no Chat Completions equivalent)",
                                           "tool type 'web_search_20250305' (no Chat Completions equivalent)"])
        # only server-side tools: no empty `tools`, and the tool_choice goes with them
        dropped = []
        body = to_openai_request({"messages": [], "tools": [{"type": "web_search_20250305", "name": "web_search"}, {"name": ""}],
                                  "params": {"tool_choice": {"type": "any"}}}, "m", None, dropped)
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)
        self.assertEqual(sorted(dropped), ["tool type 'web_search_20250305' (no Chat Completions equivalent)",
                                           "tool without a name (omitted)",
                                           "tool_choice (its tools have no Chat Completions equivalent)"])
        dropped = []
        to_openai_request({"messages": [], "params": {"tool_choice": {"type": "weird"}}}, "m", None, dropped)
        self.assertEqual(dropped, ["tool_choice 'weird' (no Chat Completions equivalent)"])
        # `adaptive` is a no-op only where it is the model's default
        dropped = []
        to_openai_request({"model": "claude-opus-4-8", "messages": [], "params": {"thinking": {"type": "adaptive"}}}, "m", None, dropped)
        self.assertEqual(dropped, ["thinking (no Chat Completions equivalent)"])
        dropped = []
        to_openai_request({"model": "claude-haiku-4-5", "messages": [], "params": {"thinking": {"type": "disabled"}}}, "m", None, dropped)
        self.assertEqual(dropped, [])
        dropped = []
        body = to_anthropic_request({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hear"}, {"type": "input_audio", "input_audio": {"data": "AAA", "format": "wav"}}]}],
            "tools": [{"name": "t", "input_schema": {"type": "object"}}],
            "params": {"user": "u" * 300, "verbosity": "medium", "modalities": ["text"], "parallel_tool_calls": False,
                       "tool_choice": {"type": "auto", "disable_parallel_tool_use": False}}}, "m", None, dropped)
        self.assertEqual(sorted(dropped), ["input_audio part (no Messages API equivalent; content omitted)",
                                           "user (longer than the 256 characters metadata.user_id allows)"])
        self.assertEqual(body["messages"][0]["content"][1], {"type": "text", "text": "[non-text content omitted by relay]"})
        # custom forced-tool form maps; the inbound snapshot is untouched by the replacement
        inbound = {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "A", "format": "wav"}}]}],
                   "params": {"tool_choice": {"type": "custom", "custom": {"name": "c"}}}}
        body2 = to_anthropic_request(inbound, "m")
        self.assertEqual(body2["tool_choice"], {"type": "tool", "name": "c"})
        self.assertEqual(inbound["messages"][0]["content"][0]["type"], "input_audio")
        self.assertNotIn("metadata", body)
        self.assertEqual(body["tool_choice"], {"type": "auto", "disable_parallel_tool_use": False})
        # one sink entry per distinct loss, however many blocks
        dropped = []
        to_openai_request({"messages": [{"role": "user", "content": [
            {"type": "document", "source": {"type": "text", "data": "d"}} for _ in range(5)]}]}, "m", None, dropped)
        self.assertEqual(dropped, ["document block (content omitted)"])

    def _drive(self, cfg: argparse.Namespace, upstream_handler: Any, inbound: dict[str, Any],
               path: str = "/v1/messages") -> dict[str, Any]:
        """Run Relay.handle against the in-process server with a fake upstream; returns
        the recorded history entry."""
        import httpx
        from puppetllm.relay import Relay
        os.environ["RELAY_TEST_KEY"] = "k"
        fs = self.mod

        async def go() -> dict[str, Any]:
            relay = Relay(cfg)
            await relay.http.aclose()
            await relay.ctl.aclose()
            relay.http = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler),
                                           headers={"Authorization": "Bearer k"})
            relay.ctl = httpx.AsyncClient(transport=httpx.ASGITransport(app=fs.app), base_url="http://puppet.test")
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fs.app), base_url="http://t") as c:
                    t = await self._pending(c, path, inbound)
                    item = (await c.get("/_control/pending")).json()["pending"][0]
                    await relay.handle(item["pending_id"], item)
                    await t
                    self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
                    return (await c.get("/_control/history")).json()["history"][-1]
            finally:
                await relay.close()
        return _run(go())

    def test_handle_records_provenance_and_prices_by_upstream(self) -> None:
        import httpx

        def upstream(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                                             "usage": {"prompt_tokens": 100, "completion_tokens": 50}})
        inbound = {**_MSG, "model": "claude-haiku-4-5", "speed": "fast"}
        e = self._drive(_relay_cfg(model="gpt-5.4", price_by="upstream"), upstream, inbound)
        self.assertEqual(e["provenance"]["upstream_provider"], "openai")
        self.assertEqual(e["provenance"]["upstream_model"], "gpt-5.4")
        self.assertEqual(e["provenance"]["usage_source"], "upstream")
        self.assertEqual(e["provenance"]["pricing_model"], "gpt-5.4")
        self.assertIn("speed=fast (Anthropic fast mode has no Chat Completions equivalent)",
                      e["provenance"]["translation"]["dropped_fields"])
        self.assertEqual((e["usage_source"], e["pricing_model"]), ("override", "gpt-5.4"))
        from puppetllm import pricing
        self.assertEqual(e["cost"]["model_family"], pricing.resolve_family("gpt-5.4"))
        # the inbound fast-mode multiplier does not reach an OpenAI price table
        self.assertEqual(e["cost"]["speed_multiplier"], 1.0)
        # an xAI upstream: 200k prompt tokens put the request on Grok's doubled card, and
        # reported cache writes count toward the prompt but carry no surcharge there
        def grok(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 200_000, "completion_tokens": 1000,
                          "prompt_tokens_details": {"cached_tokens": 100_000, "cache_write_tokens": 50_000}}})
        e = self._drive(_relay_cfg(model="grok-4.7", price_by="upstream"), grok, inbound)
        self.assertEqual((e["cost"]["model_family"], e["cost"]["long_context"]), ("grok", True))
        self.assertAlmostEqual(e["cost"]["total_usd"],
                               (50_000 * 4.0 + 50_000 * 4.0 + 100_000 * 1.0 + 1000 * 12.0) / 1e6, places=9)
        # GPT-6.x charges for the cache writes an upstream reports, at the 30-minute rate
        def gpt6(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100,
                          "prompt_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 300}}})
        e = self._drive(_relay_cfg(model="gpt-6.2-sol", price_by="upstream"), gpt6, inbound)
        self.assertEqual((e["usage"]["input_tokens"], e["usage"]["cache_creation_input_tokens"],
                          e["usage"]["cache_read_input_tokens"]), (500, 300, 200))
        self.assertAlmostEqual(e["cost"]["cache_write_usd"], 300 * 2.5 / 1e6, places=9)
        self.assertAlmostEqual(e["cost"]["total_usd"], (500 * 2.0 + 300 * 2.5 + 200 * 0.1 + 100 * 10.0) / 1e6, places=9)
        e = self._drive(_relay_cfg(model="gpt-5.4"), upstream, {**_MSG, "model": "claude-haiku-4-5"})
        self.assertNotIn("pricing_model", e["provenance"])
        self.assertEqual(e["pricing_model"], "claude-haiku-4-5")
        self.assertEqual(e["provenance"]["translation"]["dropped_fields"], [])

    def test_every_failure_path_resolves_the_pending_with_provenance(self) -> None:
        import httpx

        def upstream_400(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "no model"}})
        # no inbound model and no --model: the resolved upstream model is empty, and the
        # relayed error still ends the app's request
        e = self._drive(_relay_cfg(model=None), upstream_400, {"messages": [{"role": "user", "content": "hi"}]},
                        path="/v1/chat/completions")
        self.assertEqual(e["injected_error"]["status"], 400)
        self.assertEqual(e["provenance"]["upstream_provider"], "openai")
        self.assertNotIn("upstream_model", e["provenance"])

        def unreachable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")
        e = self._drive(_relay_cfg(model="gpt-5.4"), unreachable, _MSG)
        self.assertEqual(e["injected_error"]["status"], 502)
        self.assertEqual(e["provenance"]["upstream_model"], "gpt-5.4")

        def odd_shape(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"nothing": True})
        e = self._drive(_relay_cfg(model="gpt-5.4"), odd_shape, _MSG)
        self.assertEqual(e["injected_error"]["status"], 502)
        self.assertEqual(e["provenance"]["upstream_model"], "gpt-5.4")
        # an upstream whose table the relay does not know: one warning, costs still recorded
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            e = self._drive(_relay_cfg(model="llama3", price_by="upstream"), odd_shape, _MSG)
        self.assertEqual(buf.getvalue().count("no built-in price table"), 1)
        self.assertEqual(e["provenance"]["pricing_model"], "llama3")

    def test_rejected_error_injection_still_ends_the_pending(self) -> None:
        """When the control plane refuses the relay's own error injection, a bare 502 is
        posted so the app's request never hangs, and the pid is quarantined."""
        import httpx
        from puppetllm.relay import Relay
        posts: list[dict[str, Any]] = []

        def ctl(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            posts.append({"path": request.url.path, **body})
            if len(posts) == 1:
                return httpx.Response(400, json={"error": "provenance.upstream_model must be a non-empty string"})
            return httpx.Response(200, json={"ok": True})

        async def go() -> bool:
            relay = Relay(_relay_cfg(model="gpt-5.4"))
            await relay.ctl.aclose()
            relay.ctl = httpx.AsyncClient(transport=httpx.MockTransport(ctl), base_url="http://puppet.test")
            try:
                return await relay._inject("p1", "/_control/error", {"status": 400, "type": "invalid_request_error",
                                                                    "message": "x", "provenance": {"upstream_model": ""}})
            finally:
                await relay.close()
        os.environ["RELAY_TEST_KEY"] = "k"
        self.assertTrue(_run(go()))
        self.assertEqual([p["path"] for p in posts], ["/_control/error", "/_control/error"])
        self.assertEqual((posts[1]["status"], posts[1]["type"], posts[1]["pending_id"]), (502, "api_error", "p1"))
        self.assertNotIn("provenance", posts[1])  # provenance was the refused part: the fallback goes bare
        # a respond payload refused for another reason: the fallback keeps the (billed) provenance
        posts.clear()

        def ctl2(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            posts.append({"path": request.url.path, **body})
            if len(posts) == 1:
                return httpx.Response(400, json={"error": "content[0]: unknown block type"})
            return httpx.Response(200, json={"ok": True})

        async def go2() -> bool:
            relay = Relay(_relay_cfg(model="gpt-5.4"))
            await relay.ctl.aclose()
            relay.ctl = httpx.AsyncClient(transport=httpx.MockTransport(ctl2), base_url="http://puppet.test")
            try:
                return await relay._inject("p2", "/_control/respond", {"content": [{"type": "odd"}],
                                                                      "provenance": {"upstream_provider": "openai", "upstream_model": "gpt-5.4"}})
            finally:
                await relay.close()
        self.assertTrue(_run(go2()))
        self.assertEqual([p["path"] for p in posts], ["/_control/respond", "/_control/error"])
        self.assertEqual((posts[1]["status"], posts[1]["provenance"]["upstream_model"]), (502, "gpt-5.4"))

    def test_strict_translation_and_dry_run(self) -> None:
        import httpx
        calls: list[str] = []

        def upstream(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]})
        inbound = {**_MSG, "speed": "fast"}
        e = self._drive(_relay_cfg(model="gpt-5.4", strict_translation=True), upstream, inbound)
        self.assertEqual(e["injected_error"]["status"], 400)
        self.assertIn("speed=fast", e["injected_error"]["message"])
        self.assertEqual(e["provenance"]["upstream_model"], "gpt-5.4")
        self.assertEqual(calls, [])  # nothing was sent upstream
        # content the translation would silently replace counts as untranslatable too
        with_doc = {**_MSG, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "read"}, {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "d"}}]}]}
        e = self._drive(_relay_cfg(model="gpt-5.4", strict_translation=True), upstream, with_doc)
        self.assertEqual(e["injected_error"]["status"], 400)
        self.assertIn("document block", e["injected_error"]["message"])
        self.assertEqual(calls, [])
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            e = self._drive(_relay_cfg(model="gpt-5.4", dry_run=True), upstream, _MSG)
        self.assertEqual((e["injected_error"]["status"], e["injected_error"]["type"]), (400, "invalid_request_error"))
        self.assertEqual(calls, [])
        self.assertIn('"model": "gpt-5.4"', buf.getvalue())
        e = self._drive(_relay_cfg(model="gpt-5.4", strict_translation=True), upstream, _MSG)
        self.assertNotIn("injected_error", e)  # a fully translatable request goes through
        self.assertEqual(calls, ["/v1/chat/completions"])
        self.assertEqual((e["provenance"]["usage_source"], e["usage_source"]), ("estimate", "estimate"))

    def test_cli_flags(self) -> None:
        from puppetllm.relay import build_parser
        args = build_parser().parse_args(["--price-by", "upstream", "--strict-translation", "--dry-run"])
        self.assertEqual((args.price_by, args.strict_translation, args.dry_run), ("upstream", True, True))
        self.assertEqual(build_parser().parse_args([]).price_by, "inbound")
        from puppetllm.relay import _abbreviate
        self.assertEqual(_abbreviate({"a": ["x" * 300]})["a"][0][-12:], " (300 chars)")
        from puppetllm.__main__ import build_parser as serve_parser
        a = serve_parser().parse_args(["serve", "--log-format", "json", "--history-file", "h.jsonl", "--pricing-file", "p.json"])
        self.assertEqual((a.log_format, a.history_file, a.pricing_file), ("json", "h.jsonl", "p.json"))


class TestCacheSimulatorExplanations(unittest.TestCase):
    """The explanation paths that need a shaped conversation: driven on the simulator."""

    @staticmethod
    def _conv(n: int, tag: str = "a", tools: int = 0) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        for i in range(n):
            msgs.append({"role": "user", "content": [{"type": "text", "text": f"{tag} user {i} " * 20}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"{tag} asst {i} " * 20}]})
        for k in range(tools):  # a run of sequential tool calls, each its own position
            msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"u{k}", "name": "t", "input": {"k": k}}]})
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"u{k}", "content": "r"}]})
        msgs[-1]["content"][0]["cache_control"] = {"type": "ephemeral"}
        return msgs

    def test_lookback_and_expiry_via_the_previous_request(self) -> None:
        from puppetllm import cache_sim as cs
        sim = cs.CacheSimulator(min_cacheable_tokens=0)
        model = "claude-opus-5"
        first = sim.observe(cs.analyze_request(None, None, self._conv(2)), model, 0)
        self.assertEqual(first["reason"], "first_seen")
        # the marker moved 25 tool positions past the cached prefix: unreachable
        o = sim.observe(cs.analyze_request(None, None, self._conv(2, tools=25)), model, 1)
        self.assertEqual((o["status"], o["reason"], o["matched_through"]), ("miss", "lookback", 4))
        # pruned (> 2x TTL) and grown by a turn: the previous request's prefix expired
        sim = cs.CacheSimulator(min_cacheable_tokens=0)
        sim.observe(cs.analyze_request(None, None, self._conv(2)), model, 0)
        o = sim.observe(cs.analyze_request(None, None, self._conv(3)), model, 700)
        self.assertEqual((o["reason"], o["matched_through"]), ("ttl_expired", 4))
        # a nominated older message is what the comparison uses
        sim = cs.CacheSimulator(min_cacheable_tokens=0)
        old = sim.observe(cs.analyze_request(None, None, self._conv(1, "x")), model, 0)
        sim.remember_message("msg_old", old.pop("_fingerprint"))
        sim.observe(cs.analyze_request(None, None, self._conv(1, "y")), model, 1)
        edited = self._conv(2, "x")
        edited[1]["content"][0]["text"] = "edited"
        o = sim.observe(cs.analyze_request(None, None, edited), model, 2,
                        previous=sim.message_fingerprint("msg_old"))
        self.assertEqual((o["reason"], o["matched_through"], o["first_changed_path"]),
                         ("prefix_changed", 1, "messages[1].content[0]"))
        # without the nomination the comparison is with the model's previous request (y)
        o = sim.observe(cs.analyze_request(None, None, edited), model, 3)
        self.assertEqual(o["status"], "hit")
        o = sim.observe(cs.analyze_request(None, None, self._conv(1, "w")), model, 3)
        self.assertEqual((o["reason"], o["matched_through"]), ("first_seen", 0))
        self.assertIsNone(sim.message_fingerprint("unknown"))
        # the per-model memory keeps the models in use, not the first ones seen
        sim = cs.CacheSimulator(min_cacheable_tokens=0)
        sim.observe(cs.analyze_request(None, None, self._conv(1, "z")), model, 3)
        for i in range(255):
            sim.observe(cs.analyze_request(None, None, self._conv(1, "z")), f"m{i}", 3)
        sim.observe(cs.analyze_request(None, None, self._conv(1, "z")), model, 3)  # touched: most recent
        sim.observe(cs.analyze_request(None, None, self._conv(1, "z")), "m-new", 3)  # evicts the oldest: m0
        self.assertEqual(len(sim._last_request), 256)
        self.assertIn(model, sim._last_request)
        self.assertNotIn("m0", sim._last_request)
        # a conversation sharing only the static prefix with the model's previous request is new
        sim = cs.CacheSimulator(min_cacheable_tokens=0)
        system = [{"type": "text", "text": "shared system " * 30}]
        sim.observe(cs.analyze_request(system, None, self._conv(1, "p")), model, 0)
        o = sim.observe(cs.analyze_request(system, None, self._conv(1, "q")), model, 1)
        self.assertEqual((o["reason"], o["matched_through"]), ("first_seen", 1))
        changed_system = [{"type": "text", "text": "other system " * 30}]
        o = sim.observe(cs.analyze_request(changed_system, None, self._conv(1, "q")), model, 2)
        self.assertEqual((o["reason"], o["first_changed_path"]), ("prefix_changed", "system[0]"))
        o = sim.observe(cs.analyze_request(changed_system, None, self._conv(1, "q"), params={"speed": "fast"}), model, 2.5)
        self.assertEqual(o["reason"], "parameters_changed")
        # ... the same for the first tool, and a single string system without tools
        tools = [{"name": "t0", "input_schema": {"type": "object"}}, {"name": "t1", "input_schema": {"type": "object"}}]
        sim.observe(cs.analyze_request(changed_system, tools, self._conv(1, "q")), model, 3)
        o = sim.observe(cs.analyze_request(changed_system, [{**tools[0], "description": "d"}, tools[1]], self._conv(1, "q")), model, 4)
        self.assertEqual((o["reason"], o["first_changed_path"]), ("prefix_changed", "tools[0]"))
        sim.observe(cs.analyze_request("plain system " * 30, None, self._conv(1, "r")), model, 5)
        o = sim.observe(cs.analyze_request("other plain system " * 30, None, self._conv(1, "r")), model, 6)
        self.assertEqual((o["reason"], o["first_changed_path"]), ("prefix_changed", "system"))
        # no static prefix at all: a different first message is another conversation
        sim.observe(cs.analyze_request(None, None, self._conv(1, "s")), model, 7)
        o = sim.observe(cs.analyze_request(None, None, self._conv(1, "t")), model, 8)
        self.assertEqual((o["reason"], o["matched_through"]), ("first_seen", 0))
        # a lone surrogate anywhere in the request is fingerprinted, not fatal
        sim.observe(cs.analyze_request(None, None, [{"role": "user", "content": [
            {"type": "text", "text": "x\ud800y", "cache_control": {"type": "ephemeral"}}]}]), model, 4)


class TestPuppetHelpers(unittest.TestCase):
    def test_client_helpers(self) -> None:
        import threading

        import httpx
        from puppetllm.testing import serve
        with serve() as p:
            p.expect(model="claude-*").respond(text="one")
            p.expect(model="claude-*").respond(text="two")
            for _ in range(2):
                self.assertEqual(httpx.post(p.url + "/v1/messages", json=_MSG, timeout=10).status_code, 200)
            page = p.history_page(limit=1)
            self.assertEqual(([e["turn"] for e in page["history"]], page["has_more"]), ([1], True))
            self.assertEqual(page["next_cursor"], page["history"][0]["seq"])
            self.assertEqual([e["turn"] for e in p.history(cursor=page["next_cursor"])], [2])
            self.assertEqual(len(p.history(model="claude-*", provider="anthropic")), 2)
            self.assertEqual(p.history_entry(2)["response_blocks"][0]["text"], "two")
            self.assertIn("puppetllm_requests_total", p.metrics())
            got: list[Any] = []
            t = threading.Thread(target=lambda: got.append(httpx.post(p.url + "/v1/messages", json=_MSG, timeout=10)))
            t.start()
            p.wait_pending()
            self.assertEqual(p.pending(summary=True)[0]["request"]["message_count"], 1)
            self.assertEqual(set(p.pending(fields=["model"])[0]["request"]), {"pending_id", "model"})
            p.respond(text="three")
            t.join(10)
            self.assertEqual(got[0].status_code, 200)
            # a client that gives up while pending: the pending goes, counted as a disconnect
            def disconnects() -> float:
                for line in p.metrics().splitlines():
                    if line.startswith('puppetllm_canceled_total{reason="disconnect"}'):
                        return float(line.rsplit(" ", 1)[1])
                return 0.0
            before = disconnects()
            with self.assertRaises(httpx.TimeoutException):
                httpx.post(p.url + "/v1/messages", json=_MSG, timeout=0.5)
            for _ in range(50):
                if not p.pending():
                    break
                time.sleep(0.1)
            self.assertEqual(p.pending(), [])
            self.assertEqual(disconnects(), before + 1)


if __name__ == "__main__":
    unittest.main()
