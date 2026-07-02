"""Unit tests for puppetllm.fake_server.

Pure logic (SSE construction / response shape) is verified synchronously.
The HTTP layer is driven asynchronously with httpx.AsyncClient + ASGITransport,
and finally a round-trip through the anthropic SDK confirms SDK compatibility.

Run (from the repo root, or from under tools/):
  python3 -m unittest puppetllm.tests.test_fake_server -v

Via Docker:
  docker compose -f puppetllm/docker-compose.yml --profile test run --rm proxy-test
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import unittest
from typing import Any

# HTTP tests use small prefixes, so disable the minimum cache threshold.
# (The floor behavior itself is verified by the CacheSimulator unit tests. Setting this
# here lets the plain `python3 -m unittest ...` from the docstring run as-is.)
os.environ["PUPPETLLM_CACHE_MIN_TOKENS"] = "0"


def _import_fresh():
    """Reload the fake_server module to keep server state clean."""
    import importlib
    from puppetllm import fake_server as fs
    importlib.reload(fs)
    return fs


# ── Pure logic (no HTTP required) ────────────────────────────────────────


class TestSSEBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

    def test_text_only(self) -> None:
        events = self.mod._build_sse_stream(
            "msg_x", "claude-test",
            [{"type": "text", "text": "hello"}],
        )
        joined = b"".join(events).decode("utf-8")
        self.assertIn("event: message_start", joined)
        self.assertIn("event: content_block_start", joined)
        self.assertIn("event: content_block_delta", joined)
        self.assertIn("text_delta", joined)
        self.assertIn("event: content_block_stop", joined)
        self.assertIn("event: message_delta", joined)
        self.assertIn("event: message_stop", joined)
        # for text-only, stop_reason is end_turn
        self.assertIn('"end_turn"', joined)

    def test_tool_use_block(self) -> None:
        events = self.mod._build_sse_stream(
            "msg_y", "claude-test",
            [{"type": "tool_use", "id": "tu_1", "name": "Bash",
              "input": {"command": "ls"}}],
        )
        joined = b"".join(events).decode("utf-8")
        self.assertIn('"tool_use"', joined)
        self.assertIn("input_json_delta", joined)
        # input is embedded in partial_json as a JSON string (escaped quotes)
        self.assertIn("command", joined)
        self.assertIn("ls", joined)

    def test_mixed_blocks(self) -> None:
        events = self.mod._build_sse_stream(
            "msg_z", "claude-test",
            [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "tu_a", "name": "Bash", "input": {"x": 1}},
            ],
        )
        joined = b"".join(events).decode("utf-8")
        # the `event: content_block_start` line (excluding the same string inside data) appears twice
        self.assertEqual(joined.count("event: content_block_start"), 2)
        # when mixed, stop_reason prefers tool_use
        self.assertIn('"tool_use"', joined.split("message_delta")[-1])

    def test_empty_text(self) -> None:
        """Even empty text still emits events (start/delta/stop structure)."""
        events = self.mod._build_sse_stream("m", "c", [{"type": "text", "text": ""}])
        joined = b"".join(events).decode("utf-8")
        self.assertIn("content_block_start", joined)
        self.assertIn("content_block_stop", joined)


class TestNonStreamResponse(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

    def test_basic_shape(self) -> None:
        msg = self.mod._build_non_stream_response(
            "msg_n", "claude-test",
            [{"type": "text", "text": "hi"}],
        )
        self.assertEqual(msg["id"], "msg_n")
        self.assertEqual(msg["model"], "claude-test")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["stop_reason"], "end_turn")
        self.assertEqual(msg["content"], [{"type": "text", "text": "hi"}])

    def test_stop_reason_tool_use(self) -> None:
        msg = self.mod._build_non_stream_response(
            "msg_n", "claude-test",
            [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}],
        )
        self.assertEqual(msg["stop_reason"], "tool_use")


# ── HTTP layer (driven async via httpx ASGITransport) ────────────────


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestAsyncHttpx(unittest.TestCase):
    """Asynchronously verify fake_server's HTTP paths with httpx + ASGITransport.

    Starting a sync TestClient via threading interacts poorly with the async lock
    and becomes flaky, so this is written as a pure async path.
    """

    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app),
            base_url="http://test",
        )

    def test_health_and_idle(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                r = await c.get("/_control/health")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["turn_count"], 0)
                r2 = await c.get("/_control/pending")
                self.assertFalse(r2.json()["has_pending"])
                self.assertEqual(r2.json()["count"], 0)
        _run(run())

    def test_respond_without_pending(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                r = await c.post("/_control/respond", json={"content": []})
                self.assertEqual(r.status_code, 400)
        _run(run())

    def test_malformed_json_body_returns_400(self) -> None:
        """A broken JSON body returns a clear 400, not a 500.

        Regression test ensuring that when a responder sends render_chart's deeply
        nested input via curl and the shell escaping breaks, producing malformed JSON,
        we return an intelligible 400 rather than an opaque 500.
        """
        async def run() -> None:
            async with await self._client() as c:
                # control endpoints: the traditional plain form {"error": "invalid JSON body: ..."}
                for path in ("/_control/respond", "/_control/auto", "/_control/error"):
                    r = await c.post(
                        path, content=b"{bad json,,}",
                        headers={"Content-Type": "application/json"},
                    )
                    self.assertEqual(r.status_code, 400, f"{path} should 400 on bad JSON")
                    self.assertIn("invalid JSON body", r.json().get("error", ""))
                # Anthropic path: returns the official error envelope {"type":"error","error":{...}}
                r = await c.post(
                    "/v1/messages", content=b"{bad json,,}",
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(r.status_code, 400)
                self.assertEqual(r.json()["type"], "error")
                self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
                self.assertIn("invalid JSON body", r.json()["error"]["message"])
                # valid JSON that is not an object (an array) is also 400
                r = await c.post("/_control/respond", json=[1, 2, 3])
                self.assertEqual(r.status_code, 400)
        _run(run())

    def test_non_stream_round_trip(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # submit /v1/messages as a background task
                req_task = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "claude-test", "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                }, timeout=10))

                # wait until it becomes pending
                for _ in range(50):
                    p = await c.get("/_control/pending")
                    if p.json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                else:
                    self.fail("never became pending")

                # respond
                ar = await c.post("/_control/auto", json={"text": "auto-reply"})
                self.assertEqual(ar.status_code, 200)

                # the result of /v1/messages
                r = await req_task
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertEqual(body["role"], "assistant")
                self.assertEqual(body["content"], [{"type": "text", "text": "auto-reply"}])
                self.assertEqual(body["stop_reason"], "end_turn")

                # reflected in history
                h = await c.get("/_control/history")
                self.assertEqual(h.json()["turn_count"], 1)
                self.assertEqual(len(h.json()["history"]), 1)
        _run(run())

    def test_parallel_requests_multi_pending(self) -> None:
        """multi-pending: accept two concurrent requests and respond to each individually via pending_id."""
        async def run() -> None:
            async with await self._client() as c:
                # put two requests in-flight simultaneously
                t1 = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "a"}],
                }, timeout=10))
                t2 = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "b"}],
                }, timeout=10))

                # wait until both become pending (no 409)
                ids: list[str] = []
                for _ in range(50):
                    p = (await c.get("/_control/pending")).json()
                    if p.get("count") == 2:
                        ids = [item["pending_id"] for item in p["pending"]]
                        break
                    await asyncio.sleep(0.05)
                else:
                    self.fail("two requests never became pending")
                self.assertEqual(len(set(ids)), 2)

                # multiple pending without a pending_id → 400 (pending_id required)
                amb = await c.post("/_control/respond", json={"content": []})
                self.assertEqual(amb.status_code, 400)
                self.assertIn("multiple pending", amb.json()["error"])

                # respond to each individually by pending_id
                # map which request has which id by content
                msg_by_id = {item["pending_id"]: item["request"]["messages"][0]["content"]
                             for item in p["pending"]}
                for pid in ids:
                    text = f"reply-{msg_by_id[pid]}"
                    rr = await c.post("/_control/respond", json={
                        "pending_id": pid, "content": [{"type": "text", "text": text}]})
                    self.assertEqual(rr.status_code, 200)

                r1 = await t1
                r2 = await t2
                self.assertEqual(r1.status_code, 200)
                self.assertEqual(r2.status_code, 200)
                # each request received the response addressed to it (no mix-up)
                self.assertEqual(r1.json()["content"][0]["text"], "reply-a")
                self.assertEqual(r2.json()["content"][0]["text"], "reply-b")
        _run(run())

    def test_clear_resets_state(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # run one round-trip
                t = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "y"}],
                }, timeout=10))
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                await c.post("/_control/auto", json={"text": "ok"})
                r = await t
                self.assertEqual(r.status_code, 200)
                # one history entry
                self.assertEqual((await c.get("/_control/history")).json()["turn_count"], 1)
                # clear
                await c.post("/_control/clear")
                h = (await c.get("/_control/history")).json()
                self.assertEqual(h["turn_count"], 0)
                self.assertEqual(h["history"], [])
        _run(run())

    def test_clear_during_pending_returns_503(self) -> None:
        """When clear is called during a pending request, the main handler returns 503.

        Previously the set_exception exception propagated as-is and became a 500 (fixed in H2).
        """
        async def run() -> None:
            async with await self._client() as c:
                t = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "y"}],
                }, timeout=10))
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                await c.post("/_control/clear")
                r = await t
                self.assertEqual(r.status_code, 503)
                # errors on the Anthropic path return in the official envelope
                self.assertEqual(r.json()["type"], "error")
                self.assertIn("cleared", r.json()["error"]["message"])
                # state is emptied
                p = await c.get("/_control/pending")
                self.assertFalse(p.json()["has_pending"])
                self.assertEqual(p.json()["count"], 0)
        _run(run())

    def test_error_injection(self) -> None:
        """`/_control/error` can return an HTTP error to a pending request."""
        async def run() -> None:
            async with await self._client() as c:
                t = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "y"}],
                }, timeout=10))
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                er = await c.post("/_control/error", json={
                    "status": 429, "type": "rate_limit_error", "message": "throttled",
                })
                self.assertEqual(er.status_code, 200)
                r = await t
                self.assertEqual(r.status_code, 429)
                body = r.json()
                self.assertEqual(body["type"], "error")
                self.assertEqual(body["error"]["type"], "rate_limit_error")
                self.assertEqual(body["error"]["message"], "throttled")
                # also recorded in history
                h = (await c.get("/_control/history")).json()
                self.assertEqual(h["history"][-1]["injected_error"]["status"], 429)
                self.assertIsNone(h["history"][-1]["response_blocks"])
        _run(run())

    def test_error_injection_invalid_status_does_not_hang_pending(self) -> None:
        """Error injection with an invalid status returns 400 and does not break the pending.

        Previously an unhandled ValueError from `int("abc")` caused a 500 with the pending
        unresolved → /v1/messages hung until timeout. By validating before resolving, we
        confirm the pending stays alive after a bad injection and can still be resolved
        normally with a valid injection.
        """
        async def run() -> None:
            async with await self._client() as c:
                t = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "y"}],
                }, timeout=10))
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                # non-numeric status → 400 (the pending is untouched)
                bad = await c.post("/_control/error", json={"status": "abc", "type": "x"})
                self.assertEqual(bad.status_code, 400)
                # out-of-range status → 400
                oor = await c.post("/_control/error", json={"status": 99})
                self.assertEqual(oor.status_code, 400)
                # the pending is still alive → can be resolved with a valid injection
                self.assertTrue((await c.get("/_control/pending")).json()["has_pending"])
                ok = await c.post("/_control/error", json={"status": 500, "type": "api_error"})
                self.assertEqual(ok.status_code, 200)
                r = await t
                self.assertEqual(r.status_code, 500)
        _run(run())

    def test_streaming_round_trip(self) -> None:
        """Parse SSE streaming with httpx. Checks that raw SSE flows even without going through the SDK."""
        async def run() -> None:
            async with await self._client() as c:
                t = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": True,
                    "messages": [{"role": "user", "content": "y"}],
                }, timeout=10))

                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                await c.post("/_control/respond", json={"content": [
                    {"type": "text", "text": "streaming OK"},
                ]})
                r = await t
                self.assertEqual(r.status_code, 200)
                body = r.text
                # check that the key SSE events are present
                self.assertIn("event: message_start", body)
                self.assertIn("event: content_block_delta", body)
                self.assertIn('"text_delta"', body)
                self.assertIn("streaming OK", body)
                self.assertIn("event: message_stop", body)
        _run(run())


class TestWaitForPending(unittest.TestCase):
    """Verification of the long-polling endpoint /_control/wait_for_pending."""

    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app),
            base_url="http://test",
        )

    def test_immediate_return_when_pending_exists(self) -> None:
        """If a pending already exists, return immediately (do not block)."""
        async def run() -> None:
            async with await self._client() as c:
                # fire a request to make it pending
                req_task = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "y"}],
                }, timeout=10))
                # wait until pending is confirmed
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)

                # wait_for_pending: should return immediately
                start = time.time()
                w = await c.get("/_control/wait_for_pending?timeout=10")
                elapsed = time.time() - start
                self.assertLess(elapsed, 0.3, "should return immediately, not block")
                self.assertTrue(w.json()["has_pending"])

                # cleanup
                await c.post("/_control/auto", json={"text": "ok"})
                await req_task
        _run(run())

    def test_blocks_then_wakes_on_new_pending(self) -> None:
        """Wait while no pending exists → wake immediately when a new request arrives."""
        async def run() -> None:
            async with await self._client() as c:
                # submit a waiter in the background
                wait_task = asyncio.create_task(
                    c.get("/_control/wait_for_pending?timeout=10")
                )
                # wait a bit, then submit a new pending
                await asyncio.sleep(0.2)
                req_task = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "wakeup"}],
                }, timeout=10))

                start = time.time()
                w = await wait_task
                elapsed = time.time() - start
                # should wake within 1 second after the new pending arrives
                self.assertLess(elapsed, 1.5, f"waiter took {elapsed:.2f}s to wake")
                body = w.json()
                self.assertTrue(body["has_pending"])
                self.assertEqual(body["request"]["messages"][0]["content"], "wakeup")

                # cleanup
                await c.post("/_control/auto", json={"text": "ok"})
                await req_task
        _run(run())

    def test_timeout(self) -> None:
        """If no pending arrives, returns `{has_pending: False, timeout: True}` on timeout."""
        async def run() -> None:
            async with await self._client() as c:
                start = time.time()
                w = await c.get("/_control/wait_for_pending?timeout=0.5", timeout=5)
                elapsed = time.time() - start
                self.assertGreaterEqual(elapsed, 0.4, "should wait at least timeout")
                self.assertLess(elapsed, 2.0, "should not wait much longer")
                body = w.json()
                self.assertFalse(body["has_pending"])
                self.assertTrue(body.get("timeout"))
        _run(run())

    def test_multiple_waiters_all_woken(self) -> None:
        """Multiple waiters wait concurrently → all wake on a single new pending."""
        async def run() -> None:
            async with await self._client() as c:
                tasks = [
                    asyncio.create_task(c.get("/_control/wait_for_pending?timeout=10"))
                    for _ in range(3)
                ]
                await asyncio.sleep(0.2)
                req_task = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "broadcast"}],
                }, timeout=10))

                results = await asyncio.gather(*tasks)
                for r in results:
                    self.assertTrue(r.json()["has_pending"])
                    self.assertEqual(r.json()["request"]["messages"][0]["content"], "broadcast")

                await c.post("/_control/auto", json={"text": "ok"})
                await req_task
        _run(run())

    def test_timeout_cap(self) -> None:
        """A timeout value exceeding _WAIT_TIMEOUT_MAX is handled safely."""
        # Pass an explicitly large value. Since it does not actually wait, the test does not depend on 0.5.
        async def run() -> None:
            async with await self._client() as c:
                # since an existing pending returns immediately, it is OK even if the cap runs with timeout=1
                # here we only confirm that "passing a huge value does not cause a 4xx error"
                # (actual waiting is already covered by test_timeout)
                req_task = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "y"}],
                }, timeout=10))
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                w = await c.get("/_control/wait_for_pending?timeout=99999")
                self.assertEqual(w.status_code, 200)
                self.assertTrue(w.json()["has_pending"])
                await c.post("/_control/auto", json={"text": "ok"})
                await req_task
        _run(run())


# ── Anthropic SDK compatibility (round-trip with the real SDK) ─────────────────────


class TestAnthropicSDKCompatibility(unittest.TestCase):
    """Start on a separate port with uvicorn and actually run the anthropic SDK."""

    PORT = 18765

    @classmethod
    def setUpClass(cls) -> None:
        import uvicorn
        # fresh module
        from puppetllm import fake_server as fs
        import importlib
        importlib.reload(fs)
        cls.fs = fs

        cls._config = uvicorn.Config(
            fs.app, host="127.0.0.1", port=cls.PORT,
            log_level="critical", loop="asyncio",
        )
        cls._server = uvicorn.Server(cls._config)

        def _run_server() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(cls._server.serve())

        cls._thread = threading.Thread(target=_run_server, daemon=True)
        cls._thread.start()

        import urllib.request
        for _ in range(50):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{cls.PORT}/_control/health", timeout=0.5
                )
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("fake_server did not start")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.should_exit = True
        cls._thread.join(timeout=3)

    def setUp(self) -> None:
        """Clear server state before each test starts.

        Because setUpClass starts uvicorn only once and shares it across 3 tests,
        /_control/history and /_control/turn_count accumulate between tests.
        Clear explicitly to prevent flakiness from adding tests or reordering them.
        """
        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{self.PORT}/_control/clear", method="POST"
            ),
            timeout=2,
        )

    def _respond_when_pending(self, payload: dict[str, Any]) -> threading.Event:
        """Wait for a pending in a separate thread and POST /_control/respond.

        The return value is an Event signaling worker completion. After the SDK call,
        the test body can confirm the worker finished via `assertTrue(event.wait(timeout))`
        (detects hangs from silent exit; addresses PR #94 review #3).
        """
        return self._post_when_pending("/_control/respond", payload)

    def _respond_when_pending_error(self, payload: dict[str, Any]) -> threading.Event:
        """Wait for a pending in a separate thread and POST /_control/error."""
        return self._post_when_pending("/_control/error", payload)

    def _post_when_pending(
        self, endpoint: str, payload: dict[str, Any]
    ) -> threading.Event:
        """Spin up a polling worker and POST to endpoint once a pending is found.

        However the worker ends (success / poll exhaustion / exception), it always sets the Event.
        On poll exhaustion it **injects a fallback error to /_control/error** so the SDK
        side does not hang (a safety net to avoid turning a test hang into a mysterious
        timeout on CI).
        """
        import urllib.request
        done = threading.Event()

        def worker() -> None:
            try:
                for _ in range(100):  # 100 * 0.05s = up to 5 seconds
                    try:
                        p = json.loads(
                            urllib.request.urlopen(
                                f"http://127.0.0.1:{self.PORT}/_control/pending",
                                timeout=0.5,
                            ).read()
                        )
                        if p.get("has_pending"):
                            req = urllib.request.Request(
                                f"http://127.0.0.1:{self.PORT}{endpoint}",
                                data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            urllib.request.urlopen(req, timeout=2)
                            return
                    except Exception:
                        # ignore transient errors during polling and retry on the next sleep.
                        pass
                    time.sleep(0.05)
                # exhaust: inject a fallback error so the SDK does not hang with a pending.
                # After done.wait(), the test can confirm something arrived via the endpoint,
                # detecting the "worker ran but never found a pending" case.
                try:
                    fallback = urllib.request.Request(
                        f"http://127.0.0.1:{self.PORT}/_control/error",
                        data=json.dumps({
                            "status": 500, "type": "api_error",
                            "message": "test responder timed out",
                        }).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(fallback, timeout=2)
                except Exception:
                    # the fallback error POST is a best-effort safety net (we don't care if it
                    # becomes a 4xx/connect error due to no pending / server shutdown, etc.).
                    # finally's done.set() guarantees hang prevention, so sinking here is OK.
                    pass
            finally:
                done.set()

        threading.Thread(target=worker, daemon=True).start()
        return done

    def test_text_streaming(self) -> None:
        import anthropic
        responder_done: list[threading.Event] = []

        async def run() -> tuple[list[str], list[Any]]:
            client = anthropic.AsyncAnthropic(
                api_key="sk-mock", base_url=f"http://127.0.0.1:{self.PORT}",
            )
            responder_done.append(self._respond_when_pending({"content": [
                {"type": "text", "text": "round-trip OK"},
            ]}))
            deltas: list[str] = []
            async with client.messages.stream(
                model="claude-test", max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
            ) as s:
                async for ev in s:
                    if ev.type == "text":
                        deltas.append(ev.text)
                final = await s.get_final_message()
                return deltas, final.content

        deltas, content = _run(run())
        self.assertTrue(responder_done[0].wait(10), "responder thread did not finish")
        self.assertEqual("".join(deltas), "round-trip OK")
        self.assertEqual(content[0].type, "text")
        self.assertEqual(content[0].text, "round-trip OK")

    def test_sdk_raises_on_error_injection(self) -> None:
        """When going through `/_control/error`, the anthropic SDK raises the appropriate exception."""
        import anthropic
        responder_done: list[threading.Event] = []

        async def run() -> Exception | None:
            # max_retries=0 disables the SDK's built-in retry (observe the immediate raise)
            client = anthropic.AsyncAnthropic(
                api_key="sk-mock",
                base_url=f"http://127.0.0.1:{self.PORT}",
                max_retries=0,
            )
            responder_done.append(self._respond_when_pending_error({
                "status": 429, "type": "rate_limit_error", "message": "throttled",
            }))
            try:
                async with client.messages.stream(
                    model="claude-test", max_tokens=10,
                    messages=[{"role": "user", "content": "x"}],
                ) as s:
                    async for _ in s:
                        pass
                    await s.get_final_message()
            except Exception as e:
                return e
            return None

        exc = _run(run())
        self.assertTrue(responder_done[0].wait(10), "responder thread did not finish")
        self.assertIsNotNone(exc, "expected anthropic SDK to raise on 429")
        # check the SDK's exception type
        self.assertIsInstance(exc, anthropic.RateLimitError)

    def test_tool_use_streaming(self) -> None:
        import anthropic
        responder_done: list[threading.Event] = []

        async def run() -> Any:
            client = anthropic.AsyncAnthropic(
                api_key="sk-mock", base_url=f"http://127.0.0.1:{self.PORT}",
            )
            responder_done.append(self._respond_when_pending({"content": [
                {"type": "text", "text": "let me run"},
                {"type": "tool_use", "id": "tu_x", "name": "Bash",
                 "input": {"command": "ls"}},
            ]}))
            async with client.messages.stream(
                model="claude-test", max_tokens=100,
                tools=[{"name": "Bash", "description": "x",
                        "input_schema": {"type": "object"}}],
                messages=[{"role": "user", "content": "do something"}],
            ) as s:
                async for _ in s:
                    pass
                return await s.get_final_message()

        final = _run(run())
        self.assertTrue(responder_done[0].wait(10), "responder thread did not finish")
        self.assertEqual(final.stop_reason, "tool_use")
        blocks = {b.type: b for b in final.content}
        self.assertIn("text", blocks)
        self.assertIn("tool_use", blocks)
        self.assertEqual(blocks["tool_use"].name, "Bash")
        self.assertEqual(blocks["tool_use"].input, {"command": "ls"})


if __name__ == "__main__":
    unittest.main()
