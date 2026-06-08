"""Unit tests for puppetllm.fake_server.

純粋ロジック (SSE 構築 / レスポンス shape) は同期的に検証。
HTTP レイヤは httpx.AsyncClient + ASGITransport で async 制御し、
最後に anthropic SDK との往復で SDK 互換性を確認する。

実行 (repo root から、または tools/ 配下から):
  python3 -m unittest puppetllm.tests.test_fake_server -v

Docker 経由:
  docker compose -f puppetllm/docker-compose.yml --profile test run --rm proxy-test
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
from typing import Any


def _import_fresh():
    """fake_server module をリロードしてサーバ状態をクリーンに保つ。"""
    import importlib
    from puppetllm import fake_server as fs
    importlib.reload(fs)
    return fs


# ── 純粋ロジック (HTTP 不要) ────────────────────────────────────────


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
        # text-only の stop_reason は end_turn
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
        # input は partial_json に JSON 文字列として埋め込まれる (escaped quotes)
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
        # `event: content_block_start` ライン (data 内の同名文字列を除外) が 2 回
        self.assertEqual(joined.count("event: content_block_start"), 2)
        # 混在時の stop_reason は tool_use 優先
        self.assertIn('"tool_use"', joined.split("message_delta")[-1])

    def test_empty_text(self) -> None:
        """空 text でもイベントは出る (start/delta/stop の構造)。"""
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


# ── HTTP レイヤ (httpx ASGITransport で async 制御) ────────────────


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestAsyncHttpx(unittest.TestCase):
    """httpx + ASGITransport で fake_server の HTTP 経路を非同期検証。

    sync TestClient を threading で起動する形は async lock との相性が悪く
    flaky になるため、純粋 async 経路で書く。
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
        """壊れた JSON body は 500 でなく明確な 400 を返す。

        responder が curl で render_chart の複雑ネスト input を送る際の
        shell エスケープ崩れで malformed JSON になっても、opaque な 500 にせず
        原因の分かる 400 にする回帰テスト。
        """
        async def run() -> None:
            async with await self._client() as c:
                for path in ("/_control/respond", "/_control/auto",
                             "/_control/error", "/v1/messages"):
                    r = await c.post(
                        path, content=b"{bad json,,}",
                        headers={"Content-Type": "application/json"},
                    )
                    self.assertEqual(r.status_code, 400, f"{path} should 400 on bad JSON")
                    self.assertIn("invalid JSON body", r.json().get("error", ""))
                # JSON だが object でない (配列) も 400
                r = await c.post("/_control/respond", json=[1, 2, 3])
                self.assertEqual(r.status_code, 400)
        _run(run())

    def test_non_stream_round_trip(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # /v1/messages を background task で投入
                req_task = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "claude-test", "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                }, timeout=10))

                # pending 出るまで待つ
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

                # /v1/messages の結果
                r = await req_task
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertEqual(body["role"], "assistant")
                self.assertEqual(body["content"], [{"type": "text", "text": "auto-reply"}])
                self.assertEqual(body["stop_reason"], "end_turn")

                # history 反映
                h = await c.get("/_control/history")
                self.assertEqual(h.json()["turn_count"], 1)
                self.assertEqual(len(h.json()["history"]), 1)
        _run(run())

    def test_parallel_requests_multi_pending(self) -> None:
        """multi-pending: 同時 2 リクエストを両方受け付け、pending_id 指定で個別応答できる。"""
        async def run() -> None:
            async with await self._client() as c:
                # 2 件を同時 in-flight に
                t1 = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "a"}],
                }, timeout=10))
                t2 = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "b"}],
                }, timeout=10))

                # 両方 pending になるまで待つ (409 にならない)
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

                # pending_id 指定なしで複数 pending → 400 (要 pending_id)
                amb = await c.post("/_control/respond", json={"content": []})
                self.assertEqual(amb.status_code, 400)
                self.assertIn("multiple pending", amb.json()["error"])

                # それぞれ pending_id 指定で個別応答
                # どちらの request がどの id かを content で対応付け
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
                # 各リクエストが「自分宛て」の応答を受け取った (取り違えなし)
                self.assertEqual(r1.json()["content"][0]["text"], "reply-a")
                self.assertEqual(r2.json()["content"][0]["text"], "reply-b")
        _run(run())

    def test_clear_resets_state(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # 1 件 round-trip 走らせる
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
                # history 1 件
                self.assertEqual((await c.get("/_control/history")).json()["turn_count"], 1)
                # clear
                await c.post("/_control/clear")
                h = (await c.get("/_control/history")).json()
                self.assertEqual(h["turn_count"], 0)
                self.assertEqual(h["history"], [])
        _run(run())

    def test_clear_during_pending_returns_503(self) -> None:
        """clear が pending リクエスト中に呼ばれた場合、main handler は 503 を返す。

        以前は set_exception の例外がそのまま伝播して 500 になっていた (H2 で修正)。
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
                self.assertIn("cleared", r.json()["error"])
                # state は空に
                p = await c.get("/_control/pending")
                self.assertFalse(p.json()["has_pending"])
                self.assertEqual(p.json()["count"], 0)
        _run(run())

    def test_error_injection(self) -> None:
        """`/_control/error` で HTTP error を pending リクエストに返せる。"""
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
                # history にも記録されている
                h = (await c.get("/_control/history")).json()
                self.assertEqual(h["history"][-1]["injected_error"]["status"], 429)
                self.assertIsNone(h["history"][-1]["response_blocks"])
        _run(run())

    def test_error_injection_invalid_status_does_not_hang_pending(self) -> None:
        """不正な status の error 注入は 400 を返し、pending を壊さない。

        以前は `int("abc")` の未処理 ValueError で 500 + pending 未解決 →
        /v1/messages がタイムアウトまでハングした。検証を resolve 前に行うことで、
        不正注入後も pending は生き、正しい注入で正常に解決できることを確認。
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
                # 非数値 status → 400 (pending は触られない)
                bad = await c.post("/_control/error", json={"status": "abc", "type": "x"})
                self.assertEqual(bad.status_code, 400)
                # 範囲外 status → 400
                oor = await c.post("/_control/error", json={"status": 99})
                self.assertEqual(oor.status_code, 400)
                # pending はまだ生きている → 正しい注入で解決できる
                self.assertTrue((await c.get("/_control/pending")).json()["has_pending"])
                ok = await c.post("/_control/error", json={"status": 500, "type": "api_error"})
                self.assertEqual(ok.status_code, 200)
                r = await t
                self.assertEqual(r.status_code, 500)
        _run(run())

    def test_streaming_round_trip(self) -> None:
        """SSE streaming を httpx でパース。SDK 経由でなくとも raw SSE が flowing するか。"""
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
                # 主要 SSE イベントが含まれているか
                self.assertIn("event: message_start", body)
                self.assertIn("event: content_block_delta", body)
                self.assertIn('"text_delta"', body)
                self.assertIn("streaming OK", body)
                self.assertIn("event: message_stop", body)
        _run(run())


class TestWaitForPending(unittest.TestCase):
    """Long-polling endpoint /_control/wait_for_pending の検証。"""

    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app),
            base_url="http://test",
        )

    def test_immediate_return_when_pending_exists(self) -> None:
        """既に pending があれば即返却 (block しない)。"""
        async def run() -> None:
            async with await self._client() as c:
                # request を仕掛けて pending 化
                req_task = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "y"}],
                }, timeout=10))
                # pending 確定まで待つ
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)

                # wait_for_pending: 即返却するはず
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
        """pending なし状態で wait → 新 request 到着で即起きる。"""
        async def run() -> None:
            async with await self._client() as c:
                # waiter を background で投入
                wait_task = asyncio.create_task(
                    c.get("/_control/wait_for_pending?timeout=10")
                )
                # 少し待ってから新 pending を投入
                await asyncio.sleep(0.2)
                req_task = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "x", "stream": False,
                    "messages": [{"role": "user", "content": "wakeup"}],
                }, timeout=10))

                start = time.time()
                w = await wait_task
                elapsed = time.time() - start
                # 新 pending 来てから 1 秒以内に起きるはず
                self.assertLess(elapsed, 1.5, f"waiter took {elapsed:.2f}s to wake")
                body = w.json()
                self.assertTrue(body["has_pending"])
                self.assertEqual(body["request"]["messages"][0]["content"], "wakeup")

                # cleanup
                await c.post("/_control/auto", json={"text": "ok"})
                await req_task
        _run(run())

    def test_timeout(self) -> None:
        """pending が来ないと timeout で `{has_pending: False, timeout: True}` 返却。"""
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
        """複数 waiter が同時待機 → 1 つの新 pending で全員起きる。"""
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
        """timeout 値が _WAIT_TIMEOUT_MAX を超えても安全に処理される。"""
        # 明示的に大きな値を渡す。実際 wait はしないので 0.5 に依存する形でテスト。
        async def run() -> None:
            async with await self._client() as c:
                # 既に pending あれば即返却なので、cap が timeout=1 設定で実走しても OK
                # ここでは「巨大値を渡しても 4xx エラーにならない」ことだけ確認
                # (実 wait は test_timeout でカバー済み)
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


# ── Anthropic SDK 互換性 (実際の SDK で往復) ─────────────────────


class TestAnthropicSDKCompatibility(unittest.TestCase):
    """uvicorn で別ポート起動し、anthropic SDK を実際に走らせる。"""

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
        """各テスト開始前に server state を clear する。

        setUpClass で uvicorn を 1 回だけ起動し 3 テストで共有しているため、
        各テスト間で /_control/history と /_control/turn_count が累積する。
        テスト追加 / 順序変更で flaky 化するのを防ぐため明示的に clear。
        """
        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{self.PORT}/_control/clear", method="POST"
            ),
            timeout=2,
        )

    def _respond_when_pending(self, payload: dict[str, Any]) -> threading.Event:
        """別スレッドで pending を待って /_control/respond を投げる。

        戻り値は worker 完了通知の Event。テスト本体は SDK 呼び出し後に
        `assertTrue(event.wait(timeout))` で worker の完了を確認できる
        (silently exit による hang を検出可能、PR #94 review #3 対応)。
        """
        return self._post_when_pending("/_control/respond", payload)

    def _respond_when_pending_error(self, payload: dict[str, Any]) -> threading.Event:
        """別スレッドで pending を待って /_control/error を投げる。"""
        return self._post_when_pending("/_control/error", payload)

    def _post_when_pending(
        self, endpoint: str, payload: dict[str, Any]
    ) -> threading.Event:
        """polling worker を立ち上げ、pending を見つけたら endpoint に POST する。

        Worker はどう終わっても (成功 / poll exhaust / 例外) 必ず Event を set する。
        Poll exhaust 時は **fallback error を /_control/error に投げる** ことで
        SDK 側を hang させない (テストハングを CI 上のミステリアスな timeout に
        させないためのセーフティネット)。
        """
        import urllib.request
        done = threading.Event()

        def worker() -> None:
            try:
                for _ in range(100):  # 100 * 0.05s = 最大 5 秒
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
                        # poll 中の transient なエラーは無視して次の sleep で retry。
                        pass
                    time.sleep(0.05)
                # exhaust: SDK が pending のまま hang しないよう fallback error を投げる。
                # テストは done.wait() 後に endpoint 経由で何かが届いたことを確認することで
                # 「worker は走ったが pending を見つけられなかった」を検出可能。
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
                    # fallback error POST は best-effort セーフティネット (no pending /
                    # server shutdown 等で 4xx/connect error になっても気にしない)。
                    # finally の done.set() が hang 防止を保証するのでここは sink で OK。
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
        """`/_control/error` を経由した時、anthropic SDK は適切な例外を raise する。"""
        import anthropic
        responder_done: list[threading.Event] = []

        async def run() -> Exception | None:
            # max_retries=0 で SDK 内蔵 retry を無効化 (即 raise を観察)
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
        # SDK の例外型を判定
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
