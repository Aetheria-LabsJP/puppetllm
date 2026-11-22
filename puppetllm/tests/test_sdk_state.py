"""SDK state/sidecar tests: standard-library status and mock HTTP only."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from puppetllm.sdk_guard import SDKGuard
from puppetllm.sdk_runtime import RuntimeSnapshot, StateLock
from puppetllm.sdk_state import build_parser, create_app, main, read_status, watch_once


def snapshot(*, heartbeat=None, ready=True, requests=None):
    return {"version": 1, "heartbeat_at": time.time() if heartbeat is None else heartbeat,
            "ready": ready, "reason": "running" if ready else "stopping",
            "requests": requests or [], "guard": {"blocked": 1, "in_progress": 0}}


class StateDirectory:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def write(self, value):
        (self.path / "status.json").write_text(json.dumps(value), encoding="utf-8")


class TestStatus(StateDirectory, unittest.TestCase):
    def test_hidden_thinking_and_configured_limits_are_visible_without_private_fields(self):
        runtime = RuntimeSnapshot(self.path, log=lambda _line: None)
        runtime.start("pending-safe", "a" * 64, "claude-opus-5-5",
            generation_settings={"effort": "low", "max_output_tokens": 4096, "private": "PRIVATE_SETTING"})
        record = runtime.requests["pending-safe"]
        record.update(stage="thinking", estimated_thinking_tokens=31, thinking_token_updates=3,
            last_thinking_progress_at=100, last_stream_event_type="content_block_delta",
            stream_event_counts={"ping": 2, "content_block_delta": 3, "PRIVATE_EVENT": 1},
            stream_delta_counts={"thinking_delta": 3, "PRIVATE_DELTA": 2})
        runtime.write([])
        result = read_status(self.path)
        self.assertTrue(result["ready"])
        progress = result["requests"][0]
        self.assertEqual(progress["stage"], "thinking")
        self.assertEqual(progress["estimated_thinking_tokens"], 31)
        self.assertEqual(progress["generated_characters"], 0)
        self.assertEqual(progress["configured_effort"], "low")
        self.assertEqual(progress["configured_max_output_tokens"], 4096)
        self.assertEqual(progress["stream_delta_counts"], {"thinking_delta": 3})
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_readiness_requires_a_fresh_ready_worker(self):
        for heartbeat, ready, expected in ((100, True, True), (70, True, True),
                                            (69, True, False), (101, True, False),
                                            (100, False, False)):
            with self.subTest(heartbeat=heartbeat, ready=ready):
                self.write(snapshot(heartbeat=heartbeat, ready=ready))
                result = read_status(self.path, now=100)
                self.assertEqual(result["ready"], expected)
                self.assertEqual(result["fresh"], 0 <= 100 - heartbeat <= 30)

    def test_runtime_snapshot_preserves_numeric_usage_and_unknown_values(self):
        runtime = RuntimeSnapshot(self.path, log=lambda _line: None)
        runtime.start("pending-safe", "a" * 64, "claude-opus-5-5")
        record = runtime.requests["pending-safe"]
        record.update(sdk_estimated_cost_usd=.25, sdk_reported_num_turns=2,
                      sdk_reported_usage={"input_tokens": 100, "output_tokens": 50,
                                          "PRIVATE_BODY": "PRIVATE_TOOL_ARGUMENT"},
                      sdk_reported_model_usage=[{"costUSD": .25, "outputTokens": 50}],
                      observed_message_usage={"output_tokens": 50},
                      last_api_retry={"attempt": 1, "error_status": 429},
                      last_generation_kind="thinking_delta", raw_text="PRIVATE_THINKING")
        runtime.write([])
        result = read_status(self.path)
        self.assertTrue(result["ready"])
        progress = result["requests"][0]
        self.assertIsNone(progress["api_attempt_count"])
        self.assertIsNone(progress["structured_output_retries"])
        self.assertEqual(progress["sdk_estimated_cost_usd"], .25)
        self.assertEqual(progress["sdk_reported_model_usage"], [{"costUSD": .25, "outputTokens": 50}])
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_missing_invalid_or_private_content_fails_closed(self):
        self.assertEqual(read_status(self.path)["reason"], "status_missing")
        values = [[], {}, {**snapshot(), "version": 2}, {**snapshot(), "ready": 1},
                  {**snapshot(), "reason": "PRIVATE_EXCEPTION"},
                  {**snapshot(), "heartbeat_at": "PRIVATE_AUTH"},
                  {**snapshot(), "guard": {"blocked": True, "in_progress": 0}},
                  {**snapshot(), "requests": [{"text": "PRIVATE_BODY"}]},
                  {**snapshot(), "requests": [None]},
                  {**snapshot(), "ready": True, "reason": []}]
        for value in values:
            with self.subTest(value=type(value).__name__):
                self.write(value)
                result = read_status(self.path)
                self.assertFalse(result["ready"])
                self.assertEqual(result["requests"], [])
                self.assertNotIn("PRIVATE", json.dumps(result))
        for encoded in ('{"version": 1, "private": "PRIVATE_BODY"',
                        '{"version": NaN}', '{"version": 1, "version": 1}', '"' + 'X' * 1048576 + '"'):
            (self.path / "status.json").write_text(encoded)
            self.assertFalse(read_status(self.path)["ready"])

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "requires non-following file open")
    def test_symlinks_are_not_followed(self):
        target = self.path / "private.json"
        target.write_text(json.dumps(snapshot()))
        (self.path / "status.json").symlink_to(target)
        self.assertFalse(read_status(self.path)["ready"])

    def test_status_uses_only_standard_library_without_site_packages(self):
        self.write(snapshot())
        result = subprocess.run([sys.executable, "-S", "-B", "-m", "puppetllm.sdk_state", "status",
                                 "--state-dir", str(self.path)], capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(result.stdout)["ready"])
        self.assertEqual(result.stderr, "")

    def test_common_cli_arguments_work_before_or_after_subcommand(self):
        for argv in (("--state-dir", "chosen", "status"), ("status", "--state-dir", "chosen")):
            self.assertEqual(build_parser().parse_args(argv).state_dir, "chosen")
        args = build_parser().parse_args(["watch"])
        self.assertEqual((args.host, args.port), ("127.0.0.1", 8766))
        self.assertEqual(args.stale_after, 30)


class TestMaintenance(StateDirectory, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.guard = SDKGuard(self.path / "guard.sqlite3")
        self.addCleanup(self.guard.close)
        self.active_key = "a" * 64
        self.blocked_key = "b" * 64
        self.guard.begin(self.active_key)
        self.guard.fail(self.guard.begin(self.blocked_key), "timeout")

    def invoke(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main([*args, "--state-dir", str(self.path)])
        return code, output.getvalue(), error.getvalue()

    def test_all_maintenance_is_refused_while_a_worker_holds_the_lock(self):
        before = self.guard.status()
        with StateLock(self.path):
            for command in (("blocks",), ("clear", "--all"), ("recover", "--all")):
                with self.subTest(command=command):
                    code, output, error = self.invoke(*command)
                    self.assertEqual(code, 1)
                    self.assertEqual(output, "")
                    self.assertEqual(json.loads(error)["error"], "state_busy")
        self.assertEqual(self.guard.status(), before)

    def test_recovery_blocks_active_entries_until_explicit_clear(self):
        code, output, error = self.invoke("clear", "--all")
        self.assertEqual((code, json.loads(output), error), (0, {"cleared": 1}, ""))
        self.assertEqual(self.guard.status()[0]["state"], "active")
        code, output, error = self.invoke("recover", "--all")
        self.assertEqual((code, json.loads(output), error), (0, {"recovered": 1}, ""))
        self.assertEqual(self.guard.status()[0]["reason"], "abnormal_exit")
        code, output, error = self.invoke("blocks")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["blocked"], 1)
        self.assertEqual(json.loads(output)["in_progress"], 0)
        code, output, error = self.invoke("clear", "--key", self.active_key)
        self.assertEqual((code, json.loads(output), error), (0, {"cleared": 1}, ""))
        self.assertEqual(self.guard.status(), [])

    def test_explicit_selection_and_invalid_key_do_not_expose_input(self):
        for argv in (["clear"], ["recover"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(argv)
        code, output, error = self.invoke("clear", "--key", "PRIVATE_BODY")
        self.assertEqual(code, 1)
        self.assertNotIn("PRIVATE", error + output)
        self.assertEqual(len(self.guard.status()), 2)

    def test_tilde_state_directory_is_expanded(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"HOME": home}), \
                tempfile.TemporaryDirectory(dir=home) as directory:
            path = Path(directory) / "state"
            with SDKGuard(path / "guard.sqlite3") as guard:
                guard.fail(guard.begin("c" * 64), "timeout")
                short = "~/" + Path(directory).name + "/state"
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(["blocks", "--state-dir", short])
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(output.getvalue())["blocked"], 1)


class TestSidecar(StateDirectory, unittest.IsolatedAsyncioTestCase):
    async def test_health_and_readiness_are_independent_and_refresh_each_read(self):
        app = create_app(self.path, "http://mock-proxy")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://sidecar") as client:
            self.assertEqual((await client.get("/health")).status_code, 200)
            self.assertEqual((await client.get("/ready")).status_code, 503)
            self.write(snapshot())
            self.assertEqual((await client.get("/ready")).status_code, 200)
            self.write(snapshot(heartbeat=time.time() - 31))
            status = await client.get("/status")
            self.assertEqual(status.status_code, 200)
            self.assertFalse(status.json()["fresh"])
            self.assertEqual((await client.get("/ready")).status_code, 503)
            self.assertEqual((await client.get("/health")).status_code, 200)

    async def test_a_ready_worker_does_not_even_list_pending_requests(self):
        self.write(snapshot())
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(500)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock-proxy") as client:
            result = await watch_once(self.path, client)
        self.assertEqual(calls, [])
        self.assertTrue(result["ready"])

    async def test_stale_or_unready_worker_rejects_every_summary_id_without_retry(self):
        for ready, heartbeat in ((True, time.time() - 31), (False, time.time())):
            with self.subTest(ready=ready):
                self.write(snapshot(ready=ready, heartbeat=heartbeat))
                calls = []
                def handler(request):
                    calls.append(request)
                    if request.method == "GET":
                        self.assertEqual(request.url.path, "/_control/pending")
                        self.assertEqual(dict(request.url.params), {"fields": "pending_id"})
                        return httpx.Response(200, json={"pending": [
                            {"pending_id": "first", "request": {"text": "PRIVATE_BODY"}},
                            {"pending_id": "second"}, {"pending_id": "first"}]})
                    self.assertEqual(request.url.path, "/_control/error")
                    payload = json.loads(request.content)
                    self.assertEqual(payload["status"], 503)
                    self.assertEqual(payload["headers"], {"x-should-retry": "false"})
                    self.assertNotIn("PRIVATE", request.content.decode())
                    return httpx.Response(200, json={"ok": True})
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock-proxy") as client:
                    result = await watch_once(self.path, client)
                self.assertEqual(result["injected"], 2)
                self.assertEqual(len(calls), 3)
                self.assertNotIn("PRIVATE", json.dumps(result))

    async def test_worker_recovery_before_injection_stops_queue_drain(self):
        self.write(snapshot(ready=False))
        calls = []
        def handler(request):
            calls.append(request)
            self.write(snapshot())
            return httpx.Response(200, json={"pending": [{"pending_id": "first"}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock-proxy") as client:
            result = await watch_once(self.path, client)
        self.assertEqual(result["injected"], 0)
        self.assertEqual(len(calls), 1)

    async def test_unavailable_proxy_and_disconnected_pending_are_bounded_and_safe(self):
        def unavailable(request):
            raise httpx.ConnectError("PRIVATE_AUTH", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable), base_url="http://mock-proxy") as client:
            result = await watch_once(self.path, client)
        self.assertEqual(result["reason"], "proxy_unavailable")
        self.assertNotIn("PRIVATE", json.dumps(result))
        injected = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"pending": [{"pending_id": "gone"}, {"pending_id": "waiting"}]})
            injected.append(json.loads(request.content)["pending_id"])
            return httpx.Response(404 if injected[-1] == "gone" else 200)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock-proxy") as client:
            result = await watch_once(self.path, client)
        self.assertEqual(injected, ["gone", "waiting"])
        self.assertEqual((result["failed"], result["injected"]), (1, 1))

    async def test_background_watchdog_runs_without_a_network_service_and_stops_cleanly(self):
        injected = asyncio.Event()
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"pending": [{"pending_id": "waiting"}]})
            injected.set()
            return httpx.Response(200)
        app = create_app(self.path, "http://mock-proxy", interval=.01, transport=httpx.MockTransport(handler))
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(injected.wait(), 2)
            task = app.state.watchdog_task
            self.assertFalse(task.done())
        self.assertTrue(task.done())
        self.assertFalse((self.path / "worker.lock").exists())
        self.assertFalse((self.path / "guard.sqlite3").exists())
        self.assertFalse((self.path / "status.json").exists())


if __name__ == "__main__":
    unittest.main()
