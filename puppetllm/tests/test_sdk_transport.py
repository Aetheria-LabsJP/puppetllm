"""Framing tests for the optional SDK transport workarounds."""

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from puppetllm.testing import serve
from puppetllm.tests.test_agent_sdk import private_home


HAS_SDK = importlib.util.find_spec("claude_agent_sdk") is not None


async def _empty_stream():
    return
    yield {}


@unittest.skipUnless(HAS_SDK, "install .[agent-sdk]")
class TestByteLimit(unittest.IsolatedAsyncioTestCase):
    async def consume(self, chunks, limit):
        import anyio
        from puppetllm.sdk_transport import _ByteLimitedTextStream

        send, receive = anyio.create_memory_object_stream(len(chunks))
        async with send, receive:
            for chunk in chunks:
                await send.send(chunk)
            await send.aclose()
            async with _ByteLimitedTextStream(receive, limit) as limited:
                return [chunk async for chunk in limited]

    async def test_counts_utf8_bytes_across_chunk_boundaries(self):
        from claude_agent_sdk import CLIJSONDecodeError

        for chunks, limit in ((["日", "本", "語"], 8), (["😀"], 3), (["abc", "def"], 5), (["ok\n日", "本語"], 8)):
            with self.subTest(chunks=chunks), self.assertRaises(CLIJSONDecodeError) as caught:
                await self.consume(chunks, limit)
            self.assertIn(f"exceeds limit {limit}", str(caught.exception.original_error))
            self.assertNotIn("日本語", str(caught.exception))

    async def test_exact_limit_and_newlines_reset_each_line(self):
        for chunks, limit in ((["日", "本\n語語\n", "ab\ncdef\n"], 6),
                              (["abc\n", "def\n"], 3), (["日", "本"], 6)):
            with self.subTest(chunks=chunks):
                self.assertEqual(await self.consume(chunks, limit), chunks)


@unittest.skipUnless(HAS_SDK, "install .[agent-sdk]")
class TestCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_waits_for_cleanup_then_propagates(self):
        from puppetllm.sdk_transport import await_cleanup

        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def cleanup():
            started.set()
            await release.wait()
            finished.set()
        async def caller():
            await await_cleanup(cleanup())
        task = asyncio.create_task(caller())
        await started.wait()
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertFalse(finished.is_set())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished.is_set())

    async def test_accepts_task_future_and_gather_without_losing_results(self):
        from puppetllm.sdk_transport import await_cleanup

        self.assertEqual(await await_cleanup(asyncio.create_task(asyncio.sleep(0, result=1))), 1)
        future = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(future.set_result, 2)
        self.assertEqual(await await_cleanup(future), 2)
        self.assertEqual(await await_cleanup(asyncio.gather(asyncio.sleep(0, result=3))), [3])

    async def test_cleanup_failure_is_observed_even_when_caller_was_cancelled(self):
        from puppetllm.sdk_transport import await_cleanup

        started, release = asyncio.Event(), asyncio.Event()
        async def cleanup():
            started.set()
            await release.wait()
            raise RuntimeError("cleanup failed")
        task = asyncio.create_task(await_cleanup(cleanup()))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            await task

    async def test_cancelled_cleanup_future_propagates_without_spinning(self):
        from puppetllm.sdk_transport import await_cleanup

        future = asyncio.get_running_loop().create_future()
        future.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await await_cleanup(future)

    async def test_anyio_deadline_does_not_interrupt_cleanup_or_busy_spin(self):
        import anyio
        from puppetllm.sdk_transport import await_cleanup

        finished = asyncio.Event()
        async def cleanup():
            await asyncio.sleep(.02)
            finished.set()
        with anyio.move_on_after(.001) as scope:
            await await_cleanup(cleanup())
            await anyio.lowlevel.checkpoint()
        self.assertTrue(scope.cancelled_caught)
        self.assertTrue(finished.is_set())

    async def test_concurrent_close_owns_one_cleanup_and_defers_raw_cancels(self):
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
        from puppetllm.sdk_transport import RelaySDKTransport

        started, release = asyncio.Event(), asyncio.Event()
        calls = {"process": 0, "pipes": 0}
        process = SimpleNamespace(returncode=None)
        async def close_pipes():
            calls["pipes"] += 1
            await asyncio.sleep(0)
        process.aclose = close_pipes
        transport = RelaySDKTransport(_empty_stream(), ClaudeAgentOptions(cli_path="unused"))
        transport._process = process
        async def native_close(instance):
            calls["process"] += 1
            started.set()
            await release.wait()
            process.returncode = 0
            instance._process = None
        with patch.object(SubprocessCLITransport, "close", native_close):
            owner = asyncio.create_task(transport.close())
            await started.wait()
            concurrent = asyncio.create_task(transport.close())
            for _ in range(3):
                owner.cancel()
                await asyncio.sleep(0)
                self.assertFalse(owner.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await owner
            await concurrent
            await transport.close()
        self.assertEqual(calls, {"process": 1, "pipes": 1})

    async def test_an_unreaped_process_is_not_reported_as_successful_cleanup(self):
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
        from puppetllm.sdk_transport import RelaySDKTransport

        transport = RelaySDKTransport(_empty_stream(), ClaudeAgentOptions(cli_path="unused"))
        transport._process = SimpleNamespace(returncode=None)
        async def native_close(_instance):
            return
        with patch.object(SubprocessCLITransport, "close", native_close):
            with self.assertRaisesRegex(RuntimeError, "did not exit during bounded cleanup"):
                await transport.close()


@unittest.skipUnless(HAS_SDK and os.environ.get("PUPPETLLM_TEST_REAL_AGENT_SDK") == "1",
                     "set PUPPETLLM_TEST_REAL_AGENT_SDK=1 for local CLI integration")
class TestNativeCleanup(unittest.IsolatedAsyncioTestCase):
    """Exercise real SDK/CLI against a localhost mock, without paid inference."""

    def setUp(self):
        private_home(self)

    async def wait_until(self, predicate):
        async with asyncio.timeout(15):
            while not predicate():
                await asyncio.sleep(.01)

    def options(self, server, cwd):
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            model="claude-opus-5-5", tools=[], setting_sources=[], mcp_servers={},
            strict_mcp_config=True, permission_mode="dontAsk", cwd=cwd, max_turns=1,
            stderr=lambda _line: None, settings='{"disableAllHooks":true}',
            env={"ANTHROPIC_BASE_URL": server.url, "ANTHROPIC_API_KEY": "sk-local-test-only",
                 "ANTHROPIC_AUTH_TOKEN": "", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                 "CLAUDE_CODE_USE_BEDROCK": "0", "CLAUDE_CODE_USE_VERTEX": "0",
                 "CLAUDE_CODE_USE_FOUNDRY": "0", "CLAUDE_CODE_USE_ANTHROPIC_AWS": "0",
                 "CLAUDE_CODE_MAX_RETRIES": "0"},
            extra_args={"no-session-persistence": None, "disable-slash-commands": None})

    async def query(self, client):
        async def prompt():
            yield {"type": "user", "message": {"role": "user", "content": "local cleanup fixture"}}
        await client.query(prompt())

    def assert_reaped(self, process):
        from claude_agent_sdk._internal.transport.subprocess_cli import _ACTIVE_CHILDREN

        self.assertIsNotNone(process.returncode)
        self.assertNotIn(process, _ACTIVE_CHILDREN)
        with self.assertRaises(ProcessLookupError):
            os.kill(process.pid, 0)
        stdout = process._transport.get_pipe_transport(1)
        self.assertTrue(stdout is None or stdout.is_closing())

    async def _normal_close(self, mode):
        from claude_agent_sdk import ClaudeSDKClient
        from puppetllm.sdk_transport import RelaySDKTransport, await_cleanup

        with serve() as server, tempfile.TemporaryDirectory() as cwd:
            options = self.options(server, cwd)
            transport = RelaySDKTransport(_empty_stream(), options)
            client = ClaudeSDKClient(options=options, transport=transport)
            finish = asyncio.Event()
            async def worker():
                async with client:
                    await self.query(client)
                    await finish.wait()
            task = asyncio.create_task(worker())
            lock_held = False
            timeout_task = None
            try:
                await self.wait_until(lambda: bool(server.pending(summary=True)))
                process = transport._process
                # Hold the real transport write lock so cancellation arrives
                # deterministically during normal close's first bounded wait.
                await transport._write_lock.acquire()
                lock_held = True
                finish.set()
                await self.wait_until(lambda: getattr(transport, "_cleanup_task", None) is not None)
                if mode == "wait_for":
                    timeout_task = asyncio.create_task(asyncio.wait_for(task, .02))
                    await asyncio.sleep(.1)
                    self.assertFalse(task.done())
                    # AnyIO locks must be released by their acquiring task.
                    transport._write_lock.release()
                    lock_held = False
                    with self.assertRaises(TimeoutError):
                        await timeout_task
                else:
                    for _ in range(3):
                        task.cancel()
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        self.assertIsNone(process.returncode)
                    transport._write_lock.release()
                    lock_held = False
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assert_reaped(process)
            finally:
                if lock_held:
                    transport._write_lock.release()
                if timeout_task is not None:
                    await asyncio.gather(timeout_task, return_exceptions=True)
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                # A cancelled transport caller cannot finish SDK disconnect's
                # receive-stream bookkeeping. Finish that idempotently after
                # asserting reaping; production shields the entire __aexit__.
                await await_cleanup(client.disconnect())

    async def test_repeated_cancel_during_normal_close_reaps_the_cli(self):
        await self._normal_close("repeated")

    async def test_wait_for_expiring_during_normal_close_reaps_the_cli(self):
        await self._normal_close("wait_for")

    async def test_shielding_client_exit_covers_reader_wait_before_transport_close(self):
        import anyio
        from claude_agent_sdk import ClaudeSDKClient
        from puppetllm.sdk_transport import RelaySDKTransport, await_cleanup

        reader_stopping, release_reader = asyncio.Event(), asyncio.Event()
        class GatedReaderTransport(RelaySDKTransport):
            async def read_messages(self):
                try:
                    async for message in super().read_messages():
                        yield message
                finally:
                    # Only a scheduling gate is added; CLI, control protocol,
                    # stdout parsing and SDK Query remain the real ones.
                    with anyio.CancelScope(shield=True):
                        reader_stopping.set()
                        await release_reader.wait()

        @asynccontextmanager
        async def protected_client(client):
            await client.__aenter__()
            try:
                yield client
            finally:
                await await_cleanup(client.__aexit__(None, None, None))

        with serve() as server, tempfile.TemporaryDirectory() as cwd:
            options = self.options(server, cwd)
            transport = GatedReaderTransport(_empty_stream(), options)
            client = ClaudeSDKClient(options=options, transport=transport)
            finish = asyncio.Event()
            async def worker():
                async with protected_client(client):
                    await self.query(client)
                    await finish.wait()
            task = asyncio.create_task(worker())
            try:
                await self.wait_until(lambda: bool(server.pending(summary=True)))
                process = transport._process
                finish.set()
                await asyncio.wait_for(reader_stopping.wait(), 15)
                self.assertIsNone(transport._cleanup_task)
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                release_reader.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assert_reaped(process)
                self.assertIsNone(client._query)
            finally:
                release_reader.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await await_cleanup(client.disconnect())
