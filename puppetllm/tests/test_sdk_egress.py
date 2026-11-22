"""Loopback-only gate tests: no credentials, model calls or paid APIs."""
import asyncio
from contextlib import contextmanager
import gzip
import json
import signal
import socket
import unittest
from unittest.mock import patch

import httpx
import uvicorn

from puppetllm.sdk_egress import SDKAPIGate, _Metadata


class _Server(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        yield


class _Upstream:
    """Independent ASGI fixture recording actual upstream HTTP arrivals."""
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def __aenter__(self):
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(32)
        self.socket.setblocking(False)
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}"

        async def app(scope, receive, send):
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if not message.get("more_body"):
                    break
            self.calls.append({"path": scope["path"], "raw_path": scope["raw_path"],
                               "query": scope["query_string"], "headers": dict(scope["headers"]),
                               "body": bytes(body)})
            status, headers, content = await self.reply(self.calls[-1])
            await send({"type": "http.response.start", "status": status, "headers": headers})
            if isinstance(content, bytes):
                await send({"type": "http.response.body", "body": content})
            else:
                async for chunk in content:
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b""})

        self.server = _Server(uvicorn.Config(app, log_config=None, log_level="critical", access_log=False,
                                            lifespan="off", timeout_graceful_shutdown=.2))
        self.task = asyncio.create_task(self.server.serve(sockets=[self.socket]))
        async def started():
            while not self.server.started:
                if self.task.done():
                    self.task.result()
                await asyncio.sleep(.001)
        await asyncio.wait_for(started(), 2)
        return self

    async def __aexit__(self, *exc):
        self.server.should_exit = True
        try:
            await asyncio.wait_for(self.task, 2)
        finally:
            self.socket.close()


def _event(reason="max_tokens", **usage):
    return b"event: message_delta\ndata: " + json.dumps({
        "type": "message_delta", "delta": {"stop_reason": reason}, "usage": usage}).encode() + b"\n\n"


class TestSDKAPIGate(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_are_reserved_before_first_response(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def reply(_request):
            entered.set()
            await release.wait()
            return 200, [(b"content-type", b"text/event-stream")], _event(output_tokens=4)
        async with _Upstream(reply) as upstream, SDKAPIGate(upstream.url) as gate:
            async with httpx.AsyncClient(trust_env=False) as client:
                first = asyncio.create_task(client.post(gate.base_url + "/v1/messages", json={"stream": True}))
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    others = await asyncio.gather(*(client.post(gate.base_url + "/v1/messages", content=b"PRIVATE_BODY") for _ in range(8)))
                    self.assertEqual([response.status_code for response in others], [503] * 8)
                    self.assertTrue(all(response.headers["x-should-retry"] == "false" for response in others))
                    self.assertEqual(gate.started_requests, 1)
                    self.assertEqual(gate.blocked_requests, 8)
                    self.assertEqual(len(upstream.calls), 1)
                    self.assertTrue(all("PRIVATE" not in response.text for response in others))
                finally:
                    release.set()
                self.assertEqual((await first).status_code, 200)

    async def test_preserves_base_path_query_body_auth_and_anthropic_headers(self):
        async def reply(_request):
            return 200, [(b"content-type", b"application/json"), (b"connection", b"x-hop"),
                         (b"x-hop", b"remove"), (b"x-request-id", b"local-id")], b'{"ok":true}'
        async with _Upstream(reply) as upstream, SDKAPIGate(upstream.url + "/tenant/prefix/") as gate:
            self.assertIn("/sdk-", gate.base_url)
            self.assertTrue(gate.base_url.endswith("/tenant/prefix"))
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(gate.base_url + "/v1/messages?beta=true", content=b"PRIVATE_BODY", headers={
                    "Authorization": "Bearer PRIVATE_AUTH", "x-api-key": "PRIVATE_KEY",
                    "anthropic-version": "2023-06-01", "anthropic-beta": "local-test",
                    "Connection": "x-local-hop", "x-local-hop": "remove", "Accept-Encoding": "gzip"})
            call = upstream.calls[0]
            self.assertEqual(call["path"], "/tenant/prefix/v1/messages")
            self.assertEqual(call["query"], b"beta=true")
            self.assertEqual(call["body"], b"PRIVATE_BODY")
            self.assertEqual(call["headers"][b"authorization"], b"Bearer PRIVATE_AUTH")
            self.assertEqual(call["headers"][b"x-api-key"], b"PRIVATE_KEY")
            self.assertEqual(call["headers"][b"anthropic-version"], b"2023-06-01")
            self.assertEqual(call["headers"][b"anthropic-beta"], b"local-test")
            self.assertEqual(call["headers"][b"accept-encoding"], b"identity")
            self.assertNotIn(b"x-local-hop", call["headers"])
            self.assertNotIn("x-hop", response.headers)
            self.assertEqual(response.headers["x-request-id"], "local-id")
            self.assertEqual(response.content, b'{"ok":true}')

    async def test_non_api_wrong_prefix_and_wrong_method_never_reach_upstream(self):
        async def reply(_request):
            self.fail("rejected paths must not contact the upstream")
        async with _Upstream(reply) as upstream, SDKAPIGate(upstream.url) as gate:
            origin = gate.base_url.split("/sdk-", 1)[0]
            async with httpx.AsyncClient(trust_env=False) as client:
                for method, url in (("POST", origin + "/v1/messages"), ("GET", gate.base_url + "/v1/messages"),
                                    ("POST", gate.base_url + "/v1/messages/count_tokens"),
                                    ("POST", gate.base_url + "/other"), ("POST", origin + "/docs")):
                    with self.subTest(method=method, url=url):
                        self.assertEqual((await client.request(method, url)).status_code, 404)
            self.assertEqual(gate.started_requests, 0)
            self.assertFalse(gate.used)
            self.assertEqual(upstream.calls, [])

    async def test_redirect_cannot_escape_the_gate_even_when_client_follows_it(self):
        async def destination(_request):
            self.fail("a remote Location must never escape the quota")
        async with _Upstream(destination) as escaped:
            async def redirect(_request):
                return 307, [(b"location", (escaped.url + "/v1/messages").encode())], b"PRIVATE_REDIRECT"
            async with _Upstream(redirect) as upstream, SDKAPIGate(upstream.url) as gate:
                async with httpx.AsyncClient(follow_redirects=True, trust_env=False) as client:
                    response = await client.post(gate.base_url + "/v1/messages", content=b"{}")
                self.assertEqual(response.status_code, 502)
                self.assertNotIn("location", response.headers)
                self.assertEqual(gate.status_codes, [307])
                self.assertEqual(len(upstream.calls), 1)
                self.assertEqual(escaped.calls, [])
                self.assertNotIn("PRIVATE", response.text)

    async def test_sse_bytes_and_max_tokens_metadata_survive_chunk_boundaries(self):
        leading = b"event: content_block_delta\r\ndata: " + json.dumps({"type": "content_block_delta", "delta": {"signature": "PRIVATE_SIGNATURE" * 12000}}).encode() + b"\r\n\r\n"
        ending = _event(output_tokens=4096, input_tokens=13, private="PRIVATE_USAGE", cache_read_input_tokens=0)
        payload = leading + ending + b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        async def chunks():
            for offset in range(0, len(payload), 7919):
                yield payload[offset:offset + 7919]
        async def reply(_request):
            return 200, [(b"content-type", b"text/event-stream")], chunks()
        async with _Upstream(reply) as upstream, SDKAPIGate(upstream.url) as gate:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(gate.base_url + "/v1/messages", content=b"{}")
            self.assertEqual(response.content, payload)
            self.assertEqual(gate.stop_reason, "max_tokens")
            self.assertEqual(gate.usage, {"output_tokens": 4096, "input_tokens": 13, "cache_read_input_tokens": 0})

    async def test_gzip_remains_raw_and_the_quota_also_holds(self):
        payload = gzip.compress(_event(output_tokens=19))
        async def reply(_request):
            return 200, [(b"content-type", b"text/event-stream"), (b"content-encoding", b"gzip")], payload
        async with _Upstream(reply) as upstream, SDKAPIGate(upstream.url) as gate:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.send(client.build_request("POST", gate.base_url + "/v1/messages", content=b"{}"), stream=True)
                raw = b"".join([chunk async for chunk in response.aiter_raw()])
                await response.aclose()
                retry = await client.post(gate.base_url + "/v1/messages", content=b"{}")
            self.assertEqual(raw, payload)
            self.assertEqual(response.headers["content-encoding"], "gzip")
            self.assertEqual(gate.stop_reason, "max_tokens")
            self.assertEqual(gate.usage, {"output_tokens": 19})
            self.assertEqual(retry.status_code, 503)
            self.assertEqual(len(upstream.calls), 1)

    async def test_upstream_error_status_and_body_pass_through_without_diagnostics(self):
        async def reply(_request):
            return 401, [(b"content-type", b"application/json")], b'{"error":{"message":"PRIVATE_AUTH"}}'
        async with _Upstream(reply) as upstream, SDKAPIGate(upstream.url) as gate:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(gate.base_url + "/v1/messages", content=b"{}")
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.content, b'{"error":{"message":"PRIVATE_AUTH"}}')
            self.assertEqual(gate.status_codes, [401])

    async def test_connection_failure_is_not_retried_or_given_another_allowance(self):
        arrivals = []
        async def disconnect(_reader, writer):
            arrivals.append(True)
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(disconnect, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            async with SDKAPIGate(f"http://127.0.0.1:{port}") as gate:
                async with httpx.AsyncClient(trust_env=False) as client:
                    self.assertEqual((await client.post(gate.base_url + "/v1/messages", content=b"{}")).status_code, 502)
                    self.assertEqual((await client.post(gate.base_url + "/v1/messages", content=b"{}")).status_code, 503)
                self.assertEqual(arrivals, [True])
                self.assertEqual(gate.started_requests, 1)
        finally:
            server.close()
            await server.wait_closed()

    async def test_close_waits_through_repeated_cancellation_and_is_idempotent(self):
        gate = await SDKAPIGate("http://127.0.0.1:1").__aenter__()
        entered, release = asyncio.Event(), asyncio.Event()
        original = gate._client.aclose
        async def delayed_close():
            entered.set()
            await release.wait()
            await original()
        with patch.object(gate._client, "aclose", delayed_close):
            owner = asyncio.create_task(gate.close())
            await entered.wait()
            for _ in range(3):
                owner.cancel()
                await asyncio.sleep(0)
                self.assertFalse(owner.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await owner
        await gate.close()
        self.assertEqual(gate._socket.fileno(), -1)
        self.assertTrue(gate._client.is_closed)
        self.assertTrue(gate._server_task.done())

    async def test_close_reclaims_an_active_response_stream(self):
        release = asyncio.Event()
        async def chunks():
            yield b"event: ping\ndata: {}\n\n"
            await release.wait()
        async def reply(_request):
            return 200, [(b"content-type", b"text/event-stream")], chunks()
        async with _Upstream(reply) as upstream:
            gate = await SDKAPIGate(upstream.url).__aenter__()
            try:
                async with httpx.AsyncClient(trust_env=False) as client:
                    response = await client.send(client.build_request(
                        "POST", gate.base_url + "/v1/messages", content=b"{}"), stream=True)
                    await asyncio.wait_for(gate.close(), 2)
                    self.assertTrue(gate._client.is_closed)
                    self.assertTrue(gate._server_task.done())
                    self.assertFalse(gate._responses)
                    self.assertFalse(gate._handlers)
                    self.assertEqual(gate._socket.fileno(), -1)
                    await response.aclose()
            finally:
                release.set()
                await gate.close()

    async def test_optional_callbacks_report_safe_metadata_without_controlling_admission(self):
        class Progress:
            started = blocked = 0
            metadata = None
            def api_request_started(self):
                self.started += 1
            def api_request_blocked(self):
                self.blocked += 1
                raise RuntimeError("PRIVATE_CALLBACK")
            def api_response_metadata(self, reason, usage):
                self.metadata = (reason, dict(usage))
                usage["PRIVATE_MUTATION"] = "ignored"
        progress = Progress()
        async def reply(_request):
            return 200, [(b"content-type", b"text/event-stream")], _event(output_tokens=7)
        async with _Upstream(reply) as upstream, SDKAPIGate(upstream.url, progress=progress) as gate:
            async with httpx.AsyncClient(trust_env=False) as client:
                self.assertEqual((await client.post(gate.base_url + "/v1/messages", content=b"{}")).status_code, 200)
                self.assertEqual((await client.post(gate.base_url + "/v1/messages", content=b"{}")).status_code, 503)
            self.assertEqual((progress.started, progress.blocked), (1, 1))
            self.assertEqual(progress.metadata, ("max_tokens", {"output_tokens": 7}))
            self.assertEqual(gate.usage, {"output_tokens": 7})

    async def test_loopback_server_does_not_replace_sigterm_handler(self):
        loop = asyncio.get_running_loop()
        notified = asyncio.Event()
        loop.add_signal_handler(signal.SIGTERM, notified.set)
        try:
            async with SDKAPIGate("http://127.0.0.1:1"):
                signal.raise_signal(signal.SIGTERM)
                await asyncio.wait_for(notified.wait(), 1)
        finally:
            loop.remove_signal_handler(signal.SIGTERM)


class TestMetadata(unittest.TestCase):
    def test_retained_bytes_stay_bounded_and_only_delta_metadata_is_exposed(self):
        found = []
        parser = _Metadata(lambda reason, usage: found.append((reason, usage)))
        content = b"event: content_block_delta\ndata: " + b"PRIVATE_BODY" * 20000 + b"\n\n" + _event(output_tokens=9)
        for offset in range(0, len(content), 101):
            parser.feed(content[offset:offset + 101])
            self.assertLessEqual(len(parser.line) + len(parser.data), 64 * 1024)
        parser.finish()
        self.assertEqual(found, [("max_tokens", {"output_tokens": 9})])

    def test_target_credentials_and_invalid_limits_are_rejected_without_echo(self):
        for target in ("https://PRIVATE_AUTH@example.com", "https://user:PRIVATE_AUTH@example.com", "file:///PRIVATE_AUTH", "https://example.com?PRIVATE_AUTH"):
            with self.subTest(target=target), self.assertRaises(ValueError) as caught:
                SDKAPIGate(target)
            self.assertNotIn("PRIVATE", str(caught.exception))
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                SDKAPIGate("http://127.0.0.1:1", max_requests=limit)


if __name__ == "__main__":
    unittest.main()
