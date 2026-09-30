"""Tests for the compatibility surface: the capabilities matrix, the catch-all for
unimplemented paths, root aliases, server-side tool blocks, reported / refused unknown
blocks, default response headers, bulk injection and `after_blocks`.

Run:
  python3 -m unittest puppetllm.tests.test_compat -v
"""

from __future__ import annotations

import asyncio
import json
import os
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
_INVOKE = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
           "messages": [{"role": "user", "content": "hi"}]}
_CONVERSE = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}
_OPENAI = {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]}
_SEARCH = {"type": "server_tool_use", "name": "web_search", "input": {"query": "tokyo weather"}}
_SEARCH_RESULT = {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
                  "content": [{"type": "web_search_result", "url": "https://example.com/w",
                               "title": "Weather", "encrypted_content": "abc", "page_age": None}]}


class _Base(unittest.TestCase):
    def setUp(self) -> None:
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

    async def _pending_n(self, c, bodies: list[dict]) -> list[Any]:
        tasks = [asyncio.create_task(c.post("/v1/messages", json=b, timeout=10)) for b in bodies]
        for _ in range(100):
            if (await c.get("/_control/pending")).json().get("count") == len(bodies):
                return tasks
            await asyncio.sleep(0.05)
        self.fail("not every request became pending")

    async def _answered(self, c, path: str, body: dict, **kw: Any) -> Any:
        """A request a rule must answer (never pending)."""
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


class TestCapabilitiesAndFallback(_Base):
    def test_capabilities_matrix(self) -> None:
        async def go():
            async with await self._client() as c:
                m = await self._ctl(c, "get", "/_control/capabilities")
                from puppetllm import __version__
                self.assertEqual(m["version"], __version__)
                ops = {s["operation"] for s in m["surfaces"]}
                self.assertIn("Messages", ops)
                self.assertIn("Converse / ConverseStream", ops)
                self.assertTrue(all({"accepted", "injectable", "streamed", "relayed"} <= set(s) for s in m["surfaces"]))
                self.assertIn("server_tool_use", m["content_blocks"])
                self.assertTrue(any(n["operation"] == "Responses API" for n in m["not_implemented"]))
        _run(go())

    def test_unimplemented_paths_answer_in_the_callers_envelope(self) -> None:
        async def go():
            async with await self._client() as c:
                r = await c.post("/v1/complete", json={"prompt": "x"}, headers={"x-api-key": "k"})
                self.assertEqual(r.status_code, 404)
                self.assertEqual(r.json()["error"]["type"], "not_found_error")
                self.assertIn("POST /v1/complete (Text Completions (legacy))", r.json()["error"]["message"])
                r = await c.post("/v1/files", headers={"x-api-key": "k"})
                self.assertIn("(Files API)", r.json()["error"]["message"])
                r = await c.post("/v1/whatever", headers={"x-api-key": "k"})
                self.assertNotIn("(", r.json()["error"]["message"].split(";")[0])
                self.assertIn("request-id", r.headers)
                r = await c.post("/v1/responses", json={"input": "x"}, headers={"authorization": "Bearer sk"})
                self.assertEqual(r.status_code, 404)
                self.assertEqual(r.json()["error"]["code"], "unknown_url")
                self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
                r = await c.post(f"{_BEDROCK}/invoke-with-bidirectional-stream", json={})
                self.assertEqual(r.status_code, 404)
                self.assertEqual(r.headers["x-amzn-ErrorType"], "UnknownOperationException")
                self.assertIn("__type", r.json())
                self.assertIn("(InvokeModelWithBidirectionalStream)", r.json()["message"])
                r = await c.post(f"{_BEDROCK}/something-else", json={})
                self.assertIn("(other model operations)", r.json()["message"])
                r = await c.post("/v1/filesystem", headers={"x-api-key": "k"})
                self.assertNotIn("(Files API)", r.json()["error"]["message"])
                r = await c.post("/guardrail/g1/version/1/apply", json={})
                self.assertEqual(r.status_code, 404)
                self.assertEqual(r.headers["x-amzn-ErrorType"], "UnknownOperationException")
                r = await c.post("/async-invoke", json={})
                self.assertEqual(r.headers["x-amzn-ErrorType"], "UnknownOperationException")
                r = await c.get("/anthropic/v1/nope", headers={"x-api-key": "k"})
                self.assertEqual(r.json()["error"]["type"], "not_found_error")
                # a wrong method on a real path is still a 405 with Allow (only what is served)
                r = await c.get("/v1/messages", headers={"x-api-key": "k"})
                self.assertEqual(r.status_code, 405)
                self.assertIn("POST", r.headers["allow"])
                r = await c.head("/v1/models", headers={"x-api-key": "k"})
                self.assertEqual(r.status_code, 405)
                self.assertEqual(r.headers["allow"], "GET")
                r = await c.options("/v1/messages")
                self.assertEqual(r.status_code, 405)
                for bare in ("/v1", "/anthropic", "/chat", "/model", "/guardrail"):
                    r = await c.get(bare, headers={"x-api-key": "k"})
                    self.assertEqual(r.status_code, 404, bare)
                    self.assertIn("puppetllm does not implement", r.text)
                r = await c.put("/v1/messages", content=b"x" * 5000)
                self.assertEqual(r.status_code, 405)
                r = await c.delete(f"{_BEDROCK}/invoke")
                self.assertEqual(r.status_code, 405)
                # the S3 emulation is untouched by the fallback
                r = await c.put("/some-bucket")
                self.assertEqual(r.status_code, 200)
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
        _run(go())

    def test_root_aliases(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "hi"}}], "repeat": True})
                r = await self._answered(c, "/chat/completions", _OPENAI, headers={"authorization": "Bearer sk"})
                self.assertEqual(r.json()["choices"][0]["message"]["content"], "hi")
                r = await c.get("/models", headers={"authorization": "Bearer sk"})
                self.assertEqual(r.json()["object"], "list")
                r = await c.get("/models/gpt-5.4", headers={"authorization": "Bearer sk"})
                self.assertEqual(r.json()["id"], "gpt-5.4")
                for path in ("/models/", "/v1/models/"):
                    r = await c.get(path, headers={"authorization": "Bearer sk"})
                    self.assertEqual(r.status_code, 404, path)
                    self.assertEqual(r.json()["error"]["code"], "unknown_url")
                self.assertEqual((await c.get("/chat/nope")).status_code, 404)
        _run(go())


class TestServerToolBlocks(_Base):
    def test_passthrough_non_stream_usage_and_cost(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG)
                out = await self._ctl(c, "post", "/_control/respond", json={"content": [
                    {"type": "text", "text": "Let me search."}, _SEARCH, _SEARCH_RESULT,
                    {"type": "text", "text": "It is sunny."}]})
                self.assertNotIn("dropped", out)
                r = (await t).json()
                types = [b["type"] for b in r["content"]]
                self.assertEqual(types, ["text", "server_tool_use", "web_search_tool_result", "text"])
                self.assertTrue(r["content"][1]["id"].startswith("srvtoolu_"))
                self.assertEqual(r["content"][2], _SEARCH_RESULT)
                self.assertEqual(r["stop_reason"], "end_turn")  # a server tool call does not end the turn
                self.assertEqual(r["usage"]["server_tool_use"], {"web_search_requests": 1, "web_fetch_requests": 0})
                hist = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(hist["cost"]["server_tool_usd"], 0.01)
                self.assertEqual([b["type"] for b in hist["response_blocks"]], types)
                # the result's text is not billed as output
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/respond", json={"content": [
                    {"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_2",
                     "content": {"type": "web_fetch_result", "url": "u", "retrieved_at": None,
                                 "content": {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x" * 4000}}}}]})
                r = (await t).json()
                self.assertLessEqual(r["usage"]["output_tokens"], 1)
                self.assertIsNone(r["usage"]["server_tool_use"])
                # a usage override (a relay) keeps the search price; null input → {}
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/respond", json={
                    "content": [{**_SEARCH, "input": None}],
                    "usage": {"input_tokens": 10, "output_tokens": 10, "server_tool_use": {"web_search_requests": 3, "web_fetch_requests": 0}}})
                r = (await t).json()
                self.assertEqual(r["content"][0]["input"], {})
                self.assertEqual(r["usage"]["server_tool_use"]["web_search_requests"], 3)
                hist = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(hist["cost"]["server_tool_usd"], 0.03)
                self.assertGreater(hist["cost"]["total_usd"], 0.03)
                # the Mantle alias carries the blocks too
                t = await self._pending(c, "/anthropic/v1/messages", {**_MSG, "model": "anthropic.claude-sonnet-5", "stream": True})
                await self._ctl(c, "post", "/_control/respond", json={"content": [_SEARCH, _SEARCH_RESULT]})
                events = _sse_events((await t).text)
                self.assertEqual([d["content_block"]["type"] for n, d in events if n == "content_block_start"],
                                 ["server_tool_use", "web_search_tool_result"])
        _run(go())

    def test_private_keys_and_billing_on_dropping_routes(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {**_MSG, "stream": True})
                await self._ctl(c, "post", "/_control/respond", json={"content": [
                    {**_SEARCH_RESULT, "_openai_custom": {"x": 1}}]})
                text = (await t).text
                self.assertNotIn("_openai_custom", text)
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/respond", json={"content": [
                    {**_SEARCH_RESULT, "_openai_custom": {"x": 1}}]})
                self.assertNotIn("_openai_custom", (await t).text)
                # a dropped server-tool call is not billed as output on the OpenAI route
                big = {**_SEARCH, "input": {"query": "q" * 4000}}
                await self._ctl(c, "post", "/_control/rules", json={"steps": [
                    {"respond": {"content": [big, {"type": "text", "text": "OK"}]}}], "repeat": True})
                r = await self._answered(c, "/v1/chat/completions", {**_OPENAI, "n": 3})
                self.assertLess(r.json()["usage"]["completion_tokens"], 30)
                r = await self._answered(c, "/v1/messages", _MSG)
                self.assertGreater(r.json()["usage"]["output_tokens"], 500)
        _run(go())

    def test_batch_discount_keeps_the_search_fee_whole(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [
                    {"respond": {"content": [_SEARCH, _SEARCH_RESULT, {"type": "text", "text": "t"}]}}], "repeat": True})
                r = await c.post("/v1/messages/batches", json={"requests": [{"custom_id": "s", "params": {**_MSG}}]})
                self.assertEqual(r.status_code, 200, r.text)
                bid = r.json()["id"]
                for _ in range(100):
                    if (await c.get(f"/v1/messages/batches/{bid}")).json()["processing_status"] == "ended":
                        break
                    await asyncio.sleep(0.02)
                cost = (await c.get("/_control/history")).json()["history"][-1]["cost"]
                self.assertEqual(cost["batch_discount"], 0.5)
                self.assertEqual(cost["server_tool_usd"], 0.01)
                parts = cost["input_usd"] + cost["output_usd"] + cost["cache_write_usd"] + cost["cache_read_usd"]
                self.assertAlmostEqual(cost["total_usd"], parts + 0.01, places=6)
        _run(go())

    def test_stream_shapes(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {**_MSG, "stream": True})
                await self._ctl(c, "post", "/_control/respond", json={"content": [
                    _SEARCH, _SEARCH_RESULT,
                    {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "q", "server_name": "srv", "input": {"a": 1}},
                    {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "is_error": False, "content": []},
                    {"type": "text", "text": "done"}]})
                events = _sse_events((await t).text)
                starts = [d["content_block"] for n, d in events if n == "content_block_start"]
                self.assertEqual([b["type"] for b in starts],
                                 ["server_tool_use", "web_search_tool_result", "mcp_tool_use", "mcp_tool_result", "text"])
                self.assertEqual(starts[0]["input"], {})
                self.assertEqual(starts[1], _SEARCH_RESULT)
                self.assertEqual(starts[2]["server_name"], "srv")
                deltas = [d for n, d in events if n == "content_block_delta" and d["index"] == 0]
                self.assertEqual("".join(d["delta"]["partial_json"] for d in deltas), json.dumps(_SEARCH["input"]))
                self.assertEqual([d for n, d in events if n == "content_block_delta" and d["index"] == 1], [])
                self.assertEqual([d["index"] for n, d in events if n == "content_block_stop"], [0, 1, 2, 3, 4])
                self.assertEqual(events[-2][1]["usage"]["server_tool_use"]["web_search_requests"], 1)
        _run(go())

    def test_other_routes_drop_them_and_validation(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [
                    {"respond": {"content": [_SEARCH, _SEARCH_RESULT, {"type": "text", "text": "t"}]}}], "repeat": True})
                r = await self._answered(c, f"{_BEDROCK}/converse", _CONVERSE)
                self.assertEqual(r.json()["output"]["message"]["content"], [{"text": "t"}])
                r = await self._answered(c, "/v1/chat/completions", _OPENAI)
                self.assertEqual(r.json()["choices"][0]["message"]["content"], "t")
                # dropped there → neither counted nor billed, even with an overridden count
                for e in (await c.get("/_control/history")).json()["history"][-2:]:
                    self.assertIsNone(e["usage"]["server_tool_use"])
                    self.assertEqual(e["cost"]["server_tool_usd"], 0.0)
                await self._ctl(c, "put", "/_control/rules", json={"rules": [{"steps": [
                    {"respond": {"content": [_SEARCH, {"type": "text", "text": "t"}],
                                 "usage": {"input_tokens": 5, "output_tokens": 5,
                                           "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0}}}}], "repeat": True}]})
                await self._answered(c, f"{_BEDROCK}/converse", _CONVERSE)
                e = (await c.get("/_control/history")).json()["history"][-1]
                self.assertIsNone(e["usage"]["server_tool_use"])
                self.assertEqual(e["cost"]["server_tool_usd"], 0.0)
                await self._ctl(c, "put", "/_control/rules", json={"rules": [{"steps": [
                    {"respond": {"content": [_SEARCH, _SEARCH_RESULT, {"type": "text", "text": "t"}]}}], "repeat": True}]})
                r = await self._answered(c, f"{_BEDROCK}/invoke", _INVOKE)
                self.assertEqual([b["type"] for b in r.json()["content"]],
                                 ["server_tool_use", "web_search_tool_result", "text"])
                for bad, needle in (({"type": "web_search_tool_result", "content": []}, "tool_use_id"),
                                    ({"type": "web_search_tool_result", "tool_use_id": "x"}, "content"),
                                    ({"type": "server_tool_use", "input": {}}, "name"),
                                    ({"type": "server_tool_use", "name": "web_search", "input": 5}, "input"),
                                    ({"type": "mcp_tool_use", "name": "q", "input": {}}, "server_name")):
                    r = await c.post("/_control/rules", json={"steps": [{"respond": {"content": [bad]}}]})
                    self.assertEqual(r.status_code, 400, bad)
                    self.assertIn(needle, r.json()["error"])
        _run(go())


class TestDroppedBlocks(_Base):
    def test_reported_or_refused(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG)
                out = await self._ctl(c, "post", "/_control/respond", json={"content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": "y"},
                    {"type": "tool-use", "name": "typo"}, {"type": "text", "text": "kept"}]})
                self.assertEqual(out, {"ok": True, "dropped": ["tool_result", "tool-use"]})
                self.assertEqual([b["type"] for b in (await t).json()["content"]], ["text"])
                out = await self._ctl(c, "post", "/_control/rules", json={"id": "r", "steps": [
                    {"respond": {"content": [{"type": "image", "source": {}}]}},
                    {"error": {"status": 500, "after_events": 1, "content": [{"type": "nope"}]}}]})
                self.assertEqual(out["dropped"], [{"rule": "r", "step": 0, "type": "image"},
                                                  {"rule": "r", "step": 1, "type": "nope"}])
                self.assertNotIn("dropped", await self._ctl(c, "get", "/_control/rules"))
                await self._ctl(c, "delete", "/_control/rules")
                await self._ctl(c, "post", "/_control/config", json={"strict_blocks": True})
                t = await self._pending(c, "/v1/messages", _MSG)
                r = await c.post("/_control/respond", json={"content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]})
                self.assertEqual(r.status_code, 400)
                self.assertIn("tool_result", r.json()["error"])
                self.assertIn("next", r.json()["error"])
                r = await c.post("/_control/error", json={"status": 500, "content": [{"type": "bogus"}]})
                self.assertEqual(r.status_code, 400)
                r = await c.post("/_control/rules", json={"steps": [{"respond": {"content": [{"type": "bogus"}]}}]})
                self.assertEqual(r.status_code, 400)
                self.assertIn("rule", r.json()["error"])
                await self._ctl(c, "post", "/_control/auto", json={"text": "still fine"})
                self.assertEqual((await t).status_code, 200)
                self.assertTrue((await self._ctl(c, "get", "/_control/config"))["config"]["strict_blocks"])
                self.assertEqual((await c.post("/_control/config", json={"strict_blocks": "yes"})).status_code, 400)
                # strictness set in the same body applies to the payloads it carries
                await self._ctl(c, "post", "/_control/config", json={"strict_blocks": False})
                r = await c.post("/_control/config", json={
                    "strict_blocks": True, "default_response": {"content": [{"type": "bogus"}]}})
                self.assertEqual(r.status_code, 400)
                self.assertFalse((await self._ctl(c, "get", "/_control/config"))["config"]["strict_blocks"])
                await self._ctl(c, "post", "/_control/config", json={"strict_blocks": True})
                out = await self._ctl(c, "post", "/_control/config", json={
                    "strict_blocks": None, "default_response": {"content": [{"type": "bogus"}]}})
                self.assertFalse(out["config"]["strict_blocks"])
        _run(go())


class TestDefaultHeaders(_Base):
    def test_added_to_api_responses_only(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/config", json={"default_headers": {
                    "anthropic-ratelimit-requests-limit": "50", "x-ratelimit-remaining-tokens": 900}})
                cfg = (await self._ctl(c, "get", "/_control/config"))["config"]
                self.assertEqual(cfg["default_headers"]["x-ratelimit-remaining-tokens"], "900")
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                r = await self._answered(c, "/v1/messages", _MSG)
                self.assertEqual(r.headers["anthropic-ratelimit-requests-limit"], "50")
                r = await self._answered(c, "/v1/messages", {**_MSG, "stream": True})
                self.assertEqual(r.headers["x-ratelimit-remaining-tokens"], "900")
                r = await self._answered(c, "/v1/chat/completions", _OPENAI)
                self.assertEqual(r.headers["anthropic-ratelimit-requests-limit"], "50")
                r = await c.post("/v1/messages/count_tokens", json={"model": "m", "messages": []})
                self.assertIn("x-ratelimit-remaining-tokens", r.headers)
                r = await c.get("/_control/pending")
                self.assertNotIn("x-ratelimit-remaining-tokens", r.headers)
                r = await c.get("/openapi.json")
                self.assertNotIn("x-ratelimit-remaining-tokens", r.headers)
                r = await c.get("/docs")
                self.assertNotIn("x-ratelimit-remaining-tokens", r.headers)
                # an S3 bucket whose name merely starts like a docs path is an API response
                r = await c.put("/docs-backups")
                self.assertEqual(r.status_code, 200)
                self.assertIn("x-ratelimit-remaining-tokens", r.headers)
                # a route's own header wins; a rate-limit 429 keeps its live values
                await self._ctl(c, "post", "/_control/config", json={
                    "default_headers": {"retry-after": "99"}, "rate_limit": {"rpm": 1}})
                self.assertEqual((await self._answered(c, "/v1/messages", _MSG)).status_code, 200)
                r = await self._answered(c, "/v1/messages", _MSG)
                self.assertEqual(r.status_code, 429)
                self.assertNotEqual(r.headers["retry-after"], "99")
                for bad in ({"content-length": "1"}, {"x": "bad\nvalue"}, "x", {"": "v"},
                            {"\u0436": "1"}, {"\u00e9": "1"}):
                    self.assertEqual((await c.post("/_control/config", json={"default_headers": bad})).status_code, 400, bad)
                await self._ctl(c, "delete", "/_control/rules")
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": None})
                t = await self._pending(c, "/v1/messages", _MSG)
                self.assertEqual((await c.post("/_control/error", json={"status": 500, "headers": {"\u00e9": "1"}})).status_code, 400)
                await self._ctl(c, "post", "/_control/auto", json={"text": "ok"})
                await t
                await self._ctl(c, "post", "/_control/config", json={"default_headers": None})
                self.assertEqual((await self._ctl(c, "get", "/_control/config"))["config"]["default_headers"], {})
        _run(go())


class TestBulkInjection(_Base):
    def test_respond_all(self) -> None:
        async def go():
            async with await self._client() as c:
                self.assertEqual((await c.post("/_control/respond_all", json={"text": "x"})).status_code, 400)
                tasks = await self._pending_n(c, [{**_MSG, "messages": [{"role": "user", "content": f"q{i}"}]} for i in range(3)])
                out = await self._ctl(c, "post", "/_control/respond_all", json={
                    "content": [{"type": "tool_use", "name": "f", "input": {}}], "stop_reason": "tool_use"})
                self.assertEqual(out["count"], 3)
                self.assertEqual(len(out["pending_ids"]), 3)
                ids = {(await t).json()["content"][0]["id"] for t in tasks}
                self.assertEqual(len(ids), 3)  # each answer gets its own generated id
                r = await c.post("/_control/respond_all", json={"text": "x", "pending_id": "p"})
                self.assertEqual(r.status_code, 400)
                # batch entries are part of the fan-out
                r = await c.post("/v1/messages/batches", json={"requests": [{"custom_id": "b1", "params": {**_MSG}}]})
                self.assertEqual(r.status_code, 200, r.text)
                bid = r.json()["id"]
                for _ in range(100):
                    if (await c.get("/_control/pending")).json()["count"] == 1:
                        break
                    await asyncio.sleep(0.02)
                out = await self._ctl(c, "post", "/_control/respond_all", json={"text": "batched"})
                self.assertEqual(out["count"], 1)
                for _ in range(100):
                    b = (await c.get(f"/v1/messages/batches/{bid}")).json()
                    if b["processing_status"] == "ended":
                        break
                    await asyncio.sleep(0.02)
                self.assertEqual(b["request_counts"]["succeeded"], 1)
        _run(go())

    def test_bulk_responses_validate_everything_first(self) -> None:
        async def go():
            async with await self._client() as c:
                tasks = await self._pending_n(c, [{**_MSG, "messages": [{"role": "user", "content": f"q{i}"}]} for i in range(2)])
                pids = [p["pending_id"] for p in (await c.get("/_control/pending")).json()["pending"]]
                r = await c.post("/_control/respond", json={"responses": [
                    {"pending_id": pids[0], "text": "a"}, {"pending_id": "nope", "text": "b"}]})
                self.assertEqual(r.status_code, 400)
                self.assertIn("responses[1]", r.json()["error"])
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 2)  # nothing consumed
                r = await c.post("/_control/respond", json={"responses": [
                    {"pending_id": pids[0], "text": "a"}, {"pending_id": pids[0], "text": "b"}]})
                self.assertIn("twice", r.json()["error"])
                r = await c.post("/_control/respond", json={"responses": [{"text": "a"}]})
                self.assertEqual(r.status_code, 400)  # ambiguous target with two pendings
                out = await self._ctl(c, "post", "/_control/respond", json={"responses": [
                    {"pending_id": pids[0], "text": "a"},
                    {"pending_id": pids[1], "content": [{"type": "text", "text": "b"}, {"type": "junk"}]}]})
                self.assertEqual(out, {"ok": True, "results": [{"ok": True}, {"ok": True}], "dropped": ["junk"]})
                got = sorted([(await t).json()["content"][0]["text"] for t in tasks])
                self.assertEqual(got, ["a", "b"])
                self.assertEqual((await c.post("/_control/respond", json={"responses": []})).status_code, 400)
                self.assertEqual((await c.post("/_control/respond", json={"responses": [{"text": "a"}], "text": "x"})).status_code, 400)
                # a generated client's defaults around either form are fine
                t = await self._pending(c, "/v1/messages", _MSG)
                out = await self._ctl(c, "post", "/_control/respond", json={"text": "single", "responses": None, "content": []})
                self.assertEqual(out, {"ok": True})
                self.assertEqual((await t).json()["content"][0]["text"], "single")
                t = await self._pending(c, "/v1/messages", _MSG)
                pid = (await c.get("/_control/pending")).json()["pending"][0]["pending_id"]
                await self._ctl(c, "post", "/_control/respond", json={
                    "responses": [{"pending_id": pid, "text": "bulk"}], "content": [], "text": None,
                    "pending_id": None, "custom_id": None, "batch_id": None})
                self.assertEqual((await t).json()["content"][0]["text"], "bulk")
        _run(go())


class TestAfterBlocks(_Base):
    def test_after_blocks_on_every_stream(self) -> None:
        async def go():
            from puppetllm.providers import eventstream
            async with await self._client() as c:
                content = [{"type": "text", "text": "one"}, {"type": "tool_use", "name": "f", "input": {"k": 1}},
                           {"type": "text", "text": "three"}]
                await self._ctl(c, "post", "/_control/rules", json={"steps": [
                    {"error": {"status": 529, "after_blocks": 2, "content": content}}], "repeat": True})
                r = await self._answered(c, "/v1/messages", {**_MSG, "stream": True})
                names = [n for n, _ in _sse_events(r.text)]
                self.assertEqual(names.count("content_block_stop"), 2)
                self.assertEqual(names[-1], "error")
                self.assertNotIn("message_stop", names)
                hist = (await c.get("/_control/history")).json()["history"][-1]["injected_error"]
                self.assertEqual(hist["after_blocks"], 2)
                self.assertNotIn("after_events", hist)
                r = await self._answered(c, f"{_BEDROCK}/invoke-with-response-stream", _INVOKE)
                msgs = eventstream.decode_messages(r.content)
                self.assertEqual(sum(1 for m in msgs if m.get("type") == "content_block_stop"), 2)
                self.assertIn("_exception", msgs[-1])
                r = await self._answered(c, f"{_BEDROCK}/converse-stream", _CONVERSE)
                msgs = eventstream.decode_messages(r.content)
                self.assertEqual(sum(1 for m in msgs if m.get("_event") == "contentBlockStop"), 2)
                self.assertIn("_exception", msgs[-1])
                # 0 → the stream starts and fails at once; more blocks than exist → all of them
                await self._ctl(c, "put", "/_control/rules", json={"rules": [{"steps": [
                    {"error": {"status": 500, "after_blocks": 0, "content": content}},
                    {"error": {"status": 500, "after_blocks": 9, "content": content}}]}]})
                names = [n for n, _ in _sse_events((await self._answered(c, "/v1/messages", {**_MSG, "stream": True})).text)]
                self.assertEqual(names, ["message_start", "ping", "error"])
                names = [n for n, _ in _sse_events((await self._answered(c, "/v1/messages", {**_MSG, "stream": True})).text)]
                self.assertEqual(names.count("content_block_stop"), 3)
                self.assertNotIn("message_delta", names)
                # the OpenAI route has no mid-stream form: a plain error, nothing partial recorded
                await self._ctl(c, "put", "/_control/rules", json={"rules": [{"steps": [
                    {"error": {"status": 500, "after_blocks": 1, "content": content}}]}]})
                r = await self._answered(c, "/v1/chat/completions", {**_OPENAI, "stream": True})
                self.assertEqual(r.status_code, 500)
                err = (await c.get("/_control/history")).json()["history"][-1]["injected_error"]
                self.assertNotIn("after_blocks", err)
                self.assertNotIn("partial_content", err)
                r = await c.post("/_control/error", json={"status": 500, "after_blocks": 1, "after_events": 1})
                self.assertEqual(r.status_code, 400)
                r = await c.post("/_control/error", json={"status": 500, "after_blocks": -1})
                self.assertEqual(r.status_code, 400)
        _run(go())

    def test_auto_is_deprecated_in_the_schema(self) -> None:
        async def go():
            async with await self._client() as c:
                spec = (await c.get("/openapi.json")).json()
                self.assertTrue(spec["paths"]["/_control/auto"]["post"].get("deprecated"))
                self.assertIn("/_control/respond_all", spec["paths"])
                self.assertIn("/_control/capabilities", spec["paths"])
                self.assertNotIn("/models", spec["paths"])  # aliases stay out of the published schema
        _run(go())


if __name__ == "__main__":
    unittest.main()
