"""Tests for the test-harness features: scenario rules, the unmatched policy, pending
timeout and disconnect cleanup, latency and rate-limit simulation, the fake clock,
count_tokens / models, the typed control schema, the CLI and `puppetllm.testing`.

Run:
  python3 -m unittest puppetllm.tests.test_harness -v
"""

from __future__ import annotations

import asyncio
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
_WEATHER_TOOL = {"name": "get_weather", "description": "weather",
                 "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
_TOOL_USE = {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Tokyo"}}


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

    async def _pending(self, c, path: str, body: dict) -> Any:
        t = asyncio.create_task(c.post(path, json=body, timeout=10))
        for _ in range(50):
            if (await c.get("/_control/pending")).json().get("has_pending"):
                return t
            await asyncio.sleep(0.05)
        self.fail("never became pending")

    async def _no_pending_reply(self, c, path: str, body: dict, headers: dict | None = None) -> Any:
        """POST a request the harness must answer by itself (it must never become a pending
        a responder would have to answer; a wrong policy would hang the test forever)."""
        t = asyncio.create_task(c.post(path, json=body, timeout=10, headers=headers))
        for _ in range(200):
            if t.done():
                return t.result()
            if (await c.get("/_control/pending")).json().get("has_pending"):
                await c.post("/_control/clear")
                await asyncio.gather(t, return_exceptions=True)
                self.fail("request became pending instead of being answered by the harness")
            await asyncio.sleep(0.01)
        t.cancel()
        self.fail("request neither answered nor pending")


class TestRules(_Base):
    def test_tool_scenario_in_order_then_pending(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "put", "/_control/rules", json={"rules": [
                    {"id": "call", "match": {"tools": ["get_weather"], "has_tool_result": False},
                     "steps": [{"respond": {"content": [_TOOL_USE]}}]},
                    {"id": "final", "match": {"has_tool_result": True, "last_user_text": "sunny"},
                     "steps": [{"respond": {"text": "It is sunny in Tokyo."}}]},
                ]})
                body = {**_MSG, "tools": [_WEATHER_TOOL]}
                r = await self._no_pending_reply(c, "/v1/messages", body)
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["stop_reason"], "tool_use")
                self.assertEqual(r.json()["content"][0]["name"], "get_weather")
                # The unmatched tool_result turn (text does not say "sunny") stays pending.
                body2 = {**body, "messages": body["messages"] + [
                    {"role": "assistant", "content": [_TOOL_USE]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01",
                                                  "content": "rainy"}]}]}
                t = await self._pending(c, "/v1/messages", body2)
                view = await self._ctl(c, "get", "/_control/rules")
                self.assertEqual(view["unconsumed"], ["final"])
                self.assertFalse(view["all_consumed"])
                await self._ctl(c, "post", "/_control/respond", json={"text": "manual"})
                self.assertEqual((await t).json()["content"][0]["text"], "manual")
                body3 = {**body, "messages": body["messages"] + [
                    {"role": "assistant", "content": [_TOOL_USE]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01",
                                                  "content": "sunny"}]}]}
                r = await self._no_pending_reply(c, "/v1/messages", body3)
                self.assertEqual(r.json()["content"][0]["text"], "It is sunny in Tokyo.")
                view = await self._ctl(c, "get", "/_control/rules")
                self.assertTrue(view["all_consumed"])
                call = next(x for x in view["rules"] if x["id"] == "call")
                self.assertEqual((call["matched"], call["remaining"], call["exhausted"]), (1, 0, True))
                hist = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(hist[0]["harness"], {"source": "rule", "rule_id": "call"})
                self.assertNotIn("harness", hist[1])
        _run(go())

    def test_sequence_error_then_success_and_repeat(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={
                    "id": "flaky", "match": {"model": "claude-*", "stream": False},
                    "steps": [{"error": {"status": 429, "headers": {"retry-after": "1"}}},
                              {"respond": {"text": "second time lucky"}}],
                    "repeat": True})
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.headers["retry-after"], "1")
                self.assertEqual(r.json()["error"]["type"], "rate_limit_error")
                for _ in range(2):  # repeat: the last step answers forever
                    r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                    self.assertEqual(r.json()["content"][0]["text"], "second time lucky")
                view = await self._ctl(c, "get", "/_control/rules")
                self.assertEqual(view["rules"][0]["matched"], 3)
                self.assertIsNone(view["rules"][0]["remaining"])
                self.assertTrue(view["all_consumed"])
                hist = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(hist[0]["injected_error"]["headers"], {"retry-after": "1"})
                # A streaming request does not match `stream: false` → pending.
                t = await self._pending(c, "/v1/messages", {**_MSG, "stream": True})
                await self._ctl(c, "post", "/_control/clear")
                await asyncio.gather(t, return_exceptions=True)
        _run(go())

    def test_rules_on_every_provider_and_stream(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "put", "/_control/rules", json={"rules": [
                    {"match": {"provider": "openai"}, "steps": [{"respond": {"text": "from-openai"}}]},
                    {"match": {"provider": "bedrock"}, "steps": [{"respond": {"text": "from-bedrock"}}]},
                    {"match": {"provider": "anthropic", "turn": 3},
                     "steps": [{"respond": {"text": "third"}}]},
                ]})
                r = await self._no_pending_reply(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.json()["choices"][0]["message"]["content"], "from-openai")
                r = await self._no_pending_reply(
                    c, "/model/anthropic.claude-sonnet-4-5-20250929-v1:0/invoke",
                    {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                     "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.json()["content"][0]["text"], "from-bedrock")
                r = await self._no_pending_reply(c, "/v1/messages", {**_MSG, "stream": True})
                self.assertEqual(r.status_code, 200)
                deltas = [d["delta"]["text"] for n, d in _sse_events(r.text) if n == "content_block_delta"]
                self.assertEqual("".join(deltas), "third")
                # matched anthropic rule was turn 3 exactly; turn 4 is unmatched → pending
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/clear")
                await asyncio.gather(t, return_exceptions=True)
                self.assertEqual((await self._ctl(c, "get", "/_control/rules"))["rules"], [])
        _run(go())

    def test_rule_management_and_validation(self) -> None:
        async def go():
            async with await self._client() as c:
                out = await self._ctl(c, "post", "/_control/rules", json={"rules": [
                    {"id": "a", "steps": [{"respond": {"text": "A"}}]},
                    {"id": "b", "steps": [{"respond": {"text": "B"}}]}]})
                self.assertEqual([r["id"] for r in out["rules"]], ["a", "b"])
                r = await c.post("/_control/rules", json={"id": "a", "steps": [{"respond": {"text": "x"}}]})
                self.assertEqual(r.status_code, 400)
                self.assertIn("already present", r.json()["error"])
                out = await self._ctl(c, "put", "/_control/rules/a", json={"steps": [{"respond": {"text": "A2"}}, {"respond": {"text": "A3"}}]})
                self.assertEqual([(r["id"], r["remaining"]) for r in out["rules"]], [("a", 2), ("b", 1)])
                r = await c.put("/_control/rules/a", json={"id": "zzz", "steps": [{"respond": {"text": "x"}}]})
                self.assertEqual(r.status_code, 400)
                out = await self._ctl(c, "put", "/_control/rules/c", json={"steps": [{"respond": {"text": "C"}}]})
                self.assertEqual([r["id"] for r in out["rules"]], ["a", "b", "c"])
                await self._ctl(c, "delete", "/_control/rules/c")
                out = await self._ctl(c, "delete", "/_control/rules/a")
                self.assertEqual([r["id"] for r in out["rules"]], ["b"])
                self.assertEqual((await c.delete("/_control/rules/zzz")).status_code, 404)
                out = await self._ctl(c, "post", "/_control/rules", json=[{"steps": [{"error": {"status": 500}}]}])
                self.assertTrue(out["rules"][1]["id"].startswith("rule_"))
                await self._ctl(c, "delete", "/_control/rules")
                self.assertEqual((await self._ctl(c, "get", "/_control/rules"))["rules"], [])
                bad = [
                    ({"steps": [{"respond": {"text": "x"}}], "match": {"nope": 1}}, "unknown key"),
                    ({"steps": [{"respond": {"text": "x"}}], "match": {"last_user_text": "("}}, "invalid regex"),
                    ({"steps": [{"respond": {"text": "x"}}], "match": {"provider": "azure"}}, "provider"),
                    ({"steps": [{"respond": {"text": "x"}}], "match": {"turn": 0}}, "turn"),
                    ({"steps": [{"respond": {"text": "x"}}], "match": {"tools": "get_weather"}}, "tools"),
                    ({"steps": []}, "non-empty"),
                    ({"steps": [{"respond": {"text": "x"}, "error": {"status": 1}}]}, "exactly one"),
                    ({"steps": [{"respond": {"text": "x"}, "bogus": 1}]}, "unknown key"),
                    ({"steps": [{"respond": {"content": "nope"}}]}, "content must be"),
                    ({"steps": [{"error": {"status": 99}}]}, "status must be"),
                    ({"steps": [{"error": {"status": 500, "headers": {"content-length": "1"}}}]}, "framing"),
                    ({"steps": [{"respond": {"text": "x"}, "delay_ms": -1}]}, "delay_ms"),
                    ({"steps": [{"respond": {"text": "x"}}], "repeat": "yes"}, "repeat"),
                    ({"steps": [{"respond": {"text": "x"}}], "id": ""}, "rule.id"),
                    ({"steps": [{"respond": {"text": "x"}}], "extra": 1}, "unknown key"),
                ]
                for rule, needle in bad:
                    r = await c.post("/_control/rules", json=rule)
                    self.assertEqual(r.status_code, 400, rule)
                    self.assertIn(needle, r.json()["error"], rule)
                r = await c.post("/_control/rules", json={"rules": [
                    {"id": "dup", "steps": [{"respond": {"text": "x"}}]},
                    {"id": "dup", "steps": [{"respond": {"text": "y"}}]}]})
                self.assertIn("duplicate", r.json()["error"])
                self.assertEqual((await c.post("/_control/rules", content=b"{")).status_code, 400)
                self.assertEqual((await c.put("/_control/rules", json={"rules": "x"})).status_code, 400)
                # nothing of the above stuck
                self.assertEqual((await self._ctl(c, "get", "/_control/rules"))["rules"], [])
        _run(go())


class TestRuleEdges(_Base):
    def test_repeat_rule_hands_out_fresh_ids_and_stays_as_posted(self) -> None:
        async def go():
            async with await self._client() as c:
                step = {"respond": {"content": [{"type": "tool_use", "name": "f", "input": {}},
                                                {"type": "thinking", "thinking": "hm"}]}}
                await self._ctl(c, "post", "/_control/rules", json={"id": "r", "steps": [step], "repeat": True})
                ids, sigs = [], []
                for _ in range(2):
                    r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                    blocks = r.json()["content"]
                    sigs.append(next(b["signature"] for b in blocks if b["type"] == "thinking"))
                    ids.append(next(b["id"] for b in blocks if b["type"] == "tool_use"))
                self.assertNotEqual(ids[0], ids[1])
                self.assertNotEqual(sigs[0], sigs[1])
                view = await self._ctl(c, "get", "/_control/rules")
                self.assertEqual(view["rules"][0]["steps"], [step])
                await self._ctl(c, "post", "/_control/config", json={
                    "on_unmatched": "default", "default_response": {"content": [{"type": "tool_use", "name": "g", "input": {}}]}})
                await self._ctl(c, "delete", "/_control/rules")
                a = (await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["id"]
                b = (await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["id"]
                self.assertNotEqual(a, b)
                cfg = (await self._ctl(c, "get", "/_control/config"))["config"]
                self.assertNotIn("id", cfg["default_response"]["content"][0])
        _run(go())

    def test_rule_put_keeps_consumption_when_steps_are_extended(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "put", "/_control/rules/r", json={"steps": [{"respond": {"text": "one"}}]})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["text"], "one")
                out = await self._ctl(c, "put", "/_control/rules/r", json={"steps": [{"respond": {"text": "one"}}, {"respond": {"text": "two"}}]})
                self.assertEqual((out["rules"][0]["consumed"], out["rules"][0]["remaining"]), (1, 1))
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["text"], "two")
                # a different first step is a new script: counters restart
                out = await self._ctl(c, "put", "/_control/rules/r", json={"steps": [{"respond": {"text": "fresh"}}]})
                self.assertEqual(out["rules"][0]["consumed"], 0)
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["text"], "fresh")
        _run(go())

    def test_extending_a_repeating_rule_continues_with_the_new_steps(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "put", "/_control/rules/r", json={"steps": [{"respond": {"text": "A"}}], "repeat": True})
                for _ in range(5):
                    self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["text"], "A")
                await self._ctl(c, "put", "/_control/rules/r", json={
                    "steps": [{"respond": {"text": "A"}}, {"respond": {"text": "B"}}, {"respond": {"text": "C"}}], "repeat": True})
                got = [(await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["text"] for _ in range(3)]
                self.assertEqual(got, ["B", "C", "C"])
                out = await self._ctl(c, "put", "/_control/rules/r", json={
                    "steps": [{"respond": {"text": "A"}}, {"respond": {"text": "B"}}, {"respond": {"text": "C"}}, {"respond": {"text": "D"}}]})
                self.assertEqual((out["rules"][0]["remaining"], out["rules"][0]["exhausted"]), (1, False))
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["text"], "D")
        _run(go())

    def test_text_shorthand_next_to_an_empty_content(self) -> None:
        """What a client generated from the published RespondBody sends."""
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/respond", json={"text": "hello", "content": []})
                self.assertEqual((await t).json()["content"][0]["text"], "hello")
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ruled", "content": []}}]})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["text"], "ruled")
                await self._ctl(c, "post", "/_control/config", json={
                    "on_unmatched": "default", "default_response": {"text": "dflt", "content": []}})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["text"], "dflt")
                # `text: null` is an unset shorthand: an empty answer, not a 400
                await self._ctl(c, "post", "/_control/config", json={"on_unmatched": None})
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/respond", json={"text": None, "content": []})
                self.assertEqual((await t).json()["content"], [])
                t = await self._pending(c, "/v1/messages", _MSG)
                self.assertEqual((await c.post("/_control/respond", json={"text": 5, "content": []})).status_code, 400)
                await self._ctl(c, "post", "/_control/auto", json={"text": "ok"})
                await t
        _run(go())

    def test_nulls_from_generated_clients_are_ignored(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={
                    "id": None, "repeat": None,
                    "match": {"provider": None, "model": "claude-*", "turn": None},
                    "steps": [{"respond": {"text": "x", "stop_reason": None,
                                           "usage": {"input_tokens": 5, "output_tokens": None}},
                               "error": None, "delay_ms": None}]})
                await self._ctl(c, "post", "/_control/rules", json={"match": None, "steps": [{"respond": {"text": "later"}}]})
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.json()["usage"]["input_tokens"], 5)
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["content"][0]["text"], "later")
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/respond", json={"text": "y", "usage": {"input_tokens": None}})
                self.assertGreater((await t).json()["usage"]["input_tokens"], 5)
        _run(go())

    def test_error_step_fails_a_stream_mid_way(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [
                    {"error": {"status": 529, "after_events": 3,
                               "content": [{"type": "text", "text": "partial answer"}]}, "chunk_delay_ms": 10}]})
                r = await self._no_pending_reply(c, "/v1/messages", {**_MSG, "stream": True})
                self.assertEqual(r.status_code, 200)
                names = [n for n, _ in _sse_events(r.text)]
                self.assertEqual(names[:2], ["message_start", "ping"])
                self.assertEqual(names[-1], "error")
                self.assertNotIn("message_stop", names)
                self.assertEqual(_sse_events(r.text)[-1][1]["error"]["type"], "overloaded_error")
                hist = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(hist[-1]["injected_error"]["after_events"], 3)
                self.assertEqual(hist[-1]["harness"]["source"], "rule")
        _run(go())


class TestRulesAndBatches(_Base):
    def test_failed_batch_create_consumes_no_steps(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"id": "once", "steps": [{"respond": {"text": "scripted"}}]})
                r = await c.post("/v1/messages/batches", json={"requests": [
                    {"custom_id": "ok", "params": {**_MSG}},
                    {"custom_id": "bad", "params": {**_MSG, "messages": 123}}]})
                self.assertEqual(r.status_code, 400, r.text)
                self.assertIn("requests[1]", r.json()["error"]["message"])
                view = await self._ctl(c, "get", "/_control/rules")
                self.assertEqual(view["unconsumed"], ["once"])
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.json()["content"][0]["text"], "scripted")
        _run(go())

    def test_rules_answer_batch_entries_outside_the_limiter(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"rpm": 1, "otpm": 1}, "pending_timeout_s": 0.2})
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "batched " * 20}}], "repeat": True})
                r = await c.post("/v1/messages/batches", json={"requests": [
                    {"custom_id": "a", "params": {**_MSG}}, {"custom_id": "b", "params": {**_MSG}}]})
                self.assertEqual(r.status_code, 200, r.text)
                bid = r.json()["id"]
                for _ in range(100):
                    b = (await c.get(f"/v1/messages/batches/{bid}")).json()
                    if b["processing_status"] == "ended":
                        break
                    await asyncio.sleep(0.02)
                self.assertEqual(b["request_counts"]["succeeded"], 2)
                win = (await self._ctl(c, "get", "/_control/config"))["rate_limit_window"]
                self.assertEqual((win["window_requests"], win["window_output_tokens"]), (0, 0))
        _run(go())


class TestUnmatchedPolicyAndConfig(_Base):
    def test_default_and_error_policies(self) -> None:
        async def go():
            async with await self._client() as c:
                out = await self._ctl(c, "post", "/_control/config",
                                      json={"on_unmatched": "default", "default_response": "canned"})
                self.assertEqual(out["config"]["on_unmatched"], "default")
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.json()["content"][0]["text"], "canned")
                hist = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(hist[-1]["harness"], {"source": "default"})
                # a rule still wins over the default
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ruled"}}]})
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.json()["content"][0]["text"], "ruled")
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.json()["content"][0]["text"], "canned")
                await self._ctl(c, "post", "/_control/config", json={
                    "on_unmatched": "error",
                    "unmatched_error": {"status": 503, "message": "nothing scripted"}})
                r = await self._no_pending_reply(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.status_code, 503)
                self.assertIn("nothing scripted", r.json()["error"]["message"])
                hist = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(hist[-1]["harness"], {"source": "unmatched"})
                # `default` without a default_response gets a placeholder rather than hanging
                await self._ctl(c, "post", "/_control/config", json={"on_unmatched": "default",
                                                                     "default_response": None})
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertIn("default response", r.json()["content"][0]["text"])
                # config survives clear; rules do not
                await self._ctl(c, "post", "/_control/clear")
                cfg = (await self._ctl(c, "get", "/_control/config"))["config"]
                self.assertEqual(cfg["on_unmatched"], "default")
                await self._ctl(c, "post", "/_control/config", json={"on_unmatched": None})
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/clear")
                await asyncio.gather(t, return_exceptions=True)
        _run(go())

    def test_config_validation(self) -> None:
        async def go():
            async with await self._client() as c:
                bad = [
                    ({"bogus": 1}, "unknown config key"),
                    ({"on_unmatched": "maybe"}, "on_unmatched"),
                    ({"pending_timeout_s": -1}, "pending_timeout_s"),
                    ({"pending_timeout_s": "5"}, "pending_timeout_s"),
                    ({"default_response": {"content": "x"}}, "default_response"),
                    ({"unmatched_error": {"status": 42}}, "unmatched_error"),
                    ({"timeout_error": "x"}, "timeout_error"),
                    ({"latency": {"delay_ms": -5}}, "delay_ms"),
                    ({"latency": {"fps": 1}}, "latency"),
                    ({"rate_limit": {"rpm": 0}}, "rate_limit.rpm"),
                    ({"rate_limit": {"tps": 1}}, "rate_limit"),
                    ({"rate_limit": {}}, "rate_limit"),
                    ({"seed": "abc"}, "seed"),
                ]
                for body, needle in bad:
                    r = await c.post("/_control/config", json=body)
                    self.assertEqual(r.status_code, 400, body)
                    self.assertIn(needle, r.json()["error"], body)
                self.assertEqual((await c.put("/_control/config", content=b"[")).status_code, 400)
                cfg = (await self._ctl(c, "get", "/_control/config"))["config"]
                self.assertEqual(cfg["on_unmatched"], "pending")
                self.assertIsNone(cfg["pending_timeout_s"])
                self.assertEqual(cfg["timeout_error"]["status"], 504)
                out = await self._ctl(c, "put", "/_control/config", json={
                    "pending_timeout_s": 5, "latency": {"delay_ms": 1, "jitter_ms": 2},
                    "rate_limit": {"rpm": 10, "itpm": None}, "seed": 7})
                cfg = out["config"]
                self.assertEqual(cfg["pending_timeout_s"], 5.0)
                self.assertEqual(cfg["latency"], {"delay_ms": 1, "jitter_ms": 2})
                self.assertEqual(cfg["rate_limit"], {"rpm": 10})
                self.assertEqual(out["rate_limit_window"]["window_requests"], 0)
                out = await self._ctl(c, "post", "/_control/config", json={
                    "pending_timeout_s": None, "latency": None, "rate_limit": None})
                self.assertEqual((out["config"]["pending_timeout_s"], out["config"]["latency"],
                                  out["config"]["rate_limit"]), (None, {}, None))
        _run(go())


class TestConfigEdges(_Base):
    def test_clear_with_config_restores_defaults(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"rpm": 1}, "pending_timeout_s": 3, "seed": 4})
                await self._ctl(c, "post", "/_control/clear")
                cfg = (await self._ctl(c, "get", "/_control/config"))["config"]
                self.assertEqual(cfg["rate_limit"], {"rpm": 1})
                await self._ctl(c, "post", "/_control/clear", json={"config": True})
                cfg = (await self._ctl(c, "get", "/_control/config"))["config"]
                self.assertEqual((cfg["rate_limit"], cfg["pending_timeout_s"], cfg["seed"]), (None, None, None))
                self.assertEqual((await c.post("/_control/clear", json={"config": "yes"})).status_code, 400)
                self.assertEqual((await c.post("/_control/clear", content=b"nope")).status_code, 400)
        _run(go())

    def test_huge_numbers_are_400_not_500(self) -> None:
        async def go():
            async with await self._client() as c:
                big = b"9" * 400
                r = await c.post("/_control/config", content=b'{"pending_timeout_s": ' + big + b"}")
                self.assertEqual(r.status_code, 400)
                r = await c.post("/_control/clock/advance", content=b'{"seconds": ' + big + b"}")
                self.assertEqual(r.status_code, 400)
                r = await c.post("/_control/config", content=b'{"pending_timeout_s": 1e400}')
                self.assertEqual(r.status_code, 400)
        _run(go())

    def test_auto_reads_text_only(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG)
                # `content` and unknown keys are ignored, as they always were
                await self._ctl(c, "post", "/_control/auto", json={
                    "text": "from text", "content": [{"type": "text", "text": "not this"}], "stop_reason": 5})
                self.assertEqual((await t).json()["content"][0]["text"], "from text")
                t = await self._pending(c, "/v1/messages", _MSG)
                r = await c.post("/_control/auto", json={"text": 123})
                self.assertEqual(r.status_code, 400)
                await self._ctl(c, "post", "/_control/auto", json={"text": "ok", "delay_ms": 0})
                self.assertEqual((await t).status_code, 200)
        _run(go())


class TestPendingLifetime(_Base):
    def test_pending_timeout_answers_with_timeout_error(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/config", json={"pending_timeout_s": 0.3})
                t0 = time.monotonic()
                t = asyncio.create_task(c.post("/v1/messages", json=_MSG, timeout=10))
                await asyncio.sleep(0.05)
                listing = (await c.get("/_control/pending")).json()
                self.assertTrue(listing["has_pending"])
                item = listing["pending"][0]
                self.assertIsNotNone(item["deadline"])
                self.assertLessEqual(item["timeout_in_seconds"], 0.3)
                r = await t
                self.assertGreaterEqual(time.monotonic() - t0, 0.3)
                self.assertEqual(r.status_code, 504)
                self.assertEqual(r.json()["error"]["type"], "api_error")
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
                hist = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(hist[-1]["harness"], {"source": "timeout"})
                self.assertEqual(hist[-1]["injected_error"]["status"], 504)
                # configurable error shape, applied on the OpenAI route as well
                await self._ctl(c, "post", "/_control/config", json={
                    "timeout_error": {"status": 503, "message": "responder asleep"}})
                r = await c.post("/v1/chat/completions", json={
                    "model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]}, timeout=10)
                self.assertEqual(r.status_code, 503)
                self.assertIn("responder asleep", r.json()["error"]["message"])
        _run(go())

    def test_clock_advance_expires_a_pending(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/config", json={"pending_timeout_s": 600})
                t = await self._pending(c, "/v1/messages", _MSG)
                await asyncio.sleep(0.3)
                self.assertFalse(t.done())
                await self._ctl(c, "post", "/_control/clock/advance", json={"seconds": 601})
                r = await asyncio.wait_for(t, 5)
                self.assertEqual(r.status_code, 504)
                # a responder's answer during the wait is what the client gets
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/respond", json={"text": "in time"})
                self.assertEqual((await t).json()["content"][0]["text"], "in time")
        _run(go())

    def test_deadline_is_fixed_at_registration(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG)  # no timeout configured
                await self._ctl(c, "post", "/_control/config", json={"pending_timeout_s": 0.2})
                listing = (await c.get("/_control/pending")).json()
                self.assertIsNone(listing["pending"][0]["deadline"])
                await asyncio.sleep(0.6)
                self.assertFalse(t.done())  # the earlier pending keeps its "forever"
                t2 = asyncio.create_task(c.post("/v1/messages", json=_MSG, timeout=10))
                await asyncio.sleep(0.05)
                listing = (await c.get("/_control/pending")).json()
                deadlines = {p["pending_id"]: p["deadline"] for p in listing["pending"]}
                self.assertEqual(sum(d is None for d in deadlines.values()), 1)
                # removing the timeout does not rescue a pending registered under it
                await self._ctl(c, "post", "/_control/config", json={"pending_timeout_s": None})
                self.assertEqual((await asyncio.wait_for(t2, 5)).status_code, 504)
                await self._ctl(c, "post", "/_control/respond", json={"text": "late but fine"})
                self.assertEqual((await t).json()["content"][0]["text"], "late but fine")
        _run(go())

    def test_cancelled_handler_during_delay_drops_its_pending(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG)
                pid = (await c.get("/_control/pending")).json()["pending"][0]["pending_id"]
                await self._ctl(c, "post", "/_control/respond", json={"text": "slow", "delay_ms": 2000})
                await asyncio.sleep(0.1)
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
                await asyncio.sleep(0.05)
                self.assertNotIn(pid, self.mod.state.pending)
                self.assertEqual((await c.get("/_control/history")).json()["history"], [])
        _run(go())

    def test_clear_interrupts_a_long_delay(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/respond", json={"text": "slow", "delay_ms": 5000})
                await asyncio.sleep(0.1)
                t0 = time.monotonic()
                await self._ctl(c, "post", "/_control/clear")
                r = await asyncio.wait_for(t, 5)
                self.assertLess(time.monotonic() - t0, 1.5)
                self.assertEqual(r.status_code, 529)
        _run(go())

    def test_answer_injected_before_a_clear_is_not_delivered(self) -> None:
        async def go():
            fs = self.mod
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"rpm": 5}})
                snapshot, fut = await fs.register_request("anthropic", "claude-sonnet-5", _MSG, False)
                task = asyncio.create_task(fs.await_resolution(snapshot, fut))
                await asyncio.sleep(0)  # the handler is now waiting on the future
                fut.set_result({"content": [{"type": "text", "text": "stale"}]})
                # what /_control/clear does, in the same tick, before the handler resumes
                async with fs.state.lock:
                    fs.state.pending.clear()
                    fs.state.history.clear()
                    fs.state.clear_generation += 1
                    fs.harness.reset()
                result = await task
                self.assertEqual(result["kind"], "cleared")
                self.assertEqual(fs.state.history, [])
                self.assertEqual((await self._ctl(c, "get", "/_control/config"))["rate_limit_window"]["window_requests"], 0)
        _run(go())

    def test_answer_resolved_before_a_clear_is_not_delivered_to_a_late_waiter(self) -> None:
        async def go():
            fs = self.mod
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "stale"}, "delay_ms": 5000}]})
                snapshot, fut = await fs.register_request("anthropic", "claude-sonnet-5", _MSG, False)
                self.assertTrue(fut.done())  # answered by the rule at registration
                await self._ctl(c, "post", "/_control/clear")
                t0 = time.monotonic()
                result = await fs.await_resolution(snapshot, fut)  # a collector starting late
                self.assertEqual(result["kind"], "cleared")
                self.assertLess(time.monotonic() - t0, 1.0)
                self.assertEqual(fs.state.history, [])
        _run(go())

    def test_disconnect_during_delay_records_the_answer(self) -> None:
        async def go():
            from starlette.requests import Request
            fs = self.mod
            snapshot, fut = await fs.register_request("anthropic", "claude-sonnet-5", _MSG, False)
            fut.set_result({"content": [{"type": "text", "text": "late"}], "_latency": {"delay_ms": 5000}})
            hung_up = False

            async def receive() -> dict[str, Any]:
                if hung_up:
                    return {"type": "http.disconnect"}
                await asyncio.sleep(3600)
                return {}

            req = Request({"type": "http", "method": "POST", "path": "/v1/messages",
                           "headers": [], "query_string": b""}, receive)
            task = asyncio.create_task(fs.await_resolution(snapshot, fut, request=req))
            await asyncio.sleep(0.3)
            hung_up = True
            t0 = time.monotonic()
            result = await asyncio.wait_for(task, 3)
            # the wait ends at once, and the answer that was given is recorded anyway
            self.assertLess(time.monotonic() - t0, 1.0)
            self.assertEqual(result["kind"], "ok")
            self.assertNotIn(snapshot["pending_id"], fs.state.pending)
            self.assertEqual(len(fs.state.history), 1)
            self.assertEqual(fs.state.history[0]["response_blocks"][0]["text"], "late")
        _run(go())

    def test_disconnected_client_drops_its_pending(self) -> None:
        async def go():
            from starlette.requests import Request
            fs = self.mod
            snapshot, fut = await fs.register_request("anthropic", "claude-sonnet-5", _MSG, False)
            self.assertIn(snapshot["pending_id"], fs.state.pending)

            async def receive() -> dict[str, Any]:
                return {"type": "http.disconnect"}

            req = Request({"type": "http", "method": "POST", "path": "/v1/messages",
                           "headers": [], "query_string": b""}, receive)
            result = await asyncio.wait_for(fs.await_resolution(snapshot, fut, request=req), 5)
            self.assertEqual(result["kind"], "cleared")
            self.assertIn("disconnected", result["detail"])
            self.assertNotIn(snapshot["pending_id"], fs.state.pending)
            self.assertEqual(fs.state.history, [])
            # a connected client keeps waiting (its receive never reports a disconnect)
            snapshot, fut = await fs.register_request("anthropic", "claude-sonnet-5", _MSG, False)
            never = asyncio.get_running_loop().create_future()

            async def receive_blocking() -> dict[str, Any]:
                return await never

            req = Request({"type": "http", "method": "POST", "path": "/v1/messages",
                           "headers": [], "query_string": b""}, receive_blocking)
            task = asyncio.create_task(fs.await_resolution(snapshot, fut, request=req))
            await asyncio.sleep(0.6)
            self.assertFalse(task.done())
            fut.set_result({"content": [{"type": "text", "text": "ok"}]})
            self.assertEqual((await task)["kind"], "ok")
        _run(go())


class TestLatency(_Base):
    def test_delay_on_injection_and_config_default(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG)
                t0 = time.monotonic()
                await self._ctl(c, "post", "/_control/respond", json={"text": "slow", "delay_ms": 300})
                r = await t
                self.assertGreaterEqual(time.monotonic() - t0, 0.3)
                self.assertEqual(r.json()["content"][0]["text"], "slow")
                r = await c.post("/_control/respond", json={"text": "x", "delay_ms": -1})
                self.assertEqual(r.status_code, 400)
                self.assertIn("delay_ms", r.json()["error"])
                r = await c.post("/_control/error", json={"status": 500, "ttfb_ms": "soon"})
                self.assertEqual(r.status_code, 400)
                await self._ctl(c, "post", "/_control/config", json={"latency": {"delay_ms": 250}})
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"error": {"status": 500}}]})
                t0 = time.monotonic()
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.status_code, 500)
                self.assertGreaterEqual(time.monotonic() - t0, 0.25)
                # a per-answer value overrides the default (2 s default vs an explicit 0)
                await self._ctl(c, "post", "/_control/config", json={"latency": {"delay_ms": 2000}})
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "fast"}, "delay_ms": 0}]})
                t0 = time.monotonic()
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.json()["content"][0]["text"], "fast")
                self.assertLess(time.monotonic() - t0, 1.5)
        _run(go())

    def test_stream_pacing(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [
                    {"respond": {"text": "abc"}, "ttfb_ms": 200, "chunk_delay_ms": 50}]})
                t0 = time.monotonic()
                r = await self._no_pending_reply(c, "/v1/messages", {**_MSG, "stream": True})
                elapsed = time.monotonic() - t0
                events = _sse_events(r.text)
                self.assertEqual(events[-1][0], "message_stop")
                # ttfb once, then a chunk delay before each remaining frame (ping included)
                self.assertGreaterEqual(elapsed, 0.2 + 0.05 * (len(events) - 1))
                # the OpenAI stream honours the same knobs
                await self._ctl(c, "post", "/_control/rules", json={"steps": [
                    {"respond": {"text": "hi"}, "ttfb_ms": 150}]})
                t0 = time.monotonic()
                r = await self._no_pending_reply(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.status_code, 200)
                self.assertGreaterEqual(time.monotonic() - t0, 0.15)
                self.assertIn("[DONE]", r.text)
        _run(go())

    def test_bedrock_stream_pacing_is_reflected_in_metrics(self) -> None:
        async def go():
            from puppetllm.providers import eventstream
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [
                    {"respond": {"text": "abc"}, "ttfb_ms": 200, "chunk_delay_ms": 20}]})
                t0 = time.monotonic()
                r = await self._no_pending_reply(
                    c, "/model/anthropic.claude-sonnet-4-5-20250929-v1:0/invoke-with-response-stream",
                    {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                     "messages": [{"role": "user", "content": "hi"}]})
                elapsed = time.monotonic() - t0
                self.assertEqual(r.status_code, 200, r.text)
                msgs = eventstream.decode_messages(r.content)
                self.assertGreaterEqual(elapsed, 0.2 + 0.02 * (len(msgs) - 1))
                self.assertEqual(msgs[-1]["type"], "message_stop")
                metrics = msgs[-1]["amazon-bedrock-invocationMetrics"]
                self.assertGreaterEqual(metrics["firstByteLatency"], 200)
                self.assertGreaterEqual(metrics["invocationLatency"], 200 + 20 * (len(msgs) - 1))
        _run(go())

    def test_clear_during_delay_reports_cleared(self) -> None:
        async def go():
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", _MSG)
                await self._ctl(c, "post", "/_control/respond", json={"text": "late", "delay_ms": 400})
                await asyncio.sleep(0.05)
                await self._ctl(c, "post", "/_control/clear")
                r = await t
                self.assertEqual(r.status_code, 529)
                self.assertEqual((await c.get("/_control/history")).json()["history"], [])
        _run(go())

    def test_jitter_is_seeded(self) -> None:
        from puppetllm.harness import Harness
        h = Harness()
        h.config.seed = 42
        h.reseed()
        a = [h.latency_for({"delay_ms": 10, "jitter_ms": 1000})["delay_ms"] for _ in range(5)]
        h.reseed()
        b = [h.latency_for({"delay_ms": 10, "jitter_ms": 1000})["delay_ms"] for _ in range(5)]
        self.assertEqual(a, b)
        self.assertTrue(all(10 <= x <= 1010 for x in a))
        self.assertGreater(len(set(a)), 1)
        h.config.latency = {"ttfb_ms": 5, "jitter_ms": 0}
        self.assertEqual(h.latency_for(None), {"delay_ms": 0, "ttfb_ms": 5, "chunk_delay_ms": 0})
        self.assertEqual(h.latency_for({"ttfb_ms": 9})["ttfb_ms"], 9)


class TestRateLimit(_Base):
    def test_rpm_with_vendor_headers_and_window_expiry(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"rpm": 2}})
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                for _ in range(2):
                    r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                    self.assertEqual(r.status_code, 200)
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.json()["error"]["type"], "rate_limit_error")
                self.assertTrue(int(r.headers["retry-after"]) >= 1)
                self.assertEqual(r.headers["anthropic-ratelimit-requests-limit"], "2")
                self.assertEqual(r.headers["anthropic-ratelimit-requests-remaining"], "0")
                self.assertIn("anthropic-ratelimit-requests-reset", r.headers)
                hist = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(hist[-1]["harness"], {"source": "rate_limit"})
                r = await self._no_pending_reply(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.headers["x-ratelimit-limit-requests"], "2")
                self.assertEqual(r.headers["x-ratelimit-reset-requests"], r.headers["retry-after"] + "s")
                self.assertNotIn("anthropic-ratelimit-requests-limit", r.headers)
                r = await self._no_pending_reply(
                    c, "/model/anthropic.claude-sonnet-4-5-20250929-v1:0/invoke",
                    {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                     "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.headers["x-amzn-ErrorType"], "ThrottlingException")
                win = (await self._ctl(c, "get", "/_control/config"))["rate_limit_window"]
                self.assertEqual(win["window_requests"], 2)  # refused requests consume nothing
                # the window slides with the fake clock
                await self._ctl(c, "post", "/_control/clock/advance", json={"seconds": 61})
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.status_code, 200)
        _run(go())

    def test_token_budgets(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok " * 50}}], "repeat": True})
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                usage = r.json()["usage"]
                await self._ctl(c, "post", "/_control/config", json={
                    "rate_limit": {"itpm": usage["input_tokens"] * 2 - 1}})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.status_code, 429)
                self.assertIn("input-tokens", r.json()["error"]["message"])
                self.assertIn("anthropic-ratelimit-input-tokens-limit", r.headers)
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"otpm": 1}})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.status_code, 429)
                self.assertIn("output-tokens", r.json()["error"]["message"])
                await self._ctl(c, "post", "/_control/clear")  # clear empties the window (rules too)
                win = (await self._ctl(c, "get", "/_control/config"))["rate_limit_window"]
                self.assertEqual((win["otpm"], win["window_requests"], win["window_output_tokens"]), (1, 0, 0))
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}]})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
        _run(go())


class TestRateLimitEdges(_Base):
    def test_retry_after_is_achievable(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"rpm": 1}})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
                for _ in range(3):  # refused requests do not push the reset further out
                    r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                    self.assertEqual(r.status_code, 429)
                    self.assertIn("requests 1/1", r.json()["error"]["message"])
                wait = int(r.headers["retry-after"])
                self.assertLessEqual(wait, 61)
                await self._ctl(c, "post", "/_control/clock/advance", json={"seconds": wait})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
                # rpm 2 with the two admissions 30 s apart: retry-after points at the first
                await self._ctl(c, "post", "/_control/clear")
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"rpm": 2}})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
                await self._ctl(c, "post", "/_control/clock/advance", json={"seconds": 30})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.status_code, 429)
                self.assertLessEqual(int(r.headers["retry-after"]), 31)
                await self._ctl(c, "post", "/_control/clock/advance", json={"seconds": int(r.headers["retry-after"])})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
        _run(go())

    def test_quota_headers_are_per_dimension(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                first = (await self._no_pending_reply(c, "/v1/messages", _MSG)).json()["usage"]["input_tokens"]
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"rpm": 100, "itpm": first * 2 - 1}})
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.headers["anthropic-ratelimit-requests-remaining"], "99")
                self.assertEqual(r.headers["anthropic-ratelimit-input-tokens-remaining"], str(first - 1))
                self.assertEqual(r.headers["anthropic-ratelimit-input-tokens-limit"], str(first * 2 - 1))
                self.assertNotIn("anthropic-ratelimit-output-tokens-limit", r.headers)
                r = await self._no_pending_reply(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]})
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.headers["x-ratelimit-remaining-requests"], "99")
                self.assertEqual(r.json()["error"]["code"], "rate_limit_exceeded")
        _run(go())

    def test_output_is_charged_when_produced_and_is_not_a_request(self) -> None:
        from puppetllm.harness import Harness
        h = Harness()
        h.config.rate_limit = {"rpm": 1, "otpm": 100}
        # output charged with no admission of its own (never a request)
        h.note_output(40, 1000.0)
        self.assertEqual(h.rate_limit_snapshot(1000.0)["window_requests"], 0)
        self.assertEqual(h.rate_limit_snapshot(1000.0)["window_output_tokens"], 40)
        self.assertIsNone(h.throttle({"pending_id": "a", "provider": "anthropic", "input_tokens_total": 1}, 1001.0))
        # output produced 59 s after admission stays in the window for a full minute from
        # then, and a slow request's output is charged whether or not anything pruned in between
        h.config.rate_limit = {"otpm": 100}
        h._window.clear()
        self.assertIsNone(h.throttle({"pending_id": "slow", "provider": "anthropic", "input_tokens_total": 1}, 2000.0))
        h.note_output(500, 2059.0)
        self.assertEqual(h.rate_limit_snapshot(2061.0)["window_output_tokens"], 500)
        self.assertEqual(h.rate_limit_snapshot(2061.0)["window_requests"], 0)
        self.assertEqual(h.throttle({"pending_id": "n", "provider": "anthropic", "input_tokens_total": 1}, 2063.0)["status"], 429)
        self.assertEqual(h.rate_limit_snapshot(2120.0)["window_output_tokens"], 0)
        # a new budget starts from an empty window, so a completion after the change counts fresh
        h.config.rate_limit = {"otpm": 10}
        h._window.clear()
        h.note_output(500, 2130.0)
        self.assertEqual(h.throttle({"pending_id": "m", "provider": "anthropic", "input_tokens_total": 1}, 2131.0)["status"], 429)

    def test_openai_token_headers_follow_the_exhausted_quota(self) -> None:
        from puppetllm.harness import Harness
        h = Harness()
        h.config.rate_limit = {"itpm": 1000, "otpm": 1}
        self.assertIsNone(h.throttle({"pending_id": "a", "provider": "openai", "input_tokens_total": 10}, 1000.0))
        h.note_output(10, 1001.0)
        r = h.throttle({"pending_id": "b", "provider": "openai", "input_tokens_total": 10}, 1002.0)
        self.assertEqual(r["status"], 429)
        self.assertEqual(r["headers"]["x-ratelimit-limit-tokens"], "1")
        self.assertEqual(r["headers"]["x-ratelimit-remaining-tokens"], "0")
        self.assertEqual(r["headers"]["x-ratelimit-reset-tokens"], r["headers"]["retry-after"] + "s")
        self.assertEqual(r["_latency"]["delay_ms"], 0)
        # a request larger than the whole budget is refused with the full minute
        h.config.rate_limit = {"itpm": 1}
        h._window.clear()
        r = h.throttle({"pending_id": "c", "provider": "anthropic", "input_tokens_total": 2}, 1100.0)
        self.assertEqual(r["headers"]["retry-after"], "60")
        self.assertEqual(r["headers"]["anthropic-ratelimit-tokens-limit"], "1")

    def test_throttled_request_does_not_warm_the_cache(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                await self._ctl(c, "post", "/_control/config", json={"rate_limit": {"rpm": 1}})
                cacheable = {**_MSG, "system": [{"type": "text", "text": "long system prompt " * 40,
                                                 "cache_control": {"type": "ephemeral"}}]}
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", _MSG)).status_code, 200)
                self.assertEqual((await self._no_pending_reply(c, "/v1/messages", cacheable)).status_code, 429)
                self.assertEqual((await c.get("/_control/cache")).json()["entries"], [])
                await self._ctl(c, "post", "/_control/clock/advance", json={"seconds": 61})
                usage = (await self._no_pending_reply(c, "/v1/messages", cacheable)).json()["usage"]
                self.assertGreater(usage["cache_creation_input_tokens"], 0)
                self.assertEqual(usage["cache_read_input_tokens"], 0)
                hist = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(hist[1]["cache"]["status"], "none")
        _run(go())

    def test_limiter_answers_are_not_delayed_by_config_latency(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/config", json={
                    "rate_limit": {"rpm": 1}, "latency": {"delay_ms": 1500}, "pending_timeout_s": 0.2})
                t = await self._pending(c, "/v1/messages", _MSG)  # admitted, times out
                t0 = time.monotonic()
                r = await asyncio.wait_for(t, 5)
                self.assertEqual(r.status_code, 504)
                self.assertLess(time.monotonic() - t0, 1.0)
                t0 = time.monotonic()
                r = await self._no_pending_reply(c, "/v1/messages", _MSG)  # refused at once
                self.assertEqual(r.status_code, 429)
                self.assertLess(time.monotonic() - t0, 1.0)
                self.assertIn("anthropic-ratelimit-requests-reset", r.headers)
        _run(go())


class TestClockAndCache(_Base):
    def setUp(self) -> None:
        # a 100 s cache TTL, so a 150 s clock advance is past the 5m bucket
        previous = os.environ.get("PUPPETLLM_CACHE_TTL")
        os.environ["PUPPETLLM_CACHE_TTL"] = "100"
        self.mod = _import_fresh()
        if previous is None:
            self.addCleanup(os.environ.pop, "PUPPETLLM_CACHE_TTL", None)
        else:
            self.addCleanup(os.environ.__setitem__, "PUPPETLLM_CACHE_TTL", previous)

    def test_clock_advance_expires_the_prompt_cache(self) -> None:
        async def go():
            async with await self._client() as c:
                body = {**_MSG, "system": [{"type": "text", "text": "long system prompt " * 40,
                                            "cache_control": {"type": "ephemeral"}}]}
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "ok"}}], "repeat": True})
                r1 = (await self._no_pending_reply(c, "/v1/messages", body)).json()["usage"]
                r2 = (await self._no_pending_reply(c, "/v1/messages", body)).json()["usage"]
                self.assertGreater(r1["cache_creation_input_tokens"], 0)
                self.assertGreater(r2["cache_read_input_tokens"], 0)
                out = await self._ctl(c, "post", "/_control/clock/advance", json={"seconds": 150})
                self.assertEqual(out["clock_offset_seconds"], 150.0)
                entries = (await c.get("/_control/cache")).json()["entries"]
                self.assertTrue(entries and not any(e["alive"] for e in entries))
                r3 = (await self._no_pending_reply(c, "/v1/messages", body)).json()["usage"]
                self.assertGreater(r3["cache_creation_input_tokens"], 0)
                self.assertEqual(r3["cache_read_input_tokens"], 0)
                for bad in ({"seconds": -1}, {"seconds": "x"}, {}, {"seconds": 1e12}):
                    self.assertEqual((await c.post("/_control/clock/advance", json=bad)).status_code, 400)
                clock = (await c.get("/_control/clock")).json()
                self.assertGreater(clock["now"], time.time() + 100)
                await self._ctl(c, "post", "/_control/clear")
                self.assertEqual((await c.get("/_control/clock")).json()["clock_offset_seconds"], 0.0)
        _run(go())


class TestCountTokensAndModels(_Base):
    def test_count_tokens_matches_usage(self) -> None:
        async def go():
            async with await self._client() as c:
                body = {**_MSG, "system": "You are terse.", "tools": [_WEATHER_TOOL]}
                r = await c.post("/v1/messages/count_tokens", json={k: v for k, v in body.items() if k != "max_tokens"})
                self.assertEqual(r.status_code, 200, r.text)
                n = r.json()["input_tokens"]
                self.assertGreater(n, 0)
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
                self.assertEqual((await c.get("/_control/history")).json()["history"], [])
                t = await self._pending(c, "/v1/messages", body)
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok"})
                self.assertEqual((await t).json()["usage"]["input_tokens"], n)
                r = await c.post("/v1/messages/count_tokens", json={"model": "claude-sonnet-5"})
                self.assertEqual(r.status_code, 400)
                self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
                self.assertEqual((await c.post("/v1/messages/count_tokens", json={"messages": []})).status_code, 400)
                r = await c.post("/v1/messages/count_tokens", json={
                    "model": "claude-sonnet-5", "system": [{"type": "text", "text": "x",
                                                            "cache_control": {"type": "bogus"}}],
                    "messages": _MSG["messages"]})
                self.assertEqual(r.status_code, 400)
                # Bedrock CountTokens, both input forms
                inv = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                       "system": "You are terse.", "tools": [_WEATHER_TOOL], "messages": _MSG["messages"]}
                r = await c.post("/model/anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens",
                                 json={"input": {"invokeModel": {"body": json.dumps(inv)}}})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["inputTokens"], n)
                import base64
                r = await c.post("/model/anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens",
                                 json={"input": {"invokeModel": {"body": base64.b64encode(json.dumps(inv).encode()).decode()}}})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["inputTokens"], n)
                r = await c.post("/model/anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens",
                                 json={"input": {"converse": {
                                     "messages": [{"role": "user", "content": [{"text": "hello"}]}]}}})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertGreater(r.json()["inputTokens"], 0)
                for bad in ({"input": {}}, {"input": {"invokeModel": {"body": "{"}}},
                            {"input": {"invokeModel": {"body": "[]"}}},
                            {"input": {"converse": {"messages": "x"}}},
                            {"input": {"invokeModel": {"body": "{}"}, "converse": {}}}):
                    r = await c.post("/model/anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens", json=bad)
                    self.assertEqual(r.status_code, 400, bad)
                    self.assertEqual(r.headers["x-amzn-ErrorType"], "ValidationException")
        _run(go())

    def test_count_tokens_rejects_malformed_requests(self) -> None:
        async def go():
            async with await self._client() as c:
                bad = [{"model": "m", "messages": [123]},
                       {"model": "m", "messages": [{"role": "user", "content": 5}]},
                       {"model": "m", "messages": [], "tools": 1},
                       {"model": "m", "messages": [], "system": 5},
                       {"model": "m", "messages": _MSG["messages"], "max_tokens": 5},
                       {"model": "m", "messages": _MSG["messages"], "temperature": 0.5},
                       {"model": "m", "messages": _MSG["messages"], "stream": True}]
                for body in bad:
                    r = await c.post("/v1/messages/count_tokens", json=body)
                    self.assertEqual(r.status_code, 400, body)
                    self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
                # the request routes refuse the same shapes (instead of failing later)
                r = await c.post("/v1/messages", json={**_MSG, "tools": 1})
                self.assertEqual(r.status_code, 400)
                r = await c.post("/v1/messages", json={**_MSG, "messages": [123]})
                self.assertEqual(r.status_code, 400)
                r = await c.post("/v1/chat/completions", json={"model": "gpt-5.4", "messages": [{"role": "user", "content": "x"}], "tools": 1})
                self.assertEqual(r.status_code, 400)
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
        _run(go())

    def test_models_pagination(self) -> None:
        async def go():
            async with await self._client() as c:
                h = {"x-api-key": "k"}
                seen, after = [], None
                for _ in range(10):
                    params = {"limit": 3, **({"after_id": after} if after else {})}
                    page = (await c.get("/v1/models", params=params, headers=h)).json()
                    seen += [m["id"] for m in page["data"]]
                    if not page["has_more"]:
                        break
                    after = page["last_id"]
                from puppetllm import pricing
                self.assertEqual(seen, [m for m, _ in pricing.KNOWN_MODELS["anthropic"]])
                last = (await c.get("/v1/models", params={"limit": 3, "before_id": seen[4]}, headers=h)).json()
                self.assertEqual([m["id"] for m in last["data"]], seen[1:4])
                self.assertTrue(last["has_more"])
                for params in ({"limit": "abc"}, {"limit": 0}, {"limit": 1001}, {"limit": "1_0"},
                               {"limit": "\u00b2"}, {"limit": "9" * 40},
                               {"after_id": "nope"}, {"after_id": seen[0], "before_id": seen[3]}):
                    r = await c.get("/v1/models", params=params, headers=h)
                    self.assertEqual(r.status_code, 400, params)
                    self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
        _run(go())

    def test_models_in_both_dialects(self) -> None:
        async def go():
            async with await self._client() as c:
                r = await c.get("/v1/models", headers={"x-api-key": "k", "anthropic-version": "2023-06-01"})
                self.assertEqual(r.status_code, 200)
                data = r.json()
                self.assertFalse(data["has_more"])
                self.assertEqual(data["data"][0]["type"], "model")
                ids = [m["id"] for m in data["data"]]
                self.assertIn("claude-sonnet-5", ids)
                self.assertEqual(data["first_id"], ids[0])
                r = await c.get("/v1/models", params={"limit": 2}, headers={"x-api-key": "k"})
                self.assertEqual(len(r.json()["data"]), 2)
                self.assertTrue(r.json()["has_more"])
                r = await c.get("/v1/models", headers={"authorization": "Bearer sk-test"})
                self.assertEqual(r.json()["object"], "list")
                self.assertEqual(r.json()["data"][0]["object"], "model")
                self.assertIn("gpt-5.4", [m["id"] for m in r.json()["data"]])
                # the Anthropic SDK's OAuth flavour: Bearer plus anthropic-version → Anthropic
                r = await c.get("/v1/models", headers={"authorization": "Bearer t",
                                                       "anthropic-version": "2023-06-01"})
                self.assertIn("has_more", r.json())
                r = await c.get("/v1/models/claude-sonnet-5", headers={"x-api-key": "k"})
                self.assertEqual(r.json()["display_name"], "Claude Sonnet 5")
                r = await c.get("/v1/models/my-custom-model", headers={"x-api-key": "k"})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["display_name"], "my-custom-model")
                r = await c.get("/v1/models/gpt-5.4", headers={"authorization": "Bearer x"})
                self.assertEqual(r.json()["owned_by"], "puppetllm")
                # no header at all → Anthropic shape, and no pending was ever created
                self.assertIn("has_more", (await c.get("/v1/models")).json())
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
        _run(go())

    def test_openapi_declares_control_bodies(self) -> None:
        async def go():
            async with await self._client() as c:
                spec = (await c.get("/openapi.json")).json()
                from puppetllm import __version__
                self.assertEqual(spec["info"]["version"], __version__)
                for path in ("/_control/respond", "/_control/error", "/_control/auto",
                             "/_control/config", "/_control/rules", "/_control/clock/advance",
                             "/_control/clear", "/_control/batch/end", "/_control/batch/result",
                             "/v1/messages/count_tokens"):
                    ops = spec["paths"][path]
                    op = ops.get("post") or ops.get("put")
                    schema = op["requestBody"]["content"]["application/json"]["schema"]
                    self.assertTrue(schema.get("properties"), path)
                    self.assertNotIn("$defs", schema, path)
                respond = spec["paths"]["/_control/respond"]["post"]["requestBody"]["content"]["application/json"]["schema"]
                self.assertIn("delay_ms", respond["properties"])
                self.assertIn("content", respond["properties"])
                # every reference resolves from the document root
                refs: set[str] = set()

                def walk(o: Any) -> None:
                    if isinstance(o, dict):
                        if "$ref" in o:
                            refs.add(o["$ref"])
                        for v in o.values():
                            walk(v)
                    elif isinstance(o, list):
                        for v in o:
                            walk(v)
                walk(spec)
                comps = spec["components"]["schemas"]
                self.assertTrue(refs)
                for ref in refs:
                    self.assertTrue(ref.startswith("#/components/schemas/"), ref)
                    self.assertIn(ref.rsplit("/", 1)[1], comps, ref)
                self.assertIn("ContentBlock", comps)
        _run(go())


class TestStartupAndCli(_Base):
    def test_env_configuration(self) -> None:
        os.environ["PUPPETLLM_PENDING_TIMEOUT"] = "12.5"
        os.environ["PUPPETLLM_DEFAULT_RESPONSE"] = "env default"
        os.environ["PUPPETLLM_SEED"] = "3"
        try:
            fs = _import_fresh()
        finally:
            for k in ("PUPPETLLM_PENDING_TIMEOUT", "PUPPETLLM_DEFAULT_RESPONSE", "PUPPETLLM_SEED"):
                os.environ.pop(k)
        cfg = fs.harness.config
        self.assertEqual(cfg.pending_timeout_s, 12.5)
        self.assertEqual(cfg.on_unmatched, "default")
        self.assertEqual(cfg.default_response["content"][0]["text"], "env default")
        self.assertEqual(cfg.seed, 3)
        os.environ["PUPPETLLM_DEFAULT_RESPONSE"] = json.dumps({"content": [_TOOL_USE], "stop_reason": "tool_use"})
        os.environ["PUPPETLLM_ON_UNMATCHED"] = "error"
        try:
            fs = _import_fresh()
        finally:
            os.environ.pop("PUPPETLLM_DEFAULT_RESPONSE")
            os.environ.pop("PUPPETLLM_ON_UNMATCHED")
        self.assertEqual(fs.harness.config.default_response["content"][0]["name"], "get_weather")
        self.assertEqual(fs.harness.config.on_unmatched, "error")
        # an invalid value is ignored, not fatal
        os.environ["PUPPETLLM_PENDING_TIMEOUT"] = "soon"
        try:
            fs = _import_fresh()
        finally:
            os.environ.pop("PUPPETLLM_PENDING_TIMEOUT")
        self.assertIsNone(fs.harness.config.pending_timeout_s)
        self.mod = _import_fresh()

    def test_config_file_and_cli_parsing(self) -> None:
        from puppetllm.__main__ import build_parser, main
        fs = self.mod
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"config": {"pending_timeout_s": 3, "on_unmatched": "error"},
                       "rules": [{"id": "r1", "steps": [{"respond": {"text": "hi"}}]}]}, f)
        try:
            fs.load_config_file(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(fs.harness.config.pending_timeout_s, 3.0)
        self.assertEqual([r.id for r in fs.harness.rules], ["r1"])
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"rules": [{"steps": []}]}, f)
        try:
            with self.assertRaises(ValueError):
                fs.load_config_file(f.name)
        finally:
            os.unlink(f.name)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"id": "from-file", "steps": [{"respond": {"text": "hi"}}]}], f)
        try:
            fs.load_rules_file(f.name)
            self.assertEqual([r.id for r in fs.harness.rules], ["from-file"])
            os.environ["PUPPETLLM_CONFIG"] = f.name  # a bare list is taken as rules
            try:
                fs2 = _import_fresh()
            finally:
                os.environ.pop("PUPPETLLM_CONFIG")
            self.assertEqual([r.id for r in fs2.harness.rules], ["from-file"])
            self.mod = fs = _import_fresh()
        finally:
            os.unlink(f.name)
        args = build_parser().parse_args(["serve", "--port", "1", "--pending-timeout", "2",
                                          "--on-unmatched", "error", "--seed", "9"])
        self.assertEqual((args.command, args.port, args.pending_timeout, args.seed), ("serve", 1, 2.0, 9))
        self.assertEqual(fs.parse_default_response("plain text"), "plain text")
        self.assertEqual(fs.parse_default_response('{"text": "j"}'), {"text": "j"})
        with self.assertRaises(ValueError):
            fs.parse_default_response('{"text": oops}')
        os.environ["PUPPETLLM_DEFAULT_RESPONSE"] = '{"text": oops}'
        try:
            fs_bad = _import_fresh()
        finally:
            os.environ.pop("PUPPETLLM_DEFAULT_RESPONSE")
        self.assertIsNone(fs_bad.harness.config.default_response)
        self.assertEqual(fs_bad.harness.config.on_unmatched, "pending")
        self.mod = fs = _import_fresh()
        with self.assertRaises(SystemExit) as cm:
            main(["--version"])
        self.assertEqual(cm.exception.code, 0)
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["serve", "--on-unmatched", "sometimes"])
        # default subcommand: bare flags mean `serve`
        from puppetllm import __main__ as entry
        self.assertEqual(entry.build_parser().parse_args(["serve", "--port", "7"]).port, 7)
        self.assertEqual(entry.main(["wait", "--url", "http://127.0.0.1:9", "--timeout", "0.2"]), 1)


class TestTestingHelpers(unittest.TestCase):
    """`puppetllm.testing` against a real in-process uvicorn (skipped without uvicorn)."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import uvicorn  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("uvicorn not installed")
        from puppetllm.testing import serve
        cls._cm = serve()
        cls.puppet = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._cm.__exit__(None, None, None)

    def setUp(self) -> None:
        self.puppet.reset()

    def _post(self, body: dict[str, Any]) -> Any:
        import httpx
        return httpx.post(self.puppet.url + "/v1/messages", json=body, timeout=10)

    def test_expect_chain_and_assertions(self) -> None:
        from puppetllm.testing import PuppetAssertionError, PuppetError
        p = self.puppet
        exp = p.expect(tools=["get_weather"], has_tool_result=False).respond(content=[_TOOL_USE])
        second = p.expect(has_tool_result=True).error(429, headers={"retry-after": "0"})
        second.respond(text="sunny")
        exp.repeat()  # chaining onto an earlier expectation keeps its place in the order
        self.assertTrue(exp.id)
        self.assertEqual([r["id"] for r in p.rules()["rules"]], [exp.id, second.id])
        body = {**_MSG, "tools": [_WEATHER_TOOL]}
        r = self._post(body)
        self.assertEqual(r.json()["content"][0]["type"], "tool_use")
        with self.assertRaises(PuppetAssertionError):
            p.assert_consumed()
        body2 = {**body, "messages": body["messages"] + [
            {"role": "assistant", "content": [_TOOL_USE]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01",
                                          "content": "sunny"}]}]}
        self.assertEqual(self._post(body2).status_code, 429)
        self.assertEqual(self._post(body2).json()["content"][0]["text"], "sunny")
        p.assert_consumed()
        p.assert_no_pending()
        self.assertEqual(len(p.assert_sent(tool="get_weather")), 3)
        p.assert_sent(model="claude-*", contains="Tokyo", count=3)
        p.assert_sent(contains="sunny", count=2)
        p.assert_not_sent(provider="openai")
        with self.assertRaises(PuppetAssertionError):
            p.assert_sent(contains="Osaka")
        with self.assertRaises(PuppetError):
            p.respond(text="nobody is waiting")
        self.assertEqual(p.stats()["completed_requests"], 2)
        self.assertEqual(p.config()["on_unmatched"], "pending")
        p.config(on_unmatched="default", default_response="dflt")
        self.assertEqual(self._post(_MSG).json()["content"][0]["text"], "dflt")
        p.config(on_unmatched=None)
        self.assertEqual(p.advance_clock(5), 5.0)
        p.clear()
        self.assertEqual(p.rules()["rules"], [])

    def test_interactive_helpers_over_http(self) -> None:
        import threading
        p = self.puppet
        out: dict[str, Any] = {}

        def app_call() -> None:
            out["r"] = self._post(_MSG)

        th = threading.Thread(target=app_call)
        th.start()
        snap = p.wait_pending(timeout=5)
        self.assertEqual(snap["model"], "claude-sonnet-5")
        self.assertEqual(len(p.pending()), 1)
        p.respond(text="hand-written", pending_id=snap["pending_id"], stop_reason="end_turn")
        th.join(5)
        self.assertEqual(out["r"].json()["content"][0]["text"], "hand-written")
        th = threading.Thread(target=app_call)
        th.start()
        p.wait_pending(timeout=5)
        p.error(503, message="down", delay_ms=0)
        th.join(5)
        self.assertEqual(out["r"].status_code, 503)

    def test_stream_frames_are_paced_individually(self) -> None:
        import httpx
        p = self.puppet
        p.expect().respond(text="abcdefgh", ttfb_ms=300, chunk_delay_ms=100)
        arrivals: list[float] = []
        t0 = time.monotonic()
        with httpx.stream("POST", p.url + "/v1/messages", json={**_MSG, "stream": True}, timeout=10) as r:
            for _ in r.iter_raw():
                arrivals.append(time.monotonic() - t0)
        self.assertGreaterEqual(arrivals[0], 0.3)
        # frames are paced one by one: several distinct gaps of about the chunk delay
        # (a single sleep followed by one burst would show none)
        gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
        self.assertGreaterEqual(sum(1 for g in gaps if g >= 0.07), 2, gaps)
        self.assertGreaterEqual(arrivals[-1], 0.3 + 0.1 * 5)

    def test_wait_pending_outlives_the_client_timeout(self) -> None:
        import threading
        from puppetllm.testing import Puppet
        short = Puppet(self.puppet.url, timeout=1.0)
        out: dict[str, Any] = {}

        def late_call() -> None:
            time.sleep(1.5)
            out["r"] = self._post(_MSG)

        th = threading.Thread(target=late_call)
        th.start()
        snap = short.wait_pending(timeout=5)
        self.assertEqual(snap["model"], "claude-sonnet-5")
        short.respond(text="seen", pending_id=snap["pending_id"])
        th.join(5)
        self.assertEqual(out["r"].json()["content"][0]["text"], "seen")
        short.close()

    def test_baseline_keeps_startup_rules(self) -> None:
        p = self.puppet
        p.set_rules([{"id": "startup", "steps": [{"respond": {"text": "from startup"}}], "repeat": True}])
        base = p.baseline()
        p.expect().respond(text="test-local")
        p.config(rate_limit={"rpm": 3})
        p.reset(base)
        self.assertEqual([r["id"] for r in p.rules()["rules"]], ["startup"])
        self.assertIsNone(p.config()["rate_limit"])
        self.assertEqual(self._post(_MSG).json()["content"][0]["text"], "from startup")
        p.reset()
        self.assertEqual(p.rules()["rules"], [])

    def test_serve_baseline_includes_its_arguments(self) -> None:
        from puppetllm.testing import serve
        with serve(config={"on_unmatched": "error"},
                   rules=[{"id": "arg", "steps": [{"respond": {"text": "arg"}}], "repeat": True}]) as other:
            self.assertEqual(other.startup_baseline["config"]["on_unmatched"], "error")
            self.assertEqual([r["id"] for r in other.startup_baseline["rules"]], ["arg"])
            other.config(on_unmatched="pending")
            other.clear_rules()
            other.reset(other.startup_baseline)
            self.assertEqual(other.config()["on_unmatched"], "error")
            self.assertEqual([r["id"] for r in other.rules()["rules"]], ["arg"])
        with serve(rules=[]) as other:  # an explicit empty list clears start-up rules
            self.assertEqual(other.rules()["rules"], [])
            self.assertEqual(other.startup_baseline["rules"], [])

    def test_bulk_and_strict_helpers(self) -> None:
        import threading
        import httpx
        from puppetllm.testing import PuppetError
        p = self.puppet
        out: list[Any] = []
        errors: list[BaseException] = []

        def call() -> None:
            try:
                out.append(self._post(_MSG))
            except BaseException as e:  # noqa: BLE001 - surfaced by the assertions below
                errors.append(e)

        def fan_out(n: int) -> list[threading.Thread]:
            threads = [threading.Thread(target=call) for _ in range(n)]
            for th in threads:
                th.start()
            for _ in range(100):
                if len(p.pending()) == n:
                    break
                time.sleep(0.02)
            self.assertEqual(len(p.pending()), n)
            return threads

        def settle(threads: list[threading.Thread], n: int) -> None:
            for th in threads:
                th.join(5)
            self.assertFalse(any(th.is_alive() for th in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(out), n)

        threads = fan_out(2)
        pids = [x["pending_id"] for x in p.pending()]
        p.respond_many([{"pending_id": pids[0], "text": "one"}, {"pending_id": pids[1], "text": "two"}])
        settle(threads, 2)
        self.assertEqual(sorted(r.json()["content"][0]["text"] for r in out), ["one", "two"])
        out.clear()
        threads = fan_out(2)
        self.assertEqual(len(p.respond_all(text="all")), 2)
        settle(threads, 2)
        self.assertEqual([r.json()["content"][0]["text"] for r in out], ["all", "all"])
        p.strict_blocks()
        p.expect().error(503, after_blocks=1, content=[{"type": "text", "text": "partial"}])
        with self.assertRaises(PuppetError):
            p.expect().respond(content=[{"type": "tool_result", "tool_use_id": "x", "content": "y"}])
        # the error expectation is consumed by a stream that fails after its first block
        with httpx.stream("POST", p.url + "/v1/messages", json={**_MSG, "stream": True}, timeout=10) as r:
            self.assertEqual(r.status_code, 200)
            body = b"".join(r.iter_raw()).decode()
        self.assertEqual(body.count("event: content_block_stop"), 1)
        self.assertIn("event: error", body)
        p.assert_consumed()
        p.strict_blocks(False)
        self.assertTrue(p.config()["strict_blocks"] is False)

    def test_repeat_before_steps(self) -> None:
        p = self.puppet
        exp = p.expect(model="claude-*").repeat()
        self.assertIsNone(exp.id)
        exp.respond(text="forever")
        self.assertTrue(p.rules()["rules"][0]["repeat"])
        for _ in range(2):
            self.assertEqual(self._post(_MSG).json()["content"][0]["text"], "forever")

    def test_full_reset_restores_configuration(self) -> None:
        p = self.puppet
        p.config(rate_limit={"rpm": 1}, on_unmatched="error")
        p.clear()
        self.assertEqual(p.config()["rate_limit"], {"rpm": 1})
        p.clear(config=True)
        cfg = p.config()
        self.assertEqual((cfg["rate_limit"], cfg["on_unmatched"]), (None, "pending"))
        # a `config()` result is a valid baseline for reset(): the fixture's way of keeping
        # the server's start-up settings while isolating tests from each other
        p.config(pending_timeout_s=42, on_unmatched="default", default_response="canned",
                 unmatched_error={"status": 503, "message": "x"}, latency={"delay_ms": 1},
                 rate_limit={"itpm": 5000}, seed=3)
        baseline = p.config()
        p.config(pending_timeout_s=None, rate_limit=None, on_unmatched="error")
        p.reset(baseline)
        self.assertEqual(p.config(), baseline)
        p.reset()
        self.assertEqual(p.config()["pending_timeout_s"], None)

    def test_disconnected_client_is_forgotten(self) -> None:
        """A client that gives up while pending (a raw socket closed mid-wait) leaves no
        pending behind for a responder to answer."""
        import socket
        from urllib.parse import urlparse
        p = self.puppet
        u = urlparse(p.url)
        payload = json.dumps(_MSG).encode()
        with socket.create_connection((u.hostname, u.port), timeout=5) as s:
            s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                      + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not p.pending():
                time.sleep(0.02)
            self.assertEqual(len(p.pending()), 1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and p.pending():
            time.sleep(0.05)
        self.assertEqual(p.pending(), [])
        self.assertEqual(p.history(), [])
        p.assert_no_pending()

    def test_sdk_round_trip(self) -> None:
        try:
            import anthropic
        except ImportError:
            self.skipTest("anthropic SDK not installed")
        p = self.puppet
        p.expect(last_user_text="ping").respond(text="pong")
        client = anthropic.Anthropic(base_url=p.url, api_key="test", max_retries=0)
        msg = client.messages.create(model="claude-sonnet-5", max_tokens=8,
                                     messages=[{"role": "user", "content": "ping"}])
        self.assertEqual(msg.content[0].text, "pong")
        self.assertEqual(client.messages.count_tokens(
            model="claude-sonnet-5", messages=[{"role": "user", "content": "ping"}]).input_tokens,
            msg.usage.input_tokens)
        self.assertIn("claude-sonnet-5", [m.id for m in client.models.list()])
        from puppetllm import pricing
        self.assertEqual(len(list(client.models.list(limit=3))), len(pricing.KNOWN_MODELS["anthropic"]))
        self.assertEqual(client.models.retrieve("claude-sonnet-5").display_name, "Claude Sonnet 5")
        p.config(rate_limit={"rpm": 1})
        p.expect().respond(text="x").repeat()
        with self.assertRaises(anthropic.RateLimitError) as cm:
            for _ in range(2):
                client.messages.create(model="claude-sonnet-5", max_tokens=8,
                                       messages=[{"role": "user", "content": "again"}])
        self.assertIn("retry-after", cm.exception.response.headers)
        p.config(rate_limit=None)
        p.clear_rules()
        try:
            import openai
        except ImportError:
            return
        oc = openai.OpenAI(base_url=p.url + "/v1", api_key="test", max_retries=0)
        self.assertIn("gpt-5.4", [m.id for m in oc.models.list()])
        p.expect(provider="openai").respond(text="from rule")
        self.assertEqual(oc.chat.completions.create(
            model="gpt-5.4", messages=[{"role": "user", "content": "hi"}]).choices[0].message.content,
            "from rule")


if __name__ == "__main__":
    unittest.main()
