"""Tests for the OpenAI and Bedrock extras: individually injected `n` choices, the
logprobs fixture, the guardrail stop reason and action, the Converse / InvokeModel
`trace` passthrough and the prompt-management fixtures.

Run:
  python3 -m unittest puppetllm.tests.test_extras -v
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
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


def _sse_data(text: str) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[len("data: "):]))
    return out


_OPENAI = {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]}
_BEDROCK = "/model/anthropic.claude-sonnet-4-5-20250929-v1:0"
_ARN = "arn:aws:bedrock:us-east-1:123456789012:prompt/PROMPT123"
_ARN2 = "arn:aws:bedrock:us-east-1:123456789012:prompt/OTHER456"


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

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


class TestOpenAIExtras(_Base):
    def test_choices_are_injected_individually(self) -> None:
        async def go():
            async with await self._client() as c:
                body = {**_OPENAI, "n": 3}
                t = await self._pending(c, "/v1/chat/completions", body)
                self.assertEqual((await c.post("/_control/respond", json={"text": "top", "choices": [{"text": "a"}] * 3})).status_code, 400)
                await self._ctl(c, "post", "/_control/respond", json={"choices": [
                    {"text": "first answer"},
                    {"content": [{"type": "tool_use", "name": "get_weather", "input": {"city": "Tokyo"}}]},
                    {"text": "third", "stop_reason": "max_tokens"}]})
                r = await t
                self.assertEqual(r.status_code, 200, r.text)
                choices = r.json()["choices"]
                self.assertEqual([ch["index"] for ch in choices], [0, 1, 2])
                self.assertEqual(choices[0]["message"]["content"], "first answer")
                self.assertEqual(choices[1]["message"]["tool_calls"][0]["function"]["name"], "get_weather")
                self.assertEqual([ch["finish_reason"] for ch in choices], ["stop", "tool_calls", "length"])
                # a stop_reason on the first choice stays its own: the others derive theirs
                t = await self._pending(c, "/v1/chat/completions", body)
                await self._ctl(c, "post", "/_control/respond", json={"choices": [
                    {"text": "cut", "stop_reason": "max_tokens"},
                    {"content": [{"type": "tool_use", "name": "get_weather", "input": {}}]},
                    {"text": "plain"}]})
                r2 = await t
                self.assertEqual([ch["finish_reason"] for ch in r2.json()["choices"]], ["length", "tool_calls", "stop"])
                self.assertEqual([ch["stop_reason"] for ch in (await self._ctl(c, "get", "/_control/history/2"))["choices"]],
                                 ["max_tokens", None, None])
                # ... while a top-level stop_reason applies to every choice without one
                t = await self._pending(c, "/v1/chat/completions", body)
                await self._ctl(c, "post", "/_control/respond", json={"stop_reason": "max_tokens", "choices": [
                    {"text": "a"}, {"text": "b", "stop_reason": "end_turn"}, {"text": "c"}]})
                self.assertEqual([ch["finish_reason"] for ch in (await t).json()["choices"]], ["length", "stop", "length"])
                # a usage override describes all the injected choices: not n-scaled
                t = await self._pending(c, "/v1/chat/completions", body)
                await self._ctl(c, "post", "/_control/respond", json={"choices": [{"text": "a"}, {"text": "b"}, {"text": "c"}],
                                                                       "usage": {"output_tokens": 10}})
                self.assertEqual((await t).json()["usage"]["completion_tokens"], 10)
                # usage sums the three answers (not three copies of the first); history
                # keeps every choice
                from puppetllm import pricing
                e = (await self._ctl(c, "get", "/_control/history/1"))
                expected = sum(pricing.estimate_output_tokens(ch["content_blocks"]) for ch in e["choices"])
                self.assertEqual((r.json()["usage"]["completion_tokens"], e["usage"]["output_tokens"]), (expected, expected))
                # streaming: each index streams its own answer, interleaved
                t = await self._pending(c, "/v1/chat/completions", {**body, "n": 2, "stream": True})
                await self._ctl(c, "post", "/_control/respond", json={"choices": [{"text": "left"}, {"text": "right"}]})
                r = await t
                chunks = _sse_data(r.text)
                texts: dict[int, str] = {0: "", 1: ""}
                finishes: dict[int, str | None] = {}
                for ch in chunks:
                    for choice in ch["choices"]:
                        texts[choice["index"]] += choice["delta"].get("content") or ""
                        if choice["finish_reason"]:
                            finishes[choice["index"]] = choice["finish_reason"]
                self.assertEqual(texts, {0: "left", 1: "right"})
                self.assertEqual(finishes, {0: "stop", 1: "stop"})
                # a list shorter than n is cycled
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"choices": [
                    {"text": "p", "logprobs": [{"token": "p", "logprob": -0.5}]}, {"text": "q"}]}}]})
                r = await self._answered(c, "/v1/chat/completions", {**body, "n": 3, "logprobs": True})
                self.assertEqual([ch["message"]["content"] for ch in r.json()["choices"]], ["p", "q", "p"])
                self.assertEqual(r.json()["choices"][0]["logprobs"]["content"][0]["logprob"], -0.5)
                self.assertNotEqual(r.json()["choices"][1]["logprobs"]["content"][0]["logprob"], -0.5)
                # choices on a non-OpenAI pending: the first one answers
                t = await self._pending(c, "/v1/messages", {"model": "claude-sonnet-5", "max_tokens": 5,
                                                            "messages": [{"role": "user", "content": "hi"}]})
                await self._ctl(c, "post", "/_control/respond", json={"choices": [{"text": "only", "stop_reason": "max_tokens"}]})
                self.assertEqual(((await t).json()["content"][0]["text"], (await t).json()["stop_reason"]), ("only", "max_tokens"))
                self.assertNotIn("choices", (await self._ctl(c, "get", "/_control/history"))["history"][-1])
                # a default_response carrying choices survives a config round trip
                await self._ctl(c, "post", "/_control/config", json={"on_unmatched": "default",
                                                                      "default_response": {"choices": [{"text": "d1"}, {"text": "d2", "stop_reason": "max_tokens"}]}})
                shown = (await self._ctl(c, "get", "/_control/config"))["config"]["default_response"]
                self.assertEqual(shown["choices"], [{"content": [{"type": "text", "text": "d1"}]},
                                                    {"content": [{"type": "text", "text": "d2"}], "stop_reason": "max_tokens"}])
                await self._ctl(c, "post", "/_control/config", json={"default_response": shown})
                r = await self._answered(c, "/v1/chat/completions", {**body, "n": 2})
                self.assertEqual([ch["finish_reason"] for ch in r.json()["choices"]], ["stop", "length"])
                await self._ctl(c, "post", "/_control/config", json={"on_unmatched": "pending", "default_response": None})
        _run(go())

    def test_logprobs_fixture(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [{"respond": {"text": "Hello brave new world"}}], "repeat": True})
                # not asked: null
                r = await self._answered(c, "/v1/chat/completions", _OPENAI)
                self.assertIsNone(r.json()["choices"][0]["logprobs"])
                # asked: deterministic per-token values, top_logprobs when requested
                r = await self._answered(c, "/v1/chat/completions", {**_OPENAI, "logprobs": True})
                self.assertEqual({tuple(e["top_logprobs"]) for e in r.json()["choices"][0]["logprobs"]["content"]}, {()})
                for bad in ({"top_logprobs": 2}, {"logprobs": True, "top_logprobs": 21}, {"logprobs": True, "top_logprobs": "x"}):
                    rr = await c.post("/v1/chat/completions", json={**_OPENAI, **bad})
                    self.assertEqual((rr.status_code, rr.json()["error"]["param"]), (400, "top_logprobs"), bad)
                rr = await c.post("/v1/chat/completions", json={**_OPENAI, "logprobs": "false"})
                self.assertEqual((rr.status_code, rr.json()["error"]["param"]), (400, "logprobs"))
                r = await self._answered(c, "/v1/chat/completions", {**_OPENAI, "logprobs": True, "top_logprobs": 2})
                lp = r.json()["choices"][0]["logprobs"]
                self.assertEqual("".join(e["token"] for e in lp["content"]), "Hello brave new world")
                self.assertTrue(all(e["logprob"] <= 0 for e in lp["content"]))
                self.assertEqual(len(lp["content"][0]["top_logprobs"]), 2)
                self.assertEqual(lp["content"][0]["top_logprobs"][0]["token"], "Hello")
                self.assertEqual(lp["content"][0]["bytes"], list(b"Hello"))
                self.assertIsNone(lp["refusal"])
                r2 = await self._answered(c, "/v1/chat/completions", {**_OPENAI, "logprobs": True, "top_logprobs": 2})
                self.assertEqual(r2.json()["choices"][0]["logprobs"], lp)  # deterministic
                # streaming: the same tokens ride on the content deltas
                r = await self._answered(c, "/v1/chat/completions", {**_OPENAI, "logprobs": True, "stream": True})
                streamed = [e for ch in _sse_data(r.text) for choice in ch["choices"]
                            if choice["logprobs"] for e in choice["logprobs"]["content"]]
                self.assertEqual("".join(e["token"] for e in streamed), "Hello brave new world")
                # an injected fixture replaces the made-up values
                await self._ctl(c, "delete", "/_control/rules")
                t = await self._pending(c, "/v1/chat/completions", {**_OPENAI, "logprobs": True, "top_logprobs": 1})
                for bad in ("x", [{"token": 1, "logprob": -1}], [{"token": "a", "logprob": 0.5}],
                            [{"token": "a", "logprob": -1, "top_logprobs": [{"token": "b"}]}],
                            [{"token": "a", "logprob": -1, "bytes": [1]}],
                            [{"token": "a", "logprob": -1, "top_logprobs": [{"token": "b", "logprob": -1, "x": 1}]}]):
                    self.assertEqual((await c.post("/_control/respond", json={"text": "Hi", "logprobs": bad})).status_code, 400, bad)
                raw = b'{"text": "Hi", "logprobs": [{"token": "a", "logprob": -Infinity}]}'
                self.assertEqual((await c.post("/_control/respond", content=raw, headers={"content-type": "application/json"})).status_code, 400)
                await self._ctl(c, "post", "/_control/respond", json={"text": "Hi there", "logprobs": [
                    {"token": "Hi", "logprob": -0.1, "top_logprobs": [{"token": "Hi", "logprob": -0.1}, {"token": "Hey", "logprob": -2}]},
                    {"token": " there", "logprob": -0.3}]})
                lp = (await t).json()["choices"][0]["logprobs"]["content"]
                self.assertEqual([(e["token"], e["logprob"]) for e in lp], [("Hi", -0.1), (" there", -0.3)])
                self.assertEqual(len(lp[0]["top_logprobs"]), 1)  # cut to the requested top_logprobs
                self.assertEqual(lp[1]["top_logprobs"], [{"token": " there", "logprob": -0.3, "bytes": list(b" there")}])
                # not requested: the fixture's alternatives are not sent, the list is still there
                t = await self._pending(c, "/v1/chat/completions", {**_OPENAI, "logprobs": True})
                await self._ctl(c, "post", "/_control/respond", json={"text": "Hi", "logprobs": [
                    {"token": "Hi", "logprob": -0.1, "top_logprobs": [{"token": "Hey", "logprob": -2}]}]})
                self.assertEqual((await t).json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"], [])
                # a refusal carries its logprobs under `refusal`, non-stream and stream alike
                for stream in (False, True):
                    t = await self._pending(c, "/v1/chat/completions", {**_OPENAI, "logprobs": True, "stream": stream})
                    await self._ctl(c, "post", "/_control/respond", json={"text": "I cannot", "stop_reason": "refusal"})
                    r = await t
                    if stream:
                        parts = [choice["logprobs"] for ch in _sse_data(r.text) for choice in ch["choices"] if choice["logprobs"]]
                        self.assertTrue(parts and all(p["content"] is None for p in parts))
                        self.assertEqual("".join(e["token"] for p in parts for e in p["refusal"]), "I cannot")
                    else:
                        lp = r.json()["choices"][0]["logprobs"]
                        self.assertIsNone(lp["content"])
                        self.assertEqual("".join(e["token"] for e in lp["refusal"]), "I cannot")
                # a tool-only answer has no text, so no logprobs
                t = await self._pending(c, "/v1/chat/completions", {**_OPENAI, "logprobs": True})
                await self._ctl(c, "post", "/_control/respond", json={
                    "content": [{"type": "tool_use", "name": "f", "input": {}}], "logprobs": [{"token": "zzz", "logprob": -1}]})
                self.assertIsNone((await t).json()["choices"][0]["logprobs"])
        _run(go())


class TestBedrockExtras(_Base):
    def test_guardrail_intervened(self) -> None:
        async def go():
            async with await self._client() as c:
                await self._ctl(c, "post", "/_control/rules", json={"steps": [
                    {"respond": {"text": "blocked", "stop_reason": "guardrail_intervened"}}], "repeat": True})
                # InvokeModel: the action in the body; the Anthropic-shaped message ends normally
                invoke = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 5,
                          "messages": [{"role": "user", "content": "hi"}]}
                r = await self._answered(c, f"{_BEDROCK}/invoke", invoke)
                self.assertEqual((r.json()["amazon-bedrock-guardrailAction"], r.json()["stop_reason"]), ("INTERVENED", "end_turn"))
                # the Anthropic-shaped routes have no spelling for it: the message ends normally
                r = await self._answered(c, "/anthropic/v1/messages", {"model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                                                                      "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(r.json()["stop_reason"], "end_turn")
                r = await self._answered(c, f"{_BEDROCK}/invoke-with-response-stream", invoke)
                chunks = [json.loads(base64.b64decode(m)) for m in re.findall(rb'"bytes":\s*"([A-Za-z0-9+/=]+)"', r.content)]
                stop = next(ch for ch in chunks if ch["type"] == "message_stop")
                self.assertEqual(stop["amazon-bedrock-guardrailAction"], "INTERVENED")
                self.assertEqual(next(ch for ch in chunks if ch["type"] == "message_delta")["delta"]["stop_reason"], "end_turn")
                # a configured guardrail that lets the answer through: NONE; none configured: absent
                await self._ctl(c, "put", "/_control/rules", json={"rules": [{"steps": [{"respond": {"text": "fine"}}], "repeat": True}]})
                r = await self._answered(c, f"{_BEDROCK}/invoke", invoke, headers={"X-Amzn-Bedrock-GuardrailIdentifier": "gr-1"})
                self.assertEqual(r.json()["amazon-bedrock-guardrailAction"], "NONE")
                e = (await self._ctl(c, "get", "/_control/history/4"))
                self.assertEqual(e["request"]["bedrock_options"]["guardrail_identifier"], "gr-1")
                r = await self._answered(c, f"{_BEDROCK}/invoke", invoke)
                self.assertNotIn("amazon-bedrock-guardrailAction", r.json())
                # the OpenAI route reports a filtered answer in its own vocabulary
                await self._ctl(c, "put", "/_control/rules", json={"rules": [{"steps": [
                    {"respond": {"text": "blocked", "stop_reason": "guardrail_intervened"}}], "repeat": True}]})
                r = await self._answered(c, "/v1/chat/completions", _OPENAI)
                self.assertEqual(r.json()["choices"][0]["finish_reason"], "content_filter")
                # an InvokeModel trace is injectable too (body field, and the message_stop chunk)
                await self._ctl(c, "put", "/_control/rules", json={"rules": [{"steps": [{"respond": {
                    "text": "fine", "provider_response": {"invoke": {"trace": {"guardrail": {"input": {}}}}}}}], "repeat": True}]})
                hdrs = {"X-Amzn-Bedrock-GuardrailIdentifier": "gr-1"}
                r = await self._answered(c, f"{_BEDROCK}/invoke", invoke, headers=hdrs)
                self.assertEqual((r.json()["amazon-bedrock-trace"], r.json()["amazon-bedrock-guardrailAction"]), ({"guardrail": {"input": {}}}, "NONE"))
                r = await self._answered(c, f"{_BEDROCK}/invoke-with-response-stream", invoke, headers=hdrs)
                chunks = [json.loads(base64.b64decode(m)) for m in re.findall(rb'"bytes":\s*"([A-Za-z0-9+/=]+)"', r.content)]
                self.assertEqual(next(ch for ch in chunks if ch["type"] == "message_stop")["amazon-bedrock-trace"], {"guardrail": {"input": {}}})
        _run(go())

    def test_converse_trace_passthrough(self) -> None:
        async def go():
            async with await self._client() as c:
                trace = {"guardrail": {"inputAssessment": {"gr-1": {"topicPolicy": {"topics": []}}}}}
                body = {"messages": [{"role": "user", "content": [{"text": "hi"}]}],
                        "guardrailConfig": {"guardrailIdentifier": "gr-1", "guardrailVersion": "DRAFT", "trace": "enabled"}}
                t = await self._pending(c, f"{_BEDROCK}/converse", body)
                for bad in ("x", {"bogus": {}}, {"converse": {"trace": 1}}, {"converse": {"other": {}}}):
                    self.assertEqual((await c.post("/_control/respond", json={"text": "ok", "provider_response": bad})).status_code, 400, bad)
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok", "provider_response": {"converse": {"trace": trace}}})
                r = await t
                self.assertEqual(r.json()["trace"], trace)
                self.assertEqual((await self._ctl(c, "get", "/_control/history/1"))["provider_response"], {"converse": {"trace": trace}})
                # no trace injected: no field
                t = await self._pending(c, f"{_BEDROCK}/converse", body)
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok"})
                self.assertNotIn("trace", (await t).json())
                # the Anthropic route ignores it (recorded only)
                t = await self._pending(c, "/v1/messages", {"model": "claude-sonnet-5", "max_tokens": 5,
                                                            "messages": [{"role": "user", "content": "hi"}]})
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok", "provider_response": {"converse": {"trace": trace}}})
                self.assertNotIn("trace", (await t).json())
        _run(go())

    def test_prompt_fixtures(self) -> None:
        async def go():
            async with await self._client() as c:
                tmpl = {"system": "You answer about {{topic}}.",
                        "messages": [{"role": "user", "content": [{"type": "text", "text": "Tell me about {{topic}} in {{lang}}; {{unknown}} stays"}]}],
                        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]}
                for bad in ({"prompts": "x"}, {"prompts": {_ARN: {"bogus": 1}}}, {"prompts": {_ARN: {"messages": "no"}}}, {},
                            {"prompts": {"not-an-arn": {"messages": []}}}):
                    self.assertEqual((await c.put("/_control/prompts", json=bad)).status_code, 400, bad)
                self.assertEqual((await self._ctl(c, "put", "/_control/prompts", json={"prompts": {_ARN: tmpl}}))["count"], 1)
                self.assertEqual((await self._ctl(c, "get", "/_control/prompts"))["prompts"][_ARN]["system"], tmpl["system"])
                # a Converse request naming the ARN with only promptVariables sees the rendered prompt
                body = {"promptVariables": {"topic": {"text": "whales"}, "lang": {"text": "Japanese"}}}
                t = await self._pending(c, f"/model/{_ARN}/converse", body)
                snap = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(snap["system"], [{"type": "text", "text": "You answer about whales."}])
                self.assertEqual(snap["messages"][0]["content"][0]["text"], "Tell me about whales in Japanese; {{unknown}} stays")
                self.assertEqual(snap["tools"][0]["name"], "lookup")
                self.assertEqual((snap["converse"]["prompt_arn"], snap["converse"]["prompt_rendered"]), (_ARN, True))
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok"})
                self.assertEqual((await t).status_code, 200)
                # a versioned ARN uses the base registration; count-tokens counts the rendered prompt
                t = await self._pending(c, f"/model/{_ARN}:3/converse", body)
                snap = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual((snap["converse"]["prompt_rendered"], snap["messages"][0]["content"][0]["text"][:20]), (True, "Tell me about whales"))
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok"})
                await t
                counted = await self._ctl(c, "post", f"/model/{_ARN}/count-tokens", json={"input": {"converse": body}})
                self.assertEqual(counted["inputTokens"], snap["input_tokens_total"])
                # promptVariables must have the real shape; request messages win and are not "rendered"
                for bad in (["x"], {"topic": "whales"}, {"topic": {"text": 5}}):
                    self.assertEqual((await c.post(f"/model/{_ARN}/converse", json={"promptVariables": bad})).status_code, 400, bad)
                t = await self._pending(c, f"/model/{_ARN}/converse", {**body, "messages": [{"role": "user", "content": [{"text": "own"}]}]})
                snap = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual((snap["converse"]["prompt_rendered"], snap["messages"][0]["content"][0]["text"]), (False, "own"))
                self.assertEqual(snap["system"][0]["text"], "You answer about whales.")  # the template's system still applies
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok"})
                await t
                # placeholders are substituted in prompt text and tool descriptions only
                await self._ctl(c, "put", "/_control/prompts", json={"prompts": {_ARN: {
                    "messages": [{"role": "user", "content": "{{q}}"}],
                    "tools": [{"name": "t", "description": "about {{q}}", "input_schema": {"type": "object", "properties": {"{{q}}": {"type": "string", "description": "{{q}}"}}}}]}}})
                t = await self._pending(c, f"/model/{_ARN}/converse", {"promptVariables": {"q": {"text": "X"}}})
                snap = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual((snap["messages"][0]["content"], snap["tools"][0]["description"]), ("X", "about X"))
                self.assertIn("{{q}}", snap["tools"][0]["input_schema"]["properties"])
                self.assertEqual(snap["tools"][0]["input_schema"]["properties"]["{{q}}"]["description"], "{{q}}")
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok"})
                await t
                # ... including a tool result's own text
                await self._ctl(c, "put", "/_control/prompts", json={"prompts": {_ARN: {"messages": [
                    {"role": "user", "content": "{{q}}"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "t", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": [{"type": "text", "text": "R {{q}}"}]}]}]}}})
                t = await self._pending(c, f"/model/{_ARN}/converse", {"promptVariables": {"q": {"text": "X"}}})
                snap = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(snap["messages"][2]["content"][0]["content"][0]["text"], "R X")
                await self._ctl(c, "post", "/_control/respond", json={"text": "ok"})
                await t
                await self._ctl(c, "put", "/_control/prompts", json={"prompts": {_ARN: tmpl}})
                # a rule can match on the rendered text
                await self._ctl(c, "post", "/_control/rules", json={"match": {"last_user_text": "whales"}, "steps": [{"respond": {"text": "matched"}}]})
                r = await self._answered(c, f"/model/{_ARN}/converse", body)
                self.assertEqual(r.json()["output"]["message"]["content"][0]["text"], "matched")
                # POST adds, DELETE clears, clear() clears too
                self.assertEqual((await self._ctl(c, "post", "/_control/prompts", json={"prompts": {_ARN2: {"messages": []}}}))["count"], 2)
                await self._ctl(c, "delete", "/_control/prompts")
                self.assertEqual((await self._ctl(c, "get", "/_control/prompts"))["prompts"], {})
                await self._ctl(c, "put", "/_control/prompts", json={"prompts": {_ARN: tmpl}})
                await self._ctl(c, "post", "/_control/clear")
                self.assertEqual((await self._ctl(c, "get", "/_control/prompts"))["prompts"], {})
        _run(go())

    def test_prompts_from_a_config_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"prompts": {_ARN: {"messages": [{"role": "user", "content": "{{q}}"}]}}}, f)
        try:
            self.mod.load_config_file(f.name)
            self.assertEqual(self.mod.harness.prompts[_ARN]["messages"][0]["content"], "{{q}}")
            with open(f.name, "w") as g:
                json.dump({"prompts": {_ARN: {"nope": 1}}}, g)
            with self.assertRaises(ValueError):
                self.mod.load_config_file(f.name)
        finally:
            os.unlink(f.name)
            self.mod.harness.prompts.clear()


class TestClientHelpers(unittest.TestCase):
    def test_puppet_prompts_choices_and_baseline(self) -> None:
        import httpx
        from puppetllm.testing import serve
        with serve() as p:
            p.prompts({_ARN: {"messages": [{"role": "user", "content": "{{q}}"}]}})
            self.assertIn(_ARN, p.prompts())
            base = p.baseline()
            self.assertIn(_ARN, base["prompts"])
            p.clear()
            self.assertEqual(p.prompts(), {})
            p.reset(base)
            self.assertIn(_ARN, p.prompts())
            p.expect(provider="openai").respond(choices=[{"text": "a"}, {"text": "b"}],
                                                logprobs=[{"token": "a", "logprob": -0.5}])
            r = httpx.post(p.url + "/v1/chat/completions", json={**_OPENAI, "n": 2, "logprobs": True}, timeout=10)
            self.assertEqual([ch["message"]["content"] for ch in r.json()["choices"]], ["a", "b"])
            self.assertEqual(r.json()["choices"][0]["logprobs"]["content"][0]["token"], "a")

    def test_baseline_tolerates_an_older_server(self) -> None:
        import httpx
        from puppetllm.testing import Puppet

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/_control/prompts":
                return httpx.Response(404, json={"detail": "Not Found"})
            if request.url.path == "/_control/config":
                return httpx.Response(200, json={"config": {"pending_timeout_s": None}})
            if request.url.path == "/_control/rules":
                return httpx.Response(200, json={"rules": []})
            return httpx.Response(200, json={"ok": True})
        p = Puppet("http://old.test")
        p._http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://old.test")
        self.assertEqual(p.baseline(), {"config": {"pending_timeout_s": None}, "rules": []})


if __name__ == "__main__":
    unittest.main()
