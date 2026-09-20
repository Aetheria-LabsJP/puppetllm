"""Tests for the Bedrock extras: mid-stream error injection, the S3 emulation,
Converse / ConverseStream, and batch inference jobs.

Run:
  python3 -m unittest puppetllm.tests.test_bedrock_extras -v
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import unittest
from typing import Any

from puppetllm.providers import eventstream, s3

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


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()
        self.tmp = tempfile.mkdtemp(prefix="puppetllm-s3-test-")
        s3.reset_root(self.tmp)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # collector tasks belong to the event loop of the test that spawned them
        from puppetllm.providers import bedrock_batch as _bb
        self.addCleanup(_bb._collector_tasks.clear)

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

    async def _pending(self, c, path: str, body: dict) -> Any:
        t = asyncio.create_task(c.post(path, json=body, timeout=10))
        for _ in range(50):
            if (await c.get("/_control/pending")).json().get("has_pending"):
                return t
            await asyncio.sleep(0.05)
        self.fail("never became pending")

    async def _ctl(self, c, path: str, **kw: Any) -> Any:
        """POST to the control plane and insist on 200. A plain `await c.post(...)` there
        would let a rejected injection turn into a request nobody ever answers — and with
        `httpx.ASGITransport` that is an infinite wait, not a failed test."""
        r = await c.post(path, **kw)
        self.assertEqual(r.status_code, 200, f"{path}: {r.text}")
        return r

    async def _must_reject(self, c, path: str, body: dict) -> Any:
        """POST something that must be refused up front. A plain `await c.post` would block
        forever if a relaxed validator let it through (it becomes a pending nobody answers),
        turning a regression into a CI hang; this fails the test instead."""
        t = asyncio.create_task(c.post(path, json=body, timeout=10))
        for _ in range(100):
            if t.done():
                return t.result()
            if (await c.get("/_control/pending")).json().get("has_pending"):
                await self._ctl(c, "/_control/clear")
                await asyncio.gather(t, return_exceptions=True)
                self.fail(f"request was accepted (became pending) instead of rejected: {body}")
            await asyncio.sleep(0.01)
        t.cancel()
        self.fail("rejection never arrived")


class TestMidStreamError(_Base):
    def test_anthropic_sse_error_event_after_n_events(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {
                    "model": "claude-opus-5", "max_tokens": 10, "stream": True,
                    "messages": [{"role": "user", "content": "x"}]})
                r = await c.post("/_control/error", json={
                    "status": 529, "type": "overloaded_error", "message": "Overloaded",
                    "after_events": 3, "content": [{"type": "text", "text": "partial answer"}]})
                self.assertEqual(r.status_code, 200)
                resp = await t
                self.assertEqual(resp.status_code, 200)  # the failure is inside the stream
                events = _sse_events(resp.text)
                # `ping` rides along after message_start as it does in a normal SSE stream,
                # but `after_events` counts protocol events only.
                self.assertEqual([n for n, _ in events][:4],
                                 ["message_start", "ping", "content_block_start",
                                  "content_block_delta"])
                self.assertEqual(events[-1][0], "error")
                self.assertEqual(events[-1][1]["error"], {"type": "overloaded_error", "message": "Overloaded"})
                self.assertTrue(events[-1][1]["request_id"].startswith("req_"))
                # history records it as an injected error
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(h["injected_error"]["status"], 529)
                # after_events: 0 → the error is the only event
                t = await self._pending(c, "/v1/messages", {
                    "model": "claude-opus-5", "max_tokens": 10, "stream": True,
                    "messages": [{"role": "user", "content": "x"}]})
                await self._ctl(c, "/_control/error", json={"status": 500, "after_events": 0})
                events = _sse_events((await t).text)
                self.assertEqual([n for n, _ in events], ["error"])
                # non-streaming requests ignore after_events and get the HTTP error
                t = await self._pending(c, "/v1/messages", {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "x"}]})
                await self._ctl(c, "/_control/error", json={"status": 429, "after_events": 2})
                self.assertEqual((await t).status_code, 429)
                # validation
                t = await self._pending(c, "/v1/messages", {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "x"}]})
                r = await c.post("/_control/error", json={"status": 429, "after_events": -1})
                self.assertEqual(r.status_code, 400)
                await self._ctl(c, "/_control/error", json={"status": 429})
                await t
        _run(run())

    def test_bedrock_stream_exception_frame(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/model/anthropic.claude-opus-5/invoke-with-response-stream", {
                    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "x"}]})
                await self._ctl(c, "/_control/error", json={
                    "status": 429, "type": "rate_limit_error", "message": "slow down",
                    "after_events": 2, "content": [{"type": "text", "text": "partial"}]})
                r = await t
                self.assertEqual(r.status_code, 200)
                frames = eventstream.decode_frames(r.content)
                self.assertEqual([h.get(":event-type") for h, _ in frames[:2]], ["chunk", "chunk"])
                self.assertEqual(frames[-1][0][":message-type"], "exception")
                self.assertEqual(frames[-1][0][":exception-type"], "throttlingException")
                self.assertEqual(json.loads(frames[-1][1]), {"message": "slow down"})
                msgs = eventstream.decode_messages(r.content)
                self.assertEqual(msgs[0]["type"], "message_start")
                self.assertEqual(msgs[-1]["_exception"], "throttlingException")
        _run(run())

    def test_stream_exception_member_mapping(self) -> None:
        from puppetllm.providers.bedrock import stream_exception_member as f
        self.assertEqual(f("ThrottlingException"), "throttlingException")
        self.assertEqual(f("ValidationException"), "validationException")
        self.assertEqual(f("ModelErrorException"), "modelStreamErrorException")
        self.assertEqual(f("InternalServerException"), "internalServerException")
        self.assertEqual(f("overloaded_error"), "internalServerException")


class TestConverse(_Base):
    MODEL = "anthropic.claude-opus-5"

    def _converse(self, **over: Any) -> dict:
        b = {"messages": [{"role": "user", "content": [{"text": "hi"}]}],
             "inferenceConfig": {"maxTokens": 64}}
        b.update(over)
        return b

    def test_request_translation(self) -> None:
        from puppetllm.providers.converse import to_canonical
        out = to_canonical({
            "system": [{"text": "you are helpful"}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
            "messages": [
                {"role": "user", "content": [
                    {"text": "look"},
                    {"image": {"format": "png", "source": {"bytes": "QUJD"}}},
                    {"document": {"name": "doc", "format": "txt", "source": {"text": "body"}}},
                    {"cachePoint": {"type": "default"}}]},
                {"role": "assistant", "content": [
                    {"reasoningContent": {"reasoningText": {"text": "think", "signature": "sig"}}},
                    {"toolUse": {"toolUseId": "tu1", "name": "wx", "input": {"city": "Tokyo"}}}]},
                {"role": "user", "content": [
                    {"toolResult": {"toolUseId": "tu1", "status": "error",
                                    "content": [{"text": "boom"}, {"json": {"k": 1}}]}}]}],
            "inferenceConfig": {"maxTokens": 100, "temperature": 0.2, "topP": 0.9,
                                "stopSequences": ["END"]},
            "toolConfig": {"tools": [
                {"toolSpec": {"name": "wx", "description": "weather",
                              "inputSchema": {"json": {"type": "object"}}, "strict": True}},
                {"cachePoint": {"type": "default"}}],
                "toolChoice": {"tool": {"name": "wx"}}},
            "additionalModelRequestFields": {"thinking": {"type": "adaptive"},
                                             "output_config": {"effort": "low"}, "top_k": 5},
            "serviceTier": {"type": "priority"}})
        self.assertEqual(out["system"], [{"type": "text", "text": "you are helpful",
                                          "cache_control": {"type": "ephemeral", "ttl": "1h"}}])
        user = out["messages"][0]["content"]
        self.assertEqual(user[0], {"type": "text", "text": "look"})
        self.assertEqual(user[1], {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "QUJD"}})
        self.assertEqual(user[2]["type"], "document")
        self.assertEqual(user[2]["source"], {"type": "text", "media_type": "text/plain", "data": "body"})
        self.assertEqual(user[2]["cache_control"], {"type": "ephemeral"})  # cachePoint marks the previous block
        asst = out["messages"][1]["content"]
        self.assertEqual(asst[0], {"type": "thinking", "thinking": "think", "signature": "sig"})
        self.assertEqual(asst[1], {"type": "tool_use", "id": "tu1", "name": "wx",
                                   "input": {"city": "Tokyo"}})
        tr = out["messages"][2]["content"][0]
        self.assertEqual((tr["type"], tr["tool_use_id"], tr["is_error"]), ("tool_result", "tu1", True))
        self.assertEqual(tr["content"], [{"type": "text", "text": "boom"},
                                         {"type": "text", "text": '{"k": 1}'}])
        self.assertEqual((out["max_tokens"], out["temperature"], out["top_p"], out["stop_sequences"]),
                         (100, 0.2, 0.9, ["END"]))
        self.assertEqual(out["tools"][0], {"name": "wx", "input_schema": {"type": "object"},
                                           "description": "weather", "strict": True,
                                           "cache_control": {"type": "ephemeral"}})
        self.assertEqual(out["tool_choice"], {"type": "tool", "name": "wx"})
        # additionalModelRequestFields are merged into the canonical body
        self.assertEqual(out["thinking"], {"type": "adaptive"})
        self.assertEqual(out["output_config"], {"effort": "low"})
        self.assertEqual((out["top_k"], out["service_tier"]), (5, "auto"))
        # redactedContent round-trips as redacted_thinking
        out = to_canonical({"messages": [{"role": "assistant", "content": [
            {"reasoningContent": {"redactedContent": "b64"}}]}]})
        self.assertEqual(out["messages"][0]["content"][0], {"type": "redacted_thinking", "data": "b64"})

    def test_non_stream_response_shape(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, f"/model/{self.MODEL}/converse", self._converse(
                    system=[{"text": "S" * 400}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
                    additionalModelResponseFieldPaths=["/stop_sequence", "/usage/input_tokens", "/nope"]))
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual((p["provider"], p["model"], p["api"]), ("bedrock", "claude-opus-5", "converse"))
                await self._ctl(c, "/_control/respond", json={"content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "id": "tu9", "name": "f", "input": {"a": 1}}]})
                r = await t
                self.assertEqual(r.status_code, 200)
                self.assertIn("x-amzn-requestid", r.headers)
                j = r.json()
                msg = j["output"]["message"]
                self.assertEqual(msg["role"], "assistant")
                self.assertEqual(msg["content"], [
                    {"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "sig"}}},
                    {"text": "hello"},
                    {"toolUse": {"toolUseId": "tu9", "name": "f", "input": {"a": 1}}}])
                self.assertEqual(j["stopReason"], "tool_use")
                u = j["usage"]
                self.assertEqual(u["totalTokens"],
                                 u["inputTokens"] + u["cacheReadInputTokens"] + u["cacheWriteInputTokens"] + u["outputTokens"])
                self.assertGreater(u["cacheWriteInputTokens"], 0)
                self.assertEqual(u["cacheDetails"], [{"inputTokens": u["cacheWriteInputTokens"], "ttl": "1h"}])
                self.assertGreaterEqual(j["metrics"]["latencyMs"], 0)
                # JSON-pointer paths into the native response; missing ones are ignored
                self.assertEqual(sorted(j["additionalModelResponseFields"]), ["input_tokens", "stop_sequence"])
        _run(run())

    def test_stop_reason_mapping_and_cache_hit(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                body = self._converse(system=[{"text": "S" * 400}, {"cachePoint": {"type": "default"}}])
                t = await self._pending(c, f"/model/{self.MODEL}/converse", body)
                await self._ctl(c, "/_control/respond", json={"content": [], "stop_reason": "refusal"})
                self.assertEqual((await t).json()["stopReason"], "content_filtered")
                # the cachePoint is a real breakpoint: the second identical request reads it
                t = await self._pending(c, f"/model/{self.MODEL}/converse", body)
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                u = (await t).json()["usage"]
                self.assertGreater(u["cacheReadInputTokens"], 0)
                self.assertNotIn("cacheDetails", u)
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(h["cache"]["status"], "hit")
                for reason, expected in (("max_tokens", "max_tokens"), ("pause_turn", "end_turn"),
                                         ("guardrail_intervened", "guardrail_intervened")):
                    t = await self._pending(c, f"/model/{self.MODEL}/converse", self._converse())
                    await self._ctl(c, "/_control/respond", json={"content": [], "stop_reason": reason})
                    self.assertEqual((await t).json()["stopReason"], expected, reason)
        _run(run())

    def test_stream_event_sequence(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, f"/model/{self.MODEL}/converse-stream",
                                        self._converse(performanceConfig={"latency": "optimized"}))
                await self._ctl(c, "/_control/respond", json={"content": [
                    {"type": "thinking", "thinking": "T" * 100, "signature": "sig"},
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "id": "tu1", "name": "f", "input": {"k": "v" * 60}}]})
                r = await t
                self.assertEqual(r.status_code, 200)
                self.assertIn("vnd.amazon.eventstream", r.headers["content-type"])
                # (X-Amzn-Bedrock-Content-Type belongs to InvokeModelWithResponseStream,
                # not ConverseStream, so it is deliberately absent here)
                self.assertNotIn("X-Amzn-Bedrock-Content-Type", r.headers)
                evs = eventstream.decode_messages(r.content)
                names = [e["_event"] for e in evs]
                self.assertEqual(names[0], "messageStart")
                self.assertEqual(evs[0]["role"], "assistant")
                self.assertEqual(names[-2:], ["messageStop", "metadata"])
                self.assertEqual(evs[-2]["stopReason"], "tool_use")
                self.assertGreaterEqual(evs[-1]["metrics"]["latencyMs"], 0)
                self.assertEqual(evs[-1]["performanceConfig"], {"latency": "optimized"})
                self.assertGreater(evs[-1]["usage"]["outputTokens"], 0)
                # reasoning block: text deltas then the signature, index 0
                reasoning = [e for e in evs if e["_event"] == "contentBlockDelta"
                             and "reasoningContent" in e["delta"]]
                self.assertEqual("".join(e["delta"]["reasoningContent"].get("text", "") for e in reasoning),
                                 "T" * 100)
                self.assertEqual(reasoning[-1]["delta"]["reasoningContent"]["signature"], "sig")
                self.assertTrue(all(e["contentBlockIndex"] == 0 for e in reasoning))
                # tool use: contentBlockStart then partial-JSON deltas that concatenate
                starts = [e for e in evs if e["_event"] == "contentBlockStart"]
                self.assertEqual(starts[0]["start"]["toolUse"], {"toolUseId": "tu1", "name": "f"})
                parts = [e["delta"]["toolUse"]["input"] for e in evs
                         if e["_event"] == "contentBlockDelta" and "toolUse" in e["delta"]]
                self.assertGreater(len(parts), 1)
                self.assertEqual(json.loads("".join(parts)), {"k": "v" * 60})
                # every block gets a stop event, indices are contiguous
                stops = [e["contentBlockIndex"] for e in evs if e["_event"] == "contentBlockStop"]
                self.assertEqual(stops, [0, 1, 2])
        _run(run())

    def test_validation_and_errors(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                bad = [  # every one of these must be REFUSED, so post them through _must_reject
                    ({"messages": []}, "messages"),
                    ({"messages": [{"role": "bot", "content": [{"text": "x"}]}]}, "role"),
                    ({"messages": [{"role": "user", "content": "x"}]}, "content"),
                    (self._converse(toolConfig={"tools": []}), "toolConfig.tools"),
                    (self._converse(toolConfig={"tools": [{"toolSpec": {"name": "f"}}]}), "inputSchema"),
                    (self._converse(toolConfig={"tools": [{"toolSpec": {"name": "f", "inputSchema": {"json": {}}}}],
                                                "toolChoice": {"none": {}}}), "toolChoice"),
                    (self._converse(inferenceConfig=[]), "inferenceConfig"),
                    (self._converse(additionalModelRequestFields=[]), "additionalModelRequestFields"),
                    ({"messages": [{"role": "user", "content": [{"cachePoint": {"type": "default"}}]}]}, "cachePoint"),
                    ({"messages": [{"role": "user", "content": [
                        {"toolUse": {"name": "f", "input": {}}}]}]}, "toolUse"),
                    (self._converse(additionalModelResponseFieldPaths=["bad-pointer"]), "JSON pointer"),
                ]
                for body, needle in bad:
                    r = await self._must_reject(c, f"/model/{self.MODEL}/converse", body)
                    self.assertEqual(r.status_code, 400, body)
                    self.assertEqual(r.headers["x-amzn-ErrorType"], "ValidationException")
                    self.assertIn(needle, r.json()["message"], body)
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
                # a cache_control layout the Messages API rejects is a ValidationException here too
                # (5 cachePoints = 5 breakpoints; the limit is 4)
                r = await self._must_reject(c, f"/model/{self.MODEL}/converse", self._converse(messages=[
                    {"role": "user", "content": sum(
                        ([{"text": str(i)}, {"cachePoint": {"type": "default"}}] for i in range(5)), [])}]))
                self.assertEqual(r.status_code, 400)
                self.assertEqual(r.headers["x-amzn-ErrorType"], "ValidationException")
                self.assertIn("at most 4", r.json()["message"])
                # injected errors take the Bedrock shape
                t = await self._pending(c, f"/model/{self.MODEL}/converse", self._converse())
                await self._ctl(c, "/_control/error", json={"status": 429, "type": "rate_limit_error",
                                                      "message": "slow", "headers": {"retry-after": "2"}})
                r = await t
                self.assertEqual((r.status_code, r.headers["x-amzn-ErrorType"]), (429, "ThrottlingException"))
                self.assertEqual(r.json(), {"message": "slow", "__type": "ThrottlingException"})
                self.assertEqual(r.headers["retry-after"], "2")
                # clear while pending → ServiceUnavailableException
                t = await self._pending(c, f"/model/{self.MODEL}/converse-stream", self._converse())
                await self._ctl(c, "/_control/clear")
                r = await t
                self.assertEqual((r.status_code, r.headers["x-amzn-ErrorType"]),
                                 (503, "ServiceUnavailableException"))
        _run(run())

    def test_stream_mid_stream_exception(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, f"/model/{self.MODEL}/converse-stream", self._converse())
                await self._ctl(c, "/_control/error", json={
                    "status": 500, "type": "InternalServerException", "message": "boom",
                    "after_events": 2, "content": [{"type": "text", "text": "partial"}]})
                r = await t
                self.assertEqual(r.status_code, 200)
                frames = eventstream.decode_frames(r.content)
                self.assertEqual([h.get(":event-type") for h, _ in frames[:2]],
                                 ["messageStart", "contentBlockDelta"])
                self.assertEqual(frames[-1][0][":message-type"], "exception")
                self.assertEqual(frames[-1][0][":exception-type"], "internalServerException")
                self.assertEqual(json.loads(frames[-1][1]), {"message": "boom"})
        _run(run())

    def test_model_id_normalization_and_stats(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for mid in ("anthropic.claude-opus-5", "us.anthropic.claude-opus-5",
                            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/eu.anthropic.claude-opus-5"):
                    t = await self._pending(c, f"/model/{mid}/converse", self._converse())
                    p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                    self.assertEqual((p["model"], p["bedrock_model_id"]), ("claude-opus-5", mid))
                    await self._ctl(c, "/_control/auto", json={"text": "ok"})
                    await t
                st = (await c.get("/_control/stats")).json()
                self.assertEqual(list(st["by_model"]), ["claude-opus-5"])
                self.assertEqual(st["by_model"]["claude-opus-5"]["requests"], 3)
        _run(run())


class TestBedrockBatchInference(_Base):
    JOB = {"jobName": "myjob", "modelId": "anthropic.claude-opus-5",
           "roleArn": "arn:aws:iam::123456789012:role/BatchRole"}

    async def _seed(self, c, records: list[dict], name: str = "data.jsonl",
                    bucket: str = "inbkt") -> None:
        await c.put(f"/{bucket}")
        await c.put("/outbkt")
        body = "\n".join(json.dumps(r) for r in records).encode()
        await c.put(f"/{bucket}/in/{name}", content=body)

    @staticmethod
    def _invoke_record(rid: str, text: str = "q") -> dict:
        return {"recordId": rid, "modelInput": {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 16,
            "messages": [{"role": "user", "content": text}]}}

    async def _create(self, c, **over: Any):
        body = {**self.JOB,
                "inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://inbkt/in/data.jsonl"}},
                "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://outbkt/out/"}}}
        body.update(over)
        return await c.post("/model-invocation-job", json=body)

    async def _await_end(self, c, arn: str) -> dict:
        for _ in range(80):
            j = (await c.get(f"/model-invocation-job/{arn}")).json()
            if j["status"] in ("Completed", "Stopped", "Failed"):
                return j
            await asyncio.sleep(0.05)
        self.fail(f"job never ended: {j}")

    @staticmethod
    def _out_lines(bucket: str, key: str) -> list[dict]:
        data = s3.get_object(bucket, key) or b""
        return [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]

    def test_invoke_model_job_end_to_end(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1"), self._invoke_record("r2"),
                                     self._invoke_record("r3")])
                r = await self._create(c)
                self.assertEqual(r.status_code, 200)
                arn = r.json()["jobArn"]
                self.assertRegex(arn, r"^arn:aws:bedrock:[a-z0-9-]+:\d{12}:model-invocation-job/[a-z0-9]{12}$")
                job_id = arn.rsplit("/", 1)[1]
                j = (await c.get(f"/model-invocation-job/{arn}")).json()
                self.assertEqual((j["status"], j["totalRecordCount"], j["processedRecordCount"]),
                                 ("InProgress", 3, 0))
                self.assertEqual(j["modelInvocationType"], "InvokeModel")
                self.assertEqual(j["timeoutDurationInHours"], 24)
                # the bare 12-char id addresses the job too
                self.assertEqual((await c.get(f"/model-invocation-job/{job_id}")).json()["jobArn"], arn)
                # each record is an ordinary pending tagged with its recordId
                pend = (await c.get("/_control/pending")).json()["pending"]
                self.assertEqual(sorted(x["request"]["record_id"] for x in pend), ["r1", "r2", "r3"])
                self.assertTrue(all(x["request"]["job_arn"] == arn for x in pend))
                by_rid = {x["request"]["record_id"]: x["pending_id"] for x in pend}
                await self._ctl(c, "/_control/respond", json={"pending_id": by_rid["r1"],
                                                        "content": [{"type": "text", "text": "one"}]})
                await self._ctl(c, "/_control/error", json={"pending_id": by_rid["r2"], "status": 400,
                                                      "type": "invalid_request_error", "message": "bad"})
                await self._ctl(c, "/_control/auto", json={"pending_id": by_rid["r3"], "text": "three"})
                j = await self._await_end(c, arn)
                self.assertEqual(j["status"], "Completed")
                self.assertEqual((j["totalRecordCount"], j["processedRecordCount"],
                                  j["successRecordCount"], j["errorRecordCount"]), (3, 3, 2, 1))
                self.assertGreaterEqual(j["endTime"], j["submitTime"])
                # output JSONL: one line per record, modelInput echoed, modelOutput or error
                lines = self._out_lines("outbkt", f"out/{job_id}/data.jsonl.out")
                self.assertEqual(sorted(x["recordId"] for x in lines), ["r1", "r2", "r3"])
                by = {x["recordId"]: x for x in lines}
                self.assertEqual(by["r1"]["modelOutput"]["content"], [{"type": "text", "text": "one"}])
                self.assertEqual(by["r1"]["modelOutput"]["model"], "claude-opus-5")
                self.assertEqual(by["r1"]["modelInput"]["max_tokens"], 16)
                self.assertEqual(by["r2"]["error"], {"errorCode": 400, "errorMessage": "bad"})
                self.assertNotIn("modelOutput", by["r2"])
                manifest = json.loads((s3.get_object("outbkt", f"out/{job_id}/manifest.json.out") or b"{}").decode())
                self.assertEqual((manifest["totalRecordCount"], manifest["successRecordCount"],
                                  manifest["errorRecordCount"]), (3, 2, 1))
                self.assertGreater(manifest["inputTokenCount"], 0)
                self.assertGreater(manifest["outputTokenCount"], 0)
                # history/stats count the records like any other request
                st = (await c.get("/_control/stats")).json()
                self.assertEqual(st["completed_requests"], 2)
                self.assertEqual(st["error_requests"], 1)
        _run(run())

    def test_converse_type_job(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [{"recordId": "c1", "modelInput": {
                    "messages": [{"role": "user", "content": [{"text": "summarize"}]}],
                    "inferenceConfig": {"maxTokens": 32}}}])
                r = await self._create(c, modelInvocationType="Converse")
                arn = r.json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(p["api"], "converse")
                self.assertEqual(p["messages"], [{"role": "user", "content": [{"type": "text", "text": "summarize"}]}])
                self.assertEqual(p["max_tokens"], 32)
                await self._ctl(c, "/_control/auto", json={"text": "done"})
                j = await self._await_end(c, arn)
                self.assertEqual((j["status"], j["modelInvocationType"]), ("Completed", "Converse"))
                line = self._out_lines("outbkt", f"out/{job_id}/data.jsonl.out")[0]
                self.assertEqual(line["modelOutput"]["output"]["message"]["content"], [{"text": "done"}])
                self.assertEqual(line["modelOutput"]["stopReason"], "end_turn")
                self.assertIn("usage", line["modelOutput"])
        _run(run())

    def test_stop_job(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1"), self._invoke_record("r2")])
                arn = (await self._create(c)).json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                pend = (await c.get("/_control/pending")).json()["pending"]
                by_rid = {x["request"]["record_id"]: x["pending_id"] for x in pend}
                await self._ctl(c, "/_control/respond", json={"pending_id": by_rid["r1"],
                                                        "content": [{"type": "text", "text": "one"}]})
                for _ in range(40):
                    if (await c.get("/_control/bedrock_jobs")).json()["jobs"][0]["processedRecordCount"] == 1:
                        break
                    await asyncio.sleep(0.05)
                r = await c.post(f"/model-invocation-job/{arn}/stop")
                self.assertEqual((r.status_code, r.content), (200, b""))
                j = await self._await_end(c, arn)
                # AWS reports `Stopped` for a job the caller stopped, whether or not some
                # records had already been processed.
                self.assertEqual(j["status"], "Stopped")
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
                # the aborted record was never processed: no counter, no output line
                self.assertEqual((j["totalRecordCount"], j["processedRecordCount"],
                                  j["successRecordCount"], j["errorRecordCount"]), (2, 1, 1, 0))
                lines = self._out_lines("outbkt", f"out/{job_id}/data.jsonl.out")
                self.assertEqual([x["recordId"] for x in lines], ["r1"])
                self.assertIn("modelOutput", lines[0])
                manifest = json.loads((s3.get_object("outbkt", f"out/{job_id}/manifest.json.out") or b"{}").decode())
                self.assertEqual((manifest["processedRecordCount"], manifest["errorRecordCount"]), (1, 0))
                ctl = (await c.get("/_control/bedrock_jobs")).json()["jobs"][0]
                self.assertEqual((ctl["unresolved"], ctl["cancelled"]), ([], ["r2"]))
                # stopping a stopped job is an idempotent no-op (botocore retries a Stop
                # whose response it lost); stopping a COMPLETED job is the conflict
                r = await c.post(f"/model-invocation-job/{arn}/stop")
                self.assertEqual((r.status_code, r.content), (200, b""))
                r = await c.post("/model-invocation-job/aaaaaaaaaaaa/stop")
                self.assertEqual((r.status_code, r.headers["x-amzn-ErrorType"]),
                                 (404, "ResourceNotFoundException"))
        _run(run())

    def test_list_and_control_view(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                arns = []
                for name in ("alpha", "beta", "gamma"):
                    arns.append((await self._create(c, jobName=name)).json()["jobArn"])
                    await self._ctl(c, "/_control/auto", json={"text": "ok"})
                    await self._await_end(c, arns[-1])
                r = await c.get("/model-invocation-jobs")
                names = [j["jobName"] for j in r.json()["invocationJobSummaries"]]
                self.assertEqual(names, ["gamma", "beta", "alpha"])  # newest first by default
                r = await c.get("/model-invocation-jobs", params={"sortOrder": "Ascending"})
                self.assertEqual([j["jobName"] for j in r.json()["invocationJobSummaries"]][0], "alpha")
                r = await c.get("/model-invocation-jobs", params={"nameContains": "bet"})
                self.assertEqual([j["jobName"] for j in r.json()["invocationJobSummaries"]], ["beta"])
                r = await c.get("/model-invocation-jobs", params={"statusEquals": "Completed"})
                self.assertEqual(len(r.json()["invocationJobSummaries"]), 3)
                r = await c.get("/model-invocation-jobs", params={"statusEquals": "InProgress"})
                self.assertEqual(r.json()["invocationJobSummaries"], [])
                r = await c.get("/model-invocation-jobs", params={"maxResults": 2})
                self.assertEqual(len(r.json()["invocationJobSummaries"]), 2)
                token = r.json()["nextToken"]
                r = await c.get("/model-invocation-jobs", params={"maxResults": 2, "nextToken": token})
                self.assertEqual(len(r.json()["invocationJobSummaries"]), 1)
                self.assertNotIn("nextToken", r.json())
                for params in ({"statusEquals": "Nope"}, {"maxResults": 0}, {"maxResults": "x"}):
                    r = await c.get("/model-invocation-jobs", params=params)
                    self.assertEqual((r.status_code, r.headers["x-amzn-ErrorType"]),
                                     (400, "ValidationException"), params)
                ctl = (await c.get("/_control/bedrock_jobs")).json()
                self.assertEqual(ctl["count"], 3)
                self.assertEqual(ctl["jobs"][0]["unresolved"], [])
                # clear wipes the registry
                await self._ctl(c, "/_control/clear")
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
                self.assertEqual((await c.get(f"/model-invocation-job/{arns[0]}")).status_code, 404)
        _run(run())

    def test_create_validation(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                cases = [
                    ({"jobName": "has space"}, "jobName"),
                    ({"modelId": ""}, "modelId"),
                    ({"roleArn": "nope"}, "roleArn"),
                    ({"inputDataConfig": {"s3InputDataConfig": {"s3Uri": "http://x"}}}, "inputDataConfig"),
                    ({"outputDataConfig": {}}, "outputDataConfig"),
                    ({"inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://inbkt/in/data.jsonl",
                                                                "s3InputFormat": "CSV"}}}, "s3InputFormat"),
                    ({"modelInvocationType": "Nope"}, "modelInvocationType"),
                    ({"timeoutDurationInHours": 1}, "timeoutDurationInHours"),
                    ({"clientRequestToken": "has space"}, "clientRequestToken"),
                ]
                for over, needle in cases:
                    r = await self._create(c, **over)
                    self.assertEqual(r.status_code, 400, over)
                    self.assertEqual(r.headers["x-amzn-ErrorType"], "ValidationException", over)
                    self.assertIn(needle, r.json()["message"], over)
                # missing input object / no jsonl under a prefix
                r = await self._create(c, inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://inbkt/in/nope.jsonl"}})
                self.assertEqual(r.status_code, 400)
                self.assertIn("not found", r.json()["message"])
                r = await self._create(c, inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://inbkt/empty/"}})
                self.assertEqual(r.status_code, 400)
                self.assertIn("no .jsonl", r.json()["message"])
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
                # malformed JSONL rolls the whole create back
                await c.put("/inbkt/in/bad.jsonl", content=b'{"recordId": "x"}\n')
                r = await self._create(c, inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://inbkt/in/bad.jsonl"}})
                self.assertEqual(r.status_code, 400)
                self.assertIn("modelInput", r.json()["message"])
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
                # a duplicate running job name is a conflict
                r = await self._create(c)
                self.assertEqual(r.status_code, 200)
                r2 = await self._create(c)
                self.assertEqual((r2.status_code, r2.headers["x-amzn-ErrorType"]), (400, "ConflictException"))
                # clientRequestToken is idempotent
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                await self._await_end(c, r.json()["jobArn"])
                a = await self._create(c, jobName="tok", clientRequestToken="abc123")
                b = await self._create(c, jobName="tok2", clientRequestToken="abc123")
                self.assertEqual(a.json()["jobArn"], b.json()["jobArn"])
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 2)
        _run(run())

    def test_per_record_validation_becomes_error_record(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [
                    {"recordId": "ok1", "modelInput": {"anthropic_version": "bedrock-2023-05-31",
                                                       "max_tokens": 8,
                                                       "messages": [{"role": "user", "content": "q"}]}},
                    {"recordId": "bad1", "modelInput": {"max_tokens": 8,
                                                        "messages": [{"role": "user", "content": "q"}]}},
                ])
                arn = (await self._create(c)).json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                pend = (await c.get("/_control/pending")).json()["pending"]
                self.assertEqual([x["request"]["record_id"] for x in pend], ["ok1"])
                await self._ctl(c, "/_control/auto", json={"text": "fine"})
                j = await self._await_end(c, arn)
                self.assertEqual((j["status"], j["successRecordCount"], j["errorRecordCount"]),
                                 ("Completed", 1, 1))
                by = {x["recordId"]: x for x in self._out_lines("outbkt", f"out/{job_id}/data.jsonl.out")}
                self.assertEqual(by["bad1"]["error"]["errorCode"], 400)
                self.assertIn("anthropic_version", by["bad1"]["error"]["errorMessage"])
        _run(run())

    def test_multiple_input_files_and_generated_record_ids(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("a1")], name="one.jsonl")
                await c.put("/inbkt/in/two.jsonl",
                            content=json.dumps({"modelInput": {
                                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                                "messages": [{"role": "user", "content": "q"}]}}).encode())
                arn = (await self._create(c, inputDataConfig={
                    "s3InputDataConfig": {"s3Uri": "s3://inbkt/in/"}})).json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                pend = (await c.get("/_control/pending")).json()["pending"]
                rids = sorted(x["request"]["record_id"] for x in pend)
                self.assertEqual(rids[0], "a1")
                self.assertTrue(rids[1].startswith("record-"))  # recordId is generated when omitted
                for x in pend:
                    await self._ctl(c, "/_control/respond", json={"pending_id": x["pending_id"],
                                                            "content": [{"type": "text", "text": "ok"}]})
                await self._await_end(c, arn)
                # one output file per input file, both under <prefix>/<jobId>/
                keys = [o["key"] for o in s3.list_objects("outbkt", f"out/{job_id}/")]
                self.assertEqual(sorted(keys), [f"out/{job_id}/manifest.json.out",
                                                f"out/{job_id}/one.jsonl.out", f"out/{job_id}/two.jsonl.out"])
        _run(run())


class TestConverseValidationHardening(_Base):
    """Union / required-field / cache-point rules the real API enforces."""

    MODEL = "anthropic.claude-opus-5"

    def _post(self, body: dict) -> Any:
        async def run() -> Any:
            async with await self._client() as c:
                return await self._must_reject(c, f"/model/{self.MODEL}/converse", body)
        return _run(run())

    @staticmethod
    def _msgs(*blocks: dict) -> dict:
        return {"messages": [{"role": "user", "content": list(blocks)}]}

    def test_unions_require_exactly_one_member(self) -> None:
        cases = [
            # two members in one ContentBlock: the text would otherwise be silently dropped
            (self._msgs({"text": "keep me", "cachePoint": {"type": "default"}}), "exactly one"),
            (self._msgs({}), "exactly one"),
            (self._msgs({"text": "a"}, {"image": {"format": "png", "source": {
                "bytes": "QUJD", "s3Location": {"uri": "s3://b/k"}}}}), "exactly one"),
            ({"messages": [{"role": "assistant", "content": [{"reasoningContent": {
                "reasoningText": {"text": "t"}, "redactedContent": "b64"}}]}]}, "exactly one"),
            (dict(self._msgs({"text": "a"}),
                  toolConfig={"tools": [{"toolSpec": {"name": "f", "inputSchema": {"json": {}}},
                                         "cachePoint": {"type": "default"}}]}), "exactly one"),
            (dict(self._msgs({"text": "a"}),
                  toolConfig={"tools": [{"toolSpec": {"name": "f", "inputSchema": {"json": {}}}}],
                              "toolChoice": {"auto": {}, "any": {}}}), "exactly one"),
            (dict(self._msgs({"text": "a"}),
                  toolConfig={"tools": [{"toolSpec": {"name": "f", "inputSchema": {"json": {}}}}],
                              "toolChoice": "auto"}), "toolChoice"),
            ({"system": [{"text": "s", "cachePoint": {"type": "default"}}],
              "messages": [{"role": "user", "content": [{"text": "a"}]}]}, "exactly one"),
        ]
        for body, needle in cases:
            r = self._post(body)
            self.assertEqual(r.status_code, 400, body)
            self.assertEqual(r.headers["x-amzn-ErrorType"], "ValidationException")
            self.assertIn(needle, r.json()["message"], body)

    def test_required_fields_and_enums(self) -> None:
        cases = [
            (self._msgs({"toolUse": {"toolUseId": "t", "name": "f"}}), "input"),
            (self._msgs({"toolUse": {"name": "f", "input": {}}}), "toolUseId"),
            (self._msgs({"toolResult": {"toolUseId": "t"}}), "content"),
            (self._msgs({"toolResult": {"toolUseId": "t", "content": [{"text": "x"}],
                                        "status": "weird"}}), "status"),
            (self._msgs({"image": {"source": {"bytes": "QUJD"}}}), "format"),
            (self._msgs({"image": {"format": "bmp", "source": {"bytes": "QUJD"}}}), "format"),
            (self._msgs({"image": {"format": "png"}}), "source"),
            (self._msgs({"document": {"format": "txt", "source": {"text": "x"}}}), "name"),
            (self._msgs({"document": {"name": "d", "format": "zip", "source": {"text": "x"}}}), "format"),
            ({"messages": [{"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"signature": "s"}}}]}]}, "text"),
            (self._msgs({"guardContent": {"text": "not an object"}}), "guardContent"),
            (self._msgs({"guardContent": {"text": None}}), "guardContent"),
            (dict(self._msgs({"text": "a"}), inferenceConfig={"maxTokens": "abc"}), "maxTokens"),
            (dict(self._msgs({"text": "a"}), inferenceConfig={"maxTokens": -5}), "maxTokens"),
            (dict(self._msgs({"text": "a"}), inferenceConfig={"temperature": "hot"}), "temperature"),
            (dict(self._msgs({"text": "a"}), inferenceConfig={"topP": 5}), "topP"),
            (dict(self._msgs({"text": "a"}), inferenceConfig={"stopSequences": 5}), "stopSequences"),
            (dict(self._msgs({"text": "a"}), inferenceConfig={"stopSequences": {"a": 1}}), "stopSequences"),
            (dict(self._msgs({"text": "a"}), serviceTier={"type": "turbo"}), "serviceTier"),
            (dict(self._msgs({"text": "a"}), performanceConfig={"latency": "fastest"}), "performanceConfig"),
        ]
        for body, needle in cases:
            r = self._post(body)
            self.assertEqual(r.status_code, 400, body)
            self.assertIn(needle, r.json()["message"], body)

    def test_cache_point_rules(self) -> None:
        bad = [
            (self._msgs({"text": "a"}, {"cachePoint": {"type": "custom"}}), "cachePoint.type"),
            (self._msgs({"text": "a"}, {"cachePoint": {"type": "default", "ttl": "1d"}}), "ttl"),
            (self._msgs({"text": "a"}, {"cachePoint": {"type": "default"}},
                        {"cachePoint": {"type": "default"}}), "same block"),
            (self._msgs({"cachePoint": {"type": "default"}}), "must follow"),
        ]
        for body, needle in bad:
            r = self._post(body)
            self.assertEqual(r.status_code, 400, body)
            self.assertIn(needle, r.json()["message"], body)
        # a cachePoint opening a later message marks the previous message's last block
        from puppetllm.providers.converse import to_canonical
        out = to_canonical({"messages": [
            {"role": "user", "content": [{"text": "first"}]},
            {"role": "assistant", "content": [{"cachePoint": {"type": "default", "ttl": "1h"}},
                                              {"text": "second"}]}]})
        self.assertEqual(out["messages"][0]["content"][0]["cache_control"],
                         {"type": "ephemeral", "ttl": "1h"})
        self.assertNotIn("cache_control", out["messages"][1]["content"][0])
        # ... and one opening `system` marks the last tool
        out = to_canonical({
            "toolConfig": {"tools": [{"toolSpec": {"name": "f", "inputSchema": {"json": {}}}}]},
            "system": [{"cachePoint": {"type": "default"}}, {"text": "s"}],
            "messages": [{"role": "user", "content": [{"text": "a"}]}]})
        self.assertEqual(out["tools"][0]["cache_control"], {"type": "ephemeral"})

    def test_additional_response_field_paths(self) -> None:
        from puppetllm.providers.converse import additional_fields, ConverseValidationError
        native = {"stop_sequence": "END", "usage": {"input_tokens": 7},
                  "x": {"input_tokens": 9}, "content": [{"type": "text"}], "a~b": 1, "a/b": 2}
        self.assertEqual(additional_fields(native, ["/stop_sequence"]), {"stop_sequence": "END"})
        self.assertEqual(additional_fields(native, ["/content/0/type"]), {"type": "text"})
        self.assertEqual(additional_fields(native, ["/a~0b", "/a~1b"]), {"a~b": 1, "a/b": 2})
        self.assertEqual(additional_fields(native, ["/nope"]), None)
        # colliding last tokens keep both values
        got = additional_fields(native, ["/usage/input_tokens", "/x/input_tokens"])
        self.assertEqual(sorted(got.values()), [7, 9])
        self.assertEqual(len(got), 2)
        for bad in (["bad-pointer"], [""], [1], "/x", ["/a~2b"], ["/x"] * 11, ["/" + "a" * 300]):
            with self.assertRaises(ConverseValidationError, msg=bad):
                additional_fields(native, bad)

    def test_stream_carries_additional_response_fields(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, f"/model/{self.MODEL}/converse-stream", {
                    "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                    "additionalModelResponseFieldPaths": ["/stop_sequence"]})
                await self._ctl(c, "/_control/respond", json={
                    "content": [{"type": "text", "text": "x"}],
                    "stop_reason": "stop_sequence", "stop_sequence": "END"})
                evs = eventstream.decode_messages((await t).content)
                stop = [e for e in evs if e["_event"] == "messageStop"][0]
                self.assertEqual(stop["stopReason"], "stop_sequence")
                self.assertEqual(stop["additionalModelResponseFields"], {"stop_sequence": "END"})
        _run(run())


class TestMidStreamClamping(_Base):
    def test_after_events_never_completes_the_stream(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for path, body, decode in (
                        ("/v1/messages", {"model": "claude-opus-5", "max_tokens": 8, "stream": True,
                                          "messages": [{"role": "user", "content": "x"}]},
                         lambda r: [n for n, _ in _sse_events(r.text)]),
                        ("/model/anthropic.claude-opus-5/invoke-with-response-stream",
                         {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                          "messages": [{"role": "user", "content": "x"}]},
                         lambda r: [e.get("type", e.get("_exception"))
                                    for e in eventstream.decode_messages(r.content)]),
                        ("/model/anthropic.claude-opus-5/converse-stream",
                         {"messages": [{"role": "user", "content": [{"text": "x"}]}]},
                         lambda r: [e.get("_event", e.get("_exception"))
                                    for e in eventstream.decode_messages(r.content)])):
                    t = await self._pending(c, path, body)
                    await self._ctl(c, "/_control/error", json={
                        "status": 500, "message": "boom", "after_events": 999,
                        "content": [{"type": "text", "text": "partial"}]})
                    names = decode(await t)
                    self.assertNotIn("message_stop", names, path)
                    self.assertNotIn("messageStop", names, path)
                    self.assertNotIn("message_delta", names, path)
                    self.assertNotIn("metadata", names, path)
                # history records the content supplied for the partial stream
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(h["injected_error"]["after_events"], 999)
                self.assertEqual([b["type"] for b in h["injected_error"]["partial_content"]],
                                 ["text"])
                self.assertIsNone(h["response_blocks"])
        _run(run())

    def test_stream_exception_member_per_operation(self) -> None:
        from puppetllm.providers.bedrock import stream_exception_member as f
        self.assertEqual(f("ModelTimeoutException"), "modelTimeoutException")
        self.assertEqual(f("ModelTimeoutException", operation="converse"), "modelStreamErrorException")
        self.assertEqual(f("ThrottlingException", operation="converse"), "throttlingException")
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/model/anthropic.claude-opus-5/converse-stream",
                                        {"messages": [{"role": "user", "content": [{"text": "x"}]}]})
                await self._ctl(c, "/_control/error", json={"status": 408, "message": "slow",
                                                      "after_events": 1})
                frames = eventstream.decode_frames((await t).content)
                self.assertEqual(frames[-1][0][":exception-type"], "modelStreamErrorException")
        _run(run())



class TestBedrockBatchHardening(_Base):
    JOB = TestBedrockBatchInference.JOB
    _seed = TestBedrockBatchInference._seed
    _create = TestBedrockBatchInference._create
    _await_end = TestBedrockBatchInference._await_end
    _out_lines = staticmethod(TestBedrockBatchInference._out_lines)
    _invoke_record = staticmethod(TestBedrockBatchInference._invoke_record)

    def test_unexpected_record_failure_does_not_wedge_the_job(self) -> None:
        """A record whose translation blows up becomes an error record, not a 500 + a job
        that can never end."""
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("good"),
                                     {"recordId": "bad", "modelInput": {
                                         "anthropic_version": "bedrock-2023-05-31",
                                         "max_tokens": 8, "messages": 5}}])
                r = await self._create(c)
                self.assertEqual(r.status_code, 200, r.text)
                arn = r.json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                pend = (await c.get("/_control/pending")).json()["pending"]
                self.assertEqual([x["request"]["record_id"] for x in pend], ["good"])
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                j = await self._await_end(c, arn)
                self.assertEqual((j["status"], j["successRecordCount"], j["errorRecordCount"]),
                                 ("Completed", 1, 1))
                by = {x["recordId"]: x for x in self._out_lines("outbkt", f"out/{job_id}/data.jsonl.out")}
                # a malformed modelInput is client input: a 400 validation record
                self.assertEqual(by["bad"]["error"]["errorCode"], 400)
                self.assertIn("invalid modelInput", by["bad"]["error"]["errorMessage"])
        _run(run())

    def test_same_basename_in_different_folders(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("a1")], name="data.jsonl")
                await c.put("/inbkt/in/sub/data.jsonl",
                            content=json.dumps(self._invoke_record("b1")).encode())
                arn = (await self._create(c, inputDataConfig={
                    "s3InputDataConfig": {"s3Uri": "s3://inbkt/in/"}})).json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                for x in (await c.get("/_control/pending")).json()["pending"]:
                    await self._ctl(c, "/_control/respond", json={
                        "pending_id": x["pending_id"],
                        "content": [{"type": "text", "text": x["request"]["record_id"]}]})
                await self._await_end(c, arn)
                keys = sorted(o["key"] for o in s3.list_objects("outbkt", f"out/{job_id}/"))
                self.assertEqual(keys, [f"out/{job_id}/data.jsonl.out",
                                        f"out/{job_id}/manifest.json.out",
                                        f"out/{job_id}/sub/data.jsonl.out"])
                self.assertEqual([x["recordId"] for x in
                                  self._out_lines("outbkt", f"out/{job_id}/sub/data.jsonl.out")], ["b1"])
        _run(run())

    def test_input_output_bucket_and_empty_input_validation(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                await c.put("/inbkt/in/empty.jsonl", content=b"")
                cases = [
                    ({"inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://nobucket/in/x.jsonl"}}},
                     "bucket does not exist"),
                    ({"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://nobucket/out/"}}},
                     "output bucket does not exist"),
                    ({"inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://../escape/x.jsonl"}}},
                     "invalid bucket name"),
                    ({"outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://../escape/"}}},
                     "invalid bucket name"),
                    ({"inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://inbkt/in/empty.jsonl"}}},
                     "no records"),
                ]
                for over, needle in cases:
                    r = await self._create(c, **over)
                    self.assertEqual(r.status_code, 400, over)
                    self.assertIn(needle, r.json()["message"], over)
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
                import os
                self.assertFalse(os.path.exists(os.path.join(self.tmp, "..", "escape")))
        _run(run())

    def test_list_and_identifier_validation(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for params in ({"sortBy": "Name"}, {"sortOrder": "Up"}, {"nextToken": "-1"},
                               {"nextToken": "abc"}, {"nameContains": "has space"}):
                    r = await c.get("/model-invocation-jobs", params=params)
                    self.assertEqual((r.status_code, r.headers["x-amzn-ErrorType"]),
                                     (400, "ValidationException"), params)
                for ident in ("TOOSHORT", "not-an-arn", "arn:aws:bedrock:us-east-1:1:model-invocation-job/x"):
                    r = await c.get(f"/model-invocation-job/{ident}")
                    self.assertEqual((r.status_code, r.headers["x-amzn-ErrorType"]),
                                     (400, "ValidationException"), ident)
                    r = await c.post(f"/model-invocation-job/{ident}/stop")
                    self.assertEqual(r.status_code, 400, ident)
                # a well-formed but unknown id is still a 404
                r = await c.get("/model-invocation-job/aaaaaaaaaaaa")
                self.assertEqual(r.status_code, 404)
        _run(run())

    def test_batch_records_never_observe_the_cache(self) -> None:
        """AWS: prompt caching is not supported with the batch inference API. A record that
        asks for it is refused up front; and should that refusal ever be relaxed, the
        registration itself still runs with the cache disabled — which is only observable
        with the refusal switched off, so this test switches it off."""
        async def run() -> None:
            async with await self._client() as c:
                system = [{"type": "text", "text": "S" * 400,
                           "cache_control": {"type": "ephemeral"}}]
                for expect in ("miss", "hit"):  # first request writes the prefix, second reads it
                    warm = await self._pending(c, "/model/anthropic.claude-opus-5/invoke", {
                        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                        "system": system, "messages": [{"role": "user", "content": "q"}]})
                    await self._ctl(c, "/_control/auto", json={"text": "ok"})
                    await warm
                    warmed = (await c.get("/_control/history")).json()["history"][-1]
                    self.assertEqual(warmed["cache"]["status"], expect)
                self.assertGreater(warmed["usage"]["cache_read_input_tokens"], 0)  # a hit is possible
                cached = {"recordId": "r1", "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                    "system": system, "messages": [{"role": "user", "content": "q"}]}}
                plain = {"recordId": "r2", "modelInput": {
                    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                    "system": [{"type": "text", "text": "S" * 400}],
                    "messages": [{"role": "user", "content": "q"}]}}
                await self._seed(c, [cached, plain])
                arn = (await self._create(c)).json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                j = await self._await_end(c, arn)
                self.assertEqual((j["processedRecordCount"], j["errorRecordCount"]), (2, 1))
                out = {ln["recordId"]: ln for ln in
                       self._out_lines("outbkt", f"out/{job_id}/data.jsonl.out")}
                self.assertEqual(out["r1"]["error"]["errorCode"], 400)
                self.assertIn("prompt caching", out["r1"]["error"]["errorMessage"])
                self.assertNotIn("modelOutput", out["r1"])
                self.assertIn("modelOutput", out["r2"])
                # r2 shares the warmed prefix byte for byte, and still saw no cache
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(h["request"]["record_id"], "r2")
                self.assertEqual(h["cache"]["status"], "none")
                self.assertEqual((h["usage"]["cache_read_input_tokens"],
                                  h["usage"]["cache_creation_input_tokens"]), (0, 0))
                # now the defence in depth: with the per-record refusal switched off, a
                # marker-bearing record registers and STILL never touches the cache
                from puppetllm.providers import bedrock_batch as bb
                real = bb._reject_unsupported
                bb._reject_unsupported = lambda *_a, **_k: None
                self.addCleanup(setattr, bb, "_reject_unsupported", real)
                await self._seed(c, [cached])
                arn = (await self._create(c, jobName="nocache2")).json()["jobArn"]
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                j = await self._await_end(c, arn)
                self.assertEqual((j["status"], j["successRecordCount"]), ("Completed", 1))
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(h["request"]["record_id"], "r1")
                self.assertEqual(h["cache"]["status"], "none")   # warmed prefix + marker, no hit
                self.assertEqual((h["usage"]["cache_read_input_tokens"],
                                  h["usage"]["cache_creation_input_tokens"]), (0, 0))
        _run(run())

    def test_converse_cache_point_is_refused_per_record(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                rec = {"recordId": "r1", "modelInput": {
                    "messages": [{"role": "user", "content": [{"text": "q"},
                                                              {"cachePoint": {"type": "default"}}]}]}}
                await self._seed(c, [rec])
                arn = (await self._create(c, modelInvocationType="Converse")).json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                j = await self._await_end(c, arn)
                self.assertEqual((j["processedRecordCount"], j["errorRecordCount"]), (1, 1))
                line = self._out_lines("outbkt", f"out/{job_id}/data.jsonl.out")[0]
                self.assertEqual(line["error"]["errorCode"], 400)
                self.assertIn("prompt caching", line["error"]["errorMessage"])
        _run(run())


class TestS3PaginationAndPathSafety(_Base):
    """Listing pagination edge cases and path confinement."""

    def test_delimiter_pagination_advances(self) -> None:
        """A truncated page ending on a CommonPrefixes entry must not repeat it."""
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/pag")
                for key in ("a/1.txt", "a/2.txt", "b/1.txt", "c.txt"):
                    await c.put(f"/pag/{key}", content=b"x")
                import re as _re
                seen, token, pages = [], None, 0
                while True:
                    params = {"list-type": "2", "delimiter": "/", "max-keys": 1}
                    if token:
                        params["continuation-token"] = token
                    r = await c.get("/pag", params=params)
                    seen += (_re.findall(r"<CommonPrefixes><Prefix>([^<]+)<", r.text)
                             + _re.findall(r"<Key>([^<]+)</Key>", r.text))
                    pages += 1
                    m = _re.search(r"<NextContinuationToken>([^<]+)<", r.text)
                    self.assertLess(pages, 8, f"paginator looped: {seen}")
                    if not m:
                        break
                    self.assertNotEqual(m.group(1), token, "same token twice")
                    token = m.group(1)
                self.assertEqual(seen, ["a/", "b/", "c.txt"])
                # max-keys=0 is not "truncated"
                r = await c.get("/pag", params={"list-type": "2", "max-keys": 0})
                self.assertIn("<IsTruncated>false</IsTruncated>", r.text)
        _run(run())

    def test_symlink_cannot_escape_the_store(self) -> None:
        import os
        os.makedirs(os.path.join(self.tmp, "sbkt"), exist_ok=True)
        outside = os.path.join(self.tmp, "..", "puppetllm-escape-probe")
        os.makedirs(outside, exist_ok=True)
        link = os.path.join(self.tmp, "sbkt", "link")
        if not os.path.islink(link):
            os.symlink(outside, link)
        try:
            with self.assertRaises(s3.S3Error):
                s3.put_object("sbkt", "link/pwned.txt", b"x")
            self.assertFalse(os.path.exists(os.path.join(outside, "pwned.txt")))
            async def run() -> None:
                async with await self._client() as c:
                    r = await c.put("/sbkt/link/pwned.txt", content=b"x")
                    self.assertEqual(r.status_code, 400)
                    self.assertIn("InvalidArgument", r.text)
            _run(run())
        finally:
            os.remove(link)
            os.rmdir(outside)

    def test_converse_accepts_guard_image_and_empty_tool_result(self) -> None:
        from puppetllm.providers.converse import to_canonical
        out = to_canonical({"messages": [
            {"role": "user", "content": [{"guardContent": {"image": {
                "format": "png", "source": {"bytes": "QUJD"}}}}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "t", "content": []}}]}]})
        self.assertEqual(out["messages"][0]["content"][0]["type"], "image")
        self.assertEqual(out["messages"][1]["content"][0]["content"], [])
        # stopSequences entries must be non-empty, as AWS documents
        from puppetllm.providers.converse import ConverseValidationError
        with self.assertRaises(ConverseValidationError):
            to_canonical({"messages": [{"role": "user", "content": [{"text": "x"}]}],
                          "inferenceConfig": {"stopSequences": [""]}})

    def test_pointer_syntax_checked_past_a_missing_member(self) -> None:
        from puppetllm.providers.converse import additional_fields, ConverseValidationError
        with self.assertRaises(ConverseValidationError):
            additional_fields({"a": 1}, ["/missing/~2"])

    def test_batch_rejects_features_batch_does_not_support(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/inbkt")
                await c.put("/outbkt")
                recs = [
                    {"recordId": "tools", "modelInput": {
                        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                        "messages": [{"role": "user", "content": "q"}],
                        "tools": [{"name": "f", "input_schema": {"type": "object"}}]}},
                    {"recordId": "fmt", "modelInput": {
                        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                        "messages": [{"role": "user", "content": "q"}],
                        "output_config": {"format": {"type": "json_schema"}}}},
                    {"recordId": "ok", "modelInput": {
                        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                        "messages": [{"role": "user", "content": "q"}]}},
                ]
                await c.put("/inbkt/in/data.jsonl",
                            content=("\n".join(json.dumps(r) for r in recs)).encode())
                r = await c.post("/model-invocation-job", json={
                    "jobName": "unsup", "modelId": "anthropic.claude-opus-5",
                    "roleArn": "arn:aws:iam::123456789012:role/R",
                    "inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://inbkt/in/data.jsonl"}},
                    "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://outbkt/out/"}}})
                arn = r.json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                pend = (await c.get("/_control/pending")).json()["pending"]
                self.assertEqual([x["request"]["record_id"] for x in pend], ["ok"])
                await self._ctl(c, "/_control/auto", json={"text": "fine"})
                for _ in range(80):
                    j = (await c.get(f"/model-invocation-job/{arn}")).json()
                    if j["status"] == "Completed":
                        break
                    await asyncio.sleep(0.05)
                self.assertEqual((j["successRecordCount"], j["errorRecordCount"]), (1, 2))
                lines = {x["recordId"]: x for x in
                         [json.loads(ln) for ln in
                          (s3.get_object("outbkt", f"out/{job_id}/data.jsonl.out") or b"").decode().splitlines()]}
                self.assertEqual(lines["tools"]["error"]["errorCode"], 400)
                self.assertIn("tool calling", lines["tools"]["error"]["errorMessage"])
                self.assertIn("structured output", lines["fmt"]["error"]["errorMessage"])
        _run(run())

    def test_mid_stream_error_has_no_error_headers(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {
                    "model": "claude-opus-5", "max_tokens": 8, "stream": True,
                    "messages": [{"role": "user", "content": "x"}]})
                await self._ctl(c, "/_control/error", json={
                    "status": 429, "message": "slow", "after_events": 1,
                    "headers": {"retry-after": "3"}})
                r = await t
                self.assertEqual(r.status_code, 200)
                self.assertNotIn("retry-after", r.headers)   # nonsense on a 200
        _run(run())


class TestS3Emulation(_Base):
    def test_object_lifecycle_and_listing(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                self.assertEqual((await c.head("/mybucket")).status_code, 404)
                self.assertEqual((await c.put("/mybucket")).status_code, 200)
                self.assertEqual((await c.head("/mybucket")).status_code, 200)
                r = await c.put("/mybucket/in/a.jsonl", content=b'{"recordId":"1"}\n')
                self.assertEqual(r.status_code, 200)
                self.assertTrue(r.headers["ETag"].startswith('"'))
                r = await c.put("/mybucket/in/b.jsonl", content=b"bbb")
                r = await c.get("/mybucket/in/a.jsonl")
                self.assertEqual((r.status_code, r.content), (200, b'{"recordId":"1"}\n'))
                r = await c.head("/mybucket/in/a.jsonl")
                self.assertEqual((r.status_code, r.headers["Content-Length"]), (200, "17"))
                self.assertIn("Last-Modified", r.headers)
                r = await c.get("/mybucket/nope")
                self.assertEqual(r.status_code, 404)
                self.assertIn("<Code>NoSuchKey</Code>", r.text)
                r = await c.get("/mybucket", params={"list-type": "2", "prefix": "in/"})
                self.assertEqual(r.status_code, 200)
                self.assertIn("<KeyCount>2</KeyCount>", r.text)
                self.assertIn("<Key>in/a.jsonl</Key>", r.text)
                self.assertIn("<IsTruncated>false</IsTruncated>", r.text)
                r = await c.get("/mybucket", params={"list-type": "2", "max-keys": "1"})
                self.assertIn("<IsTruncated>true</IsTruncated>", r.text)
                self.assertIn("<NextContinuationToken>in/a.jsonl</NextContinuationToken>", r.text)
                r = await c.get("/nobucket", params={"list-type": "2"})
                self.assertEqual(r.status_code, 404)
                self.assertIn("NoSuchBucket", r.text)
                self.assertEqual((await c.delete("/mybucket/in/b.jsonl")).status_code, 204)
                self.assertEqual((await c.get("/mybucket/in/b.jsonl")).status_code, 404)
                # reserved segments are API paths, not buckets: a plain 404, not S3 XML
                r = await c.put("/model")
                self.assertEqual(r.status_code, 404)
                self.assertNotIn("<Code>", r.text)
                self.assertEqual((await c.get("/_control/health")).status_code, 200)
                self.assertEqual((await c.put("/mybucket/../x", content=b"x")).status_code, 400)
        _run(run())

    def test_aws_chunked_upload_decoded(self) -> None:
        import base64 as _b64, zlib as _z
        crc = _b64.b64encode(_z.crc32(b"hello world").to_bytes(4, "big")).decode()
        body = (b"5;chunk-signature=abc\r\nhello\r\n"
                b"6;chunk-signature=def\r\n world\r\n"
                b"0;chunk-signature=000\r\n"
                b"x-amz-checksum-crc32:" + crc.encode() + b"\r\n\r\n")
        self.assertEqual(s3.decode_aws_chunked(body),
                         (b"hello world", {"x-amz-checksum-crc32": crc}))
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/bkt")
                r = await c.put("/bkt/k", content=body,
                                headers={"content-encoding": "aws-chunked",
                                         "x-amz-decoded-content-length": "11"})
                self.assertEqual(r.status_code, 200)
                self.assertEqual((await c.get("/bkt/k")).content, b"hello world")
                # a malformed chunked body is an S3 400, not a 500
                for bad in (b"zzz", b"ffffffffffffffff\r\nab\r\n", b"5;sig\r\nhel"):
                    r = await c.put("/bkt/k2", content=bad,
                                    headers={"content-encoding": "aws-chunked"})
                    self.assertEqual(r.status_code, 400, bad)
                    self.assertIn("InvalidRequest", r.text)
        _run(run())

    def test_store_helpers(self) -> None:
        self.assertEqual(s3.parse_s3_uri("s3://bkt/in/data.jsonl"), ("bkt", "in/data.jsonl"))
        self.assertEqual(s3.parse_s3_uri("s3://bkt/out/"), ("bkt", "out/"))
        for bad in ("https://bkt/x", "s3://../escape/", "s3://BKT/x", "s3://ab/x",
                    "s3://bkt/a/../../x", "s3://model/x"):
            with self.assertRaises(ValueError, msg=bad):
                s3.parse_s3_uri(bad)
        # a write needs its bucket, like real S3
        with self.assertRaises(s3.S3Error):
            s3.put_object("bkt", "a/b.txt", b"1")
        s3.create_bucket("bkt")
        s3.put_object("bkt", "a/b.txt", b"1")
        self.assertEqual(s3.get_object("bkt", "a/b.txt"), b"1")
        self.assertEqual([o["key"] for o in s3.list_objects("bkt", "a/")], ["a/b.txt"])
        # nothing can be written outside the store root
        for bucket in ("..", "../etc", "model"):
            with self.assertRaises(s3.S3Error, msg=bucket):
                s3.put_object(bucket, "x", b"1")
        import os
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "..", "x")))

    def test_list_objects_v2_delimiter_and_paging(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/bkt1")
                for key in ("a/1.txt", "a/2.txt", "b/1.txt", "top.txt"):
                    await c.put(f"/bkt1/{key}", content=b"x")
                r = await c.get("/bkt1", params={"list-type": "2", "delimiter": "/"})
                self.assertIn("<CommonPrefixes><Prefix>a/</Prefix></CommonPrefixes>", r.text)
                self.assertIn("<CommonPrefixes><Prefix>b/</Prefix></CommonPrefixes>", r.text)
                self.assertIn("<Key>top.txt</Key>", r.text)
                self.assertNotIn("<Key>a/1.txt</Key>", r.text)
                self.assertIn("<Delimiter>/</Delimiter>", r.text)
                # a delimiter applies after `prefix`
                r = await c.get("/bkt1", params={"list-type": "2", "prefix": "a/", "delimiter": "/"})
                self.assertIn("<Key>a/1.txt</Key>", r.text)
                self.assertNotIn("CommonPrefixes", r.text)
                # page 2 really continues where page 1 stopped
                r = await c.get("/bkt1", params={"list-type": "2", "max-keys": 2})
                import re as _re
                first = _re.findall(r"<Key>([^<]+)</Key>", r.text)
                self.assertEqual(len(first), 2)
                token = _re.search(r"<NextContinuationToken>([^<]+)<", r.text).group(1)
                r = await c.get("/bkt1", params={"list-type": "2", "max-keys": 2,
                                                 "continuation-token": token})
                second = _re.findall(r"<Key>([^<]+)</Key>", r.text)
                self.assertEqual(first + second, ["a/1.txt", "a/2.txt", "b/1.txt", "top.txt"])
                self.assertIn("<IsTruncated>false</IsTruncated>", r.text)
                # max-keys validation
                self.assertIn("<KeyCount>0</KeyCount>",
                              (await c.get("/bkt1", params={"list-type": "2", "max-keys": 0})).text)
                for bad in ("abc", "1.5", "-1"):
                    r = await c.get("/bkt1", params={"list-type": "2", "max-keys": bad})
                    self.assertEqual(r.status_code, 400, bad)
                    self.assertIn("InvalidArgument", r.text)
                # the service never returns more than 1000 keys but does not reject a
                # larger request, so a caller passing MaxKeys=2000 must keep working
                r = await c.get("/bkt1", params={"list-type": "2", "max-keys": "5000"})
                self.assertEqual(r.status_code, 200)
                self.assertIn("<MaxKeys>1000</MaxKeys>", r.text)
        _run(run())

    def test_raw_path_traversal_and_encoding(self) -> None:
        """httpx normalizes `/b/../x` client-side, so drive the ASGI app directly."""
        async def call(path: str, method: str = "PUT") -> tuple[int, bytes]:
            status, body = {}, bytearray()
            async def receive():
                return {"type": "http.request", "body": b"x", "more_body": False}
            async def send(msg):
                if msg["type"] == "http.response.start":
                    status["code"] = msg["status"]
                else:
                    body.extend(msg.get("body", b""))
            await self.mod.app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                                "method": method, "path": path, "raw_path": path.encode(),
                                "root_path": "", "scheme": "http", "query_string": b"",
                                "headers": [(b"host", b"test")], "client": ("t", 1),
                                "server": ("t", 80)}, receive, send)
            return status["code"], bytes(body)
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/bkt1")
            # An ASGI server hands us the DECODED path, so `%2e%2e` arrives as `..`.
            for path in ("/bkt1/../x", "/bkt1/a/../../x", "/bkt1/./x", "/bkt1/a/.."):
                code, body = await call(path)
                self.assertEqual(code, 400, path)
                self.assertIn(b"InvalidArgument", body, path)
            import os
            self.assertFalse(os.path.exists(os.path.join(self.tmp, "..", "x")))
            # a percent-encoded slash stays part of the key (no second decode)
            async with await self._client() as c:
                r = await c.put("/bkt1/a%252Fb", content=b"v")
                self.assertEqual(r.status_code, 200)
                listing = (await c.get("/bkt1", params={"list-type": "2"})).text
                self.assertIn("a%2Fb", listing)
                self.assertNotIn("<Key>a/b</Key>", listing)
        _run(run())

    def test_reserved_segments_never_shadow_an_api_route(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # Every API path still reaches its own handler: an unparsable body makes each
                # route answer with ITS OWN 400 envelope (never S3 XML) without creating a pending.
                probes = ["/v1/messages", "/anthropic/v1/messages",
                          "/model/anthropic.claude-opus-5/invoke",
                          "/model/anthropic.claude-opus-5/invoke-with-response-stream",
                          "/model/anthropic.claude-opus-5/converse",
                          "/model/anthropic.claude-opus-5/converse-stream",
                          "/v1/chat/completions", "/v1/messages/batches", "/model-invocation-job"]
                for path in probes:
                    r = await c.post(path, content=b"{not json",
                                     headers={"content-type": "application/json"})
                    self.assertEqual(r.status_code, 400, path)
                    self.assertNotIn("<Code>", r.text, path)      # never an S3 XML error
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
                for path in ("/_control/health", "/_control/pending", "/_control/bedrock_jobs",
                             "/model-invocation-jobs"):
                    r = await c.get(path)
                    self.assertEqual(r.status_code, 200, path)
                # ... while a reserved first segment is answered as a missing API path
                # (an S3 InvalidBucketName envelope there would only mislead)
                for name in ("model", "v1", "anthropic", "_control", "model-invocation-job",
                             "docs", "openapi.json"):
                    for path in (f"/{name}", f"/{name}/x"):
                        r = await c.put(path)
                        self.assertIn(r.status_code, (404, 405), path)
                        self.assertNotIn("<Code>", r.text, path)
                        if r.status_code == 405:
                            self.assertIn("allow", r.headers, path)
                # a wrong method on a real API path is a proper 405 with `Allow`, not the
                # catch-all's 404 and never an S3 error envelope
                for method, path, allow in (
                        ("GET", "/model/anthropic.claude-opus-5/converse", "POST"),
                        ("GET", "/model/anthropic.claude-opus-5/invoke", "POST"),
                        ("DELETE", "/v1/messages", "POST"),
                        ("POST", "/model-invocation-jobs", "GET, HEAD")):
                    r = await c.request(method, path)
                    self.assertEqual(r.status_code, 405, path)
                    self.assertEqual(r.headers["allow"], allow, path)
                    self.assertNotIn("<Code>", r.text, path)
                # FastAPI's own routes keep working (they are registered before the
                # catch-all and are reserved so a bucket cannot take their names)
                for path in ("/docs", "/openapi.json"):
                    self.assertEqual((await c.get(path)).status_code, 200, path)
                # a genuinely bad bucket name is still an S3 error
                r = await c.put("/AB")
                self.assertEqual(r.status_code, 400)
                self.assertIn("InvalidBucketName", r.text)
        _run(run())


class TestS3RequestSemantics(_Base):
    """Request semantics a real boto3 client depends on."""

    async def _bucket(self, c, name: str = "sem") -> None:
        await c.put(f"/{name}")

    def test_ranged_get(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._bucket(c)
                body = bytes(range(256)) * 4  # 1024 bytes
                await c.put("/sem/obj.bin", content=body)
                for header, expect_first, expect_last in (
                        ("bytes=0-9", 0, 9), ("bytes=10-", 10, 1023), ("bytes=-16", 1008, 1023),
                        ("bytes=0-100000", 0, 1023)):
                    r = await c.get("/sem/obj.bin", headers={"range": header})
                    self.assertEqual(r.status_code, 206, header)
                    self.assertEqual(r.content, body[expect_first:expect_last + 1], header)
                    self.assertEqual(r.headers["content-range"],
                                     f"bytes {expect_first}-{expect_last}/1024", header)
                    self.assertEqual(r.headers["content-length"],
                                     str(expect_last - expect_first + 1), header)
                # past the end is 416, as real S3 answers
                r = await c.get("/sem/obj.bin", headers={"range": "bytes=2048-3000"})
                self.assertEqual(r.status_code, 416)
                self.assertIn("InvalidRange", r.text)
                # no Range (and a range form this store does not serve) is the whole object
                for headers in ({}, {"range": "bytes=0-1,5-6"}, {"range": "items=0-1"}):
                    r = await c.get("/sem/obj.bin", headers=headers)
                    self.assertEqual((r.status_code, r.content), (200, body), headers)
        _run(run())

    def test_head_reports_size_without_a_body(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._bucket(c)
                await c.put("/sem/h.bin", content=b"x" * 5000)
                r = await c.head("/sem/h.bin")
                self.assertEqual((r.status_code, r.content), (200, b""))
                self.assertEqual(r.headers["content-length"], "5000")
                self.assertEqual(r.headers["accept-ranges"], "bytes")
        _run(run())

    def test_multipart_upload_is_a_not_implemented_s3_error(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._bucket(c)
                for path, params in (("/sem/big.bin", {"uploads": ""}),
                                     ("/sem/big.bin", {"uploadId": "1", "partNumber": "1"})):
                    r = await c.post(path, params=params)
                    self.assertEqual(r.status_code, 501, params)
                    self.assertIn("NotImplemented", r.text, params)
                    self.assertIn("multipart_threshold", r.text, params)
                # a plain POST on an object is refused too, without the multipart advice
                r = await c.post("/sem/big.bin")
                self.assertEqual(r.status_code, 501)
                self.assertNotIn("multipart_threshold", r.text)
                r = await c.post("/sem", params={"delete": ""})
                self.assertEqual(r.status_code, 501)
        _run(run())

    def test_control_characters_in_a_key_are_refused(self) -> None:
        """One such key would otherwise make every later listing unparsable XML."""
        async def run() -> None:
            from xml.etree import ElementTree
            async with await self._client() as c:
                await self._bucket(c)
                await c.put("/sem/ok.txt", content=b"1")
                for key in ("bell%07key.txt", "del%7Fkey.txt", "esc%1bkey.txt"):
                    r = await c.put(f"/sem/{key}", content=b"1")
                    self.assertEqual(r.status_code, 400, key)
                    self.assertIn("InvalidArgument", r.text, key)
                # the store refuses them directly too (the batch code writes through it)
                for key in ("nl\nkey.txt", "nul\0key.txt", "cr\rkey.txt"):
                    with self.assertRaises(s3.S3Error, msg=key):
                        s3.put_object("sem", key, b"1")
                for params in ({"list-type": "2"}, {}):
                    body = (await c.get("/sem", params=params)).text
                    ElementTree.fromstring(body)  # raises if the store emitted broken XML
                    self.assertIn("<Key>ok.txt</Key>", body)
        _run(run())

    def test_a_key_is_never_silently_rewritten(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._bucket(c)
                # real S3 stores this under the key `/a`; a directory-backed store cannot,
                # and quietly writing `a` instead would hide the difference
                r = await c.put("/sem//a", content=b"v")
                self.assertEqual(r.status_code, 400)
                self.assertIn("InvalidArgument", r.text)
                self.assertIsNone(s3.get_object("sem", "a"))
        _run(run())

    def test_trailing_slash_lists_the_bucket(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._bucket(c)
                await c.put("/sem/k.txt", content=b"v")
                r = await c.get("/sem/")
                self.assertEqual(r.status_code, 200)
                self.assertIn("<Key>k.txt</Key>", r.text)
        _run(run())

    def test_bucket_names_match_the_real_rules(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for bad in ("a..b", "192.168.0.1", "AB", "a", "x" * 64, "-lead", "trail-"):
                    r = await c.put(f"/{bad}")
                    self.assertEqual(r.status_code, 400, bad)
                    self.assertIn("InvalidBucketName", r.text, bad)
                self.assertEqual((await c.put("/a.valid-name9")).status_code, 200)
        _run(run())

    def test_overwrite_is_atomic_and_the_etag_follows_it(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._bucket(c)
                await c.put("/sem/o.bin", content=b"a" * 100)
                first = (await c.head("/sem/o.bin")).headers["etag"]
                await c.put("/sem/o.bin", content=b"b" * 200)
                r = await c.get("/sem/o.bin")
                self.assertEqual(r.content, b"b" * 200)
                self.assertNotEqual(r.headers["etag"], first)
                # the listing agrees with the object (size and ETag from one moment)
                listing = (await c.get("/sem", params={"list-type": "2"})).text
                self.assertIn("<Size>200</Size>", listing)
                self.assertIn(f"<ETag>{r.headers['etag'].replace(chr(34), '&quot;')}</ETag>",
                              listing.replace(chr(34), "&quot;"))
                # no temporary files left behind
                import os as _os
                self.assertEqual([n for n in _os.listdir(_os.path.join(self.tmp, "sem"))],
                                 ["o.bin"])
        _run(run())


class TestBatchLifecycleHardening(_Base):
    """Races and configuration shapes around the batch job lifecycle."""

    JOB = TestBedrockBatchInference.JOB
    _seed = TestBedrockBatchInference._seed
    _create = TestBedrockBatchInference._create
    _await_end = TestBedrockBatchInference._await_end
    _invoke_record = staticmethod(TestBedrockBatchInference._invoke_record)
    _out_lines = staticmethod(TestBedrockBatchInference._out_lines)

    def test_a_cancelled_finalization_still_reaches_a_terminal_status(self) -> None:
        """Cancelling the request task mid-write (client disconnect, shutdown, wait_for)
        must not leave the job wedged with nothing left to finish it."""
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                arn = (await self._create(c)).json()["jobArn"]
                real_write = bb._write_outputs
                started = threading.Event()

                def slow_write(job):
                    started.set()
                    import time as _t
                    _t.sleep(1.0)
                    real_write(job)

                bb._write_outputs = slow_write
                self.addCleanup(setattr, bb, "_write_outputs", real_write)
                pend = (await c.get("/_control/pending")).json()["pending"][0]
                await self._ctl(c, "/_control/respond", json={
                    "pending_id": pend["pending_id"],
                    "content": [{"type": "text", "text": "x"}]})
                await asyncio.to_thread(started.wait, 5)   # the write is in flight
                stopper = asyncio.create_task(
                    c.post(f"/model-invocation-job/{arn}/stop", timeout=10))
                await asyncio.sleep(0.05)
                stopper.cancel()
                try:
                    await stopper
                except asyncio.CancelledError:
                    pass
                j = await self._await_end(c, arn)
                self.assertIn(j["status"], ("Stopped", "Completed", "Failed"))
                self.assertNotIn(j["status"], ("Stopping", "InProgress"))
        _run(run())

    def test_a_stop_accepted_during_finalization_still_wins(self) -> None:
        """The terminal status must not settle back to `Completed`."""
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                arn = (await self._create(c)).json()["jobArn"]
                real_write = bb._write_outputs
                started = threading.Event()

                def slow_write(job):
                    started.set()
                    import time as _t
                    _t.sleep(0.6)
                    real_write(job)

                bb._write_outputs = slow_write
                self.addCleanup(setattr, bb, "_write_outputs", real_write)
                pend = (await c.get("/_control/pending")).json()["pending"][0]
                await self._ctl(c, "/_control/respond", json={
                    "pending_id": pend["pending_id"],
                    "content": [{"type": "text", "text": "x"}]})
                await asyncio.to_thread(started.wait, 5)
                r = await c.post(f"/model-invocation-job/{arn}/stop")
                self.assertEqual(r.status_code, 200)
                j = await self._await_end(c, arn)
                self.assertEqual(j["status"], "Stopped")
        _run(run())

    def test_clear_during_creation_registers_nothing_afterwards(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record(f"r{i}") for i in range(600)])
                task = asyncio.create_task(self._create(c))
                for _ in range(200):
                    if (await c.get("/_control/pending")).json()["count"] > 0:
                        break
                    await asyncio.sleep(0.001)
                await self._ctl(c, "/_control/clear")
                r = await task
                # A modeled error of CreateModelInvocationJob that botocore does NOT retry:
                # a retryable 5xx would make the SDK quietly re-create the job the human
                # just cleared.
                self.assertEqual(r.status_code, 400)
                self.assertEqual(r.headers["x-amzn-ErrorType"], "ConflictException")
                # nothing the creation loop made survives the clear
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 0)
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
        _run(run())

    def test_list_paging_params_are_strictly_integers(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for params in ({"maxResults": "1_0"}, {"maxResults": " 5 "},
                               {"maxResults": "+5"}, {"maxResults": "0"},
                               {"nextToken": "-1"}, {"nextToken": "1_0"}):
                    r = await c.get("/model-invocation-jobs", params=params)
                    self.assertEqual(r.status_code, 400, params)
                self.assertEqual((await c.get("/model-invocation-jobs")).status_code, 200)
        _run(run())

    def test_job_configuration_shapes_are_validated(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                for over in ({"vpcConfig": "broken"},
                             {"vpcConfig": {"subnetIds": [], "securityGroupIds": ["sg-1"]}},
                             {"vpcConfig": {"subnetIds": ["s"], "securityGroupIds": ["s"],
                                            "extra": 1}},
                             {"tags": [{"key": "k"}]},
                             {"tags": "nope"},
                             {"inputDataConfig": {"s3InputDataConfig": {
                                 "s3Uri": "s3://inbkt/in/data.jsonl", "s3BucketOwner": "12"}}}):
                    r = await self._create(c, **over)
                    self.assertEqual(r.status_code, 400, over)
                    self.assertEqual(r.headers["x-amzn-ErrorType"], "ValidationException", over)
                r = await self._create(c, vpcConfig={"subnetIds": ["subnet-1"],
                                                     "securityGroupIds": ["sg-1"]},
                                       tags=[{"key": "team", "value": "x"}])
                self.assertEqual(r.status_code, 200)
                j = (await c.get(f"/model-invocation-job/{r.json()['jobArn']}")).json()
                self.assertEqual(j["vpcConfig"]["subnetIds"], ["subnet-1"])
        _run(run())

    def test_converse_jobs_accept_any_model_but_invoke_model_does_not(self) -> None:
        """`Converse` is a model-independent schema, so a Nova batch job is valid."""
        async def run() -> None:
            async with await self._client() as c:
                rec = {"recordId": "r1", "modelInput": {
                    "messages": [{"role": "user", "content": [{"text": "q"}]}]}}
                await self._seed(c, [rec])
                r = await self._create(c, modelId="amazon.nova-pro-v1:0",
                                       modelInvocationType="Converse")
                self.assertEqual(r.status_code, 200, r.text)
                arn = r.json()["jobArn"]
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                j = await self._await_end(c, arn)
                self.assertEqual((j["status"], j["successRecordCount"]), ("Completed", 1))
                job_id = arn.rsplit("/", 1)[1]
                line = self._out_lines("outbkt", f"out/{job_id}/data.jsonl.out")[0]
                self.assertEqual(line["modelOutput"]["output"]["message"]["content"],
                                 [{"text": "ok"}])
                # an InvokeModel body is model-specific, so that one is still refused
                r = await self._create(c, jobName="nova2", modelId="amazon.nova-pro-v1:0")
                self.assertEqual(r.status_code, 400)
                self.assertIn("InvokeModel", r.json()["message"])
        _run(run())


class TestConverseUnionMembers(_Base):
    """Each `*Source` union has its own members; a generic check would accept nonsense."""

    async def _reject(self, c, block: dict, expect: str) -> None:
        r = await self._must_reject(c, "/model/anthropic.claude-opus-5/converse", {
            "messages": [{"role": "user", "content": [block]}]})
        self.assertEqual(r.status_code, 400, block)
        self.assertIn(expect, r.json()["message"], block)

    def test_image_and_video_sources_reject_document_only_members(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._reject(c, {"image": {"format": "png",
                                                 "source": {"text": "not pixels"}}},
                                   "image.source")
                await self._reject(c, {"image": {"format": "png",
                                                 "source": {"content": []}}}, "image.source")
                await self._reject(c, {"video": {"format": "mp4",
                                                 "source": {"text": "x"}}}, "video.source")
                await self._reject(c, {"video": {"format": "avi",
                                                 "source": {"bytes": "aGk="}}}, "video.format")
                await self._reject(c, {"citationsContent": {"citations": "no"}},
                                   "citationsContent.citations")
                # a document source really does take `text`
                r = await self._pending(c, "/model/anthropic.claude-opus-5/converse", {
                    "messages": [{"role": "user", "content": [
                        {"document": {"name": "d", "format": "txt",
                                      "source": {"text": "hello"}}}]}]})
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                self.assertEqual((await r).status_code, 200)
        _run(run())


class TestS3UnmodelledOperations(_Base):
    """Operations that share a path with a modelled one must be refused, not guessed at."""

    def test_object_put_subresources_and_copy_are_refused(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/uns")
                await c.put("/uns/keep.txt", content=b"original")
                probes = [
                    ({}, {"x-amz-copy-source": "/uns/keep.txt"}, b""),
                    ({"tagging": ""}, {}, b"<Tagging/>"),
                    ({"acl": ""}, {}, b""),
                    ({"retention": ""}, {}, b"<Retention/>"),
                ]
                for params, headers, body in probes:
                    r = await c.put("/uns/keep.txt", params=params, headers=headers,
                                    content=body)
                    self.assertEqual(r.status_code, 501, (params, headers))
                    self.assertIn("NotImplemented", r.text, (params, headers))
                # the object is untouched
                self.assertEqual((await c.get("/uns/keep.txt")).content, b"original")
                # and so are the GET-side sub-resources
                for params in ({"tagging": ""}, {"acl": ""}, {"attributes": ""}):
                    r = await c.get("/uns/keep.txt", params=params)
                    self.assertEqual(r.status_code, 501, params)
        _run(run())

    def test_bucket_subresources_are_refused(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/uns2")
                await c.put("/uns2/k.txt", content=b"v")
                for params in ({"versions": ""}, {"policy": ""}, {"tagging": ""},
                               {"versioning": ""}, {"acl": ""}, {"location": ""}):
                    r = await c.get("/uns2", params=params)
                    self.assertEqual(r.status_code, 501, params)
                    self.assertIn("NotImplemented", r.text, params)
                # the listing itself still works
                self.assertIn("<Key>k.txt</Key>", (await c.get("/uns2")).text)
        _run(run())

    def test_conditional_headers(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/cond")
                r = await c.put("/cond/o.txt", content=b"v1")
                etag = r.headers["etag"]
                # writes
                r = await c.put("/cond/o.txt", content=b"v2", headers={"if-none-match": "*"})
                self.assertEqual(r.status_code, 412)
                self.assertIn("PreconditionFailed", r.text)
                self.assertEqual((await c.get("/cond/o.txt")).content, b"v1")
                r = await c.put("/cond/o.txt", content=b"v2",
                                headers={"if-match": chr(34) + "nope" + chr(34)})
                self.assertEqual(r.status_code, 412)
                r = await c.put("/cond/o.txt", content=b"v2", headers={"if-match": etag})
                self.assertEqual(r.status_code, 200)
                r = await c.put("/cond/new.txt", content=b"n", headers={"if-none-match": "*"})
                self.assertEqual(r.status_code, 200)
                # reads
                cur = (await c.head("/cond/o.txt")).headers["etag"]
                self.assertEqual((await c.get("/cond/o.txt",
                                              headers={"if-none-match": cur})).status_code, 304)
                self.assertEqual((await c.get("/cond/o.txt",
                                              headers={"if-match": cur})).status_code, 200)
                r = await c.get("/cond/o.txt",
                                headers={"if-match": chr(34) + "nope" + chr(34)})
                self.assertEqual(r.status_code, 412)
        _run(run())

    def test_encoding_type_url_round_trips(self) -> None:
        async def run() -> None:
            from urllib.parse import quote
            async with await self._client() as c:
                await c.put("/enc")
                keys = ["a/1.txt", "with space.txt", "plus+sign.txt", "amp&and.txt"]
                for k in keys:
                    await c.put("/enc/" + quote(k, safe=""), content=b"x")
                raw = (await c.get("/enc", params={"list-type": "2"})).text
                self.assertIn("<Key>a/1.txt</Key>", raw)
                self.assertNotIn("EncodingType", raw)
                enc = (await c.get("/enc", params={"list-type": "2",
                                                   "encoding-type": "url"})).text
                self.assertIn("<EncodingType>url</EncodingType>", enc)
                self.assertIn("<Key>a%2F1.txt</Key>", enc)
                self.assertIn("<Key>with%20space.txt</Key>", enc)
                # a continuation token we emitted encoded still selects the right position
                page = (await c.get("/enc", params={"list-type": "2", "encoding-type": "url",
                                                    "max-keys": "1"})).text
                import re as _re
                token = _re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>",
                                   page).group(1)
                nxt = (await c.get("/enc", params={"list-type": "2", "encoding-type": "url",
                                                   "continuation-token": token})).text
                self.assertNotIn(f"<Key>{token}</Key>", nxt)   # never repeats the same key
                r = await c.get("/enc", params={"list-type": "2", "encoding-type": "gzip"})
                self.assertEqual(r.status_code, 400)
        _run(run())

    def test_a_write_in_progress_is_not_an_object_but_a_tmp_named_key_is(self) -> None:
        """`_write_outputs` runs in a worker thread while the loop keeps serving S3, so the
        half-written file must live where no listing walks — and, conversely, an object
        whose KEY merely looks like a temp file must stay a perfectly visible object."""
        async def run() -> None:
            import os as _os
            async with await self._client() as c:
                await c.put("/tmpb")
                await c.put("/tmpb/real.txt", content=b"v")
                for key in (".x.tmp", "in/.data.tmp"):
                    self.assertEqual((await c.put(f"/tmpb/{key}", content=b"t")).status_code, 200)
                self.assertEqual([o["key"] for o in s3.list_objects("tmpb")],
                                 [".x.tmp", "in/.data.tmp", "real.txt"])
                # the store's own temp directory is a sibling of the buckets, never inside one
                self.assertTrue(_os.path.isfile(_os.path.join(self.tmp, "tmpb", ".x.tmp")))
                self.assertFalse(_os.path.exists(_os.path.join(self.tmp, "tmpb", ".tmp")))
                self.assertTrue(_os.path.isdir(_os.path.join(self.tmp, ".tmp")))
                self.assertEqual(_os.listdir(_os.path.join(self.tmp, ".tmp")), [])  # nothing left behind
                listing = (await c.get("/")).text
                self.assertIn("<Name>tmpb</Name>", listing)
                self.assertNotIn(".tmp", listing)
        _run(run())


class TestDocumentedBehaviours(_Base):
    """Behaviours the README promises, one test each."""

    JOB = TestBedrockBatchInference.JOB
    _seed = TestBedrockBatchInference._seed
    _create = TestBedrockBatchInference._create
    _await_end = TestBedrockBatchInference._await_end
    _invoke_record = staticmethod(TestBedrockBatchInference._invoke_record)
    _out_lines = staticmethod(TestBedrockBatchInference._out_lines)
    CONV = "/model/anthropic.claude-opus-5/converse"
    CONV_STREAM = "/model/anthropic.claude-opus-5/converse-stream"
    INVOKE_STREAM = "/model/anthropic.claude-opus-5/invoke-with-response-stream"

    # ── batch ──

    def test_an_output_write_failure_ends_failed_with_a_message(self) -> None:
        """`Completed` must always imply the objects exist."""
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                arn = (await self._create(c)).json()["jobArn"]
                real = bb._write_outputs
                bb._write_outputs = lambda job: (_ for _ in ()).throw(OSError("disk full"))
                self.addCleanup(setattr, bb, "_write_outputs", real)
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                j = await self._await_end(c, arn)
                self.assertEqual(j["status"], "Failed")
                self.assertIn("disk full", j["message"])
                self.assertEqual(s3.list_objects("outbkt"), [])
                # a stop on a failed job is the conflict
                r = await c.post(f"/model-invocation-job/{arn}/stop")
                self.assertEqual((r.status_code, r.headers["x-amzn-ErrorType"]),
                                 (400, "ConflictException"))
        _run(run())

    def test_stop_is_accepted_while_the_job_is_still_submitted(self) -> None:
        """The real service accepts a Stop on a `Submitted` job; whatever was not yet
        registered is then never processed."""
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record(f"r{i}") for i in range(400)])
                real = bb._read_input
                started = threading.Event()

                def slow_read(uri):
                    started.set()
                    import time as _t
                    _t.sleep(0.4)
                    return real(uri)

                bb._read_input = slow_read
                self.addCleanup(setattr, bb, "_read_input", real)
                task = asyncio.create_task(self._create(c))
                await asyncio.to_thread(started.wait, 5)
                jobs = (await c.get("/_control/bedrock_jobs")).json()["jobs"]
                self.assertEqual(jobs[0]["status"], "Submitted")
                r = await c.post(f"/model-invocation-job/{jobs[0]['jobArn']}/stop")
                self.assertEqual(r.status_code, 200)
                r = await task
                self.assertEqual(r.status_code, 200, r.text)
                j = await self._await_end(c, r.json()["jobArn"])
                self.assertEqual((j["status"], j["totalRecordCount"], j["processedRecordCount"]),
                                 ("Stopped", 400, 0))
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
        _run(run())

    def test_token_replay_never_returns_a_job_that_is_rolled_back(self) -> None:
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await c.put("/inbkt")
                await c.put("/outbkt")  # input object deliberately missing
                real = bb._read_input

                def slow_read(uri):
                    import time as _t
                    _t.sleep(0.3)
                    return real(uri)

                bb._read_input = slow_read
                self.addCleanup(setattr, bb, "_read_input", real)
                rs = await asyncio.gather(*(self._create(c, clientRequestToken="tok-x")
                                            for _ in range(3)))
                codes = sorted(r.status_code for r in rs)
                self.assertNotIn(200, codes, [r.text for r in rs])  # never a dangling ARN
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
                # and with a valid input, the same token yields exactly one job
                await self._seed(c, [self._invoke_record("r1")])
                rs = await asyncio.gather(*(self._create(c, clientRequestToken="tok-y")
                                            for _ in range(3)))
                self.assertEqual({r.status_code for r in rs}, {200})
                self.assertEqual(len({r.json()["jobArn"] for r in rs}), 1)
        _run(run())

    def test_malformed_jsonl_line_rolls_the_create_back(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/inbkt"); await c.put("/outbkt")
                body = json.dumps(self._invoke_record("r1")).encode() + b"\n{not json\n"
                await c.put("/inbkt/in/data.jsonl", content=body)
                r = await self._create(c)
                self.assertEqual(r.status_code, 400)
                self.assertIn("invalid JSON", r.json()["message"])
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 0)
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
        _run(run())

    def test_list_jobs_time_filters(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                arn = (await self._create(c)).json()["jobArn"]
                sub = (await c.get(f"/model-invocation-job/{arn}")).json()["submitTime"]
                for params, n in (({"submitTimeAfter": str(sub - 10)}, 1),
                                  ({"submitTimeAfter": str(sub + 10)}, 0),
                                  ({"submitTimeBefore": str(sub + 10)}, 1),
                                  ({"submitTimeBefore": "2020-01-01T00:00:00Z"}, 0)):
                    r = await c.get("/model-invocation-jobs", params=params)
                    self.assertEqual(len(r.json()["invocationJobSummaries"]), n, params)
                r = await c.get("/model-invocation-jobs", params={"submitTimeAfter": "junk"})
                self.assertEqual(r.status_code, 400)
        _run(run())

    # ── Converse ──

    def test_additional_model_request_fields_cannot_shadow_converse_fields(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for extra in ({"system": "INJECTED"}, {"tools": []}, {"max_tokens": 7},
                              {"messages": []}, {"temperature": 0.1}):
                    r = await self._must_reject(c, self.CONV, {
                        "messages": [{"role": "user", "content": [{"text": "q"}]}],
                        "additionalModelRequestFields": extra})
                    self.assertEqual(r.status_code, 400, extra)
                    self.assertIn(next(iter(extra)), r.json()["message"], extra)
                # what Converse cannot express still goes through, and outputConfig maps
                t = await self._pending(c, self.CONV, {
                    "messages": [{"role": "user", "content": [{"text": "q"}]}],
                    "additionalModelRequestFields": {"top_k": 5},
                    "outputConfig": {"effort": "low"}})
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]["params"]
                self.assertEqual((p["top_k"], p["output_config"]), (5, {"effort": "low"}))
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                await t
        _run(run())

    def test_tool_choice_auto_and_any(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for which in ("auto", "any"):
                    t = await self._pending(c, self.CONV, {
                        "messages": [{"role": "user", "content": [{"text": "q"}]}],
                        "toolConfig": {"tools": [{"toolSpec": {
                            "name": "f", "inputSchema": {"json": {"type": "object"}}}}],
                            "toolChoice": {which: {}}}})
                    p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                    self.assertEqual(p["params"]["tool_choice"], {"type": which})
                    await self._ctl(c, "/_control/auto", json={"text": "ok"})
                    await t
                r = await self._must_reject(c, self.CONV, {
                    "messages": [{"role": "user", "content": [{"text": "q"}]}],
                    "toolConfig": {"tools": [{"toolSpec": {
                        "name": "f", "inputSchema": {"json": {"type": "object"}}}}],
                        "toolChoice": {"auto": 1}}})
                self.assertEqual(r.status_code, 400)
        _run(run())

    def test_s3_location_source_and_redacted_reasoning_round_trip(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, self.CONV, {
                    "messages": [{"role": "user", "content": [
                        {"image": {"format": "png",
                                   "source": {"s3Location": {"uri": "s3://b/k.png"}}}}]}]})
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(p["messages"][0]["content"][0]["source"],
                                 {"type": "url", "url": "s3://b/k.png"})
                await self._ctl(c, "/_control/respond", json={"content": [
                    {"type": "redacted_thinking", "data": "QUJD"},
                    {"type": "text", "text": "ok"}]})
                body = (await t).json()
                self.assertEqual(body["output"]["message"]["content"][0],
                                 {"reasoningContent": {"redactedContent": "QUJD"}})
                # and on the stream
                t = await self._pending(c, self.CONV_STREAM, {
                    "messages": [{"role": "user", "content": [{"text": "q"}]}]})
                await self._ctl(c, "/_control/respond", json={"content": [
                    {"type": "redacted_thinking", "data": "QUJD"},
                    {"type": "text", "text": "ok"}]})
                evs = eventstream.decode_messages((await t).content)
                deltas = [e["delta"] for e in evs if e.get("_event") == "contentBlockDelta"]
                self.assertEqual(deltas[0], {"reasoningContent": {"redactedContent": "QUJD"}})
                # no contentBlockStart for a reasoning block: the union has no member for it
                self.assertEqual([e["_event"] for e in evs][:3],
                                 ["messageStart", "contentBlockDelta", "contentBlockStop"])
        _run(run())

    def test_converse_response_echoes_service_tier_and_performance_config(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, self.CONV, {
                    "messages": [{"role": "user", "content": [{"text": "q"}]}],
                    "serviceTier": {"type": "priority"}, "performanceConfig": {}})
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                body = (await t).json()
                self.assertEqual(body["serviceTier"], {"type": "priority"})
                self.assertEqual(body["performanceConfig"], {"latency": "standard"})
                # the InvokeModel route takes the tier / latency as HEADERS (that is how
                # boto3's `serviceTier=` / `performanceConfigLatency=` travel) and echoes them
                for path in ("/model/anthropic.claude-opus-5/invoke",
                             "/model/anthropic.claude-opus-5/invoke-with-response-stream"):
                    t = await self._pending(c, path, {
                        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                        "messages": [{"role": "user", "content": "q"}]})
                    await self._ctl(c, "/_control/auto", json={"text": "ok"})
                    r = await t
                    self.assertEqual(r.headers["x-amzn-bedrock-service-tier"], "default", path)
                    self.assertNotIn("x-amzn-bedrock-performanceconfig-latency", r.headers)
                    t = asyncio.create_task(c.post(path, json={
                        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                        "messages": [{"role": "user", "content": "q"}]},
                        headers={"X-Amzn-Bedrock-Service-Tier": "priority",
                                 "X-Amzn-Bedrock-PerformanceConfig-Latency": "optimized"}))
                    for _ in range(50):
                        if (await c.get("/_control/pending")).json()["has_pending"]:
                            break
                        await asyncio.sleep(0.02)
                    p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                    self.assertEqual(p["params"]["service_tier"], "auto")  # visible to the responder
                    await self._ctl(c, "/_control/auto", json={"text": "ok"})
                    r = await t
                    self.assertEqual(r.headers["x-amzn-bedrock-service-tier"], "priority", path)
                    self.assertEqual(r.headers["x-amzn-bedrock-performanceconfig-latency"],
                                     "optimized", path)
                r = await c.post("/model/anthropic.claude-opus-5/invoke", json={
                    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "q"}]},
                    headers={"X-Amzn-Bedrock-Service-Tier": "gold"})
                self.assertEqual(r.status_code, 400)
        _run(run())

    def test_request_metadata_constraints(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for meta in ({}, {"k" * 257: "v"}, {"k": "v" * 257}, {"bad key!": "v"}):
                    r = await self._must_reject(c, self.CONV, {
                        "messages": [{"role": "user", "content": [{"text": "q"}]}],
                        "requestMetadata": meta})
                    self.assertEqual(r.status_code, 400, meta)
        _run(run())

    # ── mid-stream ──

    def test_mid_stream_member_keeps_the_status_class(self) -> None:
        """A 429-class exception the union cannot carry must not degrade to a 500."""
        from puppetllm.providers.bedrock import stream_exception_member as f
        self.assertEqual(f("ModelNotReadyException", status=429), "throttlingException")
        self.assertEqual(f("ServiceQuotaExceededException", status=400), "validationException")
        self.assertEqual(f("overloaded_error", status=529), "internalServerException")
        self.assertEqual(f("ModelNotReadyException", operation="converse", status=429),
                         "throttlingException")
        async def run() -> None:
            async with await self._client() as c:
                for path in (self.INVOKE_STREAM, self.CONV_STREAM):
                    body = ({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                             "messages": [{"role": "user", "content": "x"}]}
                            if path == self.INVOKE_STREAM else
                            {"messages": [{"role": "user", "content": [{"text": "x"}]}]})
                    t = await self._pending(c, path, body)
                    await self._ctl(c, "/_control/error", json={
                        "status": 429, "type": "ModelNotReadyException", "message": "warming",
                        "after_events": 1})
                    r = await t
                    frames = eventstream.decode_frames(r.content)
                    self.assertEqual(frames[-1][0][":exception-type"], "throttlingException", path)
                    if path == self.INVOKE_STREAM:
                        self.assertEqual(r.headers["x-amzn-bedrock-content-type"],
                                         "application/json")
        _run(run())

    def test_model_stream_error_exception_carries_original_status(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, self.CONV_STREAM,
                                        {"messages": [{"role": "user", "content": [{"text": "x"}]}]})
                await self._ctl(c, "/_control/error", json={"status": 408, "message": "slow",
                                                      "after_events": 1})
                frames = eventstream.decode_frames((await t).content)
                self.assertEqual(frames[-1][0][":exception-type"], "modelStreamErrorException")
                payload = json.loads(frames[-1][1])
                self.assertEqual((payload["originalStatusCode"], payload["originalMessage"]),
                                 (408, "slow"))
        _run(run())

    def test_after_events_is_not_recorded_for_a_non_stream_request(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {
                    "model": "claude-opus-5", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "x"}]})
                await self._ctl(c, "/_control/error", json={"status": 500, "after_events": 2,
                                                      "content": [{"type": "text", "text": "p"}]})
                self.assertEqual((await t).status_code, 500)
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertNotIn("after_events", h["injected_error"])
                self.assertNotIn("partial_content", h["injected_error"])
        _run(run())

    # ── S3 ──

    def test_conditional_since_headers_on_reads(self) -> None:
        async def run() -> None:
            from email.utils import formatdate
            import time as _t
            async with await self._client() as c:
                await c.put("/since")
                await c.put("/since/o.txt", content=b"v")
                future = formatdate(_t.time() + 3600, usegmt=True)
                past = formatdate(_t.time() - 3600, usegmt=True)
                self.assertEqual((await c.get("/since/o.txt",
                                              headers={"if-modified-since": future})).status_code, 304)
                self.assertEqual((await c.get("/since/o.txt",
                                              headers={"if-modified-since": past})).status_code, 200)
                self.assertEqual((await c.get("/since/o.txt",
                                              headers={"if-unmodified-since": past})).status_code, 412)
                # the -Since forms are read-side only, as on the real service: a write goes through
                r = await c.put("/since/o.txt", content=b"v2", headers={"if-unmodified-since": past})
                self.assertEqual(r.status_code, 200)
                # If-Match on a missing key is NoSuchKey, not a precondition failure
                r = await c.put("/since/none.txt", content=b"x",
                                headers={"if-match": chr(34) + "abc" + chr(34)})
                self.assertEqual(r.status_code, 404)
        _run(run())

    def test_conditional_writes_have_exactly_one_winner(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/race")
                rs = await asyncio.gather(*(c.put("/race/k", content=f"w{i}".encode() * 5000,
                                                  headers={"if-none-match": "*"})
                                            for i in range(16)))
                self.assertEqual(sorted(r.status_code for r in rs), [200] + [412] * 15)
                winner = next(r for r in rs if r.status_code == 200)
                self.assertEqual((await c.head("/race/k")).headers["etag"], winner.headers["etag"])
                etag = winner.headers["etag"]
                rs = await asyncio.gather(*(c.put("/race/k", content=f"x{i}".encode() * 5000,
                                                  headers={"if-match": etag})
                                            for i in range(16)))
                self.assertEqual(sorted(r.status_code for r in rs), [200] + [412] * 15)
        _run(run())

    def test_v1_markers_are_compared_raw_under_encoding_type_url(self) -> None:
        """botocore decodes `NextMarker` before resending it as `marker`; unquoting it again
        moved a key containing `%XX` and silently skipped the next one."""
        async def run() -> None:
            from urllib.parse import quote
            import re as _re
            async with await self._client() as c:
                await c.put("/v1enc")
                for k in ("p%2Fq", "p/q", "p/r"):
                    await c.put("/v1enc/" + quote(k, safe=""), content=b"x")
                seen: list[str] = []
                marker = ""
                for _ in range(10):
                    params = {"encoding-type": "url", "max-keys": "1"}
                    if marker:
                        params["marker"] = marker
                    body = (await c.get("/v1enc", params=params)).text
                    seen += [unquote_key for unquote_key in
                             (__import__("urllib.parse").parse.unquote(k)
                              for k in _re.findall(r"<Key>([^<]*)</Key>", body))]
                    m = _re.search(r"<NextMarker>([^<]*)</NextMarker>", body)
                    if not m:
                        break
                    marker = __import__("urllib.parse").parse.unquote(m.group(1))  # as botocore does
                self.assertEqual(seen, ["p%2Fq", "p/q", "p/r"])
                # a literal StartAfter / Marker is not decoded either
                body = (await c.get("/v1enc", params={"list-type": "2", "encoding-type": "url",
                                                      "start-after": "p%2Fq"})).text
                self.assertEqual(_re.findall(r"<Key>([^<]*)</Key>", body), ["p%2Fq", "p%2Fr"])
        _run(run())

    def test_deleting_the_last_child_frees_the_directory_key(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/dirs")
                await c.put("/dirs/dir/child", content=b"c")
                self.assertEqual((await c.delete("/dirs/dir/child")).status_code, 204)
                self.assertEqual((await c.put("/dirs/dir", content=b"now a file")).status_code, 200)
                self.assertEqual((await c.get("/dirs/dir")).content, b"now a file")
        _run(run())

    def test_range_at_end_of_object_is_unsatisfiable(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/rng")
                await c.put("/rng/o", content=b"0123456789")
                for header in ("bytes=10-", "bytes=-0", "bytes=10-20"):
                    r = await c.get("/rng/o", headers={"range": header})
                    self.assertEqual(r.status_code, 416, header)
                # a syntactically invalid range (last < first, start inside the object) is
                # ignored per RFC 9110: the whole object, 200
                r = await c.get("/rng/o", headers={"range": "bytes=5-2"})
                self.assertEqual((r.status_code, r.content), (200, b"0123456789"))
        _run(run())

    def test_put_options_this_store_cannot_honour_are_refused(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/opts")
                for headers in ({"x-amz-tagging": "k=v"}, {"x-amz-meta-owner": "me"},
                                {"x-amz-server-side-encryption": "AES256"},
                                {"x-amz-acl": "public-read"}):
                    r = await c.put("/opts/o", content=b"x", headers=headers)
                    self.assertEqual(r.status_code, 501, headers)
                    self.assertIn("NotImplemented", r.text, headers)
                # plain entity headers are accepted (and not stored)
                r = await c.put("/opts/o", content=b"x", headers={"content-type": "text/plain"})
                self.assertEqual(r.status_code, 200)
                # Content-MD5 is verified
                import base64 as _b64, hashlib as _h
                good = _b64.b64encode(_h.md5(b"x").digest()).decode()
                self.assertEqual((await c.put("/opts/o", content=b"x",
                                              headers={"content-md5": good})).status_code, 200)
                r = await c.put("/opts/o", content=b"y", headers={"content-md5": good})
                self.assertEqual(r.status_code, 400)
                self.assertIn("BadDigest", r.text)
        _run(run())

    def test_bucket_listing_and_deletion(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/bk-a"); await c.put("/bk-b")
                await c.put("/bk-b/o", content=b"x")
                listing = (await c.get("/")).text
                self.assertIn("<Name>bk-a</Name>", listing)
                self.assertIn("<Name>bk-b</Name>", listing)
                r = await c.delete("/bk-b")
                self.assertEqual(r.status_code, 409)
                self.assertIn("BucketNotEmpty", r.text)
                await c.delete("/bk-b/o")
                self.assertEqual((await c.delete("/bk-b")).status_code, 204)
                self.assertEqual((await c.head("/bk-b")).status_code, 404)
                self.assertEqual((await c.delete("/bk-a")).status_code, 204)
                # presigned-URL query auth is ignored like header auth
                await c.put("/bk-c"); await c.put("/bk-c/o", content=b"x")
                r = await c.get("/bk-c/o", params={"X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                                                   "X-Amz-Signature": "deadbeef",
                                                   "X-Amz-Expires": "60"})
                self.assertEqual((r.status_code, r.content), (200, b"x"))
                r = await c.get("/bk-c", params={"list-type": "3"})
                self.assertEqual(r.status_code, 400)
        _run(run())


class TestLifecycleAndProtocolEdges(_Base):
    """Job-lifecycle races, S3 request semantics and Converse validation edges."""

    JOB = TestBedrockBatchInference.JOB
    _seed = TestBedrockBatchInference._seed
    _create = TestBedrockBatchInference._create
    _await_end = TestBedrockBatchInference._await_end
    _invoke_record = staticmethod(TestBedrockBatchInference._invoke_record)
    _out_lines = staticmethod(TestBedrockBatchInference._out_lines)
    CONV = "/model/anthropic.claude-opus-5/converse"
    INVOKE = "/model/anthropic.claude-opus-5/invoke"

    # ── batch ──

    def test_stop_on_a_completed_job_is_a_conflict(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                arn = (await self._create(c)).json()["jobArn"]
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                j = await self._await_end(c, arn)
                self.assertEqual(j["status"], "Completed")
                job_id = arn.rsplit("/", 1)[1]
                before = s3.get_object("outbkt", f"out/{job_id}/manifest.json.out")
                r = await c.post(f"/model-invocation-job/{arn}/stop")
                self.assertEqual((r.status_code, r.headers["x-amzn-ErrorType"]),
                                 (400, "ConflictException"))
                j = (await c.get(f"/model-invocation-job/{arn}")).json()
                self.assertEqual(j["status"], "Completed")  # never re-enters Stopping
                self.assertEqual(s3.get_object("outbkt", f"out/{job_id}/manifest.json.out"), before)
        _run(run())

    def test_a_fast_responder_cannot_complete_a_job_still_registering(self) -> None:
        """The `creating` guard: answering record 1 before record N is registered must not
        end the job with a partial manifest."""
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record(f"r{i}") for i in range(3)])
                real = bb._spawn
                gate = asyncio.Event()
                calls = {"n": 0}

                def slow_spawn(*args, **kwargs):
                    calls["n"] += 1
                    return real(*args, **kwargs)

                orig_register = self.mod.register_request

                async def gated_register(*args, **kwargs):
                    res = await orig_register(*args, **kwargs)
                    if calls["n"] == 1:      # after record 1 is registered, hold record 2
                        await gate.wait()
                    return res

                bb._spawn = slow_spawn
                self.mod.register_request = gated_register
                self.addCleanup(setattr, bb, "_spawn", real)
                self.addCleanup(setattr, self.mod, "register_request", orig_register)
                task = asyncio.create_task(self._create(c))
                for _ in range(100):
                    if (await c.get("/_control/pending")).json()["count"] >= 1:
                        break
                    await asyncio.sleep(0.01)
                pend = (await c.get("/_control/pending")).json()["pending"][0]
                await self._ctl(c, "/_control/respond", json={"pending_id": pend["pending_id"],
                                                        "content": [{"type": "text", "text": "x"}]})
                await asyncio.sleep(0.1)
                jobs = (await c.get("/_control/bedrock_jobs")).json()["jobs"]
                self.assertEqual(jobs[0]["status"], "InProgress")  # not Completed with 1/3
                gate.set()
                r = await task
                self.assertEqual(r.status_code, 200, r.text)
                for x in (await c.get("/_control/pending")).json()["pending"]:
                    await self._ctl(c, "/_control/respond", json={"pending_id": x["pending_id"],
                                                            "content": [{"type": "text", "text": "y"}]})
                j = await self._await_end(c, r.json()["jobArn"])
                self.assertEqual((j["status"], j["processedRecordCount"]), ("Completed", 3))
        _run(run())

    def test_an_injection_in_flight_survives_a_stop(self) -> None:
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1"), self._invoke_record("r2")])
                arn = (await self._create(c)).json()["jobArn"]
                job_id = arn.rsplit("/", 1)[1]
                pend = {x["request"]["record_id"]: x["pending_id"] for x in
                        (await c.get("/_control/pending")).json()["pending"]}
                # resolve r1's future WITHOUT giving its collector a chance to run, then Stop
                async with self.mod.state.lock:
                    fut = self.mod.state.pending[pend["r1"]]["future"]
                    fut.set_result({"content": [{"type": "text", "text": "in flight"}]})
                    self.mod.state.pending.pop(pend["r1"])
                r = await c.post(f"/model-invocation-job/{arn}/stop")
                self.assertEqual(r.status_code, 200)
                j = await self._await_end(c, arn)
                self.assertEqual((j["status"], j["processedRecordCount"]), ("Stopped", 1))
                lines = self._out_lines("outbkt", f"out/{job_id}/data.jsonl.out")
                self.assertEqual([x["recordId"] for x in lines], ["r1"])
                ctl = (await c.get("/_control/bedrock_jobs")).json()["jobs"][0]
                self.assertEqual(ctl["cancelled"], ["r2"])
        _run(run())

    def test_duplicate_or_non_string_record_ids_roll_the_create_back(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # a responder answering record 1 BEFORE the duplicate at record 2 is seen
                # produces history; the rollback must take that history with it
                orig_register = self.mod.register_request
                gate = asyncio.Event()
                seen = {"n": 0}

                async def gated_register(*a, **k):
                    seen["n"] += 1
                    if seen["n"] == 2:
                        await gate.wait()   # record 1 is registered AND spawned by now
                    return await orig_register(*a, **k)

                self.mod.register_request = gated_register
                self.addCleanup(setattr, self.mod, "register_request", orig_register)
                await self._seed(c, [self._invoke_record("same"), self._invoke_record("x2"),
                                     self._invoke_record("same")])
                task = asyncio.create_task(self._create(c))
                for _ in range(100):
                    if (await c.get("/_control/pending")).json()["count"] >= 1:
                        break
                    await asyncio.sleep(0.01)
                pend = (await c.get("/_control/pending")).json()["pending"][0]
                await self._ctl(c, "/_control/respond", json={"pending_id": pend["pending_id"],
                                                               "content": [{"type": "text", "text": "x"}]})
                await asyncio.sleep(0.05)
                self.assertEqual(len((await c.get("/_control/history")).json()["history"]), 1)
                gate.set()
                r = await task
                self.assertEqual(r.status_code, 400)
                self.assertIn("duplicate", r.json()["message"])
                self.assertEqual((await c.get("/_control/history")).json()["history"], [])
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
                self.mod.register_request = orig_register
                for records, needle in (
                        ([self._invoke_record("same"), self._invoke_record("same")], "duplicate"),
                        ([{"recordId": 7, "modelInput": self._invoke_record("x")["modelInput"]}],
                         "non-empty string"),
                        ([{"recordId": "", "modelInput": self._invoke_record("x")["modelInput"]}],
                         "non-empty string")):
                    await self._seed(c, records)
                    r = await self._create(c)
                    self.assertEqual(r.status_code, 400, r.text)
                    self.assertIn(needle, r.json()["message"])
                    self.assertEqual((await c.get("/_control/pending")).json()["count"], 0)
                    self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
                    self.assertEqual((await c.get("/_control/history")).json()["history"], [])
        _run(run())

    def test_stop_during_the_input_read_keeps_stopping_and_survives_validation(self) -> None:
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                real = bb._read_input
                started = threading.Event()
                release = threading.Event()

                def slow_read(uri):
                    started.set()
                    import time as _t
                    while not release.is_set():
                        _t.sleep(0.01)
                    return real(uri)

                bb._read_input = slow_read
                self.addCleanup(setattr, bb, "_read_input", real)
                # (a) status stays Stopping after the read, never regresses to InProgress —
                # observable because a 600-record registration yields every 200 records
                await self._seed(c, [self._invoke_record(f"r{i}") for i in range(600)])
                task = asyncio.create_task(self._create(c))
                await asyncio.to_thread(started.wait, 5)
                arn = (await c.get("/_control/bedrock_jobs")).json()["jobs"][0]["jobArn"]
                self.assertEqual((await c.post(f"/model-invocation-job/{arn}/stop")).status_code, 200)
                release.set()
                seen: list[str] = []
                import time as _t
                deadline = _t.monotonic() + 10
                while not task.done() and _t.monotonic() < deadline:
                    seen.append((await c.get(f"/model-invocation-job/{arn}")).json()["status"])
                    await asyncio.sleep(0)  # an uncontended GET never suspends: let the create run
                r = await asyncio.wait_for(task, 10)
                self.assertEqual(r.status_code, 200, r.text)
                self.assertNotIn("InProgress", seen, seen)
                self.assertIn("Stopping", seen, seen)   # the read finished with the Stop kept
                j = await self._await_end(c, arn)
                self.assertEqual((j["status"], j["totalRecordCount"]), ("Stopped", 600))
                # (b) a Stop accepted on a job whose validation then fails: the ARN the
                # stopper holds must not 404 — the job ends Failed with the reason
                await self._ctl(c, "/_control/clear")
                started.clear(); release.clear()
                await c.put("/inbkt"); await c.put("/outbkt")
                await c.put("/inbkt/in/data.jsonl", content=json.dumps(self._invoke_record("r1")).encode())
                task = asyncio.create_task(self._create(
                    c, outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://nosuchbkt/o/"}}))
                await asyncio.to_thread(started.wait, 5)
                arn = (await c.get("/_control/bedrock_jobs")).json()["jobs"][0]["jobArn"]
                self.assertEqual((await c.post(f"/model-invocation-job/{arn}/stop")).status_code, 200)
                release.set()
                r = await task
                self.assertEqual(r.status_code, 400)
                j = (await c.get(f"/model-invocation-job/{arn}")).json()
                self.assertEqual(j["status"], "Failed")
                self.assertIn("output bucket does not exist", j["message"])
        _run(run())

    def test_token_replay_waits_for_registration_too(self) -> None:
        """A twin that is past the input read but still registering can still be rolled
        back (duplicate recordId): its ARN must not be handed out until it settles."""
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("dup"), self._invoke_record("dup")])
                real = bb._spawn
                gate = asyncio.Event()
                first = {"done": False}

                def gated_spawn(*args, **kwargs):
                    return real(*args, **kwargs)

                orig_register = self.mod.register_request

                async def slow_register(*args, **kwargs):
                    res = await orig_register(*args, **kwargs)
                    if not first["done"]:
                        first["done"] = True
                        await gate.wait()   # hold after record 1; record 2 will be the duplicate
                    return res

                self.mod.register_request = slow_register
                self.addCleanup(setattr, self.mod, "register_request", orig_register)
                t1 = asyncio.create_task(self._create(c, clientRequestToken="tok-dup"))
                for _ in range(100):
                    if (await c.get("/_control/pending")).json()["count"] >= 1:
                        break
                    await asyncio.sleep(0.01)
                t2 = asyncio.create_task(self._create(c, clientRequestToken="tok-dup"))
                await asyncio.sleep(0.1)
                self.assertFalse(t2.done())  # the replayer is waiting, not answering
                gate.set()
                r1, r2 = await asyncio.gather(t1, t2)
                self.assertEqual({r1.status_code, r2.status_code}, {400}, (r1.text, r2.text))
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
        _run(run())

    # ── S3 ──

    def test_control_characters_anywhere_in_the_path_are_refused_before_routing(self) -> None:
        """Starlette's `{key:path}` is `.*` + `$`: a trailing newline would be dropped
        (`k\n` routing as `k`) and a newline elsewhere would miss every route."""
        async def call(path: str, body: bytes = b"x") -> tuple[int, bytes]:
            status, out = {}, bytearray()
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            async def send(msg):
                if msg["type"] == "http.response.start":
                    status["code"] = msg["status"]
                elif msg["type"] == "http.response.body":
                    out.extend(msg.get("body", b""))
            await self.mod.app({"type": "http", "method": "PUT", "path": path,
                                "raw_path": path.encode(), "root_path": "", "scheme": "http",
                                "query_string": b"", "headers": [(b"host", b"test")],
                                "client": ("t", 1), "server": ("t", 80)}, receive, send)
            return status["code"], bytes(out)
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/nlb")
                await c.put("/nlb/k", content=b"original")
            for path in ("/nlb/k\n", "/nlb/a\nb", "/nlb/k\r", "/nlb/x\x07y", "/nlb\n"):
                code, body = await call(path)
                self.assertEqual(code, 400, repr(path))
                self.assertIn(b"InvalidArgument", body, repr(path))
            self.assertEqual(s3.get_object("nlb", "k"), b"original")   # never rewritten
            self.assertEqual([o["key"] for o in s3.list_objects("nlb")], ["k"])
        _run(run())

    def test_a_symlinked_file_is_never_listed_or_hashed(self) -> None:
        async def run() -> None:
            import os as _os
            outside = tempfile.mkdtemp(prefix="puppetllm-outside-")
            self.addCleanup(shutil.rmtree, outside, True)
            with open(_os.path.join(outside, "secret"), "wb") as fh:
                fh.write(b"s" * 100)
            async with await self._client() as c:
                await c.put("/lnk")
                await c.put("/lnk/real", content=b"r")
                _os.symlink(_os.path.join(outside, "secret"), _os.path.join(self.tmp, "lnk", "leak"))
                self.assertEqual([o["key"] for o in s3.list_objects("lnk")], ["real"])
                self.assertNotIn("leak", (await c.get("/lnk")).text)
                self.assertEqual((await c.get("/lnk/leak")).status_code, 400)
                # a bucket directory that is a symlink out of the root is not a bucket
                _os.symlink(outside, _os.path.join(self.tmp, "sym-bkt"))
                self.assertEqual((await c.head("/sym-bkt")).status_code, 404)
                self.assertNotIn("sym-bkt", (await c.get("/")).text)
        _run(run())

    def test_etag_follows_a_same_size_overwrite_even_with_equal_mtime(self) -> None:
        async def run() -> None:
            import os as _os
            async with await self._client() as c:
                await c.put("/etg")
                r = await c.put("/etg/o", content=b"A" * 8)
                first = r.headers["etag"]
                path = _os.path.join(self.tmp, "etg", "o")
                st = _os.stat(path)
                r = await c.put("/etg/o", content=b"B" * 8)
                self.assertNotEqual(r.headers["etag"], first)
                # the write seeds the cache under the file's CURRENT identity: on a kernel with
                # coarse mtimes an inode reused for a same-size write would otherwise serve
                # the previous object's ETag
                st2 = _os.stat(path)
                self.assertEqual(s3._ETAG_CACHE.get((path, st2.st_ino, st2.st_size, st2.st_mtime_ns)),
                                 r.headers["etag"])
                # force the collision the cache key cannot distinguish: same inode reuse,
                # same size, same mtime_ns — the seed is what keeps the answer right
                _os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
                self.assertEqual((await c.head("/etg/o")).headers["etag"], r.headers["etag"])
                self.assertIn(r.headers["etag"].replace(chr(34), "&quot;"),
                              (await c.get("/etg")).text.replace(chr(34), "&quot;"))
        _run(run())

    def test_conditional_header_precedence_and_checksums(self) -> None:
        async def run() -> None:
            from email.utils import formatdate
            import base64 as _b64, hashlib as _h, time as _t, zlib as _z
            async with await self._client() as c:
                await c.put("/csum")
                etag = (await c.put("/csum/o", content=b"v")).headers["etag"]
                future = formatdate(_t.time() + 3600, usegmt=True)
                past = formatdate(_t.time() - 3600, usegmt=True)
                # RFC 9110: If-None-Match present → If-Modified-Since ignored (and vice versa)
                r = await c.get("/csum/o", headers={"if-none-match": chr(34) + "other" + chr(34),
                                                  "if-modified-since": future})
                self.assertEqual(r.status_code, 200)
                r = await c.get("/csum/o", headers={"if-match": etag, "if-unmodified-since": past})
                self.assertEqual(r.status_code, 200)
                # request checksums: botocore sends a CRC32 on every put_object
                data = b"payload"
                crc = _b64.b64encode(_z.crc32(data).to_bytes(4, "big")).decode()
                self.assertEqual((await c.put("/csum/c", content=data,
                                              headers={"x-amz-checksum-crc32": crc})).status_code, 200)
                r = await c.put("/csum/c", content=b"corrupted", headers={"x-amz-checksum-crc32": crc})
                self.assertEqual(r.status_code, 400)
                self.assertIn("BadDigest", r.text)
                sha = _b64.b64encode(_h.sha256(data).digest()).decode()
                self.assertEqual((await c.put("/csum/c", content=data,
                                              headers={"x-amz-checksum-sha256": sha})).status_code, 200)
                # a non-STANDARD storage class would be a lie
                r = await c.put("/csum/g", content=b"x", headers={"x-amz-storage-class": "GLACIER"})
                self.assertEqual(r.status_code, 501)
                self.assertEqual((await c.put("/csum/g", content=b"x",
                                              headers={"x-amz-storage-class": "STANDARD"})).status_code, 200)
                # the internal listing is in key order, like the wire
                for k in ("a/b/c", "a b/c", "a/b-d", "a/bc"):
                    s3.put_object("csum", k, b"1")
                keys = [o["key"] for o in s3.list_objects("csum", "a")]
                self.assertEqual(keys, sorted(keys))
        _run(run())

    # ── Converse ──

    def test_output_config_shapes(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                schema = {"type": "object", "properties": {"a": {"type": "string"}}}
                # JsonSchemaDefinition.schema is a JSON *string* on the wire (botocore model)
                t = await self._pending(c, self.CONV, {
                    "messages": [{"role": "user", "content": [{"text": "q"}]}],
                    "outputConfig": {"effort": "xhigh",
                                     "textFormat": {"type": "json_schema",
                                                    "structure": {"jsonSchema": {
                                                        "schema": json.dumps(schema), "name": "result"}}}}})
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]["params"]
                self.assertEqual(p["output_config"],
                                 {"effort": "xhigh", "format": {"type": "json_schema", "schema": schema,
                                                                "name": "result"}})
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                await t
                for body in (
                        {"outputConfig": {"effort": "ultra"}},
                        {"outputConfig": {"textFormat": {"type": "xml", "structure": {"jsonSchema": {"schema": "{}"}}}}},
                        {"outputConfig": {"textFormat": {"type": "json_schema"}}},
                        # the schema object itself instead of its JSON string
                        {"outputConfig": {"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {"schema": {}}}}}},
                        {"outputConfig": {"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {"schema": "[1]"}}}}},
                        {"outputConfig": {"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {"schema": "{", "name": "x"}}}}},
                        {"modelId": "some-other-model"},
                        {"messages": [{"role": "user", "content": [{"cachePoint": {"type": "default"}, "bogus": 1}]}]},
                        {"messages": [{"role": "user", "content": [{"guardContent": {"image": {"format": "gif", "source": {"bytes": "aGk="}}}}]}]},
                        {"messages": [{"role": "user", "content": [{"guardContent": {"image": {"format": "png", "source": {"s3Location": {"uri": "s3://b/k"}}}}}]}]},
                        {"outputConfig": {"bogus": 1}},
                        {"additionalModelRequestFields": {"output_config": "junk"},
                         "outputConfig": {"effort": "high"}},
                        {"unknownTopLevel": 1},
                        {"messages": [{"role": "user", "content": [{"text": "q", "foo": 1}]}]},
                        {"toolConfig": {"tools": [{"toolSpec": {"name": "f", "description": "",
                                                                "inputSchema": {"json": {"type": "object"}}}}]}},
                        {"toolConfig": {"tools": [{"toolSpec": {"name": "f",
                                                                "inputSchema": {"json": {"type": "array"}}}}]}},
                        {"toolConfig": {"tools": [{"systemTool": {}}]}}):
                    req = {"messages": [{"role": "user", "content": [{"text": "q"}]}], **body}
                    r = await self._must_reject(c, self.CONV, req)
                    self.assertEqual(r.status_code, 400, body)
        _run(run())

    def test_service_tier_vocabulary_matches_the_relay(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for tier, canonical in (("priority", "auto"), ("flex", "auto"),
                                        ("default", "standard_only"), ("reserved", "standard_only")):
                    t = await self._pending(c, self.CONV, {
                        "messages": [{"role": "user", "content": [{"text": "q"}]}],
                        "serviceTier": {"type": tier}})
                    p = (await c.get("/_control/pending")).json()["pending"][0]["request"]["params"]
                    self.assertEqual(p["service_tier"], canonical, tier)
                    await self._ctl(c, "/_control/auto", json={"text": "ok"})
                    await t
        _run(run())

    # ── responder input and error shapes ──

    def test_redacted_thinking_data_must_be_base64(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, self.CONV,
                                        {"messages": [{"role": "user", "content": [{"text": "q"}]}]})
                r = await c.post("/_control/respond", json={"content": [
                    {"type": "redacted_thinking", "data": "not base64!"}, {"type": "text", "text": "x"}]})
                self.assertEqual(r.status_code, 400)
                self.assertIn("base64", r.text)
                r = await c.post("/_control/error", json={"status": 500, "after_events": 1, "content": [
                    {"type": "redacted_thinking", "data": "nope nope"}]})
                self.assertEqual(r.status_code, 400)
                self.assertTrue((await c.get("/_control/pending")).json()["has_pending"])  # untouched
                await self._ctl(c, "/_control/respond", json={"content": [
                    {"type": "redacted_thinking", "data": "QUJD"}, {"type": "text", "text": "x"}]})
                self.assertEqual((await t).status_code, 200)
        _run(run())

    def test_untyped_injected_error_gets_the_anthropic_vocabulary_for_its_status(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for status, etype in ((429, "rate_limit_error"), (529, "overloaded_error"),
                                      (400, "invalid_request_error"), (418, "api_error")):
                    t = await self._pending(c, "/v1/messages", {
                        "model": "claude-opus-5", "max_tokens": 8,
                        "messages": [{"role": "user", "content": "x"}]})
                    await self._ctl(c, "/_control/error", json={"status": status, "message": "m"})
                    r = await t
                    self.assertEqual((r.status_code, r.json()["error"]["type"]), (status, etype))
                # and on the Bedrock non-stream route a ModelErrorException carries the
                # same original-status fields as the stream frame
                t = await self._pending(c, self.INVOKE, {
                    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "x"}]})
                await self._ctl(c, "/_control/error", json={"status": 424, "message": "upstream"})
                r = await t
                # ModelErrorException is modeled as {message, originalStatusCode, resourceName}
                # (originalMessage belongs to ModelStreamErrorException); `original_status`
                # sets the upstream status a 424 wraps, otherwise the injected one is used
                self.assertEqual(r.json(), {"message": "upstream", "__type": "ModelErrorException",
                                            "originalStatusCode": 424,
                                            "resourceName": "anthropic.claude-opus-5"})
                t = await self._pending(c, self.INVOKE, {
                    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "x"}]})
                await self._ctl(c, "/_control/error", json={"status": 424, "message": "up",
                                                             "original_status": 429})
                self.assertEqual((await t).json()["originalStatusCode"], 429)
                r = await c.post("/_control/error", json={"status": 424, "original_status": 42})
                self.assertEqual(r.status_code, 400)
        _run(run())


class TestConditionalOperationsAndRaces(_Base):
    """Conditional S3 operations, the store lock, and create/clear/stop races."""

    JOB = TestBedrockBatchInference.JOB
    _seed = TestBedrockBatchInference._seed
    _create = TestBedrockBatchInference._create
    _await_end = TestBedrockBatchInference._await_end
    _invoke_record = staticmethod(TestBedrockBatchInference._invoke_record)
    _out_lines = staticmethod(TestBedrockBatchInference._out_lines)
    CONV = "/model/anthropic.claude-opus-5/converse"
    CONV_STREAM = "/model/anthropic.claude-opus-5/converse-stream"

    # ── S3 ──

    def test_conditional_delete(self) -> None:
        """`delete_object(IfMatch=...)` is the guard against deleting a newer version."""
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/cdel")
                etag = (await c.put("/cdel/k", content=b"v1")).headers["etag"]
                r = await c.delete("/cdel/k", headers={"if-match": chr(34) + "stale" + chr(34)})
                self.assertEqual(r.status_code, 412)
                self.assertIn("PreconditionFailed", r.text)
                self.assertEqual((await c.get("/cdel/k")).content, b"v1")  # still there
                for h in ({"x-amz-if-match-size": "2"}, {"x-amz-if-match-last-modified-time": "x"}):
                    r = await c.delete("/cdel/k", headers=h)
                    self.assertEqual(r.status_code, 501, h)
                self.assertEqual((await c.delete("/cdel/k", headers={"if-match": etag})).status_code, 204)
                self.assertEqual((await c.get("/cdel/k")).status_code, 404)
                r = await c.delete("/cdel/k", headers={"if-match": "*"})
                self.assertEqual(r.status_code, 404)          # gone: Not Found, not a silent 204
                await c.put("/cdel/k2", content=b"x")
                self.assertEqual((await c.delete("/cdel/k2", headers={"if-match": "*"})).status_code, 204)
        _run(run())

    def test_aws_chunked_must_be_terminated_and_sized(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/chk")
                with self.assertRaises(s3.S3Error):
                    s3.decode_aws_chunked(b"3\r\nabc\r\n")      # no zero chunk
                r = await c.put("/chk/o", content=b"3\r\nabc\r\n",
                                headers={"content-encoding": "aws-chunked"})
                self.assertEqual(r.status_code, 400)
                r = await c.put("/chk/o", content=b"3\r\nabc\r\n0\r\n\r\n",
                                headers={"content-encoding": "aws-chunked",
                                         "x-amz-decoded-content-length": "99"})
                self.assertEqual(r.status_code, 400)
                self.assertIn("IncompleteBody", r.text)
                r = await c.put("/chk/o", content=b"3\r\nabc\r\n0\r\n\r\n",
                                headers={"content-encoding": "aws-chunked",
                                         "x-amz-decoded-content-length": "3"})
                self.assertEqual(r.status_code, 200)
        _run(run())

    def test_expected_bucket_owner_is_enforced(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/own")
                await c.put("/own/k", content=b"v")
                r = await c.get("/own/k", params={"expected-bucket-owner": "999999999999"})
                self.assertEqual(r.status_code, 403)
                self.assertIn("AccessDenied", r.text)
                r = await c.get("/own", headers={"x-amz-expected-bucket-owner": "999999999999"})
                self.assertEqual(r.status_code, 403)
                self.assertEqual((await c.get("/own/k", params={"expected-bucket-owner": "123456789012"})).status_code, 200)
        _run(run())

    def test_empty_first_segment_is_settled_before_routing(self) -> None:
        async def call(path: str, body: bytes = b"x" * 10) -> tuple[int, bytes]:
            status, out = {}, bytearray()
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            async def send(msg):
                if msg["type"] == "http.response.start":
                    status["code"] = msg["status"]
                elif msg["type"] == "http.response.body":
                    out.extend(msg.get("body", b""))
            await self.mod.app({"type": "http", "method": "PUT", "path": path,
                                "raw_path": path.encode(), "root_path": "", "scheme": "http",
                                "query_string": b"", "headers": [(b"host", b"test")],
                                "client": ("t", 1), "server": ("t", 80)}, receive, send)
            return status["code"], bytes(out)
        async def run() -> None:
            code, body = await call("//x")
            self.assertEqual(code, 400)
            self.assertIn(b"InvalidArgument", body)
        _run(run())

    def test_store_lock_serializes_a_worker_thread_against_the_handlers(self) -> None:
        """The batch emulation writes from a worker thread. Without the store lock an HTTP
        GET could stat one version, hash it, and read the next one: the ETag and
        Content-Length of A with the body of B. Hammer the same key from a thread with two
        sizes while the handler reads it, and demand every read be self-consistent."""
        async def run() -> None:
            import hashlib as _h, threading, time as _t
            async with await self._client() as c:
                await c.put("/lockb")
                s3.put_object("lockb", "o", b"A" * 100)
                stop = threading.Event()

                def writer():
                    i = 0
                    while not stop.is_set():
                        s3.put_object("lockb", "o", (b"A" * 100) if i % 2 else (b"B" * 3000))
                        i += 1

                th = threading.Thread(target=writer, daemon=True); th.start()
                try:
                    deadline = _t.monotonic() + 1.0
                    reads = 0
                    while _t.monotonic() < deadline:
                        r = await c.get("/lockb/o")
                        self.assertEqual(r.status_code, 200)
                        self.assertEqual(int(r.headers["content-length"]), len(r.content))
                        self.assertEqual(r.headers["etag"].strip(chr(34)), _h.md5(r.content).hexdigest())
                        self.assertIn(r.content, (b"A" * 100, b"B" * 3000))
                        reads += 1
                finally:
                    stop.set(); th.join(5)
                self.assertGreater(reads, 20)
        _run(run())

    # ── batch ──

    def test_clear_while_a_twin_waits_does_not_resurrect_the_job(self) -> None:
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record(f"r{i}") for i in range(3)])
                real = bb._read_input
                started = threading.Event(); release = threading.Event()

                def slow(uri):
                    started.set()
                    import time as _t
                    while not release.is_set():
                        _t.sleep(0.01)
                    return real(uri)

                bb._read_input = slow
                self.addCleanup(setattr, bb, "_read_input", real)
                ta = asyncio.create_task(self._create(c, clientRequestToken="tok-clear"))
                await asyncio.to_thread(started.wait, 5)
                tb = asyncio.create_task(self._create(c, clientRequestToken="tok-clear"))
                await asyncio.sleep(0.1)
                await self._ctl(c, "/_control/clear")
                release.set()
                ra, rb = await asyncio.gather(ta, tb)
                self.assertEqual((ra.status_code, rb.status_code), (400, 400), (ra.text, rb.text))
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 0)
        _run(run())

    def test_stop_then_duplicate_record_id_ends_failed_not_404(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                orig_register = self.mod.register_request
                gate = asyncio.Event(); seen = {"n": 0}

                async def gated(*a, **k):
                    res = await orig_register(*a, **k)
                    seen["n"] += 1
                    if seen["n"] == 1:
                        await gate.wait()
                    return res

                self.mod.register_request = gated
                self.addCleanup(setattr, self.mod, "register_request", orig_register)
                await self._seed(c, [self._invoke_record("dup"), self._invoke_record("dup")])
                task = asyncio.create_task(self._create(c))
                for _ in range(100):
                    if (await c.get("/_control/pending")).json()["count"] >= 1:
                        break
                    await asyncio.sleep(0.01)
                arn = (await c.get("/_control/bedrock_jobs")).json()["jobs"][0]["jobArn"]
                self.assertEqual((await c.post(f"/model-invocation-job/{arn}/stop")).status_code, 200)
                gate.set()
                r = await task
                self.assertEqual(r.status_code, 400)
                j = (await c.get(f"/model-invocation-job/{arn}")).json()
                self.assertEqual(j["status"], "Failed")
                self.assertIn("duplicate", j["message"])
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 0)
        _run(run())

    def test_output_failure_inside_create_is_still_a_created_failed_job(self) -> None:
        """All records errored at registration → finalize runs inside Create; a write
        failure there must yield the ARN of a `Failed` job, not a bare 500."""
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                bad = {"recordId": "r1", "modelInput": {"max_tokens": 8,   # no anthropic_version
                                                        "messages": [{"role": "user", "content": "q"}]}}
                await self._seed(c, [bad])
                real = bb._write_outputs
                bb._write_outputs = lambda job: (_ for _ in ()).throw(OSError("disk full"))
                self.addCleanup(setattr, bb, "_write_outputs", real)
                r = await self._create(c)
                self.assertEqual(r.status_code, 200, r.text)
                j = (await c.get(f"/model-invocation-job/{r.json()['jobArn']}")).json()
                self.assertEqual(j["status"], "Failed")
                self.assertIn("disk full", j["message"])
                # and a Stop that awaits a failing finalize answers 400 Conflict, never 500
                await self._seed(c, [self._invoke_record("r1")])
                arn = (await self._create(c, jobName="second")).json()["jobArn"]
                pend = (await c.get("/_control/pending")).json()["pending"][0]
                r = await c.post(f"/model-invocation-job/{arn}/stop")
                self.assertEqual(r.status_code, 200)
                j = await self._await_end(c, arn)
                self.assertEqual(j["status"], "Failed")
        _run(run())

    def test_kms_and_owner_settings_are_refused_or_enforced(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                for cfg in ({"s3Uri": "s3://inbkt/in/data.jsonl", "s3EncryptionKeyId": "alias/x"},
                            {"s3Uri": "s3://inbkt/in/data.jsonl", "s3BucketOwner": "999999999999"}):
                    r = await self._create(c, inputDataConfig={"s3InputDataConfig": cfg})
                    self.assertEqual(r.status_code, 400, cfg)
                    self.assertEqual(r.headers["x-amzn-ErrorType"], "ValidationException")
                r = await self._create(c, inputDataConfig={"s3InputDataConfig": {
                    "s3Uri": "s3://inbkt/in/data.jsonl", "s3BucketOwner": "123456789012"}})
                self.assertEqual(r.status_code, 200, r.text)
        _run(run())

    def test_output_prefix_under_an_existing_object_is_refused_at_create(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                await c.put("/outbkt/taken", content=b"i am an object")
                r = await self._create(c, outputDataConfig={"s3OutputDataConfig": {
                    "s3Uri": "s3://outbkt/taken/sub/"}})
                self.assertEqual(r.status_code, 400)
                self.assertIn("existing object", r.json()["message"])
        _run(run())

    # ── Converse ──

    def test_prompt_arn_request_without_messages_is_accepted(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/model/arn:aws:bedrock:us-east-1:123456789012:prompt%2FPROMPT123/converse",
                                        {"promptVariables": {"topic": {"text": "cats"}}})
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(p["messages"], [])
                self.assertEqual(p["converse"]["promptVariables"], {"topic": {"text": "cats"}})
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                self.assertEqual((await t).status_code, 200)
                # without promptVariables, messages stays required
                r = await self._must_reject(c, self.CONV, {})
                self.assertEqual(r.status_code, 400)
        _run(run())

    def test_converse_stream_pre_stream_http_error(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, self.CONV_STREAM,
                                        {"messages": [{"role": "user", "content": [{"text": "x"}]}]})
                await self._ctl(c, "/_control/error", json={"status": 429, "type": "ThrottlingException",
                                                             "message": "slow", "headers": {"retry-after": "2"}})
                r = await t
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.headers["x-amzn-ErrorType"], "ThrottlingException")
                self.assertEqual(r.headers["retry-after"], "2")
                self.assertEqual(r.json()["__type"], "ThrottlingException")
        _run(run())


class TestConcurrencyAndValidationEdges(_Base):
    """Symlink loops, weak validators, owner checks, chunk termination, prompt ARNs."""

    JOB = TestBedrockBatchInference.JOB
    _seed = TestBedrockBatchInference._seed
    _create = TestBedrockBatchInference._create
    _await_end = TestBedrockBatchInference._await_end
    _invoke_record = staticmethod(TestBedrockBatchInference._invoke_record)
    CONV = "/model/anthropic.claude-opus-5/converse"
    CONV_STREAM = "/model/anthropic.claude-opus-5/converse-stream"

    def test_symlink_loop_is_a_400_not_a_500(self) -> None:
        async def run() -> None:
            import os as _os
            async with await self._client() as c:
                await c.put("/loopb")
                a = _os.path.join(self.tmp, "loopb", "a"); b = _os.path.join(self.tmp, "loopb", "b")
                _os.symlink(b, a); _os.symlink(a, b)
                for method in ("get", "head", "delete"):
                    r = await getattr(c, method)("/loopb/a/x")
                    self.assertEqual(r.status_code, 400, method)
                r = await c.put("/loopb/a/x", content=b"v")
                self.assertEqual(r.status_code, 400)
                self.assertEqual((await c.get("/loopb")).status_code, 200)  # listing survives
        _run(run())

    def test_clear_during_an_inline_finalize_does_not_hand_out_the_arn(self) -> None:
        """Every record errors at registration → finalize runs inside Create; a clear that
        lands while its outputs are being written must not let Create answer 200."""
        async def run() -> None:
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                bad = {"recordId": "r1", "modelInput": {"max_tokens": 8,
                                                        "messages": [{"role": "user", "content": "q"}]}}
                await self._seed(c, [bad])
                real = bb._write_outputs
                started = threading.Event(); release = threading.Event()

                def slow_write(job):
                    started.set()
                    release.wait(5)
                    real(job)

                bb._write_outputs = slow_write
                self.addCleanup(setattr, bb, "_write_outputs", real)
                task = asyncio.create_task(self._create(c))
                await asyncio.to_thread(started.wait, 5)
                await self._ctl(c, "/_control/clear")
                release.set()
                r = await task
                self.assertEqual((r.status_code, r.headers["x-amzn-ErrorType"]),
                                 (400, "ConflictException"), r.text)
                self.assertEqual((await c.get("/_control/bedrock_jobs")).json()["count"], 0)
        _run(run())

    def test_stream_frame_original_status_override(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for path, body in ((self.CONV_STREAM, {"messages": [{"role": "user", "content": [{"text": "x"}]}]}),
                                   ("/model/anthropic.claude-opus-5/invoke-with-response-stream",
                                    {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                                     "messages": [{"role": "user", "content": "x"}]})):
                    t = await self._pending(c, path, body)
                    await self._ctl(c, "/_control/error", json={
                        "status": 424, "type": "ModelStreamErrorException", "message": "up",
                        "after_events": 1, "original_status": 503})
                    frames = eventstream.decode_frames((await t).content)
                    self.assertEqual(frames[-1][0][":exception-type"], "modelStreamErrorException", path)
                    self.assertEqual(json.loads(frames[-1][1])["originalStatusCode"], 503, path)
        _run(run())

    def test_owner_check_considers_both_sources(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/own2"); await c.put("/own2/k", content=b"v")
                r = await c.get("/own2/k", params={"expected-bucket-owner": "123456789012"},
                                headers={"x-amz-expected-bucket-owner": "999999999999"})
                self.assertEqual(r.status_code, 403)
                r = await c.get("/own2/k", params={"expected-bucket-owner": "999999999999"},
                                headers={"x-amz-expected-bucket-owner": "123456789012"})
                self.assertEqual(r.status_code, 403)
        _run(run())

    def test_weak_etag_never_authorizes_a_write_or_delete(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/weak")
                etag = (await c.put("/weak/k", content=b"v")).headers["etag"]
                weak = "W/" + etag
                self.assertEqual((await c.delete("/weak/k", headers={"if-match": weak})).status_code, 412)
                self.assertEqual((await c.put("/weak/k", content=b"v2", headers={"if-match": weak})).status_code, 412)
                self.assertEqual((await c.get("/weak/k")).content, b"v")
                # reads may use the weak comparison
                self.assertEqual((await c.get("/weak/k", headers={"if-none-match": weak})).status_code, 304)
                self.assertEqual((await c.delete("/weak/k", headers={"if-match": etag})).status_code, 204)
        _run(run())

    def test_zero_chunk_needs_its_final_crlf(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await c.put("/chk2")
                r = await c.put("/chk2/o", content=b"3\r\nabc\r\n0\r\n",
                                headers={"content-encoding": "aws-chunked",
                                         "x-amz-decoded-content-length": "3"})
                self.assertEqual(r.status_code, 400)
                self.assertIn("InvalidRequest", r.text)
                with self.assertRaises(s3.S3Error):
                    s3.decode_aws_chunked(b"0\r\n")
                self.assertEqual(s3.decode_aws_chunked(b"0\r\n\r\n"), (b"", {}))
        _run(run())

    def test_prompt_variables_do_not_excuse_messages_on_an_ordinary_model(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for body in ({"promptVariables": {"topic": {"text": "x"}}},
                             {"promptVariables": {"topic": {"text": "x"}}, "messages": None}):
                    r = await self._must_reject(c, self.CONV, body)
                    self.assertEqual(r.status_code, 400, body)
                    self.assertIn("messages", r.json()["message"])
                # the prompt-ARN form still works, version suffix included
                t = await self._pending(c, "/model/arn:aws:bedrock:us-east-1:123456789012:prompt%2FPROMPT123%3A2/converse",
                                        {"promptVariables": {"topic": {"text": "cats"}}})
                await self._ctl(c, "/_control/auto", json={"text": "ok"})
                self.assertEqual((await t).status_code, 200)
        _run(run())

    def test_output_write_failure_is_logged(self) -> None:
        async def run() -> None:
            import contextlib, io
            from puppetllm.providers import bedrock_batch as bb
            async with await self._client() as c:
                await self._seed(c, [self._invoke_record("r1")])
                arn = (await self._create(c)).json()["jobArn"]
                real = bb._write_outputs
                bb._write_outputs = lambda job: (_ for _ in ()).throw(TypeError("bug in the writer"))
                self.addCleanup(setattr, bb, "_write_outputs", real)
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    await self._ctl(c, "/_control/auto", json={"text": "ok"})
                    j = await self._await_end(c, arn)
                self.assertEqual(j["status"], "Failed")
                self.assertIn("bug in the writer", buf.getvalue())
                self.assertIn("TypeError", buf.getvalue())
        _run(run())

    def test_s3_reads_do_not_freeze_the_event_loop_while_a_thread_holds_the_lock(self) -> None:
        async def run() -> None:
            import time as _t
            async with await self._client() as c:
                await c.put("/frz"); await c.put("/frz/o", content=b"v")
                held = threading.Event(); release = threading.Event()

                def hog():
                    with s3._STORE_LOCK:
                        held.set()
                        release.wait(5)

                th = threading.Thread(target=hog, daemon=True); th.start()
                await asyncio.to_thread(held.wait, 5)
                get = asyncio.create_task(c.get("/frz/o"))     # must park in the executor
                t0 = _t.monotonic()
                await asyncio.sleep(0.05)
                self.assertLess(_t.monotonic() - t0, 0.5)        # the loop itself stayed live
                self.assertEqual((await c.get("/_control/health")).status_code, 200)
                self.assertFalse(get.done())
                release.set()
                self.assertEqual((await get).content, b"v")
                th.join(5)
        _run(run())


if __name__ == "__main__":
    unittest.main()
