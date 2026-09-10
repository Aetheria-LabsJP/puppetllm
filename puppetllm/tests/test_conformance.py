"""Conformance tests against the official Anthropic Messages / Message Batches, AWS Bedrock
InvokeModel and OpenAI Chat Completions specifications.

Run:
  python3 -m unittest puppetllm.tests.test_conformance -v
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import unittest
from typing import Any

from puppetllm import pricing
from puppetllm.cache_sim import CacheSimulator, analyze_request, _min_cacheable_for
from puppetllm.providers import eventstream

# HTTP tests use small prefixes, so disable the minimum cache threshold (see test_proxy_extensions).
os.environ["PUPPETLLM_CACHE_MIN_TOKENS"] = "0"


def _import_fresh():
    import importlib
    from puppetllm import fake_server as fs
    importlib.reload(fs)
    return fs


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    name = None
    for line in text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: "):]
        elif line.startswith("data: ") and name is not None:
            out.append((name, json.loads(line[len("data: "):])))
            name = None
    return out


def _sys(text: str, **cc: Any) -> list[dict[str, Any]]:
    b: dict[str, Any] = {"type": "text", "text": text}
    if cc:
        b["cache_control"] = cc.get("cache_control", {"type": "ephemeral"})
    return [b]


# ── cache simulator (pure) ────────────────────────────────────────────


class TestCacheSimConformance(unittest.TestCase):
    def test_min_cacheable_is_generation_aware(self) -> None:
        cases = {
            "claude-opus-5": 512, "claude-fable-5-1": 512, "claude-mythos-5": 512,
            "claude-opus-4-8": 1024, "claude-opus-4-7": 2048, "claude-opus-4-6": 4096,
            "claude-opus-4-5-20251101": 4096, "claude-opus-4-1-20250805": 1024,
            "claude-sonnet-5": 1024, "claude-sonnet-4-6": 1024,
            "claude-haiku-4-5": 4096, "claude-3-5-haiku-20241022": 2048,
            "anthropic.claude-opus-5": 512, "mystery": 1024,
        }
        for model, n in cases.items():
            self.assertEqual(_min_cacheable_for(model), n, model)

    def test_top_level_cache_control_synthesizes_breakpoint(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=0)
        msgs = [{"role": "user", "content": "hello " * 50}]
        # no markers, no top-level → none
        rc = analyze_request("S" * 200, None, msgs)
        self.assertEqual(sim.observe(rc, "m", 0.0)["status"], "none")
        # top-level cache_control → breakpoint on the LAST cacheable block (the user text)
        rc = analyze_request("S" * 200, None, msgs, top_level_cache_control={"type": "ephemeral"})
        self.assertTrue(rc.top_level)
        self.assertEqual(rc.breakpoints, [len(rc.segs) - 1])
        self.assertEqual(sim.observe(rc, "m", 1.0)["status"], "miss")
        self.assertEqual(sim.observe(rc, "m", 2.0)["status"], "hit")
        # a trailing thinking block is skipped (not cacheable) — the breakpoint lands before it
        msgs2 = msgs + [{"role": "assistant", "content": [{"type": "thinking", "thinking": "t"}]}]
        rc2 = analyze_request("S" * 200, None, msgs2, top_level_cache_control={"type": "ephemeral"})
        self.assertEqual(rc2.breakpoints, [len(rc2.segs) - 2])

    def test_ttl_1h_lives_longer_and_is_reported_separately(self) -> None:
        sim = CacheSimulator(ttl_seconds=300, ttl_1h_seconds=3600, min_cacheable_tokens=0)
        system = [{"type": "text", "text": "A" * 400, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        msgs = [{"role": "user", "content": [{"type": "text", "text": "B" * 400,
                                              "cache_control": {"type": "ephemeral"}}]}]
        r = sim.observe(analyze_request(system, None, msgs), "m", 0.0)
        self.assertEqual(r["status"], "miss")
        self.assertEqual(r["cache_creation_1h_tokens"] + r["cache_creation_5m_tokens"],
                         r["cache_creation_tokens"])
        self.assertGreater(r["cache_creation_1h_tokens"], 0)
        self.assertGreater(r["cache_creation_5m_tokens"], 0)
        # after 301s the 5m entry is dead but the 1h prefix (system) still reads
        r2 = sim.observe(analyze_request(system, None, msgs), "m", 301.0)
        self.assertEqual(r2["status"], "hit")
        self.assertEqual(r2["read_seg_count"], 1)
        self.assertEqual(r2["cache_creation_1h_tokens"], 0)
        # index entries carry their TTL
        ttls = {e["seg_count"]: e["ttl"] for e in sim.entries(301.0)}
        self.assertEqual(ttls, {1: "1h", 2: "5m"})

    def test_effort_change_invalidates_messages_but_not_system(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=0)
        system = _sys("S" * 400, cache_control={"type": "ephemeral"})
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Q" * 400,
                                              "cache_control": {"type": "ephemeral"}}]}]
        sim.observe(analyze_request(system, None, msgs, params={}), "m", 0.0)
        # same request, explicit default effort == omitted → full hit
        r = sim.observe(analyze_request(system, None, msgs,
                                        params={"output_config": {"effort": "high"}}), "m", 1.0)
        self.assertEqual(r["read_seg_count"], 2)
        # effort changed → only the system prefix (seg 1) is read, messages re-written
        r = sim.observe(analyze_request(system, None, msgs,
                                        params={"output_config": {"effort": "low"}}), "m", 2.0)
        self.assertEqual(r["status"], "hit")
        self.assertEqual(r["read_seg_count"], 1)
        self.assertGreater(r["cache_creation_tokens"], 0)
        # tool_choice / thinking changes behave the same way
        r = sim.observe(analyze_request(system, None, msgs,
                                        params={"tool_choice": {"type": "any"}}), "m", 3.0)
        self.assertEqual(r["read_seg_count"], 1)

    def test_lookback_collapses_tool_runs(self) -> None:
        marked = [{"role": "user", "content": [{"type": "text", "text": "start",
                                                "cache_control": {"type": "ephemeral"}}]}]
        plain = [{"role": "user", "content": "start"}]  # same content, no marker this turn
        # 30 consecutive tool_result blocks count as ONE lookback position, so the earlier
        # entry (right before the run) is still within the 20-position window.
        sim = CacheSimulator(min_cacheable_tokens=0)
        sim.observe(analyze_request(None, None, marked), "m", 0.0)
        results = [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "r"} for i in range(30)]
        results[-1]["cache_control"] = {"type": "ephemeral"}
        r = sim.observe(analyze_request(None, None, plain + [{"role": "user", "content": results}]), "m", 1.0)
        self.assertEqual(r["status"], "hit")
        self.assertEqual(r["read_seg_count"], 1)
        # 30 consecutive TEXT blocks do not collapse → the entry is out of the window → miss
        texts = [{"type": "text", "text": f"x{i}"} for i in range(30)]
        texts[-1]["cache_control"] = {"type": "ephemeral"}
        sim2 = CacheSimulator(min_cacheable_tokens=0)
        sim2.observe(analyze_request(None, None, marked), "m", 0.0)
        r = sim2.observe(analyze_request(None, None, plain + [{"role": "user", "content": texts}]), "m", 1.0)
        self.assertEqual(r["status"], "miss")


# ── Anthropic route (HTTP) ─────────────────────────────────────────────


class TestAnthropicRouteConformance(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

    async def _pending(self, c, body: dict, headers: dict | None = None) -> Any:
        t = asyncio.create_task(c.post("/v1/messages", json=body, headers=headers, timeout=10))
        for _ in range(50):
            if (await c.get("/_control/pending")).json().get("has_pending"):
                return t
            await asyncio.sleep(0.05)
        self.fail("never became pending")

    def test_stream_usage_and_stop_fields(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, {
                    "model": "claude-opus-5", "stream": True, "max_tokens": 10,
                    "system": [{"type": "text", "text": "S" * 400,
                                "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                    "messages": [{"role": "user", "content": "hi"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "name": "f", "input": {"a": 1, "b": "x" * 100}}]})
                events = _sse_events((await t).text)
                by = {}
                for name, data in events:
                    by.setdefault(name, []).append(data)
                start = by["message_start"][0]["message"]
                self.assertEqual(start["usage"]["output_tokens"], 1)
                self.assertIn("stop_details", start)
                self.assertIsNone(start["stop_details"])
                delta = by["message_delta"][0]
                # cumulative usage: input + cache fields repeated on message_delta
                for k in ("input_tokens", "cache_creation_input_tokens",
                          "cache_read_input_tokens", "output_tokens", "cache_creation"):
                    self.assertIn(k, delta["usage"], k)
                self.assertEqual(delta["usage"]["input_tokens"], start["usage"]["input_tokens"])
                self.assertGreater(delta["usage"]["cache_creation"]["ephemeral_1h_input_tokens"], 0)
                self.assertEqual(delta["delta"],
                                 {"stop_reason": "tool_use", "stop_sequence": None,
                                  "stop_details": None})
                # tool input is streamed as an empty opener + several partial_json chunks
                parts = [d["delta"]["partial_json"] for d in by["content_block_delta"]
                         if d["delta"]["type"] == "input_json_delta"]
                self.assertEqual(parts[0], "")
                self.assertGreater(len(parts), 2)
                self.assertEqual(json.loads("".join(parts)), {"a": 1, "b": "x" * 100})
        _run(run())

    def test_non_stream_usage_shape_and_refusal_stop_details(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]})
                await c.post("/_control/respond", json={
                    "content": [], "stop_reason": "refusal"})
                j = (await t).json()
                self.assertEqual(j["stop_reason"], "refusal")
                self.assertEqual(j["stop_details"],
                                 {"type": "refusal", "category": None, "explanation": None})
                u = j["usage"]
                self.assertEqual(u["cache_creation"],
                                 {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0})
                self.assertEqual(u["output_tokens_details"], {"thinking_tokens": 0})
                self.assertEqual(u["service_tier"], "standard")
                self.assertIsNone(u["server_tool_use"])
                # explicit category passes through; non-refusal stop → stop_details null
                t = await self._pending(c, {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]})
                await c.post("/_control/respond", json={
                    "content": [], "stop_reason": "refusal",
                    "stop_details": {"category": "cyber", "explanation": "no"}})
                self.assertEqual((await t).json()["stop_details"],
                                 {"type": "refusal", "category": "cyber", "explanation": "no"})
                t = await self._pending(c, {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]})
                await c.post("/_control/respond", json={
                    "content": [], "stop_reason": "end_turn", "stop_details": {"category": "x"}})
                self.assertIsNone((await t).json()["stop_details"])
        _run(run())

    def test_stop_sequence_defaults_to_first_configured(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, {
                    "model": "claude-opus-5", "max_tokens": 10, "stop_sequences": ["END", "X"],
                    "messages": [{"role": "user", "content": "hi"}]})
                await c.post("/_control/respond", json={
                    "content": [{"type": "text", "text": "a"}], "stop_reason": "stop_sequence"})
                j = (await t).json()
                self.assertEqual((j["stop_reason"], j["stop_sequence"]), ("stop_sequence", "END"))
                t = await self._pending(c, {
                    "model": "claude-opus-5", "max_tokens": 10, "stop_sequences": ["END"],
                    "messages": [{"role": "user", "content": "hi"}]})
                await c.post("/_control/respond", json={
                    "content": [{"type": "text", "text": "a"}], "stop_reason": "stop_sequence",
                    "stop_sequence": "OTHER"})
                self.assertEqual((await t).json()["stop_sequence"], "OTHER")
        _run(run())

    def test_error_body_request_id_and_injected_headers(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]})
                # header map must be string → string/number
                r = await c.post("/_control/error", json={"status": 429, "headers": ["x"]})
                self.assertEqual(r.status_code, 400)
                r = await c.post("/_control/error", json={
                    "status": 429, "type": "rate_limit_error", "message": "slow",
                    "headers": {"retry-after": 7,
                                "anthropic-ratelimit-requests-remaining": "0"}})
                self.assertEqual(r.status_code, 200)
                resp = await t
                self.assertEqual(resp.status_code, 429)
                self.assertEqual(resp.headers["retry-after"], "7")
                self.assertEqual(resp.headers["anthropic-ratelimit-requests-remaining"], "0")
                body = resp.json()
                self.assertEqual(body["request_id"], resp.headers["request-id"])
                self.assertEqual(body["error"]["type"], "rate_limit_error")
                # malformed JSON → 400 with request_id too
                r = await c.post("/v1/messages", content=b"{oops", headers={"content-type": "application/json"})
                self.assertEqual(r.status_code, 400)
                self.assertTrue(r.json()["request_id"].startswith("req_"))
        _run(run())

    def test_snapshot_captures_2026_params_and_beta_header(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "output_config": {"effort": "xhigh"}, "cache_control": {"type": "ephemeral"},
                    "inference_geo": "us", "fallbacks": "default", "speed": "fast",
                    "messages": [{"role": "user", "content": "hi"}]},
                    headers={"anthropic-beta": "a-2026-01-01, b-2026-02-02"})
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]["params"]
                self.assertEqual(p["output_config"], {"effort": "xhigh"})
                self.assertEqual(p["cache_control"], {"type": "ephemeral"})
                self.assertEqual(p["inference_geo"], "us")
                self.assertEqual(p["fallbacks"], "default")
                self.assertEqual(p["speed"], "fast")
                self.assertEqual(p["anthropic_beta"], ["a-2026-01-01", "b-2026-02-02"])
                await c.post("/_control/auto", json={"text": "ok"})
                j = (await t).json()
                self.assertEqual(j["usage"]["inference_geo"], "us")
                # top-level cache_control → the request was cache-observed (miss, not none)
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(h["cache"]["status"], "miss")
        _run(run())

    def test_usage_override_accepts_objects_and_prices_1h_writes(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]})
                await c.post("/_control/respond", json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1000, "output_tokens": 100,
                              "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 0,
                              "cache_creation": {"ephemeral_5m_input_tokens": 500,
                                                 "ephemeral_1h_input_tokens": 1500},
                              "output_tokens_details": {"thinking_tokens": 40},
                              "server_tool_use": {"web_search_requests": 1},
                              "service_tier": "priority", "inference_geo": "global"}})
                u = (await t).json()["usage"]
                self.assertEqual(u["cache_creation"]["ephemeral_1h_input_tokens"], 1500)
                self.assertEqual(u["service_tier"], "priority")
                self.assertEqual(u["server_tool_use"], {"web_search_requests": 1})
                h = (await c.get("/_control/history")).json()["history"][-1]
                # opus 5: input $5, output $25, 5m write $6.25, 1h write $10 per Mtok
                expected = (1000 * 5.0 + 100 * 25.0 + 500 * 6.25 + 1500 * 10.0) / 1_000_000
                self.assertAlmostEqual(h["cost"]["total_usd"], expected, places=9)
                # object keys alone (no integer counters) are rejected
                t = await self._pending(c, {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]})
                r = await c.post("/_control/respond", json={
                    "content": [], "usage": {"service_tier": "batch"}})
                self.assertEqual(r.status_code, 400)
                await c.post("/_control/auto", json={"text": "ok"})
                await t
        _run(run())


# ── Bedrock route (HTTP) ───────────────────────────────────────────────


class TestBedrockConformance(unittest.TestCase):
    MODEL = "us.anthropic.claude-opus-5"

    def setUp(self) -> None:
        self.mod = _import_fresh()

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

    @staticmethod
    def _body(**over: Any) -> dict:
        b = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
             "messages": [{"role": "user", "content": "hi"}]}
        b.update(over)
        return b

    def test_messages_alias_for_mantle(self) -> None:
        buf = io.StringIO()
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/anthropic/v1/messages", {
                    "model": "anthropic.claude-opus-5", "max_tokens": 10, "stream": True,
                    "messages": [{"role": "user", "content": "hi"}]})
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(p["provider"], "bedrock")
                self.assertEqual(p["model"], "claude-opus-5")
                self.assertEqual(p["bedrock_model_id"], "anthropic.claude-opus-5")
                await c.post("/_control/auto", json={"text": "via mantle"})
                r = await t
                self.assertEqual(r.status_code, 200)
                self.assertIn("text/event-stream", r.headers["content-type"])
                self.assertIn("x-amzn-requestid", r.headers)
                events = _sse_events(r.text)
                self.assertEqual(events[0][1]["message"]["model"], "claude-opus-5")
                self.assertIn("via mantle", r.text)
                # errors use the Anthropic envelope (what AnthropicBedrockMantle parses)
                t = await self._pending(c, "/anthropic/v1/messages", {
                    "model": "anthropic.claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]})
                await c.post("/_control/error", json={"status": 429, "type": "rate_limit_error"})
                r = await t
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.json()["error"]["type"], "rate_limit_error")
                # a Mantle client pointed at the fake's ROOT posts to plain /v1/messages — the
                # Bedrock-form model id is still detected there (first-party ids are untouched)
                t = await self._pending(c, "/v1/messages", {
                    "model": "global.anthropic.claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]})
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual((p["provider"], p["model"], p["bedrock_model_id"]),
                                 ("bedrock", "claude-opus-5", "global.anthropic.claude-opus-5"))
                await c.post("/_control/auto", json={"text": "ok"})
                self.assertEqual((await t).json()["model"], "claude-opus-5")
                t = await self._pending(c, "/v1/messages", {
                    "model": "claude-opus-5", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}]})
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(p["provider"], "anthropic")
                self.assertNotIn("bedrock_model_id", p)
                await c.post("/_control/auto", json={"text": "ok"})
                await t
                # and by_model merges with the InvokeModel route
                t = await self._pending(c, f"/model/{self.MODEL}/invoke", self._body())
                await c.post("/_control/auto", json={"text": "ok"})
                await t
                st = (await c.get("/_control/stats")).json()
                self.assertEqual(list(st["by_model"]), ["claude-opus-5"])
        with contextlib.redirect_stderr(buf):
            _run(run())
        self.assertIn("[bedrock] messages model=anthropic.claude-opus-5 -> claude-opus-5", buf.getvalue())

    def test_cache_token_headers_and_metrics(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                body = self._body(system=[{"type": "text", "text": "S" * 400,
                                           "cache_control": {"type": "ephemeral"}}])
                t = await self._pending(c, f"/model/{self.MODEL}/invoke", body)
                await c.post("/_control/auto", json={"text": "ok"})
                r = await t
                u = r.json()["usage"]
                self.assertGreater(u["cache_creation_input_tokens"], 0)
                # input header = non-cached input only; cache write/read in their own headers
                self.assertEqual(int(r.headers["X-Amzn-Bedrock-Input-Token-Count"]), u["input_tokens"])
                self.assertEqual(int(r.headers["X-Amzn-Bedrock-Cache-Write-Input-Token-Count"]),
                                 u["cache_creation_input_tokens"])
                self.assertEqual(int(r.headers["X-Amzn-Bedrock-Cache-Read-Input-Token-Count"]), 0)
                self.assertEqual(r.headers["X-Amzn-Bedrock-Service-Tier"], "default")
                # second call reads the cache; the stream's invocationMetrics carry the same split
                t = await self._pending(c, f"/model/{self.MODEL}/invoke-with-response-stream", body)
                await c.post("/_control/auto", json={"text": "ok"})
                r = await t
                self.assertEqual(r.headers["X-Amzn-Bedrock-Content-Type"], "application/json")
                events = eventstream.decode_messages(r.content)
                m = [e for e in events if e.get("type") == "message_stop"][0]["amazon-bedrock-invocationMetrics"]
                self.assertGreater(m["cacheReadInputTokenCount"], 0)
                self.assertEqual(m["cacheWriteInputTokenCount"], 0)
                self.assertLessEqual(m["firstByteLatency"], m["invocationLatency"])
        _run(run())

    def test_arn_model_ids_route_and_normalize(self) -> None:
        from puppetllm.providers.bedrock import normalize_model_id
        arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-5"
        m = normalize_model_id(arn)
        self.assertEqual((m.canonical, m.region), ("claude-opus-5", "us"))
        m = normalize_model_id("arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0")
        self.assertEqual(m.canonical, "claude-haiku-4-5-20251001")
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, f"/model/{arn}/invoke", self._body())
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(p["model"], "claude-opus-5")
                self.assertEqual(p["bedrock_model_id"], arn)
                await c.post("/_control/auto", json={"text": "ok"})
                self.assertEqual((await t).status_code, 200)
        _run(run())

    def test_exception_table_and_error_headers(self) -> None:
        from puppetllm.providers.bedrock import exception_name_for
        self.assertEqual(exception_name_for(401, None), "UnrecognizedClientException")
        self.assertEqual(exception_name_for(413, "request_too_large"), "RequestEntityTooLargeException")
        self.assertEqual(exception_name_for(529, "overloaded_error"), "overloaded_error")
        self.assertEqual(exception_name_for(400, "ServiceQuotaExceededException"), "ServiceQuotaExceededException")
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, f"/model/{self.MODEL}/invoke", self._body())
                await c.post("/_control/error", json={
                    "status": 429, "type": "ThrottlingException", "message": "slow",
                    "headers": {"retry-after": "2"}})
                r = await t
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.headers["retry-after"], "2")
                self.assertEqual(r.headers["x-amzn-ErrorType"], "ThrottlingException")
        _run(run())


# ── OpenAI route (HTTP + pure) ─────────────────────────────────────────


class TestOpenAIConformance(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

    async def _pending(self, c, body: dict) -> Any:
        t = asyncio.create_task(c.post("/v1/chat/completions", json=body, timeout=10))
        for _ in range(50):
            if (await c.get("/_control/pending")).json().get("has_pending"):
                return t
            await asyncio.sleep(0.05)
        self.fail("never became pending")

    def test_image_and_refusal_parts_normalize(self) -> None:
        from puppetllm.providers import openai as oai
        out = oai.normalize_chat_body({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                {"type": "input_audio", "input_audio": {"data": "..", "format": "wav"}}]},
            {"role": "assistant", "content": None, "refusal": "I can't help with that."},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "nope"}]},
        ]})
        blocks = out["messages"][0]["content"]
        self.assertEqual(blocks[0], {"type": "text", "text": "what is this"})
        self.assertEqual(blocks[1], {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "AAAA"}, "_openai_detail": "low"})
        self.assertEqual(blocks[2], {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}})
        self.assertEqual(blocks[3]["type"], "input_audio")  # kept verbatim
        self.assertEqual(out["messages"][1]["content"], [{"type": "text", "text": "I can't help with that."}])
        self.assertEqual(out["messages"][2]["content"], [{"type": "text", "text": "nope"}])

    def test_custom_tools_and_strict_round_trip(self) -> None:
        from puppetllm.providers import openai as oai
        out = oai.normalize_chat_body({
            "tools": [
                {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}, "strict": True}},
                {"type": "custom", "custom": {"name": "sql", "description": "run sql",
                                              "format": {"type": "text"}}}],
            "messages": [{"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "custom", "custom": {"name": "sql", "input": "select 1"}}]}]})
        self.assertTrue(out["tools"][0]["strict"])
        self.assertTrue(out["tools"][1]["_openai_custom"])
        self.assertEqual(out["tools"][1]["format"], {"type": "text"})
        call = out["messages"][0]["content"][0]
        self.assertEqual((call["name"], call["input"], call.get("_openai_custom")),
                         ("sql", {"input": "select 1"}, True))
        # encoders emit a `custom` tool call again (non-stream + stream)
        resp = oai.build_non_stream_response("c", "m", [call], {"input_tokens": 1, "output_tokens": 1}, 0)
        tc = resp["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual((tc["type"], tc["custom"]), ("custom", {"name": "sql", "input": "select 1"}))
        chunks = oai.stream_chunk_dicts("c", "m", [call], {"input_tokens": 1, "output_tokens": 1}, 0, False)
        tcs = [ch["choices"][0]["delta"]["tool_calls"][0] for ch in chunks
               if ch["choices"] and "tool_calls" in ch["choices"][0]["delta"]]
        self.assertEqual(tcs[0]["type"], "custom")
        self.assertEqual(tcs[1]["custom"]["input"], "select 1")

    def test_finish_reason_fallbacks(self) -> None:
        from puppetllm.providers.openai import finish_reason_for
        self.assertEqual(finish_reason_for("pause_turn"), "stop")
        self.assertEqual(finish_reason_for("model_context_window_exceeded"), "length")
        self.assertEqual(finish_reason_for("content_filter"), "content_filter")
        self.assertEqual(finish_reason_for("something_weird"), "stop")

    def test_response_fields_headers_and_n2_stream(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, {"model": "gpt-5.4", "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={
                    "content": [{"type": "thinking", "thinking": "T" * 80},
                                {"type": "text", "text": "ok"}]})
                r = await t
                self.assertEqual(r.headers["openai-version"], "2020-10-01")
                self.assertIn("openai-processing-ms", r.headers)
                j = r.json()
                self.assertEqual(j["service_tier"], "default")
                self.assertIsNone(j["system_fingerprint"])
                msg = j["choices"][0]["message"]
                self.assertIsNone(msg["refusal"])
                self.assertEqual(msg["annotations"], [])
                self.assertEqual(msg["content"], "ok")  # thinking has no chat representation
                u = j["usage"]
                self.assertGreater(u["completion_tokens_details"]["reasoning_tokens"], 0)
                # only fields the current SDK defines
                self.assertEqual(sorted(u["prompt_tokens_details"]),
                                 ["audio_tokens", "cache_write_tokens", "cached_tokens"])
                self.assertEqual(sorted(u["completion_tokens_details"]),
                                 ["accepted_prediction_tokens", "audio_tokens",
                                  "reasoning_tokens", "rejected_prediction_tokens"])
                one_completion = u["completion_tokens"]
                # n=2 streaming: both choice indices get content and a finish chunk
                t = await self._pending(c, {"model": "gpt-5.4", "n": 2, "stream": True,
                                            "stream_options": {"include_usage": True},
                                            "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "hi"})
                lines = [json.loads(ln[6:]) for ln in (await t).text.splitlines()
                         if ln.startswith("data: ") and not ln.endswith("[DONE]")]
                finishes = {ch["choices"][0]["index"]: ch["choices"][0]["finish_reason"]
                            for ch in lines if ch["choices"] and ch["choices"][0]["finish_reason"]}
                self.assertEqual(finishes, {0: "stop", 1: "stop"})
                self.assertEqual(lines[0]["choices"][0]["delta"], {"role": "assistant", "content": "", "refusal": None})
                # n=2 bills both choices (the usage chunk is the last one)
                self.assertEqual(lines[-1]["choices"], [])
                self.assertEqual(lines[-1]["usage"]["completion_tokens"] % 2, 0)
                self.assertGreater(lines[-1]["usage"]["completion_tokens"], 0)
                # service_tier echoes an explicit request tier
                t = await self._pending(c, {"model": "gpt-5.4", "service_tier": "flex",
                                            "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "hi"})
                self.assertEqual((await t).json()["service_tier"], "flex")
        _run(run())

    def test_error_type_vocabulary_and_headers(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                for inject, expected in (
                        ({"status": 500}, "server_error"),
                        ({"status": 429, "type": "rate_limit_error"}, "rate_limit_error"),
                        ({"status": 529, "type": "overloaded_error"}, "server_error"),
                        ({"status": 400, "type": "invalid_request_error"}, "invalid_request_error"),
                        ({"status": 402, "type": "insufficient_quota"}, "insufficient_quota"),
                        ({"status": 401, "type": "api_error"}, "authentication_error")):
                    t = await self._pending(c, {"model": "gpt-5.4",
                                                "messages": [{"role": "user", "content": "y"}]})
                    await c.post("/_control/error", json={"message": "m", "headers": {"retry-after": "1"}, **inject})
                    r = await t
                    self.assertEqual(r.status_code, inject["status"])
                    self.assertEqual(r.json()["error"]["type"], expected, inject)
                    self.assertEqual(r.headers["retry-after"], "1")
        _run(run())


# ── wire hygiene, Bedrock passthrough, cache validation, error injection ──


class TestWireAndValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

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

    # private keys never reach an Anthropic wire (relay request, non-stream response)
    def test_private_keys_stripped_from_anthropic_wire(self) -> None:
        from puppetllm.relay import to_anthropic_request
        body = to_anthropic_request({
            "messages": [
                {"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "u"},
                                              "_openai_detail": "low"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "sql",
                                                   "input": {"input": "x"}, "_openai_custom": True}]}],
            "tools": [{"name": "sql", "_openai_custom": True, "format": {"type": "text"},
                       "input_schema": {"type": "object"}}]}, "claude-opus-5")
        self.assertEqual(sorted(body["tools"][0]), ["input_schema", "name"])
        self.assertEqual(sorted(body["messages"][0]["content"][0]), ["source", "type"])
        self.assertEqual(sorted(body["messages"][1]["content"][0]), ["id", "input", "name", "type"])
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {
                    "model": "claude-opus-5", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "tool_use", "name": "sql", "input": {"input": "1"}, "_openai_custom": True}]})
                self.assertEqual(sorted((await t).json()["content"][0]), ["id", "input", "name", "type"])
        _run(run())

    # non-Anthropic Bedrock model ids pass through without the anthropic_version check
    def test_bedrock_foreign_model_passthrough(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/model/meta.llama3-70b-instruct-v1:0/invoke",
                                        {"prompt": "hi", "max_gen_len": 10})
                p = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(p["model"], "meta.llama3-70b-instruct-v1:0")
                await c.post("/_control/auto", json={"text": "ok"})
                self.assertEqual((await t).status_code, 200)
                # Anthropic ids are still checked
                r = await c.post("/model/anthropic.claude-opus-5/invoke", json={"max_tokens": 1, "messages": []})
                self.assertEqual(r.status_code, 400)
        _run(run())

    # a breakpoint early in a collapsed tool run cannot read content that follows it
    def test_lookback_never_reads_past_the_breakpoint(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=0)
        res = [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "r"} for i in range(5)]
        def turn(bp_index):
            return [{"role": "user", "content": [
                dict(b, **({"cache_control": {"type": "ephemeral"}} if i == bp_index else {}))
                for i, b in enumerate(res)]}]
        sim.observe(analyze_request(None, None, turn(4)), "m", 0.0)
        r = sim.observe(analyze_request(None, None, turn(0)), "m", 1.0)
        # the only stored entry ends at seg 5, past the breakpoint at seg 1 → nothing readable
        self.assertEqual((r["read_seg_count"], r["status"]), (0, "miss"))
        # ...and the new seg-1 entry is readable from a later breakpoint in the same run
        r = sim.observe(analyze_request(None, None, turn(4)), "m", 2.0)
        self.assertEqual((r["read_seg_count"], r["status"]), (5, "hit"))

    # the 1h TTL scales with PUPPETLLM_CACHE_TTL unless overridden
    def test_ttl_1h_scales_with_env(self) -> None:
        import importlib
        from puppetllm import fake_server as fs
        os.environ["PUPPETLLM_CACHE_TTL"] = "10"
        try:
            importlib.reload(fs)
            self.assertEqual(fs.state.cache.ttl_seconds, 10.0)
            self.assertEqual(fs.state.cache.ttl_1h_seconds, 120.0)
            os.environ["PUPPETLLM_CACHE_TTL_1H"] = "50"
            importlib.reload(fs)
            self.assertEqual(fs.state.cache.ttl_1h_seconds, 50.0)
        finally:
            os.environ.pop("PUPPETLLM_CACHE_TTL", None)
            os.environ.pop("PUPPETLLM_CACHE_TTL_1H", None)
            importlib.reload(fs)

    # cache_control layouts the real API rejects → 400 on every route
    def test_cache_control_validation_400(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                def msg(system, **top):
                    b = {"model": "claude-opus-5", "max_tokens": 5, "system": system,
                         "messages": [{"role": "user", "content": "x"}]}
                    b.update(top)
                    return b
                cases = [
                    # 1h after 5m
                    msg([{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                         {"type": "text", "text": "b", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]),
                    # unknown ttl / wrong type / non-object
                    msg([{"type": "text", "text": "a", "cache_control": {"type": "ephemeral", "ttl": "2h"}}]),
                    msg([{"type": "text", "text": "a", "cache_control": {"type": "persistent"}}]),
                    msg([{"type": "text", "text": "a", "cache_control": "ephemeral"}]),
                    # top-level + 4 explicit
                    msg([{"type": "text", "text": str(i), "cache_control": {"type": "ephemeral"}} for i in range(4)],
                        cache_control={"type": "ephemeral"}),
                    # top-level ttl disagrees with the explicit marker on the last block
                    {"model": "claude-opus-5", "max_tokens": 5, "cache_control": {"type": "ephemeral", "ttl": "1h"},
                     "messages": [{"role": "user", "content": [{"type": "text", "text": "x",
                                                                "cache_control": {"type": "ephemeral"}}]}]},
                ]
                for body in cases:
                    r = await c.post("/v1/messages", json=body)
                    self.assertEqual(r.status_code, 400, body)
                    self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
                    self.assertIn("cache_control", r.json()["error"]["message"])
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
                # Bedrock route: ValidationException
                r = await c.post("/model/anthropic.claude-opus-5/invoke", json={
                    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 5,
                    "system": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral", "ttl": "2h"}}],
                    "messages": [{"role": "user", "content": "x"}]})
                self.assertEqual((r.status_code, r.json()["__type"]), (400, "ValidationException"))
                # valid: top-level ttl equal to the explicit last marker; 1h before 5m
                t = await self._pending(c, "/v1/messages", {
                    "model": "claude-opus-5", "max_tokens": 5, "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    "system": [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "x",
                                                               "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]})
                await c.post("/_control/auto", json={"text": "ok"})
                self.assertEqual((await t).status_code, 200)
        _run(run())

    # salt: explicit defaults equal omission; speed invalidates system + messages, not tools
    def test_salt_defaults_and_speed(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=0)
        tools = [{"name": "t", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}]
        system = [{"type": "text", "text": "S" * 200, "cache_control": {"type": "ephemeral"}}]
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Q" * 200, "cache_control": {"type": "ephemeral"}}]}]
        sim.observe(analyze_request(system, tools, msgs, params={}), "m", 0.0)
        for params in ({"tool_choice": "auto"}, {"tool_choice": {"type": "auto"}},
                       {"thinking": {"type": "disabled"}}, {"output_config": {"effort": "high"}}):
            r = sim.observe(analyze_request(system, tools, msgs, params=params), "m", 1.0)
            self.assertEqual(r["read_seg_count"], 3, params)
        r = sim.observe(analyze_request(system, tools, msgs, params={"speed": "fast"}), "m", 2.0)
        self.assertEqual(r["read_seg_count"], 1)  # tools prefix survives, system + messages do not

    # refusal takes OpenAI's shape: message.refusal / delta.refusal, content null, finish "stop"
    def test_openai_refusal_shape(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/chat/completions", {"model": "gpt-5.4",
                                        "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={"content": [{"type": "text", "text": "I can't."}],
                                                        "stop_reason": "refusal"})
                ch = (await t).json()["choices"][0]
                self.assertEqual((ch["message"]["content"], ch["message"]["refusal"], ch["finish_reason"]),
                                 (None, "I can't.", "stop"))
                t = await self._pending(c, "/v1/chat/completions", {"model": "gpt-5.4", "stream": True,
                                        "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={"content": [{"type": "text", "text": "No."}],
                                                        "stop_reason": "refusal"})
                lines = [json.loads(ln[6:]) for ln in (await t).text.splitlines()
                         if ln.startswith("data: ") and not ln.endswith("[DONE]")]
                deltas = [ch["choices"][0]["delta"] for ch in lines]
                self.assertIn({"refusal": "No."}, deltas)
                self.assertFalse(any(d.get("content") for d in deltas))
        _run(run())

    # relay host parsing, geo pricing, batch error request_id, ARN partitions, header sanitizing
    def test_max_tokens_param_host_parsing(self) -> None:
        import argparse
        from puppetllm.relay import resolve_max_tokens_param as f
        self.assertEqual(f(argparse.Namespace(max_tokens_param="auto", target="https://api.openai.com:443/v1")),
                         "max_completion_tokens")
        self.assertEqual(f(argparse.Namespace(max_tokens_param="auto", target="https://evilopenai.com/v1")),
                         "max_tokens")
        self.assertEqual(f(argparse.Namespace(max_tokens_param="auto", target="https://API.OpenAI.com/v1")),
                         "max_completion_tokens")

    def test_relay_output_config_scalar_does_not_crash(self) -> None:
        from puppetllm.relay import to_anthropic_request
        body = to_anthropic_request({"messages": [{"role": "user", "content": "x"}],
                                     "params": {"output_config": "high", "reasoning_effort": "low",
                                                "service_tier": "flex"}}, "claude-opus-5")
        self.assertEqual(body["output_config"], {"effort": "low"})
        self.assertEqual(body["service_tier"], "auto")

    def test_us_inference_geo_multiplier(self) -> None:
        c = pricing.compute_cost("claude-opus-5", input_tokens=1_000_000, inference_geo="us")
        self.assertAlmostEqual(c["input_usd"], 5.5, places=6)
        self.assertEqual(c["geo_multiplier"], 1.1)
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {"model": "claude-opus-5", "max_tokens": 5,
                                        "inference_geo": "us", "messages": [{"role": "user", "content": "x"}]})
                await c.post("/_control/respond", json={"content": [{"type": "text", "text": "ok"}],
                                                        "usage": {"input_tokens": 1000, "output_tokens": 0,
                                                                  "cache_creation_input_tokens": 0,
                                                                  "cache_read_input_tokens": 0}})
                await t
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertAlmostEqual(h["cost"]["input_usd"], 1000 * 5.5 / 1_000_000, places=9)
        _run(run())

    def test_batch_errors_carry_request_id(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                r = await c.get("/v1/messages/batches", params={"limit": 0})
                self.assertEqual(r.status_code, 400)
                self.assertTrue(r.json()["request_id"].startswith("req_"))
                self.assertEqual(r.headers["request-id"], r.json()["request_id"])
        _run(run())

    def test_arn_partitions(self) -> None:
        from puppetllm.providers.bedrock import normalize_model_id
        for arn in ("arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:inference-profile/us-gov.anthropic.claude-opus-5",
                    "arn:aws-cn:bedrock:cn-north-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0"):
            self.assertTrue(normalize_model_id(arn).canonical.startswith("claude-"), arn)

    def test_inject_headers_sanitized(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {"model": "claude-opus-5", "max_tokens": 5,
                                        "messages": [{"role": "user", "content": "x"}]})
                for bad in ({"retry-after": "3\r\nx-evil: 1"}, {"content-length": "0"},
                            {"transfer-encoding": "chunked"}, {"bad name": "1"}, {"x": "\u00e9\u4e2d"},
                            {"x": True}):
                    r = await c.post("/_control/error", json={"status": 429, "headers": bad})
                    self.assertEqual(r.status_code, 400, bad)
                await c.post("/_control/error", json={"status": 429, "headers": {"retry-after": "3"}})
                self.assertEqual((await t).headers["retry-after"], "3")
        _run(run())

    def test_usage_override_shapes_and_stop_details_passthrough(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {"model": "claude-opus-5", "max_tokens": 5,
                                        "system": [{"type": "text", "text": "S" * 400,
                                                    "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                                        "messages": [{"role": "user", "content": "x"}]})
                for bad in ({"input_tokens": 1, "cache_creation": {}},
                            {"input_tokens": 1, "cache_creation": {"ephemeral_5m_input_tokens": 1}},
                            {"input_tokens": 1, "service_tier": None},
                            {"input_tokens": 1, "server_tool_use": {}}):
                    r = await c.post("/_control/respond", json={"content": [], "usage": bad})
                    self.assertEqual(r.status_code, 400, bad)
                # a partial override keeps the sim's 1h split; extra stop_details fields survive
                await c.post("/_control/respond", json={
                    "content": [], "stop_reason": "refusal",
                    "stop_details": {"category": "cyber", "recommended_model": "claude-opus-5"},
                    "usage": {"output_tokens": 7}})
                j = (await t).json()
                self.assertGreater(j["usage"]["cache_creation"]["ephemeral_1h_input_tokens"], 0)
                self.assertEqual(j["stop_details"], {"type": "refusal", "category": "cyber",
                                                     "explanation": None,
                                                     "recommended_model": "claude-opus-5"})
        _run(run())


# ── billing consistency, breakpoint limits, thinking defaults, fast mode ──


class TestBillingAndCacheDefaults(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

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

    # strip_private removes only puppetllm's keys, at block / tool level
    def test_strip_private_keeps_application_underscore_keys(self) -> None:
        from puppetllm.openai_wire import strip_private
        from puppetllm.relay import to_anthropic_request
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "u"}, "_openai_detail": "low"},
                {"type": "tool_result", "tool_use_id": "t", "content": [
                    {"type": "image", "source": {"type": "url", "url": "u2"}, "_openai_detail": "high"}]}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "c1", "name": "db", "_openai_custom": True,
                 "input": {"_id": "42", "filter": {"_rev": 1}}},
                {"type": "tool_use", "id": "c2", "name": "f", "input": {"_raw": "{broken"}}]}]
        out = strip_private(msgs)
        self.assertNotIn("_openai_detail", out[0]["content"][0])
        self.assertNotIn("_openai_detail", out[0]["content"][1]["content"][0])
        self.assertNotIn("_openai_custom", out[1]["content"][0])
        self.assertEqual(out[1]["content"][0]["input"], {"_id": "42", "filter": {"_rev": 1}})
        self.assertEqual(out[1]["content"][1]["input"], {"_raw": "{broken"})
        body = to_anthropic_request({"messages": msgs, "tools": [
            {"name": "db", "input_schema": {"type": "object", "properties": {"_id": {"type": "string"}},
                                            "required": ["_id"]}, "_openai_custom": True, "format": {"type": "text"}}]},
            "claude-opus-5")
        self.assertEqual(body["tools"][0]["input_schema"]["properties"], {"_id": {"type": "string"}})
        self.assertEqual(sorted(body["tools"][0]), ["input_schema", "name"])
        async def run() -> None:
            async with await self._client() as c:
                for stream in (False, True):
                    t = await self._pending(c, "/v1/messages", {"model": "claude-opus-5", "max_tokens": 5,
                                            "stream": stream, "messages": [{"role": "user", "content": "x"}]})
                    await c.post("/_control/respond", json={"content": [
                        {"type": "tool_use", "name": "db", "input": {"_id": "42", "q": {"_rev": 2}},
                         "_openai_custom": True}]})
                    r = await t
                    if stream:
                        parts = [d["delta"]["partial_json"] for n, d in _sse_events(r.text)
                                 if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta"]
                        got = json.loads("".join(parts))
                    else:
                        blk = r.json()["content"][0]
                        self.assertNotIn("_openai_custom", blk)
                        got = blk["input"]
                    self.assertEqual(got, {"_id": "42", "q": {"_rev": 2}})
        _run(run())

    # >4 explicit markers and a marker on a thinking block → 400
    def test_explicit_breakpoint_limits_400(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                five = {"model": "claude-opus-5", "max_tokens": 5, "messages": [
                    {"role": "user", "content": [{"type": "text", "text": str(i),
                                                  "cache_control": {"type": "ephemeral"}} for i in range(5)]}]}
                r = await c.post("/v1/messages", json=five)
                self.assertEqual(r.status_code, 400)
                self.assertIn("at most 4", r.json()["error"]["message"])
                on_thinking = {"model": "claude-opus-5", "max_tokens": 5, "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s",
                                                       "cache_control": {"type": "ephemeral"}}]},
                    {"role": "user", "content": "q2"}]}
                r = await c.post("/v1/messages", json=on_thinking)
                self.assertEqual(r.status_code, 400)
                self.assertIn("thinking", r.json()["error"]["message"])
                self.assertFalse((await c.get("/_control/pending")).json()["has_pending"])
        _run(run())

    # n>1: wire, history and stats agree
    def test_n_choices_billed_consistently(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/chat/completions", {"model": "gpt-5.4", "n": 3,
                                        "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "hello " * 20})
                wire = (await t).json()["usage"]
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(h["usage"]["output_tokens"], wire["completion_tokens"])
                self.assertEqual(h["usage"]["output_tokens"] % 3, 0)
                single = pricing.compute_cost("gpt-5.4", output_tokens=wire["completion_tokens"] // 3)["output_usd"]
                self.assertAlmostEqual(h["cost"]["output_usd"], single * 3, places=9)
                st = (await c.get("/_control/stats")).json()
                self.assertEqual(st["totals"]["output_tokens"], wire["completion_tokens"])
                # an override (one generated answer) is billed n times as well
                t = await self._pending(c, "/v1/chat/completions", {"model": "gpt-5.4", "n": 2,
                                        "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={"content": [{"type": "text", "text": "a"}],
                                                        "usage": {"input_tokens": 5, "output_tokens": 7}})
                self.assertEqual((await t).json()["usage"]["completion_tokens"], 14)
                self.assertEqual((await c.get("/_control/history")).json()["history"][-1]["usage"]["output_tokens"], 14)
        _run(run())

    # thinking default is model-aware
    def test_thinking_default_salt_per_model(self) -> None:
        from puppetllm.cache_sim import thinking_default_for
        self.assertEqual(thinking_default_for("claude-opus-5"), {"type": "adaptive"})
        self.assertEqual(thinking_default_for("claude-sonnet-5"), {"type": "adaptive"})
        self.assertEqual(thinking_default_for("claude-fable-5-1"), {"type": "adaptive"})
        self.assertEqual(thinking_default_for("claude-opus-4-8"), {"type": "disabled"})
        self.assertEqual(thinking_default_for("claude-haiku-4-5"), {"type": "disabled"})
        system = [{"type": "text", "text": "S" * 200, "cache_control": {"type": "ephemeral"}}]
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Q" * 200, "cache_control": {"type": "ephemeral"}}]}]
        for model, same, changed in (
                ("claude-opus-5", {"type": "adaptive", "display": "omitted"}, {"type": "disabled"}),
                ("claude-opus-4-8", {"type": "disabled"}, {"type": "adaptive"})):
            sim = CacheSimulator(min_cacheable_tokens=0)
            sim.observe(analyze_request(system, None, msgs, params={}, model=model), model, 0.0)
            r = sim.observe(analyze_request(system, None, msgs, params={"thinking": same}, model=model), model, 1.0)
            self.assertEqual(r["read_seg_count"], 2, (model, same))
            r = sim.observe(analyze_request(system, None, msgs, params={"thinking": changed}, model=model), model, 2.0)
            self.assertEqual(r["read_seg_count"], 1, (model, changed))
        # tool_choice with an explicit default option equals the default
        sim = CacheSimulator(min_cacheable_tokens=0)
        sim.observe(analyze_request(system, None, msgs, params={}), "m", 0.0)
        r = sim.observe(analyze_request(system, None, msgs,
                                        params={"tool_choice": {"type": "auto", "disable_parallel_tool_use": False}}), "m", 1.0)
        self.assertEqual(r["read_seg_count"], 2)

    # fast mode: 2x pricing, usage.speed, relay forwards / warns
    def test_fast_mode(self) -> None:
        c = pricing.compute_cost("claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000, speed="fast")
        self.assertAlmostEqual(c["input_usd"] + c["output_usd"], 60.0, places=6)
        self.assertEqual(c["speed_multiplier"], 2.0)
        from puppetllm.relay import to_anthropic_request, to_openai_request
        body = to_anthropic_request({"messages": [{"role": "user", "content": "x"}],
                                     "params": {"speed": "fast"}}, "claude-opus-5")
        self.assertEqual(body["speed"], "fast")
        body = to_openai_request({"messages": [{"role": "user", "content": "x"}],
                                  "params": {"speed": "fast"}}, "gpt-5.4")
        self.assertNotIn("speed", body)
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/messages", {"model": "claude-opus-5", "max_tokens": 5,
                                        "speed": "fast", "messages": [{"role": "user", "content": "x"}]})
                await c.post("/_control/respond", json={"content": [{"type": "text", "text": "ok"}],
                                                        "usage": {"input_tokens": 1000, "output_tokens": 0,
                                                                  "cache_creation_input_tokens": 0,
                                                                  "cache_read_input_tokens": 0}})
                j = (await t).json()
                self.assertEqual(j["usage"]["speed"], "fast")
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertAlmostEqual(h["cost"]["input_usd"], 1000 * 10.0 / 1_000_000, places=9)
                t = await self._pending(c, "/v1/messages", {"model": "claude-opus-5", "max_tokens": 5,
                                        "messages": [{"role": "user", "content": "x"}]})
                await c.post("/_control/auto", json={"text": "ok"})
                self.assertNotIn("speed", (await t).json()["usage"])
        _run(run())


# ── relay beta headers, partial overrides, salt / validation refinements ──


class TestRelayAndUsageRefinements(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

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

    # relay re-sends the app's anthropic-beta header to an Anthropic upstream only
    def test_relay_forwards_beta_header(self) -> None:
        import argparse
        from puppetllm.relay import Relay
        os.environ["RELAY_KEY"] = "k"
        for kind, expected in (("anthropic", {"anthropic-beta": "fast-mode-2026-02-01,compact-2026-01-12"}),
                               ("openai", {})):
            cfg = argparse.Namespace(kind=kind, target="http://127.0.0.1:1/v1", api_key_env="RELAY_KEY",
                                     model=None, model_map=None, only=None, max_concurrency=0,
                                     timeout=1, poll_timeout=1, puppet="http://127.0.0.1:1",
                                     max_tokens_param="auto", max_requests=1)
            relay = Relay(cfg)
            try:
                self.assertEqual(relay._request_headers(
                    {"params": {"anthropic_beta": ["fast-mode-2026-02-01", "compact-2026-01-12"]}}), expected)
                self.assertEqual(relay._request_headers({"params": {}}), {})
                self.assertEqual(relay._request_headers({"params": {"anthropic_beta": "x-1"}}),
                                 {"anthropic-beta": "x-1"} if kind == "anthropic" else {})
            finally:
                asyncio.run(relay.close())

    # a partial usage override with n>1 scales only the overridden keys
    def test_partial_override_not_double_folded(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._pending(c, "/v1/chat/completions", {"model": "gpt-5.4", "n": 2,
                                        "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "thinking", "thinking": "T" * 200}, {"type": "text", "text": "hello " * 20}]})
                base = (await t).json()["usage"]
                t = await self._pending(c, "/v1/chat/completions", {"model": "gpt-5.4", "n": 2,
                                        "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "thinking", "thinking": "T" * 200}, {"type": "text", "text": "hello " * 20}],
                    "usage": {"input_tokens": 5}})
                u = (await t).json()["usage"]
                self.assertEqual(u["completion_tokens"], base["completion_tokens"])
                self.assertEqual(u["completion_tokens_details"]["reasoning_tokens"],
                                 base["completion_tokens_details"]["reasoning_tokens"])
                self.assertEqual(u["prompt_tokens"], 5)
                # thinking_tokens never exceeds an overridden output total
                t = await self._pending(c, "/v1/messages", {"model": "claude-opus-5", "max_tokens": 5,
                                        "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={"content": [{"type": "thinking", "thinking": "T" * 400}],
                                                        "usage": {"output_tokens": 7}})
                u = (await t).json()["usage"]
                self.assertEqual((u["output_tokens"], u["output_tokens_details"]["thinking_tokens"]), (7, 7))
        _run(run())

    # thinking.display never salts; top-level 1h after an explicit 5m → 400; empty-text markers count
    def test_cache_salt_and_validation_refinements(self) -> None:
        system = [{"type": "text", "text": "S" * 200, "cache_control": {"type": "ephemeral"}}]
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Q" * 200, "cache_control": {"type": "ephemeral"}}]}]
        sim = CacheSimulator(min_cacheable_tokens=0)
        sim.observe(analyze_request(system, None, msgs, params={"thinking": {"type": "adaptive"}}, model="claude-opus-4-6"),
                    "claude-opus-4-6", 0.0)
        r = sim.observe(analyze_request(system, None, msgs,
                                        params={"thinking": {"type": "adaptive", "display": "summarized"}},
                                        model="claude-opus-4-6"), "claude-opus-4-6", 1.0)
        self.assertEqual(r["read_seg_count"], 2)
        async def run() -> None:
            async with await self._client() as c:
                r = await c.post("/v1/messages", json={
                    "model": "claude-opus-5", "max_tokens": 5, "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    "system": [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": "x"}]})
                self.assertEqual(r.status_code, 400)
                self.assertIn("1h", r.json()["error"]["message"])
                five = {"model": "claude-opus-5", "max_tokens": 5, "messages": [
                    {"role": "user", "content": [{"type": "text", "text": str(i), "cache_control": {"type": "ephemeral"}}
                                                 for i in range(4)] + [
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}]}]}
                r = await c.post("/v1/messages", json=five)
                self.assertEqual(r.status_code, 400)
                self.assertIn("at most 4", r.json()["error"]["message"])
                # OpenAI route never prices `speed`
                t = await self._pending(c, "/v1/chat/completions", {"model": "gpt-5.4", "speed": "fast",
                                        "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "ok"})
                await t
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual(h["cost"]["speed_multiplier"], 1.0)
                self.assertNotIn("speed", h["usage"])
        _run(run())

    # relay strips private keys from system blocks and stringified tool_result blocks too
    def test_relay_strips_private_in_system_and_tool_result_text(self) -> None:
        from puppetllm.relay import to_anthropic_request, to_openai_request
        req = {"system": [{"type": "text", "text": "s", "_openai_detail": "x"}],
               "messages": [{"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "t", "content": [
                       {"type": "image", "source": {"type": "url", "url": "u"}, "_openai_detail": "low"}]}]}]}
        body = to_anthropic_request(req, "claude-opus-5")
        self.assertEqual(body["system"], [{"type": "text", "text": "s"}])
        body = to_openai_request(req, "gpt-5.4")
        self.assertNotIn("_openai_detail", body["messages"][1]["content"])


# ── relay (pure) ───────────────────────────────────────────────────────


class TestRelayConformance(unittest.TestCase):
    def test_max_tokens_param_auto(self) -> None:
        import argparse
        from puppetllm.relay import resolve_max_tokens_param as f, to_openai_request
        self.assertEqual(f(argparse.Namespace(max_tokens_param="auto", target="https://api.openai.com/v1")),
                         "max_completion_tokens")
        self.assertEqual(f(argparse.Namespace(max_tokens_param="auto", target="https://api.x.ai/v1")),
                         "max_tokens")
        self.assertEqual(f(argparse.Namespace(max_tokens_param="max_tokens", target="https://api.openai.com/v1")),
                         "max_tokens")
        self.assertEqual(f(None), "max_tokens")
        body = to_openai_request({"messages": [{"role": "user", "content": "x"}], "max_tokens": 7}, "gpt-5.4",
                                 argparse.Namespace(max_tokens_param="auto", target="https://api.openai.com/v1"))
        self.assertEqual(body["max_completion_tokens"], 7)
        self.assertNotIn("max_tokens", body)

    def test_to_openai_request_images_tools_params(self) -> None:
        from puppetllm.relay import to_openai_request
        body = to_openai_request({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
                    {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_9", "name": "sql", "input": {"input": "select 1"},
                     "_openai_custom": True}]}],
            "tools": [
                {"name": "f", "description": "d", "input_schema": {"type": "object"}, "strict": True},
                {"name": "sql", "_openai_custom": True, "format": {"type": "text"}}],
            "params": {"seed": 3, "verbosity": "low", "prompt_cache_key": "k", "n": 2,
                       "output_config": {"effort": "max"}, "metadata": {"user_id": "u"},
                       "service_tier": "standard_only"},
        }, "gpt-5.4")
        parts = body["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "look"})
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/jpeg;base64,QUJD")
        self.assertEqual(parts[2]["image_url"]["url"], "https://x/y.png")
        tc = body["messages"][1]["tool_calls"][0]
        self.assertEqual((tc["id"], tc["type"], tc["custom"]["input"]), ("call_9", "custom", "select 1"))
        self.assertTrue(body["tools"][0]["function"]["strict"])
        self.assertEqual(body["tools"][1], {"type": "custom", "custom": {"name": "sql", "format": {"type": "text"}}})
        for k, v in (("seed", 3), ("verbosity", "low"), ("prompt_cache_key", "k"),
                     ("reasoning_effort", "xhigh"),   # Anthropic `max` clamps to OpenAI `xhigh`
                     ("service_tier", "default"),     # Anthropic `standard_only` → OpenAI `default`
                     ("metadata", {"user_id": "u"})):
            self.assertEqual(body[k], v, k)
        self.assertNotIn("n", body)  # extra samples would only be billed
        # image detail round-trips
        self.assertEqual(parts[1]["image_url"].get("detail"), None)
        body = to_openai_request({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}, "_openai_detail": "low"}]}]},
            "gpt-5.4")
        self.assertEqual(body["messages"][0]["content"][0]["image_url"]["detail"], "low")

    def test_from_openai_response_refusal_custom_cache_write(self) -> None:
        from puppetllm.relay import from_openai_response
        out = from_openai_response({"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": None, "refusal": "I can't do that."}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                      "prompt_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 30}}})
        self.assertEqual(out["content"], [{"type": "text", "text": "I can't do that."}])
        self.assertEqual(out["stop_reason"], "refusal")
        self.assertEqual(out["usage"], {"input_tokens": 50, "output_tokens": 5,
                                        "cache_read_input_tokens": 20, "cache_creation_input_tokens": 30})
        out = from_openai_response({"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "custom", "custom": {"name": "sql", "input": "select 1"}}]}}]})
        blk = out["content"][0]
        self.assertEqual((blk["type"], blk["name"], blk["input"], blk["_openai_custom"]),
                         ("tool_use", "sql", {"input": "select 1"}, True))
        self.assertEqual(out["stop_reason"], "tool_use")

    def test_from_anthropic_response_keeps_thinking_and_usage_objects(self) -> None:
        from puppetllm.relay import from_anthropic_response
        out = from_anthropic_response({
            "content": [{"type": "thinking", "thinking": "", "signature": "real"},
                        {"type": "text", "text": "hi"}],
            "stop_reason": "refusal", "stop_sequence": None,
            "stop_details": {"type": "refusal", "category": "cyber", "explanation": None},
            "usage": {"input_tokens": 10, "output_tokens": 3, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0,
                      "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                      "output_tokens_details": {"thinking_tokens": 2},
                      "service_tier": "standard", "inference_geo": "global",
                      "server_tool_use": None}})
        self.assertEqual(out["content"][0]["signature"], "real")
        self.assertEqual(out["stop_details"]["category"], "cyber")
        self.assertEqual(out["usage"]["output_tokens_details"], {"thinking_tokens": 2})
        self.assertEqual(out["usage"]["service_tier"], "standard")
        self.assertNotIn("server_tool_use", out["usage"])  # None is not forwarded
        self.assertNotIn("stop_sequence", out)


if __name__ == "__main__":
    unittest.main()
