"""Tests for puppetllm extensions: cost estimates / pseudo-cache / Bedrock path / OpenAI path.

Kept separate from the existing test_fake_server.py (Anthropic path + control API regressions).

Run:
  python3 -m unittest puppetllm.tests.test_proxy_extensions -v
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import unittest
from typing import Any

from puppetllm import pricing
from puppetllm.cache_sim import CacheSimulator, extract_cache_prefix, analyze_request
from puppetllm.providers import eventstream

# HTTP tests use small prefixes, so disable the minimum cache threshold.
# (The floor behavior itself is verified by TestCacheMinFloor against CacheSimulator alone.
# Setting it here lets the plain `python3 -m unittest ...` from the docstring run as-is.)
os.environ["PUPPETLLM_CACHE_MIN_TOKENS"] = "0"


def _import_fresh():
    """Reload fake_server to clean server state (pending/history/cache)."""
    import importlib
    from puppetllm import fake_server as fs
    importlib.reload(fs)
    return fs


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ── pricing (pure) ───────────────────────────────────────────────────


class TestPricing(unittest.TestCase):
    def test_resolve_family(self) -> None:
        self.assertEqual(pricing.resolve_family("claude-opus-4-20250514"), "opus")
        self.assertEqual(pricing.resolve_family("anthropic.claude-3-5-sonnet-20241022-v2:0"), "sonnet")
        self.assertEqual(pricing.resolve_family("us.anthropic.claude-haiku-4-5"), "haiku")
        # unknown falls back to the default (sonnet)
        self.assertEqual(pricing.resolve_family("mystery-model"), "sonnet")
        self.assertEqual(pricing.resolve_family(None), "sonnet")

    def test_resolve_family_openai(self) -> None:
        self.assertEqual(pricing.resolve_family("gpt-5.4-2026-03-01"), "gpt-5.4")
        self.assertEqual(pricing.resolve_family("gpt-5.4-mini"), "gpt-5.4-mini")
        self.assertEqual(pricing.resolve_family("gpt-5.5-pro"), "gpt-5.5-pro")
        self.assertEqual(pricing.resolve_family("gpt-5.3-codex"), "codex")
        self.assertEqual(pricing.resolve_family("gpt-5-mini-2025-08-07"), "gpt-5-mini")
        # legacy generations (plain/5.1/5.2) are absorbed into gpt-5
        self.assertEqual(pricing.resolve_family("gpt-5.2"), "gpt-5")
        self.assertEqual(pricing.resolve_family("gpt-4o-mini"), "gpt-4o-mini")
        self.assertEqual(pricing.resolve_family("o4-mini"), "o4-mini")
        self.assertEqual(pricing.resolve_family("o3-2025-04-16"), "o3")
        # unknown but containing gpt → current mid tier (gpt-5.4); otherwise sonnet as before
        self.assertEqual(pricing.resolve_family("gpt-9-experimental"), "gpt-5.4")
        self.assertEqual(pricing.resolve_family("mystery-model"), "sonnet")

    def test_compute_cost_openai(self) -> None:
        c = pricing.compute_cost("gpt-5.4", input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertEqual(c["model_family"], "gpt-5.4")
        self.assertAlmostEqual(c["input_usd"], 2.50, places=6)
        self.assertAlmostEqual(c["output_usd"], 15.0, places=6)
        # OpenAI has no cache-write surcharge (= input unit price); read uses the official cached input price
        c2 = pricing.compute_cost("gpt-5.5", cache_write_tokens=1_000_000)
        self.assertAlmostEqual(c2["cache_write_usd"], 5.0, places=6)
        c3 = pricing.compute_cost("gpt-5.5", cache_read_tokens=1_000_000)
        self.assertAlmostEqual(c3["cache_read_usd"], 0.50, places=6)

    def test_approx_tokens(self) -> None:
        self.assertEqual(pricing.approx_tokens(""), 0)
        self.assertEqual(pricing.approx_tokens(None), 0)
        self.assertEqual(pricing.approx_tokens("abcd"), 1)       # 4 chars / 4
        self.assertEqual(pricing.approx_tokens("a" * 8), 2)
        # a dict is JSON-serialized length / 4 (>0)
        self.assertGreater(pricing.approx_tokens({"k": "value"}), 0)

    def test_compute_cost_opus(self) -> None:
        c = pricing.compute_cost("claude-opus-4", input_tokens=1_000_000, output_tokens=0)
        self.assertEqual(c["model_family"], "opus")
        self.assertTrue(c["is_estimate"])
        self.assertAlmostEqual(c["input_usd"], 5.0, places=6)    # opus 4.x input $5/Mtok
        c2 = pricing.compute_cost("claude-opus-4", output_tokens=1_000_000)
        self.assertAlmostEqual(c2["output_usd"], 25.0, places=6)  # opus 4.x output $25/Mtok

    def test_compute_cost_cache_split(self) -> None:
        c = pricing.compute_cost(
            "sonnet", input_tokens=0, output_tokens=0,
            cache_write_tokens=1_000_000, cache_read_tokens=1_000_000,
        )
        self.assertAlmostEqual(c["cache_write_usd"], 3.75, places=6)
        self.assertAlmostEqual(c["cache_read_usd"], 0.30, places=6)

    def test_cache_savings(self) -> None:
        # sonnet: input 3.0, cache_read 0.30 → a 1Mtok read saves about 2.70 USD
        self.assertAlmostEqual(pricing.cache_savings_usd("sonnet", 1_000_000), 2.70, places=6)


# ── cache_sim (pure) ─────────────────────────────────────────────────


class TestCachePrefix(unittest.TestCase):
    def test_no_cache_control_returns_none(self) -> None:
        self.assertIsNone(extract_cache_prefix(
            system="plain system",
            messages=[{"role": "user", "content": "hi"}],
        ))

    def test_detects_breakpoint_and_is_stable(self) -> None:
        system = [{"type": "text", "text": "big stable prompt",
                   "cache_control": {"type": "ephemeral"}}]
        p1 = extract_cache_prefix(system=system, messages=[{"role": "user", "content": "a"}])
        p2 = extract_cache_prefix(system=system, messages=[{"role": "user", "content": "DIFFERENT"}])
        self.assertIsNotNone(p1)
        self.assertEqual(p1.breakpoints, 1)
        self.assertGreater(p1.tokens, 0)
        # even if the dynamic messages part differs, the prefix (up to system) has the same hash
        self.assertEqual(p1.hash, p2.hash)

    def test_prefix_changes_with_system(self) -> None:
        a = extract_cache_prefix(system=[{"type": "text", "text": "A",
                                          "cache_control": {"type": "ephemeral"}}])
        b = extract_cache_prefix(system=[{"type": "text", "text": "B",
                                          "cache_control": {"type": "ephemeral"}}])
        self.assertNotEqual(a.hash, b.hash)


class TestCacheSimulator(unittest.TestCase):
    def _rc(self, text: str = "stable"):
        """A RequestCache with 1 breakpoint on system."""
        return analyze_request(system=[{"type": "text", "text": text,
                                        "cache_control": {"type": "ephemeral"}}])

    def test_miss_then_hit(self) -> None:
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        rc = self._rc()
        sys_tok = rc.prefix_tokens(rc.breakpoints[-1] + 1)
        r1 = sim.observe(rc, "sonnet", now=0.0)
        self.assertEqual(r1["status"], "miss")
        self.assertEqual(r1["cache_creation_tokens"], sys_tok)
        self.assertEqual(r1["cache_read_tokens"], 0)
        r2 = sim.observe(rc, "sonnet", now=10.0)
        self.assertEqual(r2["status"], "hit")
        self.assertEqual(r2["cache_read_tokens"], sys_tok)
        self.assertEqual(r2["cache_creation_tokens"], 0)

    def test_no_breakpoint_is_none(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=0)
        rc = analyze_request(system="plain", messages=[{"role": "user", "content": "hi"}])
        r = sim.observe(rc, "sonnet", now=0.0)
        self.assertEqual(r["status"], "none")
        self.assertEqual(r["cache_read_tokens"], 0)
        self.assertEqual(r["cache_creation_tokens"], 0)

    def test_ttl_expiry(self) -> None:
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        rc = self._rc()
        sim.observe(rc, "sonnet", now=0.0)            # miss (creation)
        r = sim.observe(rc, "sonnet", now=400.0)      # TTL expired → miss, recreated
        self.assertEqual(r["status"], "miss")

    def test_ttl_honor_disabled(self) -> None:
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=False, min_cacheable_tokens=0)
        rc = self._rc()
        sim.observe(rc, "sonnet", now=0.0)
        r = sim.observe(rc, "sonnet", now=99999.0)    # TTL ignored → hit
        self.assertEqual(r["status"], "hit")

    def test_hit_refreshes_ttl(self) -> None:
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        rc = self._rc()
        sim.observe(rc, "sonnet", now=0.0)
        sim.observe(rc, "sonnet", now=250.0)          # hit, extends created_at to 250
        r = sim.observe(rc, "sonnet", now=500.0)      # 250s elapsed since 250 → still alive → hit
        self.assertEqual(r["status"], "hit")

    def test_ttl_boundary_inclusive(self) -> None:
        # age == ttl is exactly alive (`<=`). Pin the boundary that would become a miss with `<`.
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        rc = self._rc()
        sim.observe(rc, "sonnet", now=0.0)
        self.assertEqual(sim.observe(rc, "sonnet", now=300.0)["status"], "hit")

    def test_read_refreshes_ttl_independently(self) -> None:
        # Directly verify that the read path updates created_at independently of the write path.
        # turn2 reads system+u1 (sc=2), but its own BPs are system(sc=1)+u2(sc=3) and it does
        # not write sc=2 → only the read path extends the TTL of sc=2.
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        sysb = [{"type": "text", "text": "S " * 5, "cache_control": {"type": "ephemeral"}}]
        sim.observe(analyze_request(system=sysb, messages=[
            {"role": "user", "content": [{"type": "text", "text": "u1 " * 5,
                                          "cache_control": {"type": "ephemeral"}}]}]), "m", now=0.0)
        r = sim.observe(analyze_request(system=sysb, messages=[
            {"role": "user", "content": [{"type": "text", "text": "u1 " * 5}]},
            {"role": "user", "content": [{"type": "text", "text": "u2 " * 5,
                                          "cache_control": {"type": "ephemeral"}}]}]), "m", now=250.0)
        self.assertEqual(r["read_seg_count"], 2)            # read the non-BP sc=2
        ent = {e["seg_count"]: e for e in sim.entries(now=250.0)}
        self.assertEqual(ent[2]["age_seconds"], 0.0)        # read updates created_at=250 (write path never touches sc=2)

    def test_cache_is_model_scoped(self) -> None:
        # On real hardware the cache is per-model. Even identical content does not hit under a different model.
        sim = CacheSimulator(min_cacheable_tokens=0)
        rc = self._rc("STABLE " * 5)
        self.assertEqual(sim.observe(rc, "opus", now=0.0)["status"], "miss")
        self.assertEqual(sim.observe(rc, "sonnet", now=1.0)["status"], "miss")  # different model → different cache
        self.assertEqual(sim.observe(rc, "opus", now=2.0)["status"], "hit")     # same model → hit

    def test_max_breakpoints_cap(self) -> None:
        # 6 cache_control markers → only the deepest 4 are written, per the real-hardware limit.
        sim = CacheSimulator(min_cacheable_tokens=0)
        msgs = [{"role": "user", "content": [{"type": "text", "text": f"m{i} " * 3,
                                              "cache_control": {"type": "ephemeral"}}]} for i in range(5)]
        rc = analyze_request(
            system=[{"type": "text", "text": "S " * 3, "cache_control": {"type": "ephemeral"}}],
            messages=msgs)
        self.assertEqual(len(rc.breakpoints), 6)
        sim.observe(rc, "m", now=0.0)
        self.assertEqual(len(sim.index), 4)   # only the deepest 4 (the shallowest 2 are dropped)

    def test_incremental_multibreakpoint_prefix_match(self) -> None:
        """★core★ With a system anchor(BP1) + moving tail(BP2), turn2 prefix-matches and reads
        the previous turn's prefix (hits by content match even as the cache_control marker moves)."""
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        sysb = [{"type": "text", "text": "STABLE " * 30, "cache_control": {"type": "ephemeral"}}]

        # turn1: system(BP1) + BP2 at the tail of user u1
        rc1 = analyze_request(system=sysb, messages=[
            {"role": "user", "content": [{"type": "text", "text": "u1 " * 20,
                                          "cache_control": {"type": "ephemeral"}}]},
        ])
        r1 = sim.observe(rc1, "opus", now=0.0)
        self.assertEqual(r1["status"], "miss")
        up_to_u1 = rc1.prefix_tokens(rc1.breakpoints[-1] + 1)
        self.assertEqual(r1["cache_creation_tokens"], up_to_u1)  # writes system+u1

        # turn2: u1 is no longer the tail (no cc) / assistant a1 / BP2 advances onto user u2
        rc2 = analyze_request(system=sysb, messages=[
            {"role": "user", "content": [{"type": "text", "text": "u1 " * 20}]},   # no cc
            {"role": "assistant", "content": [{"type": "text", "text": "a1 " * 10}]},
            {"role": "user", "content": [{"type": "text", "text": "u2 " * 20,
                                          "cache_control": {"type": "ephemeral"}}]},
        ])
        r2 = sim.observe(rc2, "opus", now=5.0)
        self.assertEqual(r2["status"], "hit")
        # prefix-matches and reads the previous turn's "system+u1" prefix (not just system, but up to u1)
        self.assertEqual(r2["cache_read_tokens"], up_to_u1)
        self.assertGreater(r2["cache_read_tokens"], rc2.prefix_tokens(1))  # deeper than system alone
        # creation is only the delta that grew up to u2
        deepest2 = rc2.prefix_tokens(rc2.breakpoints[-1] + 1)
        self.assertEqual(r2["cache_creation_tokens"], deepest2 - up_to_u1)


class TestCacheMinFloor(unittest.TestCase):
    """Minimum cache threshold: if the prefix is below the threshold, it is not cached (prevents over-reporting)."""

    def _rc(self, n_chars: int):
        return analyze_request(system=[{"type": "text", "text": "x" * n_chars,
                                        "cache_control": {"type": "ephemeral"}}])

    def test_below_min_is_none(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=1000)
        rc = self._rc(40)  # ~10 tok << 1000
        r1 = sim.observe(rc, "m", now=0.0)
        self.assertEqual(r1["status"], "none")
        self.assertEqual(r1["cache_creation_tokens"], 0)
        r2 = sim.observe(rc, "m", now=1.0)            # nothing was written, so the 2nd time does not hit either
        self.assertEqual(r2["status"], "none")
        self.assertEqual(r2["cache_read_tokens"], 0)
        self.assertEqual(len(sim.index), 0)            # nothing is written to the index

    def test_at_or_above_min_caches(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=50)
        rc = self._rc(4000)  # ~1000 tok >> 50
        self.assertEqual(sim.observe(rc, "m", now=0.0)["status"], "miss")
        self.assertEqual(sim.observe(rc, "m", now=1.0)["status"], "hit")

    def test_model_based_default_opus_vs_sonnet(self) -> None:
        # the same prefix (~1500 tok) is cached under Sonnet(min 1024) but not under Opus(min 4096).
        rc = self._rc(6000)  # ~1500 tok
        sonnet = CacheSimulator()  # model-based
        self.assertEqual(sonnet.observe(rc, "claude-sonnet-4", now=0.0)["status"], "miss")  # write succeeds
        opus = CacheSimulator()
        self.assertEqual(opus.observe(rc, "claude-opus-4", now=0.0)["status"], "none")      # below threshold

    def test_model_based_haiku_floor(self) -> None:
        # Haiku 4.5 min = 4096: a ~1500-tok prefix is below it → not cached (pins _MIN_CACHEABLE["haiku"]).
        rc = self._rc(6000)  # ~1500 tok, well under 4096
        self.assertEqual(CacheSimulator().observe(rc, "claude-haiku-4-5", now=0.0)["status"], "none")
        # a ~4500-tok prefix is above the haiku floor → cached (miss=write)
        big = analyze_request(system=[{"type": "text", "text": "x" * 18000,
                                       "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(CacheSimulator().observe(big, "claude-haiku-4-5", now=0.0)["status"], "miss")

    def test_floor_boundary_inclusive(self) -> None:
        # prefix == threshold exactly is cached (`>=`); requiring threshold+1 is not cached. Pin against regressing to `>`.
        rc = self._rc(400)
        exact = rc.prefix_tokens(rc.breakpoints[-1] + 1)
        self.assertEqual(CacheSimulator(min_cacheable_tokens=exact).observe(rc, "m", now=0.0)["status"], "miss")
        self.assertEqual(CacheSimulator(min_cacheable_tokens=exact + 1).observe(rc, "m", now=0.0)["status"], "none")


class TestCacheLookback(unittest.TestCase):
    """20-block lookback: a prior prefix more than 20 segments away from the breakpoint cannot be read."""

    def _conv(self, n_msgs: int):
        """system(BP) + user×n. BP2 only on the last user."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": f"m{i} " * 5}]}
                for i in range(n_msgs)]
        if msgs:  # cache_control at the tail
            msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        return analyze_request(
            system=[{"type": "text", "text": "S " * 5, "cache_control": {"type": "ephemeral"}}],
            messages=msgs,
        )

    def test_far_prefix_not_read(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=0)
        # turn1: 1 message (tail BP). system(seg0) and up-to-m0(seg1) are written.
        sim.observe(self._conv(1), "m", now=0.0)
        # turn2: suddenly add 25 messages (>20 blocks in one turn). The tail BP is around seg26.
        #   the previous turn's up-to-m0 prefix (seg2) is more than 20 away → outside lookback.
        #   however system(seg1) is still readable since the system breakpoint itself is an anchor.
        r = sim.observe(self._conv(25), "m", now=1.0)
        # up to system hits (the anchor is near), but it does not reach up-to-m0
        self.assertEqual(r["read_seg_count"], 1)   # only the system prefix (seg_count=1)
        self.assertGreater(r["cache_read_tokens"], 0)

    def test_near_prefix_is_read(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=0)
        sim.observe(self._conv(1), "m", now=0.0)
        r = sim.observe(self._conv(2), "m", now=1.0)  # only +1 message → the prior prefix is near → deep read
        self.assertGreaterEqual(r["read_seg_count"], 2)  # reads up to up-to-m0 (deeper than system)

    def test_lookback_boundary_exactly_20_reads(self) -> None:
        # the distance between sc=2 and the tail BP is exactly 20 → readable (inclusive `<= 20`).
        sim = CacheSimulator(min_cacheable_tokens=0)
        sim.observe(self._conv(1), "m", now=0.0)        # write sc=1, sc=2
        r = sim.observe(self._conv(21), "m", now=1.0)   # tail BP=seg22, distance to sc=2 = 20
        self.assertEqual(r["read_seg_count"], 2)

    def test_lookback_boundary_21_not_read(self) -> None:
        # distance 21 → out of range. Only system(sc=1, anchor distance 0) is read. Also pins against regressing to `< 20`.
        sim = CacheSimulator(min_cacheable_tokens=0)
        sim.observe(self._conv(1), "m", now=0.0)
        r = sim.observe(self._conv(22), "m", now=1.0)   # tail BP=seg23, distance to sc=2 = 21
        self.assertEqual(r["read_seg_count"], 1)


# ── eventstream (pure) ───────────────────────────────────────────────


class TestEventStream(unittest.TestCase):
    def test_chunk_round_trip(self) -> None:
        ev = {"type": "content_block_delta", "index": 0,
              "delta": {"type": "text_delta", "text": "hi"}}
        frame = eventstream.encode_chunk(ev)
        decoded = eventstream.decode_messages(frame)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0], ev)

    def test_multiple_frames(self) -> None:
        a = eventstream.encode_chunk({"type": "message_start"})
        b = eventstream.encode_chunk({"type": "message_stop"})
        decoded = eventstream.decode_messages(a + b)
        self.assertEqual([d["type"] for d in decoded], ["message_start", "message_stop"])

    def test_crc_detects_corruption(self) -> None:
        frame = bytearray(eventstream.encode_chunk({"type": "x"}))
        frame[-1] ^= 0xFF  # corrupt the message CRC
        with self.assertRaises(ValueError):
            eventstream.decode_messages(bytes(frame))


# ── HTTP: cost stats / cache (Anthropic path) ────────────────────────


class TestCostAndCacheHttp(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

    async def _round_trip(self, c, body: dict, reply_text: str = "ok") -> Any:
        t = asyncio.create_task(c.post("/v1/messages", json=body, timeout=10))
        for _ in range(50):
            if (await c.get("/_control/pending")).json().get("has_pending"):
                break
            await asyncio.sleep(0.05)
        else:
            self.fail("never became pending")
        await c.post("/_control/auto", json={"text": reply_text})
        return await t

    def test_stats_reflects_cost_and_tokens(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                r = await self._round_trip(c, {
                    "model": "claude-opus-4-20250514", "stream": False,
                    "system": "x" * 400,
                    "messages": [{"role": "user", "content": "y" * 400}],
                }, reply_text="z" * 200)
                self.assertEqual(r.status_code, 200)
                # usage rides on the response
                usage = r.json()["usage"]
                self.assertGreater(usage["input_tokens"], 0)
                self.assertGreater(usage["output_tokens"], 0)
                # stats aggregation
                s = (await c.get("/_control/stats")).json()
                self.assertTrue(s["is_estimate"])
                self.assertEqual(s["completed_requests"], 1)
                self.assertGreater(s["totals"]["total_usd"], 0.0)
                self.assertIn("claude-opus-4-20250514", s["by_model"])
        _run(run())

    def test_cache_hit_across_identical_prefix(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                system = [{"type": "text", "text": "stable system " * 50,
                           "cache_control": {"type": "ephemeral"}}]
                # 1st time: miss (creation)
                await self._round_trip(c, {
                    "model": "sonnet", "stream": False, "system": system,
                    "messages": [{"role": "user", "content": "first"}],
                })
                # 2nd time: same system prefix → hit
                r2 = await self._round_trip(c, {
                    "model": "sonnet", "stream": False, "system": system,
                    "messages": [{"role": "user", "content": "second different"}],
                })
                usage2 = r2.json()["usage"]
                self.assertGreater(usage2["cache_read_input_tokens"], 0)
                self.assertEqual(usage2["cache_creation_input_tokens"], 0)

                s = (await c.get("/_control/stats")).json()
                self.assertEqual(s["cache"]["hits"], 1)
                self.assertEqual(s["cache"]["misses"], 1)
                self.assertGreater(s["totals"]["cache_savings_usd"], 0.0)

                cache = (await c.get("/_control/cache")).json()
                self.assertEqual(len(cache["entries"]), 1)
                self.assertEqual(cache["entries"][0]["hits"], 1)
        _run(run())

    def test_clear_resets_cache(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                system = [{"type": "text", "text": "s" * 100,
                           "cache_control": {"type": "ephemeral"}}]
                await self._round_trip(c, {
                    "model": "sonnet", "stream": False, "system": system,
                    "messages": [{"role": "user", "content": "a"}],
                })
                self.assertEqual(len((await c.get("/_control/cache")).json()["entries"]), 1)
                await c.post("/_control/clear")
                self.assertEqual(len((await c.get("/_control/cache")).json()["entries"]), 0)
                s = (await c.get("/_control/stats")).json()
                self.assertEqual(s["completed_requests"], 0)
        _run(run())

    def test_uncached_not_clamped_with_string_system_and_message_breakpoint(self) -> None:
        """Even with a string system + a breakpoint on a message, uncached>0 and
        uncached = total - creation (prefix ⊆ total is preserved and does not clamp to 0)."""
        async def run() -> None:
            async with await self._client() as c:
                await self._round_trip(c, {
                    "model": "opus", "stream": False,
                    "system": "plain string system prompt " * 20,  # string system
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "stable ctx " * 40,
                         "cache_control": {"type": "ephemeral"}},  # breakpoint is on the message
                        {"type": "text", "text": "dynamic tail"},
                    ]}],
                }, reply_text="ok")
                h = (await c.get("/_control/history")).json()["history"][-1]
                total = h["request"]["input_tokens_total"]
                u = h["usage"]
                self.assertEqual(u["cache_creation_input_tokens"], h["cache"]["cache_creation_tokens"])
                self.assertEqual(u["input_tokens"], total - u["cache_creation_input_tokens"])
                self.assertGreater(u["input_tokens"], 0)  # not clamped
        _run(run())

    def test_stream_carries_usage(self) -> None:
        """Approximate usage rides on the streamed response (cache on message_start, output on message_delta)."""
        async def run() -> None:
            async with await self._client() as c:
                system = [{"type": "text", "text": "stable " * 60,
                           "cache_control": {"type": "ephemeral"}}]
                # 1st time is a miss and creates the cache
                await self._round_trip(c, {"model": "opus", "stream": False, "system": system,
                                           "messages": [{"role": "user", "content": "warm"}]})
                # 2nd time streams and hits → cache_read should ride on message_start
                t = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "opus", "stream": True, "system": system,
                    "messages": [{"role": "user", "content": "go stream"}]}, timeout=10))
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                await c.post("/_control/auto", json={"text": "streamed reply text"})
                r = await t
                self.assertEqual(r.status_code, 200)
                body = r.text
                self.assertIn("message_delta", body)
                # The cache-read VALUE (not just the key, which is always emitted) must ride
                # on message_start.usage: this is a hit, so cache_read>0 and creation==0.
                start = next(ln[len("data:"):].strip()
                             for ln in body.splitlines()
                             if ln.startswith("data:") and "message_start" in ln)
                start_usage = json.loads(start)["message"]["usage"]
                self.assertGreater(start_usage["cache_read_input_tokens"], 0)
                self.assertEqual(start_usage["cache_creation_input_tokens"], 0)
                # output_tokens > 0 on message_delta
                deltas = re.findall(r'"output_tokens":\s*(\d+)', body)
                self.assertTrue(any(int(x) > 0 for x in deltas), body[:200])
        _run(run())


# ── HTTP: Bedrock path ───────────────────────────────────────────────


class TestBedrockHttp(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

    MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_invoke_non_stream(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = asyncio.create_task(c.post(
                    f"/model/{self.MODEL}/invoke",
                    json={"anthropic_version": "bedrock-2023-05-31", "max_tokens": 100,
                          "messages": [{"role": "user", "content": "hi"}]},
                    timeout=10))
                for _ in range(50):
                    p = (await c.get("/_control/pending")).json()
                    if p.get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                else:
                    self.fail("never became pending")
                # provider is bedrock, model comes from the URL
                self.assertEqual(p["pending"][0]["request"]["provider"], "bedrock")
                self.assertEqual(p["pending"][0]["request"]["model"], self.MODEL)
                await c.post("/_control/auto", json={"text": "bedrock-reply"})
                r = await t
                self.assertEqual(r.status_code, 200)
                body = r.json()
                self.assertEqual(body["content"], [{"type": "text", "text": "bedrock-reply"}])
                self.assertEqual(body["model"], self.MODEL)
        _run(run())

    def test_invoke_with_response_stream(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = asyncio.create_task(c.post(
                    f"/model/{self.MODEL}/invoke-with-response-stream",
                    json={"anthropic_version": "bedrock-2023-05-31", "max_tokens": 100,
                          "messages": [{"role": "user", "content": "hi"}]},
                    timeout=10))
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                else:
                    self.fail("never became pending")
                await c.post("/_control/respond", json={"content": [
                    {"type": "text", "text": "streamed-via-bedrock"}]})
                r = await t
                self.assertEqual(r.status_code, 200)
                self.assertIn("vnd.amazon.eventstream", r.headers.get("content-type", ""))
                # decode the eventstream binary and extract the Anthropic events
                events = eventstream.decode_messages(r.content)
                types = [e.get("type") for e in events]
                self.assertIn("message_start", types)
                self.assertIn("content_block_delta", types)
                self.assertIn("message_stop", types)
                joined = "".join(
                    e.get("delta", {}).get("text", "") for e in events
                    if e.get("type") == "content_block_delta")
                self.assertIn("streamed-via-bedrock", joined)
        _run(run())

    def test_bedrock_error_injection(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = asyncio.create_task(c.post(
                    f"/model/{self.MODEL}/invoke",
                    json={"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                          "messages": [{"role": "user", "content": "hi"}]},
                    timeout=10))
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                else:
                    self.fail("never became pending")
                await c.post("/_control/error", json={
                    "status": 429, "type": "ThrottlingException", "message": "slow down"})
                r = await t
                self.assertEqual(r.status_code, 429)
                self.assertEqual(r.headers.get("x-amzn-ErrorType"), "ThrottlingException")
                self.assertEqual(r.json()["__type"], "ThrottlingException")
        _run(run())

    def test_bedrock_multi_pending(self) -> None:
        """On the Bedrock path too, two concurrent invokes can be answered individually via pending_id (no mix-up)."""
        async def run() -> None:
            async with await self._client() as c:
                def body(u):
                    return {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                            "messages": [{"role": "user", "content": u}]}
                t1 = asyncio.create_task(c.post(f"/model/{self.MODEL}/invoke", json=body("AA"), timeout=10))
                t2 = asyncio.create_task(c.post(f"/model/{self.MODEL}/invoke", json=body("BB"), timeout=10))
                p = {}
                for _ in range(50):
                    p = (await c.get("/_control/pending")).json()
                    if p.get("count") == 2:
                        break
                    await asyncio.sleep(0.05)
                else:
                    self.fail("two bedrock requests never became pending")
                by_msg = {item["request"]["messages"][0]["content"]: item["pending_id"]
                          for item in p["pending"]}
                # both are provider=bedrock
                self.assertTrue(all(item["request"]["provider"] == "bedrock" for item in p["pending"]))
                await c.post("/_control/respond", json={
                    "pending_id": by_msg["AA"], "content": [{"type": "text", "text": "reply-AA"}]})
                await c.post("/_control/respond", json={
                    "pending_id": by_msg["BB"], "content": [{"type": "text", "text": "reply-BB"}]})
                r1 = await t1
                r2 = await t2
                self.assertEqual(r1.json()["content"][0]["text"], "reply-AA")
                self.assertEqual(r2.json()["content"][0]["text"], "reply-BB")
        _run(run())


# ── HTTP: OpenAI path ────────────────────────────────────────────────


class TestOpenAIHttp(unittest.TestCase):
    """Wire-format verification of the OpenAI Chat Completions path (/v1/chat/completions)."""

    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

    async def _round_trip(self, c, body: dict, content_blocks: list) -> tuple:
        """Fire chat/completions and return the pending snapshot and response (response, pending)."""
        t = asyncio.create_task(c.post("/v1/chat/completions", json=body, timeout=10))
        p = {}
        for _ in range(50):
            p = (await c.get("/_control/pending")).json()
            if p.get("has_pending"):
                break
            await asyncio.sleep(0.05)
        else:
            self.fail("never became pending")
        await c.post("/_control/respond", json={"content": content_blocks})
        return (await t), p

    def test_non_stream_text_and_normalization(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                r, p = await self._round_trip(c, {
                    "model": "gpt-5.4",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hi"},
                    ],
                }, [{"type": "text", "text": "openai-reply"}])
                self.assertEqual(r.status_code, 200)
                j = r.json()
                self.assertEqual(j["object"], "chat.completion")
                self.assertEqual(j["model"], "gpt-5.4")
                choice = j["choices"][0]
                self.assertEqual(choice["message"]["role"], "assistant")
                self.assertEqual(choice["message"]["content"], "openai-reply")
                self.assertEqual(choice["finish_reason"], "stop")
                u = j["usage"]
                self.assertGreater(u["prompt_tokens"], 0)
                self.assertGreater(u["completion_tokens"], 0)
                self.assertEqual(u["total_tokens"],
                                 u["prompt_tokens"] + u["completion_tokens"])
                # pending snapshot: provider=openai / system separated / cache is always none
                req = p["pending"][0]["request"]
                self.assertEqual(req["provider"], "openai")
                self.assertEqual(req["system"], "You are helpful.")
                self.assertEqual(req["messages"], [{"role": "user", "content": "hi"}])
                self.assertEqual(req["cache"]["status"], "none")
        _run(run())

    def test_tools_and_tool_result_normalization(self) -> None:
        """OpenAI-format tools / tool_calls / role:tool are normalized to canonical form."""
        async def run() -> None:
            async with await self._client() as c:
                r, p = await self._round_trip(c, {
                    "model": "gpt-5.4",
                    "messages": [
                        {"role": "user", "content": "weather?"},
                        {"role": "assistant", "content": None, "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "get_weather",
                                         "arguments": "{\"city\": \"Tokyo\"}"},
                        }]},
                        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                    ],
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "get weather",
                        "parameters": {"type": "object",
                                       "properties": {"city": {"type": "string"}}},
                    }}],
                }, [{"type": "tool_use", "id": "call_2", "name": "get_weather",
                     "input": {"city": "Osaka"}}])
                # normalization: tools become canonical (input_schema), tool_calls become tool_use blocks,
                # role:tool becomes a tool_result block in a user turn
                req = p["pending"][0]["request"]
                self.assertEqual(req["tools"][0]["name"], "get_weather")
                self.assertIn("input_schema", req["tools"][0])
                asst = req["messages"][1]
                self.assertEqual(asst["content"][0]["type"], "tool_use")
                self.assertEqual(asst["content"][0]["input"], {"city": "Tokyo"})
                toolmsg = req["messages"][2]
                self.assertEqual(toolmsg["role"], "user")
                self.assertEqual(toolmsg["content"][0]["type"], "tool_result")
                self.assertEqual(toolmsg["content"][0]["tool_use_id"], "call_1")
                # response: tool_use block → OpenAI tool_calls (arguments is a JSON string)
                j = r.json()
                choice = j["choices"][0]
                self.assertEqual(choice["finish_reason"], "tool_calls")
                self.assertIsNone(choice["message"]["content"])
                tc = choice["message"]["tool_calls"][0]
                self.assertEqual(tc["id"], "call_2")
                self.assertEqual(tc["type"], "function")
                self.assertEqual(tc["function"]["name"], "get_weather")
                self.assertEqual(json.loads(tc["function"]["arguments"]),
                                 {"city": "Osaka"})
        _run(run())

    def test_stream_chunks_and_done(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                text = "stream me " * 20  # long enough to split into multiple chunks
                r, _ = await self._round_trip(c, {
                    "model": "gpt-5.4", "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "go"}],
                }, [{"type": "text", "text": text}])
                self.assertEqual(r.status_code, 200)
                self.assertIn("text/event-stream", r.headers.get("content-type", ""))
                lines = [ln[len("data: "):] for ln in r.text.splitlines()
                         if ln.startswith("data: ")]
                self.assertEqual(lines[-1], "[DONE]")
                chunks = [json.loads(ln) for ln in lines[:-1]]
                self.assertTrue(all(ch["object"] == "chat.completion.chunk"
                                    for ch in chunks))
                # the first delta is role; concatenating content reconstructs the injected text
                self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
                joined = "".join(ch["choices"][0]["delta"].get("content", "")
                                 for ch in chunks if ch["choices"])
                self.assertEqual(joined, text)
                # a terminal delta with finish_reason + the usage chunk from include_usage
                finishes = [ch["choices"][0]["finish_reason"]
                            for ch in chunks if ch["choices"]]
                self.assertIn("stop", finishes)
                usage_chunks = [ch for ch in chunks if not ch["choices"]]
                self.assertEqual(len(usage_chunks), 1)
                self.assertGreater(usage_chunks[0]["usage"]["prompt_tokens"], 0)
        _run(run())

    def test_error_injection_openai_format(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = asyncio.create_task(c.post("/v1/chat/completions", json={
                    "model": "gpt-5.4",
                    "messages": [{"role": "user", "content": "x"}]}, timeout=10))
                for _ in range(50):
                    if (await c.get("/_control/pending")).json().get("has_pending"):
                        break
                    await asyncio.sleep(0.05)
                else:
                    self.fail("never became pending")
                await c.post("/_control/error", json={
                    "status": 429, "type": "rate_limit_error", "message": "slow down"})
                r = await t
                self.assertEqual(r.status_code, 429)
                e = r.json()["error"]
                self.assertEqual(e["type"], "rate_limit_error")
                self.assertEqual(e["message"], "slow down")
                self.assertIn("param", e)
        _run(run())

    def test_stats_no_cache_counting(self) -> None:
        """The OpenAI path does no cache observation → does not pollute the stats hit/miss."""
        async def run() -> None:
            async with await self._client() as c:
                await self._round_trip(c, {
                    "model": "gpt-5.4-mini",
                    "messages": [{"role": "user", "content": "y" * 400}],
                }, [{"type": "text", "text": "z" * 200}])
                s = (await c.get("/_control/stats")).json()
                self.assertEqual(s["completed_requests"], 1)
                self.assertIn("gpt-5.4-mini", s["by_model"])
                self.assertGreater(s["totals"]["total_usd"], 0.0)
                self.assertEqual(s["cache"]["hits"], 0)
                self.assertEqual(s["cache"]["misses"], 0)
        _run(run())


class TestOpenAISDKCompatibility(unittest.TestCase):
    """Run the real openai SDK over ASGITransport and verify the SDK can interpret the response.

    (Corresponds to the Anthropic SDK compatibility check in test_fake_server.TestAnthropicSDKCompatibility)
    """

    def setUp(self) -> None:
        self.mod = _import_fresh()

    def _make_clients(self) -> tuple:
        import httpx
        import openai
        transport = httpx.ASGITransport(app=self.mod.app)
        ctl = httpx.AsyncClient(transport=transport, base_url="http://test")
        sdk = openai.AsyncOpenAI(
            api_key="sk-mock", base_url="http://test/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=transport,
                                          base_url="http://test"))
        return ctl, sdk

    async def _inject_when_pending(self, ctl, endpoint: str, payload: dict) -> None:
        for _ in range(50):
            if (await ctl.get("/_control/pending")).json().get("has_pending"):
                break
            await asyncio.sleep(0.05)
        else:
            self.fail("never became pending")
        await ctl.post(endpoint, json=payload)

    def test_sdk_non_stream_with_tool_calls(self) -> None:
        async def run() -> None:
            ctl, sdk = self._make_clients()
            try:
                t = asyncio.create_task(sdk.chat.completions.create(
                    model="gpt-5.4", max_tokens=100,
                    messages=[{"role": "user", "content": "weather?"}],
                    tools=[{"type": "function", "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object",
                                       "properties": {"city": {"type": "string"}}},
                    }}]))
                await self._inject_when_pending(ctl, "/_control/respond", {
                    "content": [
                        {"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "call_x", "name": "get_weather",
                         "input": {"city": "Tokyo"}},
                    ]})
                completion = await t
                choice = completion.choices[0]
                self.assertEqual(choice.finish_reason, "tool_calls")
                self.assertEqual(choice.message.content, "checking")
                tc = choice.message.tool_calls[0]
                self.assertEqual(tc.function.name, "get_weather")
                self.assertEqual(json.loads(tc.function.arguments), {"city": "Tokyo"})
                self.assertGreater(completion.usage.prompt_tokens, 0)
            finally:
                await sdk.close()
                await ctl.aclose()
        _run(run())

    def test_sdk_streaming(self) -> None:
        async def run() -> None:
            ctl, sdk = self._make_clients()
            try:
                async def consume() -> str:
                    stream = await sdk.chat.completions.create(
                        model="gpt-5.4", max_tokens=100, stream=True,
                        messages=[{"role": "user", "content": "go"}])
                    parts = []
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            parts.append(chunk.choices[0].delta.content)
                    return "".join(parts)

                t = asyncio.create_task(consume())
                await self._inject_when_pending(ctl, "/_control/auto",
                                                {"text": "sdk streamed reply"})
                self.assertEqual(await t, "sdk streamed reply")
            finally:
                await sdk.close()
                await ctl.aclose()
        _run(run())

    def test_sdk_error_mapping(self) -> None:
        """429 injection → the SDK maps it to RateLimitError (max_retries=0)."""
        async def run() -> None:
            import openai
            ctl, sdk = self._make_clients()
            try:
                t = asyncio.create_task(sdk.chat.completions.create(
                    model="gpt-5.4", max_tokens=10,
                    messages=[{"role": "user", "content": "x"}]))
                await self._inject_when_pending(ctl, "/_control/error", {
                    "status": 429, "type": "rate_limit_error", "message": "throttled"})
                with self.assertRaises(openai.RateLimitError):
                    await t
            finally:
                await sdk.close()
                await ctl.aclose()
        _run(run())


# ── Regression tests for edge-case bugs / fidelity gaps ──────────────────────────────────


class TestRegressions(unittest.TestCase):
    """Regression tests pinning fixed bugs and API-fidelity edge cases."""

    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

    async def _make_pending(self, c, path: str, body: dict) -> Any:
        t = asyncio.create_task(c.post(path, json=body, timeout=10))
        for _ in range(50):
            if (await c.get("/_control/pending")).json().get("has_pending"):
                return t
            await asyncio.sleep(0.05)
        self.fail("never became pending")

    # B1: skipping an unknown block leaves a gap in the stream index and crashes the real SDK
    def test_stream_index_contiguous_with_unknown_blocks(self) -> None:
        events = self.mod.stream_event_dicts("m", "c", [
            {"type": "thinking", "thinking": "hidden"},
            {"type": "text", "text": "visible"},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
        ])
        starts = [d for n, d in events if n == "content_block_start"]
        self.assertEqual([d["index"] for d in starts], [0, 1])  # no gaps (thinking is skipped)

    # B1 continued: ping appears in the SSE (right after message_start, like the real API)
    def test_sse_stream_has_ping(self) -> None:
        raw = b"".join(self.mod._build_sse_stream("m", "c", [{"type": "text", "text": "x"}]))
        joined = raw.decode("utf-8")
        self.assertIn("event: ping", joined)
        self.assertLess(joined.index("message_start"), joined.index("event: ping"))

    # B2: an entry with a resolved future does not show up in the pending view (prevents ghost pending)
    def test_resolved_entry_hidden_from_pending_views(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                fut = asyncio.get_running_loop().create_future()
                fut.set_result({"content": []})
                self.mod.state.pending["ghost"] = {
                    "request": {"pending_id": "ghost", "received_at": 0},
                    "future": fut, "started_at": 0,
                }
                p = (await c.get("/_control/pending")).json()
                self.assertEqual(p["count"], 0)
                self.assertFalse(p["has_pending"])
                w = (await c.get("/_control/wait_for_pending?timeout=0.6")).json()
                self.assertFalse(w.get("has_pending"))  # a ghost does not fire immediately
        _run(run())

    # B2: even on client disconnect (task cancel), no pending is left behind
    def test_cancelled_request_cleans_pending(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/messages", {
                    "model": "x", "messages": [{"role": "user", "content": "y"}]})
                t.cancel()
                for _ in range(50):
                    if (await c.get("/_control/pending")).json()["count"] == 0:
                        break
                    await asyncio.sleep(0.05)
                else:
                    self.fail("cancelled request left a phantom pending")
        _run(run())

    # S1: respond validates content blocks (a broken injection is 400, the pending is intact)
    def test_respond_rejects_non_block_content(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/messages", {
                    "model": "x", "messages": [{"role": "user", "content": "y"}]})
                r = await c.post("/_control/respond", json={"content": ["oops", 42]})
                self.assertEqual(r.status_code, 400)
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 1)
                r2 = await c.post("/_control/respond", json={"content": [
                    {"type": "text", "text": "ok"}]})
                self.assertEqual(r2.status_code, 200)
                self.assertEqual((await t).status_code, 200)
        _run(run())

    # S2: malformed fields on the OpenAI path do not cause a 500
    def test_openai_malformed_fields_do_not_500(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # stream_options is a non-dict (previously crashed after committing history)
                t = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "stream": True, "stream_options": "yes",
                    "messages": [{"role": "user", "content": "x"}]})
                await c.post("/_control/auto", json={"text": "ok"})
                self.assertEqual((await t).status_code, 200)
                # function in tools / tool_calls is a non-dict
                t2 = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4",
                    "messages": [
                        {"role": "user", "content": "x"},
                        {"role": "assistant", "tool_calls": [{"id": "c1",
                         "type": "function", "function": "broken"}]},
                    ],
                    "tools": [{"type": "function", "function": "nope"}]})
                await c.post("/_control/auto", json={"text": "ok"})
                self.assertEqual((await t2).status_code, 200)
        _run(run())

    # stop_reason injection knob (for testing max_tokens branches, etc.)
    def test_stop_reason_override(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # Anthropic non-stream: verbatim
                t = await self._make_pending(c, "/v1/messages", {
                    "model": "x", "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={
                    "content": [{"type": "text", "text": "cut"}],
                    "stop_reason": "max_tokens"})
                self.assertEqual((await t).json()["stop_reason"], "max_tokens")
                # OpenAI: mapped to finish_reason (max_tokens → length)
                t2 = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={
                    "content": [{"type": "text", "text": "cut"}],
                    "stop_reason": "max_tokens"})
                self.assertEqual((await t2).json()["choices"][0]["finish_reason"], "length")
        _run(run())

    # auxiliary parameters (tool_choice, etc.) are retained in the snapshot
    def test_params_carried_in_snapshot(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/messages", {
                    "model": "x", "tool_choice": {"type": "tool", "name": "f"},
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": "y"}]})
                req = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(req["params"]["tool_choice"], {"type": "tool", "name": "f"})
                self.assertEqual(req["params"]["temperature"], 0.2)
                await c.post("/_control/auto", json={"text": "ok"})
                await t
                # the OpenAI path too (tool_choice / response_format)
                t2 = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "tool_choice": "required",
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": "y"}]})
                req2 = (await c.get("/_control/pending")).json()["pending"][0]["request"]
                self.assertEqual(req2["params"]["tool_choice"], "required")
                self.assertEqual(req2["params"]["response_format"], {"type": "json_object"})
                await c.post("/_control/auto", json={"text": "ok"})
                await t2
        _run(run())

    # OpenAI n>1 duplicates choices (for apps that read choices[i])
    def test_openai_n_choices(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "n": 3,
                    "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "same"})
                choices = (await t).json()["choices"]
                self.assertEqual([ch["index"] for ch in choices], [0, 1, 2])
                self.assertTrue(all(ch["message"]["content"] == "same" for ch in choices))
        _run(run())

    # /_control/error's code/param passes straight through to the OpenAI error format
    def test_error_code_param_passthrough(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/error", json={
                    "status": 429, "type": "rate_limit_error", "message": "slow",
                    "code": "rate_limit_exceeded"})
                e = (await t).json()["error"]
                self.assertEqual(e["code"], "rate_limit_exceeded")
        _run(run())

    # a missing tool_use id is assigned even in non-stream (unified with the stream behavior)
    def test_tool_use_id_assigned_non_stream(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/messages", {
                    "model": "x", "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "tool_use", "name": "f", "input": {}}]})
                block = (await t).json()["content"][0]
                self.assertTrue(str(block["id"]).startswith("toolu_"))
        _run(run())

    # consecutive role:tool are merged into a single user turn (the real Anthropic shape for parallel tool results)
    def test_consecutive_tool_messages_merged(self) -> None:
        from puppetllm.providers import openai as oai
        out = oai.normalize_chat_body({
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "tool_calls": [
                    {"id": "a", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}},
                    {"id": "b", "type": "function",
                     "function": {"name": "g", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": "a", "content": "ra"},
                {"role": "tool", "tool_call_id": "b", "content": "rb"},
            ]})
        self.assertEqual(len(out["messages"]), 3)  # user / assistant / merged tool_results
        results = out["messages"][2]["content"]
        self.assertEqual([r["tool_use_id"] for r in results], ["a", "b"])

    # pricing: codex is cross-generation / o3-mini / does not crash on a non-str model
    def test_pricing_family_edge_cases(self) -> None:
        self.assertEqual(pricing.resolve_family("gpt-5.4-codex"), "codex")
        self.assertEqual(pricing.resolve_family("gpt-5.3-codex"), "codex")
        self.assertEqual(pricing.resolve_family("o3-mini"), "o3-mini")
        self.assertEqual(pricing.resolve_family({"weird": True}), "sonnet")  # works even for non-str

    # request-id / token-count headers (attached like the real API)
    def test_response_headers(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/messages", {
                    "model": "x", "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "ok"})
                r = await t
                self.assertTrue(r.headers.get("request-id", "").startswith("req_"))
                t2 = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "ok"})
                self.assertIn("x-request-id", (await t2).headers)
                mid = "anthropic.claude-3-5-sonnet-20241022-v2:0"
                t3 = await self._make_pending(c, f"/model/{mid}/invoke", {
                    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "ok"})
                r3 = await t3
                self.assertIn("x-amzn-requestid", r3.headers)
                self.assertIn("X-Amzn-Bedrock-Output-Token-Count", r3.headers)
        _run(run())

    # invocationMetrics rides on the final chunk of a Bedrock stream
    def test_bedrock_invocation_metrics(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                mid = "anthropic.claude-3-5-sonnet-20241022-v2:0"
                t = await self._make_pending(
                    c, f"/model/{mid}/invoke-with-response-stream", {
                        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                        "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "ok"})
                events = eventstream.decode_messages((await t).content)
                stop = [e for e in events if e.get("type") == "message_stop"][0]
                metrics = stop["amazon-bedrock-invocationMetrics"]
                self.assertGreaterEqual(metrics["outputTokenCount"], 1)
        _run(run())

    # with include_usage, every chunk carries a usage key (null) (matches the real API spec)
    def test_openai_stream_usage_null_on_chunks(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/auto", json={"text": "ok"})
                lines = [ln[len("data: "):] for ln in (await t).text.splitlines()
                         if ln.startswith("data: ") and not ln.endswith("[DONE]")]
                chunks = [json.loads(ln) for ln in lines]
                for ch in chunks:
                    self.assertIn("usage", ch)
                self.assertTrue(all(ch["usage"] is None for ch in chunks if ch["choices"]))
        _run(run())

    # #1: unknown block types are filtered ONCE (caller content, usage, AND history agree)
    def test_unknown_blocks_filtered_consistently(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # baseline: text-only injection → output_tokens for the text alone
                t0 = await self._make_pending(c, "/v1/messages", {
                    "model": "claude-x", "messages": [{"role": "user", "content": "a"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "text", "text": "visible"}]})
                base_out = (await t0).json()["usage"]["output_tokens"]

                # thinking + text: thinking must be dropped from content, usage, and history
                t = await self._make_pending(c, "/v1/messages", {
                    "model": "claude-x", "messages": [{"role": "user", "content": "b"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "thinking", "thinking": "X" * 400},
                    {"type": "text", "text": "visible"}]})
                j = (await t).json()
                self.assertEqual([b["type"] for b in j["content"]], ["text"])
                # usage must NOT count the dropped thinking block (was the real regression)
                self.assertEqual(j["usage"]["output_tokens"], base_out)
                # history must record the same filtered blocks the caller received
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertEqual([b["type"] for b in h["response_blocks"]], ["text"])
        _run(run())

    # #2b: an OpenAI call_ id we minted (toolu_→call_) round-trips through normalization
    def test_openai_call_id_round_trips(self) -> None:
        from puppetllm.providers import openai as oai
        out = oai.normalize_chat_body({"messages": [
            {"role": "assistant", "tool_calls": [{"id": "call_abc", "type": "function",
                                                  "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_abc", "content": "r"},
        ]})
        self.assertEqual(out["messages"][0]["content"][0]["id"], "call_abc")
        self.assertEqual(out["messages"][1]["content"][0]["tool_use_id"], "call_abc")

    # #2: OpenAI tool_calls id uses the call_ prefix, not the canonical toolu_
    def test_openai_tool_call_id_prefix(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # id omitted → core assigns toolu_… → OpenAI encoder must remap to call_…
                t = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "tool_use", "name": "f", "input": {}}]})
                ns_id = (await t).json()["choices"][0]["message"]["tool_calls"][0]["id"]
                self.assertTrue(ns_id.startswith("call_"), ns_id)
                self.assertFalse(ns_id.startswith("toolu_"))
                # streaming path must produce the same call_ id shape
                t2 = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "stream": True,
                    "messages": [{"role": "user", "content": "y"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "tool_use", "name": "f", "input": {"a": 1}}]})
                ids = re.findall(r'"tool_calls":\s*\[\{[^}]*"id":\s*"([^"]+)"', (await t2).text)
                self.assertTrue(ids and all(i.startswith("call_") for i in ids), ids)
        _run(run())

    # #4: OpenAI STREAMING tool_calls delta shape (id+name delta → arguments delta → finish)
    def test_openai_stream_tool_calls_shape(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/chat/completions", {
                    "model": "gpt-5.4", "stream": True,
                    "messages": [{"role": "user", "content": "weather?"}]})
                await c.post("/_control/respond", json={"content": [
                    {"type": "tool_use", "id": "call_z", "name": "get_weather",
                     "input": {"city": "Tokyo"}}]})
                lines = [ln[len("data: "):] for ln in (await t).text.splitlines()
                         if ln.startswith("data: ") and not ln.endswith("[DONE]")]
                chunks = [json.loads(ln) for ln in lines]
                tc_deltas = [ch["choices"][0]["delta"]["tool_calls"][0]
                             for ch in chunks if ch["choices"]
                             and ch["choices"][0]["delta"].get("tool_calls")]
                # first tool-call delta carries id + name (+ empty args); a later one carries args
                self.assertEqual(tc_deltas[0]["id"], "call_z")
                self.assertEqual(tc_deltas[0]["function"]["name"], "get_weather")
                joined_args = "".join(d["function"].get("arguments", "") for d in tc_deltas)
                self.assertEqual(json.loads(joined_args), {"city": "Tokyo"})
                # subsequent tool-call deltas omit id/type/name (only index + args)
                self.assertNotIn("id", tc_deltas[-1])
                finishes = [ch["choices"][0]["finish_reason"] for ch in chunks if ch["choices"]]
                self.assertIn("tool_calls", finishes)
        _run(run())

    # #5: the _safe_set_result "already resolved" → 409 branch.
    # NOTE: this branch is defensive/near-unreachable via HTTP because
    # _resolve_target_future already filters done futures (a resolved pending yields 400,
    # not 409, and there is no await between resolve and set within one respond call), so
    # we exercise the helper directly to pin the 409 contract.
    def test_safe_set_result_already_resolved_409(self) -> None:
        async def run() -> None:
            fut = asyncio.get_running_loop().create_future()
            fut.set_result({"content": []})
            resp = self.mod._safe_set_result(fut, {"content": []})
            self.assertIsNotNone(resp)
            self.assertEqual(resp.status_code, 409)
        _run(run())

    # companion: over HTTP, injecting into an already-resolved pending returns 400 (the
    # documented behavior), because the done future is filtered before _safe_set_result.
    def test_respond_into_resolved_pending_400(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, "/v1/messages", {
                    "model": "x", "messages": [{"role": "user", "content": "y"}]})
                pid = (await c.get("/_control/pending")).json()["pending"][0]["pending_id"]
                self.mod.state.pending[pid]["future"].set_result({"content": [
                    {"type": "text", "text": "won"}]})
                r = await c.post("/_control/respond", json={
                    "pending_id": pid, "content": [{"type": "text", "text": "late"}]})
                self.assertEqual(r.status_code, 400)
                await t  # caller still completes with the first result
        _run(run())


# ── /_control/respond usage override ─────────────────────────────────


class TestUsageOverride(unittest.TestCase):
    """Optional real-usage injection via /_control/respond (for relay responders)."""

    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app), base_url="http://test")

    async def _make_pending(self, c, body: dict) -> Any:
        t = asyncio.create_task(c.post("/v1/messages", json=body, timeout=10))
        for _ in range(50):
            if (await c.get("/_control/pending")).json().get("has_pending"):
                return t
            await asyncio.sleep(0.05)
        self.fail("never became pending")

    def test_full_override_reflected_everywhere(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, {
                    "model": "claude-opus-4-1",
                    "messages": [{"role": "user", "content": "x"}]})
                await c.post("/_control/respond", json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1000, "output_tokens": 250,
                              "cache_read_input_tokens": 500,
                              "cache_creation_input_tokens": 0}})
                u = (await t).json()["usage"]
                self.assertEqual(u["input_tokens"], 1000)
                self.assertEqual(u["output_tokens"], 250)
                self.assertEqual(u["cache_read_input_tokens"], 500)
                h = (await c.get("/_control/history")).json()["history"][-1]
                self.assertTrue(h.get("usage_overridden"))
                # cost is recomputed from the overridden numbers (opus: in $5 + out $25 + read $0.5 /Mtok)
                expected = (1000 * 5.0 + 250 * 25.0 + 500 * 0.50) / 1_000_000
                self.assertAlmostEqual(h["cost"]["total_usd"], expected, places=9)
                s = (await c.get("/_control/stats")).json()
                self.assertEqual(s["totals"]["input_tokens"], 1000)
                self.assertEqual(s["totals"]["output_tokens"], 250)
        _run(run())

    def test_partial_override_keeps_approx_for_rest(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, {
                    "model": "m", "messages": [{"role": "user", "content": "y" * 400}]})
                await c.post("/_control/respond", json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"output_tokens": 7}})
                u = (await t).json()["usage"]
                self.assertEqual(u["output_tokens"], 7)      # overridden
                self.assertGreater(u["input_tokens"], 0)      # approx kept
        _run(run())

    def test_invalid_usage_rejected_pending_intact(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                t = await self._make_pending(c, {
                    "model": "m", "messages": [{"role": "user", "content": "x"}]})
                for bad in ({"output_tokens": "7"},          # non-int
                            {"output_tokens": -1},            # negative
                            {"output_tokens": True},          # bool is not accepted
                            {"unknown_key": 1},               # unknown key
                            {},                               # empty
                            "usage-as-string"):
                    r = await c.post("/_control/respond", json={
                        "content": [{"type": "text", "text": "ok"}], "usage": bad})
                    self.assertEqual(r.status_code, 400, f"usage={bad!r}")
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 1)
                await c.post("/_control/auto", json={"text": "done"})
                self.assertEqual((await t).status_code, 200)
        _run(run())


# ── relay: pure translation units ────────────────────────────────────


class TestRelayUnit(unittest.TestCase):
    def test_model_map(self) -> None:
        from puppetllm import relay
        mm = relay.parse_model_map("claude-*=grok-3, gpt-*=grok-3-mini")
        self.assertEqual(relay.map_model("claude-sonnet-4-5", None, mm), "grok-3")
        self.assertEqual(relay.map_model("gpt-5.4", None, mm), "grok-3-mini")
        self.assertEqual(relay.map_model("other", None, mm), "other")       # passthrough
        self.assertEqual(relay.map_model("claude-x", "forced", mm), "forced")  # --model wins
        with self.assertRaises(ValueError):
            relay.parse_model_map("no-equals-sign")

    def test_to_openai_request(self) -> None:
        from puppetllm import relay
        req = {
            "system": "be terse",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "tu1", "name": "wx",
                     "input": {"c": "Tokyo"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "sunny"},
                    {"type": "text", "text": "and paris?"}]},
            ],
            "tools": [{"name": "wx", "description": "d",
                       "input_schema": {"type": "object"}}],
            "max_tokens": 128,
            "params": {"temperature": 0.3, "stop_sequences": ["END"],
                       "tool_choice": {"type": "any"}},
        }
        body = relay.to_openai_request(req, "grok-3")
        self.assertEqual(body["model"], "grok-3")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "be terse"})
        asst = body["messages"][2]
        self.assertEqual(asst["content"], "checking")
        self.assertEqual(asst["tool_calls"][0]["function"]["name"], "wx")
        self.assertEqual(json.loads(asst["tool_calls"][0]["function"]["arguments"]),
                         {"c": "Tokyo"})
        # tool_result becomes a tool message (before the trailing user text)
        self.assertEqual(body["messages"][3],
                         {"role": "tool", "tool_call_id": "tu1", "content": "sunny"})
        self.assertEqual(body["messages"][4], {"role": "user", "content": "and paris?"})
        self.assertEqual(body["tools"][0]["function"]["name"], "wx")
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["stop"], ["END"])
        self.assertEqual(body["tool_choice"], "required")   # anthropic "any" -> openai
        self.assertEqual(body["max_tokens"], 128)

    def test_tool_choice_mappings(self) -> None:
        from puppetllm.relay import _tool_choice_to_openai as f
        self.assertEqual(f({"type": "auto"}), "auto")
        self.assertEqual(f({"type": "tool", "name": "wx"}),
                         {"type": "function", "function": {"name": "wx"}})
        self.assertEqual(f("required"), "required")          # openai-style passthrough
        self.assertIsNone(f(None))

    def test_from_openai_response(self) -> None:
        from puppetllm import relay
        payload = relay.from_openai_response({
            "choices": [{"index": 0, "finish_reason": "length", "message": {
                "role": "assistant", "content": "partial",
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "wx",
                                             "arguments": "{\"c\": \"Osaka\"}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 9,
                      "prompt_tokens_details": {"cached_tokens": 40}},
        })
        types = [b["type"] for b in payload["content"]]
        self.assertEqual(types, ["text", "tool_use"])
        self.assertEqual(payload["content"][1]["input"], {"c": "Osaka"})
        # tool_use present → tool_use wins over finish_reason "length" (see
        # test_tool_calls_force_tool_use_stop_reason). The plain length -> max_tokens
        # mapping (no tool calls) is asserted just below.
        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["usage"], {
            "input_tokens": 60, "output_tokens": 9,             # 100 - 40 cached
            "cache_read_input_tokens": 40, "cache_creation_input_tokens": 0})
        # length -> max_tokens when there are no tool calls
        self.assertEqual(relay.from_openai_response(
            {"choices": [{"finish_reason": "length",
                          "message": {"content": "partial"}}]})["stop_reason"], "max_tokens")

    def test_from_openai_response_list_content_joined(self) -> None:
        # Some OpenAI-compatible backends return content as a list of parts; join the text
        # instead of str()-ing the whole list into the app's message.
        from puppetllm import relay
        p = relay.from_openai_response({"choices": [{"finish_reason": "stop", "message": {
            "content": [{"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"}]}}]})
        self.assertEqual(p["content"], [{"type": "text", "text": "Hello world"}])
        # a list with no text parts yields no text block (not a stringified list)
        p2 = relay.from_openai_response({"choices": [{"finish_reason": "stop", "message": {
            "content": [{"type": "image_url", "image_url": {"url": "x"}}]}}]})
        self.assertEqual(p2["content"], [])
        # plain string content still works
        p3 = relay.from_openai_response({"choices": [{"finish_reason": "stop",
                                        "message": {"content": "plain"}}]})
        self.assertEqual(p3["content"], [{"type": "text", "text": "plain"}])

    def test_from_response_omits_usage_when_upstream_has_none(self) -> None:
        # Upstream without usage → usage=None so the relay keeps puppetllm's approx
        # (rather than forcing zeros). Applies to both directions.
        from puppetllm import relay
        no_u = {"choices": [{"finish_reason": "stop",
                             "message": {"content": "hi"}}]}
        self.assertIsNone(relay.from_openai_response(no_u)["usage"])
        self.assertIsNone(relay.from_openai_response(
            {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}},],
             "usage": {}})["usage"])
        self.assertIsNone(relay.from_anthropic_response(
            {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"})["usage"])
        # present usage still overrides
        self.assertIsNotNone(relay.from_openai_response(
            {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2}})["usage"])

    def test_to_anthropic_request_defaults(self) -> None:
        from puppetllm import relay
        body = relay.to_anthropic_request(
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": None,
             "params": {"top_k": 5}}, "claude-sonnet-4-5")
        self.assertEqual(body["max_tokens"], 4096)   # required by /v1/messages
        self.assertEqual(body["top_k"], 5)
        self.assertNotIn("system", body)

    def test_to_anthropic_request_translates_openai_params(self) -> None:
        from puppetllm import relay
        body = relay.to_anthropic_request({
            "system": "sys", "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 64,
            "tools": [{"name": "wx", "input_schema": {"type": "object"}}],
            "params": {"tool_choice": "required", "stop": "END",
                       "thinking": {"type": "enabled"}}},
            "claude-sonnet-4-5")
        self.assertEqual(body["system"], "sys")
        self.assertEqual(body["tools"][0]["name"], "wx")
        self.assertEqual(body["tool_choice"], {"type": "any"})       # required -> any
        self.assertEqual(body["stop_sequences"], ["END"])            # str wrapped in list
        self.assertNotIn("thinking", body)                          # dropped (server strips)

    def test_tool_choice_to_anthropic(self) -> None:
        from puppetllm.relay import _tool_choice_to_anthropic as f
        self.assertEqual(f("required"), {"type": "any"})
        self.assertEqual(f("auto"), {"type": "auto"})
        self.assertEqual(f({"type": "function", "function": {"name": "wx"}}),
                         {"type": "tool", "name": "wx"})
        self.assertEqual(f({"type": "tool", "name": "wx"}),          # already anthropic
                         {"type": "tool", "name": "wx"})
        self.assertIsNone(f(None))

    def test_from_anthropic_response_usage(self) -> None:
        from puppetllm import relay
        p = relay.from_anthropic_response({
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 5,
                      "cache_read_input_tokens": 10, "cache_creation_input_tokens": 2}})
        self.assertEqual(p["stop_reason"], "tool_use")
        self.assertEqual(p["usage"], {"input_tokens": 30, "output_tokens": 5,
                                      "cache_read_input_tokens": 10,
                                      "cache_creation_input_tokens": 2})

    def test_usage_clamped_non_negative(self) -> None:
        # A hostile/buggy upstream reporting negative token counts must be clamped (else the
        # server rejects the inject and the request hot-loops / hangs).
        from puppetllm import relay
        po = relay.from_openai_response({
            "choices": [{"finish_reason": "stop", "message": {"content": "x"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": -3,
                      "prompt_tokens_details": {"cached_tokens": 999}}})
        self.assertTrue(all(v >= 0 for v in po["usage"].values()))
        pa = relay.from_anthropic_response({
            "content": [], "stop_reason": "end_turn",
            "usage": {"input_tokens": -1, "output_tokens": 4}})
        self.assertTrue(all(v >= 0 for v in pa["usage"].values()))

    def test_upstream_error_fields(self) -> None:
        from puppetllm.relay import _upstream_error_fields as f
        # OpenAI envelope, with code/param
        e = f(429, {"error": {"message": "slow", "type": "rate_limit_error",
                              "code": "rate_limit_exceeded", "param": None}})
        self.assertEqual((e["status"], e["type"], e["code"]),
                         (429, "rate_limit_error", "rate_limit_exceeded"))
        # non-standard status clamped into [400,599]
        self.assertEqual(f(999, {"error": {"message": "x", "type": "y"}})["status"], 599)
        self.assertEqual(f(200, {"error": {"message": "x", "type": "y"}})["status"], 400)
        # non-envelope body → api_error with stringified message
        e2 = f(503, "plain text error")
        self.assertEqual((e2["status"], e2["type"]), (503, "api_error"))
        self.assertIn("plain text", e2["message"])

    def test_max_tokens_param(self) -> None:
        from puppetllm import relay
        import argparse
        cfg = argparse.Namespace(max_tokens_param="max_completion_tokens")
        body = relay.to_openai_request(
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": 100}, "m", cfg)
        self.assertEqual(body.get("max_completion_tokens"), 100)
        self.assertNotIn("max_tokens", body)

    def test_tool_calls_force_tool_use_stop_reason(self) -> None:
        # A tool-calling turn must report stop_reason "tool_use" even when the upstream
        # (mis)sets finish_reason to "stop"/"length" — else an Anthropic-SDK agent's tool
        # loop stalls. Presence of a tool_use block wins over finish_reason.
        from puppetllm import relay
        for finish in ("stop", "length", None, "tool_calls", "weird"):
            resp = {"choices": [{"finish_reason": finish, "message": {
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "wx", "arguments": "{}"}}]}}]}
            p = relay.from_openai_response(resp)
            self.assertEqual(p["stop_reason"], "tool_use", f"finish={finish!r}")
            self.assertEqual([b["type"] for b in p["content"]], ["tool_use"])
        # no tool calls → finish_reason still governs
        self.assertEqual(relay.from_openai_response(
            {"choices": [{"finish_reason": "stop",
                          "message": {"content": "hi"}}]})["stop_reason"], "end_turn")

    def test_malformed_prompt_tokens_details_does_not_raise(self) -> None:
        # A non-dict prompt_tokens_details (garbage upstream) must not turn a billed 200 into
        # a 502; it's treated as "no cached tokens" and the rest of usage still lands.
        from puppetllm import relay
        p = relay.from_openai_response({"choices": [
            {"finish_reason": "stop", "message": {"content": "x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                      "prompt_tokens_details": [1, 2, 3]}})   # list, not dict
        self.assertEqual(p["usage"], {"input_tokens": 10, "output_tokens": 4,
                                      "cache_read_input_tokens": 0,
                                      "cache_creation_input_tokens": 0})

    def test_unknown_finish_reason_falls_back(self) -> None:
        # A non-standard finish_reason must not reach the app as a bogus stop_reason.
        from puppetllm import relay
        p = relay.from_openai_response({"choices": [
            {"finish_reason": "weird_new_value", "message": {"content": "x"}}]})
        self.assertEqual(p["stop_reason"], "end_turn")
        # legacy single-function form maps to tool_use
        p2 = relay.from_openai_response({"choices": [
            {"finish_reason": "function_call", "message": {"content": "x"}}]})
        self.assertEqual(p2["stop_reason"], "tool_use")

    def test_empty_assistant_turn_dropped(self) -> None:
        # An assistant turn that carried only thinking (stripped by the server) becomes empty;
        # it must be dropped rather than emitted as {"content": None} with no tool_calls.
        from puppetllm import relay
        body = relay.to_openai_request({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": []},          # empty → dropped
            {"role": "user", "content": "again"},
        ]}, "m")
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["user", "user"])          # no empty assistant

    def test_anthropic_metadata_sanitized(self) -> None:
        # OpenAI-inbound apps may attach arbitrary metadata; only user_id survives to Anthropic.
        from puppetllm import relay
        keep = relay.to_anthropic_request({"messages": [], "params": {
            "metadata": {"user_id": "u1", "session": "drop-me"}}}, "m")
        self.assertEqual(keep["metadata"], {"user_id": "u1"})
        drop = relay.to_anthropic_request({"messages": [], "params": {
            "metadata": {"session": "no-user-id"}}}, "m")
        self.assertNotIn("metadata", drop)

    def test_looks_foreign(self) -> None:
        from puppetllm import relay
        self.assertTrue(relay.looks_foreign("claude-sonnet-4-5", "openai"))
        self.assertTrue(relay.looks_foreign("gpt-5.4", "anthropic"))
        self.assertTrue(relay.looks_foreign("grok-3", "anthropic"))
        self.assertFalse(relay.looks_foreign("grok-3", "openai"))       # native to openai kind
        self.assertFalse(relay.looks_foreign("claude-x", "anthropic"))  # native to anthropic kind
        self.assertFalse(relay.looks_foreign("", "openai"))

    def test_claim_filter(self) -> None:
        # --only claims matching inbound models and leaves the rest for another responder.
        import argparse
        from puppetllm import relay
        cfg = argparse.Namespace(
            model_map=None, only="gpt-*,o3-*", max_concurrency=0,
            api_key_env="RELAY_TEST_KEY_UNSET", kind="openai", target="http://x.invalid",
            timeout=1.0, poll_timeout=1.0, puppet="http://p.invalid", model=None,
            max_tokens_param="max_tokens", max_requests=0)
        r = relay.Relay(cfg)
        try:
            self.assertTrue(r._claims({"request": {"model": "gpt-5.4"}}))
            self.assertTrue(r._claims({"request": {"model": "o3-mini"}}))
            self.assertFalse(r._claims({"request": {"model": "claude-sonnet-4-5"}}))
        finally:
            _run(r.http.aclose())
            _run(r.ctl.aclose())


class TestRelayInjectResilience(unittest.TestCase):
    """The inject path must settle a pending WITHOUT ever re-forwarding to the paid upstream,
    even when the control plane is unreachable (transport error) rather than merely rejecting
    the payload. Regression for the transport-error hot-loop."""

    def _relay(self):
        import argparse
        from puppetllm import relay
        cfg = argparse.Namespace(
            model_map=None, api_key_env="RELAY_TEST_KEY_UNSET", kind="openai",
            target="http://upstream.invalid", timeout=1.0, poll_timeout=1.0,
            puppet="http://puppet.invalid", model=None,
            max_tokens_param="max_tokens", max_requests=0)
        return relay.Relay(cfg)

    def test_transport_failure_quarantines_and_does_not_reforward(self) -> None:
        import httpx

        class _AlwaysFailCtl:
            def __init__(self) -> None:
                self.calls: list = []

            async def post(self, path, json=None):
                self.calls.append((path, json))
                raise httpx.ConnectError("control plane down")

            async def aclose(self) -> None:
                pass

        async def run() -> None:
            r = self._relay()
            r.ctl = _AlwaysFailCtl()
            try:
                # An undeliverable resolution must still "settle" (return True) so handle()
                # doesn't leave the pending re-claimable → no re-forward to the upstream.
                ok = await r._inject("pid1", "/_control/respond",
                                     {"content": [{"type": "text", "text": "x"}]})
                self.assertTrue(ok)
                self.assertIn("pid1", r.quarantined)   # run loop can never re-claim it
                paths = [c[0] for c in r.ctl.calls]
                # the respond POST was retried (bounded), then a 502 fallback was attempted
                self.assertEqual(paths.count("/_control/respond"), 3)
                self.assertIn("/_control/error", paths)
            finally:
                await r.http.aclose()

        _run(run())

    def test_post_ctl_returns_response_on_first_success(self) -> None:
        class _OkCtl:
            def __init__(self) -> None:
                self.n = 0

            async def post(self, path, json=None):
                self.n += 1
                return httpx.Response(200, json={"ok": True})

            async def aclose(self) -> None:
                pass

        import httpx

        async def run() -> None:
            r = self._relay()
            r.ctl = _OkCtl()
            try:
                resp = await r._post_ctl("/_control/respond", {"pending_id": "p"})
                self.assertIsNotNone(resp)
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(r.ctl.n, 1)           # no needless retries on success
            finally:
                await r.http.aclose()

        _run(run())


# ── relay: end-to-end (real servers + real anthropic SDK + relay loop) ──


class _MockUpstream:
    """Minimal OpenAI-compatible ASGI app: scripted (status, json) responses + capture."""

    def __init__(self) -> None:
        self.queue: list = []
        self.seen: list = []
        self.seen_headers: list = []
        self.calls = 0

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        self.calls += 1
        self.seen_headers.append({k.decode(): v.decode()
                                  for k, v in scope.get("headers", [])})
        if scope.get("path", "").endswith("/_health"):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
            return
        try:
            self.seen.append(json.loads(body or b"{}"))
        except ValueError:
            self.seen.append(None)
        status, payload = (self.queue.pop(0) if self.queue
                           else (500, {"error": {"message": "mock queue empty",
                                                 "type": "test_error"}}))
        # payload may be a bytes to simulate a non-JSON body
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": data})


class TestRelayE2E(unittest.TestCase):
    """Front puppetllm (uvicorn) + mock OpenAI-compatible upstream + real relay loop.

    Proves the headline: an app speaking the anthropic SDK transparently runs against
    an OpenAI-compatible backend, with real usage/stop_reason relayed back.
    """

    @staticmethod
    def _free_port() -> int:
        """Bind :0 to let the OS pick a free port, then release it. Avoids fixed-port
        collisions with residual processes / parallel CI runs (small TOCTOU window)."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    @classmethod
    def setUpClass(cls) -> None:
        import uvicorn
        cls.FRONT = cls._free_port()
        cls.UP = cls._free_port()
        cls.mod = _import_fresh()
        cls.upstream = _MockUpstream()
        cls._servers = []
        cls._threads = []
        for app, port in ((cls.mod.app, cls.FRONT), (cls.upstream, cls.UP)):
            server = uvicorn.Server(uvicorn.Config(
                app, host="127.0.0.1", port=port, log_level="critical",
                loop="asyncio"))
            th = threading.Thread(
                target=lambda s=server: asyncio.new_event_loop().run_until_complete(
                    s.serve()),
                daemon=True)
            th.start()
            cls._servers.append(server)
            cls._threads.append(th)
        import urllib.request
        # Health-check BOTH servers (not just the front) so a silent bind failure on the
        # upstream port fails fast here instead of surfacing as a misleading relay 502.
        for url in (f"http://127.0.0.1:{cls.FRONT}/_control/health",
                    f"http://127.0.0.1:{cls.UP}/_health"):
            for _ in range(80):
                try:
                    urllib.request.urlopen(url, timeout=0.5)
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                raise RuntimeError(f"server did not start: {url}")

    @classmethod
    def tearDownClass(cls) -> None:
        for s in cls._servers:
            s.should_exit = True
        for th in cls._threads:
            th.join(timeout=3)

    def _run_relay(self, max_requests: int, kind: str = "openai",
                   target: str | None = None, extra: list | None = None) -> threading.Thread:
        from puppetllm import relay
        os.environ["RELAY_TEST_KEY"] = "k"
        target = target or (f"http://127.0.0.1:{self.UP}/v1" if kind == "openai"
                            else f"http://127.0.0.1:{self.UP}")
        args = ["--puppet", f"http://127.0.0.1:{self.FRONT}",
                "--kind", kind, "--target", target,
                "--api-key-env", "RELAY_TEST_KEY", "--model", "mock-1",
                "--poll-timeout", "3", "--timeout", "10",
                "--max-requests", str(max_requests)] + (extra or [])
        th = threading.Thread(target=relay.main, args=(args,), daemon=True)
        th.start()
        return th

    def test_anthropic_app_bridged_to_openai_upstream(self) -> None:
        import anthropic
        self.upstream.queue[:] = [
            (200, {"id": "cc1", "object": "chat.completion", "created": 1,
                   "model": "mock-1",
                   "choices": [{"index": 0, "finish_reason": "tool_calls",
                                "message": {"role": "assistant", "content": None,
                                            "tool_calls": [{
                                                "id": "call_up1", "type": "function",
                                                "function": {"name": "wx",
                                                             "arguments": "{\"c\": \"Tokyo\"}"}}]}}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 9,
                             "prompt_tokens_details": {"cached_tokens": 40}}}),
            (200, {"id": "cc2", "object": "chat.completion", "created": 2,
                   "model": "mock-1",
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant",
                                            "content": "Sunny in Tokyo."}}],
                   "usage": {"prompt_tokens": 120, "completion_tokens": 7}}),
        ]
        self.upstream.seen.clear()
        th = self._run_relay(max_requests=2)

        client = anthropic.Anthropic(
            api_key="sk-mock", base_url=f"http://127.0.0.1:{self.FRONT}",
            max_retries=0)
        tools = [{"name": "wx", "description": "weather",
                  "input_schema": {"type": "object",
                                   "properties": {"c": {"type": "string"}}}}]
        m1 = client.messages.create(model="claude-sonnet-4-5", max_tokens=64,
                                    system="be terse",
                                    messages=[{"role": "user", "content": "weather?"}],
                                    tools=tools)
        # upstream tool_calls came back as a canonical tool_use with REAL usage
        self.assertEqual(m1.stop_reason, "tool_use")
        tb = [b for b in m1.content if b.type == "tool_use"][0]
        self.assertEqual((tb.name, tb.input), ("wx", {"c": "Tokyo"}))
        self.assertEqual(m1.usage.input_tokens, 60)          # 100 - 40 cached
        self.assertEqual(m1.usage.cache_read_input_tokens, 40)
        self.assertEqual(m1.usage.output_tokens, 9)

        m2 = client.messages.create(model="claude-sonnet-4-5", max_tokens=64,
                                    system="be terse",
                                    messages=[
                                        {"role": "user", "content": "weather?"},
                                        {"role": "assistant",
                                         "content": [b.model_dump() for b in m1.content]},
                                        {"role": "user", "content": [
                                            {"type": "tool_result",
                                             "tool_use_id": tb.id,
                                             "content": "sunny, 30C"}]},
                                    ], tools=tools)
        self.assertEqual(m2.stop_reason, "end_turn")
        self.assertEqual(m2.content[0].text, "Sunny in Tokyo.")
        self.assertEqual(m2.usage.input_tokens, 120)
        th.join(timeout=10)
        self.assertFalse(th.is_alive(), "relay did not exit after max-requests")

        # the upstream saw well-formed OpenAI requests
        first, second = self.upstream.seen[0], self.upstream.seen[1]
        self.assertEqual(first["model"], "mock-1")            # model mapping applied
        self.assertEqual(first["messages"][0],
                         {"role": "system", "content": "be terse"})
        self.assertEqual(first["tools"][0]["function"]["name"], "wx")
        roles2 = [m["role"] for m in second["messages"]]
        self.assertIn("tool", roles2)                          # tool_result -> tool msg
        toolmsg = [m for m in second["messages"] if m["role"] == "tool"][0]
        self.assertEqual(toolmsg["tool_call_id"], tb.id)
        self.assertEqual(toolmsg["content"], "sunny, 30C")

        # history marks the relayed usage as real
        import urllib.request
        h = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{self.FRONT}/_control/history", timeout=5).read())
        self.assertTrue(all(e.get("usage_overridden") for e in h["history"][-2:]))

        # stats prices by the INBOUND model id (documented caveat): the aggregation is keyed
        # by "claude-sonnet-4-5", not the upstream "mock-1" that actually served the tokens.
        s = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{self.FRONT}/_control/stats", timeout=5).read())
        self.assertIn("claude-sonnet-4-5", s["by_model"])
        self.assertNotIn("mock-1", s["by_model"])
        self.assertGreater(s["totals"]["total_usd"], 0.0)

    def test_upstream_error_relayed_to_sdk(self) -> None:
        import anthropic
        self.upstream.queue[:] = [
            (429, {"error": {"message": "rate limited", "type": "rate_limit_error",
                             "code": "rate_limit_exceeded"}})]
        th = self._run_relay(max_requests=1)
        client = anthropic.Anthropic(
            api_key="sk-mock", base_url=f"http://127.0.0.1:{self.FRONT}",
            max_retries=0)
        with self.assertRaises(anthropic.RateLimitError) as ctx:
            client.messages.create(model="claude-sonnet-4-5", max_tokens=16,
                                   messages=[{"role": "user", "content": "x"}])
        self.assertIn("rate limited", str(ctx.exception))
        th.join(timeout=10)

    def test_malformed_upstream_200_does_not_hang(self) -> None:
        # A 200 with a valid-JSON but unexpected shape (no choices) must not hang the
        # app's request; the relay surfaces it as a 502 instead of a silent empty message.
        import anthropic
        self.upstream.queue[:] = [(200, {"unexpected": "shape", "no": "choices"})]
        th = self._run_relay(max_requests=1)
        client = anthropic.Anthropic(
            api_key="sk-mock", base_url=f"http://127.0.0.1:{self.FRONT}",
            max_retries=0, timeout=8)
        with self.assertRaises(anthropic.APIStatusError) as ctx:
            client.messages.create(model="claude-sonnet-4-5", max_tokens=16,
                                   messages=[{"role": "user", "content": "x"}])
        self.assertEqual(ctx.exception.status_code, 502)
        th.join(timeout=10)

    def test_non_json_2xx_upstream_does_not_hang(self) -> None:
        # 200 with a non-JSON body → 502, not a hang.
        import anthropic
        self.upstream.queue[:] = [(200, b"<html>not json</html>")]
        th = self._run_relay(max_requests=1)
        client = anthropic.Anthropic(api_key="sk-mock",
                                     base_url=f"http://127.0.0.1:{self.FRONT}",
                                     max_retries=0, timeout=8)
        with self.assertRaises(anthropic.APIStatusError) as ctx:
            client.messages.create(model="claude-sonnet-4-5", max_tokens=16,
                                   messages=[{"role": "user", "content": "x"}])
        self.assertEqual(ctx.exception.status_code, 502)
        th.join(timeout=10)

    def test_rejected_inject_does_not_hotloop(self) -> None:
        # Regression for the hot-loop BLOCKER: if the server REJECTS the relay's inject
        # payload (400), the relay must fall back to a 502 + quarantine the pending, NOT
        # re-claim it and hammer the paid upstream forever. Trigger: an anthropic upstream
        # returns content with a non-dict element, which the server's respond validation 400s.
        import anthropic
        self.upstream.queue[:] = [(200, {
            "id": "m", "type": "message", "role": "assistant", "model": "mock-1",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi"}, 12345],   # non-dict element → 400
            "usage": {"input_tokens": 3, "output_tokens": 1}})]
        self.upstream.calls = 0
        th = self._run_relay(max_requests=1, kind="anthropic")
        client = anthropic.Anthropic(api_key="sk-mock",
                                     base_url=f"http://127.0.0.1:{self.FRONT}",
                                     max_retries=0, timeout=8)
        with self.assertRaises(anthropic.APIStatusError) as ctx:
            client.messages.create(model="claude-sonnet-4-5", max_tokens=16,
                                   messages=[{"role": "user", "content": "x"}])
        self.assertEqual(ctx.exception.status_code, 502)
        th.join(timeout=10)
        self.assertEqual(self.upstream.calls, 1,
                         "upstream must be called exactly once (no hot loop)")

    def test_network_failure_relayed_as_502(self) -> None:
        # Unreachable upstream → 502 to the app (not a hang, not a crash).
        import anthropic
        th = self._run_relay(max_requests=1, target="http://127.0.0.1:9")  # nothing listens
        client = anthropic.Anthropic(api_key="sk-mock",
                                     base_url=f"http://127.0.0.1:{self.FRONT}",
                                     max_retries=0, timeout=8)
        with self.assertRaises(anthropic.APIStatusError) as ctx:
            client.messages.create(model="claude-sonnet-4-5", max_tokens=16,
                                   messages=[{"role": "user", "content": "x"}])
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("unreachable", str(ctx.exception))
        th.join(timeout=10)

    def test_relay_multi_pending_no_mixup(self) -> None:
        # Two concurrent app requests through the relay must each get a valid upstream
        # response (exercises the inflight/dedup + task fan-out; a deadlock or double-handle
        # would fail here).
        import anthropic
        self.upstream.queue[:] = []
        results = {}

        def worker(tag):
            c = anthropic.Anthropic(api_key="sk-mock",
                                    base_url=f"http://127.0.0.1:{self.FRONT}",
                                    max_retries=0, timeout=12)
            # queue an echo response right before the call so the upstream returns tag-tied text
            self.upstream.queue.append((200, {
                "id": "cc", "object": "chat.completion", "model": "mock-1",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": f"reply-{tag}"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}))
            results[tag] = c.messages.create(
                model="claude-sonnet-4-5", max_tokens=16,
                messages=[{"role": "user", "content": tag}]).content[0].text

        th = self._run_relay(max_requests=2)
        ts = [threading.Thread(target=worker, args=(t,)) for t in ("Q1", "Q2")]
        for t in ts:
            t.start()
        for t in ts:
            t.join(15)
        th.join(timeout=10)
        # both completed; each got a valid reply (queue is content-agnostic, so assert set)
        self.assertEqual(set(results.values()), {"reply-Q1", "reply-Q2"})

    def test_openai_app_bridged_to_anthropic_upstream(self) -> None:
        # The second headline: an OpenAI-SDK app transparently on a (mock) Anthropic API.
        import openai
        self.upstream.queue[:] = [(200, {
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "mock-1", "stop_reason": "max_tokens", "stop_sequence": None,
            "content": [{"type": "text", "text": "bridged to anthropic"}],
            "usage": {"input_tokens": 42, "output_tokens": 5,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}})]
        self.upstream.seen.clear()
        self.upstream.seen_headers.clear()
        th = self._run_relay(max_requests=1, kind="anthropic")
        import httpx
        client = openai.OpenAI(api_key="sk-mock",
                               base_url=f"http://127.0.0.1:{self.FRONT}/v1",
                               max_retries=0,
                               http_client=httpx.Client(timeout=12))
        r = client.chat.completions.create(
            model="gpt-5.4", max_tokens=32,
            messages=[{"role": "system", "content": "sys"},
                      {"role": "user", "content": "hello"}])
        # relayed Anthropic stop_reason -> OpenAI finish_reason, real usage
        self.assertEqual(r.choices[0].message.content, "bridged to anthropic")
        self.assertEqual(r.choices[0].finish_reason, "length")   # max_tokens -> length
        self.assertEqual(r.usage.prompt_tokens, 42)
        self.assertEqual(r.usage.completion_tokens, 5)
        th.join(timeout=10)
        # the mock Anthropic upstream saw the right auth headers, path, and body
        hdr = self.upstream.seen_headers[0]
        self.assertEqual(hdr.get("x-api-key"), "k")
        self.assertIn("anthropic-version", hdr)
        body = self.upstream.seen[0]
        self.assertEqual(body["system"], "sys")
        self.assertEqual(body["model"], "mock-1")
        self.assertEqual(body["max_tokens"], 32)


if __name__ == "__main__":
    unittest.main()
