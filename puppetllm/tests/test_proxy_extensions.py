"""puppetllm 拡張機能のテスト: コスト目安 / 擬似キャッシュ / Bedrock 経路。

既存の test_fake_server.py (Anthropic 経路 + 制御 API の回帰) とは分離。

実行:
  python3 -m unittest puppetllm.tests.test_proxy_extensions -v
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from puppetllm import pricing
from puppetllm.cache_sim import CacheSimulator, extract_cache_prefix, analyze_request
from puppetllm.providers import eventstream


def _import_fresh():
    """fake_server をリロードしてサーバ状態 (pending/history/cache) をクリーンに。"""
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
        # 不明はデフォルト (sonnet)
        self.assertEqual(pricing.resolve_family("mystery-model"), "sonnet")
        self.assertEqual(pricing.resolve_family(None), "sonnet")

    def test_approx_tokens(self) -> None:
        self.assertEqual(pricing.approx_tokens(""), 0)
        self.assertEqual(pricing.approx_tokens(None), 0)
        self.assertEqual(pricing.approx_tokens("abcd"), 1)       # 4 chars / 4
        self.assertEqual(pricing.approx_tokens("a" * 8), 2)
        # dict は JSON 直列化長 / 4 (>0)
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
        # sonnet: input 3.0, cache_read 0.30 → 1Mtok read で約 2.70 USD 節約
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
        # 動的な messages 部分が違っても prefix (system まで) は同一ハッシュ
        self.assertEqual(p1.hash, p2.hash)

    def test_prefix_changes_with_system(self) -> None:
        a = extract_cache_prefix(system=[{"type": "text", "text": "A",
                                          "cache_control": {"type": "ephemeral"}}])
        b = extract_cache_prefix(system=[{"type": "text", "text": "B",
                                          "cache_control": {"type": "ephemeral"}}])
        self.assertNotEqual(a.hash, b.hash)


class TestCacheSimulator(unittest.TestCase):
    def _rc(self, text: str = "stable"):
        """system に 1 breakpoint を持つ RequestCache。"""
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
        r = sim.observe(rc, "sonnet", now=400.0)      # TTL 切れ → miss 再作成
        self.assertEqual(r["status"], "miss")

    def test_ttl_honor_disabled(self) -> None:
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=False, min_cacheable_tokens=0)
        rc = self._rc()
        sim.observe(rc, "sonnet", now=0.0)
        r = sim.observe(rc, "sonnet", now=99999.0)    # TTL 無視 → hit
        self.assertEqual(r["status"], "hit")

    def test_hit_refreshes_ttl(self) -> None:
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        rc = self._rc()
        sim.observe(rc, "sonnet", now=0.0)
        sim.observe(rc, "sonnet", now=250.0)          # hit, created_at を 250 に延長
        r = sim.observe(rc, "sonnet", now=500.0)      # 250 から 250s 経過 → まだ生存 → hit
        self.assertEqual(r["status"], "hit")

    def test_ttl_boundary_inclusive(self) -> None:
        # age == ttl はちょうど生存 (`<=`)。`<` だと miss になる境界を固定。
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        rc = self._rc()
        sim.observe(rc, "sonnet", now=0.0)
        self.assertEqual(sim.observe(rc, "sonnet", now=300.0)["status"], "hit")

    def test_read_refreshes_ttl_independently(self) -> None:
        # write 経路と独立に read 経路が created_at を更新することを直接検証。
        # turn2 は system+u1 (sc=2) を read するが、自身の BP は system(sc=1)+u2(sc=3) で
        # sc=2 を write しない → sc=2 の TTL 延長は read 経路だけが行う。
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        sysb = [{"type": "text", "text": "S " * 5, "cache_control": {"type": "ephemeral"}}]
        sim.observe(analyze_request(system=sysb, messages=[
            {"role": "user", "content": [{"type": "text", "text": "u1 " * 5,
                                          "cache_control": {"type": "ephemeral"}}]}]), "m", now=0.0)
        r = sim.observe(analyze_request(system=sysb, messages=[
            {"role": "user", "content": [{"type": "text", "text": "u1 " * 5}]},
            {"role": "user", "content": [{"type": "text", "text": "u2 " * 5,
                                          "cache_control": {"type": "ephemeral"}}]}]), "m", now=250.0)
        self.assertEqual(r["read_seg_count"], 2)            # 非 BP の sc=2 を read
        ent = {e["seg_count"]: e for e in sim.entries(now=250.0)}
        self.assertEqual(ent[2]["age_seconds"], 0.0)        # read で created_at=250 に更新 (write 経路は sc=2 を触れない)

    def test_cache_is_model_scoped(self) -> None:
        # 実機はキャッシュが model 単位。同一 content でも別 model では hit しない。
        sim = CacheSimulator(min_cacheable_tokens=0)
        rc = self._rc("STABLE " * 5)
        self.assertEqual(sim.observe(rc, "opus", now=0.0)["status"], "miss")
        self.assertEqual(sim.observe(rc, "sonnet", now=1.0)["status"], "miss")  # 別 model → 別キャッシュ
        self.assertEqual(sim.observe(rc, "opus", now=2.0)["status"], "hit")     # 同 model → hit

    def test_max_breakpoints_cap(self) -> None:
        # cache_control 6 個 → 実機制限の deepest 4 のみ write される。
        sim = CacheSimulator(min_cacheable_tokens=0)
        msgs = [{"role": "user", "content": [{"type": "text", "text": f"m{i} " * 3,
                                              "cache_control": {"type": "ephemeral"}}]} for i in range(5)]
        rc = analyze_request(
            system=[{"type": "text", "text": "S " * 3, "cache_control": {"type": "ephemeral"}}],
            messages=msgs)
        self.assertEqual(len(rc.breakpoints), 6)
        sim.observe(rc, "m", now=0.0)
        self.assertEqual(len(sim.index), 4)   # 最深 4 のみ (最浅 2 は drop)

    def test_incremental_multibreakpoint_prefix_match(self) -> None:
        """★核心★ system anchor(BP1) + 末尾移動(BP2) で、turn2 が前 turn の prefix を
        前方一致 read する (cache_control マーカーが動いても content 一致で hit)。"""
        sim = CacheSimulator(ttl_seconds=300, honor_ttl=True, min_cacheable_tokens=0)
        sysb = [{"type": "text", "text": "STABLE " * 30, "cache_control": {"type": "ephemeral"}}]

        # turn1: system(BP1) + user u1 末尾に BP2
        rc1 = analyze_request(system=sysb, messages=[
            {"role": "user", "content": [{"type": "text", "text": "u1 " * 20,
                                          "cache_control": {"type": "ephemeral"}}]},
        ])
        r1 = sim.observe(rc1, "opus", now=0.0)
        self.assertEqual(r1["status"], "miss")
        up_to_u1 = rc1.prefix_tokens(rc1.breakpoints[-1] + 1)
        self.assertEqual(r1["cache_creation_tokens"], up_to_u1)  # system+u1 を write

        # turn2: u1 はもう末尾でない (cc 無し) / assistant a1 / user u2 に BP2 が前進
        rc2 = analyze_request(system=sysb, messages=[
            {"role": "user", "content": [{"type": "text", "text": "u1 " * 20}]},   # cc は無い
            {"role": "assistant", "content": [{"type": "text", "text": "a1 " * 10}]},
            {"role": "user", "content": [{"type": "text", "text": "u2 " * 20,
                                          "cache_control": {"type": "ephemeral"}}]},
        ])
        r2 = sim.observe(rc2, "opus", now=5.0)
        self.assertEqual(r2["status"], "hit")
        # 前 turn の「system+u1」prefix を前方一致 read している (system だけでなく u1 まで)
        self.assertEqual(r2["cache_read_tokens"], up_to_u1)
        self.assertGreater(r2["cache_read_tokens"], rc2.prefix_tokens(1))  # system 単独より深い
        # creation は u2 まで増えた差分のみ
        deepest2 = rc2.prefix_tokens(rc2.breakpoints[-1] + 1)
        self.assertEqual(r2["cache_creation_tokens"], deepest2 - up_to_u1)


class TestCacheMinFloor(unittest.TestCase):
    """最小キャッシュ閾値: prefix が閾値未満なら非キャッシュ (over-report 防止)。"""

    def _rc(self, n_chars: int):
        return analyze_request(system=[{"type": "text", "text": "x" * n_chars,
                                        "cache_control": {"type": "ephemeral"}}])

    def test_below_min_is_none(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=1000)
        rc = self._rc(40)  # ~10 tok << 1000
        r1 = sim.observe(rc, "m", now=0.0)
        self.assertEqual(r1["status"], "none")
        self.assertEqual(r1["cache_creation_tokens"], 0)
        r2 = sim.observe(rc, "m", now=1.0)            # 2 回目も write されてないので hit しない
        self.assertEqual(r2["status"], "none")
        self.assertEqual(r2["cache_read_tokens"], 0)
        self.assertEqual(len(sim.index), 0)            # index に何も書かれない

    def test_at_or_above_min_caches(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=50)
        rc = self._rc(4000)  # ~1000 tok >> 50
        self.assertEqual(sim.observe(rc, "m", now=0.0)["status"], "miss")
        self.assertEqual(sim.observe(rc, "m", now=1.0)["status"], "hit")

    def test_model_based_default_opus_vs_sonnet(self) -> None:
        # 同一 prefix (~1500 tok) が Sonnet(min 1024) ではキャッシュ、Opus(min 4096) では非キャッシュ。
        rc = self._rc(6000)  # ~1500 tok
        sonnet = CacheSimulator()  # model-based
        self.assertEqual(sonnet.observe(rc, "claude-sonnet-4", now=0.0)["status"], "miss")  # write 成功
        opus = CacheSimulator()
        self.assertEqual(opus.observe(rc, "claude-opus-4", now=0.0)["status"], "none")      # 閾値未満

    def test_floor_boundary_inclusive(self) -> None:
        # prefix == 閾値ちょうどは cached (`>=`)、閾値+1 要求では非キャッシュ。`>` への退行を固定。
        rc = self._rc(400)
        exact = rc.prefix_tokens(rc.breakpoints[-1] + 1)
        self.assertEqual(CacheSimulator(min_cacheable_tokens=exact).observe(rc, "m", now=0.0)["status"], "miss")
        self.assertEqual(CacheSimulator(min_cacheable_tokens=exact + 1).observe(rc, "m", now=0.0)["status"], "none")


class TestCacheLookback(unittest.TestCase):
    """20-block lookback: breakpoint から 20 segment 超え離れた prior prefix は read できない。"""

    def _conv(self, n_msgs: int):
        """system(BP) + user×n。最後の user にのみ BP2。"""
        msgs = [{"role": "user", "content": [{"type": "text", "text": f"m{i} " * 5}]}
                for i in range(n_msgs)]
        if msgs:  # 末尾に cache_control
            msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        return analyze_request(
            system=[{"type": "text", "text": "S " * 5, "cache_control": {"type": "ephemeral"}}],
            messages=msgs,
        )

    def test_far_prefix_not_read(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=0)
        # turn1: 1 message (末尾 BP)。system(seg0) と up-to-m0(seg1) が write される。
        sim.observe(self._conv(1), "m", now=0.0)
        # turn2: いきなり 25 message 追加 (1 turn で >20 block)。末尾 BP は seg26 付近。
        #   前 turn の up-to-m0 prefix (seg2) は 20 超え離れる → lookback 圏外。
        #   ただし system(seg1) は system breakpoint 自身が anchor なので依然 read 可。
        r = sim.observe(self._conv(25), "m", now=1.0)
        # system までは hit (anchor が近い)、しかし up-to-m0 までは届かない
        self.assertEqual(r["read_seg_count"], 1)   # system prefix (seg_count=1) のみ
        self.assertGreater(r["cache_read_tokens"], 0)

    def test_near_prefix_is_read(self) -> None:
        sim = CacheSimulator(min_cacheable_tokens=0)
        sim.observe(self._conv(1), "m", now=0.0)
        r = sim.observe(self._conv(2), "m", now=1.0)  # +1 message のみ → 前 prefix は近い → 深く read
        self.assertGreaterEqual(r["read_seg_count"], 2)  # up-to-m0 まで read (system より深い)

    def test_lookback_boundary_exactly_20_reads(self) -> None:
        # sc=2 と末尾 BP の距離がちょうど 20 → read 可 (`<= 20` 包含)。
        sim = CacheSimulator(min_cacheable_tokens=0)
        sim.observe(self._conv(1), "m", now=0.0)        # sc=1, sc=2 を write
        r = sim.observe(self._conv(21), "m", now=1.0)   # 末尾 BP=seg22, sc=2 との距離 = 20
        self.assertEqual(r["read_seg_count"], 2)

    def test_lookback_boundary_21_not_read(self) -> None:
        # 距離 21 → 圏外。system(sc=1, anchor 距離 0) のみ read。`< 20` への退行も固定。
        sim = CacheSimulator(min_cacheable_tokens=0)
        sim.observe(self._conv(1), "m", now=0.0)
        r = sim.observe(self._conv(22), "m", now=1.0)   # 末尾 BP=seg23, sc=2 との距離 = 21
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
        frame[-1] ^= 0xFF  # message CRC を壊す
        with self.assertRaises(ValueError):
            eventstream.decode_messages(bytes(frame))


# ── HTTP: cost stats / cache (Anthropic 経路) ────────────────────────


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
                # usage が応答に乗る
                usage = r.json()["usage"]
                self.assertGreater(usage["input_tokens"], 0)
                self.assertGreater(usage["output_tokens"], 0)
                # stats 集計
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
                # 1 回目: miss (creation)
                await self._round_trip(c, {
                    "model": "sonnet", "stream": False, "system": system,
                    "messages": [{"role": "user", "content": "first"}],
                })
                # 2 回目: 同一 system prefix → hit
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
        """P2 回帰: system が文字列 + breakpoint が message 上でも uncached>0 かつ
        uncached = total - creation (prefix ⊆ total が保たれ 0 クランプしない)。"""
        async def run() -> None:
            async with await self._client() as c:
                await self._round_trip(c, {
                    "model": "opus", "stream": False,
                    "system": "plain string system prompt " * 20,  # 文字列 system
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "stable ctx " * 40,
                         "cache_control": {"type": "ephemeral"}},  # breakpoint は message 上
                        {"type": "text", "text": "dynamic tail"},
                    ]}],
                }, reply_text="ok")
                h = (await c.get("/_control/history")).json()["history"][-1]
                total = h["request"]["input_tokens_total"]
                u = h["usage"]
                self.assertEqual(u["cache_creation_input_tokens"], h["cache"]["cache_creation_tokens"])
                self.assertEqual(u["input_tokens"], total - u["cache_creation_input_tokens"])
                self.assertGreater(u["input_tokens"], 0)  # クランプされていない
        _run(run())

    def test_stream_carries_usage(self) -> None:
        """ストリーム応答に概算 usage が乗る (message_start に cache、message_delta に output)。"""
        async def run() -> None:
            async with await self._client() as c:
                system = [{"type": "text", "text": "stable " * 60,
                           "cache_control": {"type": "ephemeral"}}]
                # 1回目 miss でキャッシュ作成
                await self._round_trip(c, {"model": "opus", "stream": False, "system": system,
                                           "messages": [{"role": "user", "content": "warm"}]})
                # 2回目 stream で hit → message_start に cache_read が乗るはず
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
                self.assertIn("cache_read_input_tokens", body)   # message_start usage
                self.assertIn("message_delta", body)
                # message_delta に output_tokens > 0
                import re
                deltas = re.findall(r'"output_tokens":\s*(\d+)', body)
                self.assertTrue(any(int(x) > 0 for x in deltas), body[:200])
        _run(run())


# ── HTTP: Bedrock 経路 ───────────────────────────────────────────────


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
                # provider が bedrock、model は URL から
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
                # eventstream バイナリをデコードして Anthropic イベントを取り出す
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
        """Bedrock 経路でも同時 2 invoke を pending_id 指定で個別応答できる (取り違えなし)。"""
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
                # 両方 provider=bedrock
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


if __name__ == "__main__":
    unittest.main()
