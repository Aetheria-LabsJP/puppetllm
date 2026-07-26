"""Unit tests for puppetllm.batches (Anthropic Message Batches compatible surface).

HTTP paths are driven asynchronously with httpx.AsyncClient + ASGITransport (same
pattern as test_fake_server), and a final round-trip through the anthropic SDK
(uvicorn on a separate port) confirms SDK compatibility including results_url
resolution and JSONL result parsing.

Run (from the repo root):
  python3 -m unittest puppetllm.tests.test_batches -v
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import threading
import time
import unittest
from typing import Any

# Same reasoning as test_fake_server: HTTP tests use small prefixes.
os.environ["PUPPETLLM_CACHE_MIN_TOKENS"] = "0"


def _import_fresh():
    """Reload the fake_server module to keep server state clean.

    batches.py holds only a module reference to fake_server (attributes resolved at
    call time), and importlib.reload mutates the module object in place, so the
    reloaded state/app are picked up by the batches router automatically.
    """
    import importlib
    from puppetllm import fake_server as fs
    importlib.reload(fs)
    return fs


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _batch_requests(*cids: str) -> list[dict[str, Any]]:
    return [
        {"custom_id": cid,
         "params": {"model": "claude-sonnet-test", "max_tokens": 64,
                    "messages": [{"role": "user", "content": f"question {cid}"}]}}
        for cid in cids
    ]


class TestBatchHTTP(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _import_fresh()

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app),
            base_url="http://test",
        )

    async def _create(self, c: Any, *cids: str) -> dict[str, Any]:
        r = await c.post("/v1/messages/batches", json={"requests": _batch_requests(*cids)})
        self.assertEqual(r.status_code, 200)
        return r.json()

    async def _retrieve(self, c: Any, batch_id: str) -> dict[str, Any]:
        r = await c.get(f"/v1/messages/batches/{batch_id}")
        self.assertEqual(r.status_code, 200)
        return r.json()

    async def _wait_ended(self, c: Any, batch_id: str, timeout_s: float = 5.0) -> dict[str, Any]:
        """Poll retrieve until processing_status == ended (collector tasks are async)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            b = await self._retrieve(c, batch_id)
            if b["processing_status"] == "ended":
                return b
            await asyncio.sleep(0.02)
        self.fail(f"batch {batch_id} never ended")

    async def _wait_until(self, probe: Any, what: str, timeout_s: float = 5.0) -> Any:
        """Poll an async predicate until it returns truthy (no fixed sleeps → no flakes)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            got = await probe()
            if got:
                return got
            await asyncio.sleep(0.01)
        self.fail(f"timed out waiting for: {what}")

    async def _wait_quiet(self, c: Any) -> None:
        """Wait until no pending is left and every collector task has finished.

        Replaces fixed sleeps when asserting "nothing was left behind": settling is
        driven by background tasks, so a fixed sleep is a flake waiting for a loaded box.
        """
        from puppetllm import batches as bmod

        async def probe() -> bool:
            if (await c.get("/_control/pending")).json()["count"]:
                return False
            return not any(not t.done() for t in bmod._collector_tasks)

        await self._wait_until(probe, "pendings drained and collectors finished")

    async def _wait_succeeded(self, c: Any, batch_id: str, n: int) -> None:
        """Wait until the batch has stored n succeeded results."""
        async def probe() -> bool:
            cb = (await c.get("/_control/batches")).json()
            for b in cb["batches"]:
                if b["batch_id"] == batch_id:
                    return b["request_counts"]["succeeded"] >= n
            return False

        await self._wait_until(probe, f"{n} succeeded in {batch_id}")

    async def _results(self, c: Any, batch_id: str) -> dict[str, dict[str, Any]]:
        r = await c.get(f"/v1/messages/batches/{batch_id}/results")
        self.assertEqual(r.status_code, 200)
        out: dict[str, dict[str, Any]] = {}
        for line in r.text.splitlines():
            item = json.loads(line)
            out[item["custom_id"]] = item["result"]
        return out

    # ── creation / pending exposure ──────────────────────────────────

    def test_create_registers_pendings_with_batch_metadata(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "a", "b")
                self.assertTrue(batch["id"].startswith("msgbatch_"))
                self.assertEqual(batch["type"], "message_batch")
                self.assertEqual(batch["processing_status"], "in_progress")
                self.assertEqual(batch["request_counts"]["processing"], 2)
                self.assertIsNone(batch["results_url"])
                self.assertIsNone(batch["ended_at"])
                # both sub-requests are visible as ordinary pendings, tagged with
                # batch_id/custom_id so a responder can tell what it is answering
                p = await c.get("/_control/pending")
                body = p.json()
                self.assertEqual(body["count"], 2)
                tags = {e["request"]["custom_id"]: e["request"]["batch_id"]
                        for e in body["pending"]}
                self.assertEqual(set(tags), {"a", "b"})
                self.assertTrue(all(v == batch["id"] for v in tags.values()))
                # snapshots keep the normalized request shape (messages etc.)
                self.assertEqual(
                    body["pending"][0]["request"]["messages"][0]["role"], "user")
        _run(run())

    def test_create_validation(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # empty / missing requests
                for payload in ({}, {"requests": []}, {"requests": "x"}):
                    r = await c.post("/v1/messages/batches", json=payload)
                    self.assertEqual(r.status_code, 400)
                    self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
                # duplicate custom_id
                r = await c.post("/v1/messages/batches",
                                 json={"requests": _batch_requests("dup", "dup")})
                self.assertEqual(r.status_code, 400)
                self.assertIn("duplicate", r.json()["error"]["message"])
                # streaming params are rejected
                reqs = _batch_requests("s")
                reqs[0]["params"]["stream"] = True
                r = await c.post("/v1/messages/batches", json={"requests": reqs})
                self.assertEqual(r.status_code, 400)
                self.assertIn("stream", r.json()["error"]["message"])
                # missing params object
                r = await c.post("/v1/messages/batches",
                                 json={"requests": [{"custom_id": "x"}]})
                self.assertEqual(r.status_code, 400)
                # nothing half-created after the validation failures
                cb = await c.get("/_control/batches")
                self.assertEqual(cb.json()["count"], 0)
                p = await c.get("/_control/pending")
                self.assertEqual(p.json()["count"], 0)
        _run(run())

    # ── injection by custom_id / pending_id + auto-end ───────────────

    def test_respond_and_error_by_custom_id_auto_end(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "ok1", "err1")
                # succeeded via /_control/respond addressed by custom_id
                r = await c.post("/_control/respond", json={
                    "custom_id": "ok1",
                    "content": [{"type": "text", "text": "batch says hi"}],
                })
                self.assertEqual(r.status_code, 200)
                # errored via /_control/error addressed by custom_id
                r = await c.post("/_control/error", json={
                    "custom_id": "err1", "status": 429,
                    "type": "rate_limit_error", "message": "throttled",
                })
                self.assertEqual(r.status_code, 200)

                b = await self._wait_ended(c, batch["id"])
                self.assertEqual(b["request_counts"],
                                 {"processing": 0, "succeeded": 1, "errored": 1,
                                  "canceled": 0, "expired": 0})
                self.assertIsNotNone(b["ended_at"])
                self.assertTrue(
                    b["results_url"].endswith(f"/v1/messages/batches/{batch['id']}/results"))

                results = await self._results(c, batch["id"])
                ok = results["ok1"]
                self.assertEqual(ok["type"], "succeeded")
                self.assertEqual(ok["message"]["role"], "assistant")
                self.assertEqual(ok["message"]["content"],
                                 [{"type": "text", "text": "batch says hi"}])
                self.assertEqual(ok["message"]["stop_reason"], "end_turn")
                self.assertIn("usage", ok["message"])
                err = results["err1"]
                self.assertEqual(err["type"], "errored")
                self.assertEqual(err["error"]["type"], "error")
                self.assertEqual(err["error"]["error"]["type"], "rate_limit_error")
                self.assertEqual(err["error"]["error"]["message"], "throttled")

                # no pendings left behind
                p = await c.get("/_control/pending")
                self.assertEqual(p.json()["count"], 0)
        _run(run())

    def test_respond_by_pending_id_still_works(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "solo")
                p = await c.get("/_control/pending")
                pid = p.json()["pending"][0]["pending_id"]
                r = await c.post("/_control/respond", json={
                    "pending_id": pid,
                    "content": [{"type": "text", "text": "via pending_id"}],
                })
                self.assertEqual(r.status_code, 200)
                b = await self._wait_ended(c, batch["id"])
                self.assertEqual(b["request_counts"]["succeeded"], 1)
        _run(run())

    def test_custom_id_addressing_errors(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._create(c, "x")
                # unknown custom_id
                r = await c.post("/_control/respond",
                                 json={"custom_id": "nope", "content": []})
                self.assertEqual(r.status_code, 400)
                # unknown batch_id
                r = await c.post("/_control/respond",
                                 json={"custom_id": "x", "batch_id": "msgbatch_missing",
                                       "content": []})
                self.assertEqual(r.status_code, 400)
                # same custom_id unresolved in two batches → ambiguous without batch_id
                b2 = await self._create(c, "x")
                r = await c.post("/_control/respond",
                                 json={"custom_id": "x", "content": []})
                self.assertEqual(r.status_code, 400)
                self.assertIn("batch_ids", r.json())
                # disambiguated by batch_id → OK
                r = await c.post("/_control/respond", json={
                    "custom_id": "x", "batch_id": b2["id"],
                    "content": [{"type": "text", "text": "second"}],
                })
                self.assertEqual(r.status_code, 200)
        _run(run())

    # ── control-plane lifecycle injection ────────────────────────────

    def test_control_batch_end_expires_unresolved(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "done", "left1", "left2")
                r = await c.post("/_control/respond", json={
                    "custom_id": "done",
                    "content": [{"type": "text", "text": "answered"}],
                })
                self.assertEqual(r.status_code, 200)
                await self._wait_succeeded(c, batch["id"], 1)
                # force-end: default resolves the rest as expired
                r = await c.post("/_control/batch/end", json={})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["finalized"], 2)
                self.assertTrue(r.json()["ended"])
                self.assertEqual(r.json()["in_flight"], 0)
                b = await self._retrieve(c, batch["id"])
                self.assertEqual(b["processing_status"], "ended")  # synchronous
                self.assertEqual(b["request_counts"]["succeeded"], 1)
                self.assertEqual(b["request_counts"]["expired"], 2)
                results = await self._results(c, batch["id"])
                self.assertEqual(results["left1"], {"type": "expired"})
                self.assertEqual(results["left2"], {"type": "expired"})
                # pendings were swept
                p = await c.get("/_control/pending")
                self.assertEqual(p.json()["count"], 0)
                # ending again → 400
                r = await c.post("/_control/batch/end", json={"batch_id": batch["id"]})
                self.assertEqual(r.status_code, 400)
        _run(run())

    def test_control_batch_result_individual(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "a", "b")
                # invalid type is rejected
                r = await c.post("/_control/batch/result",
                                 json={"custom_id": "a", "type": "succeeded"})
                self.assertEqual(r.status_code, 400)
                # expire just "a"
                r = await c.post("/_control/batch/result",
                                 json={"custom_id": "a", "type": "expired"})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["processing_status"], "in_progress")
                # double-inject → 400
                r = await c.post("/_control/batch/result",
                                 json={"custom_id": "a", "type": "canceled",
                                       "batch_id": batch["id"]})
                self.assertEqual(r.status_code, 400)
                # resolving the last one ends the batch
                r = await c.post("/_control/batch/result",
                                 json={"custom_id": "b", "type": "canceled"})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["processing_status"], "ended")
                results = await self._results(c, batch["id"])
                self.assertEqual(results["a"], {"type": "expired"})
                self.assertEqual(results["b"], {"type": "canceled"})
        _run(run())

    def test_cancel_endpoint(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "kept", "dropped")
                r = await c.post("/_control/respond", json={
                    "custom_id": "kept",
                    "content": [{"type": "text", "text": "made it"}],
                })
                self.assertEqual(r.status_code, 200)
                await self._wait_succeeded(c, batch["id"], 1)
                r = await c.post(f"/v1/messages/batches/{batch['id']}/cancel")
                self.assertEqual(r.status_code, 200)
                b = r.json()
                self.assertEqual(b["processing_status"], "ended")
                self.assertIsNotNone(b["cancel_initiated_at"])
                self.assertEqual(b["request_counts"]["succeeded"], 1)
                self.assertEqual(b["request_counts"]["canceled"], 1)
                # cancel on an ended batch is a no-op returning current state
                r = await c.post(f"/v1/messages/batches/{batch['id']}/cancel")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["request_counts"]["canceled"], 1)
        _run(run())

    # ── results gating / delete / list / 404s ────────────────────────

    def test_results_before_ended_and_404(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "w")
                r = await c.get(f"/v1/messages/batches/{batch['id']}/results")
                self.assertEqual(r.status_code, 400)
                self.assertIn("not ended", r.json()["error"]["message"])
                r = await c.get("/v1/messages/batches/msgbatch_missing")
                self.assertEqual(r.status_code, 404)
                self.assertEqual(r.json()["error"]["type"], "not_found_error")
                r = await c.get("/v1/messages/batches/msgbatch_missing/results")
                self.assertEqual(r.status_code, 404)
        _run(run())

    def test_delete(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "d")
                # before ended → 400
                r = await c.request("DELETE", f"/v1/messages/batches/{batch['id']}")
                self.assertEqual(r.status_code, 400)
                await c.post("/_control/batch/end",
                             json={"batch_id": batch["id"], "unresolved": "canceled"})
                r = await c.request("DELETE", f"/v1/messages/batches/{batch['id']}")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["type"], "message_batch_deleted")
                r = await c.get(f"/v1/messages/batches/{batch['id']}")
                self.assertEqual(r.status_code, 404)
        _run(run())

    def test_list(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                b1 = await self._create(c, "l1")
                b2 = await self._create(c, "l2")
                r = await c.get("/v1/messages/batches")
                body = r.json()
                self.assertFalse(body["has_more"])
                ids = [b["id"] for b in body["data"]]
                self.assertEqual(ids, [b2["id"], b1["id"]])  # newest first
                self.assertEqual(body["first_id"], b2["id"])
                self.assertEqual(body["last_id"], b1["id"])
                # limit + after_id pagination
                r = await c.get("/v1/messages/batches", params={"limit": 1})
                body = r.json()
                self.assertEqual([b["id"] for b in body["data"]], [b2["id"]])
                self.assertTrue(body["has_more"])
                r = await c.get("/v1/messages/batches",
                                params={"limit": 1, "after_id": b2["id"]})
                body = r.json()
                self.assertEqual([b["id"] for b in body["data"]], [b1["id"]])
                self.assertFalse(body["has_more"])
        _run(run())

    # ── cost discount / history / clear ──────────────────────────────

    def test_history_gets_batch_flag_and_half_cost(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                # normal (non-batch) request for a price baseline with identical shape
                req_task = asyncio.create_task(c.post("/v1/messages", json={
                    "model": "claude-sonnet-test", "max_tokens": 64,
                    "messages": [{"role": "user", "content": "question ok1"}],
                }, timeout=10))
                for _ in range(100):
                    p = await c.get("/_control/pending")
                    if p.json().get("has_pending"):
                        break
                    await asyncio.sleep(0.02)
                await c.post("/_control/respond",
                             json={"content": [{"type": "text", "text": "same reply"}]})
                await req_task

                batch = await self._create(c, "ok1")
                await c.post("/_control/respond", json={
                    "custom_id": "ok1",
                    "content": [{"type": "text", "text": "same reply"}],
                })
                await self._wait_ended(c, batch["id"])

                h = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(len(h), 2)
                normal, batched = h[0], h[1]
                self.assertNotIn("batch", normal)
                self.assertTrue(batched["batch"])
                self.assertEqual(batched["cost"]["batch_discount"], 0.5)
                # same request/response shape → batch total is exactly half
                self.assertAlmostEqual(batched["cost"]["total_usd"],
                                       round(normal["cost"]["total_usd"] * 0.5, 6),
                                       places=6)
                # stats aggregates the discounted figure without erroring
                s = (await c.get("/_control/stats")).json()
                self.assertEqual(s["completed_requests"], 2)
        _run(run())

    def test_expired_entries_not_recorded_in_history(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._create(c, "gone")
                await c.post("/_control/batch/end", json={})
                # give the collector a moment (it should be a no-op)
                await asyncio.sleep(0.05)
                h = (await c.get("/_control/history")).json()["history"]
                self.assertEqual(h, [])
        _run(run())

    def test_create_rolls_back_on_unprocessable_params(self) -> None:
        """Params that pass shallow validation but crash token analysis roll back the
        whole create: no batch, no ghost pendings, no history — and a 400, not a 500."""
        async def run() -> None:
            async with await self._client() as c:
                reqs = _batch_requests("good")
                reqs.append({"custom_id": "bad",
                             "params": {"model": "claude-sonnet-test",
                                        "messages": 42}})  # non-iterable messages
                r = await c.post("/v1/messages/batches", json={"requests": reqs})
                self.assertEqual(r.status_code, 400)
                self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
                self.assertIn("requests[1].params could not be processed",
                              r.json()["error"]["message"])
                await self._wait_quiet(c)
                self.assertEqual((await c.get("/_control/batches")).json()["count"], 0)
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 0)
                self.assertEqual((await c.get("/_control/history")).json()["history"], [])
        _run(run())

    def test_auto_by_custom_id(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "a1", "a2")
                r = await c.post("/_control/auto",
                                 json={"custom_id": "a2", "text": "auto for a2"})
                self.assertEqual(r.status_code, 200)
                r = await c.post("/_control/auto",
                                 json={"custom_id": "a1", "text": "auto for a1"})
                self.assertEqual(r.status_code, 200)
                await self._wait_ended(c, batch["id"])
                results = await self._results(c, batch["id"])
                self.assertEqual(results["a1"]["message"]["content"][0]["text"],
                                 "auto for a1")
                self.assertEqual(results["a2"]["message"]["content"][0]["text"],
                                 "auto for a2")
        _run(run())

    def test_usage_override_gets_discount(self) -> None:
        """A relay-style responder overriding usage: the 50% discount applies to the
        recomputed (real-token) cost, and both flags land on the history entry."""
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "u1")
                r = await c.post("/_control/respond", json={
                    "custom_id": "u1",
                    "content": [{"type": "text", "text": "relayed"}],
                    "usage": {"input_tokens": 1000, "output_tokens": 2000},
                })
                self.assertEqual(r.status_code, 200)
                await self._wait_ended(c, batch["id"])
                entry = (await c.get("/_control/history")).json()["history"][0]
                self.assertTrue(entry["batch"])
                self.assertTrue(entry["usage_overridden"])
                self.assertEqual(entry["usage"]["input_tokens"], 1000)
                # sonnet family: (1000*3 + 2000*15)/1e6 = 0.033 → * 0.5
                self.assertAlmostEqual(entry["cost"]["total_usd"], 0.0165, places=6)
                self.assertEqual(entry["cost"]["batch_discount"], 0.5)
        _run(run())

    def test_errored_entry_history_and_stats(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "e1")
                await c.post("/_control/error", json={
                    "custom_id": "e1", "status": 400,
                    "type": "invalid_request_error", "message": "nope",
                })
                await self._wait_ended(c, batch["id"])
                entry = (await c.get("/_control/history")).json()["history"][0]
                self.assertTrue(entry["batch"])
                self.assertEqual(entry["injected_error"]["type"], "invalid_request_error")
                s = (await c.get("/_control/stats")).json()
                self.assertEqual(s["error_requests"], 1)
                self.assertEqual(s["completed_requests"], 0)
        _run(run())

    def test_non_string_ids_return_400_not_500(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                await self._create(c, "t1")
                cases = [
                    ("/_control/respond", {"pending_id": {"x": 1}, "content": []}),
                    ("/_control/respond", {"custom_id": 5, "content": []}),
                    ("/_control/respond", {"custom_id": "t1", "batch_id": [1], "content": []}),
                    ("/_control/batch/result", {"custom_id": 5, "type": "expired"}),
                    ("/_control/batch/result", {"custom_id": "t1", "batch_id": 5,
                                                "type": "expired"}),
                    ("/_control/batch/end", {"batch_id": {"x": 1}}),
                ]
                for path, payload in cases:
                    r = await c.post(path, json=payload)
                    self.assertEqual(r.status_code, 400, f"{path} {payload}")
        _run(run())

    def test_list_bad_limit_and_before_id(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                b1 = await self._create(c, "p1")
                b2 = await self._create(c, "p2")
                b3 = await self._create(c, "p3")
                # bad limit → Anthropic envelope, not FastAPI 422
                r = await c.get("/v1/messages/batches", params={"limit": "abc"})
                self.assertEqual(r.status_code, 400)
                self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
                # backward pagination: newest-first order is [b3, b2, b1];
                # before_id=b1 → window [b3, b2], limit 1 takes the adjacent item b2
                r = await c.get("/v1/messages/batches",
                                params={"limit": 1, "before_id": b1["id"]})
                body = r.json()
                self.assertEqual([b["id"] for b in body["data"]], [b2["id"]])
                self.assertTrue(body["has_more"])
                # unbounded backward page returns the whole window, newest first
                r = await c.get("/v1/messages/batches", params={"before_id": b1["id"]})
                self.assertEqual([b["id"] for b in r.json()["data"]],
                                 [b3["id"], b2["id"]])
                self.assertFalse(r.json()["has_more"])
                # out-of-range limit and unknown cursors are rejected, not papered over
                for params in ({"limit": 0}, {"limit": -5}, {"limit": 101}):
                    r = await c.get("/v1/messages/batches", params=params)
                    self.assertEqual(r.status_code, 400, params)
                    self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
                for params in ({"after_id": "msgbatch_nope"},
                               {"before_id": "msgbatch_nope"}):
                    r = await c.get("/v1/messages/batches", params=params)
                    self.assertEqual(r.status_code, 404, params)
                    self.assertEqual(r.json()["error"]["type"], "not_found_error")
        _run(run())

    def test_custom_id_format_and_batch_size_limits(self) -> None:
        """Envelope validation matches the real API, so an app that production would
        reject does not silently pass here."""
        async def run() -> None:
            async with await self._client() as c:
                bad_ids = ["", "has space", "a" * 65, "non-ascii-café",
                           "line\nbreak", "dot.dot"]
                for cid in bad_ids:
                    reqs = _batch_requests("ok")
                    reqs[0]["custom_id"] = cid
                    r = await c.post("/v1/messages/batches", json={"requests": reqs})
                    self.assertEqual(r.status_code, 400, repr(cid))
                    self.assertIn("custom_id", r.json()["error"]["message"])
                # the allowed alphabet is accepted
                reqs = _batch_requests("ok")
                reqs[0]["custom_id"] = "A-z_0-9" + "x" * 57  # exactly 64 chars
                r = await c.post("/v1/messages/batches", json={"requests": reqs})
                self.assertEqual(r.status_code, 200)
                # request-count cap (checked before anything is registered)
                huge = {"requests": [{"custom_id": f"c{i}", "params": {"messages": []}}
                                     for i in range(100_001)]}
                r = await c.post("/v1/messages/batches", json=huge)
                self.assertEqual(r.status_code, 400)
                self.assertIn("maximum", r.json()["error"]["message"])
                self.assertEqual((await c.get("/_control/batches")).json()["count"], 1)
        _run(run())

    def _park_collector(self) -> tuple[asyncio.Event, asyncio.Event]:
        """Make the next collector block mid-flight: the injection is accepted but the
        result is not stored yet. Patches _record_and_reset (which every collector goes
        through) rather than reaching into futures, so the parked state is the real one."""
        orig = self.mod._record_and_reset
        entered, release = asyncio.Event(), asyncio.Event()

        async def wrapped(*a: Any, **k: Any) -> Any:
            entered.set()
            await release.wait()
            return await orig(*a, **k)

        self.mod._record_and_reset = wrapped
        self.addCleanup(setattr, self.mod, "_record_and_reset", orig)
        return entered, release

    def test_cancel_with_inflight_injection_reports_canceling(self) -> None:
        """Cancel must not discard an accepted injection, and must not claim `ended`
        while one is still landing — it reports the real API's `canceling`, then settles."""
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "keep", "drop")
                entered, release = self._park_collector()
                r = await c.post("/_control/respond", json={
                    "custom_id": "keep",
                    "content": [{"type": "text", "text": "landed anyway"}],
                })
                self.assertEqual(r.status_code, 200)
                await asyncio.wait_for(entered.wait(), timeout=5)

                r = await c.post(f"/v1/messages/batches/{batch['id']}/cancel")
                self.assertEqual(r.status_code, 200)
                b = r.json()
                self.assertEqual(b["processing_status"], "canceling")
                self.assertIsNotNone(b["cancel_initiated_at"])
                self.assertIsNone(b["results_url"])
                self.assertEqual(b["request_counts"]["canceled"], 1)
                self.assertEqual(b["request_counts"]["processing"], 1)
                # results are refused while canceling
                rr = await c.get(f"/v1/messages/batches/{batch['id']}/results")
                self.assertEqual(rr.status_code, 400)
                # force-end reports honestly instead of a bare ok
                r = await c.post("/_control/batch/end", json={"batch_id": batch["id"]})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["finalized"], 0)
                self.assertFalse(r.json()["ended"])
                self.assertEqual(r.json()["in_flight"], 1)

                release.set()
                b = await self._wait_ended(c, batch["id"])
                self.assertEqual(b["request_counts"]["succeeded"], 1)
                self.assertEqual(b["request_counts"]["canceled"], 1)
                results = await self._results(c, batch["id"])
                self.assertEqual(results["keep"]["message"]["content"][0]["text"],
                                 "landed anyway")
                self.assertEqual(results["drop"], {"type": "canceled"})
        _run(run())

    def test_collector_crash_is_contained_as_errored(self) -> None:
        """An unexpected collector failure becomes an `errored` result instead of
        killing the task and leaving the batch unable to ever end."""
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "boom")
                orig = self.mod._build_non_stream_response

                def exploding(*a: Any, **k: Any) -> Any:
                    raise ValueError("synthetic encoder failure")

                self.mod._build_non_stream_response = exploding
                self.addCleanup(setattr, self.mod, "_build_non_stream_response", orig)
                r = await c.post("/_control/respond", json={
                    "custom_id": "boom",
                    "content": [{"type": "text", "text": "will not encode"}],
                })
                self.assertEqual(r.status_code, 200)
                # the containment path logs the traceback by design; keep it out of the
                # test report (it would read as a failure) while still exercising it
                with contextlib.redirect_stderr(io.StringIO()) as log:
                    b = await self._wait_ended(c, batch["id"])
                self.assertIn("batch collector failed", log.getvalue())
                self.assertEqual(b["request_counts"]["errored"], 1)
                results = await self._results(c, batch["id"])
                self.assertEqual(results["boom"]["type"], "errored")
                self.assertIn("failed internally",
                              results["boom"]["error"]["error"]["message"])
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 0)
        _run(run())

    def test_entry_whose_collector_vanished_is_still_finalizable(self) -> None:
        """If a collector ends without storing a result, the entry must stay finalizable
        — otherwise the batch could never end and only /_control/clear would recover it."""
        async def run() -> None:
            async with await self._client() as c:
                orig = self.mod.await_resolution

                async def vanishing(snapshot: dict[str, Any], fut: Any, **k: Any) -> Any:
                    # pending reaped, collector returns without storing anything
                    self.mod.state.pending.pop(snapshot["pending_id"], None)
                    return {"kind": "cleared", "detail": "synthetic"}

                self.mod.await_resolution = vanishing
                self.addCleanup(setattr, self.mod, "await_resolution", orig)
                batch = await self._create(c, "lost")
                await self._wait_quiet(c)
                b = await self._retrieve(c, batch["id"])
                self.assertEqual(b["processing_status"], "in_progress")
                # the escape hatch works: force-end finalizes the orphaned entry
                r = await c.post("/_control/batch/end", json={"batch_id": batch["id"]})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["finalized"], 1)
                self.assertTrue(r.json()["ended"])
                results = await self._results(c, batch["id"])
                self.assertEqual(results["lost"], {"type": "expired"})
        _run(run())

    def test_cache_savings_are_batch_discounted(self) -> None:
        """A cache hit on a batch entry saved half as much, since batch tokens bill at 50%."""
        async def run() -> None:
            async with await self._client() as c:
                system = [{"type": "text", "text": "shared preamble " * 40,
                           "cache_control": {"type": "ephemeral"}}]

                async def one(cid: str) -> None:
                    r = await c.post("/v1/messages/batches", json={"requests": [
                        {"custom_id": cid,
                         "params": {"model": "claude-sonnet-test", "max_tokens": 64,
                                    "system": system,
                                    "messages": [{"role": "user", "content": "q"}]}},
                    ]})
                    self.assertEqual(r.status_code, 200)
                    await c.post("/_control/respond",
                                 json={"custom_id": cid,
                                       "content": [{"type": "text", "text": "a"}]})
                    await self._wait_ended(c, r.json()["id"])

                await one("miss")   # writes the prefix
                await one("hit")    # reads it back
                s = (await c.get("/_control/stats")).json()
                self.assertEqual(s["cache"]["hits"], 1)
                read = s["totals"]["cache_read_input_tokens"]
                self.assertGreater(read, 0)
                from puppetllm import pricing
                full = pricing.cache_savings_usd("claude-sonnet-test", read)
                self.assertAlmostEqual(s["totals"]["cache_savings_usd"],
                                       round(full * 0.5, 6), places=6)
        _run(run())

    def test_wait_for_pending_sees_batch_entries(self) -> None:
        """The documented responder entry point works for batch pendings too."""
        async def run() -> None:
            async with await self._client() as c:
                waiter = asyncio.create_task(
                    c.get("/_control/wait_for_pending", params={"timeout": 5}, timeout=10))
                await asyncio.sleep(0)  # let the waiter register before creating
                batch = await self._create(c, "w1")
                r = await waiter
                body = r.json()
                self.assertTrue(body["has_pending"])
                self.assertEqual(body["request"]["custom_id"], "w1")
                self.assertEqual(body["request"]["batch_id"], batch["id"])
                self.assertEqual(body["pending_id"], body["request"]["pending_id"])
        _run(run())

    def test_clear_wipes_batches(self) -> None:
        async def run() -> None:
            async with await self._client() as c:
                batch = await self._create(c, "c1", "c2")
                r = await c.post("/_control/clear")
                self.assertEqual(r.status_code, 200)
                cb = await c.get("/_control/batches")
                self.assertEqual(cb.json()["count"], 0)
                r = await c.get(f"/v1/messages/batches/{batch['id']}")
                self.assertEqual(r.status_code, 404)
                p = await c.get("/_control/pending")
                self.assertEqual(p.json()["count"], 0)
        _run(run())


class TestBatchCreationRaces(unittest.TestCase):
    """Deterministically exercise the create-loop windows (the reviewer's findings):
    control endpoints and clear racing the entry-registration loop.

    register_request is wrapped so the test can hold the creation loop right before
    it registers the SECOND entry (batch published, entry 1 live, entry 2 absent),
    fire control calls into that window, then let the loop proceed.
    """

    def setUp(self) -> None:
        import importlib
        from puppetllm import fake_server as fs
        importlib.reload(fs)
        self.mod = fs

    async def _client(self) -> Any:
        import httpx
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.mod.app),
            base_url="http://test",
        )

    def _gate_second_registration(self) -> tuple[asyncio.Event, asyncio.Event]:
        """Patch register_request: pause before the 2nd call until `proceed` is set."""
        orig = self.mod.register_request
        gate, proceed = asyncio.Event(), asyncio.Event()
        calls = 0

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 2:
                gate.set()
                await proceed.wait()
            return await orig(*args, **kwargs)

        self.mod.register_request = wrapped
        self.addCleanup(setattr, self.mod, "register_request", orig)
        return gate, proceed

    def test_controls_rejected_while_creating(self) -> None:
        """batch/end, batch/result, and cancel must not touch a batch mid-creation
        (they would end it over partial entries → null results + ghost pendings)."""
        async def run() -> None:
            gate, proceed = self._gate_second_registration()
            async with await self._client() as c:
                create_task = asyncio.create_task(c.post(
                    "/v1/messages/batches",
                    json={"requests": _batch_requests("a", "b")}, timeout=10))
                await asyncio.wait_for(gate.wait(), timeout=5)
                # window: batch registered, entry "a" live, entry "b" not yet
                cb = (await c.get("/_control/batches")).json()
                self.assertTrue(cb["batches"][0]["creating"])
                bid = cb["batches"][0]["batch_id"]
                r = await c.post("/_control/batch/end", json={})
                self.assertEqual(r.status_code, 400)
                self.assertIn("being created", r.json()["error"])
                r = await c.post("/_control/batch/end", json={"batch_id": bid})
                self.assertEqual(r.status_code, 400)
                r = await c.post("/_control/batch/result",
                                 json={"custom_id": "a", "type": "expired"})
                self.assertEqual(r.status_code, 400)
                r = await c.post(f"/v1/messages/batches/{bid}/cancel")
                self.assertEqual(r.status_code, 400)
                self.assertIn("being created", r.json()["error"]["message"])
                # release the loop: create completes normally with both entries
                proceed.set()
                resp = await create_task
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertEqual(body["processing_status"], "in_progress")
                self.assertEqual(body["request_counts"]["processing"], 2)
                # and now the controls work
                r = await c.post("/_control/batch/end", json={"batch_id": bid})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["finalized"], 2)
        _run(run())

    def test_clear_during_creation_leaves_no_ghosts(self) -> None:
        """clear racing the create loop: the create returns 503 and every pending it
        registered (before AND after the clear) is swept."""
        async def run() -> None:
            gate, proceed = self._gate_second_registration()
            async with await self._client() as c:
                create_task = asyncio.create_task(c.post(
                    "/v1/messages/batches",
                    json={"requests": _batch_requests("a", "b")}, timeout=10))
                await asyncio.wait_for(gate.wait(), timeout=5)
                r = await c.post("/_control/clear")
                self.assertEqual(r.status_code, 200)
                proceed.set()
                resp = await create_task
                self.assertEqual(resp.status_code, 503)
                self.assertEqual(resp.json()["error"]["type"], "api_error")
                await self._wait_quiet(c)
                self.assertEqual((await c.get("/_control/batches")).json()["count"], 0)
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 0)
                self.assertEqual((await c.get("/_control/history")).json()["history"], [])
        _run(run())

    def test_rollback_purges_history_of_an_already_answered_entry(self) -> None:
        """A responder may answer an early entry while later ones are still being
        registered. If the create then fails, that entry's history/cost must go too —
        otherwise stats bill a request belonging to a batch that never existed."""
        async def run() -> None:
            gate, proceed = self._gate_second_registration()
            async with await self._client() as c:
                reqs = _batch_requests("good")
                reqs.append({"custom_id": "bad",
                             "params": {"model": "claude-sonnet-test",
                                        "messages": 42}})  # explodes in register_request
                create_task = asyncio.create_task(c.post(
                    "/v1/messages/batches", json={"requests": reqs}, timeout=10))
                await asyncio.wait_for(gate.wait(), timeout=5)
                r = await c.post("/_control/respond", json={
                    "custom_id": "good",
                    "content": [{"type": "text", "text": "answered mid-create"}],
                })
                self.assertEqual(r.status_code, 200)
                # wait until that answer really is recorded, so the purge is exercised
                await self._wait_until(
                    lambda: self._history_len(c, 1), "history recorded before rollback")

                proceed.set()
                resp = await create_task
                self.assertEqual(resp.status_code, 400)
                await self._wait_quiet(c)
                self.assertEqual((await c.get("/_control/batches")).json()["count"], 0)
                self.assertEqual((await c.get("/_control/pending")).json()["count"], 0)
                self.assertEqual((await c.get("/_control/history")).json()["history"], [])
                s = (await c.get("/_control/stats")).json()
                self.assertEqual(s["completed_requests"], 0)
                self.assertEqual(s["totals"]["total_usd"], 0.0)
        _run(run())

    async def _history_len(self, c: Any, n: int) -> bool:
        return len((await c.get("/_control/history")).json()["history"]) >= n

    async def _wait_until(self, probe: Any, what: str, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if await probe():
                return
            await asyncio.sleep(0.01)
        self.fail(f"timed out waiting for: {what}")

    async def _wait_quiet(self, c: Any) -> None:
        from puppetllm import batches as bmod

        async def probe() -> bool:
            if (await c.get("/_control/pending")).json()["count"]:
                return False
            return not any(not t.done() for t in bmod._collector_tasks)

        await self._wait_until(probe, "pendings drained and collectors finished")

    def test_respond_during_creation_is_allowed(self) -> None:
        """Injecting a response mid-creation is fine (auto-end is suppressed until the
        create completes; the batch then settles — even to ended if all are answered)."""
        async def run() -> None:
            gate, proceed = self._gate_second_registration()
            async with await self._client() as c:
                create_task = asyncio.create_task(c.post(
                    "/v1/messages/batches",
                    json={"requests": _batch_requests("a", "b")}, timeout=10))
                await asyncio.wait_for(gate.wait(), timeout=5)
                r = await c.post("/_control/respond", json={
                    "custom_id": "a",
                    "content": [{"type": "text", "text": "early answer"}],
                })
                self.assertEqual(r.status_code, 200)
                await asyncio.sleep(0.02)  # collector stores while still creating
                cb = (await c.get("/_control/batches")).json()
                self.assertEqual(cb["batches"][0]["processing_status"], "in_progress")
                proceed.set()
                resp = await create_task
                self.assertEqual(resp.status_code, 200)
                bid = resp.json()["id"]
                await c.post("/_control/respond", json={
                    "custom_id": "b",
                    "content": [{"type": "text", "text": "late answer"}],
                })
                for _ in range(100):
                    b = (await c.get(f"/v1/messages/batches/{bid}")).json()
                    if b["processing_status"] == "ended":
                        break
                    await asyncio.sleep(0.02)
                self.assertEqual(b["request_counts"]["succeeded"], 2)
        _run(run())


class TestAnthropicSDKBatches(unittest.TestCase):
    """Start uvicorn on a separate port and drive the batch flow with the anthropic SDK.

    Verifies the pieces the ASGI tests can't: pydantic validation of the batch object,
    absolute results_url resolution, and JSONL result decoding by the SDK.
    """

    PORT = 18766

    @classmethod
    def setUpClass(cls) -> None:
        import uvicorn
        from puppetllm import fake_server as fs
        import importlib
        importlib.reload(fs)
        cls.fs = fs

        cls._config = uvicorn.Config(
            fs.app, host="127.0.0.1", port=cls.PORT,
            log_level="critical", loop="asyncio",
        )
        cls._server = uvicorn.Server(cls._config)

        def _run_server() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(cls._server.serve())

        cls._thread = threading.Thread(target=_run_server, daemon=True)
        cls._thread.start()

        import urllib.request
        for _ in range(50):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{cls.PORT}/_control/health", timeout=0.5
                )
                return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("fake_server did not start")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.should_exit = True
        cls._thread.join(timeout=3)

    def setUp(self) -> None:
        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{self.PORT}/_control/clear", method="POST"
            ),
            timeout=2,
        )

    def _control_post(self, endpoint: str, payload: dict[str, Any]) -> None:
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.PORT}{endpoint}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)

    def test_batch_round_trip(self) -> None:
        import anthropic
        client = anthropic.Anthropic(
            api_key="sk-mock", base_url=f"http://127.0.0.1:{self.PORT}"
        )
        batch = client.messages.batches.create(requests=[
            {"custom_id": "r1",
             "params": {"model": "claude-sonnet-test", "max_tokens": 64,
                        "messages": [{"role": "user", "content": "hi"}]}},
            {"custom_id": "r2",
             "params": {"model": "claude-sonnet-test", "max_tokens": 64,
                        "messages": [{"role": "user", "content": "yo"}]}},
        ])
        self.assertEqual(batch.processing_status, "in_progress")
        self.assertEqual(batch.request_counts.processing, 2)

        # create() has returned, so the pendings already exist — inject directly
        self._control_post("/_control/respond", {
            "custom_id": "r1",
            "content": [{"type": "text", "text": "batch-reply"}],
        })
        self._control_post("/_control/error", {
            "custom_id": "r2", "status": 500,
            "type": "api_error", "message": "boom",
        })

        for _ in range(100):
            b = client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(0.05)
        self.assertEqual(b.processing_status, "ended")
        self.assertEqual(b.request_counts.succeeded, 1)
        self.assertEqual(b.request_counts.errored, 1)
        self.assertIsNotNone(b.results_url)

        results = {r.custom_id: r.result for r in client.messages.batches.results(batch.id)}
        self.assertEqual(set(results), {"r1", "r2"})
        self.assertEqual(results["r1"].type, "succeeded")
        self.assertEqual(results["r1"].message.content[0].text, "batch-reply")
        self.assertEqual(results["r1"].message.stop_reason, "end_turn")
        self.assertEqual(results["r2"].type, "errored")

    def test_sdk_sees_expired_and_canceled(self) -> None:
        import anthropic
        client = anthropic.Anthropic(
            api_key="sk-mock", base_url=f"http://127.0.0.1:{self.PORT}"
        )
        batch = client.messages.batches.create(requests=[
            {"custom_id": "e1",
             "params": {"model": "claude-sonnet-test", "max_tokens": 64,
                        "messages": [{"role": "user", "content": "hi"}]}},
            {"custom_id": "c1",
             "params": {"model": "claude-sonnet-test", "max_tokens": 64,
                        "messages": [{"role": "user", "content": "hi"}]}},
        ])
        self._control_post("/_control/batch/result",
                           {"custom_id": "c1", "type": "canceled"})
        self._control_post("/_control/batch/end",
                           {"batch_id": batch.id, "unresolved": "expired"})
        b = client.messages.batches.retrieve(batch.id)
        self.assertEqual(b.processing_status, "ended")
        self.assertEqual(b.request_counts.expired, 1)
        self.assertEqual(b.request_counts.canceled, 1)
        results = {r.custom_id: r.result for r in client.messages.batches.results(batch.id)}
        self.assertEqual(results["e1"].type, "expired")
        self.assertEqual(results["c1"].type, "canceled")

    def test_sdk_cancel(self) -> None:
        import anthropic
        client = anthropic.Anthropic(
            api_key="sk-mock", base_url=f"http://127.0.0.1:{self.PORT}"
        )
        batch = client.messages.batches.create(requests=[
            {"custom_id": "x",
             "params": {"model": "claude-sonnet-test", "max_tokens": 64,
                        "messages": [{"role": "user", "content": "hi"}]}},
        ])
        cancelled = client.messages.batches.cancel(batch.id)
        self.assertEqual(cancelled.processing_status, "ended")
        self.assertEqual(cancelled.request_counts.canceled, 1)


if __name__ == "__main__":
    unittest.main()
