"""SDK-level interoperability tests: real boto3 / botocore against the fake server.

The unit tests elsewhere drive the ASGI app directly and decode event streams with
puppetllm's own codec, so a matching encoder/decoder mistake would pass unnoticed. These
tests run an actual uvicorn instance and point real `bedrock-runtime` / `bedrock` / `s3`
clients at it, which is the only evidence that botocore parses what the server emits.

Skipped when boto3 / uvicorn are unavailable — set `PUPPETLLM_REQUIRE_SDK_TESTS=1` to turn
that skip into a hard failure instead (docker-compose's test profile does).

Run:
  python3 -m unittest puppetllm.tests.test_boto3_interop -v
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest
from typing import Any

try:
    import boto3
    import uvicorn
    from botocore.config import Config
    from botocore.exceptions import ClientError, EventStreamError
    _DEPS = True
except ImportError:  # pragma: no cover - exercised only where boto3 is absent
    _DEPS = False

if not _DEPS and os.environ.get("PUPPETLLM_REQUIRE_SDK_TESTS"):
    # These are the only tests that prove botocore parses what the server emits, so a run
    # that was meant to include them must fail rather than print OK with every one skipped.
    raise RuntimeError(
        "PUPPETLLM_REQUIRE_SDK_TESTS is set but boto3 / uvicorn are missing: "
        "install requirements.txt (or rebuild the image) before running this module")

os.environ.setdefault("PUPPETLLM_CACHE_MIN_TOKENS", "0")

MODEL = "anthropic.claude-opus-5"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@unittest.skipUnless(_DEPS, "boto3 / uvicorn not installed")
class TestBoto3Interop(unittest.TestCase):
    """One server for the whole class; each test drives it through real AWS clients."""

    @classmethod
    def setUpClass(cls) -> None:
        import importlib

        import httpx

        from puppetllm import fake_server as fs
        from puppetllm.providers import s3 as s3mod
        cls.fs = importlib.reload(fs)
        cls.tmp = tempfile.mkdtemp(prefix="puppetllm-boto3-")
        s3mod.reset_root(cls.tmp)
        cls.port = _free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.server = uvicorn.Server(uvicorn.Config(cls.fs.app, host="127.0.0.1", port=cls.port,
                                                   log_level="critical"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        cls.http = httpx.Client(base_url=cls.base, timeout=20)
        for _ in range(100):
            try:
                cls.http.get("/_control/health")
                break
            except Exception:
                time.sleep(0.1)
        else:  # pragma: no cover
            raise RuntimeError("server did not start")
        kw = dict(region_name="us-east-1", aws_access_key_id="dummy",
                  aws_secret_access_key="dummy", endpoint_url=cls.base,
                  config=Config(retries={"max_attempts": 0}, read_timeout=20))
        cls.rt = boto3.client("bedrock-runtime", **kw)
        cls.ctl = boto3.client("bedrock", **kw)
        cls.s3 = boto3.client("s3", **{**kw, "config": Config(
            retries={"max_attempts": 0}, s3={"addressing_style": "path"})})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        cls.http.close()
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self) -> None:
        self.http.post("/_control/clear")

    # ── helpers ──

    def _wait_pending(self, n: int = 1) -> list[dict[str, Any]]:
        for _ in range(200):
            p = self.http.get("/_control/pending").json()
            if p.get("count", 0) >= n:
                return p["pending"]
            time.sleep(0.05)
        self.fail(f"no pending after waiting (wanted {n})")

    def _in_thread(self, fn: Any) -> tuple[threading.Thread, dict]:
        out: dict[str, Any] = {}

        def run() -> None:
            try:
                out["value"] = fn()
            except BaseException as e:  # noqa: BLE001 - the test inspects the exception
                out["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t, out

    def _respond(self, **payload: Any) -> None:
        r = self.http.post("/_control/respond", json=payload)
        self.assertEqual(r.status_code, 200, r.text)

    # ── Converse ──

    def test_converse_round_trip(self) -> None:
        t, out = self._in_thread(lambda: self.rt.converse(
            modelId=MODEL,
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
            system=[{"text": "be brief"}],
            inferenceConfig={"maxTokens": 64, "temperature": 0.2},
            toolConfig={"tools": [{"toolSpec": {
                "name": "wx", "description": "weather",
                "inputSchema": {"json": {"type": "object"}}}}]},
            additionalModelRequestFields={"thinking": {"type": "adaptive"}}))
        req = self._wait_pending()[0]["request"]
        self.assertEqual((req["provider"], req["model"], req["api"]),
                         ("bedrock", "claude-opus-5", "converse"))
        self.assertEqual(req["params"]["thinking"], {"type": "adaptive"})
        self._respond(content=[
            {"type": "thinking", "thinking": "considering", "signature": "sig1"},
            {"type": "text", "text": "hi there"},
            {"type": "tool_use", "id": "tu1", "name": "wx", "input": {"city": "Tokyo"}}])
        t.join(20)
        self.assertNotIn("error", out, out.get("error"))
        r = out["value"]
        self.assertEqual(r["stopReason"], "tool_use")
        self.assertEqual([list(b)[0] for b in r["output"]["message"]["content"]],
                         ["reasoningContent", "text", "toolUse"])
        self.assertEqual(r["output"]["message"]["content"][0]["reasoningContent"]
                         ["reasoningText"]["signature"], "sig1")
        self.assertEqual(r["output"]["message"]["content"][2]["toolUse"]["input"],
                         {"city": "Tokyo"})
        self.assertGreater(r["usage"]["totalTokens"], 0)
        self.assertGreaterEqual(r["metrics"]["latencyMs"], 0)

    def test_converse_stream_events_decode(self) -> None:
        def call() -> list[dict[str, Any]]:
            resp = self.rt.converse_stream(
                modelId=MODEL, inferenceConfig={"maxTokens": 64},
                messages=[{"role": "user", "content": [{"text": "stream"}]}])
            return list(resp["stream"])

        t, out = self._in_thread(call)
        self._wait_pending()
        self._respond(content=[{"type": "text", "text": "streamed answer"},
                               {"type": "tool_use", "id": "t1", "name": "f",
                                "input": {"k": "v" * 40}}],
                      stop_reason="tool_use")
        t.join(20)
        self.assertNotIn("error", out, out.get("error"))
        events = out["value"]
        names = [list(e)[0] for e in events]
        self.assertEqual(names[0], "messageStart")
        self.assertEqual(names[-1], "metadata")
        text = "".join(e["contentBlockDelta"]["delta"].get("text", "")
                       for e in events if "contentBlockDelta" in e)
        self.assertEqual(text, "streamed answer")
        args = "".join(e["contentBlockDelta"]["delta"]["toolUse"]["input"]
                       for e in events if "contentBlockDelta" in e
                       and "toolUse" in e["contentBlockDelta"]["delta"])
        self.assertEqual(json.loads(args), {"k": "v" * 40})
        self.assertEqual([e["messageStop"]["stopReason"]
                          for e in events if "messageStop" in e], ["tool_use"])
        meta = [e["metadata"] for e in events if "metadata" in e][0]
        self.assertGreater(meta["usage"]["outputTokens"], 0)

    def test_invoke_model_with_response_stream(self) -> None:
        def call() -> list[dict[str, Any]]:
            resp = self.rt.invoke_model_with_response_stream(modelId=MODEL, body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}]}))
            return [json.loads(ev["chunk"]["bytes"]) for ev in resp["body"]]

        t, out = self._in_thread(call)
        self._wait_pending()
        self.http.post("/_control/auto", json={"text": "invoke streamed"})
        t.join(20)
        self.assertNotIn("error", out, out.get("error"))
        types = [e["type"] for e in out["value"]]
        self.assertEqual((types[0], types[-1]), ("message_start", "message_stop"))
        metrics = [e for e in out["value"]
                   if e["type"] == "message_stop"][0]["amazon-bedrock-invocationMetrics"]
        self.assertGreater(metrics["outputTokenCount"], 0)

    def test_mid_stream_exception_surfaces_as_event_stream_error(self) -> None:
        for op, kwargs, member in (
                ("converse_stream",
                 {"messages": [{"role": "user", "content": [{"text": "x"}]}]},
                 "throttlingException"),
                ("invoke_model_with_response_stream",
                 {"body": json.dumps({"anthropic_version": "bedrock-2023-05-31",
                                      "max_tokens": 8,
                                      "messages": [{"role": "user", "content": "x"}]})},
                 "throttlingException")):
            with self.subTest(op=op):
                self.http.post("/_control/clear")

                def call(op=op, kwargs=kwargs) -> list[Any]:
                    resp = getattr(self.rt, op)(modelId=MODEL, **kwargs)
                    seen: list[Any] = []
                    # NOT `list(...)`: that discards everything already decoded when the
                    # exception fires, so it could not tell a partial stream from an
                    # immediate failure. Keep the prefix, then let the error propagate.
                    try:
                        for ev in resp["stream" if op == "converse_stream" else "body"]:
                            seen.append(ev)
                    except BaseException as e:
                        e.seen = seen  # type: ignore[attr-defined]
                        raise
                    return seen

                t, out = self._in_thread(call)
                self._wait_pending()
                r = self.http.post("/_control/error", json={
                    "status": 429, "type": "ThrottlingException", "message": "slow down",
                    "after_events": 2, "content": [{"type": "text", "text": "partial"}]})
                self.assertEqual(r.status_code, 200, r.text)
                t.join(20)
                err = out.get("error")
                self.assertIsInstance(err, EventStreamError)
                self.assertIn(member, str(err))
                self.assertIn("slow down", str(err))
                # botocore delivered exactly the events requested before the exception,
                # and none of the terminal ones that would make the stream look complete.
                seen = getattr(err, "seen", [])
                self.assertEqual(len(seen), 2, seen)
                if op == "converse_stream":
                    self.assertEqual([next(iter(e)) for e in seen],
                                     ["messageStart", "contentBlockDelta"])
                    self.assertNotIn("messageStop", [next(iter(e)) for e in seen])
                else:
                    kinds = [json.loads(e["chunk"]["bytes"])["type"] for e in seen]
                    self.assertEqual(kinds, ["message_start", "content_block_start"])

    def test_error_mapping_raises_the_right_botocore_class(self) -> None:
        for status, etype, code in ((429, "ThrottlingException", "ThrottlingException"),
                                    (400, "ValidationException", "ValidationException"),
                                    (503, "ServiceUnavailableException", "ServiceUnavailableException")):
            with self.subTest(status=status):
                self.http.post("/_control/clear")
                t, out = self._in_thread(lambda: self.rt.converse(
                    modelId=MODEL,
                    messages=[{"role": "user", "content": [{"text": "x"}]}]))
                self._wait_pending()
                self.http.post("/_control/error",
                               json={"status": status, "type": etype, "message": "boom"})
                t.join(20)
                err = out.get("error")
                self.assertIsInstance(err, ClientError)
                self.assertEqual(err.response["Error"]["Code"], code)

    # ── S3 ──

    def test_s3_object_and_listing_round_trip(self) -> None:
        self.s3.create_bucket(Bucket="iop")
        self.s3.put_object(Bucket="iop", Key="a/1.txt", Body=b"one")
        self.s3.put_object(Bucket="iop", Key="a/2.txt", Body=b"two")
        self.s3.put_object(Bucket="iop", Key="top.txt", Body=b"three")
        self.assertEqual(self.s3.get_object(Bucket="iop", Key="a/1.txt")["Body"].read(), b"one")
        self.assertEqual(self.s3.head_object(Bucket="iop", Key="a/1.txt")["ContentLength"], 3)
        listing = self.s3.list_objects_v2(Bucket="iop", Delimiter="/")
        self.assertEqual([p["Prefix"] for p in listing.get("CommonPrefixes", [])], ["a/"])
        self.assertEqual([o["Key"] for o in listing["Contents"]], ["top.txt"])
        # both paginators must terminate (a stale token would loop forever)
        for api in ("list_objects_v2", "list_objects"):
            with self.subTest(api=api):
                keys: list[str] = []
                for page in self.s3.get_paginator(api).paginate(
                        Bucket="iop", PaginationConfig={"PageSize": 1}):
                    keys += [o["Key"] for o in page.get("Contents", [])]
                self.assertEqual(sorted(keys), ["a/1.txt", "a/2.txt", "top.txt"])
        with self.assertRaises(ClientError) as ctx:
            self.s3.get_object(Bucket="iop", Key="missing")
        self.assertEqual(ctx.exception.response["Error"]["Code"], "NoSuchKey")
        with self.assertRaises(ClientError) as ctx:
            self.s3.get_object(Bucket="nosuch", Key="k")
        self.assertEqual(ctx.exception.response["Error"]["Code"], "NoSuchBucket")

    def test_s3_managed_download_of_a_multipart_sized_object(self) -> None:
        """boto3 splits any download over `multipart_threshold` into concurrent ranged
        GETs and concatenates them: a server that ignores `Range` hands back a corrupt
        file and raises nothing."""
        from boto3.s3.transfer import TransferConfig
        self.s3.create_bucket(Bucket="iopbig")
        blob = os.urandom(12 * 1024 * 1024)  # over boto3's 8 MB default threshold
        self.s3.put_object(Bucket="iopbig", Key="big.bin", Body=blob)
        cfg = TransferConfig(multipart_threshold=8 * 1024 * 1024,
                             multipart_chunksize=5 * 1024 * 1024, max_concurrency=4)
        dest = os.path.join(self.tmp, "downloaded.bin")
        self.s3.download_file("iopbig", "big.bin", dest, Config=cfg)
        with open(dest, "rb") as fh:
            self.assertEqual(fh.read(), blob)
        # and the explicit range API behaves too
        part = self.s3.get_object(Bucket="iopbig", Key="big.bin", Range="bytes=5-14")
        self.assertEqual(part["Body"].read(), blob[5:15])
        self.assertEqual(part["ContentRange"], f"bytes 5-14/{len(blob)}")
        with self.assertRaises(ClientError) as ctx:
            self.s3.get_object(Bucket="iopbig", Key="big.bin",
                               Range=f"bytes={len(blob)}-{len(blob) + 10}")
        self.assertEqual(ctx.exception.response["Error"]["Code"], "InvalidRange")

    def test_s3_multipart_upload_is_refused_in_an_s3_envelope(self) -> None:
        """Not implemented — but it must fail as S3, not as a bare 405."""
        from boto3.s3.transfer import TransferConfig
        self.s3.create_bucket(Bucket="iopmp")
        with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as fh:
            fh.write(b"x" * (3 * 1024 * 1024))
            name = fh.name
        try:
            with self.assertRaises(Exception) as ctx:
                self.s3.upload_file(name, "iopmp", "big.bin", Config=TransferConfig(
                    multipart_threshold=1024, multipart_chunksize=1024 * 1024))
        finally:
            os.unlink(name)
        self.assertIn("NotImplemented", str(ctx.exception))

    def test_s3_operations_this_store_does_not_model_are_refused(self) -> None:
        """All of these arrive as a PUT on the object path. Writing their body — or their
        empty body — over the object would destroy it and report success."""
        self.s3.create_bucket(Bucket="iopns")
        self.s3.put_object(Bucket="iopns", Key="keep.txt", Body=b"original")
        calls = [
            ("copy_object", dict(Bucket="iopns", Key="keep.txt",
                                 CopySource={"Bucket": "iopns", "Key": "keep.txt"})),
            ("copy_object", dict(Bucket="iopns", Key="copy.txt",
                                 CopySource={"Bucket": "iopns", "Key": "keep.txt"})),
            ("put_object_tagging", dict(Bucket="iopns", Key="keep.txt",
                                        Tagging={"TagSet": [{"Key": "k", "Value": "v"}]})),
            ("put_object_acl", dict(Bucket="iopns", Key="keep.txt", ACL="private")),
            ("get_object_tagging", dict(Bucket="iopns", Key="keep.txt")),
            ("list_object_versions", dict(Bucket="iopns")),
            ("get_bucket_policy", dict(Bucket="iopns")),
            ("get_bucket_tagging", dict(Bucket="iopns")),
            ("put_bucket_versioning", dict(Bucket="iopns",
                                           VersioningConfiguration={"Status": "Enabled"})),
        ]
        for name, kwargs in calls:
            with self.subTest(op=name):
                with self.assertRaises(ClientError) as ctx:
                    getattr(self.s3, name)(**kwargs)
                self.assertEqual(ctx.exception.response["Error"]["Code"], "NotImplemented")
        # and nothing touched the object
        self.assertEqual(self.s3.get_object(Bucket="iopns", Key="keep.txt")["Body"].read(),
                         b"original")
        with self.assertRaises(ClientError) as ctx:
            self.s3.get_object(Bucket="iopns", Key="copy.txt")
        self.assertEqual(ctx.exception.response["Error"]["Code"], "NoSuchKey")

    def test_s3_conditional_writes_and_reads(self) -> None:
        self.s3.create_bucket(Bucket="iopcond")
        self.s3.put_object(Bucket="iopcond", Key="c.txt", Body=b"v1")
        etag = self.s3.head_object(Bucket="iopcond", Key="c.txt")["ETag"]
        # If-None-Match: * is how callers avoid clobbering — it must not overwrite
        with self.assertRaises(ClientError) as ctx:
            self.s3.put_object(Bucket="iopcond", Key="c.txt", Body=b"v2", IfNoneMatch="*")
        self.assertEqual(ctx.exception.response["Error"]["Code"], "PreconditionFailed")
        self.assertEqual(self.s3.get_object(Bucket="iopcond", Key="c.txt")["Body"].read(), b"v1")
        with self.assertRaises(ClientError) as ctx:
            self.s3.put_object(Bucket="iopcond", Key="c.txt", Body=b"v2", IfMatch='"deadbeef"')
        self.assertEqual(ctx.exception.response["Error"]["Code"], "PreconditionFailed")
        # a matching If-Match does go through, and a fresh key accepts If-None-Match: *
        self.s3.put_object(Bucket="iopcond", Key="c.txt", Body=b"v2", IfMatch=etag)
        self.s3.put_object(Bucket="iopcond", Key="new.txt", Body=b"n", IfNoneMatch="*")
        # conditional reads
        cur = self.s3.head_object(Bucket="iopcond", Key="c.txt")["ETag"]
        with self.assertRaises(ClientError) as ctx:
            self.s3.get_object(Bucket="iopcond", Key="c.txt", IfNoneMatch=cur)
        self.assertEqual(ctx.exception.response["ResponseMetadata"]["HTTPStatusCode"], 304)
        with self.assertRaises(ClientError) as ctx:
            self.s3.get_object(Bucket="iopcond", Key="c.txt", IfMatch='"deadbeef"')
        self.assertEqual(ctx.exception.response["Error"]["Code"], "PreconditionFailed")

    def test_s3_keys_needing_url_encoding_round_trip(self) -> None:
        """botocore asks for `encoding-type=url` on every listing and decodes the keys only
        when the response echoes it, so the two have to be switched together."""
        self.s3.create_bucket(Bucket="iopenc")
        keys = ["plain.txt", "with space.txt", "plus+sign.txt", "amp&and.txt",
                "hash#mark.txt", "pct%25.txt", "uni/\u65e5\u672c\u8a9e.txt"]
        for k in keys:
            self.s3.put_object(Bucket="iopenc", Key=k, Body=b"x")
        listed = [o["Key"] for o in self.s3.list_objects_v2(Bucket="iopenc")["Contents"]]
        self.assertEqual(sorted(listed), sorted(keys))
        for k in keys:
            self.assertEqual(self.s3.get_object(Bucket="iopenc", Key=k)["Body"].read(), b"x")
        # and paging over them terminates with every key seen exactly once
        seen: list[str] = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(
                Bucket="iopenc", PaginationConfig={"PageSize": 2}):
            seen += [o["Key"] for o in page.get("Contents", [])]
        self.assertEqual(sorted(seen), sorted(keys))

    def test_batch_create_is_idempotent_under_concurrency(self) -> None:
        """The name check and the registration must be one critical section: reading the
        input takes real time, and a client retrying inside that window must get a replay."""
        self.s3.create_bucket(Bucket="cin")
        self.s3.create_bucket(Bucket="cout")
        self.s3.put_object(Bucket="cin", Key="in.jsonl", Body=b"\n".join(
            json.dumps({"recordId": f"c{i}", "modelInput": {
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                "messages": [{"role": "user", "content": "q"}]}}).encode()
            for i in range(50)))
        kwargs = dict(jobName="concurrent", modelId=MODEL,
                      roleArn="arn:aws:iam::123456789012:role/BatchRole",
                      clientRequestToken="tok1",
                      inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://cin/in.jsonl"}},
                      outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://cout/o/"}})
        results: list[Any] = []
        threads = [self._in_thread(lambda: self.ctl.create_model_invocation_job(**kwargs))
                   for _ in range(4)]
        for t, out in threads:
            t.join(30)
            results.append(out)
        arns = {r["value"]["jobArn"] for r in results if "value" in r}
        errors = [r["error"] for r in results if "error" in r]
        self.assertEqual(len(arns), 1, results)      # one job, replayed
        self.assertEqual(errors, [], errors)
        self.assertEqual(len(self.http.get("/_control/bedrock_jobs").json()["jobs"]), 1)

    def test_a_refused_put_does_not_poison_the_connection(self) -> None:
        """botocore sends `Expect: 100-continue` on every PutObject with a body and, on a
        non-100 answer, keeps the connection WITHOUT sending the body. Unless the server
        asked for that body first, the client's next request on the same socket is eaten
        as the phantom body: a bare 400, a 60 s stall, or a wrong request executed."""
        self.s3.create_bucket(Bucket="iopexp")
        self.s3.put_object(Bucket="iopexp", Key="fine.txt", Body=b"ok")
        refused = [
            ("put_object", dict(Bucket="iopexp", Key="x/../y", Body=b"1" * 50)),
            ("put_object", dict(Bucket="iopexp", Key="trail/", Body=b"z" * 100_000)),
            ("put_object", dict(Bucket="iopexp", Key="k", Body=b"m" * 5000,
                                Metadata={"owner": "me"})),
            ("put_object", dict(Bucket="Bad_Name", Key="k", Body=b"q" * 5000)),
            ("put_object", dict(Bucket="iopexp", Key="fine.txt", Body=b"n" * 5000,
                                IfNoneMatch="*")),
        ]
        for name, kwargs in refused:
            with self.subTest(op=name, key=kwargs.get("Key")):
                with self.assertRaises(ClientError):
                    getattr(self.s3, name)(**kwargs)
                # the very next calls on the same client must be normal
                t0 = time.monotonic()
                self.assertEqual(self.s3.head_object(Bucket="iopexp", Key="fine.txt")["ContentLength"], 2)
                self.assertEqual([o["Key"] for o in self.s3.list_objects_v2(Bucket="iopexp")["Contents"]],
                                 ["fine.txt"])
                self.assertLess(time.monotonic() - t0, 2.0)
        # a refused request against a reserved API path, same story
        with self.assertRaises(ClientError):
            self.s3.put_object(Bucket="v1", Key="messages", Body=b"j" * 5000)
        self.assertEqual(self.s3.get_object(Bucket="iopexp", Key="fine.txt")["Body"].read(), b"ok")

    def test_conditional_put_has_one_winner_under_concurrency(self) -> None:
        self.s3.create_bucket(Bucket="iopwin")
        results: list[dict[str, Any]] = []
        threads = [self._in_thread(lambda i=i: self.s3.put_object(
            Bucket="iopwin", Key="k", Body=f"w{i}".encode() * 20_000, IfNoneMatch="*"))
            for i in range(16)]
        for t, out in threads:
            t.join(30)
            results.append(out)
        winners = [r for r in results if "value" in r]
        losers = [r["error"] for r in results if "error" in r]
        self.assertEqual(len(winners), 1, results)
        self.assertTrue(all(isinstance(e, ClientError)
                            and e.response["Error"]["Code"] == "PreconditionFailed"
                            for e in losers), losers)

    def test_v1_list_paginates_keys_containing_percent(self) -> None:
        """botocore decodes `NextMarker` before resending it as `marker`; a server that
        decoded it again skipped `p/q`."""
        self.s3.create_bucket(Bucket="ioppct")
        keys = ["p%2Fq", "p/q", "p/r", "a b", "a%20b", "a+b", "a+c"]
        for k in keys:
            self.s3.put_object(Bucket="ioppct", Key=k, Body=b"x")
        for api in ("list_objects", "list_objects_v2"):
            for size in (1, 2):
                with self.subTest(api=api, page_size=size):
                    seen: list[str] = []
                    for page in self.s3.get_paginator(api).paginate(
                            Bucket="ioppct", PaginationConfig={"PageSize": size}):
                        seen += [o["Key"] for o in page.get("Contents", [])]
                    self.assertEqual(seen, sorted(keys))
        self.assertEqual([o["Key"] for o in self.s3.list_objects(Bucket="ioppct", Marker="p%2Fq")["Contents"]],
                         ["p/q", "p/r"])
        self.assertEqual([o["Key"] for o in self.s3.list_objects_v2(Bucket="ioppct", StartAfter="p%2Fq")["Contents"]],
                         ["p/q", "p/r"])

    def test_bucket_listing_and_deletion(self) -> None:
        self.s3.create_bucket(Bucket="iopdel")
        self.assertIn("iopdel", [b["Name"] for b in self.s3.list_buckets()["Buckets"]])
        self.s3.put_object(Bucket="iopdel", Key="o", Body=b"x")
        with self.assertRaises(ClientError) as ctx:
            self.s3.delete_bucket(Bucket="iopdel")
        self.assertEqual(ctx.exception.response["Error"]["Code"], "BucketNotEmpty")
        self.s3.delete_object(Bucket="iopdel", Key="o")
        self.s3.delete_bucket(Bucket="iopdel")
        self.assertNotIn("iopdel", [b["Name"] for b in self.s3.list_buckets()["Buckets"]])

    def test_keys_with_newlines_are_refused_not_rewritten(self) -> None:
        """A trailing newline would otherwise route as the key WITHOUT it (Python's `$`
        matches before a final newline), rewriting `k` and defeating `If-None-Match: *`;
        a newline elsewhere would miss every route and poison the connection."""
        self.s3.create_bucket(Bucket="iopnl")
        self.s3.put_object(Bucket="iopnl", Key="k", Body=b"original")
        for key in ("k\n", "a\nb", "k\r", "c\x07d"):
            with self.subTest(key=repr(key)):
                with self.assertRaises(ClientError) as ctx:
                    self.s3.put_object(Bucket="iopnl", Key=key, Body=b"n" * 5000)
                self.assertEqual(ctx.exception.response["Error"]["Code"], "InvalidArgument")
                # same client, same pooled connection: still healthy
                self.assertEqual(self.s3.get_object(Bucket="iopnl", Key="k")["Body"].read(), b"original")
        self.assertEqual([o["Key"] for o in self.s3.list_objects_v2(Bucket="iopnl")["Contents"]], ["k"])
        with self.assertRaises(ClientError) as ctx:
            self.s3.delete_object(Bucket="iopnl", Key="k\n")
        self.assertEqual(ctx.exception.response["Error"]["Code"], "InvalidArgument")
        self.assertEqual(self.s3.head_object(Bucket="iopnl", Key="k")["ContentLength"], 8)

    def test_invoke_model_echoes_the_requested_tier_and_latency(self) -> None:
        body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                           "messages": [{"role": "user", "content": "x"}]})
        t, out = self._in_thread(lambda: self.rt.invoke_model(
            modelId=MODEL, body=body, serviceTier="priority", performanceConfigLatency="optimized"))
        pend = self._wait_pending()[0]
        self.assertEqual(pend["request"]["params"]["service_tier"], "auto")
        self.http.post("/_control/auto", json={"text": "ok"})
        t.join(20)
        self.assertNotIn("error", out, out.get("error"))
        self.assertEqual((out["value"]["serviceTier"], out["value"]["performanceConfigLatency"]),
                         ("priority", "optimized"))
        self.http.post("/_control/clear")
        t, out = self._in_thread(lambda: self.rt.invoke_model(modelId=MODEL, body=body))
        self._wait_pending()
        self.http.post("/_control/auto", json={"text": "ok"})
        t.join(20)
        self.assertEqual(out["value"]["serviceTier"], "default")

    def test_conditional_delete_via_boto3(self) -> None:
        self.s3.create_bucket(Bucket="iopcd")
        etag = self.s3.put_object(Bucket="iopcd", Key="k", Body=b"v1")["ETag"]
        with self.assertRaises(ClientError) as ctx:
            self.s3.delete_object(Bucket="iopcd", Key="k", IfMatch='"stale"')
        self.assertEqual(ctx.exception.response["Error"]["Code"], "PreconditionFailed")
        self.assertEqual(self.s3.get_object(Bucket="iopcd", Key="k")["Body"].read(), b"v1")
        self.s3.delete_object(Bucket="iopcd", Key="k", IfMatch=etag)
        with self.assertRaises(ClientError) as ctx:
            self.s3.head_object(Bucket="iopcd", Key="k")
        self.assertEqual(ctx.exception.response["ResponseMetadata"]["HTTPStatusCode"], 404)

    def test_structured_output_request_shape_reaches_the_responder(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        t, out = self._in_thread(lambda: self.rt.converse(
            modelId=MODEL, messages=[{"role": "user", "content": [{"text": "x"}]}],
            outputConfig={"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {
                "schema": json.dumps(schema), "name": "answer"}}}}))
        pend = self._wait_pending()[0]
        self.assertEqual(pend["request"]["params"]["output_config"]["format"],
                         {"type": "json_schema", "schema": schema, "name": "answer"})
        self.http.post("/_control/auto", json={"text": '{"answer": "ok"}'})
        t.join(20)
        self.assertNotIn("error", out, out.get("error"))

    def test_model_error_exception_shape_via_boto3(self) -> None:
        body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
                           "messages": [{"role": "user", "content": "x"}]})
        t, out = self._in_thread(lambda: self.rt.invoke_model(modelId=MODEL, body=body))
        self._wait_pending()
        self.http.post("/_control/error", json={"status": 424, "type": "ModelErrorException",
                                                "message": "upstream", "original_status": 429})
        t.join(20)
        err = out.get("error")
        self.assertIsInstance(err, ClientError)
        self.assertEqual(err.response["Error"]["Code"], "ModelErrorException")
        self.assertEqual(err.response["originalStatusCode"], 429)
        self.assertEqual(err.response["resourceName"], MODEL)

    def test_s3_upload_file_uses_aws_chunked(self) -> None:
        self.s3.create_bucket(Bucket="iopup")
        with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as fh:
            fh.write(b'{"a": 1}\n')
            name = fh.name
        try:
            self.s3.upload_file(name, "iopup", "in/data.jsonl")
        finally:
            os.unlink(name)
        self.assertEqual(self.s3.get_object(Bucket="iopup", Key="in/data.jsonl")["Body"].read(),
                         b'{"a": 1}\n')

    # ── batch inference ──

    def test_batch_job_lifecycle(self) -> None:
        self.s3.create_bucket(Bucket="bin")
        self.s3.create_bucket(Bucket="bout")
        records = "\n".join(json.dumps({"recordId": f"rec{i}", "modelInput": {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 32,
            "messages": [{"role": "user", "content": f"question {i}"}]}}) for i in (1, 2))
        self.s3.put_object(Bucket="bin", Key="jobs/input.jsonl", Body=records.encode())
        job = self.ctl.create_model_invocation_job(
            jobName="interop", modelId=MODEL,
            roleArn="arn:aws:iam::123456789012:role/BatchRole",
            inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://bin/jobs/input.jsonl"}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://bout/results/"}})
        arn = job["jobArn"]
        for item in self._wait_pending(2):
            self._respond(pending_id=item["pending_id"],
                          content=[{"type": "text",
                                    "text": "answer for " + item["request"]["record_id"]}])
        for _ in range(200):
            desc = self.ctl.get_model_invocation_job(jobIdentifier=arn)
            if desc["status"] in ("Completed", "Stopped", "Failed"):
                break
            time.sleep(0.05)
        self.assertEqual(desc["status"], "Completed")
        self.assertEqual((desc["totalRecordCount"], desc["successRecordCount"]), (2, 2))
        self.assertGreaterEqual(desc["submitTime"].timestamp(), 0)  # parsed as a datetime
        job_id = arn.rsplit("/", 1)[1]
        keys = sorted(o["Key"] for o in self.s3.list_objects_v2(
            Bucket="bout", Prefix=f"results/{job_id}/")["Contents"])
        self.assertEqual(keys, [f"results/{job_id}/input.jsonl.out",
                                f"results/{job_id}/manifest.json.out"])
        lines = [json.loads(x) for x in self.s3.get_object(
            Bucket="bout", Key=keys[0])["Body"].read().decode().splitlines()]
        self.assertEqual(sorted(x["recordId"] for x in lines), ["rec1", "rec2"])
        self.assertTrue(lines[0]["modelOutput"]["content"][0]["text"].startswith("answer for"))
        manifest = json.loads(self.s3.get_object(Bucket="bout", Key=keys[1])["Body"].read())
        self.assertEqual(manifest["successRecordCount"], 2)
        self.assertTrue(any(j["jobArn"] == arn for j in self.ctl.list_model_invocation_jobs(
            statusEquals="Completed")["invocationJobSummaries"]))
        # batch traffic is billed at the 50% batch rate
        hist = self.http.get("/_control/history").json()["history"][-1]
        self.assertTrue(hist["batch"])
        self.assertEqual(hist["cost"]["batch_discount"], 0.5)
        self.assertEqual(hist["usage"]["service_tier"], "batch")

    def test_batch_job_stop(self) -> None:
        self.s3.create_bucket(Bucket="sin")
        self.s3.create_bucket(Bucket="sout")
        records = "\n".join(json.dumps({"recordId": f"s{i}", "modelInput": {
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8,
            "messages": [{"role": "user", "content": "q"}]}}) for i in (1, 2))
        self.s3.put_object(Bucket="sin", Key="in.jsonl", Body=records.encode())
        job = self.ctl.create_model_invocation_job(
            jobName="interopstop", modelId=MODEL,
            roleArn="arn:aws:iam::123456789012:role/BatchRole",
            inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://sin/in.jsonl"}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://sout/out/"}})
        self._wait_pending(2)
        self.ctl.stop_model_invocation_job(jobIdentifier=job["jobArn"])
        for _ in range(200):
            desc = self.ctl.get_model_invocation_job(jobIdentifier=job["jobArn"])
            if desc["status"] in ("Stopped", "Completed", "Failed"):
                break
            time.sleep(0.05)
        self.assertEqual(desc["status"], "Stopped")
        self.assertEqual((desc["processedRecordCount"], desc["errorRecordCount"]), (0, 0))
        with self.assertRaises(ClientError) as ctx:
            self.ctl.get_model_invocation_job(jobIdentifier="aaaaaaaaaaaa")
        self.assertEqual(ctx.exception.response["Error"]["Code"], "ResourceNotFoundException")


if __name__ == "__main__":
    unittest.main()
