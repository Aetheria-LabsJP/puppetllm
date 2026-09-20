"""Minimal S3-compatible object store (for Bedrock batch inference against the fake).

Bedrock batch inference reads its input JSONL from S3 and writes its output there. The
fake ships a tiny, directory-backed S3 surface so a `boto3` S3 client pointed at the
same server (`endpoint_url=http://localhost:8765`, path-style addressing) can upload
inputs and download outputs, and the batch-inference emulation reads / writes the
same objects. Nothing here is authenticated (SigV4 is ignored), and only the calls
boto3 needs for that flow are implemented:

- `GET /`                    list_buckets
- `PUT /{bucket}`            create bucket        `HEAD /{bucket}`          bucket exists?
- `DELETE /{bucket}`         delete bucket (409 `BucketNotEmpty` unless empty)
- `PUT /{bucket}/{key}`      put_object           `GET /{bucket}/{key}`     get_object
- `HEAD /{bucket}/{key}`     head_object          `DELETE /{bucket}/{key}`  delete_object
- `GET /{bucket}[?list-type=2][&prefix=…]`  list_objects_v2 and the V1 list_objects, with
  `delimiter`, `max-keys` (clamped to 1000), `marker` / `continuation-token` /
  `start-after` and `encoding-type=url`
- `GET /{bucket}/{key}` with `Range:`       ranged get_object (206 / 416)
- `If-Match` / `If-None-Match` on reads (304) and writes (412); `If-Modified-Since` /
  `If-Unmodified-Since` on reads; `Content-MD5` verified on a write

Objects live under `PUPPETLLM_S3_ROOT` (default: a fresh temp directory per process) as
`<root>/<bucket>/<key>`. `aws-chunked` uploads (the checksum-trailer encoding recent
boto3 versions use for put_object) are decoded, and writes go through a rename from a
temp directory outside the bucket tree, so a reader never observes a half-written object.

`Range` and `encoding-type=url` are not optional in practice: boto3's managed download
splits any object over `multipart_threshold` (8 MB) into concurrent ranged GETs and
concatenates the bodies, and botocore asks for `encoding-type=url` on every listing.

Everything else is refused rather than approximated, in an S3 envelope (`501
NotImplemented`): multipart upload, `CopyObject`, tagging / ACL / metadata / SSE options on
a PUT, object and bucket sub-resources (`?tagging`, `?acl`, `?versions`, `?policy`, …),
`DeleteObjects`. `Content-Type` and the other plain entity headers on a PUT are accepted but
not stored (objects read back as `binary/octet-stream`).

Reserved top-level path segments (`model`, `v1`, `anthropic`, `_control`,
`model-invocation-job`, `model-invocation-jobs`, `docs`, `redoc`, `openapi.json`) are
routed to the other APIs, so do not name a bucket like that.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import itertools
import os
import shutil
import threading
import zlib
import re
import tempfile
import time
from email.utils import formatdate, parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote
from xml.sax.saxutils import escape

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

_TMP_SEQ = itertools.count()
# The HTTP handlers run their file I/O on the event-loop thread with no `await` in between,
# which makes stat -> precondition -> write atomic against each other — but the batch
# emulation writes its outputs from a worker thread (`asyncio.to_thread`). Every store
# operation, and every handler's read/check/write sequence, therefore takes this lock so
# the thread and the loop can never interleave inside one. Re-entrant: a handler holds it
# across a sequence that itself calls the store functions.
_STORE_LOCK = threading.RLock()
_ACCOUNT_ID = "123456789012"  # the owner of every bucket here (matches the Bedrock ARNs)
_TMP_DIR = ".tmp"  # under the root, never inside a bucket: no key can look like a temp file
_ROOT: Path | None = None
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_IPV4_RE = re.compile(r"^[0-9]{1,3}(\.[0-9]{1,3}){3}$")
RESERVED_SEGMENTS = frozenset((
    "model", "v1", "anthropic", "_control", "model-invocation-job", "model-invocation-jobs",
    "docs", "redoc", "openapi.json",
))


class S3Error(ValueError):
    """An S3 request the emulation refuses (surfaced as the matching XML error)."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def root() -> Path:
    """The store's root directory (created lazily; `PUPPETLLM_S3_ROOT` overrides)."""
    global _ROOT
    if _ROOT is None:
        env = os.environ.get("PUPPETLLM_S3_ROOT")
        _ROOT = Path(env) if env else Path(tempfile.mkdtemp(prefix="puppetllm-s3-"))
        _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT


def reset_root(path: str | None = None) -> None:
    """Point the store at a new directory (tests)."""
    global _ROOT
    _ROOT = Path(path) if path else None
    if _ROOT is not None:
        _ROOT.mkdir(parents=True, exist_ok=True)


_MAX_KEY_BYTES = 1024
_MAX_SEGMENT_BYTES = 255  # what the filesystem backing the store accepts
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _safe_key(key: str) -> str:
    # NOT stripped: real S3 would store `//a` under the key `/a`, and a directory-backed
    # store cannot. Silently rewriting it to `a` would hide the difference, so refuse.
    if not isinstance(key, str) or any(part in ("", ".", "..") for part in key.split("/")):
        raise S3Error("InvalidArgument", f"invalid object key: {key!r}")
    # XML 1.0 cannot represent C0/C1 control characters, so a key holding one would make
    # every later ListObjects response unparsable for the client. Real S3 accepts them and
    # relies on `encoding-type=url`; a directory-backed store has no use for them.
    if _CTRL_RE.search(key):
        raise S3Error("InvalidArgument",
                      "object key must not contain control characters (this emulation "
                      "renders keys into XML without encoding-type=url)")
    if "\ufffd" in key:
        # The ASGI server decodes an invalid percent-escape (`%ff`, `%c0%ae`) to U+FFFD
        # before we see it; storing that would be a silent rewrite of the caller's key.
        raise S3Error("InvalidArgument", "object key contains an invalid percent-encoding")
    if len(key.encode("utf-8")) > _MAX_KEY_BYTES:
        raise S3Error("KeyTooLongError", f"object key exceeds {_MAX_KEY_BYTES} bytes")
    # Directory-backed store: a single segment must still fit a filesystem name.
    if any(len(part.encode("utf-8")) > _MAX_SEGMENT_BYTES for part in key.split("/")):
        raise S3Error("InvalidArgument",
                      f"this emulation limits one key segment to {_MAX_SEGMENT_BYTES} bytes")
    return key


def _safe_bucket(bucket: str) -> str:
    """Validate a bucket name everywhere, not just on the HTTP surface — `parse_s3_uri`
    feeds the batch code, and `s3://../x` must not escape the store root."""
    if (not isinstance(bucket, str) or bucket in RESERVED_SEGMENTS
            or not _BUCKET_RE.match(bucket)
            # Same rules as the real service: no adjacent periods, no IPv4-looking name.
            or ".." in bucket or _IPV4_RE.match(bucket)):
        raise S3Error("InvalidBucketName", f"invalid bucket name: {bucket!r}")
    return bucket


def _obj_path(bucket: str, key: str) -> Path:
    p = root() / _safe_bucket(bucket) / _safe_key(key)
    # Belt and braces: the FULLY RESOLVED path (symlinks included) must stay under the root,
    # so neither a crafted key nor a symlink planted in the store can reach outside it.
    base = root().resolve()
    probe = p
    while True:
        try:
            resolved = probe.resolve()
            break
        except (OSError, RuntimeError):  # unreadable component, or a symlink loop
            raise S3Error("InvalidArgument", f"invalid object path: {key!r}") from None
    if not (resolved == base or base in resolved.parents):
        raise S3Error("InvalidArgument", f"object path escapes the store root: {key!r}")
    return p


# ── store API (used by the batch-inference emulation) ────────────────


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/prefix/key` → (bucket, key-or-prefix).

    The bucket is validated here too: callers outside the HTTP surface (the batch
    emulation) must not be able to reach outside the store with `s3://../x`.
    """
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise S3Error("InvalidArgument", f"not an s3:// URI: {uri!r}")
    # Parsed by hand: `?` and `#` are legal characters in an S3 key, so urlparse would
    # silently truncate the key at the first one.
    rest = uri[len("s3://"):]
    bucket, _sep, path = rest.partition("/")
    _safe_bucket(bucket)
    if any(part in (".", "..") for part in path.split("/")):
        raise S3Error("InvalidArgument", f"invalid s3:// path: {uri!r}")
    return bucket, path


def _inside_root(p: Path) -> bool:
    """Whether the FULLY RESOLVED path stays under the store root (symlink-proof)."""
    try:
        r, base = p.resolve(), root().resolve()
    except (OSError, RuntimeError):  # RuntimeError: a symlink loop
        return False
    return r == base or base in r.parents


def bucket_exists(bucket: str) -> bool:
    try:
        d = root() / _safe_bucket(bucket)
    except S3Error:
        return False
    # A bucket directory that is itself a symlink out of the root is not a bucket.
    with _STORE_LOCK:
        return d.is_dir() and _inside_root(d)


def create_bucket(bucket: str) -> None:
    with _STORE_LOCK:
        (root() / _safe_bucket(bucket)).mkdir(parents=True, exist_ok=True)


def put_object(bucket: str, key: str, data: bytes, *, require_bucket: bool = True) -> str:
    """Store an object; returns the ETag (quoted MD5). Real S3 refuses a write to a bucket
    that does not exist, so the check is on by default."""
    p = _obj_path(bucket, key)
    with _STORE_LOCK:
        return _put_object_locked(bucket, key, p, data, require_bucket)


def _put_object_locked(bucket: str, key: str, p: Path, data: bytes, require_bucket: bool) -> str:
    if require_bucket and not bucket_exists(bucket):
        raise S3Error("NoSuchBucket", "The specified bucket does not exist", 404)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a concurrent GET or listing must never observe a half-written
        # object, or see a size and an ETag taken from two different moments. The temp file
        # lives in a sibling directory of the buckets, so no listing can ever see it and no
        # object key can be mistaken for one.
        tmp_dir = root() / _TMP_DIR
        tmp_dir.mkdir(exist_ok=True)
        tmp = tmp_dir / f"{os.getpid()}.{next(_TMP_SEQ)}"
        try:
            tmp.write_bytes(data)
            os.replace(tmp, p)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        etag = _etag(data)
        try:
            st = p.stat()
            # Inodes are reused at once and a coarse-mtime kernel can give two same-size
            # writes the same mtime_ns: seed the cache with the value we KNOW is right.
            _ETAG_CACHE[(str(p), st.st_ino, st.st_size, st.st_mtime_ns)] = etag
        except OSError:
            pass
    except (NotADirectoryError, IsADirectoryError, FileExistsError) as e:
        # Real S3 is a flat key space, so `p/q` and `p` can coexist; a directory-backed
        # store cannot represent that. Report it as a 4xx instead of a 500.
        raise S3Error("InvalidArgument",
                      f"key {key!r} collides with an existing key in this emulation "
                      f"(a directory-backed store cannot hold both): {type(e).__name__}") from None
    except OSError as e:
        raise S3Error("InvalidArgument", f"could not store {key!r}: {e.strerror or e}") from None
    return etag


def get_object(bucket: str, key: str) -> bytes | None:
    p = _obj_path(bucket, key)
    with _STORE_LOCK:
        try:
            return p.read_bytes() if p.is_file() else None
        except OSError:
            return None


def delete_object(bucket: str, key: str) -> None:
    p = _obj_path(bucket, key)
    with _STORE_LOCK:
        _delete_object_locked(bucket, p)


def _delete_object_locked(bucket: str, p: Path) -> None:
    try:
        if p.is_file():
            p.unlink()
    except OSError:
        return
    # Real S3 has no directories: once `dir/child` is gone, `dir` is a free key again. Prune
    # the empty parents so a directory-backed store agrees (a non-empty one stops the loop).
    bucket_dir = root() / bucket
    parent = p.parent
    while parent != bucket_dir and bucket_dir in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def list_objects(bucket: str, prefix: str = "", *, etag: bool = True) -> list[dict[str, Any]]:
    """Objects under `prefix` (recursive), sorted by key.

    `etag=False` skips the MD5: an internal caller (the batch emulation scanning an input
    prefix) is about to read every object anyway and has no use for a hash of each one.
    """
    try:
        base = root() / _safe_bucket(bucket)
    except S3Error:
        return []
    if not base.is_dir():
        return []
    # The object paths are confined by `_obj_path`; the bucket directory itself must be
    # too, or a symlinked bucket would let a listing walk (and hash) files outside the root.
    if not _inside_root(base):
        return []
    with _STORE_LOCK:
        return _list_objects_locked(base, prefix, etag)


def _list_objects_locked(base: Path, prefix: str, etag: bool) -> list[dict[str, Any]]:
    out = []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue  # a symlink could point outside the store; never list or hash it
        key = p.relative_to(base).as_posix()
        if key.startswith(prefix):
            st = p.stat()
            out.append({"key": key, "size": st.st_size, "mtime": st.st_mtime,
                        "etag": _file_etag(p, st) if etag else ""})
    # `rglob` yields in path-part order (`a/b/c` before `a b/c`); S3 orders by key bytes.
    out.sort(key=lambda o: o["key"])
    return out


def _etag(data: bytes) -> str:
    return '"' + hashlib.md5(data).hexdigest() + '"'

_ETAG_CACHE: dict[tuple[str, int, int, int], str] = {}


def _file_etag(p: Path, st: os.stat_result | None = None) -> str:
    """MD5 of a file, streamed so a listing never holds a whole object in memory, and
    memoized on (path, size, mtime) so listing a bucket does not re-hash every object."""
    st = st or p.stat()
    ck = (str(p), st.st_ino, st.st_size, st.st_mtime_ns)
    hit = _ETAG_CACHE.get(ck)
    if hit is not None:
        return hit
    h = hashlib.md5()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    etag = chr(34) + h.hexdigest() + chr(34)
    if len(_ETAG_CACHE) > 4096:
        _ETAG_CACHE.clear()
    _ETAG_CACHE[ck] = etag
    return etag



def stat_object(bucket: str, key: str) -> os.stat_result | None:
    """Size / mtime without reading the object (what HEAD and a ranged GET need)."""
    p = _obj_path(bucket, key)
    with _STORE_LOCK:
        try:
            return p.stat() if p.is_file() and not p.is_symlink() else None
        except OSError:
            return None


def read_range(bucket: str, key: str, start: int, length: int) -> bytes:
    """Read `length` bytes from `start` without materializing the whole object."""
    p = _obj_path(bucket, key)
    with _STORE_LOCK, p.open("rb") as fh:
        fh.seek(start)
        return fh.read(length)


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """`Range: bytes=a-b` / `bytes=a-` / `bytes=-n` -> (first, last) inclusive.

    Returns None when the header is not a byte range this store honours (RFC 9110 says an
    unsatisfiable-syntax range is ignored); raises S3Error(416) when it is well formed but
    starts past the end, which is what real S3 answers with `InvalidRange`.
    """
    spec = header.strip()
    if not spec.lower().startswith("bytes="):
        return None
    spec = spec[len("bytes="):].strip()
    if "," in spec:  # multipart/byteranges is not emulated; serve the whole object
        return None
    first_s, _sep, last_s = spec.partition("-")
    if not _sep:
        return None

    def _int(raw: str) -> int | None:
        try:
            return int(raw)
        except ValueError:
            return None

    # Parse first, judge after: S3Error is a ValueError, so raising it inside a
    # `try ... except ValueError` would silently turn a 416 into "serve the whole object".
    if not first_s:
        if not last_s:
            return None
        suffix = _int(last_s)
        if suffix is None:
            return None
        if suffix <= 0:
            raise S3Error("InvalidRange", "The requested range is not satisfiable", 416)
        first, last = max(0, size - suffix), size - 1
    else:
        first = _int(first_s)
        last = _int(last_s) if last_s else size - 1
        if first is None or last is None:
            return None
    if first < 0:
        return None
    if size == 0 or first >= size:
        raise S3Error("InvalidRange", "The requested range is not satisfiable", 416)
    if last < first:
        return None  # syntactically invalid (`bytes=10-5`): RFC 9110 says ignore it
    return first, min(last, size - 1)


# ── aws-chunked decoding (boto3 put_object with checksum trailers) ──


def decode_aws_chunked(body: bytes) -> tuple[bytes, dict[str, str]]:
    """Decode a `Content-Encoding: aws-chunked` body (hex chunk sizes, optional trailers).

    Returns (payload, trailers) — the trailers carry the request checksum boto3 computes
    for file uploads. Raises S3Error on a malformed body instead of a ValueError → 500.
    """
    out = bytearray()
    trailers: dict[str, str] = {}
    off = 0
    n = len(body)
    terminated = False
    while off < n:
        nl = body.find(b"\r\n", off)
        if nl < 0:
            raise S3Error("InvalidRequest", "malformed aws-chunked body: missing chunk header")
        size_field = body[off:nl].split(b";", 1)[0].strip()
        try:
            size = int(size_field, 16)
        except ValueError:
            raise S3Error("InvalidRequest",
                          f"malformed aws-chunked body: bad chunk size {size_field!r}") from None
        off = nl + 2
        if size == 0:
            # Only trailers may follow, terminated by a blank line.
            tail = body[off:]
            if not tail.endswith(b"\r\n"):
                # `0\r\n` alone is a body cut off before its final CRLF (`0\r\n\r\n`)
                raise S3Error("InvalidRequest",
                              "malformed aws-chunked body: unterminated trailers")
            for line in tail.split(b"\r\n"):
                name, sep, value = line.partition(b":")
                if sep:
                    trailers[name.strip().decode("latin-1").lower()] = value.strip().decode("latin-1")
            terminated = True
            break
        if size < 0 or off + size + 2 > n:
            raise S3Error("InvalidRequest",
                          "malformed aws-chunked body: chunk size exceeds the body")
        if body[off + size:off + size + 2] != b"\r\n":
            raise S3Error("InvalidRequest",
                          "malformed aws-chunked body: missing CRLF after chunk data")
        out += body[off:off + size]
        off += size + 2  # skip data + CRLF
    if not terminated:
        # A body that simply ends after a data chunk was cut off (or never finished):
        # storing it would silently keep a truncated object.
        raise S3Error("InvalidRequest",
                      "malformed aws-chunked body: missing the terminating zero-size chunk")
    return bytes(out), trailers


def verify_checksums(data: bytes, headers: dict[str, str]) -> None:
    """`Content-MD5` and the `x-amz-checksum-*` request checksums (header or trailer form).

    botocore attaches a CRC32 to every `put_object` by default; real S3 refuses a mismatch,
    so a body corrupted in transit must not be stored quietly. The algorithms Python ships
    are verified; CRC32C / CRC64NVME are accepted unverified.
    """
    def _b64(raw: str, what: str) -> bytes:
        try:
            return base64.b64decode(raw, validate=True)
        except (ValueError, binascii.Error):
            raise S3Error("InvalidDigest", f"The {what} you specified was invalid") from None

    md5 = headers.get("content-md5")
    if md5 is not None and _b64(md5, "Content-MD5") != hashlib.md5(data).digest():
        raise S3Error("BadDigest", "The Content-MD5 you specified did not match what we received")
    for name, want in (("x-amz-checksum-crc32", zlib.crc32(data).to_bytes(4, "big")),
                       ("x-amz-checksum-sha1", hashlib.sha1(data).digest()),
                       ("x-amz-checksum-sha256", hashlib.sha256(data).digest())):
        sent = headers.get(name)
        if sent is not None and _b64(sent, name) != want:
            raise S3Error("BadDigest", f"The {name} you specified did not match what we received")

# ── HTTP surface ──────────────────────────────────────────────────────


def _xml(status: int, body: str, headers: dict[str, str] | None = None) -> Response:
    return Response(content='<?xml version="1.0" encoding="UTF-8"?>\n' + body,
                    status_code=status, media_type="application/xml", headers=headers)


def _s3_error(e: S3Error, **extra: str) -> Response:
    return _error(e.status, e.code, str(e), **extra)


def _error(status: int, code: str, message: str, **extra: str) -> Response:
    fields = "".join(f"<{k}>{escape(v)}</{k}>" for k, v in extra.items())
    return _xml(status, f"<Error><Code>{code}</Code><Message>{escape(message)}</Message>"
                        f"{fields}<RequestId>puppetllm</RequestId></Error>")


def _http_date(ts: float) -> str:
    return formatdate(ts, usegmt=True)


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(ts))


def _api_routes(app: Any) -> list[tuple[Any, frozenset[str]]]:
    """(path regex, methods) for every non-S3 route, so the catch-all below can answer a
    wrong method on a real API path with 405 + `Allow` instead of a misleading 404.

    `app.routes` holds the app's own routes plus one wrapper per `include_router`; the
    wrapper exposes the router it wrapped as `original_router`.
    """
    cached = getattr(app, "_puppetllm_api_routes", None)
    if cached is None:
        found: list[tuple[Any, frozenset[str]]] = []
        stack, seen = list(getattr(app, "routes", [])), set()
        while stack:
            r = stack.pop()
            sub = getattr(r, "original_router", None)
            if sub is not None:
                if id(sub) not in seen:
                    seen.add(id(sub))
                    stack.extend(getattr(sub, "routes", []) or [])
                continue
            regex, methods = getattr(r, "path_regex", None), getattr(r, "methods", None)
            if regex is None or not methods:
                continue
            if str(getattr(r, "path", "")).startswith("/{bucket}"):
                continue  # this router's own catch-alls
            found.append((regex, frozenset(methods)))
        # Cached on the app, not on the module: the tests reload `fake_server`, and a
        # process-wide cache would answer for whichever app instance came first.
        cached = found
        try:
            app._puppetllm_api_routes = found
        except AttributeError:  # pragma: no cover - a non-FastAPI app object
            pass
    return cached


# Query keys each surface actually implements. Every other key names an S3 operation this
# store does not model (`?tagging`, `?acl`, `?versions`, `?uploads`, …); answering it with a
# 200 and the wrong body is far worse than refusing it, because the SDK reports success.
# Query-string auth (presigned URLs) is ignored like header auth is.
_AUTH_PARAMS = frozenset((
    "AWSAccessKeyId", "Signature", "Expires", "X-Amz-Algorithm", "X-Amz-Credential",
    "X-Amz-Date", "X-Amz-Expires", "X-Amz-SignedHeaders", "X-Amz-Signature",
    "X-Amz-Security-Token", "x-id", "expected-bucket-owner"))
_BUCKET_GET_PARAMS = _AUTH_PARAMS | frozenset((
    "list-type", "prefix", "delimiter", "max-keys", "marker", "continuation-token",
    "start-after", "encoding-type", "fetch-owner"))
_OBJECT_PARAMS = _AUTH_PARAMS
# PutObject options this store cannot honour. Accepting them would mean `head_object`
# later contradicts what the caller wrote (empty Metadata, no tags) — refuse instead.
_UNMODELLED_PUT_HEADERS = ("x-amz-tagging", "x-amz-meta-", "x-amz-server-side-encryption",
                           "x-amz-acl", "x-amz-grant-", "x-amz-object-lock-",
                           "x-amz-website-redirect-location", "x-amz-object-ownership",
                           "x-amz-bucket-object-lock-enabled")


def _wrong_owner(request: Request, bucket: str) -> Response | None:
    """`ExpectedBucketOwner` is a confused-deputy guard; every bucket here belongs to the
    one account the Bedrock ARNs name, so any other value is the real service's 403."""
    supplied = [v for v in (request.query_params.get("expected-bucket-owner"),
                            request.headers.get("x-amz-expected-bucket-owner")) if v is not None]
    for owner in supplied:  # both forms are checked: a wrong one anywhere is a 403
        if owner != _ACCOUNT_ID:
            return _error(403, "AccessDenied",
                          f"the bucket is owned by account {_ACCOUNT_ID}, not {owner}",
                          BucketName=bucket)
    return None


def _unsupported_op(request: Request, allowed: frozenset[str], bucket: str,
                    key: str | None = None) -> Response | None:
    """Refuse an operation this emulation does not model, in an S3 envelope."""
    extra = sorted(k for k in request.query_params if k not in allowed)
    if request.method in ("PUT", "POST"):
        if request.headers.get("x-amz-copy-source"):
            extra.append("x-amz-copy-source")
        extra += sorted(h for h in request.headers
                        if any(h.startswith(pfx) for pfx in _UNMODELLED_PUT_HEADERS))
        if request.headers.get("x-amz-storage-class", "STANDARD").upper() != "STANDARD":
            extra.append("x-amz-storage-class")  # everything here is STANDARD; do not lie
    if not extra:
        return None
    fields = {"BucketName": bucket}
    if key is not None:
        fields["Key"] = key
    return _error(501, "NotImplemented",
                  f"this emulation does not implement {', '.join(extra)} — it models only "
                  f"plain object get/put/head/delete, bucket create/head/delete and listing",
                  **fields)


def _precondition(request: Request, etag: str | None, mtime: float | None) -> Response | None:
    """RFC 9110 conditional headers. Ignoring these is not harmless: `If-None-Match: *` is
    exactly the primitive callers use to avoid clobbering an object, and a store that
    answers 200 overwrites the thing they were protecting.

    Deliberately synchronous: the caller evaluates it and performs the write with no
    `await` in between, so concurrent conditional writers cannot all pass the check.
    Real S3 honours only the `*-Match` forms on a write; the `*-Since` forms are read-side.
    """
    h = request.headers
    is_read = request.method in ("GET", "HEAD")
    exists = etag is not None

    def _match(header: str) -> bool:
        raw = h[header]
        if raw.strip() == "*":
            return exists
        wanted = set()
        for t in raw.split(","):
            t = t.strip()
            if t.startswith("W/"):
                if not is_read:
                    continue  # RFC 9110 §13.1.1: writes use the strong comparison
                t = t[2:]
            wanted.add(t.strip(chr(34)))
        return exists and (etag or "").strip(chr(34)) in wanted

    if "if-match" in h and not _match("if-match"):
        return _error(412, "PreconditionFailed", "At least one of the pre-conditions you "
                      "specified did not hold", Condition="If-Match")
    if "if-none-match" in h and _match("if-none-match"):
        if is_read:
            return Response(status_code=304, headers={"ETag": etag or ""})
        return _error(412, "PreconditionFailed", "At least one of the pre-conditions you "
                      "specified did not hold", Condition="If-None-Match")
    if is_read and exists and mtime is not None:
        # RFC 9110 §13.1.3 / §13.1.4: a `*-Since` header is evaluated only when the
        # corresponding `*-Match` header is absent (the entity tag is the stronger test).
        since = None if "if-none-match" in h else h.get("if-modified-since")
        if since is not None and _parse_http_date(since) is not None \
                and int(mtime) <= _parse_http_date(since):
            return Response(status_code=304, headers={"ETag": etag or ""})
        unmod = None if "if-match" in h else h.get("if-unmodified-since")
        if unmod is not None and _parse_http_date(unmod) is not None \
                and int(mtime) > _parse_http_date(unmod):
            return _error(412, "PreconditionFailed", "At least one of the pre-conditions "
                          "you specified did not hold", Condition="If-Unmodified-Since")
    return None


def _parse_http_date(raw: str) -> int | None:
    try:
        return int(parsedate_to_datetime(raw).timestamp())
    except (TypeError, ValueError):
        return None


async def _read_body(request: Request) -> bytes:
    """Read the body FIRST, before any refusal can be sent.

    botocore sends `Expect: 100-continue` on every PutObject with a body and, on any
    non-100 status, keeps the connection in its pool WITHOUT sending the body. If the
    server had not asked for it, uvicorn is still waiting for `Content-Length` bytes on
    that socket, so the client's next request is swallowed as the phantom body: a bare
    400, a 60 s stall, or (with a crafted length) a different request executed. Asking for
    the body up front sends the `100 Continue`, so the exchange completes either way.
    """
    if request.method in ("PUT", "POST", "PATCH"):
        return await request.body()
    return b""


class ControlCharGuard:
    """Pure-ASGI middleware, installed OUTSIDE the router.

    Starlette's `{key:path}` convertor is `.*` with a `$` appended, and in Python `$` also
    matches just before a trailing newline: `PUT /b/k\n` would route as key `k` (a silent
    rewrite that also defeats `If-None-Match: *` on `k`), while a newline anywhere else
    matches no route and gets Starlette's own 404 — sent without reading the body, which
    poisons the keep-alive connection for botocore's `Expect: 100-continue` PUTs. Both are
    settled here, before routing, with the body drained and an S3 envelope.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http" or not (_CTRL_RE.search(path) or path.startswith("//")):
            await self.app(scope, receive, send)
            return
        while True:  # drain the body so the connection stays in sync
            msg = await receive()
            if msg.get("type") != "http.request" or not msg.get("more_body"):
                break
        resp = _error(400, "InvalidArgument",
                      "the request path contains a control character or an empty bucket "
                      "segment; object keys must not" if _CTRL_RE.search(path) else
                      "the request path has an empty bucket segment")
        await resp(scope, receive, send)


def build_router() -> APIRouter:
    router = APIRouter()

    def _not_s3(bucket: str, request: Request) -> Response | None:
        """A reserved first segment means the caller mistyped an API path (or used the
        wrong method) — answering with an S3 `InvalidBucketName` envelope would be actively
        misleading. Because this catch-all matches every method, a wrong method on a real
        API path lands here instead of the router's own 405, so reproduce that 405.
        """
        if bucket not in RESERVED_SEGMENTS:
            return None
        path = request.url.path
        allowed: set[str] = set()
        for regex, methods in _api_routes(request.app):
            if regex.match(path):
                allowed |= set(methods)
        if allowed and request.method not in allowed:
            if "GET" in allowed:
                allowed.add("HEAD")
            return JSONResponse({"detail": "Method Not Allowed"}, status_code=405,
                                headers={"Allow": ", ".join(sorted(allowed))})
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    @router.api_route("/", methods=["GET", "PUT", "POST", "DELETE", "HEAD", "PATCH"])
    async def list_buckets(request: Request) -> Any:
        await _read_body(request)  # before ANY early return — see _read_body
        if request.method != "GET":
            return _error(501, "NotImplemented",
                          f"{request.method} on the service root is not implemented; "
                          f"GET / lists the buckets")
        bad = _unsupported_op(request, _AUTH_PARAMS, "")
        if bad is not None:
            return bad
        entries = []
        for d in sorted(root().iterdir()):
            if not d.is_dir() or d.name.startswith(".") or not _inside_root(d):
                continue  # `.tmp` (writes in progress) and symlinks out of the root are not buckets
            try:
                _safe_bucket(d.name)
            except S3Error:
                continue
            entries.append(f"<Bucket><Name>{escape(d.name)}</Name>"
                           f"<CreationDate>{_iso(d.stat().st_mtime)}</CreationDate></Bucket>")
        return _xml(200, (
            '<ListAllMyBucketsResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            "<Owner><ID>puppetllm</ID><DisplayName>puppetllm</DisplayName></Owner>"
            f"<Buckets>{''.join(entries)}</Buckets></ListAllMyBucketsResult>"))

    @router.api_route("/{bucket}", methods=["PUT", "HEAD", "GET", "POST", "DELETE", "PATCH"])
    async def bucket_ops(bucket: str, request: Request) -> Any:
        await _read_body(request)  # before ANY early return — see _read_body
        miss = _not_s3(bucket, request)
        if miss is not None:
            return miss
        try:
            _safe_bucket(bucket)
        except S3Error as e:
            return _s3_error(e, BucketName=bucket)
        if request.method in ("POST", "PATCH"):
            return _error(501, "NotImplemented",
                          "This emulation implements only PUT/GET/HEAD/DELETE on an object "
                          "and GET/PUT/HEAD/DELETE on a bucket",
                          BucketName=bucket)
        bad = (_unsupported_op(request, _BUCKET_GET_PARAMS if request.method == "GET"
                               else _AUTH_PARAMS, bucket) or _wrong_owner(request, bucket))
        if bad is not None:
            return bad
        if request.method == "PUT":
            create_bucket(bucket)
            return Response(status_code=200, headers={"Location": f"/{bucket}"})
        if not bucket_exists(bucket):
            return (Response(status_code=404) if request.method == "HEAD"
                    else _error(404, "NoSuchBucket", "The specified bucket does not exist",
                                BucketName=bucket))
        if request.method == "HEAD":
            return Response(status_code=200)
        if request.method == "DELETE":
            return await asyncio.to_thread(_bucket_delete_locked, bucket)
        return await asyncio.to_thread(_bucket_list, request, bucket)

    def _bucket_delete_locked(bucket: str) -> Any:
        with _STORE_LOCK:  # the emptiness check and the removal are one step
            if list_objects(bucket, etag=False):
                return _error(409, "BucketNotEmpty",
                              "The bucket you tried to delete is not empty",
                              BucketName=bucket)
            try:
                shutil.rmtree(root() / bucket)
            except OSError as e:
                return _error(500, "InternalError",
                              f"could not delete bucket: {e.strerror or type(e).__name__}",
                              BucketName=bucket)
        return Response(status_code=204)

    def _bucket_list(request: Request, bucket: str) -> Any:
        qp = request.query_params
        prefix = qp.get("prefix", "")
        raw_max = qp.get("max-keys")
        if raw_max in (None, ""):
            max_keys = 1000
        else:
            try:
                max_keys = int(raw_max)
            except ValueError:
                return _error(400, "InvalidArgument",
                              "Provided max-keys not an integer or within integer range",
                              ArgumentName="max-keys", ArgumentValue=str(raw_max))
            if max_keys < 0:
                return _error(400, "InvalidArgument",
                              "Argument max-keys must be an integer between 0 and 2147483647",
                              ArgumentName="max-keys", ArgumentValue=str(raw_max))
            # The service never returns more than 1000 keys but does not reject a larger
            # request, so clamp rather than 400 — a caller written against a real bucket
            # with MaxKeys=2000 must keep working here.
            max_keys = min(max_keys, 1000)
        delimiter = qp.get("delimiter", "")
        enc = qp.get("encoding-type")
        if enc is not None and enc != "url":
            return _error(400, "InvalidArgument", "Invalid Encoding Method specified in Request",
                          ArgumentName="encoding-type", ArgumentValue=str(enc))
        list_type = qp.get("list-type")
        if list_type is not None and list_type != "2":
            return _error(400, "InvalidArgument", "Invalid list type specified in Request",
                          ArgumentName="list-type", ArgumentValue=str(list_type))
        # botocore URL-decodes the keys it reads back only when the response echoes
        # `EncodingType`, so the two must be switched together.
        def _xml_key(value: str) -> str:
            return escape(quote(value, safe="") if enc == "url" else value)
        v2 = list_type == "2"
        objs = list_objects(bucket, prefix)
        if v2:
            # `NextContinuationToken` is the one value the SDK sends back VERBATIM, so it is
            # the only one to undo our own percent-encoding on. botocore decodes `NextMarker`
            # before resending it as `marker`, and never encodes a user-supplied `Marker` or
            # `StartAfter`; unquoting those would move a key containing `%XX` to the wrong
            # position and silently skip or repeat keys on the V1 paginator.
            token_in = qp.get("continuation-token") or ""
            start_after = (unquote(token_in) if (enc == "url" and token_in)
                           else token_in or qp.get("start-after") or "")
        else:
            start_after = qp.get("marker") or ""
        if start_after:
            objs = [o for o in objs if o["key"] > start_after]
        # Delimiter grouping: keys sharing a prefix up to the first delimiter after `prefix`
        # collapse into one CommonPrefixes entry, exactly as ListObjectsV2 does.
        flat: list[dict[str, Any]] = []
        common: list[str] = []
        for o in objs:
            rest_key = o["key"][len(prefix):]
            idx = rest_key.find(delimiter) if delimiter else -1
            if idx >= 0:
                cp = prefix + rest_key[:idx + len(delimiter)]
                if cp not in common:
                    common.append(cp)
            else:
                flat.append(o)
        entries: list[tuple[str, dict[str, Any] | None]] = (
            sorted([(o["key"], o) for o in flat] + [(c, None) for c in common]))
        page, rest = entries[:max_keys], entries[max_keys:]
        contents = "".join(
            f"<Contents><Key>{_xml_key(o['key'])}</Key><LastModified>{_iso(o['mtime'])}</LastModified>"
            f"<ETag>{escape(o['etag'])}</ETag><Size>{o['size']}</Size>"
            f"<StorageClass>STANDARD</StorageClass></Contents>"
            for _k, o in page if o is not None)
        prefixes = "".join(f"<CommonPrefixes><Prefix>{_xml_key(k)}</Prefix></CommonPrefixes>"
                           for k, o in page if o is None)
        # The continuation token must sort PAST everything the page covered: for a
        # CommonPrefixes entry that is the largest key inside the group, not the prefix
        # itself (which would re-match the group and loop the paginator forever).
        token = ""
        if rest and page:
            last_name, last_obj = page[-1]
            token = (last_name if last_obj is not None else
                     max(o["key"] for o in objs if o["key"].startswith(last_name)))
        truncated = bool(rest and max_keys)
        # V2 pages with a continuation token, V1 (list_objects) with a marker — emitting a
        # V2 body for a V1 request loops botocore's paginator forever.
        if v2:
            nxt = f"<NextContinuationToken>{_xml_key(token)}</NextContinuationToken>" if token else ""
            echo = f"<KeyCount>{len(page)}</KeyCount>"
            if qp.get("continuation-token"):
                # echoed as received: the client sends it back exactly as we emitted it,
                # so encoding it again would double-encode
                echo += f"<ContinuationToken>{escape(qp['continuation-token'])}</ContinuationToken>"
            if qp.get("start-after"):
                echo += f"<StartAfter>{_xml_key(qp['start-after'])}</StartAfter>"
        else:
            nxt = f"<NextMarker>{_xml_key(token)}</NextMarker>" if token else ""
            echo = f"<Marker>{_xml_key(start_after)}</Marker>"
        if delimiter:
            echo += f"<Delimiter>{_xml_key(delimiter)}</Delimiter>"
        if enc:
            echo += f"<EncodingType>{escape(enc)}</EncodingType>"
        return _xml(200, (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Name>{escape(bucket)}</Name><Prefix>{_xml_key(prefix)}</Prefix>"
            f"<MaxKeys>{max_keys}</MaxKeys>{echo}"
            f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>{nxt}"
            f"{contents}{prefixes}</ListBucketResult>"))

    @router.api_route("/{bucket}/{key:path}", methods=["PUT", "GET", "HEAD", "DELETE", "POST", "PATCH"])
    async def object_ops(bucket: str, key: str, request: Request) -> Any:
        # NOTE: `key` arrives already percent-decoded by Starlette — do not decode again,
        # or an object literally named `a%2Fb` would be stored under `a/b`.
        body = await _read_body(request)  # before ANY early return — see _read_body
        miss = _not_s3(bucket, request)
        if miss is not None:
            return miss
        if key == "" and request.method in ("GET", "HEAD"):
            return await bucket_ops(bucket, request)  # `GET /bucket/` lists the bucket
        if request.method in ("POST", "PATCH"):
            # CreateMultipartUpload (`?uploads`) is what boto3's managed transfer reaches for
            # above `multipart_threshold`; say so in an S3 envelope rather than a bare 405.
            if "uploads" in request.query_params or "uploadId" in request.query_params:
                return _error(501, "NotImplemented",
                              "multipart uploads are not implemented by this emulation; keep "
                              "uploads under boto3's multipart_threshold (8 MB) or lower it in "
                              "boto3.s3.transfer.TransferConfig",
                              BucketName=bucket, Key=key)
            return _error(501, "NotImplemented",
                          f"{request.method} on an object (RestoreObject, SelectObjectContent, "
                          f"POST-form uploads, …) is not implemented by this emulation",
                          BucketName=bucket, Key=key)
        try:
            _safe_bucket(bucket)
            _safe_key(key)
        except S3Error as e:
            return _s3_error(e, BucketName=bucket)
        # `copy_object`, `put_object_tagging`, `put_object_acl` … all arrive as a PUT on this
        # same path; writing their body (or their empty body) over the object would destroy
        # it and report success.
        bad = _unsupported_op(request, _OBJECT_PARAMS, bucket, key) or _wrong_owner(request, bucket)
        if bad is not None:
            return bad
        if request.method == "DELETE":
            # Conditional deletes on a general-purpose bucket take `If-Match` only; the
            # size / last-modified forms are directory-bucket features → refused, not ignored.
            extra = sorted(h for h in ("x-amz-if-match-size", "x-amz-if-match-last-modified-time")
                           if h in request.headers)
            if extra:
                return _error(501, "NotImplemented",
                              f"this emulation does not implement {', '.join(extra)}",
                              BucketName=bucket, Key=key)
        # From here to the write / the read there is no `await`: the handler runs its file
        # I/O on the loop thread, so stat -> precondition -> write is atomic with respect to
        # every other request. That is what makes `If-None-Match: *` a real guard.
        conditional = any(h in request.headers for h in
                          ("if-match", "if-none-match", "if-modified-since", "if-unmodified-since"))
        # Off the loop thread: the store lock may be held by the batch writer for the
        # length of a large write, and a blocking acquire here would stall EVERY route.
        return await asyncio.to_thread(_object_ops_locked, request, bucket, key, body, conditional)

    def _object_ops_locked(request: Request, bucket: str, key: str, body: bytes,
                           conditional: bool) -> Any:
        with _STORE_LOCK:
            return _object_ops_inner(request, bucket, key, body, conditional)

    def _object_ops_inner(request: Request, bucket: str, key: str, body: bytes,
                          conditional: bool) -> Any:
        try:
            cur = stat_object(bucket, key)
            # Hashing the CURRENT object is only needed to answer a conditional request or
            # to report an ETag on a read; an unconditional overwrite must not pay for it.
            cur_etag = (_file_etag(_obj_path(bucket, key), cur)
                        if cur is not None and (conditional or request.method != "PUT") else None)
        except S3Error as e:
            return _s3_error(e, BucketName=bucket, Key=key)
        except OSError:
            cur, cur_etag = None, None
        if request.method == "PUT":
            if "if-match" in request.headers and cur is None:
                return _error(404, "NoSuchKey", "The specified key does not exist.", Key=key)
            failed = _precondition(request, cur_etag, cur.st_mtime if cur else None)
            if failed is not None:
                return failed
            data = body
            try:
                checksums: dict[str, str] = {k: v for k, v in request.headers.items()
                                             if k == "content-md5" or k.startswith("x-amz-checksum-")}
                if "aws-chunked" in request.headers.get("content-encoding", ""):
                    data, trailers = decode_aws_chunked(data)
                    checksums.update(trailers)
                    declared = request.headers.get("x-amz-decoded-content-length")
                    if declared is not None and declared.strip() != str(len(data)):
                        raise S3Error("IncompleteBody",
                                      f"x-amz-decoded-content-length says {declared.strip()} "
                                      f"bytes, the chunks carried {len(data)}")
                verify_checksums(data, checksums)
                etag = put_object(bucket, key, data)
            except S3Error as e:
                return _s3_error(e, BucketName=bucket, Key=key)
            return Response(status_code=200, headers={"ETag": etag})
        if not bucket_exists(bucket):
            return (Response(status_code=404) if request.method == "HEAD"
                    else _error(404, "NoSuchBucket", "The specified bucket does not exist",
                                BucketName=bucket))
        if request.method == "DELETE":
            if "if-match" in request.headers:
                # Conditional delete: `If-Match: *` = only if it exists, `If-Match: "<etag>"`
                # = only if unchanged. Real S3: 412 on a mismatch, Not Found when the object
                # is gone. Deleting anyway would defeat the guard callers rely on.
                if cur is None:
                    return _error(404, "NoSuchKey", "The specified key does not exist.", Key=key)
                failed = _precondition(request, cur_etag, cur.st_mtime)
                if failed is not None:
                    return failed
            try:
                delete_object(bucket, key)
            except S3Error as e:
                return _s3_error(e, BucketName=bucket, Key=key)
            return Response(status_code=204)
        stat = cur
        if stat is None:
            return (Response(status_code=404) if request.method == "HEAD"
                    else _error(404, "NoSuchKey", "The specified key does not exist.", Key=key))
        failed = _precondition(request, cur_etag, stat.st_mtime)
        if failed is not None:
            return failed
        size = stat.st_size
        headers = {"ETag": cur_etag or "",
                   "Content-Length": str(size),
                   "Last-Modified": _http_date(stat.st_mtime),
                   "Accept-Ranges": "bytes"}
        if request.method == "HEAD":
            # No body is sent, so never read the object just to measure it.
            return Response(status_code=200, headers=headers, media_type="binary/octet-stream")
        # boto3's managed download (`download_file` / `Object.download_fileobj`) splits any
        # object over `multipart_threshold` into concurrent ranged GETs and concatenates the
        # bodies: ignoring `Range` here would hand the caller a corrupt file with no error.
        raw_range = request.headers.get("range")
        rng = None
        if raw_range:
            try:
                rng = parse_range(raw_range, size)
            except S3Error as e:
                return _s3_error(e, BucketName=bucket, Key=key,
                                 ActualObjectSize=str(size), RangeRequested=raw_range)
        try:
            if rng is not None:
                first, last = rng
                body_out = read_range(bucket, key, first, last - first + 1)
                headers["Content-Length"] = str(len(body_out))
                headers["Content-Range"] = f"bytes {first}-{last}/{size}"
                return Response(content=body_out, status_code=206, headers=headers,
                                media_type="binary/octet-stream")
            data = get_object(bucket, key)
        except S3Error as e:
            return _s3_error(e, BucketName=bucket, Key=key)
        except OSError:
            data = None
        if data is None:
            return _error(404, "NoSuchKey", "The specified key does not exist.", Key=key)
        return Response(content=data, status_code=200, headers=headers,
                        media_type="binary/octet-stream")

    return router
