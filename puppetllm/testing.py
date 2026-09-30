"""Test-side helpers: a client for the control plane and an in-process server.

    from puppetllm.testing import serve

    with serve() as puppet:                       # uvicorn on a free localhost port
        client = anthropic.Anthropic(base_url=puppet.url, api_key="test")
        puppet.expect(tools=["get_weather"]).respond(
            content=[{"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                      "input": {"city": "Tokyo"}}])
        puppet.expect(has_tool_result=True).respond(text="It is sunny.")
        ...run the app under test...
        puppet.assert_consumed()
        puppet.assert_sent(model="claude-*", contains="Tokyo")

`Puppet(url)` alone talks to a server started elsewhere (docker compose, `puppetllm serve`).
Every method raises `PuppetError` on a non-2xx control response. Only `httpx` (already a
server dependency) is needed; the pytest fixture lives in `puppetllm.pytest_plugin`.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import socket
import threading
import time
import warnings
from typing import Any, Iterator

import httpx

from .harness import LATENCY_KEYS


class PuppetError(RuntimeError):
    """A control call was refused (the server's error message is the text)."""


class PuppetAssertionError(AssertionError):
    """An `assert_*` helper found the recorded traffic did not match."""


def _pick(kw: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: kw.pop(k) for k in tuple(kw) if k in keys}


class Expectation:
    """A rule under construction: `puppet.expect(...)` then `.respond(...)` / `.error(...)`
    (chain several to script a sequence; `.repeat()` makes the last step permanent).
    Each call re-posts the rule in place; steps already consumed stay consumed, so a rule
    may be extended after traffic."""

    def __init__(self, puppet: "Puppet", match: dict[str, Any], rule_id: str | None) -> None:
        self._puppet = puppet
        self._match = match
        self._id = rule_id
        self._steps: list[dict[str, Any]] = []
        self._repeat = False

    def respond(self, text: str | None = None, *, content: list[dict[str, Any]] | None = None,
                stop_reason: str | None = None, usage: dict[str, Any] | None = None,
                **latency: int) -> "Expectation":
        body: dict[str, Any] = {}
        if content is not None:
            body["content"] = content
        elif text is not None:
            body["text"] = text
        else:
            body["content"] = []
        if stop_reason is not None:
            body["stop_reason"] = stop_reason
        if usage is not None:
            body["usage"] = usage
        return self._add({"respond": body, **_latency_only(latency)})

    def error(self, status: int = 500, *, type: str | None = None, message: str | None = None,
              headers: dict[str, str] | None = None, after_events: int | None = None,
              after_blocks: int | None = None,
              content: list[dict[str, Any]] | None = None, **latency: int) -> "Expectation":
        body: dict[str, Any] = {"status": status}
        if type is not None:
            body["type"] = type
        if message is not None:
            body["message"] = message
        if headers is not None:
            body["headers"] = headers
        if after_events is not None:
            body["after_events"] = after_events
        if after_blocks is not None:
            body["after_blocks"] = after_blocks
        if content is not None:
            body["content"] = content
        return self._add({"error": body, **_latency_only(latency)})

    def repeat(self) -> "Expectation":
        self._repeat = True
        return self._sync()

    def _add(self, step: dict[str, Any]) -> "Expectation":
        self._steps.append(step)
        return self._sync()

    def _sync(self) -> "Expectation":
        """(Re)post the rule so each chained call takes effect immediately; a re-post
        keeps the rule's place in the order. Nothing is posted until the first step
        exists (so `.repeat()` may come first)."""
        if not self._steps:
            return self
        rule = {"match": self._match, "steps": self._steps, "repeat": self._repeat}
        if self._id is None:
            out = self._puppet._post("/_control/rules", rule)
            self._id = out["rules"][-1]["id"]
        else:
            self._puppet._put(f"/_control/rules/{self._id}", rule)
        return self

    @property
    def id(self) -> str | None:
        return self._id


def _latency_only(kw: dict[str, Any]) -> dict[str, Any]:
    bad = sorted(k for k in kw if k not in LATENCY_KEYS)
    if bad:
        raise TypeError(f"unknown keyword(s) {bad}; latency keys are {list(LATENCY_KEYS)}")
    return dict(kw)


class Puppet:
    """Client for a running puppetllm server's control plane."""

    def __init__(self, url: str = "http://127.0.0.1:8765", *, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self._http = httpx.Client(base_url=self.url, timeout=timeout)

    # ── transport ──

    def _check(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            raise PuppetError(f"{r.request.method} {r.request.url.path} -> {r.status_code}: {detail}")
        return r.json()

    def _get(self, path: str, **params: Any) -> Any:
        return self._check(self._http.get(path, params=params or None))

    def _post(self, path: str, body: Any = None) -> Any:
        return self._check(self._http.post(path, json=body if body is not None else {}))

    def _put(self, path: str, body: Any) -> Any:
        return self._check(self._http.put(path, json=body))

    def _delete(self, path: str) -> Any:
        return self._check(self._http.delete(path))

    def close(self) -> None:
        self._http.close()

    # ── liveness ──

    def wait_ready(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                if self._http.get("/_control/health", timeout=2.0).status_code == 200:
                    return
            except httpx.HTTPError as e:
                last = e
            time.sleep(0.05)
        raise PuppetError(f"server at {self.url} not ready after {timeout}s ({last})")

    def health(self) -> dict[str, Any]:
        return self._get("/_control/health")

    # ── the interactive surface ──

    def pending(self) -> list[dict[str, Any]]:
        return self._get("/_control/pending")["pending"]

    def wait_pending(self, timeout: float = 30.0) -> dict[str, Any]:
        """Block until a request is pending; returns its snapshot (`pending_id` inside)."""
        # The long-poll must outlive the client's default read timeout.
        out = self._check(self._http.get("/_control/wait_for_pending", params={"timeout": timeout},
                                         timeout=httpx.Timeout(timeout + 5.0)))
        if not out.get("has_pending"):
            raise PuppetError(f"no request arrived within {timeout}s")
        return out["request"]

    def respond(self, text: str | None = None, *, content: list[dict[str, Any]] | None = None,
                pending_id: str | None = None, custom_id: str | None = None,
                batch_id: str | None = None, **extra: Any) -> None:
        """Answer a pending (`/_control/respond`). `extra` takes stop_reason / usage /
        the latency keys."""
        body: dict[str, Any] = dict(extra)
        if content is not None:
            body["content"] = content
        elif text is not None:
            body["text"] = text
        else:
            body["content"] = []
        for k, v in (("pending_id", pending_id), ("custom_id", custom_id), ("batch_id", batch_id)):
            if v is not None:
                body[k] = v
        self._post("/_control/respond", body)

    def respond_all(self, text: str | None = None, *, content: list[dict[str, Any]] | None = None,
                    **extra: Any) -> list[str]:
        """The same answer for every live pending (`/_control/respond_all`); returns the
        pending ids it answered."""
        body: dict[str, Any] = dict(extra)
        if content is not None:
            body["content"] = content
        elif text is not None:
            body["text"] = text
        else:
            body["content"] = []
        return self._post("/_control/respond_all", body)["pending_ids"]

    def respond_many(self, responses: list[dict[str, Any]]) -> None:
        """Several answers at once, each item a `respond` body with its own target
        (`{"responses": [...]}`): all are validated before any is applied."""
        self._post("/_control/respond", {"responses": responses})

    def error(self, status: int = 500, *, pending_id: str | None = None, **extra: Any) -> None:
        """Fail a pending (`/_control/error`); `extra` takes type / message / headers /
        after_events / content / the latency keys."""
        body = {"status": status, **extra}
        if pending_id is not None:
            body["pending_id"] = pending_id
        self._post("/_control/error", body)

    # ── scenario rules ──

    def expect(self, *, id: str | None = None, provider: str | None = None,
               model: str | None = None, tools: list[str] | None = None,
               has_tool_result: bool | None = None, last_user_text: str | None = None,
               turn: int | None = None, stream: bool | None = None) -> Expectation:
        """Start a rule: the keyword arguments are the match (all omitted = any request)."""
        match = {k: v for k, v in (("provider", provider), ("model", model), ("tools", tools),
                                   ("has_tool_result", has_tool_result),
                                   ("last_user_text", last_user_text), ("turn", turn),
                                   ("stream", stream)) if v is not None}
        return Expectation(self, match, id)

    def rules(self) -> dict[str, Any]:
        return self._get("/_control/rules")

    def set_rules(self, rules: list[dict[str, Any]]) -> dict[str, Any]:
        return self._put("/_control/rules", {"rules": rules})

    def add_rules(self, rules: list[dict[str, Any]]) -> dict[str, Any]:
        return self._post("/_control/rules", {"rules": rules})

    def clear_rules(self) -> None:
        self._delete("/_control/rules")

    # ── configuration / clock ──

    def strict_blocks(self, enabled: bool = True) -> None:
        """Refuse (400) injected block types the server does not model instead of dropping
        them — `config.strict_blocks`."""
        self.config(strict_blocks=enabled)

    def config(self, **changes: Any) -> dict[str, Any]:
        """No arguments: read. With arguments: change those keys (see `/_control/config`)."""
        if changes:
            return self._post("/_control/config", changes)["config"]
        return self._get("/_control/config")["config"]

    def advance_clock(self, seconds: float) -> float:
        return self._post("/_control/clock/advance", {"seconds": seconds})["clock_offset_seconds"]

    # ── observation ──

    def history(self) -> list[dict[str, Any]]:
        return self._get("/_control/history")["history"]

    def stats(self) -> dict[str, Any]:
        return self._get("/_control/stats")

    def cache(self) -> dict[str, Any]:
        return self._get("/_control/cache")

    def clear(self, *, config: bool = False) -> None:
        """Drop pendings, history, cache, batches, rules and the clock offset. The
        configuration stays unless `config=True` restores its defaults too."""
        self._post("/_control/clear", {"config": True} if config else None)

    def baseline(self) -> dict[str, Any]:
        """The configuration and rules as they are now (`{"config", "rules"}`), in the
        form `reset()` re-applies — take it once the server has started to preserve what
        the environment / a `--config` file set up."""
        return {"config": self.config(),
                "rules": [{k: r[k] for k in ("id", "match", "steps", "repeat")}
                          for r in self.rules()["rules"]]}

    def reset(self, baseline: dict[str, Any] | None = None) -> None:
        """`clear()` plus the configuration and rules set back to `baseline` (a
        `baseline()` result; a bare `config()` result is accepted too) — or to the
        defaults, with no rules, when None."""
        self.clear(config=True)
        if not baseline:
            return
        if "config" in baseline or "rules" in baseline:
            if baseline.get("config"):
                self.config(**baseline["config"])
            if baseline.get("rules"):
                self.set_rules(baseline["rules"])
        else:
            self.config(**baseline)

    # ── assertions ──

    def sent(self, *, provider: str | None = None, model: str | None = None,
             tool: str | None = None, contains: str | None = None,
             stream: bool | None = None) -> list[dict[str, Any]]:
        """History entries whose request matches every given filter (`model` is a glob,
        `tool` a tool name the request offered, `contains` a substring of any text in the
        conversation)."""
        out = []
        for e in self.history():
            req = e.get("request") or {}
            if provider is not None and e.get("provider") != provider:
                continue
            if model is not None and not fnmatch.fnmatchcase(str(e.get("model") or ""), model):
                continue
            if stream is not None and bool(req.get("stream")) != stream:
                continue
            if tool is not None and tool not in {t.get("name") for t in req.get("tools") or []
                                                 if isinstance(t, dict)}:
                continue
            if contains is not None and contains not in _all_text(req):
                continue
            out.append(e)
        return out

    def assert_sent(self, *, count: int | None = None, **filters: Any) -> list[dict[str, Any]]:
        """At least one (or exactly `count`) recorded request matches the filters."""
        found = self.sent(**filters)
        if count is None and not found:
            raise PuppetAssertionError(f"no recorded request matches {filters}")
        if count is not None and len(found) != count:
            raise PuppetAssertionError(
                f"{len(found)} recorded request(s) match {filters}, expected {count}")
        return found

    def assert_not_sent(self, **filters: Any) -> None:
        found = self.sent(**filters)
        if found:
            raise PuppetAssertionError(f"{len(found)} recorded request(s) match {filters}")

    def assert_consumed(self) -> None:
        """Every rule with a finite step list was used up (nothing the test scripted went
        unasked)."""
        view = self.rules()
        if not view["all_consumed"]:
            raise PuppetAssertionError(f"rules with unconsumed steps: {view['unconsumed']}")

    def assert_no_pending(self) -> None:
        left = self.pending()
        if left:
            raise PuppetAssertionError(
                f"{len(left)} request(s) still pending: {[p['pending_id'] for p in left]}")

    def __enter__(self) -> "Puppet":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _all_text(req: dict[str, Any]) -> str:
    parts: list[str] = []
    sysm = req.get("system")
    if isinstance(sysm, str):
        parts.append(sysm)
    elif isinstance(sysm, list):
        parts.extend(str(b.get("text", "")) for b in sysm if isinstance(b, dict))
    for m in req.get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    if isinstance(b.get("text"), str):
                        parts.append(b["text"])
                    elif b.get("type") == "tool_use":
                        parts.append(json.dumps(b.get("input"), ensure_ascii=False))
                    elif b.get("type") == "tool_result":
                        parts.append(json.dumps(b.get("content"), ensure_ascii=False))
    return "\n".join(parts)


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


class ServerHandle(Puppet):
    """A `Puppet` bound to the in-process server that `serve()` started."""

    def __init__(self, url: str, server: Any, thread: threading.Thread) -> None:
        super().__init__(url)
        self._server = server
        self._thread = thread
        # What the server started with (set by serve()); the pytest fixture resets to it.
        self.startup_baseline: dict[str, Any] = {}

    def stop(self, timeout: float = 10.0) -> None:
        # Release the pendings still waiting (their handlers return at once), then stop
        # the server thread; a long-poll still open ends with the graceful-shutdown timeout.
        try:
            self.clear()
        except Exception:
            pass
        self._server.should_exit = True
        self._thread.join(timeout)
        if self._thread.is_alive():
            warnings.warn("puppetllm server thread did not stop within "
                          f"{timeout}s; it keeps running in the background", RuntimeWarning)
        self.close()


@contextlib.contextmanager
def serve(*, host: str = "127.0.0.1", port: int | None = None,
          config: dict[str, Any] | None = None, rules: list[dict[str, Any]] | None = None,
          ready_timeout: float = 15.0) -> Iterator[ServerHandle]:
    """Run the fake server on a background thread for the duration of the block.

    The server is the module-global app, so one process hosts one server at a time
    (parallel test workers each get their own process and port). `config` / `rules` are
    applied before the first request; the configuration persists across serve() calls in
    one process (use `handle.reset()` for a clean slate)."""
    import uvicorn

    from . import fake_server as fs

    port = port or free_port(host)
    server = uvicorn.Server(uvicorn.Config(fs.app, host=host, port=port, log_level="warning",
                                           timeout_graceful_shutdown=5))
    thread = threading.Thread(target=server.run, name="puppetllm-server", daemon=True)
    thread.start()
    handle = ServerHandle(f"http://{host}:{port}", server, thread)
    try:
        handle.wait_ready(ready_timeout)
        # The app is module-global: drop state left by an earlier serve() in this process
        # while keeping the configuration and the rules the environment / a `--config`
        # file set up (or an earlier `config(...)`); `config=` / `rules=` layer on top.
        handle.reset(handle.baseline())
        if config:
            handle.config(**config)
        if rules is not None:
            handle.set_rules(rules)
        # What this server starts with, `config=` / `rules=` included: the pytest fixture
        # resets to it around every test.
        handle.startup_baseline = handle.baseline()
        yield handle
    finally:
        handle.stop()
