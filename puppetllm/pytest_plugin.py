"""pytest plugin: a `puppet` fixture backed by an in-process server.

Registered through the `pytest11` entry point when puppetllm is pip-installed, so tests
can use it without any conftest:

    def test_weather(puppet):
        puppet.expect(tools=["get_weather"]).respond(text="sunny")
        client = anthropic.Anthropic(base_url=puppet.url, api_key="test", max_retries=0)
        ...

One server per test session (`puppet_server`); `puppet` clears the state and restores the
configuration and rules the server started with before and after every test — so settings
from the `PUPPETLLM_*` environment variables (or from a `--config` file on an external
server) stay in force, while what one test changed through `puppet.config(...)` or
`puppet.expect(...)` never reaches the next.
`PUPPETLLM_URL` points the fixture at an already-running server instead of starting one.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import pytest

from .testing import Puppet, serve


_BASELINES: dict[int, dict[str, Any]] = {}


@pytest.fixture(scope="session")
def puppet_server() -> Iterator[Puppet]:
    url = os.environ.get("PUPPETLLM_URL")
    if url:
        with Puppet(url) as p:
            p.wait_ready()
            # Captured before any test can touch the server.
            _BASELINES[id(p)] = p.baseline()
            yield p
        return
    with serve() as handle:
        _BASELINES[id(handle)] = handle.startup_baseline
        yield handle


@pytest.fixture(scope="session")
def puppet_baseline(puppet_server: Puppet) -> dict[str, Any]:
    """The configuration and rules the server started with (environment / `--config`)."""
    return _BASELINES[id(puppet_server)]


@pytest.fixture
def puppet(puppet_server: Puppet, puppet_baseline: dict[str, Any]) -> Iterator[Puppet]:
    puppet_server.reset(puppet_baseline)
    yield puppet_server
    puppet_server.reset(puppet_baseline)


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "puppet: tests that talk to the puppetllm fake server")
