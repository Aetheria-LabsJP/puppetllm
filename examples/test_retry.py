"""A pytest example: the SDK's retry on a rate limit, exercised against puppetllm.

The `puppet` fixture comes with `pip install .` (one server per test session; state is
cleared around every test). The scenario: the first request gets a 429 with
`retry-after`, the retried request gets the answer. The test then asserts both requests
reached the server and that the app saw the final text.

    pip install ".[test]"
    python3 -m pytest examples/test_retry.py
"""

from __future__ import annotations

import anthropic


def test_sdk_retries_a_rate_limit(puppet):
    puppet.expect(model="claude-*").error(429, type="rate_limit_error",
                                          message="slow down", headers={"retry-after": "0"})
    puppet.expect(model="claude-*").respond(text="second time lucky")

    client = anthropic.Anthropic(base_url=puppet.url, api_key="sk-mock", max_retries=2)
    msg = client.messages.create(model="claude-sonnet-4-5", max_tokens=32,
                                 messages=[{"role": "user", "content": "hello"}])

    assert msg.content[0].text == "second time lucky"
    puppet.assert_sent(count=2, contains="hello")  # the original and the retry
    puppet.assert_consumed()
    assert puppet.history()[0]["injected_error"]["status"] == 429


def test_without_retries_the_error_surfaces(puppet):
    puppet.expect().error(429, type="rate_limit_error", message="slow down")

    client = anthropic.Anthropic(base_url=puppet.url, api_key="sk-mock", max_retries=0)
    try:
        client.messages.create(model="claude-sonnet-4-5", max_tokens=32,
                               messages=[{"role": "user", "content": "hello"}])
    except anthropic.RateLimitError as e:
        assert e.status_code == 429
    else:
        raise AssertionError("expected a RateLimitError")
    puppet.assert_sent(count=1)
