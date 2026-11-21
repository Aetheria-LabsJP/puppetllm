"""A complete tool-use loop against puppetllm, scripted end to end.

The server runs in-process on a free port, two rules play the model (first a `tool_use`
for `get_weather`, then the final text once the tool result is back), and an ordinary
Anthropic SDK agent loop runs against it — no responder, no real API, no cost.

    pip install ".[test]"        # anthropic + the server
    python3 examples/tool_loop.py
"""

from __future__ import annotations

import json

import anthropic

from puppetllm.testing import serve

TOOLS = [{
    "name": "get_weather",
    "description": "Current weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}]


def get_weather(city: str) -> dict:
    """The app's real tool (here a stub)."""
    return {"city": city, "temperature_c": 23, "sky": "clear"}


def main() -> None:
    with serve() as puppet:
        # The script: a tool call first, then — once the app has sent the tool result —
        # the final answer. Rules are consumed in order; `assert_consumed()` checks both ran.
        puppet.expect(tools=["get_weather"]).respond(content=[
            {"type": "tool_use", "name": "get_weather", "input": {"city": "Tokyo"}}])
        puppet.expect(has_tool_result=True).respond(text="It is 23 °C and clear in Tokyo.")

        client = anthropic.Anthropic(base_url=puppet.url, api_key="sk-mock", max_retries=0)
        messages = [{"role": "user", "content": "What is the weather in Tokyo?"}]
        while True:
            msg = client.messages.create(model="claude-sonnet-4-5", max_tokens=256,
                                         tools=TOOLS, messages=messages)
            if msg.stop_reason != "tool_use":
                break
            # Run every tool the model asked for and send the results back.
            messages.append({"role": "assistant", "content": [b.model_dump() for b in msg.content]})
            results = []
            for block in msg.content:
                if block.type == "tool_use":
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(get_weather(**block.input))})
            messages.append({"role": "user", "content": results})

        print("final answer:", msg.content[0].text)
        puppet.assert_consumed()
        # What the app sent, as the server recorded it.
        sent = puppet.assert_sent(count=2, tool="get_weather")
        print("tool result the app sent:", sent[1]["request"]["messages"][-1]["content"][0]["content"])
        print("estimated cost:", puppet.stats()["totals"]["total_usd"], "USD")


if __name__ == "__main__":
    main()
