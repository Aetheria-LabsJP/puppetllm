"""Agent SDK adapter tests. No Anthropic requests or paid inference.

Install .[test,agent-sdk] to include these tests. Most tests replace the SDK
client's response stream; opt-in CLI tests use only local fake API responses.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import importlib
import importlib.util
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from puppetllm.agent_sdk import AgentSDKBackend, SDKOutputError, SDKRunError, UnsupportedSDKRequest, input_content, prepare_request
from puppetllm.testing import serve
from puppetllm.relay import Relay, build_parser, looks_foreign, resolve_defaults


HAS_DEPS = all(importlib.util.find_spec(n) for n in ("claude_agent_sdk", "jsonschema"))
TOOL = {"name": "Read", "description": "Read a file in the caller's workspace",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}},
                         "required": ["file_path"], "additionalProperties": False}}
REQUEST = {"model": "claude-sonnet-4-6", "max_tokens": 1024,
           "system": [{"type": "text", "text": "日本語で答える"}],
           "messages": [{"role": "user", "content": "README.mdを読んで"}], "tools": [TOOL]}
CALL = {"content": [{"type": "text", "text": "読みます。"},
                    {"type": "tool_use", "name": "Read", "input_json": '{"file_path":"README.md"}'}],
        "stop_reason": "tool_use"}
TEXT = {"content": [{"type": "text", "text": "読みました。"}], "stop_reason": "end_turn"}
IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
    "data": "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg=="}}
QUESTION_TOOL = {"name": "AskUserQuestion", "input_schema": {
    "type": "object", "properties": {
        "questions": {"type": "array", "minItems": 1, "maxItems": 4, "items": {
            "type": "object", "properties": {
                "question": {"type": "string"}, "header": {"type": "string"},
                "options": {"type": "array", "minItems": 2, "maxItems": 4, "items": {
                    "type": "object", "properties": {"label": {"type": "string"},
                        "description": {"type": "string"}},
                    "required": ["label", "description"], "additionalProperties": False}},
                "multiSelect": {"type": "boolean", "default": False}},
            "required": ["question", "header", "options", "multiSelect"], "additionalProperties": False}},
        "answers": {"type": "object", "additionalProperties": {"type": "string"}}},
    "required": ["questions"], "additionalProperties": False}}
QUESTIONS = [{"question": "Which option?", "header": "Choice", "options": [
    {"label": "A", "description": "First option"}, {"label": "B", "description": "Second option"}]}]


def question_call(args):
    return {"content": [{"type": "tool_use", "name": "AskUserQuestion",
                          "input_json": json.dumps(args)}], "stop_reason": "tool_use"}


def thinking_request():
    req = copy.deepcopy(REQUEST)
    req["messages"] += [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "PRIVATE_REASONING", "signature": "PRIVATE_SIGNATURE"},
            {"type": "redacted_thinking", "data": "UFJJVkFURV9EQVRB"},
            {"type": "tool_use", "id": "old_read", "name": "Read", "input": {"file_path": "README.md"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old_read", "content": "README contents"}]}]
    return req


def inline_tool_request():
    req = thinking_request()
    definition = {**copy.deepcopy(TOOL), "name": "Probe"}
    req["messages"].append({"role": "system", "content": [
        {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": definition}},
        {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "Read"}}]})
    return req


_STATE = tempfile.TemporaryDirectory(prefix="puppetllm-sdk-test-")  # shared with the other SDK test modules
atexit.register(_STATE.cleanup)


def config(*args):
    return resolve_defaults(build_parser().parse_args([
        "--kind", "agent-sdk", "--puppet", "http://test", "--poll-timeout", "0.1",
        "--sdk-state-dir", _STATE.name, *args]))


def private_home(case):
    """Keep the bundled CLI's own files, and `~` paths, inside a temporary directory."""
    home = tempfile.TemporaryDirectory(prefix="puppetllm-sdk-home-")
    case.addCleanup(home.cleanup)
    patcher = patch.dict(os.environ, {"HOME": home.name, "CLAUDE_CONFIG_DIR": os.path.join(home.name, ".claude")})
    patcher.start()
    case.addCleanup(patcher.stop)


def answer_without_gate(case):
    """A fake SDK answers without an API call, which the backend otherwise refuses."""
    from puppetllm.sdk_egress import SDKAPIGate
    patcher = patch.object(SDKAPIGate, "used", True)
    patcher.start()
    case.addCleanup(patcher.stop)


def fake_sdk(query):
    import claude_agent_sdk as sdk
    class Client:
        def __init__(self, *, options, transport=None):
            self.options = options
            self.stream = None

        async def __aenter__(self):
            return self

        async def query(self, prompt):
            self.stream = query(prompt=prompt, options=self.options)

        def receive_response(self):
            return self.stream

        async def __aexit__(self, *exc):
            if self.stream is not None:
                await self.stream.aclose()

    return SimpleNamespace(query=query, ClaudeAgentOptions=sdk.ClaudeAgentOptions,
                           ResultMessage=sdk.ResultMessage, AssistantMessage=sdk.AssistantMessage,
                           ToolUseBlock=sdk.ToolUseBlock, TextBlock=sdk.TextBlock,
                           StreamEvent=sdk.StreamEvent, CLIJSONDecodeError=sdk.CLIJSONDecodeError,
                           ResultError=sdk.ResultError, ClaudeSDKClient=Client)


def stream_event(kind, **fields):
    import claude_agent_sdk as sdk
    return sdk.StreamEvent(uuid="event-local", session_id="sdk-test", event={"type": kind, **fields})


def native_events(blocks, *, model="claude-opus-5-5", stop_reason="tool_use"):
    """The real CLI emits one AssistantMessage per completed API content block."""
    import claude_agent_sdk as sdk
    yield stream_event("message_start", message={"id": "msg-local", "model": model,
        "usage": {"input_tokens": 100, "output_tokens": 2}})
    for block in blocks:
        yield sdk.AssistantMessage(model=model, message_id="msg-local", content=[block])
    yield stream_event("message_delta", delta={"stop_reason": stop_reason}, usage={"output_tokens": 77})
    yield stream_event("message_stop")


async def sdk_transcript(prompt):
    """Read the actual SDK input stream, reconstructing its external transcript."""
    messages = [message async for message in prompt]
    blocks = messages[0]["message"]["content"]
    first = json.loads(blocks[0]["text"])
    transcript = {"system": first["system"], "tools": first["tools"], "messages": []}
    for block in blocks[1:]:
        if block["type"] != "text" or block["text"].startswith("Attachment image_"):
            continue
        record = json.loads(block["text"])
        if record["record_type"] == "message":
            transcript["messages"].append({"role": record["role"], "content": record["content"]})
        elif record["record_type"] == "request":
            transcript["tool_choice"] = record["tool_choice"]
    return transcript


def result(data=TEXT, **kw):
    from claude_agent_sdk import ResultMessage
    fields = dict(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                  num_turns=1, session_id="sdk-test", structured_output=copy.deepcopy(data),
                  usage={"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 5})
    fields.update(kw)
    return ResultMessage(**fields)


class TestSDKPreparation(unittest.TestCase):
    def test_request_effort_is_kept_and_other_output_config_keys_are_reported(self):
        def prepared(config_value):
            dropped = []
            body = prepare_request({**REQUEST, "params": {"output_config": config_value}}, "m", dropped)
            return body.get("effort"), [key for key in dropped if key.startswith("output_config")]
        self.assertEqual(prepared({"effort": "low"}), ("low", []))
        self.assertEqual(prepared({"effort": "low", "format": {}}), ("low", ["output_config.format"]))
        self.assertEqual(prepared({"effort": "unknown"}), (None, ["output_config"]))

    @unittest.skipUnless(HAS_DEPS, "install .[test,agent-sdk]")
    def test_configured_effort_caps_the_request_effort(self):
        for limit, asked, used in (("high", "low", "low"), ("medium", "max", "medium"), ("high", None, "high"),
                                   ("auto", "xhigh", "xhigh"), ("auto", None, None)):
            backend = AgentSDKBackend(config("--sdk-effort", limit, "--dry-run"))
            body = {"effort": asked} if asked else {}
            with self.subTest(limit=limit, asked=asked):
                self.assertEqual(backend.effort(body), used)
                self.assertEqual(backend.generation_settings(body).get("effort"), used)
        dropped = []
        AgentSDKBackend(config("--sdk-effort", "medium", "--dry-run")).prepare(
            {**REQUEST, "params": {"output_config": {"effort": "max"}}}, "m", dropped)
        self.assertIn("output_config.effort (capped to medium)", dropped)

    def test_cacheable_input_keeps_unchanged_history_blocks_stable(self):
        first = prepare_request(REQUEST, "m", [])
        followup = copy.deepcopy(REQUEST)
        followup["messages"] += [{"role": "assistant", "content": "Read completed."},
                                 {"role": "user", "content": "Continue."}]
        second = prepare_request(followup, "m", [])
        original = copy.deepcopy(first)
        before = input_content(first)
        after = input_content(second)
        without_cache = lambda block: {k: v for k, v in block.items() if k != "cache_control"}
        self.assertEqual([without_cache(b) for b in before[:-1]],
                         [without_cache(b) for b in after[:len(before) - 1]])
        self.assertEqual(sum("cache_control" in b for b in before), 1)
        self.assertEqual(sum("cache_control" in b for b in after), 1)
        self.assertIn("cache_control", after[-2])
        self.assertEqual([b["cache_control"] for b in input_content(second, cache_ttl="1h") if "cache_control" in b],
                         [{"type": "ephemeral", "ttl": "1h"}])
        self.assertNotIn("cache_control", after[-1])
        self.assertEqual(first, original)
        restored = asyncio.run(sdk_transcript(self._stream(after)))
        self.assertEqual(restored, second["transcript"])

    @staticmethod
    async def _stream(content):
        yield {"type": "user", "message": {"role": "user", "content": content}}

    def test_tool_choice_is_after_cached_history_and_context_changes_invalidate_prefix(self):
        body = prepare_request(REQUEST, "m", [])
        before = input_content(body)
        changed = copy.deepcopy(body)
        changed["transcript"]["tool_choice"] = {"type": "none"}
        after = input_content(changed)
        self.assertEqual(before[:-1], after[:-1])
        self.assertNotEqual(before[-1], after[-1])
        for field in ("system", "tools"):
            edited = copy.deepcopy(body)
            edited["transcript"][field] = []
            self.assertNotEqual(input_content(edited)[0], before[0])
        compacted = copy.deepcopy(body)
        compacted["transcript"]["messages"] = [{"role": "user", "content": "Compacted summary."}]
        self.assertNotEqual(input_content(compacted)[1], before[1])
        self.assertEqual(len(input_content(compacted)), 3)

    def test_transcript_preserves_history_and_does_not_mutate_input(self):
        req = copy.deepcopy(REQUEST)
        req["messages"] += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "old",
                "name": "Read", "input": {"file_path": "README.md"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old",
                "content": [{"type": "text", "text": "File contents"}], "is_error": False}]}]
        original = copy.deepcopy(req)
        dropped = []
        body = prepare_request(req, "chosen-model", dropped)
        self.assertEqual(body["model"], "chosen-model")
        self.assertEqual(body["transcript"]["messages"], req["messages"])
        self.assertEqual(req, original)
        self.assertTrue(any("max_tokens" in key for key in dropped))

    def test_unsupported_media_and_nonassistant_thinking_fail_explicitly(self):
        for block in ({"type": "image"}, {"type": "thinking", "thinking": "x"},
                      {"type": "redacted_thinking", "data": "hidden"},
                      {"type": "tool_result", "tool_use_id": "x", "content": [{"type": "image"}]}):
            with self.subTest(block=block):
                req = {**REQUEST, "messages": [{"role": "user", "content": [block]}]}
                with self.assertRaises(UnsupportedSDKRequest):
                    prepare_request(req, "m", [])

    def test_assistant_thinking_is_omitted_without_losing_text_or_tools(self):
        visible = [{"type": "text", "text": "ファイルを読みます。"},
                   {"type": "tool_use", "id": "old_read", "name": "Read", "input": {"file_path": "README.md"}}]
        req = copy.deepcopy(REQUEST)
        req["messages"] += [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "PRIVATE_REASONING", "signature": "PRIVATE_SIGNATURE"},
                visible[0], {"type": "redacted_thinking", "data": "PRIVATE_REDACTED_DATA"},
                visible[1], {"type": "thinking", "thinking": "PRIVATE_SECOND_REASONING"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old_read",
                                            "content": "README contents", "is_error": False}]}]
        original = copy.deepcopy(req)
        dropped = []
        body = prepare_request(req, "m", dropped)
        self.assertEqual(req, original)
        self.assertEqual(body["transcript"]["messages"][1], {"role": "assistant", "content": visible})
        self.assertEqual(body["transcript"]["messages"][2], req["messages"][2])
        self.assertNotIn("PRIVATE", json.dumps(body["transcript"]))
        self.assertNotIn("PRIVATE", json.dumps(input_content(body)))
        self.assertEqual([field for field in dropped if "content blocks" in field], [
            "redacted_thinking content blocks (omitted from SDK history)",
            "thinking content blocks (omitted from SDK history)"])

    def test_thinking_only_assistant_turn_remains_a_valid_history_record(self):
        req = copy.deepcopy(REQUEST)
        req["messages"] += [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "PRIVATE_REASONING"}]},
            {"role": "user", "content": "Continue."}]
        body = prepare_request(req, "m", [])
        self.assertEqual(body["transcript"]["messages"][1], {"role": "assistant", "content": []})
        content = input_content(body)
        restored = asyncio.run(sdk_transcript(self._stream(content)))
        self.assertEqual(restored, body["transcript"])
        self.assertNotIn("PRIVATE", json.dumps(content))

    def test_inline_tool_changes_preserve_history_and_update_active_schema(self):
        req = inline_tool_request()
        original = copy.deepcopy(req)
        body = prepare_request(req, "m", [])
        self.assertEqual(req, original)
        self.assertEqual(body["transcript"]["messages"][-1], req["messages"][-1])
        self.assertEqual([tool["name"] for tool in body["transcript"]["tools"]], ["Probe"])
        replacement = {"name": "Probe", "input_schema": {"type": "object", "properties": {"count": {"type": "integer"}}}}
        req["messages"].append({"role": "system", "content": [
            {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": replacement}}]})
        self.assertEqual(prepare_request(req, "m", [])["transcript"]["tools"], [replacement])

    def test_deferred_tools_load_from_search_results_and_can_be_removed_and_reoffered(self):
        deferred = {**copy.deepcopy(TOOL), "name": "Probe", "defer_loading": True}
        req = {**copy.deepcopy(REQUEST), "tools": [TOOL, deferred]}
        names = lambda: [tool["name"] for tool in prepare_request(req, "m", [])["transcript"]["tools"]]
        self.assertEqual(names(), ["Read"])
        reference = {"type": "tool_reference", "tool_name": "Probe"}
        req["messages"] += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "search", "name": "ToolSearch", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "search", "content": [reference]}]}]
        self.assertEqual(names(), ["Read", "Probe"])
        body = prepare_request(req, "m", [])
        self.assertEqual(body["transcript"]["messages"][-1]["content"][0]["content"], [reference])
        for change, expected in (("tool_removal", ["Read"]), ("tool_addition", ["Read", "Probe"])):
            req["messages"].append({"role": "system", "content": [
                {"type": change, "tool": {"type": "tool_reference", "name": "Probe"}}]})
            self.assertEqual(names(), expected)

    def test_tool_choice_is_checked_after_historical_tool_changes(self):
        req = inline_tool_request()
        req["params"] = {"tool_choice": {"type": "tool", "name": "Probe"}}
        self.assertEqual(prepare_request(req, "m", [])["transcript"]["tool_choice"]["name"], "Probe")
        req["params"]["tool_choice"]["name"] = "Read"
        with self.assertRaises(UnsupportedSDKRequest):
            prepare_request(req, "m", [])

    def test_invalid_or_unresolved_tool_changes_are_rejected(self):
        cases = [
            {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "unknown"}}]},
            {"role": "system", "content": [{"type": "tool_removal", "tool": {"type": "tool_reference", "name": "unknown"}}]},
            {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_definition", "definition": {"type": "web_search_20250305", "name": "web_search"}}}]},
            {"role": "user", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "Read"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "search", "content": [{"type": "tool_reference", "tool_name": "unknown"}]}]},
        ]
        for message in cases:
            with self.subTest(message=message), self.assertRaises(UnsupportedSDKRequest):
                prepare_request({**REQUEST, "messages": REQUEST["messages"] + [message]}, "m", [])

    def test_images_in_user_messages_and_tool_results_keep_their_positions(self):
        req = copy.deepcopy(REQUEST)
        req["messages"] += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "read_image",
                "name": "Read", "input": {"file_path": "image.png"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read_image",
                "content": [IMAGE, {"type": "text", "text": "screenshot"}]}]},
            {"role": "user", "content": [IMAGE]}]
        original = copy.deepcopy(req)
        body = prepare_request(req, "m", [])
        messages = body["transcript"]["messages"]
        self.assertEqual(messages[2]["content"][0]["tool_use_id"], "read_image")
        self.assertEqual(messages[2]["content"][0]["content"][0],
                         {"type": "image", "attachment_id": "image_1"})
        self.assertEqual(messages[3]["content"][0], {"type": "image", "attachment_id": "image_2"})
        self.assertEqual(body["images"], [IMAGE, IMAGE])
        self.assertNotIn(IMAGE["source"]["data"], json.dumps(body["transcript"]))
        self.assertEqual(req, original)

    def test_mid_conversation_system_instructions_keep_their_role_and_position(self):
        req = copy.deepcopy(REQUEST)
        req["model"] = "claude-opus-5-5"
        req["messages"] += [
            {"role": "system", "content": [{"type": "text", "text": "Use brief answers now."}],
             "output_config": {"effort": "xhigh"}},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Read the file."},
            {"role": "system", "content": "Reply in Japanese."},
        ]
        original = copy.deepcopy(req)
        dropped = []
        body = prepare_request(req, req["model"], dropped)
        transcript = body["transcript"]
        self.assertEqual(transcript["system"], req["system"])
        self.assertEqual(transcript["messages"], [
            {"role": m["role"], "content": m["content"]} for m in req["messages"]])
        self.assertIn("messages[1].output_config", dropped)
        self.assertEqual(req, original)

    def test_invalid_message_roles_still_fail_with_the_message_index(self):
        for message in ({"role": "bogus", "content": "x"}, {"content": "x"}, "bad message"):
            with self.subTest(message=message), self.assertRaisesRegex(UnsupportedSDKRequest, r"messages\[1\]"):
                prepare_request({**REQUEST, "messages": [REQUEST["messages"][0], message]}, "m", [])

    def test_server_tools_and_bad_tool_choice_fail_explicitly(self):
        for changes in ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
                        {"params": {"tool_choice": {"type": "tool", "name": "missing"}}}):
            with self.subTest(changes=changes), self.assertRaises(UnsupportedSDKRequest):
                prepare_request({**REQUEST, **changes}, "m", [])

    def test_upstream_default_ignores_frontend_base_url(self):
        with patch.dict(os.environ, {"ANTHROPIC_BASE_URL": "http://localhost:8765"}):
            self.assertEqual(config().target, "https://api.anthropic.com")
        with self.assertRaisesRegex(ValueError, "point back"):
            AgentSDKBackend(config("--puppet", "http://localhost:8765", "--target", "http://127.0.0.1:8765/v1"))
        self.assertFalse(looks_foreign("claude-sonnet-4-6", "agent-sdk"))
        self.assertTrue(looks_foreign("gpt-4o", "agent-sdk"))


@unittest.skipUnless(HAS_DEPS, "install .[test,agent-sdk]")
class TestSDKBackend(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        answer_without_gate(self)
        self.backend = AgentSDKBackend(config())
        self.body = self.backend.prepare(REQUEST, "claude-sonnet-4-6", [])

    def test_tool_input_validation_and_ids(self):
        first = self.backend.decode(CALL, self.body["transcript"])
        second = self.backend.decode(CALL, self.body["transcript"])
        self.assertEqual(first["content"][1]["input"], {"file_path": "README.md"})
        self.assertNotEqual(first["content"][1]["id"], second["content"][1]["id"])
        for invalid in ('[]', '{"file_path":42}', '{"file_path":NaN}', 'bad json'):
            data = copy.deepcopy(CALL)
            data["content"][1]["input_json"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(SDKOutputError):
                self.backend.decode(data, self.body["transcript"])

    def test_unknown_tools_and_missing_output_are_rejected(self):
        invalid = copy.deepcopy(CALL)
        invalid["content"][1]["name"] = "Bash"
        for data in (invalid, None, {"content": [], "stop_reason": "end_turn"}):
            with self.subTest(data=data), self.assertRaises(SDKOutputError):
                self.backend.decode(data, self.body["transcript"])

    def test_declared_boolean_strings_are_normalized_in_tool_input(self):
        body = copy.deepcopy(self.body)
        body["transcript"]["tools"][0]["input_schema"]["properties"]["allow_large"] = {"type": "boolean"}
        for value, expected in (("true", True), ("false", False), (True, True), (False, False)):
            data = copy.deepcopy(CALL)
            data["content"][1]["input_json"] = json.dumps({"file_path": "README.md", "allow_large": value})
            original = copy.deepcopy(data)
            with self.subTest(value=value):
                decoded = self.backend.decode(data, body["transcript"])
                self.assertIs(decoded["content"][1]["input"]["allow_large"], expected)
                self.assertEqual(data, original)

    def test_required_tool_defaults_are_applied_without_guessing_other_values(self):
        body = self.backend.prepare({**REQUEST, "tools": [{"name": "ToolSearch", "input_schema": {
            "type": "object", "properties": {"query": {"type": "string"},
                "max_results": {"type": "number", "default": 5}},
            "required": ["query", "max_results"], "additionalProperties": False}}]}, "m", [])
        data = {"content": [{"type": "tool_use", "name": "ToolSearch",
                             "input_json": '{"query":"select:Read"}'}], "stop_reason": "tool_use"}
        decoded = self.backend.decode(data, body["transcript"])
        self.assertEqual(decoded["content"][0]["input"], {"query": "select:Read", "max_results": 5})
        data["content"][0]["input_json"] = '{}'
        with self.assertRaisesRegex(RuntimeError, "query"):
            self.backend.decode(data, body["transcript"])

    def test_declared_numeric_strings_are_normalized_but_still_validated(self):
        body = copy.deepcopy(self.body)
        fields = body["transcript"]["tools"][0]["input_schema"]["properties"]
        fields.update({"timeout": {"type": "number", "minimum": 0},
                       "offset": {"type": "integer", "minimum": 0}})
        for value, expected in (("120000", 120000), ("1.5", 1.5), ("1e3", 1000), (120000, 120000)):
            data = copy.deepcopy(CALL)
            data["content"][1]["input_json"] = json.dumps({"file_path": "123", "timeout": value, "offset": "2"})
            with self.subTest(value=value):
                decoded = self.backend.decode(data, body["transcript"])["content"][1]["input"]
                self.assertEqual(decoded, {"file_path": "123", "timeout": expected, "offset": 2})
        for key, value in (("timeout", "NaN"), ("timeout", "1e999"), ("timeout", "-1"),
                           ("timeout", "120s"), ("timeout", "true"), ("offset", "1.5")):
            data["content"][1]["input_json"] = json.dumps({"file_path": "123", key: value})
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.backend.decode(data, body["transcript"])

    def test_stringified_question_arrays_and_nested_defaults_are_normalized(self):
        body = self.backend.prepare({**REQUEST, "tools": [QUESTION_TOOL]}, "m", [])
        questions = copy.deepcopy(QUESTIONS)
        questions[0]["options"] = json.dumps(questions[0]["options"])
        for multi_select in (None, "false", "true"):
            args = copy.deepcopy(questions)
            if multi_select is not None:
                args[0]["multiSelect"] = multi_select
            data = question_call({"questions": json.dumps(args)})
            original = copy.deepcopy(data)
            decoded = self.backend.decode(data, body["transcript"])["content"][0]["input"]
            expected = [{**QUESTIONS[0], "multiSelect": multi_select == "true"}]
            with self.subTest(multi_select=multi_select):
                self.assertEqual(decoded["questions"], expected)
                self.assertIs(decoded["questions"][0]["multiSelect"], multi_select == "true")
                self.assertEqual(data, original)

    def test_typed_objects_are_decoded_and_string_values_are_preserved(self):
        body = self.backend.prepare({**REQUEST, "tools": [QUESTION_TOOL]}, "m", [])
        questions = [json.dumps({**QUESTIONS[0], "multiSelect": "false"})]
        data = question_call({"questions": json.dumps(questions), "answers": '{"answer":"[true]"}'})
        decoded = self.backend.decode(data, body["transcript"])["content"][0]["input"]
        self.assertEqual(decoded["questions"], [{**QUESTIONS[0], "multiSelect": False}])
        self.assertEqual(decoded["answers"], {"answer": "[true]"})

    def test_invalid_question_arrays_are_not_guessed_or_silently_repaired(self):
        body = self.backend.prepare({**REQUEST, "tools": [QUESTION_TOOL]}, "m", [])
        invalid_questions = ["not JSON", "null", "{}", QUESTIONS[0], "[]", "[1]",
            json.dumps([{**QUESTIONS[0], "options": QUESTIONS[0]["options"][:1]}]),
            json.dumps([{**QUESTIONS[0], "extra": "SECRET"}]),
            json.dumps([{**QUESTIONS[0], "multiSelect": "SECRET"}]),
            json.dumps(QUESTIONS * 5)]
        for value in invalid_questions:
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "questions") as caught:
                self.backend.decode(question_call({"questions": value}), body["transcript"])
            self.assertNotIn("SECRET", str(caught.exception))

    def test_union_types_keep_valid_json_strings_as_strings(self):
        tool = {"name": "Echo", "input_schema": {"type": "object", "properties": {
            "value": {"type": ["string", "array"], "items": {"type": "boolean"}}}, "required": ["value"]}}
        body = self.backend.prepare({**REQUEST, "tools": [tool]}, "m", [])
        data = {"content": [{"type": "tool_use", "name": "Echo", "input_json": '{"value":"[true]"}'}],
                "stop_reason": "tool_use"}
        decoded = self.backend.decode(data, body["transcript"])["content"][0]["input"]
        self.assertEqual(decoded, {"value": "[true]"})

    async def test_images_are_sent_as_sdk_content_blocks(self):
        received = []
        async def query(*, prompt, options):
            received.extend([msg async for msg in prompt])
            yield result(TEXT)
        self.backend.sdk = fake_sdk(query)
        req = {**REQUEST, "messages": [{"role": "user", "content": [IMAGE]}]}
        await self.backend.generate(self.backend.prepare(req, "m", []))
        content = received[0]["message"]["content"]
        self.assertEqual(content[2], {"type": "text", "text": "Attachment image_1:"})
        self.assertEqual(content[3], {**IMAGE, "cache_control": {"type": "ephemeral"}})
        self.assertEqual(json.loads(content[1]["text"])["content"][0]["attachment_id"], "image_1")

    def test_nonfinite_numbers_are_rejected_before_serializing(self):
        schema = {"type": "object", "properties": {"timeout": {"type": "number"},
            "values": {"type": "array", "items": {"type": "number"}}}}
        body = self.backend.prepare({**REQUEST, "tools": [{"name": "Probe", "input_schema": schema}]}, "m", [])
        for literal in ('{"timeout":1e999}', '{"timeout":-1e999}', '{"values":[1e999]}',
                        '{"extra":{"value":1e999}}'):
            data = {"content": [{"type": "tool_use", "name": "Probe", "input_json": literal}],
                    "stop_reason": "tool_use"}
            with self.subTest(literal=literal), self.assertRaisesRegex(SDKOutputError, "finite JSON"):
                self.backend.decode(data, body["transcript"])

    def test_dynamic_map_keys_are_hidden_but_schema_fields_are_preserved(self):
        for mapping in ({"additionalProperties": {"type": "object", "properties": {"count": {"type": "number"}}}},
                        {"patternProperties": {".*": {"type": "object", "properties": {"count": {"type": "number"}}}}}):
            schema = {"type": "object", "properties": {"answers": {"type": "object", **mapping}}}
            body = self.backend.prepare({**REQUEST, "tools": [{"name": "Probe", "input_schema": schema}]}, "m", [])
            data = {"content": [{"type": "tool_use", "name": "Probe", "input_json": json.dumps(
                {"answers": {"PRIVATE_QUESTION": {"count": "PRIVATE_VALUE"}}})}], "stop_reason": "tool_use"}
            with self.subTest(mapping=mapping), self.assertRaises(SDKOutputError) as caught:
                self.backend.decode(data, body["transcript"])
            self.assertIn('$["answers"]["<key>"]["count"]', str(caught.exception))
            self.assertNotIn("PRIVATE", str(caught.exception))

    def test_tool_choice_enforced_after_generation(self):
        for choice, data in (({"type": "none"}, CALL), ({"type": "any"}, TEXT),
                             ({"type": "tool", "name": "Read"}, TEXT),
                             ({"type": "auto", "disable_parallel_tool_use": True},
                              {"content": CALL["content"] * 2, "stop_reason": "tool_use"})):
            transcript = {**self.body["transcript"], "tool_choice": choice}
            with self.subTest(choice=choice), self.assertRaisesRegex(SDKOutputError, "tool_choice"):
                self.backend.decode(data, transcript)

    async def test_sdk_is_isolated_and_receives_structured_output_contract(self):
        seen = []
        async def query(*, prompt, options):
            seen.append(options)
            self.assertTrue(os.path.isdir(options.cwd))
            self.assertEqual((await sdk_transcript(prompt))["tools"], [TOOL])
            self.assertEqual(options.tools, [])
            self.assertEqual(options.setting_sources, [])
            self.assertTrue(options.strict_mcp_config)
            self.assertEqual(options.mcp_servers, {})
            self.assertEqual(json.loads(options.settings), {"disableAllHooks": True})
            self.assertIn("no-session-persistence", options.extra_args)
            self.assertTrue(options.verbatim_prompts)
            self.assertTrue(options.include_partial_messages)
            self.assertEqual(options.max_buffer_size, 4 * 1024 * 1024)
            self.assertTrue(options.env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:"))
            self.assertEqual(options.env["CLAUDE_CODE_USE_MANTLE"], "0")
            self.assertIn("CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE", options.env)
            self.assertNotIn("ENABLE_PROMPT_CACHING_1H", options.env)
            yield result(CALL)
        self.backend.sdk = fake_sdk(query)
        payload = await self.backend.generate(self.body)
        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["usage"]["input_tokens"], 100)
        self.assertEqual(seen[0].cwd, os.path.join(self.backend.cfg.sdk_state_dir, "sdk-cwd"))
        self.assertEqual(os.listdir(seen[0].cwd), [])

    async def test_request_effort_reaches_the_sdk_options(self):
        seen = []
        async def query(*, prompt, options):
            seen.append(options.effort)
            yield result(TEXT)
        self.backend.sdk = fake_sdk(query)
        await self.backend.generate({**self.body, "effort": "low"})
        await self.backend.generate(self.body)
        self.assertEqual(seen, ["low", "high"])

    async def test_an_answer_that_never_used_the_api_gate_is_refused(self):
        from puppetllm.agent_sdk import SDKGateBypassError
        from puppetllm.sdk_egress import SDKAPIGate
        async def query(**kw):
            yield result(TEXT)
        self.backend.sdk = fake_sdk(query)
        with patch.object(SDKAPIGate, "used", False), self.assertRaises(SDKGateBypassError):
            await self.backend.generate(self.body)

    async def test_a_blocked_extra_request_reports_the_first_upstream_outcome(self):
        from puppetllm.agent_sdk import SDKAPILimitError
        from puppetllm.sdk_egress import SDKAPIGate
        async def query(**kw):
            yield result(None, is_error=True, subtype="error_during_execution")
        self.backend.sdk = fake_sdk(query)
        original = SDKAPIGate.__aenter__

        def gate_state(**values):
            async def enter(gate):
                entered = await original(gate)
                vars(gate).update(blocked_requests=1, **values)
                return entered
            return patch.object(SDKAPIGate, "__aenter__", enter)

        with gate_state(status_codes=[404]), self.assertRaises(SDKRunError) as caught:
            await self.backend.generate(self.body)
        self.assertEqual((caught.exception.status, caught.exception.error_type), (404, "not_found_error"))
        with gate_state(status_codes=[200], stop_reason="refusal", usage={"output_tokens": 3}):
            self.assertEqual(await self.backend.generate(self.body),
                             {"content": [], "stop_reason": "refusal", "usage": {"output_tokens": 3}})
        with gate_state(status_codes=[200]), self.assertRaises(SDKAPILimitError):
            await self.backend.generate(self.body)

    async def test_failed_or_unstructured_result_never_becomes_an_answer(self):
        for msg in (result(None, result="internal status, not an answer"),
                    result(TEXT, is_error=True), result(TEXT, subtype="error_max_turns")):
            async def query(**kw):
                yield msg
            self.backend.sdk = fake_sdk(query)
            with self.subTest(msg=msg), self.assertRaises(RuntimeError):
                await self.backend.generate(self.body)

    async def test_direct_tool_request_is_returned_before_inner_disabled_error(self):
        import claude_agent_sdk as sdk
        closed = asyncio.Event()
        async def query(**kw):
            try:
                for event in native_events([
                    sdk.TextBlock("ファイルを確認します。"),
                    sdk.ToolUseBlock(id="inner_read", name="Read", input={"file_path": "README.md"})]):
                    yield event
                self.fail("must hand off before the inner runtime denies the tool")
            finally:
                closed.set()
        self.backend.sdk = fake_sdk(query)
        response = await self.backend.generate(self.body)
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["content"][1]["name"], "Read")
        self.assertEqual(response["content"][1]["input"], {"file_path": "README.md"})
        self.assertTrue(closed.is_set())

    async def test_direct_tool_request_still_validates_input_and_tool_choice(self):
        import claude_agent_sdk as sdk
        for args, choice in (({"file_path": 42}, {"type": "auto"}),
                             ({"file_path": "README.md"}, {"type": "none"})):
            async def query(**kw):
                for event in native_events([sdk.ToolUseBlock(id="inner", name="Read", input=args)]):
                    yield event
                yield result(TEXT)
            self.backend.sdk = fake_sdk(query)
            body = copy.deepcopy(self.body)
            body["transcript"]["tool_choice"] = choice
            with self.subTest(args=args, choice=choice), self.assertRaises(SDKOutputError):
                await self.backend.generate(body)

    async def test_direct_tools_wait_for_complete_message_and_final_usage(self):
        import claude_agent_sdk as sdk
        closed = asyncio.Event()
        async def query(**kw):
            try:
                for event in native_events([sdk.TextBlock("Before."),
                        sdk.ToolUseBlock(id="a", name="Read", input={"file_path": "A.txt"}),
                        sdk.ToolUseBlock(id="b", name="Read", input={"file_path": "B.txt"})]):
                    yield event
                self.fail("do not wait for disabled-tool feedback after message_stop")
            finally:
                closed.set()
        self.backend.sdk = fake_sdk(query)
        payload = await self.backend.generate(self.body)
        self.assertEqual([b["type"] for b in payload["content"]], ["text", "tool_use", "tool_use"])
        self.assertEqual(payload["content"][0]["text"], "Before.")
        self.assertEqual([b["input"]["file_path"] for b in payload["content"][1:]], ["A.txt", "B.txt"])
        self.assertEqual(payload["usage"]["output_tokens"], 77)
        self.assertTrue(closed.is_set())

    async def test_incomplete_or_truncated_native_tools_are_never_handed_off(self):
        import claude_agent_sdk as sdk
        for stop_reason in (None, "max_tokens"):
            async def query(**kw):
                events = list(native_events([sdk.ToolUseBlock(id="a", name="Read", input={"file_path": "A.txt"})],
                                             stop_reason=stop_reason))
                for event in (events if stop_reason else events[:-2]):
                    yield event
                yield result(TEXT)
            self.backend.sdk = fake_sdk(query)
            with self.subTest(stop_reason=stop_reason), self.assertRaises(SDKOutputError):
                await self.backend.generate(self.body)

    async def test_midstream_error_discards_tools_and_preserves_error_status(self):
        import claude_agent_sdk as sdk
        async def query(**kw):
            yield stream_event("message_start", message={"id": "partial"})
            yield sdk.AssistantMessage(model="m", message_id="partial", content=[
                sdk.ToolUseBlock(id="a", name="Read", input={"file_path": "A.txt"})])
            yield stream_event("error", error={"type": "api_error"})
            yield result(None, is_error=True, subtype="error_during_execution", api_error_status=500)
        self.backend.sdk = fake_sdk(query)
        with self.assertRaises(SDKRunError) as caught:
            await self.backend.generate(self.body)
        self.assertEqual(caught.exception.status, 500)

    async def test_oversize_cli_message_reports_limit_without_raw_output(self):
        import claude_agent_sdk as sdk
        async def query(**kw):
            self.assertEqual(kw["options"].max_buffer_size, 2097152)
            raise sdk.CLIJSONDecodeError("PRIVATE_BODY", ValueError("Buffer size 2097153 exceeds limit 2097152"))
            yield
        backend = AgentSDKBackend(config("--sdk-max-buffer-bytes", "2097152"))
        backend.sdk = fake_sdk(query)
        with self.assertRaisesRegex(SDKOutputError, "sdk-max-buffer-bytes=2097152") as caught:
            await backend.generate(self.body)
        self.assertNotIn("PRIVATE", str(caught.exception))

    async def test_sdk_error_reports_the_limit_without_leaking_result_text(self):
        import claude_agent_sdk as sdk
        async def query(**kw):
            raise sdk.ResultError("SECRET raw result", data={
                "subtype": "error_max_turns", "terminal_reason": "max_turns", "errors": ["SECRET"]})
            yield
        self.backend.sdk = fake_sdk(query)
        with self.assertRaisesRegex(RuntimeError, "sdk-max-turns") as caught:
            await self.backend.generate(self.body)
        self.assertNotIn("SECRET", str(caught.exception))


@unittest.skipUnless(HAS_DEPS, "install .[test,agent-sdk]")
class TestSDKRelay(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from puppetllm import fake_server
        answer_without_gate(self)
        self.fs = importlib.reload(fake_server)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.fs.app), base_url="http://test")
        self.state_directory = tempfile.TemporaryDirectory()
        self.relay = Relay(config("--sdk-state-dir", self.state_directory.name))
        await self.relay.ctl.aclose()
        self.relay.ctl = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.fs.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.relay.close()
        await self.client.aclose()
        self.state_directory.cleanup()

    def test_cancel_request_uses_only_python_310_task_api_and_is_not_repeated(self):
        class Py310Task:
            def __init__(self):
                self.cancel_count = 0
                self.callbacks = []
            def done(self):
                return False
            def cancel(self):
                self.cancel_count += 1
            def add_done_callback(self, callback):
                self.callbacks.append(callback)
        task = Py310Task()  # deliberately has no Task.cancelling()
        self.assertTrue(self.relay._cancel_task(task))
        self.assertFalse(self.relay._cancel_task(task))
        self.assertEqual(task.cancel_count, 1)
        for callback in task.callbacks:
            callback(task)
        self.assertFalse(self.relay._cancel_requested)

    async def exchange(self, body=REQUEST):
        task = asyncio.create_task(self.client.post("/v1/messages", json=body))
        for _ in range(100):
            pending = (await self.client.get("/_control/pending")).json()["pending"]
            if pending:
                await self.relay.handle(pending[0]["pending_id"], pending[0])
                return await asyncio.wait_for(task, 2)
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.fail("request never became pending")

    async def test_tool_roundtrip_streaming_and_provenance(self):
        prompts = []
        async def query(*, prompt, options):
            transcript = await sdk_transcript(prompt)
            prompts.append(transcript)
            yield result(CALL if len(prompts) == 1 else TEXT)
        self.relay.agent_sdk.sdk = fake_sdk(query)
        first = (await self.exchange()).json()
        self.assertEqual(first["stop_reason"], "tool_use")
        call = first["content"][1]
        followup = {**REQUEST, "stream": True, "messages": REQUEST["messages"] + [
            {"role": "assistant", "content": first["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"],
                                           "content": "README contents"}]}]}
        second = await self.exchange(followup)
        self.assertEqual(second.status_code, 200)
        self.assertIn("event: message_stop", second.text)
        self.assertIn("読みました。", second.text)
        self.assertEqual(prompts[1]["messages"][-1]["content"][0]["tool_use_id"], call["id"])
        history = (await self.client.get("/_control/history")).json()
        # History layout is part of the existing control API.
        records = history["history"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["provenance"]["upstream_provider"], "agent-sdk")

    async def test_invalid_generated_call_is_502_and_resolves_pending(self):
        async def query(**kw):
            data = copy.deepcopy(CALL)
            data["content"][1]["name"] = "unoffered_tool_SECRET"
            yield result(data)
        self.relay.agent_sdk.sdk = fake_sdk(query)
        response = await self.exchange()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.headers.get("x-should-retry"), "false")
        self.assertNotIn("SECRET", response.text)
        self.assertFalse((await self.client.get("/_control/pending")).json()["has_pending"])

    async def test_invalid_tool_argument_reports_field_without_echoing_values(self):
        async def query(**kw):
            data = copy.deepcopy(CALL)
            data["content"][1]["input_json"] = json.dumps({"file_path": "PRIVATE_PATH", "allow_large": "SECRET_INVALID"})
            yield result(data)
        self.relay.agent_sdk.sdk = fake_sdk(query)
        req = copy.deepcopy(REQUEST)
        req["tools"][0]["input_schema"]["properties"]["allow_large"] = {"type": "boolean"}
        response = await self.exchange(req)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.headers.get("x-should-retry"), "false")
        self.assertRegex(response.json()["error"]["message"], "Read.*allow_large.*boolean")
        self.assertNotIn("SECRET", response.text)
        self.assertNotIn("PRIVATE_PATH", response.text)
        self.assertFalse((await self.client.get("/_control/pending")).json()["has_pending"])

    async def test_gate_bypass_is_refused_and_stops_the_worker(self):
        from puppetllm.sdk_egress import SDKAPIGate
        calls = []
        async def query(**kw):
            calls.append(1)
            yield result(TEXT)
        self.relay.agent_sdk.sdk = fake_sdk(query)
        with patch.object(SDKAPIGate, "used", False):
            response = await self.exchange()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["x-should-retry"], "false")
        self.assertIn("API gate", response.json()["error"]["message"])
        self.assertFalse(self.relay.sdk_runtime.ready)
        other = await self.exchange({**REQUEST, "messages": [{"role": "user", "content": "another"}]})
        self.assertEqual(other.status_code, 503)
        self.assertEqual(len(calls), 1)

    async def test_too_long_prompt_is_reported_in_the_messages_api_wording(self):
        for kw, message in (
                ({"terminal_reason": "prompt_too_long", "api_error_status": 400,
                  "result": "Prompt is too long · the request is ~250000 tokens (limit 200000) SECRET"},
                 "prompt is too long: 250000 tokens > 200000 maximum"),
                ({"terminal_reason": "blocking_limit", "result": "Prompt is too long"}, "prompt is too long")):
            async def query(kw=kw, **_):
                yield result(None, is_error=True, **kw)
            self.relay.agent_sdk.sdk = fake_sdk(query)
            with self.subTest(reason=kw["terminal_reason"]):
                for _ in range(2):  # nothing ran, so a repeat is not blocked
                    response = await self.exchange()
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json()["error"],
                                     {"type": "invalid_request_error", "message": message})
                self.assertEqual(self.relay.sdk_guard.status(), [])

    async def test_sdk_api_error_preserves_status_without_raw_result(self):
        async def query(**kw):
            yield result(None, is_error=True, api_error_status=401,
                         terminal_reason="api_error", result="SECRET credential details")
        self.relay.agent_sdk.sdk = fake_sdk(query)
        response = await self.exchange()
        self.assertEqual(response.status_code, 401)
        error = response.json()["error"]
        self.assertEqual(error["type"], "authentication_error")
        self.assertIn("HTTP 401", error["message"])
        self.assertNotIn("SECRET", response.text)

    async def test_opus_system_message_reaches_sdk_and_returns_success(self):
        instruction = {"role": "system", "content": [{"type": "text", "text": "Continue in Japanese."}],
                       "output_config": {"effort": "xhigh"}}
        received = []
        async def query(*, prompt, options):
            received.append(await sdk_transcript(prompt))
            self.assertEqual(options.model, "claude-opus-5-5")
            yield result(TEXT)
        self.relay.agent_sdk.sdk = fake_sdk(query)
        response = await self.exchange({**REQUEST, "model": "claude-opus-5-5",
                                        "messages": REQUEST["messages"] + [instruction]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["content"], TEXT["content"])
        self.assertEqual(received[0]["messages"][-1],
                         {k: instruction[k] for k in ("role", "content")})
        history = (await self.client.get("/_control/history")).json()["history"]
        self.assertIn("messages[1].output_config", history[-1]["provenance"]["translation"]["dropped_fields"])

    async def test_unsupported_input_is_400_without_sdk_call(self):
        async def query(**kw):
            self.fail("must not invoke SDK")
            yield
        self.relay.agent_sdk.sdk = fake_sdk(query)
        response = await self.exchange({**REQUEST, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.invalid/x.png"}}]}]})
        self.assertEqual(response.status_code, 400)

    async def test_strict_translation_still_rejects_omitted_thinking_before_sdk_call(self):
        async def query(**kw):
            self.fail("strict mode must not invoke SDK after omitting thinking")
            yield
        self.relay.agent_sdk.sdk = fake_sdk(query)
        self.relay.cfg.strict_translation = True
        response = await self.exchange(thinking_request())
        self.assertEqual(response.status_code, 400)
        self.assertIn("thinking content blocks", response.json()["error"]["message"])

    async def test_inline_tool_catalogue_is_used_for_the_next_declaration(self):
        async def query(*, prompt, options):
            transcript = await sdk_transcript(prompt)
            self.assertEqual([tool["name"] for tool in transcript["tools"]], ["Probe"])
            data = copy.deepcopy(CALL)
            data["content"][1]["name"] = "Probe"
            yield result(data)
        self.relay.agent_sdk.sdk = fake_sdk(query)
        response = await self.exchange(inline_tool_request())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["content"][1]["name"], "Probe")

    async def test_timeout_cancels_sdk_and_resolves_pending(self):
        cancelled = asyncio.Event()
        async def query(**kw):
            try:
                await asyncio.sleep(60)
                yield result()
            finally:
                cancelled.set()
        self.relay.cfg.timeout = 1
        self.relay.agent_sdk.sdk = fake_sdk(query)
        response = await self.exchange()
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.headers.get("x-should-retry"), "false")
        self.assertTrue(cancelled.is_set())

    async def test_timeout_blocks_resends_and_survives_worker_recreation(self):
        calls = []
        async def query(**kw):
            calls.append(1)
            await asyncio.sleep(60)
            yield result()
        self.relay.cfg.timeout = 1
        self.relay.agent_sdk.sdk = fake_sdk(query)
        self.assertEqual((await self.exchange()).status_code, 504)
        repeated = {**REQUEST, "metadata": {"user_id": "a-different-transport-request-id"}}
        self.relay.cfg.timeout = 600
        second = await self.exchange(repeated)
        self.assertEqual(second.status_code, 503)
        self.assertEqual(second.headers.get("x-should-retry"), "false")
        self.assertEqual(len(calls), 1)
        await self.relay.close()
        self.relay = Relay(config("--sdk-state-dir", self.state_directory.name))
        await self.relay.ctl.aclose()
        self.relay.ctl = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.fs.app), base_url="http://test")
        self.relay.agent_sdk.sdk = fake_sdk(query)
        self.assertEqual((await self.exchange()).status_code, 503)
        self.assertEqual(len(calls), 1)

    async def test_a_different_request_is_allowed_after_a_timeout(self):
        calls = []
        async def query(**kw):
            calls.append(1)
            if len(calls) == 1:
                await asyncio.sleep(60)
            yield result(TEXT)
        self.relay.cfg.timeout = 1
        self.relay.agent_sdk.sdk = fake_sdk(query)
        self.assertEqual((await self.exchange()).status_code, 504)
        other = {**REQUEST, "messages": [{"role": "user", "content": "A different job"}]}
        self.assertEqual((await self.exchange(other)).status_code, 200)
        self.assertEqual(len(calls), 2)

    async def test_unknown_delivery_keeps_a_failure_guard(self):
        calls = []
        async def query(**kw):
            calls.append(1)
            yield result(TEXT)
        self.relay.agent_sdk.sdk = fake_sdk(query)
        async def lost_delivery(*args, **kw):
            return httpx.Response(409)
        with patch.object(self.relay, "_post_ctl", lost_delivery):
            await self.relay.handle("lost-delivery", {"request": REQUEST})
        self.assertEqual(self.relay.sdk_guard.status()[0]["reason"], "delivery_error")
        self.assertEqual((await self.exchange()).status_code, 503)
        self.assertEqual(len(calls), 1)

    async def test_shutdown_status_failure_still_closes_guard_and_releases_locks(self):
        from puppetllm.sdk_runtime import StateLock
        guard = self.relay.sdk_guard
        with patch.object(self.relay.sdk_runtime, "write", side_effect=OSError("private detail")):
            await self.relay.close()
        self.assertIsNone(guard._connection)
        with StateLock(self.state_directory.name):
            pass
        await self.relay.close()

    async def test_timeout_cleanup_survives_an_additional_external_cancel(self):
        cleanup_started = asyncio.Event()
        cleanup_finished = asyncio.Event()
        async def query(**kw):
            try:
                await asyncio.sleep(60)
                yield result(TEXT)
            finally:
                cleanup_started.set()
                await asyncio.sleep(0.05)
                cleanup_finished.set()
        self.relay.agent_sdk.sdk = fake_sdk(query)
        self.relay.cfg.timeout = 1
        body = self.relay.agent_sdk.prepare(REQUEST, "m", [])
        key = self.relay.sdk_guard.fingerprint(body)
        self.relay.sdk_runtime.start("cleanup", key, "m")
        task = asyncio.create_task(self.relay._generate_sdk(body, "cleanup"))
        await asyncio.wait_for(cleanup_started.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cleanup_finished.is_set())

    async def test_tilde_state_directory_is_expanded_before_locking(self):
        from pathlib import Path
        private_home(self)
        from puppetllm.sdk_runtime import StateLock, StateBusyError
        with tempfile.TemporaryDirectory(dir=Path.home()) as directory:
            state_path = Path(directory) / "state"
            other = Relay(config("--sdk-state-dir", "~/" + Path(directory).name + "/state"))
            try:
                self.assertEqual(Path(other.cfg.sdk_state_dir), state_path.resolve())
                with self.assertRaises(StateBusyError):
                    StateLock(state_path)
            finally:
                await other.close()

    async def test_a_second_worker_on_the_same_state_is_refused_without_disturbing_the_first(self):
        from puppetllm.sdk_runtime import StateLock, StateBusyError
        with self.assertRaises(ValueError):
            Relay(config("--sdk-state-dir", self.state_directory.name))
        self.assertTrue(self.relay.sdk_runtime.ready)
        self.assertEqual(self.relay.sdk_guard.status(), [])
        with self.assertRaises(StateBusyError):
            StateLock(self.state_directory.name)

    async def test_disconnected_client_releases_slot_for_next_request(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        calls = []
        async def query(**kw):
            calls.append(kw)
            if len(calls) == 1:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            yield result(TEXT)
        self.relay.agent_sdk.sdk = fake_sdk(query)
        self.relay.max_concurrency = 1
        self.relay.cfg.max_requests = 2
        worker = asyncio.create_task(self.relay.run())
        first = asyncio.create_task(self.client.post("/v1/messages", json=REQUEST))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            followup = {**REQUEST, "messages": [{"role": "user", "content": "A different request"}]}
            second = asyncio.create_task(self.client.post("/v1/messages", json=followup))
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            response = await asyncio.wait_for(second, 3)
            self.assertEqual(response.status_code, 200)
            await asyncio.wait_for(worker, 3)
            self.assertTrue(cancelled.is_set())
            self.assertEqual(len(calls), 2)
            self.assertFalse(self.relay.inflight)
            self.assertFalse(self.relay._pending_tasks)
        finally:
            for task in (first, second, worker):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(t for t in (first, second, worker) if t is not None), return_exceptions=True)

    async def test_relay_shutdown_cancels_active_sdk_queries(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def query(**kw):
            try:
                entered.set()
                await asyncio.sleep(60)
                yield result()
            finally:
                cancelled.set()
        self.relay.agent_sdk.sdk = fake_sdk(query)
        self.relay.inflight.add("shutdown-test")
        task = asyncio.create_task(self.relay.handle("shutdown-test", {"request": REQUEST}))
        self.relay._tasks.add(task)
        await asyncio.wait_for(entered.wait(), 2)
        await self.relay.close()
        self.assertTrue(cancelled.is_set())
        self.assertTrue(task.done())
        self.assertFalse(self.relay.inflight)

    async def test_dry_run_and_strict_translation_do_not_invoke_sdk(self):
        async def query(**kw):
            self.fail("must not invoke SDK")
            yield
        self.relay.agent_sdk.sdk = fake_sdk(query)
        for mode in ("dry_run", "strict_translation"):
            setattr(self.relay.cfg, mode, True)
            response = await self.exchange()
            self.assertEqual(response.status_code, 400)
            setattr(self.relay.cfg, mode, False)


@unittest.skipUnless(HAS_DEPS and os.environ.get("PUPPETLLM_TEST_REAL_AGENT_SDK") == "1",
                     "set PUPPETLLM_TEST_REAL_AGENT_SDK=1 for local CLI integration")
class TestNativeSDK(unittest.TestCase):
    """Exercise the real bundled CLI against localhost, never a paid model."""

    def setUp(self):
        private_home(self)

    def test_real_sdk_accepts_signed_history_and_inline_tool_changes(self):
        req = inline_tool_request()
        answer = copy.deepcopy(CALL)
        answer["content"][1]["name"] = "Probe"
        with serve() as server, patch.dict(os.environ, {"PUPPET_TEST_SDK_KEY": "sk-local-test-only"}):
            server.expect().respond(content=[{"type": "tool_use", "id": "envelope",
                "name": "StructuredOutput", "input": answer}]).repeat()
            backend = AgentSDKBackend(config("--target", server.url, "--api-key-env", "PUPPET_TEST_SDK_KEY"))
            payload = asyncio.run(asyncio.wait_for(backend.generate(
                backend.prepare(req, req["model"], [])), 20))
            self.assertEqual(payload["content"][1]["name"], "Probe")
            history = server.history()
            self.assertTrue(history)
            sent = json.dumps([entry["request"] for entry in history])
            self.assertNotIn("PRIVATE", sent)
            self.assertNotIn("UFJJVkFURV9EQVRB", sent)
            self.assertIn("README contents", sent)

    def test_real_sdk_large_output_and_configured_limit(self):
        req = {**REQUEST, "tools": []}
        for text in ("あ" * 400_000,):
            large = {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
            with self.subTest(character=text[0]), serve() as server, \
                    patch.dict(os.environ, {"PUPPET_TEST_SDK_KEY": "sk-local-test-only"}):
                server.expect().respond(content=[{"type": "tool_use", "id": "large", "name": "StructuredOutput",
                                                 "input": large}]).repeat()
                backend = AgentSDKBackend(config("--target", server.url, "--api-key-env", "PUPPET_TEST_SDK_KEY"))
                payload = asyncio.run(asyncio.wait_for(backend.generate(backend.prepare(req, req["model"], [])), 20))
                self.assertEqual(payload["content"], large["content"])
                backend.cfg.sdk_max_buffer_bytes = 1024 * 1024
                with self.assertRaisesRegex(SDKOutputError, "sdk-max-buffer-bytes=1048576"):
                    asyncio.run(asyncio.wait_for(backend.generate(backend.prepare(req, req["model"], [])), 20))

    def test_real_sdk_sends_a_long_history_and_reports_an_upstream_length_error(self):
        req = {**REQUEST, "tools": [], "messages": [{"role": "user", "content": "Long history. " * 220_000}]}
        with serve() as server, patch.dict(os.environ, {"PUPPET_TEST_SDK_KEY": "sk-local-test-only"}):
            server.expect().respond(content=[{"type": "tool_use", "id": "envelope",
                                             "name": "StructuredOutput", "input": TEXT}])
            server.expect().error(status=400, type="invalid_request_error",
                                  message="prompt is too long: 900000 tokens > 200000 maximum")
            backend = AgentSDKBackend(config("--target", server.url, "--api-key-env", "PUPPET_TEST_SDK_KEY"))
            payload = asyncio.run(asyncio.wait_for(backend.generate(backend.prepare(req, req["model"], [])), 30))
            self.assertEqual(payload["content"], TEXT["content"])
            with self.assertRaisesRegex(SDKRunError, r"^prompt is too long: 900000 tokens > 200000 maximum$") as caught:
                asyncio.run(asyncio.wait_for(backend.generate(backend.prepare(req, req["model"], [])), 30))
            self.assertEqual((caught.exception.status, caught.exception.error_type), (400, "invalid_request_error"))
            self.assertEqual(len(server.history()), 2)

    def test_real_sdk_reports_the_upstream_error_behind_a_blocked_retry(self):
        with serve() as server, patch.dict(os.environ, {"PUPPET_TEST_SDK_KEY": "sk-local-test-only"}):
            server.expect().error(status=404, type="not_found_error", message="model: unknown").repeat()
            backend = AgentSDKBackend(config("--target", server.url, "--api-key-env", "PUPPET_TEST_SDK_KEY"))
            with self.assertRaises(SDKRunError) as caught:
                asyncio.run(asyncio.wait_for(backend.generate(backend.prepare(REQUEST, REQUEST["model"], [])), 30))
            self.assertEqual((caught.exception.status, caught.exception.error_type), (404, "not_found_error"))
            self.assertEqual(len(server.history()), 1)

    def test_history_cache_is_reused_across_fresh_sdk_sessions(self):
        # Older models get the CLI's environment (with the working directory)
        # ahead of the history; newer ones after it.
        for model, ttl in (("claude-opus-5-5", "5m"), ("claude-sonnet-4-6", "5m"), ("claude-opus-5-5", "1h")):
            with self.subTest(model=model, ttl=ttl):
                self._check_history_cache(model, ttl)

    def _check_history_cache(self, model, ttl):
        with serve() as server, patch.dict(os.environ, {"PUPPET_TEST_SDK_KEY": "sk-local-test-only"}):
            server.expect().respond(content=[{"type": "tool_use", "id": "envelope",
                                             "name": "StructuredOutput", "input": TEXT}]).repeat()
            req = {**REQUEST, "model": model, "tools": [],
                   "messages": [{"role": "user", "content": "Stable history. " * 1000}]}
            backend = AgentSDKBackend(config("--target", server.url, "--api-key-env", "PUPPET_TEST_SDK_KEY",
                                             "--sdk-cache-ttl", ttl))
            async def run():
                await backend.generate(backend.prepare(req, req["model"], []))
                followup = {**req, "messages": req["messages"] + [
                    {"role": "assistant", "content": TEXT["content"]},
                    {"role": "user", "content": "Continue."}]}
                await backend.generate(backend.prepare(followup, req["model"], []))
            asyncio.run(asyncio.wait_for(run(), 30))
            history = server.history()
            self.assertEqual(len(history), 2)
            before, after = (h["cache"] for h in history)
            self.assertGreater(after["cache_read_tokens"], before["cache_read_tokens"] + 1000)
            self.assertEqual(before["cache_creation_1h_tokens"] > 0, ttl == "1h")
            self.assertEqual(before["cache_creation_5m_tokens"] > 0, ttl == "5m")

    def test_real_sdk_generates_a_tool_declaration_without_executing_it(self):
        self._check_backend_tool_declaration()

    def _check_backend_tool_declaration(self, *, image=False):
        req = copy.deepcopy(REQUEST)
        if image:
            req["messages"] += [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "read_image",
                    "name": "Read", "input": {"file_path": "image.png"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read_image",
                    "content": [IMAGE]}]}]
        with serve() as server, patch.dict(os.environ, {"PUPPET_TEST_SDK_KEY": "sk-local-test-only"}):
            server.expect().respond(content=[{"type": "tool_use", "id": "toolu_envelope",
                                             "name": "StructuredOutput", "input": CALL}]).repeat()
            backend = AgentSDKBackend(config("--target", server.url, "--api-key-env", "PUPPET_TEST_SDK_KEY"))
            payload = asyncio.run(asyncio.wait_for(backend.generate(backend.prepare(req, req["model"], [])), 30))
            calls = [b for b in payload["content"] if b["type"] == "tool_use"]
            self.assertEqual(calls[0]["input"], {"file_path": "README.md"})
            self.assertEqual(payload["stop_reason"], "tool_use")
            (entry,) = server.history()
            sent = entry["request"]
            self.assertEqual([t["name"] for t in sent["tools"]], ["StructuredOutput"])
            self.assertLessEqual(json.dumps([sent.get("system"), sent["tools"], sent["messages"]])
                                 .count('"cache_control"'), 4)
            blocks = [b for m in sent["messages"] if m["role"] == "user" for b in m["content"]]
            context = next(json.loads(b["text"]) for b in blocks if b["type"] == "text"
                           and b["text"].startswith('{"record_type":"context"'))
            self.assertEqual(context["tools"], req["tools"])
            if image:
                self.assertEqual([b["source"] for b in blocks if b["type"] == "image"], [IMAGE["source"]])

    def test_real_sdk_receives_an_image_from_an_external_tool_result(self):
        self._check_backend_tool_declaration(image=True)

    def test_real_claude_frontend_executes_read_and_sends_back_result(self):
        self._check_frontend_read_roundtrip(REQUEST["model"])

    def test_real_opus_frontend_executes_read_and_sends_back_result(self):
        self._check_frontend_read_roundtrip("claude-opus-5-5")

    def _check_frontend_read_roundtrip(self, model):
        import claude_agent_sdk as sdk
        from pathlib import Path
        with serve() as server, tempfile.TemporaryDirectory(prefix="puppetllm-front-") as cwd:
            path = Path(cwd) / "fixture.txt"
            path.write_text("UNIQUE_FRONTEND_FILE_CONTENT", encoding="utf-8")
            prompts = []
            async def go():
                relay = Relay(config("--puppet", server.url, "--max-requests", "2", "--timeout", "20",
                                     "--sdk-state-dir", str(Path(cwd) / "state")))
                async def inner_query(*, prompt, options):
                    prompts.append(await sdk_transcript(prompt))
                    data = copy.deepcopy(CALL if len(prompts) == 1 else TEXT)
                    if len(prompts) == 1:
                        data["content"][1]["input_json"] = json.dumps({"file_path": str(path)})
                        if model == "claude-opus-5-5":
                            for event in native_events([
                                sdk.ToolUseBlock(id="direct_read", name="Read", input={"file_path": str(path)})], model=model):
                                yield event
                            self.fail("a direct Read should end the inner query")
                    yield result(data)
                answer_without_gate(self)
                relay.agent_sdk.sdk = fake_sdk(inner_query)
                worker = asyncio.create_task(relay.run())
                try:
                    options = sdk.ClaudeAgentOptions(
                        model=model, tools=["Read"], allowed_tools=["Read"],
                        cwd=cwd, setting_sources=[], strict_mcp_config=True,
                        permission_mode="dontAsk", max_turns=3,
                        settings='{"disableAllHooks":true}',
                        env={"ANTHROPIC_BASE_URL": server.url, "ANTHROPIC_API_KEY": "sk-local-only",
                             "ANTHROPIC_AUTH_TOKEN": "", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                             "CLAUDE_CODE_USE_BEDROCK": "0", "CLAUDE_CODE_USE_VERTEX": "0",
                             "CLAUDE_CODE_USE_FOUNDRY": "0", "CLAUDE_CODE_USE_ANTHROPIC_AWS": "0"},
                        extra_args={"no-session-persistence": None, "disable-slash-commands": None})
                    results = [m async for m in sdk.query(prompt=f"Read {path}", options=options)
                               if isinstance(m, sdk.ResultMessage)]
                    self.assertFalse(results[-1].is_error)
                    await asyncio.wait_for(worker, 5)
                finally:
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
            asyncio.run(asyncio.wait_for(go(), 40))
            self.assertEqual(len(prompts), 2)
            returned = [b for m in prompts[1]["messages"] if isinstance(m["content"], list)
                        for b in m["content"] if b.get("type") == "tool_result"]
            self.assertIn("UNIQUE_FRONTEND_FILE_CONTENT", json.dumps(returned))


if __name__ == "__main__":
    unittest.main()
