"""Opt-in SDK progress/retry checks against an isolated localhost fake API.

PUPPETLLM_TEST_REAL_AGENT_SDK=1 enables the bundled CLI. These tests never use
an inference service; they also run under Docker --network none.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from puppetllm.agent_sdk import AgentSDKBackend, SDKOutputLimitError, SDKRunError
from puppetllm.sdk_runtime import RuntimeSnapshot
from puppetllm.testing import serve
from puppetllm.tests.test_agent_sdk import (HAS_DEPS, QUESTION_TOOL, QUESTIONS, REQUEST, answer_without_gate,
                                            config, fake_sdk, private_home, result, stream_event)


RETRY_ENV = {
    "CLAUDE_CODE_MAX_RETRIES": "0",
    "CLAUDE_CODE_RETRY_WATCHDOG": "0",
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
    "MAX_STRUCTURED_OUTPUT_RETRIES": "1",
}


@unittest.skipUnless(HAS_DEPS, "install .[agent-sdk]")
class TestBackendProgressWiring(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        answer_without_gate(self)

    async def test_effort_and_output_limit_override_inheritance_and_change_guard_settings(self):
        from puppetllm.sdk_guard import SDKGuard

        backend = AgentSDKBackend(config("--sdk-effort", "low", "--sdk-max-output-tokens", "4096"))
        body = backend.prepare(REQUEST, REQUEST["model"], [])
        async def query(*, prompt, options):
            self.assertEqual(options.effort, "low")
            self.assertEqual(options.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "4096")
            yield result({"content": [{"type": "text", "text": "fixture"}], "stop_reason": "end_turn"})
        backend.sdk = fake_sdk(query)
        with patch.dict(os.environ, {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"}):
            await backend.generate(body)
        default = AgentSDKBackend(config())
        high = AgentSDKBackend(config("--sdk-effort", "high", "--sdk-max-output-tokens", "auto"))
        auto = AgentSDKBackend(config("--sdk-effort", "auto", "--sdk-max-output-tokens", "auto"))
        self.assertEqual(default.generation_settings(), high.generation_settings())
        self.assertEqual(default.generation_settings()["effort"], "high")
        self.assertNotIn("effort", auto.generation_settings())
        self.assertNotIn("max_output_tokens", auto.generation_settings())
        with tempfile.TemporaryDirectory() as directory, SDKGuard(Path(directory) / "guard.sqlite3") as guard:
            original = guard.fingerprint(body, generation_settings=default.generation_settings())
            changed = guard.fingerprint(body, generation_settings=backend.generation_settings())
            self.assertNotEqual(original, changed)

    async def test_backend_overrides_inherited_retries_and_forwards_progress(self):
        with tempfile.TemporaryDirectory(prefix="sdk-progress-wiring-") as directory:
            logs = []
            runtime = RuntimeSnapshot(directory, log=logs.append, interval=0)
            runtime.start("local-pending", "local-key", "claude-sonnet-4-6")
            backend = AgentSDKBackend(config())
            payload = {"content": [{"type": "text", "text": "PRIVATE_RESPONSE"}],
                       "stop_reason": "end_turn"}

            async def query(*, prompt, options):
                self.assertTrue(options.include_partial_messages)
                self.assertEqual(options.effort, "high")
                self.assertEqual(options.max_turns, 1)
                for name, value in RETRY_ENV.items():
                    self.assertEqual(options.env[name], value)
                options.stderr("PRIVATE_STDERR")
                yield stream_event("message_start", message={"usage": {"input_tokens": 10}})
                yield stream_event("content_block_delta", index=0,
                    delta={"type": "text_delta", "text": "PRIVATE_PARTIAL_TEXT"})
                yield result(payload, num_turns=2, total_cost_usd=0.25)

            backend.sdk = fake_sdk(query)
            with patch.dict(os.environ, {
                "CLAUDE_CODE_MAX_RETRIES": "5", "CLAUDE_CODE_RETRY_WATCHDOG": "1",
                "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "0",
                "MAX_STRUCTURED_OUTPUT_RETRIES": "5",
            }):
                response = await backend.generate(backend.prepare(REQUEST, REQUEST["model"], []),
                    progress=runtime.for_request("local-pending"))
            self.assertEqual(response["content"], payload["content"])
            record = runtime.requests["local-pending"]
            self.assertEqual(record["stderr_lines"], 1)
            self.assertEqual(record["event_count"], 3)
            self.assertEqual(record["generation_events"], 1)
            self.assertEqual(record["sdk_estimated_cost_usd"], 0.25)
            self.assertEqual(record["sdk_reported_num_turns"], 2)
            stages = [json.loads(line.removeprefix("sdk_progress "))["stage"] for line in logs]
            self.assertIn("connected", stages)
            self.assertIn("waiting_sdk_event", stages)
            self.assertEqual(stages[-1], "validating")
            runtime.write([])
            persisted = (Path(directory) / "status.json").read_text()
            self.assertNotIn("PRIVATE", persisted + "\n".join(logs))


@unittest.skipUnless(HAS_DEPS and os.environ.get("PUPPETLLM_TEST_REAL_AGENT_SDK") == "1",
                     "set PUPPETLLM_TEST_REAL_AGENT_SDK=1 for local CLI integration")
class TestRealSDKRuntime(unittest.TestCase):
    def setUp(self):
        private_home(self)

    def test_partial_tool_at_output_limit_cannot_trigger_another_upstream_request(self):
        from puppetllm import fake_server as fs
        original = fs.stream_event_dicts

        def truncated(*args, **kwargs):
            emitted = False
            events = []
            for kind, event in original(*args, **kwargs):
                if kind == "content_block_stop":
                    continue
                if kind == "content_block_delta" and (event.get("delta") or {}).get("type") == "input_json_delta":
                    if emitted:
                        continue
                    emitted = True
                    event = {**event, "delta": {"type": "input_json_delta",
                             "partial_json": '{"private":"PRIVATE_UNFINISHED'}}
                events.append((kind, event))
            return events

        for name in ("Read", "StructuredOutput"):
            with self.subTest(tool=name), serve() as server, tempfile.TemporaryDirectory() as directory:
                server.expect().respond(content=[{"type": "tool_use", "id": "local-truncated",
                    "name": name, "input": {}}], stop_reason="max_tokens").repeat()
                runtime = RuntimeSnapshot(directory, log=lambda _line: None)
                runtime.start("pending-safe", "a" * 64, REQUEST["model"])
                backend = AgentSDKBackend(config("--target", server.url, "--api-key-env", "LOCAL_LIMIT_TEST_KEY",
                    "--sdk-effort", "low", "--sdk-max-output-tokens", "4096"))
                with patch.dict(os.environ, {"LOCAL_LIMIT_TEST_KEY": "sk-local-fixture-only"}), \
                        patch.object(fs, "stream_event_dicts", truncated):
                    with self.assertRaises(SDKOutputLimitError):
                        asyncio.run(backend.generate(backend.prepare(REQUEST, REQUEST["model"], []),
                            progress=runtime.for_request("pending-safe")))
                self.assertEqual(len(server.history()), 1)
                record = runtime.requests["pending-safe"]
                self.assertEqual(record["api_attempt_count"], 1)
                self.assertEqual(record["upstream_stop_reason"], "max_tokens")
                self.assertNotIn("PRIVATE_UNFINISHED", json.dumps(record))

    def test_output_limit_is_reported_before_native_handoff_without_another_api_call(self):
        from puppetllm.sdk_state import read_status
        with serve() as server, tempfile.TemporaryDirectory() as directory:
            server.expect().respond(content=[{"type": "tool_use", "id": "local-truncated",
                "name": "Read", "input": {"file_path": "PRIVATE_UNFINISHED_PATH"}}], stop_reason="max_tokens").repeat()
            runtime = RuntimeSnapshot(directory, log=lambda _line: None)
            runtime.start("pending-safe", "a" * 64, REQUEST["model"])
            backend = AgentSDKBackend(config("--target", server.url, "--api-key-env", "LOCAL_LIMIT_TEST_KEY",
                "--sdk-max-output-tokens", "4096"))
            with patch.dict(os.environ, {"LOCAL_LIMIT_TEST_KEY": "sk-local-fixture-only"}):
                with self.assertRaisesRegex(SDKOutputLimitError, "sdk-max-output-tokens=4096") as caught:
                    asyncio.run(backend.generate(backend.prepare(REQUEST, REQUEST["model"], []),
                        progress=runtime.for_request("pending-safe")))
            self.assertEqual(len(server.history()), 1)
            self.assertNotIn("PRIVATE", str(caught.exception))
            runtime.write([])
            status = read_status(directory)
            self.assertEqual(status["requests"][0]["observed_stop_reason"], "max_tokens")

    def test_explicit_effort_and_token_cap_reach_the_real_cli_request(self):
        output = {"content": [{"type": "text", "text": "fixture"}], "stop_reason": "end_turn"}
        with serve() as server:
            server.expect().respond(content=[{"type": "tool_use", "id": "local-envelope",
                "name": "StructuredOutput", "input": output}]).repeat()
            backend = AgentSDKBackend(config("--target", server.url, "--api-key-env", "LOCAL_LIMIT_TEST_KEY",
                "--sdk-effort", "low", "--sdk-max-output-tokens", "4096"))
            with patch.dict(os.environ, {"LOCAL_LIMIT_TEST_KEY": "sk-local-fixture-only"}):
                asyncio.run(backend.generate(backend.prepare({**REQUEST, "tools": []}, REQUEST["model"], [])))
            history = server.history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["request"]["max_tokens"], 4096)
            self.assertEqual(history[0]["request"]["params"]["output_config"]["effort"], "low")

    def run_backend(self, server, directory, *, request=None):
        logs = []
        runtime = RuntimeSnapshot(directory, log=logs.append, interval=0)
        runtime.start("local-pending", "local-key", REQUEST["model"])
        backend = AgentSDKBackend(config("--target", server.url,
            "--api-key-env", "PUPPET_RUNTIME_TEST_KEY"))
        request = request if request is not None else {**REQUEST, "tools": []}
        body = backend.prepare(request, request["model"], [])

        async def generate():
            return await asyncio.wait_for(backend.generate(body,
                progress=runtime.for_request("local-pending")), 20)

        # Conflicting inherited settings must not re-enable retry loops.
        with patch.dict(os.environ, {
            "PUPPET_RUNTIME_TEST_KEY": "sk-local-fixture-only",
            "CLAUDE_CODE_MAX_RETRIES": "5", "CLAUDE_CODE_RETRY_WATCHDOG": "1",
            "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "0",
            "MAX_STRUCTURED_OUTPUT_RETRIES": "5",
        }):
            try:
                response = asyncio.run(generate())
            except SDKRunError as error:
                response = error
        runtime.write([])
        return response, runtime.requests["local-pending"], "\n".join(logs) + (
            Path(directory) / "status.json").read_text()

    def test_http_500_has_one_api_request_and_no_invented_attempt_count(self):
        with serve() as server, tempfile.TemporaryDirectory(prefix="sdk-http-error-") as directory:
            server.expect().error(status=500, message="PRIVATE_UPSTREAM_ERROR").repeat()
            response, record, serialized = self.run_backend(server, directory)
            self.assertIsInstance(response, SDKRunError)
            self.assertEqual(response.status, 500)
            self.assertEqual(len(server.history()), 1)
            self.assertEqual(record["message_start_events"], 0)
            self.assertEqual(record["api_retry_events"], 0)
            self.assertEqual(record["api_attempt_count"], 1)
            self.assertNotIn("PRIVATE_UPSTREAM_ERROR", serialized + str(response))

    def test_sse_error_does_not_fall_back_to_a_second_nonstreaming_request(self):
        with serve() as server, tempfile.TemporaryDirectory(prefix="sdk-sse-error-") as directory:
            server.expect().error(status=500, after_blocks=0, content=[
                {"type": "text", "text": "PRIVATE_UNREACHED_TEXT"}]).respond(
                content=[{"type": "tool_use", "id": "would-be-fallback", "name": "StructuredOutput",
                    "input": {"content": [{"type": "text", "text": "PRIVATE_FALLBACK_ANSWER"}],
                              "stop_reason": "end_turn"}}]).repeat()
            response, record, serialized = self.run_backend(server, directory)
            self.assertIsInstance(response, SDKRunError)
            history = server.history()
            self.assertEqual(len(history), 1)
            self.assertTrue(history[0]["request"]["stream"])
            self.assertEqual(record["api_attempt_count"], 1)
            self.assertEqual(record["generation_events"], 0)
            self.assertNotIn("PRIVATE", serialized + str(response))

    def test_invalid_structured_output_stops_after_its_first_api_response(self):
        with serve() as server, tempfile.TemporaryDirectory(prefix="sdk-invalid-output-") as directory:
            server.expect().respond(content=[{"type": "tool_use", "id": "invalid-envelope",
                "name": "StructuredOutput", "input": {"private": "PRIVATE_INVALID_OUTPUT"}}]).repeat()
            response, record, serialized = self.run_backend(server, directory)
            self.assertIsInstance(response, SDKRunError)
            self.assertIn("structured-output", str(response))
            self.assertEqual(len(server.history()), 1)
            self.assertEqual(record["structured_output_calls_observed"], 1)
            self.assertIsNone(record["structured_output_retries"])
            self.assertNotIn("PRIVATE_INVALID_OUTPUT", serialized + str(response))

    def test_one_turn_allows_initial_valid_structured_output_and_reports_metadata(self):
        output = {"content": [{"type": "text", "text": "PRIVATE_VALID_OUTPUT"}],
                  "stop_reason": "end_turn"}
        with serve() as server, tempfile.TemporaryDirectory(prefix="sdk-valid-output-") as directory:
            server.expect().respond(content=[{"type": "tool_use", "id": "valid-envelope",
                "name": "StructuredOutput", "input": output}]).repeat()
            response, record, serialized = self.run_backend(server, directory)
            self.assertIsInstance(response, dict)
            self.assertEqual(response["content"], output["content"])
            self.assertEqual(len(server.history()), 1)
            self.assertIsNotNone(record["first_sdk_event_at"])
            self.assertIsNotNone(record["last_generation_at"])
            self.assertGreater(record["generation_events"], 0)
            self.assertEqual(record["structured_output_calls_observed"], 1)
            self.assertEqual(record["api_attempt_count"], 1)
            self.assertIsNone(record["structured_output_retries"])
            # The CLI reports 2 for this single API response; keep its value as reported.
            self.assertEqual(record["sdk_reported_num_turns"], 2)
            self.assertIsNotNone(record["sdk_reported_usage"])
            self.assertIsNotNone(record["sdk_reported_model_usage"])
            self.assertIsInstance(record["sdk_estimated_cost_usd"], (int, float))
            self.assertEqual(record["sdk_terminal_reason"], "completed")
            self.assertNotIn("PRIVATE_VALID_OUTPUT", serialized)

    def assert_native_handoff(self, server, response, record, expected):
        self.assertIsInstance(response, dict)
        self.assertEqual(response["stop_reason"], "tool_use")
        actual = [{key: value for key, value in block.items() if key != "id"}
                  for block in response["content"]]
        self.assertEqual(actual, expected)
        self.assertEqual(len(server.history()), 1)
        self.assertEqual(record["message_start_events"], 1)
        self.assertEqual(record["api_retry_events"], 0)
        self.assertEqual(record["api_attempt_count"], 1)
        # Native handoff precedes ResultMessage: its cost and turns remain unknown.
        self.assertIsNone(record["sdk_reported_num_turns"])
        self.assertIsNone(record["sdk_estimated_cost_usd"])

    def test_native_read_is_handed_off_with_exactly_one_api_request(self):
        call = {"type": "tool_use", "id": "native-read", "name": "Read",
                "input": {"file_path": "PRIVATE_NATIVE_PATH"}}
        with serve() as server, tempfile.TemporaryDirectory(prefix="sdk-native-read-") as directory:
            server.expect().respond(content=[call]).repeat()
            response, record, serialized = self.run_backend(server, directory, request=REQUEST)
            self.assert_native_handoff(server, response, record,
                [{key: value for key, value in call.items() if key != "id"}])
            self.assertNotIn("PRIVATE_NATIVE_PATH", serialized)

    def test_native_parallel_tools_preserve_leading_text_with_one_api_request(self):
        blocks = [{"type": "text", "text": "PRIVATE_LEADING_TEXT"},
                  {"type": "tool_use", "id": "parallel-a", "name": "Read",
                   "input": {"file_path": "A.txt"}},
                  {"type": "tool_use", "id": "parallel-b", "name": "Read",
                   "input": {"file_path": "B.txt"}}]
        with serve() as server, tempfile.TemporaryDirectory(prefix="sdk-native-parallel-") as directory:
            server.expect().respond(content=blocks).repeat()
            response, record, serialized = self.run_backend(server, directory, request=REQUEST)
            self.assert_native_handoff(server, response, record,
                [{key: value for key, value in block.items() if key != "id"} for block in blocks])
            self.assertNotIn("PRIVATE_LEADING_TEXT", serialized)

    def test_native_bash_is_not_executed_and_uses_one_api_request(self):
        with serve() as server, tempfile.TemporaryDirectory(prefix="sdk-native-bash-") as directory:
            sentinel = Path(directory) / "must-not-be-created"
            call = {"type": "tool_use", "id": "native-bash", "name": "Bash",
                    "input": {"command": "touch " + shlex.quote(str(sentinel))}}
            request = {**REQUEST, "tools": [{"name": "Bash", "input_schema": {
                "type": "object", "properties": {"command": {"type": "string"}},
                "required": ["command"]}}]}
            server.expect().respond(content=[call]).repeat()
            response, record, serialized = self.run_backend(server, directory, request=request)
            self.assert_native_handoff(server, response, record,
                [{key: value for key, value in call.items() if key != "id"}])
            self.assertFalse(sentinel.exists(), "the inner SDK must never execute the outer Bash command")
            self.assertNotIn(str(sentinel), serialized)

    def test_native_question_normalizes_its_array_with_one_api_request(self):
        questions = [dict(QUESTIONS[0], question="PRIVATE_QUESTION")]
        call = {"type": "tool_use", "id": "native-question", "name": "AskUserQuestion",
                "input": {"questions": json.dumps(questions)}}
        request = {**REQUEST, "tools": [QUESTION_TOOL]}
        with serve() as server, tempfile.TemporaryDirectory(prefix="sdk-native-question-") as directory:
            server.expect().respond(content=[call]).repeat()
            response, record, serialized = self.run_backend(server, directory, request=request)
            self.assert_native_handoff(server, response, record, [{"type": "tool_use",
                "name": "AskUserQuestion", "input": {"questions": [dict(questions[0], multiSelect=False)]}}])
            self.assertNotIn("PRIVATE_QUESTION", serialized)

    def test_partial_native_tool_before_sse_error_never_hands_off_or_repeats_api(self):
        with serve() as server, tempfile.TemporaryDirectory(prefix="sdk-partial-native-") as directory:
            server.expect().error(status=500, after_blocks=1, content=[
                {"type": "tool_use", "id": "partial-native", "name": "Read",
                 "input": {"file_path": "PRIVATE_PARTIAL_NATIVE_PATH"}},
                {"type": "text", "text": "PRIVATE_UNFINISHED_TAIL"}]).repeat()
            response, record, serialized = self.run_backend(server, directory, request=REQUEST)
            self.assertIsInstance(response, SDKRunError)
            self.assertEqual(len(server.history()), 1)
            self.assertEqual(record["message_start_events"], 1)
            self.assertEqual(record["api_attempt_count"], 1)
            self.assertNotIn("PRIVATE", serialized + str(response))


if __name__ == "__main__":
    unittest.main()
