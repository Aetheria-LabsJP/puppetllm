"""Content-free SDK progress snapshots; no model or network calls."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from puppetllm.sdk_runtime import RuntimeSnapshot, StateBusyError, StateLock


HAS_SDK = importlib.util.find_spec("claude_agent_sdk") is not None


class RuntimeCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sdk-runtime-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.now = 1000.0
        clock = patch("puppetllm.sdk_runtime.time.time", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.logs = []
        self.runtime = RuntimeSnapshot(self.directory, log=self.logs.append,
                                       idle_warning=10, interval=0)
        self.runtime.start("pending-local", "a" * 64, "claude-sonnet-4-6")
        self.progress = self.runtime.for_request("pending-local")

    @property
    def record(self):
        return self.runtime.requests["pending-local"]

    def snapshot(self, guard=()):
        self.runtime.write(list(guard))
        return json.loads((self.directory / "status.json").read_text())

    def last_log(self):
        return json.loads(self.logs[-1].removeprefix("sdk_progress "))


class TestRuntimeState(RuntimeCase):
    def test_egress_counts_and_metadata_are_distinct_from_sdk_event_estimates(self):
        self.progress.api_request_started()
        self.progress.api_request_blocked()
        self.progress.api_response_metadata("max_tokens", {"output_tokens": 4096, "private": "PRIVATE_BODY"})
        self.assertEqual(self.record["api_attempt_count"], 1)
        self.assertEqual(self.record["api_requests_blocked"], 1)
        self.assertEqual(self.record["upstream_stop_reason"], "max_tokens")
        self.assertEqual(self.record["upstream_reported_usage"], {"output_tokens": 4096})
        self.assertEqual(self.record["event_count"], 0)
        self.assertIsNone(self.record["last_generation_at"])
        self.assertNotIn("PRIVATE", json.dumps(self.snapshot()) + "\n".join(self.logs))

    def test_unobserved_sdk_counters_and_cost_remain_null(self):
        unknown = ("api_attempt_count", "structured_output_retries", "sdk_reported_num_turns",
                   "sdk_estimated_cost_usd", "sdk_reported_usage", "sdk_reported_model_usage",
                   "first_sdk_event_at", "last_sdk_event_at", "last_generation_at")
        self.now += 30
        status = self.snapshot()
        record = status["requests"][0]
        for field in unknown:
            with self.subTest(field=field):
                self.assertIsNone(record[field])
        self.assertEqual(record["event_count"], 0)
        self.assertEqual(record["generation_events"], 0)
        self.assertEqual(self.last_log()["warning"], "no_generation_event_observed")

    def test_heartbeat_and_stderr_never_claim_sdk_or_generation_progress(self):
        self.now += 5
        self.progress.stderr("PRIVATE_STDERR command=secret token=secret")
        first = self.snapshot()
        self.now += 20
        second = self.snapshot()
        self.assertGreater(second["heartbeat_at"], first["heartbeat_at"])
        self.assertEqual(self.record["stderr_lines"], 1)
        self.assertEqual(self.record["event_count"], 0)
        self.assertIsNone(self.record["first_sdk_event_at"])
        self.assertIsNone(self.record["last_generation_at"])
        self.assertEqual(self.last_log()["warning"], "no_generation_event_observed")
        self.assertNotIn("PRIVATE_STDERR", json.dumps(second) + "\n".join(self.logs))

    def test_status_retains_only_guard_counts_and_masks_invalid_model_names(self):
        self.runtime.start("other-pending", "other-fingerprint", "PRIVATE MODEL CONTENT")
        status = self.snapshot([
            {"state": "blocked", "request": "PRIVATE_GUARD_REQUEST"},
            {"state": "active", "error": "PRIVATE_GUARD_ERROR"},
        ])
        self.assertEqual(status["guard"], {"blocked": 1, "in_progress": 1})
        self.assertEqual(status["requests"][1]["model"], "custom")
        self.assertNotIn("PRIVATE", json.dumps(status) + "\n".join(self.logs))
        self.assertEqual((self.directory / "status.json").stat().st_mode & 0o777, 0o600)

    def test_finish_preserves_first_completion_and_stops_idle_warning(self):
        self.now += 20
        self.runtime.finish("pending-local", "timeout")
        completed = dict(self.record)
        self.now += 20
        self.runtime.finish("pending-local", "different_reason")
        self.snapshot()
        self.assertEqual(self.record, completed)
        self.assertEqual(self.record["stage"], "timeout")
        self.assertNotIn("warning", self.last_log())


@unittest.skipUnless(HAS_SDK, "install .[agent-sdk]")
class TestPublicSDKEvents(RuntimeCase):
    def setUp(self):
        super().setUp()
        import claude_agent_sdk as sdk
        self.sdk = sdk

    def event(self, kind, **fields):
        return self.sdk.StreamEvent(uuid="local-event", session_id="local-session",
                                    event={"type": kind, **fields})

    def test_sdk_liveness_is_distinct_from_generation(self):
        self.now += 1
        self.progress.message(self.sdk.SystemMessage(subtype="init", data={"private": "PRIVATE_INIT"}))
        self.assertEqual(self.record["first_sdk_event_at"], self.now)
        self.now += 1
        self.progress.message(self.event("message_start", message={"id": "local-api-id"}))
        self.now += 20
        for message in (self.event("ping"), self.event("content_block_delta", index=0,
                delta={"type": "signature_delta", "signature": "PRIVATE_SIGNATURE"})):
            self.progress.message(message)
        self.assertEqual(self.record["last_sdk_event_at"], self.now)
        self.assertEqual(self.record["message_start_events"], 1)
        self.assertEqual(self.record["generation_events"], 0)
        self.assertIsNone(self.record["last_generation_at"])
        self.assertIsNone(self.record["api_attempt_count"])
        self.assertEqual(self.last_log()["warning"], "no_generation_event_observed")

    def test_generation_tracks_only_counts_type_and_time(self):
        chunks = (("text_delta", "text", "PRIVATE_TEXT"),
                  ("thinking_delta", "thinking", "PRIVATE_THINKING"),
                  ("input_json_delta", "partial_json", '{"secret":"PRIVATE_TOOL_ARGUMENT"}'))
        for kind, field, value in chunks:
            self.now += 1
            self.progress.message(self.event("content_block_delta", index=0,
                                              delta={"type": kind, field: value}))
        self.assertEqual(self.record["generation_events"], len(chunks))
        self.assertEqual(self.record["generated_characters"], sum(len(value) for _, _, value in chunks))
        self.assertEqual(self.record["last_generation_at"], self.now)
        self.assertEqual(self.record["last_generation_kind"], "input_json_delta")
        self.assertEqual(self.record["stage"], "generating")
        self.assertNotIn("warning", self.last_log())
        status = self.snapshot()
        self.assertNotIn("PRIVATE", json.dumps(status) + "\n".join(self.logs))

    def test_empty_deltas_do_not_refresh_generation_or_clear_idle_warning(self):
        self.now += 1
        self.progress.message(self.event("content_block_delta", index=0,
            delta={"type": "text_delta", "text": "one generated chunk"}))
        generated_at = self.record["last_generation_at"]
        event_count = self.record["event_count"]
        self.now += 20
        for kind, field in (("text_delta", "text"), ("thinking_delta", "thinking"),
                            ("input_json_delta", "partial_json")):
            self.progress.message(self.event("content_block_delta", index=0,
                                              delta={"type": kind, field: ""}))
        self.assertEqual(self.record["event_count"], event_count + 3)
        self.assertEqual(self.record["last_sdk_event_at"], self.now)
        self.assertEqual(self.record["last_generation_at"], generated_at)
        self.assertEqual(self.record["generation_events"], 1)
        self.assertEqual(self.record["generated_characters"], len("one generated chunk"))
        self.assertEqual(self.last_log()["warning"], "no_generation_event_observed")

    def test_hidden_thinking_estimates_advance_without_text_or_double_counting(self):
        total = 0
        for increment in (7, 11, 13):
            self.now += 20
            total += increment
            messages = [self.event("content_block_delta", index=0, delta={
                "type": "thinking_delta", "thinking": "", "estimated_tokens": increment}),
                self.sdk.SystemMessage(subtype="thinking_tokens", data={
                    "estimated_tokens": total, "estimated_tokens_delta": increment,
                    "private": "PRIVATE_THINKING"})]
            # Both event orderings occur in SDK consumers; neither may double count.
            for message in reversed(messages) if increment == 11 else messages:
                self.progress.message(message)
        self.assertEqual(self.record["estimated_thinking_tokens"], 31)
        self.assertEqual(self.record["thinking_token_updates"], 3)
        self.assertEqual(self.record["last_thinking_progress_at"], self.now)
        self.assertEqual(self.record["generated_characters"], 0)
        self.assertEqual(self.record["generation_events"], 0)
        self.assertIsNone(self.record["last_generation_at"])
        self.assertEqual(self.record["stage"], "thinking")
        self.assertNotIn("warning", self.last_log())
        self.assertNotIn("PRIVATE", json.dumps(self.snapshot()) + "\n".join(self.logs))
        self.now += 20
        self.snapshot()
        self.assertEqual(self.last_log()["warning"], "no_generation_event_observed")

    def test_system_only_thinking_increments_survive_per_block_counter_reset(self):
        for total, delta in ((7, 7), (18, 11), (5, 5)):
            self.now += 1
            self.progress.message(self.sdk.SystemMessage(subtype="thinking_tokens", data={
                "estimated_tokens": total, "estimated_tokens_delta": delta}))
        self.assertEqual(self.record["estimated_thinking_tokens"], 23)
        last = self.record["last_thinking_progress_at"]
        self.now += 20
        for value in (0, -1, True, math.inf, math.nan, "PRIVATE_COUNTER"):
            self.progress.message(self.event("content_block_delta", delta={
                "type": "thinking_delta", "thinking": "", "estimated_tokens": value}))
            self.progress.message(self.sdk.SystemMessage(subtype="thinking_tokens", data={
                "estimated_tokens": value, "estimated_tokens_delta": value}))
        self.assertEqual(self.record["estimated_thinking_tokens"], 23)
        self.assertEqual(self.record["last_thinking_progress_at"], last)
        self.assertIn("warning", self.last_log())

    def test_stream_counters_use_fixed_keys_and_do_not_count_signatures_as_progress(self):
        for message in (self.event("ping"), self.event("PRIVATE_EVENT"),
                        self.event("content_block_delta", delta={"type": "PRIVATE_DELTA"}),
                        self.event("content_block_delta", delta={
                            "type": "signature_delta", "signature": "PRIVATE_SIGNATURE", "estimated_tokens": 50})):
            self.progress.message(message)
        self.assertEqual(self.record["stream_event_counts"], {"ping": 1, "other": 1, "content_block_delta": 2})
        self.assertEqual(self.record["stream_delta_counts"], {"other": 1, "signature_delta": 1})
        self.assertIsNone(self.record["estimated_thinking_tokens"])
        self.assertNotIn("PRIVATE", json.dumps(self.snapshot()) + "\n".join(self.logs))

    def test_retry_metadata_is_numeric_and_does_not_infer_total_attempts(self):
        self.progress.message(self.sdk.SystemMessage(subtype="api_retry", data={
            "attempt": 1, "max_retries": 2, "retry_delay_ms": 10, "error_status": 500,
            "error": {"message": "PRIVATE_API_ERROR"}, "request_id": "PRIVATE_REQUEST_ID"}))
        self.assertEqual(self.record["api_retry_events"], 1)
        self.assertEqual(self.record["last_api_retry"], {
            "attempt": 1, "max_retries": 2, "retry_delay_ms": 10, "error_status": 500})
        self.assertEqual(self.record["stage"], "sdk_retry")
        self.assertIsNone(self.record["api_attempt_count"])
        self.assertIsNone(self.record["structured_output_retries"])
        self.progress.message(self.sdk.SystemMessage(subtype="api_retry", data={
            "attempt": True, "max_retries": -1, "retry_delay_ms": math.inf,
            "error_status": "PRIVATE_HTTP_STATUS"}))
        self.assertEqual(self.record["last_api_retry"], {})
        self.assertNotIn("PRIVATE", json.dumps(self.snapshot()) + "\n".join(self.logs))

    def test_message_usage_is_replaced_at_start_and_updated_by_final_delta(self):
        self.progress.message(self.event("message_start", message={"usage": {
            "input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 2,
            "private": "PRIVATE_USAGE"}}))
        self.progress.message(self.event("message_delta", usage={"output_tokens": 70}))
        self.assertEqual(self.record["observed_message_usage"], {
            "input_tokens": 10, "output_tokens": 70, "cache_read_input_tokens": 2})
        self.progress.message(self.event("message_start", message={"usage": {"input_tokens": 20}}))
        self.assertEqual(self.record["observed_message_usage"], {"input_tokens": 20})
        self.assertIsNone(self.record["sdk_reported_usage"])
        self.assertIsNone(self.record["sdk_estimated_cost_usd"])

    def test_result_keeps_reported_usage_and_cost_separate_without_private_fields(self):
        message = self.sdk.ResultMessage(
            subtype="error_max_budget_usd", duration_ms=100, duration_api_ms=90,
            is_error=True, num_turns=2, session_id="local-session", total_cost_usd=18,
            usage={"input_tokens": 0, "output_tokens": 0, "private": "PRIVATE_USAGE"},
            model_usage={"PRIVATE_MODEL_KEY": {"inputTokens": 1_000_000,
                "outputTokens": 1_000_000, "costUSD": 18, "private": "PRIVATE_MODEL_USAGE"}},
            errors=["PRIVATE_ERROR"], result="PRIVATE_RESULT",
            structured_output={"private": "PRIVATE_STRUCTURED_OUTPUT"})
        self.progress.message(message)
        status = self.snapshot()
        self.assertEqual(self.record["sdk_reported_num_turns"], 2)
        self.assertEqual(self.record["sdk_estimated_cost_usd"], 18)
        self.assertEqual(self.record["sdk_reported_usage"], {"input_tokens": 0, "output_tokens": 0})
        self.assertEqual(self.record["sdk_reported_model_usage"], [{
            "inputTokens": 1_000_000, "outputTokens": 1_000_000, "costUSD": 18}])
        self.assertNotIn("PRIVATE", json.dumps(status) + "\n".join(self.logs))

    def test_real_cli_terminal_reasons_survive_runtime_and_status_filter(self):
        from puppetllm.sdk_state import read_status
        for reason in ("completed", "budget_exhausted", "structured_output_retry_exhausted"):
            self.progress.message(self.sdk.ResultMessage(subtype="success", duration_ms=1,
                duration_api_ms=1, is_error=False, num_turns=1, session_id="local-session",
                terminal_reason=reason, usage={}))
            self.snapshot()
            status = read_status(self.directory, now=self.now)
            self.assertEqual(status["requests"][0]["sdk_terminal_reason"], reason)

    def test_absent_or_invalid_reported_usage_and_cost_stay_unknown(self):
        message = self.sdk.ResultMessage(
            subtype="error_during_execution", duration_ms=1, duration_api_ms=0,
            is_error=True, num_turns=-1, session_id="local-session", total_cost_usd=math.nan,
            usage={"input_tokens": True, "output_tokens": -1,
                   "cache_read_input_tokens": math.inf, "private": "PRIVATE_USAGE"})
        self.progress.message(message)
        self.assertIsNone(self.record["sdk_reported_num_turns"])
        self.assertIsNone(self.record["sdk_estimated_cost_usd"])
        self.assertIsNone(self.record["sdk_reported_usage"])
        self.assertIsNone(self.record["sdk_reported_model_usage"])
        self.snapshot()  # Invalid numeric data must not break allow_nan=False persistence.

    def test_structured_calls_are_observed_without_assuming_validation_retries(self):
        self.progress.message(self.sdk.AssistantMessage(model="local-model", content=[
            self.sdk.TextBlock("PRIVATE_ASSISTANT_TEXT"),
            self.sdk.ToolUseBlock(id="first", name="StructuredOutput", input={"secret": "PRIVATE_INPUT"}),
            self.sdk.ToolUseBlock(id="external", name="PRIVATE_EXTERNAL_TOOL", input={"secret": "PRIVATE_INPUT"})]))
        self.assertEqual(self.record["structured_output_calls_observed"], 1)
        self.assertIsNone(self.record["structured_output_retries"])
        self.assertIsNone(self.record["sdk_reported_num_turns"])
        self.assertNotIn("PRIVATE", json.dumps(self.snapshot()) + "\n".join(self.logs))


class TestStateLock(unittest.TestCase):
    def test_a_state_directory_has_one_holder_at_a_time(self):
        with tempfile.TemporaryDirectory(prefix="sdk-state-lock-") as directory:
            with StateLock(directory):
                with self.assertRaises(StateBusyError):
                    StateLock(directory)
            with StateLock(directory):
                pass


if __name__ == "__main__":
    unittest.main()
