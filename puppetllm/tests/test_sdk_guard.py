"""The persistent execution guard needs no optional SDK or paid API calls."""

import copy
import json
import multiprocessing
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from puppetllm.sdk_guard import Blocked, GuardError, Lease, SDKGuard


BODY = {"model": "claude-opus-5-5", "transcript": {
    "system": "PRIVATE_SYSTEM", "tools": [], "tool_choice": {"type": "auto"},
    "messages": [{"role": "user", "content": "PRIVATE_BODY"}]},
    "images": [{"source": {"data": "PRIVATE_IMAGE"}}], "prompt": "derived"}


def _compete(path, key, start, results):
    try:
        with SDKGuard(path) as guard:
            start.wait(10)
            admitted = guard.begin(key)
            results.put((isinstance(admitted, Lease), getattr(admitted, "reason", None)))
    except Exception as exc:
        results.put(("error", type(exc).__name__))


class TestSDKGuard(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "guard.sqlite3"
        self.guard = SDKGuard(self.path)
        self.addCleanup(self.guard.close)
        self.key = self.guard.fingerprint(BODY, generation_settings={"max_turns": 3})

    def test_fingerprint_is_canonical_and_persistent_but_keyed(self):
        reordered = dict(reversed(list(BODY.items())))
        reordered["transcript"] = dict(reversed(list(BODY["transcript"].items())))
        self.assertEqual(self.key, self.guard.fingerprint(reordered, generation_settings={"max_turns": 3}))
        with SDKGuard(self.path) as second:
            self.assertEqual(self.key, second.fingerprint(BODY, generation_settings={"max_turns": 3}))
        with SDKGuard(Path(self.directory.name) / "other.sqlite3") as other:
            self.assertNotEqual(self.key, other.fingerprint(BODY, generation_settings={"max_turns": 3}))
        self.assertRegex(self.key, r"^[0-9a-f]{64}$")

    def test_ids_and_derived_prompt_do_not_reset_failure(self):
        settings = {"max_turns": 3}
        original = self.guard.fingerprint(BODY, generation_settings=settings)
        changed = {**BODY, "request_id": "two", "pending_id": "new", "prompt": "changed"}
        self.assertEqual(original, self.guard.fingerprint(changed, generation_settings=settings))
        lease = self.guard.begin(original)
        self.guard.fail(lease, "timeout")
        self.assertEqual(self.guard.begin(self.guard.fingerprint(changed, generation_settings=settings)),
                         Blocked(original, "timeout", "blocked"))

    def test_all_effective_input_and_generation_changes_remain_distinct(self):
        changes = []
        for field, value in (("model", "sonnet"), ("images", [{"source": {"data": "different"}}])):
            changes.append(({**BODY, field: value}, {"max_turns": 3}))
        for field, value in (("system", "different"), ("tools", [{"name": "Read"}]),
                             ("tool_choice", {"type": "none"}), ("messages", [{"role": "user", "content": "different"}])):
            changes.append(({**BODY, "transcript": {**BODY["transcript"], field: value}}, {"max_turns": 3}))
        changes.extend([(BODY, {"max_turns": 4}), (BODY, {"max_turns": 3, "input_format": "json"}),
                        (BODY, {"max_turns": 3, "system_prompt": "adapter instructions"})])
        for body, settings in changes:
            with self.subTest(body=body, settings=settings):
                self.assertNotEqual(self.key, self.guard.fingerprint(body, generation_settings=settings))
        nested = copy.deepcopy(BODY)
        nested["transcript"]["messages"][0]["request_id"] = "meaningful input"
        self.assertNotEqual(self.key, self.guard.fingerprint(nested, generation_settings={"max_turns": 3}))

    def test_concurrent_instances_only_admit_one_and_success_releases(self):
        with SDKGuard(self.path) as second:
            lease = self.guard.begin(self.key)
            self.assertIsInstance(lease, Lease)
            self.assertEqual(second.begin(self.key), Blocked(self.key, "in_progress", "active"))
            self.guard.succeed(lease)
            self.assertEqual(second.status(), [])
            next_lease = second.begin(self.key)
            self.assertIsInstance(next_lease, Lease)
            second.succeed(next_lease)

    def test_different_requests_run_independently(self):
        second_key = self.guard.fingerprint({**BODY, "model": "sonnet"})
        first, second = self.guard.begin(self.key), self.guard.begin(second_key)
        self.assertIsInstance(first, Lease)
        self.assertIsInstance(second, Lease)
        self.guard.fail(first, "timeout")
        self.guard.succeed(second)
        self.assertEqual([row["key"] for row in self.guard.status()], [self.key])

    def test_failures_survive_restarts_without_expiration(self):
        for reason in ("timeout", "error", "cancelled", "abnormal_exit"):
            with self.subTest(reason=reason):
                key = self.guard.fingerprint({**BODY, "model": reason})
                lease = self.guard.begin(key)
                self.guard.fail(lease, reason)
                with SDKGuard(self.path) as restarted:
                    self.assertEqual(restarted.begin(key), Blocked(key, reason, "blocked"))

    def test_clear_never_deletes_active_and_selective_clear_keeps_others(self):
        active = self.guard.begin(self.key)
        second_key = self.guard.fingerprint({**BODY, "model": "second"})
        third_key = self.guard.fingerprint({**BODY, "model": "third"})
        self.guard.fail(self.guard.begin(second_key), "timeout")
        self.guard.fail(self.guard.begin(third_key), "error")
        self.assertEqual(self.guard.clear(self.key), 0)
        self.assertEqual(self.guard.clear(second_key), 1)
        self.assertEqual(self.guard.clear(), 1)
        self.assertEqual(self.guard.begin(self.key).state, "active")
        self.assertEqual(self.guard.status()[0]["key"], active.key)

    def test_stale_or_failed_leases_cannot_release_a_new_execution(self):
        lease = self.guard.begin(self.key)
        self.guard.fail(lease, "timeout")
        self.assertFalse(self.guard.fail(lease, "error"))
        with self.assertRaises(GuardError):
            self.guard.succeed(lease)
        self.assertEqual(self.guard.status()[0]["reason"], "timeout")
        self.guard.clear(self.key)
        replacement = self.guard.begin(self.key)
        with self.assertRaises(GuardError):
            self.guard.succeed(lease)
        self.assertEqual(self.guard.begin(self.key).state, "active")
        self.guard.succeed(replacement)

    def test_close_marks_only_its_own_unfinished_leases_failed(self):
        second = SDKGuard(self.path)
        self.addCleanup(second.close)
        self.guard.begin(self.key)
        second_key = second.fingerprint({**BODY, "model": "second"})
        second.begin(second_key)
        self.guard.close()
        self.assertEqual(second.begin(self.key), Blocked(self.key, "abnormal_exit", "blocked"))
        self.assertEqual(second.begin(second_key), Blocked(second_key, "in_progress", "active"))

    def test_process_crash_keeps_active_until_explicit_recover_then_clear(self):
        code = "from puppetllm.sdk_guard import SDKGuard; import os,sys; g=SDKGuard(sys.argv[1]); g.begin(sys.argv[2]); os._exit(0)"
        subprocess.run([sys.executable, "-B", "-c", code, str(self.path), self.key], check=True)
        with SDKGuard(self.path) as restarted:
            self.assertEqual(restarted.begin(self.key), Blocked(self.key, "in_progress", "active"))
            self.assertEqual(restarted.clear(), 0)
            self.assertEqual(restarted.recover_active(), 1)
            self.assertEqual(restarted.begin(self.key), Blocked(self.key, "abnormal_exit", "blocked"))
            self.assertEqual(restarted.clear(self.key), 1)
            restarted.succeed(restarted.begin(self.key))

    def test_capacity_fails_closed_without_evicting_failures(self):
        path = Path(self.directory.name) / "bounded.sqlite3"
        with SDKGuard(path, max_entries=1) as bounded:
            first_key, second_key = bounded.fingerprint(BODY), bounded.fingerprint({**BODY, "model": "second"})
            bounded.fail(bounded.begin(first_key), "timeout")
            self.assertEqual(bounded.begin(second_key), Blocked(second_key, "capacity", "capacity"))
            self.assertEqual(bounded.begin(first_key), Blocked(first_key, "timeout", "blocked"))
            bounded.clear(first_key)
            bounded.succeed(bounded.begin(second_key))

    def test_database_and_status_never_contain_payload_auth_or_raw_errors(self):
        lease = self.guard.begin(self.key)
        with self.assertRaises(GuardError) as caught:
            self.guard.fail(lease, "PRIVATE_EXCEPTION_WITH_AUTH")
        self.assertNotIn("PRIVATE", str(caught.exception))
        self.guard.fail(lease, "error")
        status = self.guard.status()
        self.assertEqual(set(status[0]), {"key", "state", "created_at", "updated_at", "reason"})
        for marker in (b"PRIVATE_BODY", b"PRIVATE_SYSTEM", b"PRIVATE_IMAGE", b"PRIVATE_AUTH", b"PRIVATE_EXCEPTION"):
            self.assertNotIn(marker, self.path.read_bytes())
            self.assertNotIn(marker.decode(), json.dumps(status))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_corrupt_unwritable_and_closed_databases_fail_closed(self):
        corrupt = Path(self.directory.name) / "corrupt.sqlite3"
        corrupt.write_bytes(b"PRIVATE_CORRUPT_DATA")
        for path in (corrupt, Path("/proc/self/puppetllm-guard.sqlite3")):
            with self.subTest(path=path), self.assertRaises(GuardError) as caught:
                SDKGuard(path)
            self.assertNotIn("PRIVATE", str(caught.exception))
        self.guard._connection.execute("PRAGMA query_only=ON")
        with self.assertRaises(GuardError):
            self.guard.begin(self.key)
        self.guard._connection.execute("PRAGMA query_only=OFF")
        self.guard.close()
        with self.assertRaises(GuardError):
            self.guard.begin(self.key)

    def test_missing_hmac_metadata_cannot_reset_existing_rows(self):
        self.guard.fail(self.guard.begin(self.key), "error")
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM sdk_guard_meta")
        with self.assertRaises(GuardError):
            SDKGuard(self.path)

    def test_failed_success_write_leaves_execution_blocked(self):
        lease = self.guard.begin(self.key)
        self.guard._connection.execute("PRAGMA query_only=ON")
        with self.assertRaises(GuardError):
            self.guard.succeed(lease)
        with SDKGuard(self.path) as second:
            self.assertEqual(second.begin(self.key), Blocked(self.key, "in_progress", "active"))
        self.guard._connection.execute("PRAGMA query_only=OFF")
        self.guard.fail(lease, "error")

    def test_multiple_processes_race_for_exactly_one_attempt(self):
        context = multiprocessing.get_context("spawn")
        # Include simultaneous first creation, not only an initialized database.
        for path in (self.path, Path(self.directory.name) / "fresh.sqlite3"):
            with self.subTest(path=path):
                start, results = context.Event(), context.Queue()
                processes = [context.Process(target=_compete, args=(str(path), self.key, start, results)) for _ in range(4)]
                try:
                    for process in processes:
                        process.start()
                    start.set()
                    outcomes = [results.get(timeout=15) for _ in processes]
                    self.assertEqual(sum(admitted is True for admitted, _ in outcomes), 1, outcomes)
                    self.assertFalse(any(admitted == "error" for admitted, _ in outcomes), outcomes)
                finally:
                    for process in processes:
                        process.join(15)
                        if process.is_alive():
                            process.terminate()
                            process.join(5)
                        self.assertEqual(process.exitcode, 0)
                    results.close()
                    results.join_thread()


if __name__ == "__main__":
    unittest.main()
