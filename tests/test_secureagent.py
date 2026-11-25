import os
import tempfile
import unittest
from pathlib import Path

from secureagent import (COMPUTE, FS_READ, FS_WRITE, Agent, AuditLog, Final, Policy,
                         ScriptedLLM, ToolCall, default_registry)
from secureagent.tools import Blocked, safe_path


def make_policy(sandbox, granted=(FS_READ, FS_WRITE, COMPUTE), **kwargs):
    return Policy(sandbox=sandbox, granted=frozenset(granted), **kwargs)


def run(sandbox, steps, **policy_kwargs):
    granted = policy_kwargs.pop("granted", (FS_READ, FS_WRITE, COMPUTE))
    audit = AuditLog()
    agent = Agent(ScriptedLLM(steps), default_registry(),
                  make_policy(sandbox, granted, **policy_kwargs), audit)
    return agent.run("task"), audit


class SandboxCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        (self.root / "notes.txt").write_text("hello", encoding="utf-8")


class SafePathTests(SandboxCase):
    def test_inside_paths_are_allowed(self):
        self.assertEqual(safe_path(self.root, "notes.txt"), self.root / "notes.txt")

    def test_parent_traversal_is_blocked(self):
        for bad in ("../x", "a/../../x", "../../etc/passwd"):
            with self.subTest(path=bad), self.assertRaises(Blocked):
                safe_path(self.root, bad)

    def test_absolute_path_is_blocked(self):
        with self.assertRaises(Blocked):
            safe_path(self.root, "/etc/passwd")

    def test_sibling_directory_with_same_prefix_is_blocked(self):
        sibling = Path(str(self.root) + "-evil")
        sibling.mkdir()
        self.addCleanup(sibling.rmdir)
        with self.assertRaises(Blocked):
            safe_path(self.root, "../" + sibling.name)

    @unittest.skipUnless(hasattr(os, "symlink"), "needs symlink support")
    def test_symlink_escape_is_blocked(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        link = self.root / "link"
        try:
            os.symlink(outside.name, link)
        except OSError:
            self.skipTest("cannot create symlinks here")
        with self.assertRaises(Blocked):
            safe_path(self.root, "link/secret.txt")

    def test_bad_input_is_blocked(self):
        for bad in ("", "a\x00b", None, 5):
            with self.subTest(path=bad), self.assertRaises(Blocked):
                safe_path(self.root, bad)


class PermissionTests(SandboxCase):
    def test_tool_without_permission_is_denied(self):
        result, audit = run(self.root,
                            [ToolCall("write_file", {"path": "a.txt", "content": "x"}), Final("ok")],
                            granted=(FS_READ,))
        self.assertFalse(result.observations[0].allowed)
        self.assertIn("missing permission", audit.events[0]["reason"])
        self.assertFalse((self.root / "a.txt").exists())

    def test_unknown_tool_is_denied(self):
        result, audit = run(self.root, [ToolCall("delete_everything", {}), Final("ok")])
        self.assertFalse(result.observations[0].allowed)
        self.assertEqual(audit.events[0]["reason"], "unknown tool")

    def test_granted_tool_runs(self):
        result, _ = run(self.root, [ToolCall("read_file", {"path": "notes.txt"}), Final("ok")])
        self.assertTrue(result.observations[0].allowed)
        self.assertEqual(result.observations[0].text, "hello")


class ArgumentTests(SandboxCase):
    def test_extra_missing_and_wrong_type_arguments_are_denied(self):
        cases = [
            {"path": "notes.txt", "mode": "w"},
            {},
            {"path": 5},
        ]
        for args in cases:
            with self.subTest(args=args):
                result, _ = run(self.root, [ToolCall("read_file", args), Final("ok")])
                self.assertFalse(result.observations[0].allowed)

    def test_traversal_through_the_agent_is_denied(self):
        result, audit = run(self.root, [ToolCall("read_file", {"path": "../../etc/passwd"}), Final("ok")])
        self.assertFalse(result.observations[0].allowed)
        self.assertIn("blocked by tool", audit.events[0]["reason"])


class InjectionTests(SandboxCase):
    steps = [ToolCall("read_file", {"path": "notes.txt"}),
             ToolCall("write_file", {"path": "pwned.txt", "content": "stolen"}),
             Final("done")]

    def test_write_after_untrusted_read_is_denied_by_default(self):
        result, audit = run(self.root, list(self.steps))
        self.assertTrue(result.observations[0].allowed)
        self.assertFalse(result.observations[1].allowed)
        self.assertIn("approval required", audit.events[1]["reason"])
        self.assertFalse((self.root / "pwned.txt").exists())

    def test_write_without_untrusted_read_is_allowed(self):
        result, _ = run(self.root, [ToolCall("write_file", {"path": "ok.txt", "content": "fine"}), Final("done")])
        self.assertTrue(result.observations[0].allowed)
        self.assertTrue((self.root / "ok.txt").exists())

    def test_human_approval_can_allow_the_write(self):
        seen = []

        def approver(tool, args, reason):
            seen.append((tool, reason))
            return True

        result, audit = run(self.root, list(self.steps), approver=approver)
        self.assertTrue(result.observations[1].allowed)
        self.assertEqual(seen[0][0], "write_file")
        self.assertEqual(audit.events[1]["reason"], "approved after untrusted read")
        self.assertTrue((self.root / "pwned.txt").exists())

    def test_approval_can_be_declined(self):
        result, _ = run(self.root, list(self.steps), approver=lambda *a: False)
        self.assertFalse(result.observations[1].allowed)
        self.assertFalse((self.root / "pwned.txt").exists())

    def test_taint_check_can_be_turned_off(self):
        result, _ = run(self.root, list(self.steps), approve_after_untrusted=False)
        self.assertTrue(result.observations[1].allowed)


class BudgetTests(SandboxCase):
    def test_run_stops_at_the_call_budget(self):
        steps = [ToolCall("calculator", {"expression": "1+1"})] * 50
        result, audit = run(self.root, steps, max_calls=3)
        self.assertEqual(result.stopped, "budget")
        self.assertEqual(len(audit.events), 3)

    def test_denied_calls_still_use_budget(self):
        steps = [ToolCall("nope", {})] * 50
        result, _ = run(self.root, steps, max_calls=4)
        self.assertEqual(result.stopped, "budget")
        self.assertEqual(len(result.observations), 4)


class ToolTests(SandboxCase):
    def calc(self, expression):
        result, _ = run(self.root, [ToolCall("calculator", {"expression": expression}), Final("ok")])
        return result.observations[0]

    def test_arithmetic_works(self):
        self.assertEqual(self.calc("2 + 3 * 4").text, "14")
        self.assertEqual(self.calc("-(2 + 3)").text, "-5")

    def test_code_is_not_executed(self):
        for payload in ("__import__('os').system('echo hi')", "open('notes.txt')",
                        "[x for x in range(5)]", "(1).__class__", "2 ** 100000"):
            with self.subTest(payload=payload):
                obs = self.calc(payload)
                self.assertTrue(obs.text.startswith("ERROR"), obs.text)

    def test_division_by_zero_is_a_clean_error(self):
        self.assertEqual(self.calc("1 / 0").text, "ERROR: division by zero")

    def test_output_is_truncated(self):
        (self.root / "big.txt").write_text("a" * 5000, encoding="utf-8")
        result, _ = run(self.root, [ToolCall("read_file", {"path": "big.txt"}), Final("ok")],
                        max_output_chars=100)
        self.assertTrue(result.observations[0].text.endswith("[truncated]"))
        self.assertLess(len(result.observations[0].text), 200)


class AuditTests(SandboxCase):
    def test_every_decision_is_recorded_and_content_is_not(self):
        secret = "SUPER-SECRET-" * 20
        result, audit = run(self.root, [
            ToolCall("write_file", {"path": "a.txt", "content": secret}),
            ToolCall("nope", {}),
            Final("ok")])
        self.assertEqual(len(audit.events), 2)
        self.assertEqual([e["allowed"] for e in audit.events], [True, False])
        self.assertNotIn("SUPER-SECRET", repr(audit.events))
        self.assertIn("chars", audit.events[0]["args"]["content"])

    def test_log_can_be_written_to_a_file(self):
        log_path = self.root / "audit.jsonl"
        agent = Agent(ScriptedLLM([ToolCall("calculator", {"expression": "1+1"}), Final("ok")]),
                      default_registry(), make_policy(self.root), AuditLog(log_path))
        agent.run("task")
        self.assertEqual(len(log_path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
