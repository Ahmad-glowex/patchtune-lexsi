import tempfile
import unittest
from pathlib import Path

from toyrepo.generate import generate_instance
from patchtune.tools import ToolExecutor, validate_call, parse_unified_diff, PatchError


class TestTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = generate_instance("is_even_inverted", "parity")
        self.repo.materialize(Path(self._tmp.name))
        self.ex = ToolExecutor(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_and_search(self):
        r = self.ex.dispatch({"tool": "read_file",
                              "arguments": {"path": self.repo.target_file}})
        self.assertTrue(r["ok"])
        self.assertIn("is_even", r["content"])
        s = self.ex.dispatch({"tool": "search_code", "arguments": {"query": "is_even"}})
        self.assertTrue(s["ok"])
        self.assertTrue(any("def is_even" in m for m in s["matches"]))

    def test_path_escape_rejected(self):
        r = self.ex.dispatch({"tool": "read_file", "arguments": {"path": "../secret.txt"}})
        self.assertFalse(r["ok"])

    def test_missing_file_is_failure_not_malformed(self):
        r = self.ex.dispatch({"tool": "read_file", "arguments": {"path": "nope.py"}})
        self.assertFalse(r["ok"])
        self.assertFalse(r["malformed"])

    def test_malformed_calls(self):
        for bad in ({"tool": "read_file", "arguments": {}},
                    {"tool": "nope", "arguments": {}},
                    {"tool": "search_code", "arguments": {"query": 42}},
                    {"tool": "apply_patch", "arguments": {"diff": "not a diff"}},
                    "not-a-dict"):
            ok, _ = validate_call(bad)
            self.assertFalse(ok, bad)
            r = self.ex.dispatch(bad)
            self.assertFalse(r["ok"])
            self.assertTrue(r["malformed"], bad)

    def test_bad_patch_fails_cleanly(self):
        # Corrupt the removed line so the hunk context cannot match the file.
        bad_diff = self.repo.fix_diff.replace("n % 2 == 1", "n % 2 == 9")
        self.assertNotEqual(bad_diff, self.repo.fix_diff)
        r = self.ex.dispatch({"tool": "apply_patch", "arguments": {"diff": bad_diff}})
        self.assertFalse(r["ok"])

    def test_parse_diff_rejects_garbage(self):
        with self.assertRaises(PatchError):
            parse_unified_diff("hello world")


if __name__ == "__main__":
    unittest.main()
