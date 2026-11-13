import tempfile
import unittest
from pathlib import Path

from toyrepo.generate import BUG_KINDS, generate_instance
from patchtune.tools import ToolExecutor


class TestGenerator(unittest.TestCase):
    def test_every_bug_kind_fails_then_fix_passes(self):
        for bk in BUG_KINDS:
            module = bk.train_modules[0]
            repo = generate_instance(bk.kind, module)
            with tempfile.TemporaryDirectory() as tmp:
                repo.materialize(Path(tmp))
                ex = ToolExecutor(tmp)
                before = ex.run_tests()
                self.assertFalse(before["passed"], f"{bk.kind} should fail pre-fix")
                res = ex.dispatch({"tool": "apply_patch",
                                   "arguments": {"diff": repo.fix_diff}})
                self.assertTrue(res["ok"], f"{bk.kind} fix should apply: {res}")
                after = ex.run_tests()
                self.assertTrue(after["passed"], f"{bk.kind} should pass post-fix")

    def test_target_file_exists_in_file_map(self):
        repo = generate_instance("add_becomes_subtract", "arith_core")
        self.assertIn(repo.target_file, repo.files)
        self.assertIn("tests/test_arith_core.py", repo.files)


if __name__ == "__main__":
    unittest.main()
