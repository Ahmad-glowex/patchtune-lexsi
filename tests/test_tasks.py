import unittest

from patchtune.tasks import build_dataset
from patchtune.tools import validate_call


class TestTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_dataset(n_train=30, n_heldout=12, seed=99)

    def test_sizes(self):
        self.assertEqual(len(self.data["train"]), 30)
        self.assertEqual(len(self.data["heldout"]), 12)

    def test_module_pools_disjoint(self):
        def mods(tasks):
            return {t["target_file"] for t in tasks}
        self.assertTrue(mods(self.data["train"]).isdisjoint(mods(self.data["heldout"])))

    def test_expected_calls_well_formed(self):
        for split in ("train", "heldout"):
            for t in self.data[split]:
                self.assertEqual(len(t["expected_calls"]), 4)
                for c in t["expected_calls"]:
                    ok, reason = validate_call(c)
                    self.assertTrue(ok, f"{t['id']}: {reason}")
                self.assertIn(t["target_file"], t["files"])
                self.assertTrue(t["expected_patch"].startswith("--- a/"))


if __name__ == "__main__":
    unittest.main()
