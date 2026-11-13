import unittest

from patchtune.evaluate import MockAgentModel, evaluate_model
from patchtune.tasks import build_dataset


class TestEvaluateMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = build_dataset(n_train=0, n_heldout=8, seed=5)["heldout"]

    def test_tuned_beats_base(self):
        base = evaluate_model(MockAgentModel("base"), self.tasks)
        tuned = evaluate_model(MockAgentModel("tuned"), self.tasks)
        self.assertGreater(tuned["task_success_rate"], base["task_success_rate"])
        self.assertGreaterEqual(tuned["task_success_rate"], 0.99)
        self.assertLess(tuned["malformed_rate"], base["malformed_rate"])
        for summary in (base, tuned):
            for key in ("task_success_rate", "tool_call_accuracy",
                        "avg_calls_per_task", "malformed_rate"):
                self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
