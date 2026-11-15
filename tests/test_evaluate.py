import json
import unittest

from patchtune.evaluate import (
    MockAgentModel,
    evaluate_model,
    first_call_from_text,
    run_interactive_rollout,
)
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


class TestFirstCallFromText(unittest.TestCase):
    def test_json_line_after_prose(self):
        text = 'Let me look.\n{"tool": "search_code", "arguments": {"query": "is_even"}}\nmore'
        call, prefix, bad = first_call_from_text(text)
        self.assertEqual(call, {"tool": "search_code", "arguments": {"query": "is_even"}})
        self.assertTrue(prefix.endswith('{"tool": "search_code", "arguments": {"query": "is_even"}}'))
        self.assertEqual(bad, 0)

    def test_function_syntax_line(self):
        call, _, bad = first_call_from_text('search_code(query="is_even")')
        self.assertEqual(call["tool"], "search_code")
        self.assertEqual(call["arguments"], {"query": "is_even"})
        self.assertEqual(bad, 0)

    def test_malformed_line_before_good_call_counts(self):
        text = '{"tool": "read_file", "arguments":\n{"tool": "run_tests", "arguments": {}}'
        call, _, bad = first_call_from_text(text)
        self.assertEqual(call["tool"], "run_tests")
        self.assertEqual(bad, 1)

    def test_no_call(self):
        call, _, bad = first_call_from_text("I cannot fix this bug.")
        self.assertIsNone(call)
        self.assertEqual(bad, 0)


class _ScriptedInteractive:
    """Interactive-model double: replays scripted completions per turn.

    ``script(task, conversation) -> str`` decides each turn's generation
    (the turn index is recoverable from the count of "Result:" lines in
    the conversation). No GPU / transformers involved.
    """

    interactive = True

    def __init__(self, script, max_turns=6):
        self.script = script
        self.max_turns = max_turns

    def generate(self, task, executor):
        return run_interactive_rollout(
            task, executor, lambda conv: self.script(task, conv),
            max_turns=self.max_turns)


def _canonical_script(task, conversation):
    turn = conversation.count("Result:")
    if turn >= len(task["expected_calls"]):
        return ""  # task done: emit no further call, loop stops
    return json.dumps(task["expected_calls"][turn])


class TestInteractiveRollout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = build_dataset(n_train=0, n_heldout=4, seed=5)["heldout"]

    def test_canonical_sequence_succeeds(self):
        summary = evaluate_model(_ScriptedInteractive(_canonical_script), self.tasks)
        self.assertEqual(summary["task_success_rate"], 1.0)
        self.assertEqual(summary["tool_call_accuracy"], 1.0)
        self.assertEqual(summary["malformed_rate"], 0.0)
        for row in summary["per_task"]:
            self.assertEqual(row["calls"], 4)
            self.assertTrue(row["success"])

    def test_all_calls_in_one_generation_still_one_per_turn(self):
        def rambling(task, conversation):
            if "Result:" not in conversation:
                return "\n".join(json.dumps(c) for c in task["expected_calls"])
            return _canonical_script(task, conversation)

        summary = evaluate_model(_ScriptedInteractive(rambling), self.tasks)
        self.assertEqual(summary["task_success_rate"], 1.0)
        for row in summary["per_task"]:
            self.assertEqual(row["calls"], 4)  # tail of gen 1 discarded

    def test_prose_only_model_fails_cleanly(self):
        summary = evaluate_model(
            _ScriptedInteractive(lambda task, conv: "I give up."), self.tasks)
        self.assertEqual(summary["task_success_rate"], 0.0)
        for row in summary["per_task"]:
            self.assertEqual(row["calls"], 0)

    def test_turn_cap_stops_a_looping_model(self):
        loop = lambda task, conv: json.dumps(task["expected_calls"][0])  # noqa: E731
        summary = evaluate_model(
            _ScriptedInteractive(loop, max_turns=3), self.tasks)
        for row in summary["per_task"]:
            self.assertEqual(row["calls"], 3)
            self.assertFalse(row["success"])


if __name__ == "__main__":
    unittest.main()
