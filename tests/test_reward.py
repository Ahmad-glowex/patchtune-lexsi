import json
import unittest

from patchtune.reward import parse_rollout, reward_func, score_trajectory
from patchtune.tasks import make_task
import random


def _task():
    return make_task(random.Random(7), "train", 0)


class TestParseRollout(unittest.TestCase):
    def test_list_of_dicts(self):
        calls, bad = parse_rollout([{"tool": "run_tests", "arguments": {}}])
        self.assertEqual(len(calls), 1)
        self.assertEqual(bad, 0)

    def test_json_string(self):
        s = json.dumps([{"tool": "read_file", "arguments": {"path": "a.py"}}])
        calls, bad = parse_rollout(s)
        self.assertEqual(calls[0]["tool"], "read_file")
        self.assertEqual(bad, 0)

    def test_flattened_function_lines(self):
        # AgentTune may pass completions as flattened strings (research R1)
        text = 'search_code(query="is_even")\nread_file(path="tinylib/parity.py")\nrun_tests()'
        calls, bad = parse_rollout(text)
        self.assertEqual([c["tool"] for c in calls],
                         ["search_code", "read_file", "run_tests"])
        self.assertEqual(calls[0]["arguments"], {"query": "is_even"})
        self.assertEqual(bad, 0)

    def test_flattened_malformed_line_counted(self):
        calls, bad = parse_rollout('read_file(path=)\nrun_tests()')
        self.assertEqual(len(calls), 1)
        self.assertEqual(bad, 1)

    def test_prose_is_ignored_not_penalised(self):
        calls, bad = parse_rollout("Let me think about this bug first.\nrun_tests()")
        self.assertEqual(len(calls), 1)
        self.assertEqual(bad, 0)

    def test_openai_style_message(self):
        msg = [{"role": "assistant", "tool_calls": [
            {"function": {"name": "run_tests", "arguments": "{}"}}]}]
        calls, bad = parse_rollout(msg)
        self.assertEqual(calls, [{"tool": "run_tests", "arguments": {}}])


class TestScoring(unittest.TestCase):
    def test_perfect_trajectory_scores_highest(self):
        task = _task()
        perfect = score_trajectory(task["expected_calls"], task, final_tests_pass=True)
        empty = score_trajectory([], task, final_tests_pass=False)
        self.assertGreater(perfect["total"], empty["total"])
        self.assertEqual(perfect["matched_calls"], len(task["expected_calls"]))
        self.assertAlmostEqual(perfect["total"], 4.0)

    def test_loops_and_malformed_penalised(self):
        task = _task()
        good = score_trajectory(task["expected_calls"], task, final_tests_pass=True)
        loopy = task["expected_calls"] + [task["expected_calls"][0]] * 3 + \
            [{"tool": "read_file", "arguments": {}}]
        bad = score_trajectory(loopy, task, final_tests_pass=True)
        self.assertLess(bad["total"], good["total"])
        self.assertEqual(bad["loop_repeats"], 3)
        self.assertEqual(bad["malformed"], 1)

    def test_reward_func_with_flattened_strings(self):
        task = _task()
        completion = "\n".join(
            f'{c["tool"]}({", ".join(f"{k}={v!r}" for k, v in c["arguments"].items() if k != "diff")})'
            if c["tool"] != "apply_patch" else "run_tests()"
            for c in task["expected_calls"])
        rewards = reward_func([completion], task_json=[json.dumps(task)])
        self.assertEqual(len(rewards), 1)
        self.assertIsInstance(rewards[0], float)


if __name__ == "__main__":
    unittest.main()
