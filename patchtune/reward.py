"""Reward / verifier functions for PatchTune-mini.

Scoring a trajectory (list of tool calls) against a task:

+ matched expected calls (right tool, right args, in order) .. up to +2.0
+ verified final outcome (test suite passes) ................ +2.0
- each malformed call ....................................... -0.5
- each repeated identical call (looping) .................... -0.3
- each call beyond (expected + 2) ........................... -0.1

IMPORTANT — AgentTune calling convention (per AgentTune's RL docs, see
research-report R1): GRPO reward functions may receive the completion as a
**flattened string**, not a list of message dicts. All parsing here is
therefore defensive: :func:`parse_rollout` accepts

* a list of call dicts: ``[{"tool": ..., "arguments": {...}}, ...]``
* a list of message dicts with ``tool_calls`` / JSON ``content``
* a single JSON string (object, list, or JSON-lines)
* flattened text with one call per line, in forms like
  ``read_file(path="tinylib/x.py")``, ``search_code(query='is_even')``,
  ``TOOL: run_tests {}`` or ``{"tool": "run_tests", "arguments": {}}``

Lines that *look like* attempted calls but cannot be parsed count as
malformed (they are returned in ``unparsed``), which the scorer penalises.
"""

from __future__ import annotations

import ast
import json
import re

from .tools import TOOL_SPECS, validate_call

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_CALLISH_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>.*)\)\s*$", re.DOTALL)
_TOOL_PREFIX_RE = re.compile(r"^(?:TOOL|CALL|ACTION)\s*:\s*", re.IGNORECASE)


def _call_from_function_syntax(name: str, arg_src: str) -> dict | None:
    """Parse ``name(k=v, ...)`` where values are Python literals."""
    arg_src = arg_src.strip()
    if not arg_src:
        return {"tool": name, "arguments": {}}
    try:
        tree = ast.parse(f"f({arg_src})", mode="eval")
        call = tree.body
        args = {}
        for kw in call.keywords:  # type: ignore[attr-defined]
            args[kw.arg] = ast.literal_eval(kw.value)
        if call.args:  # type: ignore[attr-defined]
            # one positional arg allowed (e.g. read_file("path"))
            spec_keys = list(TOOL_SPECS.get(name, {}))
            if len(call.args) == 1 and spec_keys:  # type: ignore[attr-defined]
                args[spec_keys[0]] = ast.literal_eval(call.args[0])  # type: ignore[attr-defined]
            else:
                return None
        return {"tool": name, "arguments": args}
    except (SyntaxError, ValueError):
        return None


def _parse_one(text: str) -> dict | None:
    """Best-effort parse of one text chunk into a call dict (or None)."""
    text = _TOOL_PREFIX_RE.sub("", text.strip())
    if not text:
        return None
    # JSON object / JSON with tool key
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            obj.setdefault("arguments", {})
            return obj
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return {"tool": obj["name"], "arguments": obj["arguments"]}
    except (json.JSONDecodeError, TypeError):
        pass
    # function-call syntax
    m = _CALLISH_RE.match(text)
    if m:
        return _call_from_function_syntax(m.group("name"), m.group("args"))
    return None


def parse_rollout(completion) -> tuple[list[dict], int]:
    """Parse a model completion into (calls, malformed_attempts).

    Never raises. ``malformed_attempts`` counts chunks that looked like
    tool-call attempts but could not be parsed.
    """
    calls: list[dict] = []
    malformed = 0

    if completion is None:
        return calls, 0
    if isinstance(completion, dict):  # single call dict
        return [completion], 0
    if isinstance(completion, (list, tuple)):
        for item in completion:
            if isinstance(item, dict):
                if "tool" in item:  # direct call dict
                    calls.append({"tool": item["tool"],
                                  "arguments": item.get("arguments", {})})
                elif item.get("tool_calls"):  # OpenAI-style message
                    for tc in item["tool_calls"]:
                        fn = tc.get("function", tc)
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                malformed += 1
                                continue
                        calls.append({"tool": fn.get("name"), "arguments": args})
                elif isinstance(item.get("content"), str):
                    sub, bad = parse_rollout(item["content"])
                    calls.extend(sub)
                    malformed += bad
                else:
                    malformed += 1
            elif isinstance(item, str):
                sub, bad = parse_rollout(item)
                calls.extend(sub)
                malformed += bad
            else:
                malformed += 1
        return calls, malformed
    if isinstance(completion, str):
        text = completion.strip()
        if not text:
            return calls, 0
        # Whole-string JSON (list or object)?
        try:
            obj = json.loads(text)
            return parse_rollout(obj)
        except (json.JSONDecodeError, TypeError):
            pass
        # Line-oriented: JSONL or function-call lines. Lines that are not
        # call-shaped (prose, markdown fences) are ignored, not penalised;
        # lines that start like a call but fail to parse are malformed.
        for raw in text.splitlines():
            line = raw.strip().strip("`")
            if not line:
                continue
            parsed = _parse_one(line)
            if parsed is not None:
                calls.append(parsed)
            elif line.startswith("{") or _CALLISH_RE.match(line) or \
                    _TOOL_PREFIX_RE.match(line):
                malformed += 1
        return calls, malformed
    return calls, 1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _args_equal(a: dict, b: dict) -> bool:
    return a == b


def match_expected(calls: list[dict], expected: list[dict]) -> list[int]:
    """Greedy in-order match: indices into ``calls`` that realise expected
    calls (same tool + same arguments), scanning forward."""
    matched: list[int] = []
    pos = 0
    for exp in expected:
        for j in range(pos, len(calls)):
            if calls[j].get("tool") == exp["tool"] and \
                    _args_equal(calls[j].get("arguments", {}), exp.get("arguments", {})):
                matched.append(j)
                pos = j + 1
                break
    return matched


def score_trajectory(calls: list[dict], task: dict, *,
                     final_tests_pass: bool | None = None,
                     malformed_attempts: int = 0) -> dict:
    """Score a parsed trajectory. Returns a breakdown dict with ``total``."""
    expected = task["expected_calls"]
    malformed = malformed_attempts
    for c in calls:
        ok, _ = validate_call(c)
        if not ok:
            malformed += 1

    matched = match_expected(calls, expected)
    match_reward = 2.0 * len(matched) / max(1, len(expected))
    outcome_reward = 2.0 if final_tests_pass else 0.0

    seen: dict[str, int] = {}
    loop_repeats = 0
    for c in calls:
        key = json.dumps(c, sort_keys=True, default=str)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            loop_repeats += 1
    extra = max(0, len(calls) - len(expected) - 2)

    total = (match_reward + outcome_reward
             - 0.5 * malformed - 0.3 * loop_repeats - 0.1 * extra)
    return {
        "total": round(total, 4),
        "match_reward": round(match_reward, 4),
        "outcome_reward": outcome_reward,
        "matched_calls": len(matched),
        "expected_calls": len(expected),
        "malformed": malformed,
        "loop_repeats": loop_repeats,
        "extra_calls": extra,
    }


def verify_outcome(executor, calls: list[dict]) -> bool:
    """Execute calls in a sandbox executor; True iff the suite passes at
    the end of the trajectory (checked with a fresh run_tests)."""
    for c in calls:
        executor.dispatch(c)
    return bool(executor.run_tests().get("passed"))


# ---------------------------------------------------------------------------
# Trainer-facing reward function (AgentTune / TRL GRPO style)
# ---------------------------------------------------------------------------

def reward_func(completions, task_json=None, **kwargs) -> list[float]:
    """GRPO reward function.

    ``completions``: list of completions in any shape parse_rollout accepts
    (AgentTune may pass flattened strings — handled). ``task_json``: list of
    JSON strings, one per completion, each encoding the task record (TRL
    passes dataset columns as kwargs). Outcome reward is only included when
    the caller provides pre-computed ``final_tests_pass`` per completion —
    executing model output inside a reward function is deliberately *not*
    done here (sandboxing belongs in the rollout harness, see evaluate.py).
    """
    rewards: list[float] = []
    passes = kwargs.get("final_tests_pass") or [None] * len(completions)
    for i, completion in enumerate(completions):
        calls, bad = parse_rollout(completion)
        if task_json is not None:
            task = json.loads(task_json[i]) if isinstance(task_json[i], str) else task_json[i]
        else:  # scoring without a task: only format/loop penalties apply
            task = {"expected_calls": []}
        bd = score_trajectory(calls, task,
                              final_tests_pass=passes[i] if i < len(passes) else None,
                              malformed_attempts=bad)
        rewards.append(bd["total"])
    return rewards
