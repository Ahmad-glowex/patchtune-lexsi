"""Toy-repo generator for PatchTune-mini.

Creates a tiny Python package (``tinylib``) containing exactly one planted,
programmatically-fixable bug, plus a unittest suite that fails because of
the bug and passes once the bug is fixed.

Every bug kind is a single-line defect, so the correct fix is a small,
deterministic unified diff — which is what makes the agent's behaviour
programmatically verifiable (see ``patchtune.reward``).
"""

from __future__ import annotations

import difflib
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BugKind:
    """One family of single-line bugs."""

    kind: str
    func: str                      # function defined in the module
    signature: str                 # e.g. "def add_numbers(a, b):"
    correct_line: str              # the line as it should be
    buggy_line: str                # the planted defect
    docstring: str
    tests: tuple[str, ...]         # assert statements (inside test methods)
    train_modules: tuple[str, ...] = field(default_factory=tuple)
    heldout_modules: tuple[str, ...] = field(default_factory=tuple)
    train_phrasings: tuple[str, ...] = field(default_factory=tuple)
    heldout_phrasings: tuple[str, ...] = field(default_factory=tuple)


_COMMON_TRAIN_PHRASINGS = (
    "The test suite is failing: {func} in {module}.py does not behave as documented. Find the bug and fix it.",
    "Users report that {func} (in {module}.py) returns wrong results. Investigate the code, patch the bug, and make the tests pass.",
    "Something is wrong with {func} in module {module}. Locate the defect, apply a fix, and verify with the test suite.",
)
_COMMON_HELDOUT_PHRASINGS = (
    "CI is red because of {module}.py — the {func} helper misbehaves. Track it down, repair it, and confirm the suite goes green.",
    "A regression slipped into {func} inside {module}.py. Please diagnose the faulty line, correct it, and re-run the tests.",
)


BUG_KINDS: tuple[BugKind, ...] = (
    BugKind(
        kind="add_becomes_subtract",
        func="add_numbers",
        signature="def add_numbers(a, b):",
        correct_line="    return a + b",
        buggy_line="    return a - b",
        docstring="Return the sum of a and b.",
        tests=("self.assertEqual(add_numbers(2, 3), 5)",
               "self.assertEqual(add_numbers(-1, 1), 0)",
               "self.assertEqual(add_numbers(10, 5), 15)"),
        train_modules=("arith_core", "math_helpers", "calc_utils"),
        heldout_modules=("number_tools", "sum_engine"),
    ),
    BugKind(
        kind="multiply_becomes_add",
        func="multiply",
        signature="def multiply(a, b):",
        correct_line="    return a * b",
        buggy_line="    return a + b",
        docstring="Return the product of a and b.",
        tests=("self.assertEqual(multiply(3, 4), 12)",
               "self.assertEqual(multiply(0, 7), 0)",
               "self.assertEqual(multiply(-2, 5), -10)"),
        train_modules=("arith_core", "product_lib", "calc_utils"),
        heldout_modules=("number_tools", "times_table"),
    ),
    BugKind(
        kind="reverse_is_identity",
        func="reverse_text",
        signature="def reverse_text(s):",
        correct_line="    return s[::-1]",
        buggy_line="    return s[:]",
        docstring="Return the string s reversed.",
        tests=("self.assertEqual(reverse_text('abc'), 'cba')",
               "self.assertEqual(reverse_text(''), '')",
               "self.assertEqual(reverse_text('racecar'), 'racecar')"),
        train_modules=("text_ops", "string_lab", "word_tools"),
        heldout_modules=("text_kit", "glyph_utils"),
    ),
    BugKind(
        kind="max_becomes_min",
        func="list_max",
        signature="def list_max(xs):",
        correct_line="    return max(xs)",
        buggy_line="    return min(xs)",
        docstring="Return the largest value in xs.",
        tests=("self.assertEqual(list_max([1, 9, 3]), 9)",
               "self.assertEqual(list_max([-5, -2]), -2)",
               "self.assertEqual(list_max([4]), 4)"),
        train_modules=("seq_ops", "list_lab", "array_tools"),
        heldout_modules=("seq_kit", "vector_utils"),
    ),
    BugKind(
        kind="discount_sign_flip",
        func="apply_discount",
        signature="def apply_discount(price, pct):",
        correct_line="    return round(price * (1 - pct / 100), 2)",
        buggy_line="    return round(price * (1 + pct / 100), 2)",
        docstring="Return price reduced by pct percent, rounded to 2dp.",
        tests=("self.assertEqual(apply_discount(100, 10), 90.0)",
               "self.assertEqual(apply_discount(50, 0), 50.0)",
               "self.assertEqual(apply_discount(80, 25), 60.0)"),
        train_modules=("pricing", "shop_math", "billing_ops"),
        heldout_modules=("checkout_math", "price_engine"),
    ),
    BugKind(
        kind="clamp_swapped",
        func="clamp_value",
        signature="def clamp_value(x, lo, hi):",
        correct_line="    return max(lo, min(hi, x))",
        buggy_line="    return min(lo, max(hi, x))",
        docstring="Clamp x into the inclusive range [lo, hi].",
        tests=("self.assertEqual(clamp_value(5, 0, 10), 5)",
               "self.assertEqual(clamp_value(-3, 0, 10), 0)",
               "self.assertEqual(clamp_value(99, 0, 10), 10)"),
        train_modules=("range_ops", "bounds_lib", "limit_tools"),
        heldout_modules=("range_kit", "guard_utils"),
    ),
    BugKind(
        kind="is_even_inverted",
        func="is_even",
        signature="def is_even(n):",
        correct_line="    return n % 2 == 0",
        buggy_line="    return n % 2 == 1",
        docstring="Return True when n is even.",
        tests=("self.assertTrue(is_even(4))",
               "self.assertFalse(is_even(7))",
               "self.assertTrue(is_even(0))"),
        train_modules=("parity", "num_checks", "int_ops"),
        heldout_modules=("parity_kit", "bool_utils"),
    ),
    BugKind(
        kind="greeting_missing_bang",
        func="greet",
        signature="def greet(name):",
        correct_line="    return f\"Hello, {name}!\"",
        buggy_line="    return f\"Hello, {name}\"",
        docstring="Return a friendly greeting ending with an exclamation mark.",
        tests=("self.assertEqual(greet('Ada'), 'Hello, Ada!')",
               "self.assertEqual(greet('Bo'), 'Hello, Bo!')"),
        train_modules=("messages", "hello_ops", "chat_text"),
        heldout_modules=("greet_kit", "salutation_utils"),
    ),
)

for _bk in BUG_KINDS:
    object.__setattr__(_bk, "train_phrasings", _COMMON_TRAIN_PHRASINGS)
    object.__setattr__(_bk, "heldout_phrasings", _COMMON_HELDOUT_PHRASINGS)

BUG_BY_KIND = {bk.kind: bk for bk in BUG_KINDS}

PACKAGE = "tinylib"


def _module_source(bk: BugKind, *, buggy: bool) -> str:
    line = bk.buggy_line if buggy else bk.correct_line
    return (
        f'"""{bk.docstring}"""\n\n\n'
        f"{bk.signature}\n"
        f'    """{bk.docstring}"""\n'
        f"{line}\n"
    )


def _test_source(bk: BugKind, module: str) -> str:
    asserts = "\n        ".join(bk.tests)
    return (
        "import unittest\n\n"
        f"from tinylib.{module} import {bk.func}\n\n\n"
        f"class Test{bk.func.title().replace('_', '')}(unittest.TestCase):\n"
        "    def test_behaviour(self):\n"
        f"        {asserts}\n\n\n"
        'if __name__ == "__main__":\n'
        "    unittest.main()\n"
    )


def unified_fix(bk: BugKind, module: str) -> str:
    """The canonical correct patch (unified diff) for this bug instance."""
    buggy = _module_source(bk, buggy=True).splitlines(keepends=True)
    fixed = _module_source(bk, buggy=False).splitlines(keepends=True)
    path = f"{PACKAGE}/{module}.py"
    diff = difflib.unified_diff(buggy, fixed, fromfile=f"a/{path}", tofile=f"b/{path}")
    return "".join(diff)


@dataclass
class ToyRepo:
    """A generated toy-repo instance (in-memory file map)."""

    bug: BugKind
    module: str
    files: dict[str, str]           # relative path -> content
    fix_diff: str                   # canonical correct unified diff
    target_file: str                # relative path of the buggy file

    def materialize(self, root: Path) -> Path:
        root = Path(root)
        for rel, content in self.files.items():
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        return root


def generate_instance(kind: str, module: str) -> ToyRepo:
    """Build one toy repo with the given bug kind in the given module."""
    bk = BUG_BY_KIND[kind]
    target = f"{PACKAGE}/{module}.py"
    files = {
        f"{PACKAGE}/__init__.py": "",
        target: _module_source(bk, buggy=True),
        f"tests/__init__.py": "",
        f"tests/test_{module}.py": _test_source(bk, module),
        "README.md": f"# tinylib\n\nTiny library. Run tests with "
                     f"`python -m unittest discover -s tests`.\n",
    }
    return ToyRepo(bug=bk, module=module, files=files,
                   fix_diff=unified_fix(bk, module), target_file=target)


def random_instance(rng: random.Random, split: str) -> ToyRepo:
    """Random instance; module pools and phrasings are disjoint per split."""
    bk = rng.choice(BUG_KINDS)
    pool = bk.train_modules if split == "train" else bk.heldout_modules
    return generate_instance(bk.kind, rng.choice(pool))
