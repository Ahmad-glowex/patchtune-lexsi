"""Synthetic task generation for PatchTune-mini.

Each task is a self-contained record:

- ``id`` / ``split``         : unique id; "train" or "heldout"
- ``instruction`` / ``prompt``: natural-language bug report (prompt adds the
  tool manual + expected JSON call format, ready to feed a model)
- ``files``                  : {relative path: content} of the buggy toy repo
- ``target_file``            : the file containing the planted bug
- ``bug_kind``               : which defect family was planted
- ``expected_patch``         : the canonical correct unified diff
- ``expected_calls``         : the canonical correct tool-call sequence
  (search -> read -> apply_patch -> run_tests)
- ``seed``                   : generation seed (reproducibility)

Train and held-out tasks use *disjoint* module-name pools and *disjoint*
instruction phrasings (see ``toyrepo.generate``), so held-out performance
is not inflated by memorised surface forms.

CLI:
    python -m patchtune.tasks --out-dir data --train 150 --heldout 50
Writes ``data/train.jsonl`` and ``data/heldout.jsonl`` — plain JSONL that
loads directly with ``datasets.load_dataset("json", ...)``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from toyrepo.generate import random_instance

TOOL_MANUAL = """You are a coding agent. Fix the bug using tools, one call per step.
Reply with exactly one JSON object per step: {"tool": <name>, "arguments": {...}}.

Tools:
- read_file(path): read a file in the repo.
- search_code(query): find lines containing a string.
- apply_patch(diff): apply a unified diff.
- run_tests(): run the test suite."""


def make_task(rng: random.Random, split: str, index: int) -> dict:
    repo = random_instance(rng, split)
    bk = repo.bug
    phrasings = bk.train_phrasings if split == "train" else bk.heldout_phrasings
    instruction = rng.choice(phrasings).format(func=bk.func, module=repo.module)
    expected_calls = [
        {"tool": "search_code", "arguments": {"query": bk.func}},
        {"tool": "read_file", "arguments": {"path": repo.target_file}},
        {"tool": "apply_patch", "arguments": {"diff": repo.fix_diff}},
        {"tool": "run_tests", "arguments": {}},
    ]
    return {
        "id": f"{split}-{index:04d}",
        "split": split,
        "seed": rng.randrange(2**31),
        "instruction": instruction,
        "prompt": f"{TOOL_MANUAL}\n\nTask: {instruction}",
        "files": repo.files,
        "target_file": repo.target_file,
        "bug_kind": bk.kind,
        "expected_patch": repo.fix_diff,
        "expected_calls": expected_calls,
    }


def build_dataset(n_train: int = 150, n_heldout: int = 50, seed: int = 20261113) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    return {
        "train": [make_task(rng, "train", i) for i in range(n_train)],
        "heldout": [make_task(rng, "heldout", i) for i in range(n_heldout)],
    }


def write_jsonl(tasks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")


def load_jsonl(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PatchTune-mini datasets")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--heldout", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()
    data = build_dataset(args.train, args.heldout, args.seed)
    out = Path(args.out_dir)
    for split, tasks in data.items():
        write_jsonl(tasks, out / f"{split}.jsonl")
        print(f"{split}: {len(tasks)} tasks -> {out / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
