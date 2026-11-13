"""Evaluation harness for PatchTune-mini: base vs tuned, before/after metrics.

Metrics (the hackathon rules require more than one; we report four):
- ``task_success_rate``   : fraction of tasks whose suite passes after the
                            model's trajectory is executed in the sandbox
- ``tool_call_accuracy``  : matched expected calls / total calls the model
                            made (greedy in-order match, tool + args)
- ``avg_calls_per_task``  : efficiency (Problem Statement #2)
- ``malformed_rate``      : malformed calls / total attempted calls

Models
------
``MockAgentModel`` runs WITHOUT a GPU or a real model, so the whole
pipeline can be verified end-to-end today:

- mode="base"  mimics an untuned small model: sometimes emits a malformed
  call, loops a search, picks the wrong tool, or gives up.
- mode="tuned" mimics the post-trained model: follows the canonical
  sequence, with only occasional slips.

``HFModel`` is the real-model adapter (transformers, greedy JSON-line
generation) — used on Colab for the actual base/tuned runs; it is only
imported when requested.

CLI:
    python -m patchtune.evaluate --model mock-base  --tasks data/heldout.jsonl
    python -m patchtune.evaluate --model mock-tuned --tasks data/heldout.jsonl
Writes a JSON summary (and prints it).
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from toyrepo.generate import ToyRepo, BUG_BY_KIND

from .reward import match_expected, parse_rollout
from .tasks import load_jsonl
from .tools import ToolExecutor, validate_call


# ---------------------------------------------------------------------------
# Mock model (no GPU, deterministic per task)
# ---------------------------------------------------------------------------

class MockAgentModel:
    def __init__(self, mode: str = "base"):
        assert mode in ("base", "tuned")
        self.mode = mode

    def generate(self, task: dict) -> list[dict]:
        rng = random.Random(hash((task["id"], self.mode)) & 0xFFFFFFFF)
        expected = [dict(c, arguments=dict(c["arguments"])) for c in task["expected_calls"]]
        if self.mode == "tuned":
            calls = expected
            if rng.random() < 0.10:  # occasional slip: skip the read step
                calls = [c for c in calls if c["tool"] != "read_file"]
            return calls
        # --- base behaviour: noisy, unreliable ---------------------------
        calls: list[dict] = []
        r = rng.random()
        if r < 0.30:
            # malformed first attempt (missing args / unknown tool)
            calls.append(rng.choice([
                {"tool": "read_file", "arguments": {}},
                {"tool": "open_file", "arguments": {"path": task["target_file"]}},
            ]))
        if r < 0.75:  # usually at least searches
            calls.append(expected[0])
            if rng.random() < 0.35:  # loop: repeat the same search
                calls.append(dict(expected[0]))
        if rng.random() < 0.55:  # often reads the file
            calls.append(expected[1])
        if rng.random() < 0.45:  # sometimes attempts the (correct) patch
            calls.append(expected[2])
        calls.append(expected[3])  # runs tests at the end
        return calls


class HFModel:  # pragma: no cover - requires GPU/transformers (Colab)
    """Real-model adapter: greedy generation, one JSON call per line."""

    def __init__(self, model_id: str, adapter_id: str | None = None, max_calls: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(adapter_id or model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        if adapter_id:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_id)
        self.max_calls = max_calls

    def generate(self, task: dict, executor: ToolExecutor) -> list[dict]:
        raise NotImplementedError(
            "Interactive rollout loop for the real model is run inside the "
            "Colab notebook (train_grpo_colab.ipynb), where generation and "
            "tool execution interleave; this stub marks the integration point.")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def evaluate_model(model, tasks: list[dict]) -> dict:
    rows = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix="patchtune-eval-") as tmp:
            repo = ToyRepo(bug=BUG_BY_KIND[task["bug_kind"]], module="",
                           files=task["files"], fix_diff=task["expected_patch"],
                           target_file=task["target_file"])
            repo.materialize(Path(tmp))
            executor = ToolExecutor(tmp)
            calls = model.generate(task)
            if not isinstance(calls, list):  # raw completion from a real model
                calls, extra_bad = parse_rollout(calls)
            else:
                extra_bad = 0
            malformed = extra_bad
            for c in calls:
                ok, _ = validate_call(c)
                if not ok:
                    malformed += 1
                executor.dispatch(c)
            final = executor.run_tests()
            matched = len(match_expected(calls, task["expected_calls"]))
            rows.append({
                "id": task["id"],
                "success": bool(final.get("passed")),
                "calls": len(calls),
                "matched": matched,
                "malformed": malformed,
            })
    n = max(1, len(rows))
    total_calls = sum(r["calls"] for r in rows)
    return {
        "tasks": len(rows),
        "task_success_rate": round(sum(r["success"] for r in rows) / n, 4),
        "tool_call_accuracy": round(sum(r["matched"] for r in rows) / max(1, total_calls), 4),
        "avg_calls_per_task": round(total_calls / n, 3),
        "malformed_rate": round(sum(r["malformed"] for r in rows) / max(1, total_calls), 4),
        "per_task": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PatchTune-mini models")
    ap.add_argument("--model", choices=["mock-base", "mock-tuned"], required=True)
    ap.add_argument("--tasks", default="data/heldout.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tasks = load_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    model = MockAgentModel("base" if args.model == "mock-base" else "tuned")
    summary = evaluate_model(model, tasks)
    summary["model"] = args.model
    summary["tasks_file"] = args.tasks
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))
    out = args.out or f"results/eval_{args.model.replace('-', '_')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
