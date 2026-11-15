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

``HFModel`` is the real-model adapter (transformers, greedy generation):
the base CohereLabs/tiny-aya-global model loaded exactly as training
loads it (bf16, 4-bit bitsandbytes by default — see
``train_grpo._model_init_kwargs``), optionally with a LoRA adapter
attached via peft. It runs an interactive rollout per task: generate,
take the FIRST tool call in the generation, execute it in the sandbox,
feed the result back, repeat — up to ``max_turns`` turns.

Rollout-format assumptions (training consistency)
-------------------------------------------------
PATH B (TRL) trained on the raw string ``task["prompt"]`` (TOOL_MANUAL +
instruction: "Reply with exactly one JSON object per step") and scored
the whole single completion with :func:`parse_rollout`; it never fed
tool results back. The eval loop therefore:

- starts from the identical ``task["prompt"]`` string, raw (no chat
  template — TRL treats string prompts as plain text);
- honours "one call per step": only the first call of each generation
  is executed, and only the text up to that call is appended to the
  conversation (a model that emits all four calls in one shot still
  gets them executed one per turn, each time seeing the real result);
- feeds each result back as one line ``Result: {json}`` (truncated) —
  a format training never showed the model, kept deliberately plain;
- caps turns at 6 (PATH A's ``max_steps_per_turn``; the canonical
  trajectory is 4 calls) and each generation at 512 new tokens
  (training's ``max_completion_length`` default).

CLI:
    python -m patchtune.evaluate --model mock-base  --tasks data/heldout.jsonl
    python -m patchtune.evaluate --model mock-tuned --tasks data/heldout.jsonl
    python -m patchtune.evaluate --model hf --data data/heldout.jsonl
    python -m patchtune.evaluate --model hf --adapter runs/grpo-full \
        --data data/heldout.jsonl --out results/eval_hf_tuned.json
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

# Same base model as train_grpo.BASE_MODEL (duplicated as a literal so
# this module stays importable without the training script).
BASE_MODEL_ID = "CohereLabs/tiny-aya-global"


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


# ---------------------------------------------------------------------------
# Interactive rollout (real model; the loop itself is model-agnostic and
# unit-tested with scripted completions — only _complete needs a GPU)
# ---------------------------------------------------------------------------

def first_call_from_text(text: str) -> tuple[dict | None, str, int]:
    """Extract the first tool call from one generation.

    Returns ``(call, prefix, malformed)``: ``call`` is the first parsed
    call dict (or None if the generation contains none), ``prefix`` is
    the text up to and including the line that produced it (the rest of
    the generation is discarded — the manual says one call per step),
    and ``malformed`` counts call-shaped lines that failed to parse up
    to that point, using :func:`parse_rollout`'s own accounting.
    """
    prefix_lines: list[str] = []
    malformed = 0
    for line in text.splitlines():
        prefix_lines.append(line)
        calls, bad = parse_rollout(line)
        malformed += bad
        if calls:
            return calls[0], "\n".join(prefix_lines), malformed
    return None, text, malformed


def run_interactive_rollout(task: dict, executor: ToolExecutor,
                            complete_fn, *,
                            max_turns: int = 6,
                            feedback_chars: int = 1200) -> tuple[list[dict], int]:
    """Multi-turn tool-call loop driven by ``complete_fn(prompt) -> str``.

    Each turn: generate from the running conversation, execute the
    first call in the sandbox, append the call text and a one-line
    ``Result: {json}`` feedback to the conversation. Stops when the
    model produces a generation with no parseable call, or after
    ``max_turns`` executed calls. Returns ``(calls, malformed)`` where
    ``malformed`` counts unparseable call-shaped lines (structural
    validation of the parsed calls is the harness's job, as with the
    mock path).
    """
    conversation = task["prompt"]
    calls: list[dict] = []
    malformed = 0
    for _ in range(max_turns):
        text = complete_fn(conversation)
        call, prefix, bad = first_call_from_text(text)
        malformed += bad
        if call is None:
            break
        calls.append(call)
        result = executor.dispatch(call)
        feedback = json.dumps(result, default=str)
        if len(feedback) > feedback_chars:
            feedback = feedback[:feedback_chars] + "...[truncated]"
        conversation += f"\n{prefix.strip()}\nResult: {feedback}"
    return calls, malformed


class HFModel:  # pragma: no cover - requires GPU/transformers (Colab)
    """Real-model adapter: tiny-aya-global (+ optional LoRA adapter),
    greedy generation, interactive one-call-per-turn rollouts."""

    interactive = True  # evaluate_model lets it dispatch during the loop

    def __init__(self, model_id: str = BASE_MODEL_ID,
                 adapter_id: str | None = None, *,
                 max_turns: int = 6, max_completion: int = 512,
                 load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Tokenizer from the BASE model: LoRA checkpoints may or may
        # not ship tokenizer files; the base always does.
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Mirror train_grpo._model_init_kwargs: bf16 weights pinned to
        # the single GPU, base frozen in 4-bit by default (a 14.5 GB T4
        # cannot hold the bf16 base plus full-vocab logits).
        init_kwargs: dict = {
            "torch_dtype": "bfloat16",
            "device_map": {"": "cuda"} if torch.cuda.is_available() else "auto",
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            init_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **init_kwargs)
        if adapter_id:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_id)
        self.model.eval()
        self.max_turns = max_turns
        self.max_completion = max_completion

    def _complete(self, prompt_text: str) -> str:
        """One greedy generation; returns only the newly generated text."""
        import torch

        inputs = self.tok(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_completion,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True)

    def generate(self, task: dict, executor: ToolExecutor) -> tuple[list[dict], int]:
        return run_interactive_rollout(
            task, executor, self._complete, max_turns=self.max_turns)


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
            if getattr(model, "interactive", False):
                # Real model: it dispatches each call itself during the
                # rollout (it needs the results as feedback); the calls
                # are already executed, in order, when it returns.
                calls, extra_bad = model.generate(task, executor)
                malformed = extra_bad
                for c in calls:
                    ok, _ = validate_call(c)
                    if not ok:
                        malformed += 1
            else:
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
    ap.add_argument("--model", choices=["mock-base", "mock-tuned", "hf"], required=True)
    ap.add_argument("--tasks", "--data", dest="tasks", default="data/heldout.jsonl")
    ap.add_argument("--limit", "--max-tasks", dest="limit", type=int, default=0,
                    help="0 = all tasks")
    ap.add_argument("--out", default="")
    ap.add_argument("--adapter", default="",
                    help="hf only: LoRA adapter path or Hub repo id "
                         "(e.g. runs/grpo-full or Tashfiq99/patchtune-mini-grpo)")
    ap.add_argument("--model-id", default=BASE_MODEL_ID,
                    help="hf only: base model id")
    ap.add_argument("--max-turns", type=int, default=6,
                    help="hf only: max executed calls per task")
    ap.add_argument("--max-completion", type=int, default=512,
                    help="hf only: max new tokens per generation turn")
    ap.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="hf only: load the base in 4-bit (bitsandbytes), "
                         "as in training; --no-load-in-4bit on bigger GPUs")
    args = ap.parse_args()

    tasks = load_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    if args.model == "hf":
        model = HFModel(args.model_id, args.adapter or None,
                        max_turns=args.max_turns,
                        max_completion=args.max_completion,
                        load_in_4bit=args.load_in_4bit)
        label = "hf-tuned" if args.adapter else "hf-base"
    else:
        model = MockAgentModel("base" if args.model == "mock-base" else "tuned")
        label = args.model
    summary = evaluate_model(model, tasks)
    summary["model"] = label
    summary["tasks_file"] = args.tasks
    if args.model == "hf":
        summary["model_id"] = args.model_id
        summary["adapter"] = args.adapter or None
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))
    out = args.out or f"results/eval_{label.replace('-', '_')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
