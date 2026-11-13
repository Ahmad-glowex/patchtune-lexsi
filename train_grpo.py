"""GRPO training entry point for PatchTune-mini (run on Colab GPU).

Two training paths — the notebook tries PATH A first and falls back to
PATH B. Both optimise the same reward (``patchtune.reward.reward_func``)
on the same data (``data/train.jsonl``).

PATH A — AgentTune (preferred; it is the track toolkit)
    Uses ``agenttune.create_agentic_trainer(algorithm="grpo", ...)`` as
    documented at https://agenttune.lexsi.ai/user-guide/rl-training/ .
    Per those docs, agentic RL is the only training path that currently
    runs end-to-end (``core/sft`` is documented broken), and reward
    functions receive **flattened completion strings**, not message
    dicts — ``patchtune.reward.parse_rollout`` is written for exactly
    that convention.

PATH B — TRL GRPO fallback (if agenttune is not pip-installable or its
    API differs in the installed version)
    Plain ``trl.GRPOTrainer`` over text completions of the task prompt,
    scored by the same reward function. The model still learns the
    canonical tool-call sequence format; only the rollout machinery
    differs.

Usage (Colab):
    python train_grpo.py --smoke            # 5 steps, verifies the path
    python train_grpo.py --path trl         # force the fallback
    python train_grpo.py --steps 300 --out runs/patchtune-grpo

Base model: CohereLabs/tiny-aya-global (hackathon Rules §3: dense
open-weights Cohere Labs LLM). Other models may only be teacher/judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_MODEL = "CohereLabs/tiny-aya-global"


def load_prompts(data_dir: str) -> list[dict]:
    rows = []
    for line in Path(data_dir, "train.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        rows.append({"prompt": t["prompt"], "task_json": json.dumps(t)})
    return rows


# ---------------------------------------------------------------------------
# PATH A — AgentTune
# ---------------------------------------------------------------------------

def train_agenttune(args) -> str:
    """Returns the output dir on success; raises on any failure so the
    caller can fall back to PATH B."""
    import agenttune  # noqa: F401  (ImportError -> caller falls back)
    from agenttune import create_agentic_trainer

    from patchtune.reward import reward_func
    from patchtune.tools import TOOL_SPECS  # tool names for the trainer

    trainer = create_agentic_trainer(
        algorithm="grpo",
        model=args.model,
        tools=sorted(TOOL_SPECS),          # read_file/search_code/run_tests/apply_patch
        reward_funcs=[reward_func],
        train_dataset=load_prompts(args.data_dir),
        max_steps=args.steps,
        output_dir=args.out,
    )
    trainer.train()
    trainer.save_model(args.out)
    return args.out


# ---------------------------------------------------------------------------
# PATH B — TRL GRPO fallback
# ---------------------------------------------------------------------------

def train_trl(args) -> str:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from patchtune.reward import reward_func

    dataset = Dataset.from_list(load_prompts(args.data_dir))
    config = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        per_device_train_batch_size=2,
        num_generations=4,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=[reward_func],
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.out)
    return args.out


def main() -> None:
    ap = argparse.ArgumentParser(description="PatchTune-mini GRPO training")
    ap.add_argument("--path", choices=["auto", "agenttune", "trl"], default="auto")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="runs/patchtune-grpo")
    ap.add_argument("--smoke", action="store_true", help="5 steps only")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 5

    if args.path in ("auto", "agenttune"):
        try:
            out = train_agenttune(args)
            print(f"PATH A (AgentTune) done -> {out}")
            return
        except Exception as exc:  # noqa: BLE001 - deliberate fallback
            print(f"PATH A (AgentTune) unavailable/failed: {type(exc).__name__}: {exc}")
            if args.path == "agenttune":
                raise
            print("Falling back to PATH B (TRL GRPO)...")
    out = train_trl(args)
    print(f"PATH B (TRL) done -> {out}")


if __name__ == "__main__":
    main()
