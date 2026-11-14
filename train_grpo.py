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
import os
from pathlib import Path

# agenttune hard-depends on wandb and its trainer prompts interactively
# for a W&B login in Colab, which blocks the run. Disable it outright;
# PATH B also passes report_to=[].
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

BASE_MODEL = "CohereLabs/tiny-aya-global"


def _save_kwargs(args) -> dict:
    """Checkpoint settings shared by both paths.

    Training saves a checkpoint DURING the run (every --save-steps
    steps), not only at the end, and — when --push-to names a Hub
    repo — pushes each checkpoint there as it is written. On an
    ephemeral-disk runner (HF Jobs) a job killed mid-run (timeout,
    OOM, spot loss) therefore keeps every completed step's progress
    on the Hub instead of losing the whole run.
    """
    kw = {"save_strategy": "steps", "save_steps": args.save_steps}
    if args.push_to:
        kw["push_to_hub"] = True
        kw["hub_model_id"] = args.push_to
    return kw


def _resolve_resume(args) -> str | None:
    """Download the --resume-from Hub repo and return the local path
    of its newest checkpoint-* folder (full Trainer checkpoint,
    optimizer state included), or None to start fresh."""
    if not args.resume_from:
        return None
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=args.resume_from, repo_type="model")
    ckpts = sorted(
        Path(local).glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if not ckpts:
        print(f"RESUME: no checkpoint-* in {args.resume_from}; starting fresh")
        return None
    print(f"RESUME: continuing from {ckpts[-1].name} in {args.resume_from}")
    return str(ckpts[-1])


def _model_init_kwargs(args) -> dict:
    """Shared model-loading kwargs. --load-in-4bit shrinks the frozen
    base from ~6.7 GB (bf16) to ~2 GB: on a 14.5 GB T4 the bf16 base
    plus GRPO's full-vocab (262k) logits left no room and both paths
    OOM'd even at batch 2 / group 2 (Nov 14 Colab smoke)."""
    kw = {"torch_dtype": "bfloat16", "device_map": {"": "cuda"}}
    if args.load_in_4bit:
        kw["quantization_config"] = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        }
    return kw


def load_prompts(data_dir: str) -> list[dict]:
    rows = []
    for line in Path(data_dir, "train.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        rows.append({"prompt": t["prompt"], "task_json": json.dumps(t)})
    return rows


# ---------------------------------------------------------------------------
# PATH A — AgentTune
# ---------------------------------------------------------------------------

def _tool_ack(name: str, arguments: dict) -> str:
    """Validate one tool call and acknowledge it.

    Rollout-time execution against a live repo is deliberately out of
    scope (see patchtune.reward.reward_func): GRPO's learning signal
    comes from the reward function, which re-parses the emitted calls
    and scores them against the task. These callables exist so the
    trainer receives plain named functions — TRL's trainer glue indexes
    tools by ``__name__`` and crashes on name strings.
    """
    from patchtune.tools import validate_call

    ok, _msg = validate_call({"tool": name, "arguments": arguments})
    return json.dumps({"tool": name, "accepted": ok})


def read_file(path: str) -> str:
    """Read a file from the project repository."""
    return _tool_ack("read_file", {"path": path})


def search_code(query: str) -> str:
    """Search the project code for a text query."""
    return _tool_ack("search_code", {"query": query})


def run_tests() -> str:
    """Run the project test suite."""
    return _tool_ack("run_tests", {})


def apply_patch(diff: str) -> str:
    """Apply a unified-diff patch to the project."""
    return _tool_ack("apply_patch", {"diff": diff})


def train_agenttune(args) -> str:
    """Returns the output dir on success; raises on any failure so the
    caller can fall back to PATH B."""
    import agenttune  # noqa: F401  (ImportError -> caller falls back)
    from agenttune import create_agentic_trainer
    from datasets import Dataset

    from patchtune.reward import reward_func

    # Memory levers (agenttune 1.1.6's TRL backend routes kwargs by
    # introspection and accepts peft_config as a dict): LoRA + bf16 +
    # everything pinned to the single GPU. Without these, PATH A tried
    # full fine-tuning with disk offload and OOM'd the 14.5 GB T4
    # (Nov 14 aligned-env run).
    trainer = create_agentic_trainer(
        algorithm="grpo",
        model=args.model,
        tools=[read_file, search_code, run_tests, apply_patch],
        reward_funcs=[reward_func],
        train_dataset=Dataset.from_list(load_prompts(args.data_dir)),
        max_steps=args.steps,
        output_dir=args.out,
        report_to=[],
        peft_config={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "task_type": "CAUSAL_LM",
            "target_modules": "all-linear",
        },
        model_init_kwargs=_model_init_kwargs(args),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=args.batch,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        learning_rate=5e-5,
        # Rollout caps (agenttune backend kwargs, read via _get):
        # a trajectory defaults to max_steps_per_turn=20 turns, run
        # synchronously with a growing context — the Nov 14 L40S smoke
        # job sat in step 1 for 18+ minutes without finishing a single
        # step. Our canonical trajectory is exactly 4 tool calls, so
        # 6 turns is ample slack; 256 new tokens per turn is the
        # backend's own default, set here explicitly.
        max_steps_per_turn=args.max_steps_per_turn,
        max_new_tokens=256,
        **_save_kwargs(args),
    )
    if args.resume_ckpt:
        try:
            trainer.train(resume_from_checkpoint=args.resume_ckpt)
        except TypeError:
            # agenttune's train() wrapper may not forward the kwarg;
            # checkpoints still save+push, only the resume is skipped.
            print("RESUME: PATH A train() rejected resume_from_checkpoint; starting fresh")
            trainer.train()
    else:
        trainer.train()
    trainer.save_model(args.out)
    return args.out


# ---------------------------------------------------------------------------
# PATH B — TRL GRPO fallback
# ---------------------------------------------------------------------------

def train_trl(args) -> str:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from patchtune.reward import reward_func

    dataset = Dataset.from_list(load_prompts(args.data_dir))
    # LoRA + bf16 + gradient checkpointing: full fine-tuning a 3.35B
    # model does not fit a free T4 (15 GB) once AdamW states are counted;
    # the first smoke attempt thrashed on CPU/disk offload. LoRA keeps
    # the trainable state small and the base frozen in bf16.
    #
    # The config is built defensively: TRL renamed/removed GRPOConfig
    # fields across versions (a hard TypeError on 'max_prompt_length'
    # killed the Nov 14 fresh-env attempt), so drop any kwarg the
    # installed GRPOConfig rejects and retry.
    import re

    config_kwargs = dict(
        output_dir=args.out,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        num_generations=args.num_generations,
        learning_rate=5e-5,  # LoRA-scale rate (full-FT would use ~1e-5)
        logging_steps=1,
        report_to=[],
        **_save_kwargs(args),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=1024,
        max_completion_length=args.max_completion,
        # Pin every parameter to the GPU: without device_map, the
        # Nov 14 aligned-env run left weights on meta/CPU/disk offload
        # and accelerate died with "Cannot copy out of meta tensor".
        model_init_kwargs=_model_init_kwargs(args),
    )
    while True:
        try:
            config = GRPOConfig(**config_kwargs)
            break
        except TypeError as exc:
            m = re.search(r"unexpected keyword argument '([A-Za-z_]+)'", str(exc))
            if not m or m.group(1) not in config_kwargs:
                raise
            print(f"GRPOConfig: dropping unsupported kwarg {m.group(1)!r}")
            del config_kwargs[m.group(1)]
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=[reward_func],
        train_dataset=dataset,
        peft_config=peft_config,
    )
    try:
        # PEFT + gradient checkpointing: the checkpoint hook needs input
        # grads enabled on the frozen base (documented PEFT caveat).
        trainer.model.enable_input_require_grads()
    except Exception:  # noqa: BLE001 - best effort; training still runs
        pass
    if args.resume_ckpt:
        trainer.train(resume_from_checkpoint=args.resume_ckpt)
    else:
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
    ap.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Prompts per training step. Use 2 on a 16 GB T4: the "
        "scoring phase materialises full-vocab logits and OOMs at 4.",
    )
    ap.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="GRPO group size (completions per prompt). 2 is the "
        "minimum valid group; use it on small GPUs.",
    )
    ap.add_argument(
        "--max-steps-per-turn",
        type=int,
        default=6,
        help="AgentTune PATH A only: max tool turns per trajectory "
        "(backend default is 20 — far too slow; canonical task = 4).",
    )
    ap.add_argument(
        "--max-completion",
        type=int,
        default=512,
        help="Max completion tokens per generation. 256 halves the "
        "logits memory on small GPUs.",
    )
    ap.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the frozen base in 4-bit (bitsandbytes). Frees "
        "~4.7 GB vs bf16 — the lever that lets a 16 GB T4 hold the "
        "base plus GRPO's full-vocab logits.",
    )
    ap.add_argument(
        "--push-to",
        default=None,
        help="HF model repo id (e.g. Tashfiq99/patchtune-mini-grpo) to upload "
        "the trained adapter to, using the ambient HF_TOKEN. Needed on HF "
        "Jobs, where the local disk disappears when the job ends.",
    )
    ap.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save (and, with --push-to, upload) a checkpoint every N "
        "steps. Default: 1 for --smoke, 25 otherwise.",
    )
    ap.add_argument(
        "--resume-from",
        default=None,
        help="HF model repo id holding checkpoint-* folders from an "
        "earlier interrupted run; training continues from the newest "
        "checkpoint instead of starting over.",
    )
    args = ap.parse_args()
    if args.smoke:
        args.steps = 5
    if args.save_steps is None:
        args.save_steps = 1 if args.smoke else 25
    args.resume_ckpt = _resolve_resume(args)

    out = None
    if args.path in ("auto", "agenttune"):
        try:
            out = train_agenttune(args)
            print(f"PATH A (AgentTune) done -> {out}")
        except Exception as exc:  # noqa: BLE001 - deliberate fallback
            print(f"PATH A (AgentTune) unavailable/failed: {type(exc).__name__}: {exc}")
            if args.path == "agenttune":
                raise
            print("Falling back to PATH B (TRL GRPO)...")
            # PATH A may have loaded the model before failing; free what
            # we can so PATH B does not start against a full GPU/RAM.
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    if out is None:
        out = train_trl(args)
        print(f"PATH B (TRL) done -> {out}")

    if args.push_to:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.push_to, repo_type="model", exist_ok=True)
            api.upload_folder(folder_path=out, repo_id=args.push_to, repo_type="model")
            print(f"Pushed adapter -> https://huggingface.co/{args.push_to}")
        except Exception as exc:  # noqa: BLE001 - report, never mask
            print(f"PUSH FAILED ({type(exc).__name__}: {exc}) — adapter only at {out}")


if __name__ == "__main__":
    main()
