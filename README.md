Read the full submission write-up: [ARTICLE.md](ARTICLE.md)

# PatchTune-mini

**Post-train CohereLabs/tiny-aya-global with GRPO so a coding agent makes
reliable tool calls** — the right tool, well-formed arguments, in the right
order, with a verified final outcome.

- **Event:** Lexsi AI Safety & Alignment Hackathon — **Track 3: Agentic
  Fine-tuning (AgentTune + AuditKIT)**
- **Problem statements:** #1 *Reliable tool calling* (primary),
  #2 *Efficient trajectories* (secondary — calls/task metric)
- **Deadline:** Stage 1 — Nov 15, 2026, 11:59 PM AoE (Mon Nov 16, 17:59 Dhaka)
- **Status:** environment, data, reward, and evaluation scaffolding are
  built and tested (see `STATUS.md`). GPU training + real-model eval are
  the Sat/Sun steps (`train_grpo_colab.ipynb`).

## Layout

| Path | What |
|---|---|
| `toyrepo/` | Generator for a tiny Python package (`tinylib`) with one planted, single-line, programmatically-fixable bug per instance + a unittest suite that fails pre-fix and passes post-fix. 8 bug families. |
| `patchtune/tools.py` | The 4 agent tools — `read_file`, `search_code`, `run_tests`, `apply_patch` — executed by `ToolExecutor` inside a sandbox dir, plus structural call validation and a unified-diff applier. |
| `patchtune/tasks.py` | Synthetic task generator → `data/train.jsonl` (150) + `data/heldout.jsonl` (50), HF-datasets-compatible JSONL. |
| `patchtune/reward.py` | Verifier/reward: matched expected calls (+≤2), verified outcome (+2), penalties for malformed calls, loops, excess calls. Defensive parsing of flattened-string completions (AgentTune convention). |
| `patchtune/evaluate.py` | Before/after harness: task success, tool-call accuracy, calls/task, malformed rate. Mock models run it end-to-end with no GPU. |
| `train_grpo.py` | Training entry point. **PATH A:** AgentTune `create_agentic_trainer(algorithm="grpo")`. **PATH B:** TRL `GRPOTrainer` fallback (auto). |
| `train_grpo_colab.ipynb` | Colab runbook: install → data → **smoke test** → train → eval → publish. |
| `tests/` | Unit tests (unittest-style, also run under pytest). |
| `data/`, `results/` | Generated datasets; eval summaries. |

## Quickstart (no GPU)

```bash
python -m patchtune.tasks --out-dir data --train 150 --heldout 50
python -m unittest discover -s tests -t .          # 21 tests
python -m patchtune.evaluate --model mock-base   --tasks data/heldout.jsonl
python -m patchtune.evaluate --model mock-tuned  --tasks data/heldout.jsonl
```

Mock-mode output on the 50 held-out tasks (pipeline sanity check only —
**not** a result; the real before/after comes from Tiny Aya runs):

| model | success | tool-call acc. | calls/task | malformed |
|---|---|---|---|---|
| mock-base | 0.50 | 0.817 | 3.5 | 0.109 |
| mock-tuned | 1.00 | 1.000 | 3.8 | 0.000 |

## Design notes

- **Tasks.** Each task is self-contained: the buggy repo as a file map, a
  natural-language bug report, the canonical correct tool sequence
  (`search_code → read_file → apply_patch → run_tests`) and the canonical
  fix diff. Train and held-out splits use **disjoint module-name pools and
  disjoint instruction phrasings**, so held-out scores are not memorised
  surface forms.
- **Reward.** Outcome is *verified*, not judged: a trajectory earns the
  outcome reward only when the sandbox test suite actually passes. Format
  is scored separately (structural validation in `tools.validate_call`),
  so a well-formed-but-wrong patch and a malformed call are penalised
  differently.
- **Flattened-string rewards (AgentTune R1).** AgentTune's RL docs warn
  that GRPO reward functions receive flattened completion strings rather
  than message dicts. `reward.parse_rollout` therefore accepts dicts,
  JSON strings, JSON-lines and `tool(arg=...)` text lines, and never
  raises.
- **Stale-bytecode gotcha (found during this build).** Planted fixes are
  same-length one-line edits; a patched file can share size+mtime with
  the buggy original and Python would run the stale `.pyc`. The executor
  clears `__pycache__` after patching and before every test run, and runs
  tests with `-B`.
- **Tests are unittest-style on purpose:** the whole local pipeline runs
  with zero third-party packages on the stdlib alone.

## Training (GPU, Colab)

Open `train_grpo_colab.ipynb`. It smoke-tests first (5 steps), then runs
`python train_grpo.py --steps 300`. PATH A (AgentTune) is tried first;
any import/API failure prints the error and falls back to PATH B (TRL)
automatically. AgentTune's own docs state its `core/sft` path is broken
and agentic RL is the only path that runs end-to-end — this project is
GRPO-only by design.

## Evaluation (real model)

For each of base / tuned: run the interactive rollout (generation
interleaved with `ToolExecutor.dispatch` in a per-task sandbox) on the
identical held-out set, score with `patchtune.evaluate` metrics, and pass
the same traces through **AuditKIT** agent scorers (tool-call F1,
validity, redundant calls, loop detection) — chaining both track
toolkits in one pipeline. Report ≥2 metrics before/after (we report 4).

## Submission-checklist mapping

| Required item (hackathon "Submit" + kickoff checklist) | Where / status |
|---|---|
| Registered email | tashfiq668@gmail.com (team lead) |
| Abstract < 500 words | To write Mon from README intro + results |
| HF **model** collection (public) | Pending GPU run: base ref + GRPO adapter |
| HF **dataset** collection (public) | `data/*.jsonl` + traces + eval JSONs, ready to upload |
| Write-up (HF Article, 9 sections; GitHub README acceptable) | This README is the scaffold; article = Mon step |
| GitHub repo + reproducibility README | This repo; quickstart above reproduces data/tests/eval |
| Track + problem statement | Track 3; PS #1 (+#2) |
| Hackathon repos used | AgentTune (training, PATH A) + AuditKIT (trace scoring) |
| Models + licences | CohereLabs/tiny-aya-global — quote licence from its HF model card in the article (verify at write-up time) |
| Datasets + licences | Synthetic, generated by `toyrepo/`+`patchtune/tasks.py` in this repo during the hackathon; project licence MIT |
| Before/after metrics (>1) | 4 metrics via `patchtune/evaluate.py` (+ AuditKIT scorers) |

## Licences

- This project's code: MIT.
- AgentTune and AuditKIT are **source-available under LSAL** (not
  MIT/Apache) — described as such everywhere in the write-up.
- Base model licence: as stated on the CohereLabs/tiny-aya-global HF
  model card (to be quoted verbatim in the article).
- Datasets: synthetic, self-generated; no third-party data.
