# STATUS — PatchTune-mini scaffolding

**Updated:** Fri Nov 13, 2026 (Asia/Dhaka) · **Deadline:** Mon Nov 16, 17:59 Dhaka
**Scope of this step:** code scaffolding only — no accounts, no submissions,
no GPU runs, no spending.

## Built

- `toyrepo/` — toy-repo generator: 8 single-line bug families, each with a
  failing unittest suite and a canonical fix diff; train/held-out module
  pools and phrasings are disjoint.
- `patchtune/tools.py` — sandboxed `ToolExecutor` with `read_file`,
  `search_code`, `run_tests`, `apply_patch` (+ unified-diff parser/applier,
  structural `validate_call`, path-escape protection).
- `patchtune/tasks.py` — task generator; `data/train.jsonl` (**150**) and
  `data/heldout.jsonl` (**50**) generated and committed in `data/`.
- `patchtune/reward.py` — verifier/reward (match ≤ +2, verified outcome +2,
  malformed −0.5, loop −0.3, excess −0.1) with defensive parsing of
  flattened-string completions (AgentTune convention, research R1) and a
  trainer-facing `reward_func`.
- `patchtune/evaluate.py` — 4-metric harness (success, tool-call accuracy,
  calls/task, malformed rate) with deterministic mock models.
- `train_grpo.py` + `train_grpo_colab.ipynb` — PATH A AgentTune
  `create_agentic_trainer(grpo)` / PATH B TRL GRPO fallback (auto), smoke
  mode, publish + submit steps.
- `README.md` (reproducibility + submission-checklist mapping),
  `requirements.txt`, `tests/` (21 unit tests).

## Verified (exact results)

- Unit tests: **21 ran, 21 passed, 0 failed** —
  `python -m unittest discover -s tests -t .` → `OK` (~3.4 s).
  Covers: every bug family fails pre-fix and passes post-fix; tools incl.
  malformed/path-escape/bad-patch handling; reward parsing (dicts, JSON,
  flattened function-lines, malformed counting, prose ignored) and scoring
  (perfect > empty, loops/malformed penalised); dataset disjointness and
  expected-call validity; mock eval (tuned beats base).
- Mock eval end-to-end on all 50 held-out tasks (sanity only, **not** a
  result): mock-base success 0.50 / acc 0.817 / 3.5 calls / malformed
  0.109 → mock-tuned 1.00 / 1.000 / 3.8 / 0.000.
  Summaries in `results/eval_mock_base.json`, `results/eval_mock_tuned.json`.
- Notebook JSON validated (14 cells).

## Blockers found & fixed during the build

1. **No pytest in this environment** → all tests written unittest-style
   (stdlib-only local pipeline; still pytest-compatible). Toy-repo suites
   are unittest too, so `run_tests` needs no third-party packages.
2. **Stale `__pycache__` bug (real):** same-length one-line fixes share
   size+mtime with the buggy file, so Python ran stale bytecode and fixed
   repos still "failed". Fixed: executor clears `__pycache__` after
   patching and before test runs; tests run with `-B`. (Initial run:
   19/21 passed; after fix: 21/21.)
3. One test expectation was itself wrong (a comment-appended patch is
   valid); corrected to a context-corrupted patch.

## Remaining — GPU steps (Sat/Sun, Colab)

1. **Fri-evening smoke test (still the #1 de-risk):** notebook §4 —
   `train_grpo.py --smoke`. Unknown: does AgentTune GRPO actually run on
   `CohereLabs/tiny-aya-global` (docs verify Qwen 0.5B–1.7B only, R2)?
   PATH B (TRL) is wired as automatic fallback.
2. Baseline real-model rollout + AuditKIT trace scoring (harness metrics
   exist; the interactive real-model rollout loop lives in the notebook —
   `evaluate.HFModel.generate` is a marked stub).
3. Full GRPO run (~300 steps), checkpoint to the HF model collection.
4. Post-training eval on the identical held-out set; before/after table;
   3–5 failure examples.
5. Mon: HF Article (9 sections), public checks, submit by ~15:00.

## Open items outside this build (parent/user)

- HF $20 credit coupon email (Nov 1) — check inbox/spam; redemption may
  need a payment method (user's no-card decision). Plan assumes free
  Colab is enough.
- Push this folder to a new public GitHub repo; fill the repo URL and
  HF username placeholders in the notebook.
- Discord #report_issues ticket (scope validation + pre-existing-code
  question, R5) — user is the Discord member.
- Tiny Aya licence text — quote from its HF model card in the article.
