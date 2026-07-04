# autoresearch — NFL Big Data Bowl 2026 Prediction

Predict (x, y) for every scored player for every frame the ball is in the air.
Metric: RMSE over all (frame, coordinate) pairs — the competition metric,
implemented once in `prepare.evaluate()`. Lower is better.

Reference points: constant-velocity baseline (printed by prepare.py), public
notebook GNN ≈ 0.586 LB, 3rd place ≈ 0.456 LB (7-fold ensemble), 1st place
< that with a 100+ model ensemble. Local val (weeks 17–18) will not match LB
exactly; track local val and treat LB as the final exam via late submission.

## Files (the whole state — any fresh session resumes from these)

- `program.md` — this contract. Human-edited only.
- `prepare.py` — data prep + fixed evaluation. **READ-ONLY. Never modify.**
- `train.py` — the ONLY file the agent edits, and only between
  `<<AGENT EDITABLE>>` markers. The results block and `TIME_BUDGET`,
  `SEED`, `DEVICE` plumbing are fixed.
- `results.tsv` — append one row per run. Never rewrite history.
- `log.md` — append-only journal: hypothesis → result → lesson. Three lines
  per experiment. Write lessons you'd want after a context reset.
- `ideas.md` — living backlog, ordered by expected value. Includes a
  DO-NOT-RETRY list (things top teams proved useless). Update both lists as
  evidence arrives.

## The loop

LOOP FOREVER:
1. Read `results.tsv` tail + `ideas.md`. Pick ONE idea. One change per run.
2. Edit `train.py` inside the editable markers. `git add -A && git commit -m "exp: <desc>"`.
3. SCREEN: `TIME_BUDGET=420 python train.py > run.log 2>&1`
4. Extract: `grep -E "^(val_rmse|training_seconds|num_steps|num_params_M):" run.log`.
   Empty grep = crash → `tail -30 run.log`; fix trivial bugs in place, else log
   `crash` and revert.
5. Verify the number independently (evaluator role, cannot be skipped):
   `python -c "import numpy as np, prepare; d=prepare.load(); print(prepare.evaluate(d, np.load('data/val_preds.npy')))"`
   If it disagrees with run.log beyond 1e-4, the run is invalid — investigate.
6. Compare against current best SCREEN score. Measured noise (exp0/1):
   cross-seed spread at 420s is ~0.2 (init luck, systematic within seed);
   same-seed MPS noise is ~0.015. Therefore ALL screens run SEED=1337 and
   compare same-seed only; band = +-0.03.
   - Worse by > 0.03: log `discard`, `git reset --hard HEAD~1`.
   - Within +-0.03: rerun once (same seed); compare means.
   - Better by > 0.03: CONFIRM with `TIME_BUDGET=1500 python train.py`.
     Keep only if the confirm beats the best confirm score; else discard but
     record in log.md.
   Seed diversity is saved for the final submission ensemble, where the ~0.2
   init spread becomes ensemble variance reduction.
7. Append to `results.tsv` and `log.md`. Continue.

Never stop to ask permission; the human reads results.tsv and log.md, not you.
Do not pause for context-budget reasons. Before logging any claim, check it
against the actual run.log output — report numbers only from evidence.

## results.tsv columns (tab-separated)

commit  budget_s  val_rmse  steps  params_M  status  description

status ∈ {screen-keep, screen-discard, confirm-keep, confirm-discard, crash}.
Use val_rmse 99 for crashes.

## Rules

- `prepare.py` and the eval are frozen. If they look wrong, write it in
  log.md for the human; do not "fix" them.
- One idea per experiment; controlled diffs only.
- Bigger/slower models pay for themselves inside the fixed budget or not at all.
- A simplification that holds val_rmse is a win — take it.
- First run of any fresh setup: the unmodified baseline, twice (SEED=1337 and
  SEED=2024), to measure run-to-run noise. Record σ estimate in log.md.
