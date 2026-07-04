# One-pager: Beating BDB 2026 Prediction with an autoresearch loop

**Goal.** Match/beat the 1st-place score of the NFL Big Data Bowl 2026
Prediction competition (RMSE on player x/y while the ball is in the air),
using an autonomous experiment loop on a MacBook (M2, 16 GB), then verify the
true score by late-submitting through the Kaggle evaluation API.

**Starting point (research).** All three top solutions share one recipe:
raw last-N-frame tracking features per player → per-player temporal encoder →
attention/graph interaction across the 22 players → decoder emitting per-frame
displacements (not absolute positions), trained with uncertainty-aware losses
(GaussianNLL / temporal Huber) + velocity/acceleration auxiliary losses, heavy
geometric augmentation (uniform rotation, flips, early-start), EMA, and large
ensembles (1st place: 100+ models). The differentiators were augmentation
design and loss shaping — not model size. Their published ablations also give
a free "do-not-try" list (dropout, complex geometry features, deep-narrow
models), which seeds the loop's backlog.

**Method.** Karpathy-style autoresearch, hardened:
one editable `train.py`, frozen data+eval in `prepare.py` (competition metric,
weeks 17–18 temporal holdout), fixed wall-clock budget per run, git
keep/revert, and three on-disk state files (results, journal, backlog) so the
loop survives restarts. Improvements over the prior numa_model attempt:
independent re-scoring of every run (no self-graded metrics), two-tier
screen/confirm budgets with an explicit noise band and second-seed reruns, an
evidence-seeded idea backlog, and Fable-5-tuned program language for long
autonomous runs.

**Baseline model (iteration 1).** 1.4M-param play-level model: 12 dynamic
features × 20 frames → conv encoder; 14 static features; 3-layer SwiGLU
transformer over player slots; conv decoder → 48 frames × (Δx, Δy, σx, σy)
from the final input frame; GaussianNLL + velocity/accel aux; rotation,
h-flip, early-start augmentation; EMA; AdamW + cosine.

**Results** (weeks 17–18 holdout, 420s screen budget on M2 unless noted).
All top-3 solutions published only writeups except 1st place, whose actual
train + inference notebooks we pulled via the Kaggle API and cloned faithfully.
| run | change | val RMSE |
|-----|--------|----------|
| CV baseline | constant-velocity extrapolation | 1.645 |
| exp0 | 1st-place recipe, scaled to M2 (1.4M→2.5M params) | 0.944 |
| exp1 | + SWA tail-averaging, best-of{raw,ema,swa} | 0.949 (keep: stabilizes) |
| exp2 | batch 64→128 | 1.274 (discard) |
| exp3 | conv-stem width 32→16 (more steps/budget) | 0.922 |
| exp4 | aux-loss weight 0.8→0.5 | 0.728 |
| exp5 | aux-loss weight →0.2 | 0.687 |
| exp6 | aux-loss weight →0.0 (dropped) | 0.652 |
| exp7 | cosine LR decay | 0.699 (discard) |
| exp8 | LR 5e-4→1e-3 constant | 0.631 |
| — | overnight knob sweep (7 trials: LR/width/DIM1) | no adoptions — knob-local optimum |
| exp9 | per-frame delta + cumsum decoding (2nd place idea) | 0.602 |
| exp10 | 30 input frames | 0.617 (discard) |
| exp11 | geometric endpoint prior features (GNN notebook idea) | 0.605 (discard: within noise) |
| exp12 | 3→2 transformer layers (wide-shallow) | 0.597 |
| exp13 | h-flip test-time augmentation | 0.590 |
| confirm | champion config @ 900s budget | 0.553 |
| R1 finals | champion @ 1800s, seeds 1337/2024/7 | 0.5156 / 0.5148 / 0.5137 |
| R1 ensemble | 3 seeds + h-flip TTA | 0.5121 |
| R2 finals | @7200s on 3 data folds (MULT16 / MULT32 / AUX0.4) | 0.4837 / 0.4801 / 0.4866 |
| R2 ensemble | greedy subset (MULT32+MULT16) + h-flip TTA | **0.4784** |

Two biggest levers, in order. **(1) Loss balance.** The 1st-place 0.8 weight on
velocity/acceleration auxiliaries is tuned for their ~37k-step training (35
epochs × a hidden `train_data_repeat_factor=6`); at our budgets the auxiliaries
are pure drag — a clean monotone (AUX 0.0→0.4→0.8 = 0.51→0.59→0.65 at 2400s),
so dropping them plus raising LR took the model 0.94→0.63. **(2) Budget, then
scale.** Longer training moved a single model 0.590 (420s) → 0.515 (1800s) →
0.484 (7200s, 34k steps ≈ 1st-place count). At that point a *wider* model
(MULT 32) overtook the base *with fewer steps* — proof the earlier width
screens were throughput-limited, not quality-limited. Ensembling gains came
from **data-fold diversity** (1st place: 5 folds × 3 seeds), not seeds: our
seed-only round-1 ensemble gained 0.0016, while two fold-diverse round-2 models
gained 0.0017 over the best single and 0.034 over round 1.

**Round 3 (Kaggle T4, server-side).** Ported training to a Kaggle GPU kernel so
it runs without the laptop. The 12h kernel limit plus two season-shift screens
left time for only one of five folds, but the screens were the point:
- `shift_lean` (our lean config, 12k steps, validated on a held-out *season*
  weeks 14–18) = 0.490;
- `shift_1stplace` (the 1st-place full-scale config, 12k steps, same split) =
  0.641 — but this is a starved 4.9M model at 12k steps, not a verdict on scale;
- `kfold_0` (the full-scale config trained the full 37k steps, game-fold split)
  = **0.4685** — our best-trained checkpoint, though on an in-distribution
  (game-fold, not season) split.
The lesson: apparent config verdicts are confounded by step count — a 4.9M
model needs ~37k steps to breathe, and comparing it to a lean model at 12k
steps is meaningless. kf0 (0.4685) was submitted to the real leaderboard.

**Protocol lessons** (improvements over the earlier numa_model loop): the
dominant error source at short budgets is initialization/data-order luck
(~0.2 across seeds) not run noise (same-seed MPS ~0.003–0.015), so all screens
fix the seed and reserve seed diversity for the final ensemble. Every run's
number is re-scored independently from saved predictions; SWA tail-averaging
makes the estimator stable.

**True test score (the real exam).** The 0.4784 ensemble was late-submitted
through Kaggle's evaluation API (dataset + script kernel + `submit-notebook`,
driven by REST since the account token is Bearer-only). Kaggle reran it on the
full hidden test and returned a **private score of 0.57915** (submission
54275816, status complete). The private leaderboard top is 0.46340 (1st) with
the 20th-place cutoff at 0.50460, so this lands **outside the top 20** — we did
not beat 1st place.

The instructive part is the gap structure: on our weeks-17–18 holdout we were
0.478 vs the top's ~0.465 (≈0.013 back), but on the real test that gap widened
to ≈0.116 (0.579 vs 0.463). Our holdout was *same-season* (held-out weeks of
the 2023–24 data we trained on), i.e. in-distribution, while the hidden test is
the **2025 season** — fully out-of-distribution. The top solutions' heavy
augmentation and 15-model fold ensembles are what buy robustness to that
season shift; our lean 2-model ensemble overfit the same-season holdout and
generalized worse. The lesson for a next attempt is that the holdout should
have simulated season shift (e.g. train 2023 / validate 2024), and that the
distance to close is generalization, not holdout RMSE.

**Round 3 (Kaggle-GPU, full 1st-place scale).** Real LB progression:
| submission | what | private RMSE |
|-----------|------|--------------|
| r2 ensemble | 2 lean models (MULT16/32) | 0.57915 |
| kf0 | single full-scale (MULT64, 37k steps) | 0.53559 |
| kf0+kf1+kf2 | 3-fold full-scale ensemble | **0.53405** |

The 0.579 -> 0.536 gain came from model **scale + full 37k-step training**.
Fold-ensembling added only 0.0015 (correlated members). We attributed the
residual ~0.07 gap to "out-of-distribution generalization" — and were **wrong**.

**Round 4 (the real story: an inference bug, not distribution shift).** A code
audit (subagent vs the reference) plus a numeric **parity test** — feeding raw
weeks-17/18 plays through the exact submission-notebook API path and comparing
to the local eval path on the same checkpoint — found the two paths disagreed
by 0.052 while every input array matched **except one**: `n_out` (frames-to-
predict, a static feature). `prepare.preprocess` sets it from the input csv's
`num_frames_output` for all 22 players; the notebook derived it from the
*scored*-player test frame, leaving `n_out=0` for every unscored player — a
value the model only ever saw on padding slots. It silently poisoned the
transformer context on **every play of every submission**. One-line fix
(read `num_frames_output` from `test_input`); parity gap collapsed to -0.0003.

Resubmitting the *identical* 3-fold ensemble with the fix: **0.53405 -> 0.47490**,
clearing the 20th-place cutoff (0.50460) and landing 0.0115 from 1st. So ~77%
of the apparent OOD gap was this bug. A parallel Kaggle A/B also settled two
levers: the true 1st-place **direct** displacement head (no cumsum) marginally
beats our 2nd-place cumsum decoder and adds ensemble diversity, while **heavy
augmentation hurts** (0.55 vs 0.46) — the opposite of the pre-fix hypothesis.

Real LB progression:
| submission | what | private RMSE |
|-----------|------|--------------|
| r2 ensemble | 2 lean models, n_out-bugged | 0.57915 |
| kf0 | single full-scale, bugged | 0.53559 |
| kf0+kf1+kf2 | 3-fold, bugged | 0.53405 |
| **kf0+kf1+kf2** | **3-fold, n_out FIXED** | **0.47490** |

**Status.** Beat the top-20 cutoff (0.4749 < 0.5046); 0.0115 from 1st (0.4634).
The autoresearch loop's discipline — frozen eval, independent re-scoring, and a
parity harness that distrusted the "distribution shift" story enough to diff
the two code paths — is what surfaced a bug that four prior submissions had
silently paid for. Pushing the final gap via decode-diverse ensembling.
